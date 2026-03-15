"""World state and tick loop for GPU simulation.

WorldState is a flat collection of JAX arrays (pytree-compatible).
All operations are jittable - no Python control flow on traced values.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp


from semgpu.brain import (
    INPUTS,
    MEMORY_SIZE,
    NUM_SYMBOLS,
    batched_forward,
    softmax_emit,
)
from semgpu.signal import receive_signals_grid
from semgpu.spatial import (
    build_cell_grid,
    has_neighbor_in_radius,
    move_zones,
    nearest_in_grid,
    nearest_zone_edge_dist,
    wrap_delta,
    zone_drain_amount,
)

# Input layout offsets (must match Rust)
SIGNAL_INPUT_START = 9
MEMORY_INPUT_START = SIGNAL_INPUT_START + NUM_SYMBOLS * 3  # 27
ENERGY_INPUT_IDX = MEMORY_INPUT_START + MEMORY_SIZE  # 35

INPUT_NAMES = [
    "zone_damage", "energy_delta", "dead_spare",
    "food_dx", "food_dy", "food_dist",
    "ally_dx", "ally_dy", "ally_dist",
    "sig0_str", "sig0_dx", "sig0_dy",
    "sig1_str", "sig1_dx", "sig1_dy",
    "sig2_str", "sig2_dx", "sig2_dy",
    "sig3_str", "sig3_dx", "sig3_dy",
    "sig4_str", "sig4_dx", "sig4_dy",
    "sig5_str", "sig5_dx", "sig5_dy",
    "mem0", "mem1", "mem2", "mem3", "mem4", "mem5", "mem6", "mem7",
    "energy",
]
assert len(INPUT_NAMES) == INPUTS


class WorldState(NamedTuple):
    """All simulation state as flat JAX arrays. NamedTuple for pytree compat."""
    # Prey state (N = max population)
    prey_x: jnp.ndarray          # (N,) int32
    prey_y: jnp.ndarray          # (N,) int32
    energy: jnp.ndarray          # (N,) float32
    alive: jnp.ndarray           # (N,) bool
    zone_damage: jnp.ndarray     # (N,) float32
    prev_energy: jnp.ndarray     # (N,) float32
    memory: jnp.ndarray          # (N, 8) float32
    ticks_alive: jnp.ndarray     # (N,) int32
    food_eaten: jnp.ndarray      # (N,) int32
    weights: jnp.ndarray         # (N, 5491) float32
    base_hidden: jnp.ndarray     # (N,) int32
    signal_hidden: jnp.ndarray   # (N,) int32

    # Kill zones (Z = num_zones)
    zone_x: jnp.ndarray          # (Z,) float32
    zone_y: jnp.ndarray          # (Z,) float32
    zone_radius: jnp.ndarray     # (Z,) float32
    zone_speed: jnp.ndarray      # (Z,) float32

    # Food (F = max food slots)
    food_x: jnp.ndarray          # (F,) int32
    food_y: jnp.ndarray          # (F,) int32
    food_is_patch: jnp.ndarray   # (F,) bool
    food_alive: jnp.ndarray      # (F,) bool
    food_count: jnp.ndarray      # scalar int32, current number of alive food

    # Signal ring buffer (S = max_signals)
    sig_x: jnp.ndarray           # (S,) int32
    sig_y: jnp.ndarray           # (S,) int32
    sig_symbol: jnp.ndarray      # (S,) int32
    sig_tick: jnp.ndarray        # (S,) int32
    sig_valid: jnp.ndarray       # (S,) bool
    sig_cursor: jnp.ndarray      # scalar int32, write position in ring buffer

    # Counters
    tick: jnp.ndarray             # scalar int32
    signals_emitted: jnp.ndarray  # scalar int32
    zone_deaths: jnp.ndarray      # scalar int32

    # RNG
    key: jnp.ndarray              # PRNGKey

    # Metrics accumulators (on-device, updated each tick)
    mi_counts: jnp.ndarray          # (6, 4) int32 - signal x zone_dist_bin contingency
    symbol_counts: jnp.ndarray      # (6,) int32 - symbol frequencies
    iconicity_in_zone: jnp.ndarray  # scalar int32 - signals emitted near zone
    m_ticks_in_zone: jnp.ndarray    # scalar int32 - alive-prey-ticks inside any zone
    m_total_prey_ticks: jnp.ndarray # scalar int32 - total alive ticks
    recv_counts: jnp.ndarray        # (7, 2, 5) int32 - [no_sig+6sym][ctx][action]
    per_tick_signals: jnp.ndarray   # (T,) float32 - signals emitted per tick
    per_tick_alive: jnp.ndarray     # (T,) float32 - alive count per tick
    per_tick_min_zdist: jnp.ndarray # (T,) float32 - min zone edge dist per tick
    prey_signals_sent: jnp.ndarray  # (N,) int32
    prey_recv_with: jnp.ndarray     # (N, 2, 5) int32 - actions while hearing signal
    prey_recv_without: jnp.ndarray  # (N, 2, 5) int32 - actions while not hearing
    was_hearing: jnp.ndarray        # (N,) bool - silence onset tracking
    onset_actions: jnp.ndarray      # (N, 2, 5) int32 - actions at silence onset
    present_actions: jnp.ndarray    # (N, 2, 5) int32 - actions during signal
    # Event buffer for input_mi (ring buffer)
    evt_symbol: jnp.ndarray         # (E,) int32
    evt_inputs: jnp.ndarray         # (E, 36) float32
    evt_cursor: jnp.ndarray         # scalar int32
    evt_count: jnp.ndarray          # scalar int32 - total events seen


def init_world(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    weights: jnp.ndarray,
    base_hidden: jnp.ndarray,
    signal_hidden: jnp.ndarray,
    num_zones: int,
    food_count: int,
    grid_size: int,
    zone_radius: float,
    zone_speed: float,
    patch_ratio: float,
    max_signals: int,
    ticks_per_eval: int,
    max_events: int,
    key: jnp.ndarray,
) -> WorldState:
    """Initialize a WorldState for one generation evaluation."""
    N = prey_x.shape[0]
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Zone positions (continuous float)
    zx = jax.random.uniform(k1, (num_zones,)) * grid_size
    zy = jax.random.uniform(k2, (num_zones,)) * grid_size

    # Food positions
    fx = jax.random.randint(k3, (food_count,), 0, grid_size)
    fy = jax.random.randint(k4, (food_count,), 0, grid_size)
    k5, k6 = jax.random.split(k4)
    fp = jax.random.uniform(k5, (food_count,)) < patch_ratio

    # Memory init: small random [-0.1, 0.1]
    mem = jax.random.uniform(k6, (N, MEMORY_SIZE), minval=-0.1, maxval=0.1)

    return WorldState(
        prey_x=prey_x,
        prey_y=prey_y,
        energy=jnp.ones(N),
        alive=jnp.ones(N, dtype=jnp.bool_),
        zone_damage=jnp.zeros(N),
        prev_energy=jnp.ones(N),
        memory=mem,
        ticks_alive=jnp.zeros(N, dtype=jnp.int32),
        food_eaten=jnp.zeros(N, dtype=jnp.int32),
        weights=weights,
        base_hidden=base_hidden,
        signal_hidden=signal_hidden,
        zone_x=zx,
        zone_y=zy,
        zone_radius=jnp.full(num_zones, zone_radius),
        zone_speed=jnp.full(num_zones, zone_speed),
        food_x=fx,
        food_y=fy,
        food_is_patch=fp,
        food_alive=jnp.ones(food_count, dtype=jnp.bool_),
        food_count=jnp.int32(food_count),
        sig_x=jnp.zeros(max_signals, dtype=jnp.int32),
        sig_y=jnp.zeros(max_signals, dtype=jnp.int32),
        sig_symbol=jnp.zeros(max_signals, dtype=jnp.int32),
        sig_tick=jnp.zeros(max_signals, dtype=jnp.int32),
        sig_valid=jnp.zeros(max_signals, dtype=jnp.bool_),
        sig_cursor=jnp.int32(0),
        tick=jnp.int32(0),
        signals_emitted=jnp.int32(0),
        zone_deaths=jnp.int32(0),
        key=key,
        # Metrics accumulators
        mi_counts=jnp.zeros((NUM_SYMBOLS, 4), dtype=jnp.int32),
        symbol_counts=jnp.zeros(NUM_SYMBOLS, dtype=jnp.int32),
        iconicity_in_zone=jnp.int32(0),
        m_ticks_in_zone=jnp.int32(0),
        m_total_prey_ticks=jnp.int32(0),
        recv_counts=jnp.zeros((1 + NUM_SYMBOLS, 2, 5), dtype=jnp.int32),
        per_tick_signals=jnp.zeros(ticks_per_eval, dtype=jnp.float32),
        per_tick_alive=jnp.zeros(ticks_per_eval, dtype=jnp.float32),
        per_tick_min_zdist=jnp.zeros(ticks_per_eval, dtype=jnp.float32),
        prey_signals_sent=jnp.zeros(N, dtype=jnp.int32),
        prey_recv_with=jnp.zeros((N, 2, 5), dtype=jnp.int32),
        prey_recv_without=jnp.zeros((N, 2, 5), dtype=jnp.int32),
        was_hearing=jnp.zeros(N, dtype=jnp.bool_),
        onset_actions=jnp.zeros((N, 2, 5), dtype=jnp.int32),
        present_actions=jnp.zeros((N, 2, 5), dtype=jnp.int32),
        evt_symbol=jnp.zeros(max_events, dtype=jnp.int32),
        evt_inputs=jnp.zeros((max_events, INPUTS), dtype=jnp.float32),
        evt_cursor=jnp.int32(0),
        evt_count=jnp.int32(0),
    )


PREY_MAX_PER_CELL = 32


def build_inputs(
    state: WorldState,
    grid_size: int,
    signal_range: float,
    prey_cells: jnp.ndarray | None = None,
    prey_counts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build (N, 36) input vectors for all prey.

    Layout: [zone_damage, energy_delta, spare, food(3), ally(3), signals(18), memory(8), energy]
    """
    N = state.prey_x.shape[0]
    gs = float(grid_size)
    inputs = jnp.zeros((N, INPUTS))

    # 0: zone_damage
    inputs = inputs.at[:, 0].set(state.zone_damage)

    # 1: energy_delta
    inputs = inputs.at[:, 1].set(state.energy - state.prev_energy)

    # 2: spare (zero)

    # 3-5: nearest food (dx, dy, distance)
    alive_food_x = jnp.where(state.food_alive, state.food_x, -grid_size * 10)
    alive_food_y = jnp.where(state.food_alive, state.food_y, -grid_size * 10)
    food_idx, food_dx, food_dy = nearest_in_grid(
        state.prey_x, state.prey_y,
        alive_food_x, alive_food_y,
        None, None,  # cells/counts unused in brute force
        grid_size, grid_size // 2,
        jnp.full(N, -1, dtype=jnp.int32),  # don't skip any food
    )
    food_dist = jnp.sqrt(food_dx ** 2 + food_dy ** 2)
    has_food = food_idx >= 0
    inputs = inputs.at[:, 3].set(jnp.where(has_food, food_dx / gs, 0.0))
    inputs = inputs.at[:, 4].set(jnp.where(has_food, food_dy / gs, 0.0))
    inputs = inputs.at[:, 5].set(jnp.where(has_food, jnp.minimum(food_dist / gs, 1.0), 0.0))

    # 6-8: nearest ally (dx, dy, distance) - uses cell grid to avoid O(N^2)
    prey_indices = jnp.arange(N, dtype=jnp.int32)
    alive_prey_x = jnp.where(state.alive, state.prey_x, -grid_size * 10)
    alive_prey_y = jnp.where(state.alive, state.prey_y, -grid_size * 10)
    ally_idx, ally_dx, ally_dy = nearest_in_grid(
        state.prey_x, state.prey_y,
        alive_prey_x, alive_prey_y,
        prey_cells, prey_counts,
        grid_size, grid_size // 2,
        prey_indices,  # skip self
    )
    ally_dist = jnp.sqrt(ally_dx ** 2 + ally_dy ** 2)
    has_ally = ally_idx >= 0
    inputs = inputs.at[:, 6].set(jnp.where(has_ally, ally_dx / gs, 0.0))
    inputs = inputs.at[:, 7].set(jnp.where(has_ally, ally_dy / gs, 0.0))
    inputs = inputs.at[:, 8].set(jnp.where(has_ally, jnp.minimum(ally_dist / gs, 1.0), 1.0))

    # 9-26: signal inputs (6 symbols * 3 = 18)
    sig_inputs = receive_signals_grid(
        state.prey_x, state.prey_y,
        state.sig_x, state.sig_y, state.sig_symbol, state.sig_tick,
        state.sig_valid, state.tick,
        grid_size, signal_range,
    )
    inputs = inputs.at[:, SIGNAL_INPUT_START:SIGNAL_INPUT_START + NUM_SYMBOLS * 3].set(sig_inputs)

    # 27-34: memory
    inputs = inputs.at[:, MEMORY_INPUT_START:MEMORY_INPUT_START + MEMORY_SIZE].set(state.memory)

    # 35: own energy
    inputs = inputs.at[:, ENERGY_INPUT_IDX].set(jnp.clip(state.energy, 0.0, 1.0))

    return inputs


def step(
    state: WorldState,
    grid_size: int,
    signal_range: float,
    base_drain: float,
    signal_cost: float,
    zone_drain_rate: float,
    patch_ratio: float,
    food_count: int,
    signal_ticks: int,
    no_signals: bool,
    max_signals: int,
    mi_bin0: float,
    mi_bin1: float,
    mi_bin2: float,
    zone_radius_scalar: float,
    max_events: int,
) -> WorldState:
    """Execute one tick. All jittable."""
    key, k1, k2, k3, k4 = jax.random.split(state.key, 5)

    tick = state.tick + 1

    # Expire old signals
    sig_valid = state.sig_valid & ((tick - state.sig_tick) <= signal_ticks)

    state = state._replace(tick=tick, sig_valid=sig_valid, key=key)

    # Snapshot energy for delta
    state = state._replace(prev_energy=state.energy)

    # Metabolism
    energy = state.energy - base_drain
    alive = state.alive & (energy > 0)
    energy = jnp.where(alive, energy, 0.0)
    state = state._replace(energy=energy, alive=alive)

    # Zone edge distance for all prey (before zone movement)
    zone_edge_dists = nearest_zone_edge_dist(
        state.prey_x, state.prey_y,
        state.zone_x, state.zone_y, state.zone_radius,
        grid_size,
    )

    # Build prey cell grid (1x1 cells) - used for ally nearest + cooperative eating
    prey_cells, prey_counts = build_cell_grid(
        state.prey_x, state.prey_y, state.alive, grid_size, PREY_MAX_PER_CELL,
    )

    # Build inputs and run brains
    inputs = build_inputs(state, grid_size, signal_range, prey_cells, prey_counts)

    actions_raw, signals_raw, mem_write = batched_forward(
        state.weights, state.base_hidden, state.signal_hidden, inputs
    )

    # Action selection: argmax
    action = jnp.argmax(actions_raw, axis=-1)  # (N,)

    # -- Metrics: receiver action counts --
    N = state.prey_x.shape[0]
    sig_strengths = inputs[:, SIGNAL_INPUT_START::3][:, :NUM_SYMBOLS]  # (N, 6)
    max_strength = jnp.max(sig_strengths, axis=1)
    is_hearing = max_strength > 0
    dominant_sym = jnp.argmax(sig_strengths, axis=1)
    signal_state = jnp.where(is_hearing, dominant_sym + 1, 0)  # 0=none, 1-6=sym
    in_zone = zone_edge_dists < 0
    context = in_zone.astype(jnp.int32)

    # Population-level recv_counts: (7, 2, 5) += one_hot contributions
    state_oh = jax.nn.one_hot(signal_state, 1 + NUM_SYMBOLS)  # (N, 7)
    ctx_oh = jax.nn.one_hot(context, 2)  # (N, 2)
    act_oh = jax.nn.one_hot(action, 5)  # (N, 5)
    alive_f = alive.astype(jnp.float32)
    recv_contrib = (
        state_oh[:, :, None, None] * ctx_oh[:, None, :, None] * act_oh[:, None, None, :]
        * alive_f[:, None, None, None]
    )
    recv_update = jnp.sum(recv_contrib, axis=0).astype(jnp.int32)

    # Per-prey with/without signal
    per_prey_act_ctx = ctx_oh[:, :, None] * act_oh[:, None, :]  # (N, 2, 5)
    hearing_mask = (is_hearing & alive).astype(jnp.float32)[:, None, None]
    not_hearing_mask = (~is_hearing & alive).astype(jnp.float32)[:, None, None]

    # Silence onset detection
    now_hearing = is_hearing & alive
    silence_onset = state.was_hearing & ~now_hearing & alive
    onset_contrib = (per_prey_act_ctx * silence_onset.astype(jnp.float32)[:, None, None]).astype(jnp.int32)
    present_contrib = (per_prey_act_ctx * now_hearing.astype(jnp.float32)[:, None, None]).astype(jnp.int32)

    state = state._replace(
        recv_counts=state.recv_counts + recv_update,
        prey_recv_with=state.prey_recv_with + (per_prey_act_ctx * hearing_mask).astype(jnp.int32),
        prey_recv_without=state.prey_recv_without + (per_prey_act_ctx * not_hearing_mask).astype(jnp.int32),
        was_hearing=now_hearing,
        onset_actions=state.onset_actions + onset_contrib,
        present_actions=state.present_actions + present_contrib,
    )

    # Movement: all simultaneous (0=up, 1=down, 2=right, 3=left, 4=eat)
    dy = jnp.where(action == 0, -1, jnp.where(action == 1, 1, 0))
    dx = jnp.where(action == 2, 1, jnp.where(action == 3, -1, 0))
    new_x = (state.prey_x + jnp.where(alive, dx, 0)) % grid_size
    new_y = (state.prey_y + jnp.where(alive, dy, 0)) % grid_size

    state = state._replace(prey_x=new_x, prey_y=new_y)

    # Eating: find nearest food for eaters, resolve conflicts via priority scatter
    is_eating = alive & (action == 4)

    # Find nearest food for each eater
    alive_food_x = jnp.where(state.food_alive, state.food_x, -grid_size * 10)
    alive_food_y = jnp.where(state.food_alive, state.food_y, -grid_size * 10)
    N = state.prey_x.shape[0]
    food_target, _, _ = nearest_in_grid(
        state.prey_x, state.prey_y,
        alive_food_x, alive_food_y,
        None, None,
        grid_size, 1,  # eat range = 1 cell
        jnp.full(N, -1, dtype=jnp.int32),
    )

    # Check distance <= 1 for eating
    eat_dx = wrap_delta(state.prey_x, state.food_x[jnp.clip(food_target, 0)], grid_size)
    eat_dy = wrap_delta(state.prey_y, state.food_y[jnp.clip(food_target, 0)], grid_size)
    eat_dist = jnp.abs(eat_dx) + jnp.abs(eat_dy)
    can_eat = is_eating & (food_target >= 0) & (eat_dist <= 1)

    # Patch food: need 2+ nearby prey (Chebyshev distance 2)
    # For simplicity, check if any other alive prey within Chebyshev 2
    is_patch = state.food_is_patch[jnp.clip(food_target, 0)]

    # Cooperative check: for each eating prey, is there another alive prey within Chebyshev 2?
    prey_indices = jnp.arange(N, dtype=jnp.int32)
    has_partner = has_neighbor_in_radius(
        state.prey_x, state.prey_y,
        prey_cells,
        state.prey_x, state.prey_y, state.alive,
        grid_size, 2,
        prey_indices,
    )

    can_eat = can_eat & (~is_patch | has_partner)

    # Conflict resolution: shuffled priority (random permutation)
    # For each food item, only the first prey to claim it gets it
    priority = jax.random.permutation(k1, N)
    F = state.food_x.shape[0]
    food_best_priority = jnp.full(F, N + 1, dtype=jnp.int32)
    clipped_target = jnp.clip(food_target, 0, F - 1)

    # Scatter min priority per food - lowest priority value wins
    food_best_priority = food_best_priority.at[clipped_target].min(
        jnp.where(can_eat, priority, N + 1)
    )
    won_food = can_eat & (priority == food_best_priority[clipped_target])

    # Apply eating
    new_energy = jnp.where(won_food, jnp.minimum(state.energy + 0.3, 1.0), state.energy)
    new_food_eaten = state.food_eaten + won_food.astype(jnp.int32)
    # Mark eaten food via boolean scatter (clipped_target defaults to 0 for non-eaters,
    # so use won_food mask to avoid corrupting food 0)
    eaten_mask = jnp.zeros(F, dtype=jnp.bool_)
    eaten_mask = eaten_mask.at[clipped_target].set(
        eaten_mask[clipped_target] | won_food
    )
    new_food_alive = state.food_alive & ~eaten_mask

    state = state._replace(
        energy=new_energy,
        food_eaten=new_food_eaten,
        food_alive=new_food_alive,
    )

    # Signal emission
    symbol_indices, emit_mask = softmax_emit(signals_raw)
    can_emit = alive & emit_mask & (state.energy > signal_cost) & (not no_signals)
    emit_cost = jnp.where(can_emit, signal_cost, 0.0)
    state = state._replace(energy=state.energy - emit_cost)

    # Write signals to ring buffer - parallel scatter via cumsum
    emit_count = jnp.sum(can_emit.astype(jnp.int32))
    sig_local_idx = jnp.cumsum(can_emit.astype(jnp.int32)) - 1
    sig_write_pos = (state.sig_cursor + sig_local_idx) % max_signals

    new_sig_x = state.sig_x.at[sig_write_pos].set(
        jnp.where(can_emit, state.prey_x, state.sig_x[sig_write_pos]))
    new_sig_y = state.sig_y.at[sig_write_pos].set(
        jnp.where(can_emit, state.prey_y, state.sig_y[sig_write_pos]))
    new_sig_sym = state.sig_symbol.at[sig_write_pos].set(
        jnp.where(can_emit, symbol_indices, state.sig_symbol[sig_write_pos]))
    new_sig_tick = state.sig_tick.at[sig_write_pos].set(
        jnp.where(can_emit, tick, state.sig_tick[sig_write_pos]))
    new_sig_valid = state.sig_valid.at[sig_write_pos].set(
        jnp.where(can_emit, True, state.sig_valid[sig_write_pos]))
    new_cursor = state.sig_cursor + emit_count

    state = state._replace(
        sig_x=new_sig_x, sig_y=new_sig_y, sig_symbol=new_sig_sym,
        sig_tick=new_sig_tick, sig_valid=new_sig_valid, sig_cursor=new_cursor,
        signals_emitted=state.signals_emitted + emit_count,
    )

    # -- Metrics: signal emission counts --
    # MI contingency table: bin zone_edge_dist for emitters
    emitter_zone_bin = jnp.where(
        zone_edge_dists < mi_bin0, 0,
        jnp.where(zone_edge_dists < mi_bin1, 1,
                  jnp.where(zone_edge_dists < mi_bin2, 2, 3)))
    sym_oh = jax.nn.one_hot(symbol_indices, NUM_SYMBOLS)  # (N, 6)
    bin_oh = jax.nn.one_hot(emitter_zone_bin, 4)  # (N, 4)
    emit_f = can_emit.astype(jnp.float32)
    mi_update = jnp.sum(
        sym_oh[:, :, None] * bin_oh[:, None, :] * emit_f[:, None, None],
        axis=0,
    ).astype(jnp.int32)

    sym_update = jnp.sum(sym_oh * emit_f[:, None], axis=0).astype(jnp.int32)
    in_zone_signal = can_emit & (zone_edge_dists <= zone_radius_scalar)
    iconicity_update = jnp.sum(in_zone_signal)

    # Event buffer for input_mi: scatter emitter symbols + inputs
    emitter_count = jnp.sum(can_emit)
    local_idx = jnp.cumsum(can_emit.astype(jnp.int32)) - 1
    write_pos = (state.evt_cursor + local_idx) % max_events
    new_evt_symbol = state.evt_symbol.at[write_pos].set(
        jnp.where(can_emit, symbol_indices, state.evt_symbol[write_pos])
    )
    new_evt_inputs = state.evt_inputs.at[write_pos].set(
        jnp.where(can_emit[:, None], inputs, state.evt_inputs[write_pos])
    )

    state = state._replace(
        mi_counts=state.mi_counts + mi_update,
        symbol_counts=state.symbol_counts + sym_update,
        iconicity_in_zone=state.iconicity_in_zone + iconicity_update,
        prey_signals_sent=state.prey_signals_sent + can_emit.astype(jnp.int32),
        evt_symbol=new_evt_symbol,
        evt_inputs=new_evt_inputs,
        evt_cursor=state.evt_cursor + emitter_count,
        evt_count=state.evt_count + emitter_count,
    )

    # Memory EMA update: new = 0.9 * old + 0.1 * output
    new_memory = jnp.where(
        alive[:, None],
        0.9 * state.memory + 0.1 * mem_write,
        state.memory,
    )
    state = state._replace(memory=new_memory)

    # Ticks alive
    state = state._replace(ticks_alive=state.ticks_alive + alive.astype(jnp.int32))

    # Move zones
    new_zx, new_zy = move_zones(state.zone_x, state.zone_y, state.zone_speed, grid_size, k2)
    state = state._replace(zone_x=new_zx, zone_y=new_zy)

    # Zone drain
    drain = zone_drain_amount(
        state.prey_x, state.prey_y,
        state.zone_x, state.zone_y, state.zone_radius,
        grid_size, zone_drain_rate,
    )
    new_zone_damage = state.zone_damage + jnp.where(alive, drain, 0.0)
    zone_killed = alive & (new_zone_damage >= 1.0)
    new_alive = state.alive & ~zone_killed
    state = state._replace(
        zone_damage=new_zone_damage,
        alive=new_alive,
        zone_deaths=state.zone_deaths + jnp.sum(zone_killed),
    )

    # -- Metrics: per-tick and zone stats --
    tick_idx = tick - 1  # tick is 1-based
    alive_now = state.alive
    alive_count_f = jnp.sum(alive_now).astype(jnp.float32)
    alive_zone_dist = jnp.where(alive_now, zone_edge_dists, jnp.float32(1e10))
    min_zd = jnp.min(alive_zone_dist)

    state = state._replace(
        per_tick_signals=state.per_tick_signals.at[tick_idx].set(emit_count.astype(jnp.float32)),
        per_tick_alive=state.per_tick_alive.at[tick_idx].set(alive_count_f),
        per_tick_min_zdist=state.per_tick_min_zdist.at[tick_idx].set(min_zd),
        m_ticks_in_zone=state.m_ticks_in_zone + jnp.sum(alive_now & in_zone),
        m_total_prey_ticks=state.m_total_prey_ticks + jnp.sum(alive_now),
    )

    # Food refill
    current_food = jnp.sum(state.food_alive)
    need_refill = current_food < (food_count // 2)
    # When refilling, regenerate all dead food slots
    new_food_x = state.food_x
    new_food_y = state.food_y
    new_food_patch = state.food_is_patch
    new_food_alive_arr = state.food_alive

    # Only refill when below threshold
    k3a, k3b, k3c = jax.random.split(k3, 3)
    refill_x = jax.random.randint(k3a, state.food_x.shape, 0, grid_size)
    refill_y = jax.random.randint(k3b, state.food_y.shape, 0, grid_size)
    refill_patch = jax.random.uniform(k3c, state.food_x.shape) < patch_ratio

    new_food_x = jnp.where(need_refill & ~state.food_alive, refill_x, state.food_x)
    new_food_y = jnp.where(need_refill & ~state.food_alive, refill_y, state.food_y)
    new_food_patch = jnp.where(need_refill & ~state.food_alive, refill_patch, state.food_is_patch)
    new_food_alive_arr = jnp.where(need_refill, jnp.ones_like(state.food_alive), state.food_alive)

    state = state._replace(
        food_x=new_food_x, food_y=new_food_y,
        food_is_patch=new_food_patch, food_alive=new_food_alive_arr,
    )

    return state


class EvalResult(NamedTuple):
    """Results from one generation evaluation."""
    fitness: jnp.ndarray          # (N,) float32
    ticks_alive: jnp.ndarray      # (N,) int32
    food_eaten: jnp.ndarray       # (N,) int32
    total_signals: jnp.ndarray    # scalar int32
    zone_deaths: jnp.ndarray      # scalar int32
    final_state: WorldState       # for extracting signal events etc on host


@functools.partial(jax.jit, static_argnames=[
    'grid_size', 'signal_range', 'base_drain', 'signal_cost',
    'zone_drain_rate', 'patch_ratio', 'food_count', 'signal_ticks',
    'no_signals', 'max_signals', 'ticks_per_eval', 'mi_bins',
    'zone_radius_scalar', 'max_events',
])
def evaluate_generation(
    state: WorldState,
    grid_size: int,
    signal_range: float,
    base_drain: float,
    signal_cost: float,
    zone_drain_rate: float,
    patch_ratio: float,
    food_count: int,
    signal_ticks: int,
    no_signals: bool,
    max_signals: int,
    ticks_per_eval: int,
    mi_bins: tuple[float, float, float] = (8.0, 8.0, 11.0),
    zone_radius_scalar: float = 8.0,
    max_events: int = 100_000,
) -> EvalResult:
    """Run one generation: ticks_per_eval ticks via lax.fori_loop.

    Returns fitness and collected metrics.
    """
    def body_fn(_, s):
        return step(
            s,
            grid_size=grid_size,
            signal_range=signal_range,
            base_drain=base_drain,
            signal_cost=signal_cost,
            zone_drain_rate=zone_drain_rate,
            patch_ratio=patch_ratio,
            food_count=food_count,
            signal_ticks=signal_ticks,
            no_signals=no_signals,
            max_signals=max_signals,
            mi_bin0=mi_bins[0],
            mi_bin1=mi_bins[1],
            mi_bin2=mi_bins[2],
            zone_radius_scalar=zone_radius_scalar,
            max_events=max_events,
        )

    final = jax.lax.fori_loop(0, ticks_per_eval, body_fn, state)

    # Fitness: ticks_alive * (1 + food_eaten * 0.1) - matches Rust
    fitness = final.ticks_alive.astype(jnp.float32) * (
        1.0 + final.food_eaten.astype(jnp.float32) * 0.1
    )

    return EvalResult(
        fitness=fitness,
        ticks_alive=final.ticks_alive,
        food_eaten=final.food_eaten,
        total_signals=final.signals_emitted,
        zone_deaths=final.zone_deaths,
        final_state=final,
    )
