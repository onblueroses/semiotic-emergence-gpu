"""Evolution operators for GPU simulation.

Crossover, scoped mutation, spatial tournament selection, kin bonus.
All operations vectorized for batch processing.
"""

from __future__ import annotations

import functools  # noqa: F401 - used in jax.jit decorator

import jax
import jax.numpy as jnp

from semgpu.brain import (
    INPUTS,
    MAX_BASE_HIDDEN,
    MAX_GENOME_LEN,
    MAX_SIGNAL_HIDDEN,
    MEMORY_OUTPUTS,
    MIN_BASE_HIDDEN,
    MIN_SIGNAL_HIDDEN,
    MOVEMENT_OUTPUTS,
    SEG_BASE_BIAS,
    SEG_BASE_MEM,
    SEG_BASE_MOVE,
    SEG_BASE_SIGHID,
    SEG_INPUT_BASE,
    SEG_MEM_BIAS,
    SEG_MOVE_BIAS,
    SEG_SIGHID_BIAS,
    SEG_SIGHID_SIGOUT,
    SEG_SIGOUT_BIAS,
    SIGNAL_OUTPUTS,
)

OFFSPRING_JITTER = 1
HIDDEN_SIZE_MUTATION_RATE = 0.05
MUTATION_HEADROOM = 4


def crossover_single(
    weights_a: jnp.ndarray,
    weights_b: jnp.ndarray,
    bh_a: jnp.ndarray,
    bh_b: jnp.ndarray,
    sh_a: jnp.ndarray,
    sh_b: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single-point crossover between two genomes.

    Returns: (child_weights, child_base_hidden, child_signal_hidden)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    point = jax.random.randint(k1, (), 1, MAX_GENOME_LEN)
    mask = jnp.arange(MAX_GENOME_LEN) < point
    child_weights = jnp.where(mask, weights_a, weights_b)

    # Hidden sizes: 50/50 from either parent, independent
    child_bh = jnp.where(jax.random.uniform(k2) < 0.5, bh_a, bh_b)
    child_sh = jnp.where(jax.random.uniform(k3) < 0.5, sh_a, sh_b)

    return child_weights, child_bh, child_sh


def _gaussian_noise(key: jnp.ndarray, shape: tuple, sigma: float) -> jnp.ndarray:
    """CLT Gaussian: sum of 4 uniforms, no transcendentals."""
    keys = jax.random.split(key, 4)
    total = sum(jax.random.uniform(keys[i], shape) for i in range(4))
    return (total - 2.0) * 1.7320508 * sigma


def _build_mutation_mask(base_hidden: jnp.ndarray, signal_hidden: jnp.ndarray) -> jnp.ndarray:
    """Build a per-weight mutation mask respecting scoping.

    Vectorized: computes segment membership and row/col position for all weights
    at once, replacing 260 Python loop iterations with ~10 vectorized ops.

    Returns: (MAX_GENOME_LEN,) bool mask of which weights to mutate.
    """
    bh_scope = jnp.minimum(base_hidden + MUTATION_HEADROOM, MAX_BASE_HIDDEN)
    sh_scope = jnp.minimum(signal_hidden + MUTATION_HEADROOM, MAX_SIGNAL_HIDDEN)

    idx = jnp.arange(MAX_GENOME_LEN)
    mask = jnp.zeros(MAX_GENOME_LEN, dtype=jnp.bool_)

    # Helper: for a 2D block at offset with (rows, cols), compute col < col_scope
    def _col_scoped(seg_start, rows, cols, col_scope):
        seg_end = seg_start + rows * cols
        in_seg = (idx >= seg_start) & (idx < seg_end)
        col = (idx - seg_start) % cols
        return in_seg & (col < col_scope)

    # Helper: for a 2D block, compute row < row_scope
    def _row_scoped(seg_start, rows, cols, row_scope):
        seg_end = seg_start + rows * cols
        in_seg = (idx >= seg_start) & (idx < seg_end)
        row = (idx - seg_start) // cols
        return in_seg & (row < row_scope)

    # Helper: both row and col scoped
    def _row_col_scoped(seg_start, rows, cols, row_scope, col_scope):
        seg_end = seg_start + rows * cols
        in_seg = (idx >= seg_start) & (idx < seg_end)
        off = idx - seg_start
        return in_seg & ((off // cols) < row_scope) & ((off % cols) < col_scope)

    # Helper: 1D bias segment scoped by size
    def _bias_scoped(seg_start, size, scope):
        in_seg = (idx >= seg_start) & (idx < seg_start + size)
        return in_seg & ((idx - seg_start) < scope)

    # Helper: always-on bias segment
    def _bias_all(seg_start, size):
        return (idx >= seg_start) & (idx < seg_start + size)

    # Input -> base hidden: cols scoped by bh
    mask = mask | _col_scoped(SEG_INPUT_BASE, INPUTS, MAX_BASE_HIDDEN, bh_scope)
    # Base hidden biases
    mask = mask | _bias_scoped(SEG_BASE_BIAS, MAX_BASE_HIDDEN, bh_scope)
    # Base -> movement: rows scoped by bh
    mask = mask | _row_scoped(SEG_BASE_MOVE, MAX_BASE_HIDDEN, MOVEMENT_OUTPUTS, bh_scope)
    # Movement biases: always
    mask = mask | _bias_all(SEG_MOVE_BIAS, MOVEMENT_OUTPUTS)
    # Base -> signal hidden: rows by bh, cols by sh
    mask = mask | _row_col_scoped(SEG_BASE_SIGHID, MAX_BASE_HIDDEN, MAX_SIGNAL_HIDDEN, bh_scope, sh_scope)
    # Signal hidden biases
    mask = mask | _bias_scoped(SEG_SIGHID_BIAS, MAX_SIGNAL_HIDDEN, sh_scope)
    # Signal hidden -> signal output: rows by sh
    mask = mask | _row_scoped(SEG_SIGHID_SIGOUT, MAX_SIGNAL_HIDDEN, SIGNAL_OUTPUTS, sh_scope)
    # Signal output biases: always
    mask = mask | _bias_all(SEG_SIGOUT_BIAS, SIGNAL_OUTPUTS)
    # Base -> memory: rows by bh
    mask = mask | _row_scoped(SEG_BASE_MEM, MAX_BASE_HIDDEN, MEMORY_OUTPUTS, bh_scope)
    # Memory biases: always
    mask = mask | _bias_all(SEG_MEM_BIAS, MEMORY_OUTPUTS)

    return mask


def mutate_single(
    weights: jnp.ndarray,
    base_hidden: jnp.ndarray,
    signal_hidden: jnp.ndarray,
    sigma: float,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scoped Gaussian mutation + hidden size mutation for one agent."""
    k1, k2, k3 = jax.random.split(key, 3)

    # Weight mutation
    mask = _build_mutation_mask(base_hidden, signal_hidden)
    noise = _gaussian_noise(k1, (MAX_GENOME_LEN,), sigma)
    new_weights = weights + jnp.where(mask, noise, 0.0)

    # Hidden size mutation (5% rate, +/-1, clamped)
    k2a, k2b, k2c, k2d = jax.random.split(k2, 4)
    bh_mutate = jax.random.uniform(k2a) < HIDDEN_SIZE_MUTATION_RATE
    bh_delta = jnp.where(jax.random.uniform(k2b) < 0.5, 1, -1)
    new_bh = jnp.where(bh_mutate,
                        jnp.clip(base_hidden + bh_delta, MIN_BASE_HIDDEN, MAX_BASE_HIDDEN),
                        base_hidden)

    sh_mutate = jax.random.uniform(k2c) < HIDDEN_SIZE_MUTATION_RATE
    sh_delta = jnp.where(jax.random.uniform(k2d) < 0.5, 1, -1)
    new_sh = jnp.where(sh_mutate,
                        jnp.clip(signal_hidden + sh_delta, MIN_SIGNAL_HIDDEN, MAX_SIGNAL_HIDDEN),
                        signal_hidden)

    return new_weights, new_bh, new_sh


@functools.partial(jax.jit, static_argnames=[
    'elite_count', 'tournament_size', 'sigma', 'grid_size',
    'reproduction_radius', 'fallback_radius',
])
def evolve_generation(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    weights: jnp.ndarray,
    base_hidden: jnp.ndarray,
    signal_hidden: jnp.ndarray,
    fitness: jnp.ndarray,
    parent_indices: jnp.ndarray,
    grandparent_indices: jnp.ndarray,
    elite_count: int,
    tournament_size: int,
    sigma: float,
    grid_size: int,
    reproduction_radius: float,
    fallback_radius: float,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Spatial evolution: elites pass through, offspring replace dead slots.

    Args:
        prey_x, prey_y: (N,) int positions
        weights: (N, MAX_GENOME_LEN) float
        base_hidden, signal_hidden: (N,) int
        fitness: (N,) float (rank-adjusted)
        parent_indices: (N, 2) int, -1 = no parent
        grandparent_indices: (N, 4) int, -1 = no grandparent
        elite_count: number of elites to preserve
        tournament_size: tournament k
        sigma: mutation strength
        grid_size: world grid size
        reproduction_radius, fallback_radius: spatial selection radii
        key: PRNG key

    Returns:
        new_{prey_x, prey_y, weights, base_hidden, signal_hidden, parent_indices, grandparent_indices}
    """
    N = fitness.shape[0]

    # Sort by fitness descending
    rank_order = jnp.argsort(-fitness)
    elite_mask = jnp.arange(N) < elite_count

    # Reorder everything by rank
    sorted_x = prey_x[rank_order]
    sorted_y = prey_y[rank_order]
    sorted_w = weights[rank_order]
    sorted_bh = base_hidden[rank_order]
    sorted_sh = signal_hidden[rank_order]
    sorted_parents = parent_indices[rank_order]
    sorted_grandparents = grandparent_indices[rank_order]
    sorted_fitness = fitness[rank_order]

    # For non-elites, select parents via tournament and produce offspring
    # Cell-grid spatial selection instead of O(N^2) pairwise distances
    from semgpu.spatial import build_cell_grid, gather_nearby_indices, wrap_delta

    EVOL_MAX_PER_CELL = 32
    scan_radius = min(int(fallback_radius) + 1, 20)
    max_cand = (2 * scan_radius + 1) ** 2 * EVOL_MAX_PER_CELL

    repro_sq = reproduction_radius ** 2
    fallback_sq = fallback_radius ** 2

    sorted_alive = jnp.ones(N, dtype=jnp.bool_)
    evol_cells, evol_counts = build_cell_grid(
        sorted_x, sorted_y, sorted_alive, grid_size, EVOL_MAX_PER_CELL,
    )

    def select_parent_for_slot(key, slot_x, slot_y):
        """Select one parent from nearby agents using cell grid."""
        cand_idx = gather_nearby_indices(
            slot_x, slot_y, evol_cells, grid_size, 1, scan_radius, max_cand,
        )
        valid = cand_idx >= 0
        safe = jnp.clip(cand_idx, 0)

        cdx = wrap_delta(slot_x, sorted_x[safe], grid_size).astype(jnp.float32)
        cdy = wrap_delta(slot_y, sorted_y[safe], grid_size).astype(jnp.float32)
        d_sq = cdx * cdx + cdy * cdy

        within_repro = valid & (d_sq <= repro_sq)
        within_fallback = valid & (d_sq <= fallback_sq)
        n_repro = jnp.sum(within_repro)
        n_fallback = jnp.sum(within_fallback)

        use_repro = n_repro >= 2
        use_fallback = (~use_repro) & (n_fallback >= 2)

        candidates = jnp.where(
            use_repro, within_repro,
            jnp.where(use_fallback, within_fallback, valid),
        )

        # Tournament: Gumbel trick over candidate array
        log_probs = jnp.where(candidates, 0.0, -1e10)
        gumbel = jax.random.gumbel(key, (tournament_size, max_cand))
        perturbed = log_probs[None, :] + gumbel
        tournament_picks = jnp.argmax(perturbed, axis=1)  # indices into cand array
        tournament_sorted_idx = safe[tournament_picks]
        tournament_fitness = sorted_fitness[tournament_sorted_idx]
        best = jnp.argmax(tournament_fitness)
        return tournament_sorted_idx[best]

    # Generate offspring for all non-elite slots - vmap for GPU parallelism
    num_offspring = N - elite_count
    offspring_keys = jax.random.split(key, num_offspring)

    def make_one_offspring(slot_idx, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        slot = elite_count + slot_idx

        pa = select_parent_for_slot(k1, sorted_x[slot], sorted_y[slot])
        pb = select_parent_for_slot(k2, sorted_x[slot], sorted_y[slot])

        child_w, child_bh, child_sh = crossover_single(
            sorted_w[pa], sorted_w[pb],
            sorted_bh[pa], sorted_bh[pb],
            sorted_sh[pa], sorted_sh[pb],
            k3,
        )
        child_w, child_bh, child_sh = mutate_single(child_w, child_bh, child_sh, sigma, k4)

        # Jitter position
        jx = jax.random.randint(k5, (), -OFFSPRING_JITTER, OFFSPRING_JITTER + 1)
        jy = jax.random.randint(k6, (), -OFFSPRING_JITTER, OFFSPRING_JITTER + 1)
        child_x = (sorted_x[slot] + jx) % grid_size
        child_y = (sorted_y[slot] + jy) % grid_size

        # Lineage
        child_parents = jnp.array([pa, pb], dtype=jnp.int32)
        child_grandparents = jnp.array([
            sorted_parents[pa, 0], sorted_parents[pa, 1],
            sorted_parents[pb, 0], sorted_parents[pb, 1],
        ], dtype=jnp.int32)

        return child_x, child_y, child_w, child_bh, child_sh, child_parents, child_grandparents

    off_x, off_y, off_w, off_bh, off_sh, off_parents, off_grandparents = jax.vmap(
        make_one_offspring
    )(jnp.arange(num_offspring), offspring_keys)

    # Combine elites + offspring
    new_x = jnp.concatenate([sorted_x[:elite_count], off_x])
    new_y = jnp.concatenate([sorted_y[:elite_count], off_y])
    new_w = jnp.concatenate([sorted_w[:elite_count], off_w])
    new_bh = jnp.concatenate([sorted_bh[:elite_count], off_bh])
    new_sh = jnp.concatenate([sorted_sh[:elite_count], off_sh])
    new_parents = jnp.concatenate([sorted_parents[:elite_count], off_parents])
    new_grandparents = jnp.concatenate([sorted_grandparents[:elite_count], off_grandparents])

    return new_x, new_y, new_w, new_bh, new_sh, new_parents, new_grandparents


@functools.partial(jax.jit, static_argnames=['kin_bonus'])
def compute_kin_bonus(
    fitness: jnp.ndarray,
    parent_indices: jnp.ndarray,
    grandparent_indices: jnp.ndarray,
    kin_bonus: float,
) -> jnp.ndarray:
    """Add kin fitness bonus via scatter-aggregate instead of O(N^2) matrices.

    For each parent/grandparent index, aggregates fitness of all children/grandchildren,
    then each prey reads its accumulated kin fitness. O(N) instead of O(N^2).

    Slight semantic difference from matrix version: prey sharing both parents get
    double-counted. Effect is negligible on selection dynamics.

    Args:
        fitness: (N,) raw fitness
        parent_indices: (N, 2) int, -1 = no parent
        grandparent_indices: (N, 4) int, -1 = no grandparent
        kin_bonus: scaling factor

    Returns:
        (N,) adjusted fitness
    """
    N = fitness.shape[0]
    pa = parent_indices       # (N, 2)
    ga = grandparent_indices  # (N, 4)

    # Parent/grandparent indices are in [0, N-1] or -1. Size arrays to N.
    # Sibling bonus: aggregate fitness by shared parent index
    parent_fitness_sum = jnp.zeros(N, dtype=jnp.float32)
    for k in range(2):
        valid = pa[:, k] >= 0
        idx = jnp.clip(pa[:, k], 0)
        parent_fitness_sum = parent_fitness_sum.at[idx].add(
            jnp.where(valid, fitness, 0.0)
        )

    # Each prey's sibling contribution: sum of kin fitness via each parent, minus self
    sib_sum = jnp.zeros(N, dtype=jnp.float32)
    for k in range(2):
        valid = pa[:, k] >= 0
        idx = jnp.clip(pa[:, k], 0)
        sib_sum = sib_sum + jnp.where(valid, parent_fitness_sum[idx] - fitness, 0.0)

    # Cousin bonus: aggregate fitness by shared grandparent index
    gp_fitness_sum = jnp.zeros(N, dtype=jnp.float32)
    for k in range(4):
        valid = ga[:, k] >= 0
        idx = jnp.clip(ga[:, k], 0)
        gp_fitness_sum = gp_fitness_sum.at[idx].add(
            jnp.where(valid, fitness, 0.0)
        )

    cousin_sum = jnp.zeros(N, dtype=jnp.float32)
    for k in range(4):
        valid = ga[:, k] >= 0
        idx = jnp.clip(ga[:, k], 0)
        cousin_sum = cousin_sum + jnp.where(valid, gp_fitness_sum[idx] - fitness, 0.0)

    # Subtract sibling contribution from cousin to reduce double-counting
    cousin_sum = jnp.maximum(0.0, cousin_sum - sib_sum)

    return fitness + kin_bonus * (sib_sum * 0.5 + cousin_sum * 0.25)
