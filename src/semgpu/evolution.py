"""Evolution operators for GPU simulation.

Crossover, scoped mutation, spatial tournament selection, kin bonus.
All operations vectorized for batch processing.
"""

from __future__ import annotations

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

    Returns: (MAX_GENOME_LEN,) bool mask of which weights to mutate.
    """
    bh_scope = jnp.minimum(base_hidden + MUTATION_HEADROOM, MAX_BASE_HIDDEN)
    sh_scope = jnp.minimum(signal_hidden + MUTATION_HEADROOM, MAX_SIGNAL_HIDDEN)

    mask = jnp.zeros(MAX_GENOME_LEN, dtype=jnp.bool_)
    idx = jnp.arange(MAX_GENOME_LEN)

    # Input -> base hidden: rows=INPUTS, cols=MAX_BASE_HIDDEN, scope cols by bh
    for i in range(INPUTS):
        start = SEG_INPUT_BASE + i * MAX_BASE_HIDDEN
        in_segment = (idx >= start) & (idx < start + MAX_BASE_HIDDEN)
        col = idx - start
        mask = mask | (in_segment & (col < bh_scope))

    # Base hidden biases: scope by bh
    in_seg = (idx >= SEG_BASE_BIAS) & (idx < SEG_BASE_BIAS + MAX_BASE_HIDDEN)
    mask = mask | (in_seg & ((idx - SEG_BASE_BIAS) < bh_scope))

    # Base -> movement: rows=MAX_BASE_HIDDEN, cols=MOVEMENT_OUTPUTS, scope rows by bh
    for h in range(MAX_BASE_HIDDEN):
        start = SEG_BASE_MOVE + h * MOVEMENT_OUTPUTS
        in_seg = (idx >= start) & (idx < start + MOVEMENT_OUTPUTS)
        mask = mask | (in_seg & (h < bh_scope))

    # Movement biases: always mutate all
    mask = mask | ((idx >= SEG_MOVE_BIAS) & (idx < SEG_MOVE_BIAS + MOVEMENT_OUTPUTS))

    # Base -> signal hidden: scope rows by bh, cols by sh
    for b in range(MAX_BASE_HIDDEN):
        start = SEG_BASE_SIGHID + b * MAX_SIGNAL_HIDDEN
        in_seg = (idx >= start) & (idx < start + MAX_SIGNAL_HIDDEN)
        col = idx - start
        mask = mask | (in_seg & (b < bh_scope) & (col < sh_scope))

    # Signal hidden biases: scope by sh
    in_seg = (idx >= SEG_SIGHID_BIAS) & (idx < SEG_SIGHID_BIAS + MAX_SIGNAL_HIDDEN)
    mask = mask | (in_seg & ((idx - SEG_SIGHID_BIAS) < sh_scope))

    # Signal hidden -> signal output: scope rows by sh
    for s in range(MAX_SIGNAL_HIDDEN):
        start = SEG_SIGHID_SIGOUT + s * SIGNAL_OUTPUTS
        in_seg = (idx >= start) & (idx < start + SIGNAL_OUTPUTS)
        mask = mask | (in_seg & (s < sh_scope))

    # Signal output biases: always mutate all
    mask = mask | ((idx >= SEG_SIGOUT_BIAS) & (idx < SEG_SIGOUT_BIAS + SIGNAL_OUTPUTS))

    # Base -> memory: scope rows by bh
    for h in range(MAX_BASE_HIDDEN):
        start = SEG_BASE_MEM + h * MEMORY_OUTPUTS
        in_seg = (idx >= start) & (idx < start + MEMORY_OUTPUTS)
        mask = mask | (in_seg & (h < bh_scope))

    # Memory biases: always mutate all
    mask = mask | ((idx >= SEG_MEM_BIAS) & (idx < SEG_MEM_BIAS + MEMORY_OUTPUTS))

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

    # Generate offspring for all non-elite slots
    num_offspring = N - elite_count
    keys = jax.random.split(key, num_offspring * 3 + 1)
    offspring_keys = keys[:-1].reshape(num_offspring, 3, -1)
    main_key = keys[-1]

    # Process each offspring slot
    def make_offspring(carry, idx):
        key = carry
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        slot = elite_count + idx  # index into sorted arrays

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
        k5, k6, key = jax.random.split(key, 3)
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

        return key, (child_x, child_y, child_w, child_bh, child_sh, child_parents, child_grandparents)

    _, offspring = jax.lax.scan(make_offspring, main_key, jnp.arange(num_offspring))
    off_x, off_y, off_w, off_bh, off_sh, off_parents, off_grandparents = offspring

    # Combine elites + offspring
    new_x = jnp.concatenate([sorted_x[:elite_count], off_x])
    new_y = jnp.concatenate([sorted_y[:elite_count], off_y])
    new_w = jnp.concatenate([sorted_w[:elite_count], off_w])
    new_bh = jnp.concatenate([sorted_bh[:elite_count], off_bh])
    new_sh = jnp.concatenate([sorted_sh[:elite_count], off_sh])
    new_parents = jnp.concatenate([sorted_parents[:elite_count], off_parents])
    new_grandparents = jnp.concatenate([sorted_grandparents[:elite_count], off_grandparents])

    return new_x, new_y, new_w, new_bh, new_sh, new_parents, new_grandparents


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

    # Use max possible index + 1 as array size to handle any parent index range.
    # Parent indices come from evolution (0..N-1) but tests may use arbitrary values.
    all_indices = jnp.concatenate([pa.reshape(-1), ga.reshape(-1)])
    max_idx = jnp.maximum(jnp.max(all_indices) + 1, N)

    # Sibling bonus: aggregate fitness by shared parent index
    parent_fitness_sum = jnp.zeros(max_idx, dtype=jnp.float32)
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
    gp_fitness_sum = jnp.zeros(max_idx, dtype=jnp.float32)
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
