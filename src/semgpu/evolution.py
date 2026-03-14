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
from semgpu.spatial import wrap_dist_sq

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
    # Use vectorized approach: for each offspring slot, pick parents from nearby agents

    # Pairwise distances for spatial selection
    dist_sq = wrap_dist_sq(
        sorted_x[:, None], sorted_y[:, None],
        sorted_x[None, :], sorted_y[None, :],
        grid_size,
    ).astype(jnp.float32)

    repro_sq = reproduction_radius ** 2
    fallback_sq = fallback_radius ** 2

    # For each slot, determine which agents are within radius
    nearby_repro = dist_sq <= repro_sq  # (N, N)
    nearby_fallback = dist_sq <= fallback_sq

    def select_parent_for_slot(key, slot_idx):
        """Select one parent for offspring at slot_idx position."""
        k1, k2, k3 = jax.random.split(key, 3)

        # Try reproduction radius first
        candidates_repro = nearby_repro[slot_idx]
        n_repro = jnp.sum(candidates_repro)

        # Try fallback radius
        candidates_fallback = nearby_fallback[slot_idx]
        n_fallback = jnp.sum(candidates_fallback)

        # Global fallback: all candidates
        candidates_global = jnp.ones(N, dtype=jnp.bool_)

        # Pick which candidate set to use
        use_repro = n_repro >= 2
        use_fallback = (~use_repro) & (n_fallback >= 2)
        use_global = (~use_repro) & (~use_fallback)

        candidates = jnp.where(use_repro, candidates_repro,
                               jnp.where(use_fallback, candidates_fallback, candidates_global))

        # Tournament selection within candidates
        # Sample tournament_size random candidates, pick best
        # Use Gumbel trick for weighted sampling
        # Assign -inf to non-candidates
        log_probs = jnp.where(candidates, 0.0, -1e10)
        gumbel = jax.random.gumbel(k1, (tournament_size, N))
        perturbed = log_probs[None, :] + gumbel  # (tournament_size, N)
        tournament_picks = jnp.argmax(perturbed, axis=1)  # (tournament_size,)
        tournament_fitness = sorted_fitness[tournament_picks]
        best = jnp.argmax(tournament_fitness)
        return tournament_picks[best]

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

        pa = select_parent_for_slot(k1, slot)
        pb = select_parent_for_slot(k2, slot)

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
    """Add kin fitness bonus: +0.5 for siblings, +0.25 for cousins, scaled by kin_bonus.

    Args:
        fitness: (N,) raw fitness
        parent_indices: (N, 2) int, -1 = no parent
        grandparent_indices: (N, 4) int, -1 = no grandparent
        kin_bonus: scaling factor

    Returns:
        (N,) adjusted fitness
    """
    N = fitness.shape[0]

    # Sibling detection: shared parent indices
    # (N, N) matrix of whether any parent matches
    pa = parent_indices  # (N, 2)
    # Check if any parent of i matches any parent of j (and both are valid)
    sibling = jnp.zeros((N, N), dtype=jnp.bool_)
    for pi in range(2):
        for pj in range(2):
            valid = (pa[:, pi, None] >= 0) & (pa[None, :, pj] >= 0)
            matches = (pa[:, pi, None] == pa[None, :, pj]) & valid
            sibling = sibling | matches

    # Cousin detection: shared grandparent indices
    ga = grandparent_indices  # (N, 4)
    cousin = jnp.zeros((N, N), dtype=jnp.bool_)
    for gi in range(4):
        for gj in range(4):
            valid = (ga[:, gi, None] >= 0) & (ga[None, :, gj] >= 0)
            matches = (ga[:, gi, None] == ga[None, :, gj]) & valid
            cousin = cousin | matches

    # Remove self-relatedness
    self_mask = jnp.eye(N, dtype=jnp.bool_)
    sibling = sibling & ~self_mask
    cousin = cousin & ~self_mask & ~sibling

    # Sum kin fitness: siblings contribute 0.5, cousins 0.25
    kin_sum = (
        jnp.sum(jnp.where(sibling, fitness[None, :], 0.0), axis=1) * 0.5
        + jnp.sum(jnp.where(cousin, fitness[None, :], 0.0), axis=1) * 0.25
    )

    return fitness + kin_bonus * kin_sum
