"""Tests for evolution operators."""

import jax
import jax.numpy as jnp

from semgpu.brain import (
    DEFAULT_BASE_HIDDEN,
    DEFAULT_SIGNAL_HIDDEN,
    MAX_BASE_HIDDEN,
    MAX_GENOME_LEN,
    MIN_BASE_HIDDEN,
)
from semgpu.evolution import (
    compute_kin_bonus,
    crossover_single,
    evolve_generation,
    mutate_single,
)


def test_crossover_preserves_length():
    key = jax.random.key(42)
    k1, k2, k3 = jax.random.split(key, 3)
    wa = jax.random.normal(k1, (MAX_GENOME_LEN,))
    wb = jax.random.normal(k2, (MAX_GENOME_LEN,))
    child_w, child_bh, child_sh = crossover_single(
        wa, wb,
        jnp.int32(12), jnp.int32(20),
        jnp.int32(6), jnp.int32(16),
        k3,
    )
    assert child_w.shape == (MAX_GENOME_LEN,)
    assert int(child_bh) in (12, 20)
    assert int(child_sh) in (6, 16)


def test_crossover_mixes_weights():
    key = jax.random.key(7)
    wa = jnp.zeros(MAX_GENOME_LEN)
    wb = jnp.ones(MAX_GENOME_LEN)
    child_w, _, _ = crossover_single(
        wa, wb,
        jnp.int32(12), jnp.int32(12),
        jnp.int32(6), jnp.int32(6),
        key,
    )
    # Child should have some 0s (from a) and some 1s (from b)
    assert jnp.sum(child_w == 0) > 0
    assert jnp.sum(child_w == 1) > 0


def test_mutate_changes_weights():
    key = jax.random.key(42)
    w = jnp.zeros(MAX_GENOME_LEN)
    bh = jnp.int32(DEFAULT_BASE_HIDDEN)
    sh = jnp.int32(DEFAULT_SIGNAL_HIDDEN)
    new_w, new_bh, new_sh = mutate_single(w, bh, sh, 0.1, key)
    # Some weights should have changed
    assert not jnp.allclose(new_w, w)
    # Most should still be near zero with sigma=0.1
    assert jnp.max(jnp.abs(new_w)) < 2.0


def test_mutate_hidden_size_bounded():
    key = jax.random.key(123)
    w = jnp.zeros(MAX_GENOME_LEN)
    for _ in range(100):
        key, subkey = jax.random.split(key)
        _, bh, sh = mutate_single(w, jnp.int32(MIN_BASE_HIDDEN), jnp.int32(2), 0.1, subkey)
        assert int(bh) >= MIN_BASE_HIDDEN
        assert int(bh) <= MAX_BASE_HIDDEN
        assert int(sh) >= 2
        assert int(sh) <= 32


def test_evolve_preserves_population():
    key = jax.random.key(42)
    N = 20
    k1, k2, k3 = jax.random.split(key, 3)

    prey_x = jax.random.randint(k1, (N,), 0, 20)
    prey_y = jax.random.randint(k2, (N,), 0, 20)
    weights = jax.random.normal(k3, (N, MAX_GENOME_LEN)) * 0.5
    bh = jnp.full(N, DEFAULT_BASE_HIDDEN, dtype=jnp.int32)
    sh = jnp.full(N, DEFAULT_SIGNAL_HIDDEN, dtype=jnp.int32)
    fitness = jax.random.uniform(key, (N,))
    parents = jnp.full((N, 2), -1, dtype=jnp.int32)
    grandparents = jnp.full((N, 4), -1, dtype=jnp.int32)

    new_x, new_y, new_w, new_bh, new_sh, new_p, new_g = evolve_generation(
        prey_x, prey_y, weights, bh, sh, fitness,
        parents, grandparents,
        elite_count=3, tournament_size=3, sigma=0.1,
        grid_size=20,
        reproduction_radius=6.0, fallback_radius=10.0,
        key=key,
    )

    assert new_x.shape == (N,)
    assert new_w.shape == (N, MAX_GENOME_LEN)
    assert new_bh.shape == (N,)
    assert new_p.shape == (N, 2)


def test_elites_unchanged():
    key = jax.random.key(42)
    N = 10
    k1, k2, k3 = jax.random.split(key, 3)

    prey_x = jax.random.randint(k1, (N,), 0, 20)
    prey_y = jax.random.randint(k2, (N,), 0, 20)
    weights = jax.random.normal(k3, (N, MAX_GENOME_LEN))
    bh = jnp.full(N, DEFAULT_BASE_HIDDEN, dtype=jnp.int32)
    sh = jnp.full(N, DEFAULT_SIGNAL_HIDDEN, dtype=jnp.int32)
    # Give agent 5 the best fitness
    fitness = jnp.arange(N, dtype=jnp.float32)
    parents = jnp.full((N, 2), -1, dtype=jnp.int32)
    grandparents = jnp.full((N, 4), -1, dtype=jnp.int32)

    new_x, new_y, new_w, new_bh, new_sh, _, _ = evolve_generation(
        prey_x, prey_y, weights, bh, sh, fitness,
        parents, grandparents,
        elite_count=2, tournament_size=3, sigma=0.1,
        grid_size=20,
        reproduction_radius=6.0, fallback_radius=10.0,
        key=key,
    )

    # Top 2 elites (agents with highest fitness = agents N-1, N-2) should be unchanged
    # They're now at indices 0 and 1 after sorting
    assert jnp.allclose(new_w[0], weights[N - 1])
    assert jnp.allclose(new_w[1], weights[N - 2])


def test_kin_bonus():
    N = 6
    fitness = jnp.ones(N)
    # Agents 0 and 1 share parent 10 -> siblings
    parents = jnp.array([
        [10, 11], [10, 12], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
    ], dtype=jnp.int32)
    grandparents = jnp.full((N, 4), -1, dtype=jnp.int32)

    adjusted = compute_kin_bonus(fitness, parents, grandparents, 1.0)
    # Agent 0 should get bonus from agent 1 (sibling, 0.5 * 1.0)
    assert float(adjusted[0]) > float(fitness[0])
    # Agent 3 should have no bonus
    assert float(adjusted[3]) == float(fitness[3])
