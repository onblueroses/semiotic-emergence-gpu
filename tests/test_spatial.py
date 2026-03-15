"""Tests for cell-grid spatial primitives and grid-based signal reception."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from semgpu.signal import receive_signals, receive_signals_grid
from semgpu.spatial import build_coarse_grid, gather_nearby_indices


def test_build_coarse_grid_basic():
    """Coarse grid places entities in correct cells."""
    grid_size = 20
    cell_size = 5
    cells_per_side = 4  # 20 / 5
    x = jnp.array([0, 4, 5, 19], dtype=jnp.int32)
    y = jnp.array([0, 0, 5, 19], dtype=jnp.int32)
    valid = jnp.ones(4, dtype=jnp.bool_)

    cells, counts, cps = build_coarse_grid(x, y, valid, grid_size, cell_size, 8)
    assert cps == cells_per_side

    # Entity 0 at (0,0) -> cell (0,0) = flat 0
    # Entity 1 at (4,0) -> cell (0,0) = flat 0 (4//5=0)
    # Entity 2 at (5,5) -> cell (1,1) = flat 5 (5//5=1, 1*4+1=5)
    # Entity 3 at (19,19) -> cell (3,3) = flat 15 (19//5=3, 3*4+3=15)
    assert int(counts[0]) == 2  # two entities in cell (0,0)
    assert int(counts[5]) == 1  # one entity in cell (1,1)
    assert int(counts[15]) == 1  # one entity in cell (3,3)

    # Check that entity indices are in the right cells
    cell_0_contents = set(int(v) for v in cells[0] if v >= 0)
    assert cell_0_contents == {0, 1}


def test_build_coarse_grid_respects_valid_mask():
    """Invalid entities are excluded from the grid."""
    x = jnp.array([0, 5, 10], dtype=jnp.int32)
    y = jnp.array([0, 5, 10], dtype=jnp.int32)
    valid = jnp.array([True, False, True])

    cells, counts, _ = build_coarse_grid(x, y, valid, 20, 5, 8)

    # Entity 1 is invalid, should not appear
    all_indices = set()
    for row in cells:
        for v in row:
            if int(v) >= 0:
                all_indices.add(int(v))
    assert 1 not in all_indices
    assert 0 in all_indices
    assert 2 in all_indices


def test_gather_nearby_indices_finds_neighbors():
    """Gather returns indices from nearby cells."""
    grid_size = 20
    cell_size = 5
    cells_per_side = 4

    x = jnp.array([2, 7, 18], dtype=jnp.int32)
    y = jnp.array([2, 7, 18], dtype=jnp.int32)
    valid = jnp.ones(3, dtype=jnp.bool_)

    cells, counts, cps = build_coarse_grid(x, y, valid, grid_size, cell_size, 8)

    # Query from (2, 2), scan_radius=1: should find entity 0 (same cell)
    # and entity 1 (cell (1,1), which is 1 cell away)
    max_cand = (2 * 1 + 1) ** 2 * 8  # 9 cells * 8 = 72
    result = gather_nearby_indices(2, 2, cells, cells_per_side, cell_size, 1, max_cand)

    found = set(int(v) for v in result if v >= 0)
    assert 0 in found  # same cell
    assert 1 in found  # adjacent cell


def test_gather_nearby_indices_wraps_toroidally():
    """Gather handles toroidal wrapping at grid edges."""
    grid_size = 20
    cell_size = 5
    cells_per_side = 4

    x = jnp.array([1, 19], dtype=jnp.int32)
    y = jnp.array([1, 19], dtype=jnp.int32)
    valid = jnp.ones(2, dtype=jnp.bool_)

    cells, counts, cps = build_coarse_grid(x, y, valid, grid_size, cell_size, 8)

    # Query from (1, 1) with scan_radius=1: should find entity at (19, 19)
    # because cell (3,3) wraps to be adjacent to cell (0,0)
    max_cand = (2 * 1 + 1) ** 2 * 8
    result = gather_nearby_indices(1, 1, cells, cells_per_side, cell_size, 1, max_cand)

    found = set(int(v) for v in result if v >= 0)
    assert 0 in found
    assert 1 in found  # wrapped neighbor


def test_receive_signals_grid_matches_brute_force():
    """Grid-based reception produces same results as brute-force for small N."""
    key = jax.random.key(99)
    N = 30
    S = 200
    grid_size = 20
    signal_range = 8.0

    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    prey_x = jax.random.randint(k1, (N,), 0, grid_size)
    prey_y = jax.random.randint(k2, (N,), 0, grid_size)

    sig_x = jax.random.randint(k3, (S,), 0, grid_size)
    sig_y = jax.random.randint(k4, (S,), 0, grid_size)
    sig_symbol = jax.random.randint(k5, (S,), 0, 6)
    sig_tick = jax.random.randint(k6, (S,), 0, 5)
    sig_valid = jnp.ones(S, dtype=jnp.bool_)
    current_tick = jnp.int32(5)

    brute = receive_signals(
        prey_x, prey_y, sig_x, sig_y, sig_symbol, sig_tick,
        sig_valid, current_tick, grid_size, signal_range,
    )
    grid = receive_signals_grid(
        prey_x, prey_y, sig_x, sig_y, sig_symbol, sig_tick,
        sig_valid, current_tick, grid_size, signal_range,
    )

    # Strength values (every 3rd column starting at 0) must match exactly.
    # Direction columns may differ when two signals have identical strength
    # (argmax tie-breaking differs between brute-force and grid ordering).
    brute_str = brute[:, 0::3]
    grid_str = grid[:, 0::3]
    npt.assert_allclose(brute_str, grid_str, atol=1e-5)


def test_receive_signals_grid_no_signals():
    """Grid reception handles case with no valid signals."""
    N = 10
    S = 50
    grid_size = 20
    signal_range = 8.0

    key = jax.random.key(42)
    k1, k2 = jax.random.split(key)

    prey_x = jax.random.randint(k1, (N,), 0, grid_size)
    prey_y = jax.random.randint(k2, (N,), 0, grid_size)
    sig_x = jnp.zeros(S, dtype=jnp.int32)
    sig_y = jnp.zeros(S, dtype=jnp.int32)
    sig_symbol = jnp.zeros(S, dtype=jnp.int32)
    sig_tick = jnp.zeros(S, dtype=jnp.int32)
    sig_valid = jnp.zeros(S, dtype=jnp.bool_)  # no valid signals
    current_tick = jnp.int32(5)

    result = receive_signals_grid(
        prey_x, prey_y, sig_x, sig_y, sig_symbol, sig_tick,
        sig_valid, current_tick, grid_size, signal_range,
    )

    assert result.shape == (N, 18)
    npt.assert_allclose(result, 0.0, atol=1e-7)
