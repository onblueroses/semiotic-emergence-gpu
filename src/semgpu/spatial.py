"""Toroidal spatial primitives for GPU.

Vectorized wrap_delta, wrap_dist_sq, and cell grid operations.
All functions operate on JAX arrays for jit compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def wrap_delta(a: jnp.ndarray, b: jnp.ndarray, size: int) -> jnp.ndarray:
    """Wrap-aware signed delta on toroidal grid. Returns values in (-size/2, size/2].

    Works for both integer and float coordinates. Vectorized over arrays.
    """
    d = b - a
    half = size / 2.0
    return jnp.where(d > half, d - size, jnp.where(d < -half, d + size, d))


def wrap_dist_sq(
    ax: jnp.ndarray, ay: jnp.ndarray,
    bx: jnp.ndarray, by: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """Squared toroidal distance between two sets of points."""
    dx = wrap_delta(ax, bx, grid_size)
    dy = wrap_delta(ay, by, grid_size)
    return dx * dx + dy * dy


def nearest_zone_edge_dist(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    zone_x: jnp.ndarray,
    zone_y: jnp.ndarray,
    zone_radius: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """Vectorized nearest zone edge distance for all prey.

    Args:
        prey_x: (N,) int prey x positions
        prey_y: (N,) int prey y positions
        zone_x: (Z,) float zone x positions
        zone_y: (Z,) float zone y positions
        zone_radius: (Z,) float zone radii
        grid_size: grid dimension

    Returns:
        (N,) float, negative = inside zone, positive = outside
    """
    # (N, Z) pairwise distances
    dx = wrap_delta(prey_x[:, None].astype(jnp.float32), zone_x[None, :], grid_size)
    dy = wrap_delta(prey_y[:, None].astype(jnp.float32), zone_y[None, :], grid_size)
    center_dist = jnp.sqrt(dx * dx + dy * dy)
    edge_dist = center_dist - zone_radius[None, :]
    return jnp.min(edge_dist, axis=1)


def zone_drain_amount(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    zone_x: jnp.ndarray,
    zone_y: jnp.ndarray,
    zone_radius: jnp.ndarray,
    grid_size: int,
    drain_rate: float,
) -> jnp.ndarray:
    """Compute zone damage for all prey. Gradient: full drain at center, zero at edge.

    Stacks across overlapping zones.

    Returns:
        (N,) float damage to add this tick
    """
    dx = wrap_delta(prey_x[:, None].astype(jnp.float32), zone_x[None, :], grid_size)
    dy = wrap_delta(prey_y[:, None].astype(jnp.float32), zone_y[None, :], grid_size)
    dist = jnp.sqrt(dx * dx + dy * dy)
    # gradient: drain_rate * (1 - dist/radius), clamped to 0 outside zone
    gradient = jnp.maximum(0.0, 1.0 - dist / zone_radius[None, :])
    per_zone_drain = drain_rate * gradient
    return jnp.sum(per_zone_drain, axis=1)


def move_zones(
    zone_x: jnp.ndarray,
    zone_y: jnp.ndarray,
    zone_speed: jnp.ndarray,
    grid_size: int,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Probabilistic random walk for zones. Each zone moves 1 cell with probability=speed.

    Returns:
        (zone_x, zone_y) updated positions with toroidal wrapping
    """
    Z = zone_x.shape[0]
    k1, k2 = jax.random.split(key)
    should_move = jax.random.uniform(k1, (Z,)) < zone_speed
    direction = jax.random.randint(k2, (Z,), 0, 4)

    # 0=up(-y), 1=down(+y), 2=right(+x), 3=left(-x)
    dy = jnp.where(direction == 0, -1.0, jnp.where(direction == 1, 1.0, 0.0))
    dx = jnp.where(direction == 2, 1.0, jnp.where(direction == 3, -1.0, 0.0))

    new_x = jnp.where(should_move, zone_x + dx, zone_x) % grid_size
    new_y = jnp.where(should_move, zone_y + dy, zone_y) % grid_size
    return new_x, new_y


def build_cell_grid(
    x: jnp.ndarray,
    y: jnp.ndarray,
    alive: jnp.ndarray,
    grid_size: int,
    max_per_cell: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a cell grid spatial index for alive entities.

    Args:
        x: (N,) int positions
        y: (N,) int positions
        alive: (N,) bool mask
        grid_size: grid dimension
        max_per_cell: max entities per cell (overflow silently dropped)

    Returns:
        cells: (grid_size*grid_size, max_per_cell) int32 indices, -1 = empty
        counts: (grid_size*grid_size,) int32 count per cell
    """
    N = x.shape[0]
    num_cells = grid_size * grid_size
    cell_idx = y * grid_size + x  # (N,)

    # Sort by cell index for grouped scatter
    # Use alive mask: dead entities get cell_idx = num_cells (out of range, ignored)
    effective_idx = jnp.where(alive, cell_idx, num_cells)

    # Count per cell
    counts = jnp.zeros(num_cells, dtype=jnp.int32)
    counts = counts.at[cell_idx].add(alive.astype(jnp.int32))

    # Build cell contents using scan-based approach
    # Sort entities by cell, then assign slots
    sort_order = jnp.argsort(effective_idx)
    sorted_cells = effective_idx[sort_order]
    sorted_agents = sort_order

    # Compute within-cell position for each sorted entity
    # Use a cumulative count approach
    cell_starts = jnp.zeros(num_cells + 1, dtype=jnp.int32)
    cell_starts = cell_starts.at[1:].set(jnp.cumsum(counts))

    cells = jnp.full((num_cells, max_per_cell), -1, dtype=jnp.int32)

    # Scatter sorted agents into grid
    # For each sorted alive agent, compute its slot in its cell
    is_valid = sorted_cells < num_cells  # alive entities
    # Compute position within cell
    same_as_prev = jnp.concatenate([jnp.array([False]), sorted_cells[1:] == sorted_cells[:-1]])
    within_cell_pos = jnp.zeros(N, dtype=jnp.int32)

    def scan_fn(carry, same):
        pos = jnp.where(same, carry + 1, 0)
        return pos, pos

    _, within_cell_pos = jax.lax.scan(scan_fn, jnp.int32(0), same_as_prev)

    # Only scatter if position < max_per_cell and entity is valid
    valid_scatter = is_valid & (within_cell_pos < max_per_cell)
    # Flat index into cells array
    flat_idx = sorted_cells * max_per_cell + within_cell_pos
    # Clip to valid range for safety
    flat_idx = jnp.clip(flat_idx, 0, num_cells * max_per_cell - 1)
    cells_flat = cells.reshape(-1)
    cells_flat = cells_flat.at[flat_idx].set(
        jnp.where(valid_scatter, sorted_agents.astype(jnp.int32), -1)
    )
    cells = cells_flat.reshape(num_cells, max_per_cell)

    # Clamp counts to max_per_cell
    counts = jnp.minimum(counts, max_per_cell)

    return cells, counts


def nearest_in_grid(
    query_x: jnp.ndarray,
    query_y: jnp.ndarray,
    target_x: jnp.ndarray,
    target_y: jnp.ndarray,
    cells: jnp.ndarray,
    counts: jnp.ndarray,
    grid_size: int,
    max_radius: int,
    skip_self: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Find nearest target for each query point using cell grid.

    For GPU, we do a brute-force distance computation over all targets
    instead of ring search (more parallelizable, avoids control flow).

    Args:
        query_x, query_y: (N,) query positions
        target_x, target_y: (M,) target positions
        cells, counts: cell grid (unused in brute-force, kept for API compat)
        grid_size: grid size
        max_radius: max search distance (unused in brute force)
        skip_self: (N,) int32, target index to skip (-1 = don't skip)

    Returns:
        nearest_idx: (N,) int32 index into targets, -1 if none found
        nearest_dx: (N,) float32 wrap delta x
        nearest_dy: (N,) float32 wrap delta y
    """
    # Brute force: (N, M) pairwise distances
    dx = wrap_delta(query_x[:, None], target_x[None, :], grid_size).astype(jnp.float32)
    dy = wrap_delta(query_y[:, None], target_y[None, :], grid_size).astype(jnp.float32)
    dist_sq = dx * dx + dy * dy

    # Mask out self
    M = target_x.shape[0]
    target_indices = jnp.arange(M)[None, :]  # (1, M)
    self_mask = target_indices == skip_self[:, None]  # (N, M)
    dist_sq = jnp.where(self_mask, jnp.float32(1e10), dist_sq)

    nearest_idx = jnp.argmin(dist_sq, axis=1)
    nearest_dx_val = jnp.take_along_axis(dx, nearest_idx[:, None], axis=1).squeeze(1)
    nearest_dy_val = jnp.take_along_axis(dy, nearest_idx[:, None], axis=1).squeeze(1)

    # Check if any valid target was found (all distances might be 1e10)
    min_dist = jnp.min(dist_sq, axis=1)
    found = min_dist < 1e9
    nearest_idx = jnp.where(found, nearest_idx, -1)

    return nearest_idx, nearest_dx_val, nearest_dy_val
