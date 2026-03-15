"""Signal ring buffer and reception for GPU.

Fixed-size pre-allocated arrays with write cursor. Signals persist for
signal_ticks ticks with 1-tick delay (emitted on T, receivable from T+1).
Linear decay over signal_range.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from semgpu.brain import NUM_SYMBOLS
from semgpu.spatial import build_coarse_grid, gather_nearby_indices, wrap_delta


def receive_signals(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    sig_x: jnp.ndarray,
    sig_y: jnp.ndarray,
    sig_symbol: jnp.ndarray,
    sig_tick: jnp.ndarray,
    sig_valid: jnp.ndarray,
    current_tick: jnp.ndarray,
    grid_size: int,
    signal_range: float,
) -> jnp.ndarray:
    """Compute signal inputs for all prey. Strongest-per-symbol reception.

    O(N*S) brute-force version. Use receive_signals_grid for large populations.

    Args:
        prey_x: (N,) int prey x positions
        prey_y: (N,) int prey y positions
        sig_x: (S,) int signal x positions
        sig_y: (S,) int signal y positions
        sig_symbol: (S,) int signal symbol (0-5)
        sig_tick: (S,) int tick when emitted
        sig_valid: (S,) bool, whether signal slot is occupied
        current_tick: scalar int
        grid_size: grid dimension
        signal_range: max reception distance

    Returns:
        signal_inputs: (N, NUM_SYMBOLS*3) float32
            For each symbol: [strength, dx, dy] where dx/dy point to strongest emitter
    """
    N = prey_x.shape[0]
    S = sig_x.shape[0]

    # 1-tick delay: only signals emitted before current tick
    receivable = sig_valid & (sig_tick < current_tick)

    # (N, S) pairwise distances
    dx = wrap_delta(
        prey_x[:, None].astype(jnp.float32),
        sig_x[None, :].astype(jnp.float32),
        grid_size
    )
    dy = wrap_delta(
        prey_y[:, None].astype(jnp.float32),
        sig_y[None, :].astype(jnp.float32),
        grid_size
    )
    dist = jnp.sqrt(dx * dx + dy * dy)

    # Linear decay: strength = 1 - dist/signal_range, clamped to 0
    strength = jnp.maximum(0.0, 1.0 - dist / signal_range)

    # Mask out invalid/unreceivable signals
    strength = jnp.where(receivable[None, :], strength, 0.0)

    # Normalize direction
    safe_dist = jnp.maximum(dist, 1e-6)
    norm_dx = dx / (safe_dist * grid_size)
    norm_dy = dy / (safe_dist * grid_size)

    # For each prey and each symbol, find strongest signal
    # sig_symbol is (S,), expand to (1, S) for comparison
    signal_inputs = jnp.zeros((N, NUM_SYMBOLS * 3))

    for sym in range(NUM_SYMBOLS):
        sym_mask = (sig_symbol[None, :] == sym)  # (1, S) broadcast to (N, S)
        sym_strength = jnp.where(sym_mask, strength, 0.0)  # (N, S)
        best_idx = jnp.argmax(sym_strength, axis=1)  # (N,)
        best_str = jnp.take_along_axis(sym_strength, best_idx[:, None], axis=1).squeeze(1)
        best_dx = jnp.take_along_axis(norm_dx, best_idx[:, None], axis=1).squeeze(1)
        best_dy = jnp.take_along_axis(norm_dy, best_idx[:, None], axis=1).squeeze(1)

        base = sym * 3
        signal_inputs = signal_inputs.at[:, base].set(best_str)
        signal_inputs = signal_inputs.at[:, base + 1].set(jnp.where(best_str > 0, best_dx, 0.0))
        signal_inputs = signal_inputs.at[:, base + 2].set(jnp.where(best_str > 0, best_dy, 0.0))

    return signal_inputs


# -- Grid-based signal reception (O(N * K) where K = max_candidates << S) --

# Tuning constants for the coarse signal grid
SIGNAL_CELL_DIVISOR = 4     # cell_size = signal_range // this
SIGNAL_MAX_PER_CELL = 128   # max signals stored per coarse cell
SIGNAL_SCAN_CELLS = 5       # scan +/- this many cells (must cover signal_range)


def _signal_cell_size(signal_range: float) -> int:
    return max(1, int(signal_range) // SIGNAL_CELL_DIVISOR)


def _signal_scan_radius(signal_range: float) -> int:
    """Scan radius in coarse cells that guarantees full signal_range coverage."""
    cs = _signal_cell_size(signal_range)
    # ceil(signal_range / cell_size) to cover the full range
    return int(signal_range / cs) + 1


def _signal_max_candidates(signal_range: float) -> int:
    sr = _signal_scan_radius(signal_range)
    side = 2 * sr + 1
    return side * side * SIGNAL_MAX_PER_CELL


def receive_signals_grid(
    prey_x: jnp.ndarray,
    prey_y: jnp.ndarray,
    sig_x: jnp.ndarray,
    sig_y: jnp.ndarray,
    sig_symbol: jnp.ndarray,
    sig_tick: jnp.ndarray,
    sig_valid: jnp.ndarray,
    current_tick: jnp.ndarray,
    grid_size: int,
    signal_range: float,
) -> jnp.ndarray:
    """Compute signal inputs using spatial binning. Same API as receive_signals.

    Builds a coarse cell grid over signals, then each prey scans only nearby
    cells. VRAM scales as O(N * max_candidates) instead of O(N * S).
    """
    cell_size = _signal_cell_size(signal_range)
    scan_radius = _signal_scan_radius(signal_range)
    max_candidates = _signal_max_candidates(signal_range)
    cells_per_side = (grid_size + cell_size - 1) // cell_size

    # 1-tick delay
    receivable = sig_valid & (sig_tick < current_tick)

    # Build coarse spatial index over receivable signals
    sig_cells, sig_counts, cps = build_coarse_grid(
        sig_x, sig_y, receivable, grid_size, cell_size, SIGNAL_MAX_PER_CELL,
    )

    def process_one_prey(prey_pos):
        """Process a single prey: gather nearby signals, find strongest per symbol."""
        px, py = prey_pos

        # Gather candidate signal indices from nearby cells
        cand_idx = gather_nearby_indices(
            px, py, sig_cells, cells_per_side, cell_size,
            scan_radius, max_candidates,
        )

        # Filter to valid candidates
        valid = cand_idx >= 0
        safe_idx = jnp.clip(cand_idx, 0)  # clamp for indexing, mask later

        # Compute distances to candidates
        cand_sx = sig_x[safe_idx].astype(jnp.float32)
        cand_sy = sig_y[safe_idx].astype(jnp.float32)
        cand_dx = wrap_delta(
            jnp.full(max_candidates, px, dtype=jnp.float32), cand_sx, grid_size
        )
        cand_dy = wrap_delta(
            jnp.full(max_candidates, py, dtype=jnp.float32), cand_sy, grid_size
        )
        cand_dist = jnp.sqrt(cand_dx * cand_dx + cand_dy * cand_dy)

        # Linear decay
        cand_strength = jnp.maximum(0.0, 1.0 - cand_dist / signal_range)
        cand_strength = jnp.where(valid, cand_strength, 0.0)

        # Normalized direction
        safe_dist = jnp.maximum(cand_dist, 1e-6)
        cand_norm_dx = cand_dx / (safe_dist * grid_size)
        cand_norm_dy = cand_dy / (safe_dist * grid_size)

        cand_sym = sig_symbol[safe_idx]

        # For each symbol, find strongest
        result = jnp.zeros(NUM_SYMBOLS * 3)
        for sym in range(NUM_SYMBOLS):
            sym_mask = valid & (cand_sym == sym)
            sym_str = jnp.where(sym_mask, cand_strength, 0.0)
            best = jnp.argmax(sym_str)
            best_s = sym_str[best]
            best_ndx = cand_norm_dx[best]
            best_ndy = cand_norm_dy[best]

            base = sym * 3
            result = result.at[base].set(best_s)
            result = result.at[base + 1].set(jnp.where(best_s > 0, best_ndx, 0.0))
            result = result.at[base + 2].set(jnp.where(best_s > 0, best_ndy, 0.0))

        return result

    # vmap over all prey
    prey_positions = jnp.stack([prey_x, prey_y], axis=1)  # (N, 2)
    return jax.vmap(process_one_prey)(prey_positions)
