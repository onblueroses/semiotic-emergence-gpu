"""Signal ring buffer and reception for GPU.

Fixed-size pre-allocated arrays with write cursor. Signals persist for
signal_ticks ticks with 1-tick delay (emitted on T, receivable from T+1).
Linear decay over signal_range.
"""

from __future__ import annotations

import jax.numpy as jnp

from semgpu.brain import NUM_SYMBOLS
from semgpu.spatial import wrap_delta


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
