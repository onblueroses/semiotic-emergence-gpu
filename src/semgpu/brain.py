"""Batched neural network forward pass for GPU.

Mirrors Rust brain.rs: split-head topology with evolvable hidden layers.
Weights stored as flat (N, MAX_GENOME_LEN) array. Per-agent neuron masks
handle variable hidden sizes without dynamic shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Architecture constants
INPUTS = 39  # 36 base + freeze_pressure replaces dead_spare + death_nearby/dx/dy

MAX_BASE_HIDDEN = 64
MIN_BASE_HIDDEN = 4
DEFAULT_BASE_HIDDEN = 12

MAX_SIGNAL_HIDDEN = 32
MIN_SIGNAL_HIDDEN = 2
DEFAULT_SIGNAL_HIDDEN = 6

MEMORY_SIZE = 8

MOVEMENT_OUTPUTS = 5
SIGNAL_OUTPUTS = 6
MEMORY_OUTPUTS = MEMORY_SIZE
NUM_SYMBOLS = SIGNAL_OUTPUTS

# Segment offsets into flat genome (column-major layout)
# Each offset computed from previous; all shift by +192 due to INPUTS 36->39
SEG_INPUT_BASE = 0
SEG_BASE_BIAS = SEG_INPUT_BASE + INPUTS * MAX_BASE_HIDDEN  # 2496
SEG_BASE_MOVE = SEG_BASE_BIAS + MAX_BASE_HIDDEN  # 2560
SEG_MOVE_BIAS = SEG_BASE_MOVE + MAX_BASE_HIDDEN * MOVEMENT_OUTPUTS  # 2880
SEG_BASE_SIGHID = SEG_MOVE_BIAS + MOVEMENT_OUTPUTS  # 2885
SEG_SIGHID_BIAS = SEG_BASE_SIGHID + MAX_BASE_HIDDEN * MAX_SIGNAL_HIDDEN  # 4933
SEG_SIGHID_SIGOUT = SEG_SIGHID_BIAS + MAX_SIGNAL_HIDDEN  # 4965
SEG_SIGOUT_BIAS = SEG_SIGHID_SIGOUT + MAX_SIGNAL_HIDDEN * SIGNAL_OUTPUTS  # 5157
SEG_BASE_MEM = SEG_SIGOUT_BIAS + SIGNAL_OUTPUTS  # 5163
SEG_MEM_BIAS = SEG_BASE_MEM + MAX_BASE_HIDDEN * MEMORY_OUTPUTS  # 5675

MAX_GENOME_LEN = SEG_MEM_BIAS + MEMORY_OUTPUTS  # 5683

assert MAX_GENOME_LEN == 5683


def _make_mask(size: jnp.ndarray, max_size: int) -> jnp.ndarray:
    """Create a boolean mask of shape (max_size,) with True for indices < size."""
    return jnp.arange(max_size) < size


def _forward_single(
    weights: jnp.ndarray,
    base_hidden_size: jnp.ndarray,
    signal_hidden_size: jnp.ndarray,
    inputs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass for a single agent. Designed to be vmapped.

    Args:
        weights: (MAX_GENOME_LEN,) flat weight array
        base_hidden_size: scalar int
        signal_hidden_size: scalar int
        inputs: (INPUTS,) input vector

    Returns:
        actions: (MOVEMENT_OUTPUTS,) raw movement logits
        signals: (SIGNAL_OUTPUTS,) raw signal logits
        memory_write: (MEMORY_OUTPUTS,) tanh-bounded memory outputs
    """
    w = weights
    bh_mask = _make_mask(base_hidden_size, MAX_BASE_HIDDEN)
    sh_mask = _make_mask(signal_hidden_size, MAX_SIGNAL_HIDDEN)

    # 1. Input -> Base hidden (tanh)
    # Weight matrix is stored column-major: w[SEG_INPUT_BASE + i*MAX_BASE_HIDDEN + h]
    w_input_base = w[SEG_INPUT_BASE:SEG_BASE_BIAS].reshape(INPUTS, MAX_BASE_HIDDEN)
    base_bias = w[SEG_BASE_BIAS:SEG_BASE_MOVE]
    base_hidden = base_bias + inputs @ w_input_base  # (MAX_BASE_HIDDEN,)
    base_hidden = jnp.tanh(base_hidden) * bh_mask

    # 2. Base hidden -> Movement (raw)
    w_base_move = w[SEG_BASE_MOVE:SEG_MOVE_BIAS].reshape(MAX_BASE_HIDDEN, MOVEMENT_OUTPUTS)
    move_bias = w[SEG_MOVE_BIAS:SEG_BASE_SIGHID]
    actions = move_bias + base_hidden @ w_base_move

    # 3. Base hidden -> Signal hidden (tanh)
    w_base_sighid = w[SEG_BASE_SIGHID:SEG_SIGHID_BIAS].reshape(MAX_BASE_HIDDEN, MAX_SIGNAL_HIDDEN)
    sighid_bias = w[SEG_SIGHID_BIAS:SEG_SIGHID_SIGOUT]
    sig_hidden = sighid_bias + base_hidden @ w_base_sighid
    sig_hidden = jnp.tanh(sig_hidden) * sh_mask

    # 4. Signal hidden -> Signal outputs (raw)
    w_sighid_sigout = w[SEG_SIGHID_SIGOUT:SEG_SIGOUT_BIAS].reshape(MAX_SIGNAL_HIDDEN, SIGNAL_OUTPUTS)
    sigout_bias = w[SEG_SIGOUT_BIAS:SEG_BASE_MEM]
    signals = sigout_bias + sig_hidden @ w_sighid_sigout

    # 5. Base hidden -> Memory (tanh)
    w_base_mem = w[SEG_BASE_MEM:SEG_MEM_BIAS].reshape(MAX_BASE_HIDDEN, MEMORY_OUTPUTS)
    mem_bias = w[SEG_MEM_BIAS:SEG_MEM_BIAS + MEMORY_OUTPUTS]
    memory_write = jnp.tanh(mem_bias + base_hidden @ w_base_mem)

    return actions, signals, memory_write


# Batched forward: vmap over (weights, base_hidden_size, signal_hidden_size, inputs)
batched_forward = jax.vmap(_forward_single, in_axes=(0, 0, 0, 0))


def softmax_emit(
    signal_logits: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batched softmax + emission decision.

    Args:
        signal_logits: (N, SIGNAL_OUTPUTS) raw signal outputs

    Returns:
        symbol_indices: (N,) int, which symbol each agent would emit
        emit_mask: (N,) bool, True if max(softmax) > 1/NUM_SYMBOLS
    """
    probs = jax.nn.softmax(signal_logits, axis=-1)
    symbol_indices = jnp.argmax(probs, axis=-1)
    max_probs = jnp.max(probs, axis=-1)
    threshold = 1.0 / NUM_SYMBOLS
    emit_mask = max_probs > threshold
    return symbol_indices, emit_mask
