"""Tests for brain module - mirrors Rust brain.rs tests."""

import jax
import jax.numpy as jnp

from semgpu.brain import (
    DEFAULT_BASE_HIDDEN,
    DEFAULT_SIGNAL_HIDDEN,
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
    SEG_MEM_BIAS,
    SEG_MOVE_BIAS,
    SEG_SIGOUT_BIAS,
    SEG_SIGHID_BIAS,
    SEG_SIGHID_SIGOUT,
    SIGNAL_OUTPUTS,
    _forward_single,
    batched_forward,
    softmax_emit,
)


def test_genome_length():
    assert MAX_GENOME_LEN == 5683


def test_segment_offsets_contiguous():
    assert SEG_MEM_BIAS + MEMORY_OUTPUTS == MAX_GENOME_LEN


def test_segment_offsets_inputs39():
    # All offsets shift by +192 (3 extra inputs * 64 base hidden) vs INPUTS=36
    assert INPUTS == 39
    assert SEG_BASE_BIAS == 2496   # 39*64
    assert SEG_BASE_MOVE == 2560   # +64
    assert SEG_MOVE_BIAS == 2880   # +64*5
    assert SEG_BASE_SIGHID == 2885 # +5
    assert SEG_SIGHID_BIAS == 4933 # +64*32
    assert SEG_SIGHID_SIGOUT == 4965  # +32
    assert SEG_SIGOUT_BIAS == 5157 # +32*6
    assert SEG_BASE_MEM == 5163    # +6
    assert SEG_MEM_BIAS == 5675    # +64*8


def test_zero_weights_zero_output():
    w = jnp.zeros(MAX_GENOME_LEN)
    inp = jnp.zeros(INPUTS)
    bh = jnp.array(DEFAULT_BASE_HIDDEN)
    sh = jnp.array(DEFAULT_SIGNAL_HIDDEN)
    actions, signals, memory = _forward_single(w, bh, sh, inp)
    assert jnp.allclose(actions, 0.0, atol=1e-6)
    assert jnp.allclose(signals, 0.0, atol=1e-6)
    assert jnp.allclose(memory, 0.0, atol=1e-6)


def test_forward_deterministic():
    key = jax.random.key(42)
    w = jax.random.normal(key, (MAX_GENOME_LEN,))
    inp = jnp.ones(INPUTS) * 0.5
    bh = jnp.array(DEFAULT_BASE_HIDDEN)
    sh = jnp.array(DEFAULT_SIGNAL_HIDDEN)
    a1, s1, m1 = _forward_single(w, bh, sh, inp)
    a2, s2, m2 = _forward_single(w, bh, sh, inp)
    assert jnp.allclose(a1, a2, atol=1e-10)
    assert jnp.allclose(s1, s2, atol=1e-10)
    assert jnp.allclose(m1, m2, atol=1e-10)


def test_forward_respects_base_hidden_size():
    w = jnp.ones(MAX_GENOME_LEN) * 0.1
    inp = jnp.ones(INPUTS)
    sh = jnp.array(DEFAULT_SIGNAL_HIDDEN)

    a_full, _, _ = _forward_single(w, jnp.array(MAX_BASE_HIDDEN), sh, inp)
    a_small, _, _ = _forward_single(w, jnp.array(MIN_BASE_HIDDEN), sh, inp)
    assert not jnp.allclose(a_full, a_small, atol=1e-6)


def test_forward_respects_signal_hidden_size():
    w = jnp.ones(MAX_GENOME_LEN) * 0.1
    inp = jnp.ones(INPUTS)
    bh = jnp.array(DEFAULT_BASE_HIDDEN)

    _, s_full, _ = _forward_single(w, bh, jnp.array(MAX_SIGNAL_HIDDEN), inp)
    _, s_small, _ = _forward_single(w, bh, jnp.array(MIN_SIGNAL_HIDDEN), inp)
    assert not jnp.allclose(s_full, s_small, atol=1e-6)


def test_memory_bounded():
    key = jax.random.key(99)
    w = jax.random.normal(key, (MAX_GENOME_LEN,)) * 5.0
    inp = jax.random.normal(jax.random.key(100), (INPUTS,))
    bh = jnp.array(DEFAULT_BASE_HIDDEN)
    sh = jnp.array(DEFAULT_SIGNAL_HIDDEN)
    _, _, memory = _forward_single(w, bh, sh, inp)
    assert jnp.all(memory >= -1.0)
    assert jnp.all(memory <= 1.0)


def test_batched_forward():
    N = 16
    key = jax.random.key(7)
    k1, k2 = jax.random.split(key)
    weights = jax.random.normal(k1, (N, MAX_GENOME_LEN))
    inputs = jax.random.normal(k2, (N, INPUTS))
    bh = jnp.full(N, DEFAULT_BASE_HIDDEN)
    sh = jnp.full(N, DEFAULT_SIGNAL_HIDDEN)

    actions, signals, memory = batched_forward(weights, bh, sh, inputs)
    assert actions.shape == (N, MOVEMENT_OUTPUTS)
    assert signals.shape == (N, SIGNAL_OUTPUTS)
    assert memory.shape == (N, MEMORY_OUTPUTS)

    # Verify matches single forward for first agent
    a0, s0, m0 = _forward_single(weights[0], bh[0], sh[0], inputs[0])
    assert jnp.allclose(actions[0], a0, atol=1e-5)
    assert jnp.allclose(signals[0], s0, atol=1e-5)
    assert jnp.allclose(memory[0], m0, atol=1e-5)


def test_softmax_uniform():
    logits = jnp.zeros((4, SIGNAL_OUTPUTS))
    symbols, emit = softmax_emit(logits)
    # Uniform softmax: max = 1/6, threshold is >, so no emission
    assert jnp.all(~emit)


def test_softmax_concentrated():
    logits = jnp.zeros((4, SIGNAL_OUTPUTS))
    logits = logits.at[:, 2].set(10.0)
    symbols, emit = softmax_emit(logits)
    assert jnp.all(emit)
    assert jnp.all(symbols == 2)


def test_softmax_emit_threshold():
    # Just above uniform: one symbol slightly higher
    logits = jnp.zeros((1, SIGNAL_OUTPUTS)).at[0, 0].set(0.1)
    _, emit = softmax_emit(logits)
    # With logit 0.1, softmax gives slightly above 1/6 -> should emit
    assert emit[0]


def test_zero_weights_min_hidden():
    w = jnp.zeros(MAX_GENOME_LEN)
    inp = jnp.ones(INPUTS)
    bh = jnp.array(MIN_BASE_HIDDEN)
    sh = jnp.array(MIN_SIGNAL_HIDDEN)
    actions, signals, memory = _forward_single(w, bh, sh, inp)
    assert jnp.allclose(actions, 0.0, atol=1e-6)
    assert jnp.allclose(signals, 0.0, atol=1e-6)
    assert jnp.allclose(memory, 0.0, atol=1e-6)


def test_config_defaults():
    from semgpu.config import SimParams
    p = SimParams()
    assert p.pop_size == 384
    assert p.grid_size == 56
    assert p.num_zones == 3
    assert p.food_count == 100
    assert p.ticks_per_eval == 500
    assert p.zone_radius == 8.0
    assert p.zone_speed == 0.5
    assert p.zone_drain_rate == 0.02
    assert p.signal_cost == 0.002
    assert p.base_drain == 0.0008
    assert p.neuron_cost == 0.0
    assert p.patch_ratio == 0.5
    assert p.kin_bonus == 0.1
    assert p.signal_ticks == 4


def test_config_from_cli():
    from semgpu.config import SimParams
    p = SimParams.from_cli(["42", "100", "--pop", "1000", "--grid", "100"])
    assert p.pop_size == 1000
    assert p.grid_size == 100
    # scale = 100/20 = 5.0, signal_range = 8.0 * 5.0 = 40.0
    assert abs(p.signal_range - 40.0) < 1e-6
    assert p.elite_count == max(1000 // 6, 2)
