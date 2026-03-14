"""Tests for metrics computation."""

import numpy as np

from semgpu.metrics import (
    compute_iconicity,
    compute_mutual_info,
    compute_receiver_jsd,
    compute_signal_entropy,
    compute_silence_onset_metrics,
    cross_population_divergence,
    inter_symbol_jsd,
    jsd,
    mi_from_contingency,
    normalize_matrix,
    pearson,
    per_prey_receiver_jsd,
    rolling_fluctuation_ratio,
    trajectory_jsd,
)

NUM_SYMBOLS = 6


def test_mi_uniform_is_zero():
    counts = np.ones((6, 4), dtype=np.int32) * 100
    assert mi_from_contingency(counts) < 1e-10


def test_mi_correlated_is_positive():
    counts = np.zeros((6, 4), dtype=np.int32)
    for i in range(4):
        counts[i, i] = 100
    mi = mi_from_contingency(counts)
    assert mi > 0.5


def test_mi_below_threshold_returns_zero():
    counts = np.ones((6, 4), dtype=np.int32)  # 24 total < 20? No, 24 > 20
    # Need < 20 total
    counts = np.zeros((6, 4), dtype=np.int32)
    counts[0, 0] = 10
    assert compute_mutual_info(counts) == 0.0


def test_iconicity_no_bias():
    # Same proportion in and out of zone -> iconicity = 0
    assert abs(compute_iconicity(50, 100, 250, 500)) < 1e-10


def test_iconicity_positive():
    # More signals in zone than baseline
    val = compute_iconicity(80, 100, 250, 500)
    assert val > 0  # 0.8 - 0.5 = 0.3


def test_iconicity_empty():
    assert compute_iconicity(0, 0, 0, 0) == 0.0


def test_signal_entropy_uniform_is_max():
    counts = np.array([100, 100, 100, 100, 100, 100])
    expected = np.log(6)
    assert abs(compute_signal_entropy(counts) - expected) < 1e-4


def test_signal_entropy_single_symbol_is_zero():
    counts = np.array([100, 0, 0, 0, 0, 0])
    assert compute_signal_entropy(counts) < 1e-10


def test_signal_entropy_two_symbols():
    counts = np.array([100, 100, 0, 0, 0, 0])
    expected = np.log(2)
    assert abs(compute_signal_entropy(counts) - expected) < 1e-4


def test_jsd_identical_is_zero():
    p = np.array([0.2, 0.3, 0.1, 0.15, 0.25])
    assert jsd(p, p) < 1e-10


def test_jsd_different_is_positive():
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert jsd(p, q) > 0.5


def test_receiver_jsd_empty():
    counts = np.zeros((7, 2, 5), dtype=np.int32)
    a, b = compute_receiver_jsd(counts)
    assert abs(a) < 1e-10
    assert abs(b) < 1e-10


def test_receiver_jsd_with_signal_difference():
    counts = np.zeros((7, 2, 5), dtype=np.int32)
    # No signal: all eat (action 4), context 0
    counts[0, 0, 4] = 100
    # Symbol 0: all move up (action 0), context 0
    counts[1, 0, 0] = 100
    jsd_no_pred, _ = compute_receiver_jsd(counts)
    assert jsd_no_pred > 0.5


def test_pearson_perfect_positive():
    xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ys = np.array([2, 4, 6, 8, 10], dtype=np.float64)
    assert abs(pearson(xs, ys) - 1.0) < 1e-6


def test_pearson_uncorrelated():
    xs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    ys = np.array([1, -1, 0, -1, 1], dtype=np.float64)
    assert abs(pearson(xs, ys)) < 1e-6


def test_normalize_matrix_uniform():
    counts = np.ones((6, 4), dtype=np.int32) * 25
    norm = normalize_matrix(counts)
    assert norm is not None
    assert np.allclose(norm, 0.25)


def test_normalize_matrix_zero_row():
    counts = np.ones((6, 4), dtype=np.int32)
    counts[3] = 0
    assert normalize_matrix(counts) is None


def test_inter_symbol_jsd_identical():
    norm = np.full((6, 4), 0.25)
    result = inter_symbol_jsd(norm)
    assert all(v < 1e-10 for v in result)
    assert len(result) == 15


def test_inter_symbol_jsd_distinct():
    norm = np.full((6, 4), 0.25)
    norm[0] = [0.9, 0.03, 0.03, 0.04]
    norm[1] = [0.03, 0.9, 0.04, 0.03]
    result = inter_symbol_jsd(norm)
    assert result[0] > 0.1  # pair (0,1)


def test_trajectory_jsd_identical():
    m = np.full((6, 4), 0.25)
    assert trajectory_jsd(m, m) < 1e-10


def test_rolling_fluct_insufficient():
    assert rolling_fluctuation_ratio([1.0, 2.0], 5) == 0.0


def test_rolling_fluct_stable_near_one():
    series = [np.sin(i * 0.1) for i in range(40)]
    ratio = rolling_fluctuation_ratio(series, 10)
    assert abs(ratio - 1.0) < 0.5


def test_cross_pop_divergence_identical():
    m = np.full((6, 4), 0.25)
    assert cross_population_divergence(m, m) < 1e-10


def test_cross_pop_divergence_permuted():
    a = np.full((6, 4), 0.25)
    a[0] = [0.8, 0.1, 0.05, 0.05]
    a[1] = [0.1, 0.7, 0.1, 0.1]
    a[2] = [0.05, 0.05, 0.1, 0.8]
    b = np.full((6, 4), 0.25)
    b[0] = [0.1, 0.7, 0.1, 0.1]
    b[1] = [0.05, 0.05, 0.1, 0.8]
    b[2] = [0.8, 0.1, 0.05, 0.05]
    assert cross_population_divergence(a, b) < 1e-10


def test_per_prey_jsd_different():
    w = np.array([[50, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # all action 0 with signal
    wo = np.array([[0, 0, 0, 0, 50], [0, 0, 0, 0, 0]])  # all action 4 without
    assert per_prey_receiver_jsd(w, wo) > 0.5


def test_per_prey_jsd_identical():
    d = np.array([[10, 10, 10, 10, 10], [0, 0, 0, 0, 0]])
    assert per_prey_receiver_jsd(d, d) < 1e-10


def test_per_prey_jsd_insufficient():
    w = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])  # 5 total < 10 min
    wo = np.array([[10, 10, 10, 10, 10], [0, 0, 0, 0, 0]])
    assert per_prey_receiver_jsd(w, wo) == 0.0


def test_silence_onset_detects_shift():
    N = 5
    onset = np.zeros((N, 2, 5), dtype=np.int32)
    present = np.zeros((N, 2, 5), dtype=np.int32)
    # At onset: all movement (action 0), context 0
    onset[:, 0, 0] = 20
    # During signal: all eat (action 4), context 0
    present[:, 0, 4] = 20
    jsd_val, move_delta = compute_silence_onset_metrics(onset, present)
    assert jsd_val > 0.5
    assert move_delta > 0.5


def test_silence_onset_no_shift():
    N = 5
    d = np.zeros((N, 2, 5), dtype=np.int32)
    d[:, 0, :] = 10
    jsd_val, move_delta = compute_silence_onset_metrics(d, d)
    assert jsd_val < 1e-10
    assert abs(move_delta) < 1e-10


def test_silence_onset_insufficient():
    N = 1
    onset = np.zeros((N, 2, 5), dtype=np.int32)
    onset[0, 0, 0] = 1  # too few
    present = np.zeros((N, 2, 5), dtype=np.int32)
    present[0, 0, 0] = 20
    jsd_val, move_delta = compute_silence_onset_metrics(onset, present)
    assert jsd_val == 0.0
