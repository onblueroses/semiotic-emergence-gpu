"""Metrics computation - host-side numpy after device transfer.

All functions take numpy arrays (transferred from JAX device arrays).
Matches the Rust metrics.rs implementations.
"""

from __future__ import annotations

import numpy as np

NUM_SYMBOLS = 6
NUM_ACTIONS = 5
INPUTS = 36
MIN_RECEIVER_SAMPLES = 10


def mi_from_contingency(counts: np.ndarray) -> float:
    """Mutual information from a contingency table.

    Args:
        counts: (S, B) int array where S=symbols, B=bins
    """
    n = float(counts.sum())
    if n == 0:
        return 0.0
    mi = 0.0
    n_sym, n_bin = counts.shape
    for s in range(n_sym):
        p_s = counts[s].sum() / n
        if p_s == 0:
            continue
        for b in range(n_bin):
            p_b = counts[:, b].sum() / n
            if p_b == 0:
                continue
            p_joint = counts[s, b] / n
            if p_joint > 0:
                mi += p_joint * np.log(p_joint / (p_s * p_b))
    return float(mi)


def compute_mutual_info(mi_counts: np.ndarray) -> float:
    """MI from pre-accumulated (6, 4) contingency table."""
    if mi_counts.sum() < 20:
        return 0.0
    return mi_from_contingency(mi_counts)


def compute_iconicity(
    signals_in_zone: int,
    total_signals: int,
    ticks_in_zone: int,
    total_prey_ticks: int,
) -> float:
    """Signal-in-zone rate minus baseline zone rate."""
    if total_signals == 0 or total_prey_ticks == 0:
        return 0.0
    signal_zone_rate = signals_in_zone / total_signals
    baseline_zone_rate = ticks_in_zone / total_prey_ticks
    return float(signal_zone_rate - baseline_zone_rate)


def compute_signal_entropy(symbol_counts: np.ndarray) -> float:
    """Shannon entropy of symbol frequencies. Max = ln(6)."""
    total = symbol_counts.sum()
    if total == 0:
        return 0.0
    p = symbol_counts[symbol_counts > 0].astype(np.float64) / total
    return float(-np.sum(p * np.log(p)))


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = (p + q) * 0.5
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def normalize_action_dist(counts: np.ndarray):
    """Normalize action counts to probabilities. Returns None if all zero."""
    total = counts.sum()
    if total == 0:
        return None
    return counts.astype(np.float64) / total


def compute_receiver_jsd(recv_counts: np.ndarray) -> tuple[float, float]:
    """Receiver response spectrum JSD.

    Args:
        recv_counts: (7, 2, 5) int [no_sig + 6 symbols][context][actions]

    Returns:
        (jsd_no_pred, jsd_pred)
    """
    jsd_per_ctx = [0.0, 0.0]
    for ctx in range(2):
        p_base = normalize_action_dist(recv_counts[0, ctx])
        if p_base is None:
            continue
        total_jsd = 0.0
        n_sym = 0
        for sym in range(NUM_SYMBOLS):
            p_sig = normalize_action_dist(recv_counts[sym + 1, ctx])
            if p_sig is None:
                continue
            total_jsd += jsd(p_base, p_sig)
            n_sym += 1
        if n_sym > 0:
            jsd_per_ctx[ctx] = total_jsd / n_sym
    return jsd_per_ctx[0], jsd_per_ctx[1]


def compute_per_symbol_jsd(recv_counts: np.ndarray) -> np.ndarray:
    """Per-symbol JSD pooled across contexts. Returns (6,) float."""
    base_pooled = recv_counts[0, 0] + recv_counts[0, 1]
    result = np.zeros(NUM_SYMBOLS)
    for sym in range(NUM_SYMBOLS):
        sig_pooled = recv_counts[sym + 1, 0] + recv_counts[sym + 1, 1]
        p_base = normalize_action_dist(base_pooled)
        p_sig = normalize_action_dist(sig_pooled)
        if p_base is None or p_sig is None:
            continue
        result[sym] = jsd(p_base, p_sig)
    return result


def pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    dx = xs - xs.mean()
    dy = ys - ys.mean()
    denom = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def normalize_matrix(counts: np.ndarray):
    """Normalize (6, 4) counts to row-wise probabilities. None if any row is zero."""
    result = np.zeros((NUM_SYMBOLS, 4), dtype=np.float64)
    for s in range(NUM_SYMBOLS):
        total = counts[s].sum()
        if total == 0:
            return None
        result[s] = counts[s].astype(np.float64) / total
    return result


def inter_symbol_jsd(norm: np.ndarray) -> list[float]:
    """Pairwise JSD between symbols' context distributions. Returns 15 values."""
    result = []
    for i in range(NUM_SYMBOLS):
        for j in range(i + 1, NUM_SYMBOLS):
            result.append(jsd(norm[i], norm[j]))
    return result


def trajectory_jsd(prev: np.ndarray, curr: np.ndarray) -> float:
    """Average row-wise JSD between two generation matrices."""
    total = sum(jsd(prev[s], curr[s]) for s in range(NUM_SYMBOLS))
    return total / NUM_SYMBOLS


def rolling_fluctuation_ratio(series: list[float] | np.ndarray, window: int) -> float:
    """Std(recent) / std(early). Rising ratio precedes phase transitions."""
    s = np.asarray(series, dtype=np.float64)
    if len(s) < window * 2:
        return 0.0
    early_std = np.std(s[:window])
    recent_std = np.std(s[-window:])
    if early_std < 1e-12:
        return 0.0
    return float(recent_std / early_std)


def per_prey_receiver_jsd(
    with_counts: np.ndarray,
    without_counts: np.ndarray,
    min_samples: int = MIN_RECEIVER_SAMPLES,
) -> float:
    """Per-prey JSD between actions with vs without signal.

    Args:
        with_counts: (2, 5) int [context][actions]
        without_counts: (2, 5) int [context][actions]
    """
    w_pooled = with_counts[0] + with_counts[1]
    wo_pooled = without_counts[0] + without_counts[1]
    if w_pooled.sum() < min_samples or wo_pooled.sum() < min_samples:
        return 0.0
    p_w = normalize_action_dist(w_pooled)
    p_wo = normalize_action_dist(wo_pooled)
    if p_w is None or p_wo is None:
        return 0.0
    return jsd(p_w, p_wo)


def cross_population_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Min-over-permutations average row-wise JSD. Heap's algorithm."""
    perm = list(range(NUM_SYMBOLS))
    min_div = float("inf")

    def _eval_perm():
        total = sum(jsd(a[i], b[perm[i]]) for i in range(NUM_SYMBOLS))
        return total / NUM_SYMBOLS

    def _heap(k):
        nonlocal min_div
        if k == 1:
            val = _eval_perm()
            if val < min_div:
                min_div = val
            return
        for i in range(k):
            _heap(k - 1)
            j = i if k % 2 == 0 else 0
            perm[j], perm[k - 1] = perm[k - 1], perm[j]

    _heap(NUM_SYMBOLS)
    return min_div


def compute_silence_onset_metrics(
    onset_actions: np.ndarray,
    present_actions: np.ndarray,
) -> tuple[float, float]:
    """Silence onset: behavior when signals disappear vs during signals.

    Args:
        onset_actions: (N, 2, 5) int per prey
        present_actions: (N, 2, 5) int per prey

    Returns:
        (onset_jsd, move_delta)
    """
    # Pool across prey, context 0 (not in zone) only
    onset_pooled = onset_actions[:, 0, :].sum(axis=0)
    present_pooled = present_actions[:, 0, :].sum(axis=0)

    if onset_pooled.sum() < 5 or present_pooled.sum() < 10:
        return 0.0, 0.0

    p_onset = normalize_action_dist(onset_pooled)
    p_present = normalize_action_dist(present_pooled)
    if p_onset is None or p_present is None:
        return 0.0, 0.0

    onset_jsd_val = jsd(p_onset, p_present)
    onset_flight = float(p_onset[0] + p_onset[1] + p_onset[2] + p_onset[3])
    present_flight = float(p_present[0] + p_present[1] + p_present[2] + p_present[3])
    return float(onset_jsd_val), onset_flight - present_flight


def compute_input_mi(symbols: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """MI between each symbol and each input dimension at emission time.

    Uses quartile-based binning.

    Args:
        symbols: (E,) int
        inputs: (E, 36) float

    Returns:
        (36,) float MI per input dimension
    """
    if len(symbols) < 20:
        return np.zeros(INPUTS)

    result = np.zeros(INPUTS)
    syms = np.clip(symbols.astype(np.intp), 0, NUM_SYMBOLS - 1)

    for dim in range(INPUTS):
        vals = inputs[:, dim]
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])

        bin_idx = np.where(
            vals <= q1, 0, np.where(vals <= q2, 1, np.where(vals <= q3, 2, 3))
        )

        counts = np.zeros((NUM_SYMBOLS, 4), dtype=np.int32)
        for s in range(NUM_SYMBOLS):
            s_mask = syms == s
            for b in range(4):
                counts[s, b] = np.sum(s_mask & (bin_idx == b))
        result[dim] = mi_from_contingency(counts)

    return result
