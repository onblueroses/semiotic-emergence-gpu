#!/usr/bin/env python3
"""Analyze semiotic-emergence simulation output.

Usage:
    python analyze.py output.csv                              # single run
    python analyze.py output.csv --plot                       # + charts
    python analyze.py output.csv --json                       # structured output
    python analyze.py run1.csv run2.csv                       # compare
    python analyze.py output.csv --trajectory t.csv           # + trajectory
    python analyze.py output.csv --input-mi i.csv             # + input MI
    python analyze.py output.csv --all t.csv i.csv            # everything
    python analyze.py output.csv --counterfactual nosig.csv   # signal value
    python analyze.py output.csv --all t.csv i.csv --plot     # full analysis + charts
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── Format detection ─────────────────────────────────────────────

CURRENT_COLUMNS = frozenset({
    "generation", "avg_fitness", "max_fitness", "signals_emitted",
    "iconicity", "mutual_info", "jsd_no_pred", "jsd_pred",
    "silence_corr", "sender_fit_corr", "traj_fluct_ratio",
    "receiver_fit_corr", "response_fit_corr", "silence_onset_jsd",
    "silence_move_delta", "signal_entropy",
    "avg_base_hidden", "min_base_hidden", "max_base_hidden",
    "avg_signal_hidden", "min_signal_hidden", "max_signal_hidden",
})

LEGACY_22_COLUMNS = frozenset({
    "generation", "avg_fitness", "max_fitness", "signals_emitted",
    "iconicity", "mutual_info", "confusion_ticks",
    "jsd_no_pred", "jsd_pred", "silence_corr",
    "mi_kin", "mi_rnd", "jsd_no_pred_kin", "jsd_no_pred_rnd",
    "jsd_pred_kin", "jsd_pred_rnd",
    "sender_fit_corr", "traj_fluct_ratio",
    "receiver_fit_corr", "response_fit_corr",
    "silence_onset_jsd", "silence_move_delta",
})

NEAT_COLUMNS = frozenset({
    "generation", "avg_fitness", "max_fitness", "species_count",
    "prey_alive", "signal_count", "mutual_information",
    "topographic_similarity", "iconicity",
})

# Renames for backward compat across formats
COLUMN_RENAMES = {
    "mutual_information": "mutual_info",
    "signal_count": "signals_emitted",
}

KEY_METRICS = ["avg_fitness", "avg_base_hidden", "avg_signal_hidden", "mutual_info", "jsd_pred", "silence_corr"]

COMPARISON_ROWS = [
    ("Generations",       None,                   None,        ","),
    ("Final avg fitness", "avg_fitness",          "final",     ".1f"),
    ("Peak avg fitness",  "avg_fitness",          "peak",      ".1f"),
    ("Sustained fitness", "avg_fitness",          "sustained", ".1f"),
    ("Final base hidden", "avg_base_hidden",     "final",     ".1f"),
    ("Peak base hidden",  "avg_base_hidden",     "peak",      ".1f"),
    ("Final sig hidden",  "avg_signal_hidden",   "final",     ".1f"),
    ("Peak sig hidden",   "avg_signal_hidden",   "peak",      ".1f"),
    ("Final MI",          "mutual_info",          "final",     ".4f"),
    ("Peak MI",           "mutual_info",          "peak",      ".4f"),
    ("MI peak gen",       "mutual_info",          "peak_gen",  ","),
    ("Sustained MI",      "mutual_info",          "sustained", ".4f"),
    ("JSD (pred)",        "jsd_pred",             "final",     ".4f"),
    ("Silence corr",      "silence_corr",         "final",     ".4f"),
    ("Silence min",       "silence_corr",         "min",       ".4f"),
    ("Sender-fitness",    "sender_fit_corr",      "final",     ".4f"),
    ("Response-fitness",  "response_fit_corr",    "final",     ".4f"),
    ("Max fluct ratio",   "traj_fluct_ratio",     "peak",      ".4f"),
    ("Signal entropy",    "signal_entropy",        "final",     ".4f"),
    ("Min entropy",       "signal_entropy",        "min",       ".4f"),
]


# ── Data loading ─────────────────────────────────────────────────

@dataclass
class Run:
    name: str
    generations: np.ndarray
    metrics: dict[str, np.ndarray]
    is_legacy: bool = False

    @staticmethod
    def from_csv(path: Path) -> "Run":
        path = Path(path)
        with open(path) as f:
            header = f.readline().strip().split(",")
        cols = frozenset(header)

        if "generation" not in cols:
            raise ValueError(f"No 'generation' column in {path}")

        is_legacy = "avg_base_hidden" not in cols and "avg_hidden" not in cols
        if is_legacy and cols != NEAT_COLUMNS:
            print(f"  [warn] Legacy format (no brain columns): {path.name}", file=sys.stderr)

        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if raw.size == 0:
            raise ValueError(f"No data rows in {path}")

        generations = raw[:, 0].astype(np.int64)
        metrics = {}
        for i, col in enumerate(header):
            if col == "generation":
                continue
            name = COLUMN_RENAMES.get(col, col)
            metrics[name] = raw[:, i]

        # Map v1 single-hidden to split-head columns for display compatibility
        if "avg_hidden" in metrics and "avg_base_hidden" not in metrics:
            metrics["avg_base_hidden"] = metrics["avg_hidden"]
            if "min_hidden" in metrics:
                metrics["min_base_hidden"] = metrics["min_hidden"]
            if "max_hidden" in metrics:
                metrics["max_base_hidden"] = metrics["max_hidden"]

        run_name = path.parent.name if path.parent.name not in (".", "") else path.stem
        return Run(name=run_name, generations=generations, metrics=metrics, is_legacy=is_legacy)

    @property
    def n(self) -> int:
        return len(self.generations)

    def get(self, metric: str) -> np.ndarray | None:
        return self.metrics.get(metric)


@dataclass
class TrajectoryData:
    generations: np.ndarray
    num_symbols: int
    matrix: np.ndarray       # (n, num_symbols, 4) symbol x distance bin counts
    jsd_sym: np.ndarray      # (n, num_symbols)
    trajectory_jsd: np.ndarray
    contrast_labels: list[str]
    contrasts: np.ndarray    # (n, num_contrasts)

    @staticmethod
    def from_csv(path: Path) -> "TrajectoryData":
        with open(path) as f:
            header = f.readline().strip().split(",")
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        col = {name: i for i, name in enumerate(header)}

        gen = raw[:, col["generation"]].astype(np.int64)
        n = len(gen)

        # Auto-detect number of symbols from columns
        num_symbols = 0
        while f"s{num_symbols}d0" in col:
            num_symbols += 1

        mat = np.zeros((n, num_symbols, 4))
        for s in range(num_symbols):
            for d in range(4):
                mat[:, s, d] = raw[:, col[f"s{s}d{d}"]]

        jsd = np.column_stack([raw[:, col[f"jsd_sym{i}"]] for i in range(num_symbols)])
        tj = raw[:, col["trajectory_jsd"]]

        # Auto-detect contrast pairs from columns
        contrast_labels = [c.replace("contrast_", "") for c in header if c.startswith("contrast_")]
        con = np.column_stack([raw[:, col[f"contrast_{p}"]] for p in contrast_labels]) if contrast_labels else np.zeros((n, 0))
        return TrajectoryData(gen, num_symbols, mat, jsd, tj, contrast_labels, con)

    @property
    def n(self) -> int:
        return len(self.generations)


@dataclass
class InputMIData:
    generations: np.ndarray
    columns: list[str]
    values: np.ndarray  # (n, num_cols)

    @staticmethod
    def from_csv(path: Path) -> "InputMIData":
        with open(path) as f:
            header = f.readline().strip().split(",")
        mi_cols = [c for c in header if c.startswith("mi_")]
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        col = {name: i for i, name in enumerate(header)}
        gen = raw[:, 0].astype(np.int64)
        vals = np.column_stack([raw[:, col[c]] for c in mi_cols])
        return InputMIData(gen, mi_cols, vals)


# ── Numerical utilities ──────────────────────────────────────────

def window_size(n: int) -> int:
    return max(100, n // 200)


def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    if len(arr) < w:
        return arr.copy()
    kernel = np.ones(w) / w
    smooth = np.convolve(arr, kernel, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), smooth])


def rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    n = len(arr)
    if n < w:
        return np.zeros(n)
    cs = np.cumsum(np.insert(arr, 0, 0.0))
    cs2 = np.cumsum(np.insert(arr ** 2, 0, 0.0))
    s = cs[w:] - cs[:-w]
    s2 = cs2[w:] - cs2[:-w]
    var = np.maximum(0.0, s2 / w - (s / w) ** 2)
    return np.concatenate([np.full(w - 1, np.nan), np.sqrt(var)])


def cusum_changepoints(arr: np.ndarray, threshold: float = 4.0, drift: float = 0.5) -> list[int]:
    """CUSUM on z-scored series. Returns indices of structural breaks."""
    if len(arr) < 20:
        return []
    sigma = np.std(arr)
    if sigma < 1e-10:
        return []
    z = (arr - np.mean(arr)) / sigma
    s_hi = s_lo = 0.0
    points = []
    for i in range(1, len(z)):
        s_hi = max(0.0, s_hi + z[i] - drift)
        s_lo = max(0.0, s_lo - z[i] - drift)
        if s_hi > threshold or s_lo > threshold:
            points.append(i)
            s_hi = s_lo = 0.0
    return points


def merge_changepoints(points_by_metric: dict[str, list[int]], merge_window: int) -> list[int]:
    """Merge change points from multiple metrics within merge_window of each other."""
    all_pts = sorted({p for pts in points_by_metric.values() for p in pts})
    if not all_pts:
        return []
    merged = [all_pts[0]]
    for p in all_pts[1:]:
        if p - merged[-1] > merge_window:
            merged.append(p)
        else:
            merged[-1] = (merged[-1] + p) // 2
    return merged


def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


# ── Single-run analysis ─────────────────────────────────────────

def summary_stats(run: Run) -> dict:
    tail_start = max(0, int(run.n * 0.9))
    result = {}
    for name, arr in run.metrics.items():
        peak_idx = int(np.argmax(arr))
        entry = {
            "final": float(arr[-1]),
            "peak": float(np.max(arr)),
            "peak_gen": int(run.generations[peak_idx]),
            "sustained": float(np.mean(arr[tail_start:])),
        }
        if name in ("silence_corr", "signal_entropy"):
            min_idx = int(np.argmin(arr))
            entry["min"] = float(np.min(arr))
            entry["min_gen"] = int(run.generations[min_idx])
        result[name] = entry
    return result


def compute_rolling(run: Run) -> dict[str, dict]:
    w = window_size(run.n)
    result = {}
    for m in KEY_METRICS:
        arr = run.get(m)
        if arr is None:
            continue
        result[m] = {"window": w, "mean": rolling_mean(arr, w), "std": rolling_std(arr, w)}
    return result


def correlation_matrix(run: Run) -> tuple[list[str], np.ndarray]:
    # Skip metrics with zero variance (all-zero columns produce NaN correlations)
    available = [m for m in KEY_METRICS
                 if run.get(m) is not None and np.std(run.get(m)) > 1e-10]
    if len(available) < 2:
        return available, np.array([[1.0]])
    arrays = np.array([run.get(m) for m in available])
    return available, np.corrcoef(arrays)


def lag_correlation(run: Run, metric_a: str, metric_b: str, max_lag: int) -> tuple[int, float] | None:
    """Find the lag with strongest correlation between two metrics."""
    a, b = run.get(metric_a), run.get(metric_b)
    if a is None or b is None or run.n < max_lag * 3:
        return None
    best_lag, best_r = 0, 0.0
    step = max(1, max_lag // 20)
    for lag in range(-max_lag, max_lag + 1, step):
        if lag > 0:
            r = np.corrcoef(a[:-lag], b[lag:])[0, 1]
        elif lag < 0:
            r = np.corrcoef(a[-lag:], b[:lag])[0, 1]
        else:
            r = np.corrcoef(a, b)[0, 1]
        if not np.isnan(r) and abs(r) > abs(best_r):
            best_lag, best_r = lag, float(r)
    return best_lag, best_r


def detect_changepoints(run: Run) -> dict[str, list[tuple[int, float]]]:
    smooth_w = max(500, run.n // 20)
    result = {}
    for m in KEY_METRICS:
        arr = run.get(m)
        if arr is None:
            continue
        smoothed = rolling_mean(arr, smooth_w)
        valid = smoothed[~np.isnan(smoothed)]
        if len(valid) < 20:
            continue
        raw_indices = cusum_changepoints(valid, threshold=6.0)
        offset = smooth_w - 1
        indices = [i + offset for i in raw_indices if i + offset < run.n]
        # Enforce minimum gap
        if indices:
            filtered = [indices[0]]
            for idx in indices[1:]:
                if idx - filtered[-1] >= smooth_w:
                    filtered.append(idx)
            indices = filtered
        result[m] = [(int(run.generations[i]), float(arr[i])) for i in indices]
    return result


def classify_segment(run: Run, start: int, end: int) -> str:
    deviations = {}
    for m in KEY_METRICS:
        arr = run.get(m)
        if arr is None:
            continue
        full_std = np.std(arr)
        if full_std < 1e-10:
            continue
        deviations[m] = (np.mean(arr[start:end]) - np.mean(arr)) / full_std

    if not deviations:
        return "unknown"

    dominant = max(deviations, key=lambda k: abs(deviations[k]))
    z = deviations[dominant]
    if abs(z) < 0.5:
        return "baseline"

    labels = {
        "avg_fitness": "high-fitness" if z > 0 else "low-fitness",
        "avg_base_hidden": "large-base-brain" if z > 0 else "small-base-brain",
        "avg_signal_hidden": "large-signal-brain" if z > 0 else "small-signal-brain",
        "mutual_info": "high-MI" if z > 0 else "low-MI",
        "jsd_pred": "high-response" if z > 0 else "low-response",
        "silence_corr": "silence-active" if z < 0 else "silence-weak",
    }
    return labels.get(dominant, f"{dominant}={'high' if z > 0 else 'low'}")


def detect_epochs(run: Run) -> list[tuple[int, int, str]]:
    # Reuse smoothed change points from detect_changepoints
    cps = detect_changepoints(run)
    cp_by_metric = {}
    for m, pts in cps.items():
        # Convert gen values back to indices for merging
        gen_to_idx = {int(g): i for i, g in enumerate(run.generations)}
        cp_by_metric[m] = [gen_to_idx.get(g, 0) for g, _ in pts]

    # Merge CPs from different metrics that are close together
    merge_w = max(200, run.n // 100)
    boundaries = merge_changepoints(cp_by_metric, merge_w)

    if not boundaries:
        return [(int(run.generations[0]), int(run.generations[-1]), classify_segment(run, 0, run.n))]

    raw_epochs = []
    segments = [0] + boundaries + [run.n - 1]
    for i in range(len(segments) - 1):
        s, e = segments[i], segments[i + 1]
        label = classify_segment(run, s, e)
        raw_epochs.append((int(run.generations[s]), int(run.generations[min(e, run.n - 1)]), label))

    # Merge consecutive epochs with the same label
    if not raw_epochs:
        return raw_epochs
    merged = [raw_epochs[0]]
    for start, end, label in raw_epochs[1:]:
        if label == merged[-1][2]:
            merged[-1] = (merged[-1][0], end, label)
        else:
            merged.append((start, end, label))
    return merged


def metric_health(run: Run) -> dict[str, dict]:
    w = window_size(run.n)
    result = {}
    for m in ["response_fit_corr", "silence_onset_jsd", "silence_move_delta", "jsd_pred", "jsd_no_pred"]:
        arr = run.get(m)
        if arr is None:
            continue
        nonzero = int(np.count_nonzero(arr))
        n_windows = max(1, run.n // w)
        active = sum(1 for i in range(n_windows) if np.any(arr[i * w:(i + 1) * w] != 0))
        result[m] = {
            "nonzero": nonzero,
            "pct": 100 * nonzero / run.n,
            "active_windows_pct": 100 * active / n_windows,
        }
    return result


# ── Counterfactual analysis ──────────────────────────────────────

def auto_detect_counterfactual(main_path: Path) -> Path | None:
    if main_path.parent.name == "standard":
        candidate = main_path.parent.parent / "nosignal" / main_path.name
        return candidate if candidate.exists() else None
    return None


def counterfactual_analysis(standard: Run, control: Run) -> dict:
    s_fit = standard.get("avg_fitness")
    c_fit = control.get("avg_fitness")
    n = min(len(s_fit), len(c_fit))
    delta = s_fit[:n] - c_fit[:n]
    gens = standard.generations[:n]
    signal_value = float(np.trapezoid(delta, gens))
    return {
        "common_gens": n,
        "standard_final": float(s_fit[n - 1]),
        "control_final": float(c_fit[n - 1]),
        "delta_final": float(delta[-1]),
        "signal_value_integral": signal_value,
        "pct_advantage": float(100 * delta[-1] / c_fit[n - 1]) if c_fit[n - 1] != 0 else 0.0,
    }


# ── Output formatting ───────────────────────────────────────────

def print_summary(run: Run, stats: dict):
    print(f"\n{'=' * 60}")
    print(f"  {run.name}  ({run.n:,} generations)")
    print(f"{'=' * 60}")
    s = stats

    af = s.get("avg_fitness", {})
    mf = s.get("max_fitness", {})
    print("\n  FITNESS")
    print(f"    Final avg/max:  {af.get('final', 0):.1f} / {mf.get('final', 0):.1f}")
    print(f"    Peak avg/max:   {af.get('peak', 0):.1f} / {mf.get('peak', 0):.1f}")
    print(f"    Sustained avg:  {af.get('sustained', 0):.1f}  (last 10%)")

    if not run.is_legacy:
        ab = s.get("avg_base_hidden", {})
        ash = s.get("avg_signal_hidden", {})
        print("\n  BRAIN SIZE (split-head)")
        print(f"    Base hidden:    {ab.get('final', 0):.1f}  "
              f"[{s.get('min_base_hidden', {}).get('final', 0):.0f}-{s.get('max_base_hidden', {}).get('final', 0):.0f}]"
              f"  (peak {ab.get('peak', 0):.1f} at gen {ab.get('peak_gen', 0):,})")
        print(f"    Signal hidden:  {ash.get('final', 0):.1f}  "
              f"[{s.get('min_signal_hidden', {}).get('final', 0):.0f}-{s.get('max_signal_hidden', {}).get('final', 0):.0f}]"
              f"  (peak {ash.get('peak', 0):.1f} at gen {ash.get('peak_gen', 0):,})")
        total = ab.get("final", 0) + ash.get("final", 0)
        drain = 0.0008 + total * 0.00001
        print(f"    Total neurons:  {total:.1f}  drain={drain:.5f}/tick  ({drain * 500:.2f} over 500 ticks)")

    mi = s.get("mutual_info", {})
    print("\n  MUTUAL INFORMATION")
    print(f"    Final:          {mi.get('final', 0):.4f}")
    print(f"    Peak:           {mi.get('peak', 0):.4f}  (gen {mi.get('peak_gen', 0):,})")
    print(f"    Sustained:      {mi.get('sustained', 0):.4f}  (last 10%)")

    jp = s.get("jsd_pred", {})
    jn = s.get("jsd_no_pred", {})
    sc = s.get("silence_corr", {})
    print("\n  RECEIVER BEHAVIOR")
    print(f"    JSD (pred):     {jp.get('final', 0):.4f}  (peak {jp.get('peak', 0):.4f})")
    print(f"    JSD (no pred):  {jn.get('final', 0):.4f}")
    print(f"    Silence corr:   {sc.get('final', 0):.4f}  (min {sc.get('min', 0):.4f})")

    sf = s.get("sender_fit_corr", {})
    rf = s.get("receiver_fit_corr", {})
    rsf = s.get("response_fit_corr", {})
    print("\n  FITNESS COUPLING")
    print(f"    Sender-fitness: {sf.get('final', 0):.4f}")
    print(f"    Receiver-fit:   {rf.get('final', 0):.4f}")
    print(f"    Response-fit:   {rsf.get('final', 0):.4f}")


def print_metric_health(run: Run, health: dict):
    print(f"\n  METRIC HEALTH ({run.n:,} gens, window={window_size(run.n)})")
    for m, h in health.items():
        print(f"    {m:>22s}: {h['nonzero']:>6,} non-zero ({h['pct']:.1f}%), "
              f"{h['active_windows_pct']:.0f}% windows active")


def print_changepoints(cps: dict):
    has_any = any(pts for pts in cps.values())
    if not has_any:
        return
    print("\n  CHANGE POINTS (CUSUM)")
    for m, pts in cps.items():
        if pts:
            desc = ", ".join(f"gen {g:,} ({v:.4f})" for g, v in pts)
            print(f"    {m}: {desc}")


def print_epochs(epochs: list):
    if not epochs:
        return
    print("\n  EVOLUTIONARY EPOCHS (data-driven)")
    for start, end, regime in epochs:
        print(f"    gen {start:>7,} - {end:>7,}  ({end - start:>6,} gens)  {regime}")


def print_correlations(names: list[str], corr: np.ndarray):
    if len(names) < 2:
        return
    w = 12
    print("\n  CROSS-METRIC CORRELATIONS")
    print(f"    {'':>15s}" + "".join(f"{n[:w]:>{w}s}" for n in names))
    for i, name in enumerate(names):
        vals = "".join(f"{corr[i, j]:>{w}.3f}" for j in range(len(names)))
        print(f"    {name:>15s}{vals}")


def print_lag_correlations(run: Run):
    if run.is_legacy:
        return
    max_lag = min(200, run.n // 10)
    if max_lag < 10:
        return
    print("\n  LAG CORRELATIONS")
    for brain_metric, label in [("avg_base_hidden", "base brain"), ("avg_signal_hidden", "sig brain")]:
        result = lag_correlation(run, brain_metric, "mutual_info", max_lag)
        if result is None:
            continue
        lag, r = result
        print(f"    {label} -> MI: strongest r={r:.3f} at lag={lag:+d} gens", end="")
        if lag > 0:
            print(f"  ({label} LEADS MI by ~{lag} gens)")
        elif lag < 0:
            print(f"  (MI LEADS {label} by ~{-lag} gens)")
        else:
            print("  (simultaneous)")


def compare_runs(runs: list[Run], stats_list: list[dict]):
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")

    headers = [r.name[:20] for r in runs]
    col_w = max(max(len(h) for h in headers) + 2, 14)
    show_delta = len(runs) == 2

    hdr_line = f"\n    {'':25s}" + "".join(f"{h:>{col_w}s}" for h in headers)
    if show_delta:
        hdr_line += f"{'delta':>{col_w}s}"
    print(hdr_line)
    sep_cols = len(runs) + (1 if show_delta else 0)
    print(f"    {'-' * 25}" + ("-" * col_w) * sep_cols)

    for label, metric, subkey, fmt in COMPARISON_ROWS:
        vals = []
        for run, s in zip(runs, stats_list):
            if metric is None:
                vals.append(run.n)
            elif metric in s and subkey in s[metric]:
                vals.append(s[metric][subkey])
            else:
                vals.append(None)

        if all(v is None for v in vals):
            continue

        formatted = []
        for v in vals:
            if v is None:
                formatted.append("-")
            else:
                formatted.append(f"{v:{fmt}}")

        line = f"    {label:<25s}" + "".join(f"{v:>{col_w}s}" for v in formatted)

        if show_delta and vals[0] is not None and vals[1] is not None:
            try:
                delta = vals[1] - vals[0]
                line += f"{delta:>+{col_w}{fmt}}"
            except (TypeError, ValueError):
                line += " " * col_w
        print(line)


def print_counterfactual(result: dict):
    print(f"\n{'=' * 60}")
    print("  COUNTERFACTUAL ANALYSIS (signal value)")
    print(f"{'=' * 60}")
    print(f"    Common generations:      {result['common_gens']:,}")
    print(f"    Standard final fitness:  {result['standard_final']:.1f}")
    print(f"    No-signal final fitness: {result['control_final']:.1f}")
    print(f"    Fitness delta:           {result['delta_final']:+.1f}")
    print(f"    Signal value (integral): {result['signal_value_integral']:,.0f}")
    print(f"    Advantage:               {result['pct_advantage']:+.1f}%")


def print_divergence(path: Path):
    with open(path) as f:
        header = f.readline().strip().split(",")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if raw.size == 0:
        return
    s_cols = [c for c in header if c.startswith("s") and c != "seed"]
    n_seeds = len(s_cols)

    print(f"\n{'=' * 60}")
    print(f"  CROSS-POPULATION DIVERGENCE  ({n_seeds} seeds)")
    print(f"{'=' * 60}")
    w = 10
    print(f"\n    {'':>6s}" + "".join(f"{c:>{w}s}" for c in s_cols))
    for i in range(len(raw)):
        seed = int(raw[i, 0])
        vals = "".join(f"{raw[i, j + 1]:>{w}.4f}" for j in range(n_seeds))
        print(f"    s{seed:>4d}{vals}")


# ── Trajectory analysis ──────────────────────────────────────────

def print_trajectory(traj: TrajectoryData):
    n = traj.n
    ns = traj.num_symbols
    print(f"\n{'=' * 60}")
    print(f"  TRAJECTORY ANALYSIS  ({n:,} generations, {ns} symbols)")
    print(f"{'=' * 60}")

    tail_start = max(0, int(n * 0.9))

    # Symbol contrast - show top contrasts by tail average
    if traj.contrasts.shape[1] > 0:
        print("\n  SYMBOL CONTRAST (pairwise JSD, top 6 by tail avg)")
        tail_avgs = []
        for i, label in enumerate(traj.contrast_labels):
            col = traj.contrasts[:, i]
            tail_nz = col[tail_start:][col[tail_start:] != 0]
            tail_avg = float(np.mean(tail_nz)) if len(tail_nz) else 0.0
            tail_avgs.append((tail_avg, i, label))
        tail_avgs.sort(reverse=True)
        for tail_avg, i, label in tail_avgs[:6]:
            col = traj.contrasts[:, i]
            nz = col[col != 0]
            if len(nz):
                print(f"    contrast_{label}: avg {np.mean(nz):.4f}, tail avg {tail_avg:.4f}, max {np.max(nz):.4f}")

    # Top trajectory JSD spikes
    top_idx = np.argsort(traj.trajectory_jsd)[-10:][::-1]
    print("\n  TOP 10 TRAJECTORY JSD SPIKES")
    for idx in top_idx:
        jsd = traj.trajectory_jsd[idx]
        if jsd > 0:
            print(f"    gen {traj.generations[idx]:>7,}: {jsd:.4f}")

    # Symbol usage evolution
    totals = traj.matrix.sum(axis=2)  # (n, ns)
    checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    print("\n  SYMBOL USAGE EVOLUTION")
    for idx in checkpoints:
        gen = traj.generations[idx]
        t = totals[idx]
        total = t.sum() or 1
        pcts = 100 * t / total
        parts = "  ".join(f"s{s}={pcts[s]:5.1f}%" for s in range(ns))
        print(f"    gen {gen:>7,}: {parts}")

    # Herfindahl index
    total_per_gen = totals.sum(axis=1, keepdims=True)
    shares = totals / np.maximum(total_per_gen, 1)
    hhi = (shares ** 2).sum(axis=1)
    equal_hhi = 1.0 / ns
    print(f"\n  SYMBOL CONCENTRATION (Herfindahl: {equal_hhi:.2f}=equal, 1.0=monopoly)")
    for idx in checkpoints:
        print(f"    gen {traj.generations[idx]:>7,}: HHI={hhi[idx]:.3f}")

    # Predator-proximal analysis
    near = traj.matrix[:, :, 0]
    tail_near = near[tail_start:].sum(axis=0)
    total_near = tail_near.sum() or 1
    print("\n  PREDATOR-PROXIMAL SYMBOLS (d0 = closest distance bin, last 10%)")
    for s in range(ns):
        print(f"    sym{s}: {100 * tail_near[s] / total_near:.1f}%")


# ── Input MI analysis ────────────────────────────────────────────

def print_input_mi(data: InputMIData):
    n = data.values.shape[0]
    print(f"\n{'=' * 60}")
    print(f"  INPUT MI ANALYSIS  ({n:,} generations)")
    print(f"{'=' * 60}")

    tail_start = max(0, int(n * 0.9))
    tail_means = data.values[tail_start:].mean(axis=0)
    ranking = np.argsort(tail_means)[::-1]

    print("\n  SUSTAINED ENCODING (avg MI last 10%)")
    for i in ranking:
        val = tail_means[i]
        bar = "#" * int(val * 200)
        print(f"    {data.columns[i]:>15s}: {val:.4f}  {bar}")

    # Top 3 evolution
    top3_idx = ranking[:3]
    top3_names = [data.columns[i] for i in top3_idx]
    print("\n  TOP 3 ENCODING EVOLUTION")
    print("    gen".ljust(12) + "".join(f"{c:>16s}" for c in top3_names))
    checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in checkpoints:
        gen = data.generations[idx]
        vals = "".join(f"{data.values[idx, i]:>16.4f}" for i in top3_idx)
        print(f"    {gen:>7,}{vals}")

    # Ranking stability
    if n > 100:
        early = data.values[:n // 10].mean(axis=0)
        mid = data.values[n // 3:2 * n // 3].mean(axis=0)
        late = data.values[tail_start:].mean(axis=0)
        print("\n  ENCODING STABILITY (Spearman rank correlation)")
        print(f"    early vs mid:   {spearman_rank_corr(early, mid):.3f}")
        print(f"    mid vs late:    {spearman_rank_corr(mid, late):.3f}")
        print(f"    early vs late:  {spearman_rank_corr(early, late):.3f}")


# ── Visualization ────────────────────────────────────────────────

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_run(run: Run, rolling: dict, cps: dict, output_dir: Path, traj: TrajectoryData | None = None):
    plt = _import_plt()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{run.name}  ({run.n:,} generations)", fontsize=14)
    gens = run.generations

    # Panel 1: Fitness + Brain
    ax1 = axes[0, 0]
    ax1.set_title("Fitness & Brain Size")
    fit = run.get("avg_fitness")
    if fit is not None:
        ax1.plot(gens, fit, alpha=0.3, color="#2196F3", linewidth=0.5)
        rm = rolling.get("avg_fitness", {}).get("mean")
        if rm is not None:
            ax1.plot(gens, rm, color="#2196F3", linewidth=1.5, label="avg_fitness")
    ax1.set_ylabel("avg_fitness", color="#2196F3")
    ax1.set_xlabel("generation")

    if not run.is_legacy:
        ax1b = ax1.twinx()
        for brain_m, color, label in [("avg_base_hidden", "#4CAF50", "base_hidden"),
                                       ("avg_signal_hidden", "#8BC34A", "sig_hidden")]:
            hid = run.get(brain_m)
            if hid is not None:
                ax1b.plot(gens, hid, alpha=0.3, color=color, linewidth=0.5)
                rm = rolling.get(brain_m, {}).get("mean")
                if rm is not None:
                    ax1b.plot(gens, rm, color=color, linewidth=1.5, label=label)
        ax1b.set_ylabel("hidden size", color="#4CAF50")
        ax1b.legend(fontsize=7, loc="upper left")

    # Panel 2: Signal metrics
    ax2 = axes[0, 1]
    ax2.set_title("Signal Metrics")
    colors = {"mutual_info": "#FF9800", "jsd_pred": "#9C27B0", "silence_corr": "#F44336"}
    for m in ["mutual_info", "jsd_pred", "silence_corr"]:
        arr = run.get(m)
        if arr is None:
            continue
        ax2.plot(gens, arr, alpha=0.2, color=colors[m], linewidth=0.5)
        rm = rolling.get(m, {}).get("mean")
        if rm is not None:
            ax2.plot(gens, rm, color=colors[m], linewidth=1.5, label=m)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("generation")

    # Panel 3: Phase transitions + change points
    ax3 = axes[1, 0]
    ax3.set_title("Phase Transitions")
    if traj is not None:
        ax3.plot(traj.generations, traj.trajectory_jsd, color="#795548", linewidth=0.8, label="trajectory_jsd")
    else:
        fluct = run.get("traj_fluct_ratio")
        if fluct is not None:
            ax3.plot(gens, fluct, color="#795548", linewidth=0.8, label="traj_fluct_ratio")
    all_cp_gens = {g for pts in cps.values() for g, _ in pts}
    for g in all_cp_gens:
        ax3.axvline(x=g, color="red", alpha=0.4, linestyle="--", linewidth=0.8)
    ax3.legend(fontsize=8)
    ax3.set_xlabel("generation")

    # Panel 4: Metric health
    ax4 = axes[1, 1]
    ax4.set_title("Metric Health (% windows active)")
    health = metric_health(run)
    names = list(health.keys())
    if names:
        pcts = [health[n]["active_windows_pct"] for n in names]
        short = [n.replace("_corr", "").replace("_", "\n") for n in names]
        y_pos = range(len(names))
        ax4.barh(list(y_pos), pcts, color="#607D8B")
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(short, fontsize=7)
        ax4.set_xlim(0, 100)
        ax4.set_xlabel("% windows active")

    plt.tight_layout()
    out = output_dir / f"{run.name}_analysis.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


def plot_counterfactual(standard: Run, control: Run, result: dict, output_dir: Path):
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Counterfactual: Signal Value", fontsize=14)

    n = result["common_gens"]
    s_fit = standard.get("avg_fitness")[:n]
    c_fit = control.get("avg_fitness")[:n]
    g = standard.generations[:n]

    ax.plot(g, s_fit, label=standard.name, color="#2196F3")
    ax.plot(g, c_fit, label=control.name, color="#F44336")
    ax.fill_between(g, c_fit, s_fit, where=s_fit > c_fit, alpha=0.2, color="#4CAF50", label="signal advantage")
    ax.fill_between(g, c_fit, s_fit, where=s_fit < c_fit, alpha=0.2, color="#FF5722")
    ax.set_xlabel("Generation")
    ax.set_ylabel("avg_fitness")
    ax.legend()

    plt.tight_layout()
    out = output_dir / f"{standard.name}_counterfactual.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


def plot_comparison(runs: list[Run], output_dir: Path):
    plt = _import_plt()
    metrics = [("avg_fitness", "Fitness"), ("avg_base_hidden", "Base Hidden"),
               ("avg_signal_hidden", "Signal Hidden"), ("mutual_info", "Mutual Info")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Run Comparison", fontsize=14)

    for ax, (m, label) in zip(axes.flat, metrics):
        for run in runs:
            arr = run.get(m)
            if arr is not None:
                ax.plot(run.generations, arr, label=run.name, alpha=0.7)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlabel("generation")

    plt.tight_layout()
    out = output_dir / "comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ── JSON output ──────────────────────────────────────────────────

def _np_convert(obj):
    if isinstance(obj, dict):
        return {k: _np_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_convert(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_json(run, stats, health, cps, epochs, corr_names, corr_mat):
    return _np_convert({
        "name": run.name,
        "generations": run.n,
        "is_legacy": run.is_legacy,
        "stats": stats,
        "health": health,
        "changepoints": {m: [{"gen": g, "val": v} for g, v in pts] for m, pts in cps.items()},
        "epochs": [{"start": s, "end": e, "regime": r} for s, e, r in epochs],
        "correlations": {"metrics": corr_names, "matrix": corr_mat.tolist()},
    })


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze semiotic-emergence output")
    parser.add_argument("output_csvs", nargs="+", help="output.csv file(s)")
    parser.add_argument("--trajectory", "-t", help="trajectory.csv file")
    parser.add_argument("--input-mi", "-m", help="input_mi.csv file")
    parser.add_argument("--all", "-a", nargs=2, metavar=("TRAJ", "INPUT_MI"),
                        help="trajectory.csv and input_mi.csv")
    parser.add_argument("--counterfactual", "-c", help="no-signals output.csv")
    parser.add_argument("--plot", "-p", action="store_true", help="save charts as PNG")
    parser.add_argument("--json", "-j", action="store_true", help="structured JSON output")
    parser.add_argument("--epochs", "-e", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    traj_path = args.trajectory or (args.all[0] if args.all else None)
    mi_path = args.input_mi or (args.all[1] if args.all else None)

    # Load runs
    runs = []
    for p in args.output_csvs:
        runs.append(Run.from_csv(Path(p)))

    # Load optional data
    traj = TrajectoryData.from_csv(Path(traj_path)) if traj_path else None
    mi_data = InputMIData.from_csv(Path(mi_path)) if mi_path else None

    cf_path = args.counterfactual
    if not cf_path:
        detected = auto_detect_counterfactual(Path(args.output_csvs[0]))
        if detected:
            cf_path = str(detected)
    cf_run = Run.from_csv(Path(cf_path)) if cf_path else None

    json_out = {} if args.json else None

    # Per-run analysis
    all_stats = []
    for run in runs:
        stats = summary_stats(run)
        cps = detect_changepoints(run)
        epochs = detect_epochs(run)
        health = metric_health(run)
        corr_names, corr_mat = correlation_matrix(run)
        all_stats.append(stats)

        if json_out is not None:
            json_out[run.name] = build_json(run, stats, health, cps, epochs, corr_names, corr_mat)
        else:
            print_summary(run, stats)
            print_metric_health(run, health)
            print_changepoints(cps)
            print_epochs(epochs)
            print_correlations(corr_names, corr_mat)
            print_lag_correlations(run)

        if args.plot:
            rolling = compute_rolling(run)
            plot_run(run, rolling, cps, Path(args.output_csvs[0]).parent, traj if run is runs[0] else None)

    # Divergence auto-detect
    div_path = Path(args.output_csvs[0]).parent / "divergence.csv"
    if div_path.exists() and json_out is None:
        print_divergence(div_path)

    # Multi-run comparison
    if len(runs) > 1:
        if json_out is None:
            compare_runs(runs, all_stats)
        if args.plot:
            plot_comparison(runs, Path(args.output_csvs[0]).parent)

    # Counterfactual
    if cf_run:
        result = counterfactual_analysis(runs[0], cf_run)
        if json_out is None:
            print_counterfactual(result)
        else:
            json_out["counterfactual"] = _np_convert(result)
        if args.plot:
            plot_counterfactual(runs[0], cf_run, result, Path(args.output_csvs[0]).parent)

    # Trajectory
    if traj:
        if json_out is None:
            print_trajectory(traj)

    # Input MI
    if mi_data:
        if json_out is None:
            print_input_mi(mi_data)

    if json_out is not None:
        print(json.dumps(json_out, indent=2))


if __name__ == "__main__":
    main()
