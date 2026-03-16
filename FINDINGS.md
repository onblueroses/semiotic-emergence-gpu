# Findings

Experimental results from the GPU (JAX/XLA) port of semiotic-emergence. This documents the first large-scale run: 5,000 population, 100,000 generations on an A100 80GB.

## Run Registry

| ID | Pop | Gens | Grid | Seed | GPU | Runtime | Data |
|----|-----|------|------|------|-----|---------|------|
| 5k-100k-s42 | 5,000 | 100,000 | 150x150 | 42 | A100 SXM4 80GB | ~23 hours | `runs/5k-100k-seed42/` |

Parameters: 5 flee zones, 2 freeze zones, zone_radius=60, zone_speed=3.75, zone_drain=0.15, signal_cost=0.015, signal_range=60, food=750, ticks=500, patch_ratio=0.5, kin_bonus=0.10. All derived from `scale = grid_size / 20 = 7.5`.

---

## Run: 5k-100k-s42

### Summary

Signals have adaptive value. Context-dependent receiver behavior emerges with a phase transition around generation 40,000. The signal system is simpler than initial analysis suggested - effectively a single-channel intensity signal with slight per-symbol variation, not a differentiated vocabulary. A broken measurement artifact (response_fit_corr) was identified and explains a persistent zero result across the entire project history.

### Fitness trajectory

| Epoch | Avg fitness | Zone deaths/gen |
|-------|-------------|-----------------|
| Gen 0-10k | 1089 | 67 |
| Gen 10-20k | 1099 | 68 |
| Gen 30-50k | 1036 | 63 |
| Gen 50-70k | 1003 | 59 |
| Gen 90-100k | 984 | 58 |

Fitness plateaus by Gen ~5k (1100), then *slowly declines* to ~984 by Gen 100k. This is not collapse - max_fitness stays at ~25,499 throughout. The population trades raw survival for signal complexity (see regime analysis below). Zone deaths do not improve over 100k generations - the population gets better at surviving despite zones, not at avoiding them.

### Three evolutionary regimes

K-means clustering (k=3) on 1000-gen windows of 7 metrics finds three temporally contiguous phases with only 5 transitions:

| Phase | Generations | Fitness | JSD (pred) | Entropy | Deaths | Character |
|-------|-------------|---------|------------|---------|--------|-----------|
| Early | 0-16k | 1103 | 0.033 | 1.699 | 68 | Survival optimization, symbol frequency biased |
| Plateau | 18k-48k | 1048 | 0.033 | 1.731 | 63 | Leveling off, signal usage equalizing |
| Late | 48k-100k | 994 | 0.057 | 1.751 | 58 | Signal context-dependence rises, fitness declines |

The late regime shows 73% higher context-dependence (JSD) than the plateau, at the cost of 5% lower fitness.

### Confirmed findings (survive null checks)

#### 1. Signals have adaptive value

Detrended correlation between signals_emitted and avg_fitness: r=+0.51, p=0.00. Within each 10k-gen block, high-signal generations average +52 fitness points over low-signal generations. Consistent across all 100k gens (range: +48 to +63). sender_fit_corr is positive throughout (0.047-0.054).

silence_corr is consistently negative (-0.169 mean): prey that signal are fitter than prey that don't.

#### 2. Signal inputs dominate information budget (87%)

Per-input MI comparison (last 10k gens):

| Input category | Inputs | Mean MI per input | % of total |
|----------------|--------|-------------------|------------|
| Signal strength | 6 | 0.0377 | - |
| Signal direction | 12 | 0.0155 | - |
| Body state | 3 | 0.0013 | - |
| Food | 3 | 0.0009 | - |
| Memory | 8 | 0.0065 | - |
| Ally | 3 | 0.0006 | - |
| **Signals (total)** | **18** | **0.0229** | **87.3%** |

This is not a dimensionality artifact. Per-input signal MI is 7x higher than per-input non-signal MI. Comparing only always-present continuous inputs (signal strength vs body state), signals win 30x. Signal direction MI is also high (0.0155), not just "is signal present."

The signal system is primarily self-referential: what a prey signals depends overwhelmingly on what signals it's hearing, not on food, danger, or allies. This is meta-communication.

#### 3. Symbol specialization is real but narrow

Kruskal-Wallis test across 6 symbols' JSD: H=5444, p=0.00. All pairwise Mann-Whitney comparisons significant at p << 0.001.

Symbol 5 develops the highest context-dependence (JSD=0.057 by Gen 90k), diverging from the pack starting around Gen 40k (sym5/sym4 ratio goes from 0.9x to 3.1x). This is statistically real.

However, the cross-symbol MI correlation matrix is almost rank-1: **PC1 explains 89.9% of variance**. All 6 symbols' informativeness rises and falls together. There is effectively one "signal channel" with slight per-symbol amplitude differences, not 6 independent symbols. The system is closer to alarm pheromones (one signal, varying intensity) than vocabulary.

#### 4. The ~40k phase transition is real

Piecewise linear regression with breakpoint at Gen 40k vs simple linear: F=216.6, p=1.1e-16. Gen 40k is the optimal breakpoint across all tested (10k-90k in 5k steps). JSD slope flips from -1.3e-7 (slightly declining) to +4.5e-7 (rising). Symbol 5's JSD ramps steadily from 0.009 to 0.025 across Gen 35k-50k.

#### 5. Senders are noisy, receivers extract meaning

mutual_info (context -> symbol choice) is only 0.001. Senders barely encode environmental context into which symbol they choose. But jsd_pred (symbol -> receiver behavior near zones) is 0.066. Receivers respond differently to different symbols near zones, despite senders not deliberately choosing symbols based on context. The system works through statistical regularity in the sender population, not intentional encoding.

#### 6. Signals are arbitrary conventions

Iconicity is flat at 0.076 across 100k generations (linear trend slope=9.4e-10, p=0.70). No structural resemblance between signals and referents.

Note: without a null distribution (what would random weights produce?), we cannot confirm that 0.076 is above baseline noise. The "arbitrary convention" interpretation requires additional validation.

### Findings with caveats

#### 7. Receiver benefit ratio (6x) is partly architectural

receiver_fit_corr (0.28) is consistently ~6x sender_fit_corr (0.047). But the NN architecture makes receiver-fitness coupling easier by construction: signals are direct movement inputs. A random-weight baseline would be needed to determine how much of this ratio is learned vs structural.

The ratio is slowly declining (0.293 -> 0.282 over 100k gens, -2.5%), suggesting the population is optimizing other fitness factors that reduce signal-following's relative contribution.

#### 8. Memory tracks general signal activity, not specific symbols

All 8 memory cells correlate with Symbol 4 MI (r=0.78-0.90 raw). But controlling for total signal MI, the partial correlation drops to r=-0.017. Memory tracks overall signal activity, not Symbol 4 specifically. Memory MI grows 2-5x from baseline over 100k gens, with onset around Gen 57k for most cells.

### Broken measurement: response_fit_corr

**response_fit_corr = 0.000 across all 100,000 generations. This is a measurement artifact, not a biological result.**

The metric computes per-prey receiver JSD between "hearing signal" and "not hearing signal" conditions, requiring min 10 samples in each bucket. With signal_range=60 on a 150x150 grid and ~3,740 emitters per tick, every prey hears at least one signal on every tick. The "without signal" bucket never reaches the 10-sample threshold. per_prey_receiver_jsd returns 0.0 for every prey, making Pearson correlation undefined (constant zero array).

This was also true in the Rust version: at 384 population on 56x56 grid with signal_range=22.4, P(no signal at a given position) = 7.6e-30.

**This explains the single most persistent negative result in the project's history** (documented in the Rust repo's FINDINGS.md as open question #3). response_fit_corr has been zero in every run across all eras because the measurement is data-starved, not because the causal chain doesn't close.

Proposed fixes:
1. **Dominant symbol JSD**: compare actions when hearing symbol X vs symbol Y (both buckets always full)
2. **Signal strength threshold**: above-median vs below-median strength (splits by intensity, not presence)
3. **Per-symbol response profiles**: separate JSD per symbol, bypassing the binary hearing/not-hearing split

### Structural observations

**Action space is collapsed.** One action dimension (d1) gets ~86% of all counts across every symbol. Action entropy is 0.72 bits out of 2.0 max. All symbols produce nearly identical behavioral responses. The JSD differences are real but are small variations on a heavily biased baseline.

**Signal entropy converges to 98% of maximum** (1.756 / 1.792). All symbols used nearly equally by Gen 100k, up from 93.8% at Gen 5k. Functional specialization is happening *without* frequency bias - symbols trigger slightly different responses at equal usage rates.

**Signal network size decreases while MI increases.** avg_signal_hidden declines from 5.7 to 5.3 neurons while signal input MI grows from 0.166 to 0.226. Evolution compresses the signal processing network while increasing its information throughput.

**No cyclical patterns.** FFT analysis shows dominant frequencies are harmonics of the total run length, not real oscillations. Metric autocorrelation varies: signal_entropy is highly stable (AC(1)=0.94), receiver_fit_corr is noisy (AC(1)=0.004).

**JSD-fitness anti-correlation is a trend confound.** Raw r=-0.85 between jsd_pred and fitness, but detrended r=-0.014 (p=0.17). Within 1k-gen windows, mean correlation is -0.007. Context-dependent signaling is neither helpful nor harmful within epochs - the raw negative correlation is just two independent trends (JSD rising, fitness slightly declining) happening simultaneously.

### Comparison to Rust results

| Metric | Rust (Era 4, 384 pop, 56x56) | GPU (5k pop, 150x150) |
|--------|------------------------------|----------------------|
| Population | 384 | 5,000 |
| Grid | 56x56 | 150x150 |
| Generations | 37k-93k | 100k |
| avg_fitness | 620-736 | 984-1103 |
| mutual_info | 0.107-0.114 (food encoding) | 0.001 (near-zero) |
| jsd_pred | 0.018-0.215 | 0.033-0.066 |
| sender_fit_corr | 0.36-0.46 | 0.047-0.054 |
| signal_hidden | 25-31 | 5.3-5.7 |
| Top encoding | Food location | Other signals (meta) |
| response_fit_corr | 0.000 (broken) | 0.000 (broken) |

Key differences:
- **Food encoding vanished.** Rust runs encoded food location (MI 0.10-0.12). GPU run encodes other signals instead. At 13x population on a larger grid, the signal environment itself becomes the dominant input.
- **Much smaller signal networks.** Rust evolved to 25-31 signal hidden neurons. GPU stays at 5-6. Larger population may reduce individual signal processing demands.
- **Lower sender fitness correlation.** Rust: 0.36-0.46. GPU: ~0.05. With 5,000 agents, the correlation between any individual's signaling and fitness is diluted.

### Data files

- `output.csv` - 10,001 rows, 23 columns (fitness, MI, JSD, correlations, brain sizes)
- `trajectory.csv` - 10,001 rows, 47 columns (per-symbol context matrices, contrast)
- `input_mi.csv` - 10,001 rows, 37 columns (MI per input dimension)
- `run.log` - full runtime log

Analysis scripts in `runs/5k-100k-seed42/`: `analyze.py`, `analyze2.py`, `deep1.py`, `deep2.py`, `nullcheck.py`.
