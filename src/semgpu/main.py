"""Entry point for semiotic-emergence-gpu simulation."""

from __future__ import annotations

import csv
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from semgpu.brain import DEFAULT_BASE_HIDDEN, DEFAULT_SIGNAL_HIDDEN, MAX_GENOME_LEN
from semgpu.config import SimParams
from semgpu.evolution import compute_kin_bonus, evolve_generation
from semgpu.metrics import (
    compute_iconicity,
    compute_input_mi,
    compute_mutual_info,
    compute_per_symbol_jsd,
    compute_receiver_jsd,
    compute_signal_entropy,
    compute_silence_onset_metrics,
    inter_symbol_jsd,
    normalize_matrix,
    pearson,
    per_prey_receiver_jsd,
    rolling_fluctuation_ratio,
    trajectory_jsd,
)
from semgpu.world import evaluate_generation, init_world

# CSV column headers matching Rust output.csv (27 columns)
OUTPUT_COLUMNS = [
    "generation", "avg_fitness", "max_fitness", "signals_emitted",
    "iconicity", "mutual_info", "jsd_no_pred", "jsd_pred",
    "silence_corr", "sender_fit_corr", "traj_fluct_ratio",
    "receiver_fit_corr", "response_fit_corr",
    "silence_onset_jsd", "silence_move_delta",
    "avg_base_hidden", "min_base_hidden", "max_base_hidden",
    "avg_signal_hidden", "min_signal_hidden", "max_signal_hidden",
    "zone_deaths", "signal_entropy",
    "freeze_zone_deaths", "food_mi", "poison_eaten", "energy_delta_mi",
]

# trajectory.csv: 47 columns
TRAJ_COLUMNS = (
    ["generation"]
    + [f"s{s}d{d}" for s in range(6) for d in range(4)]  # 24 matrix entries
    + [f"jsd_sym{s}" for s in range(6)]  # 6 per-symbol JSD
    + ["trajectory_jsd"]
    + [f"contrast_{i}{j}" for i in range(6) for j in range(i + 1, 6)]  # 15 pairs
)

FLUCT_WINDOW = 10


def _buffer_sizes(pop_size: int, ticks: int = 500) -> tuple[int, int, int]:
    """Compute signal, event, and death buffer sizes.

    Signal ring buffer: holds active signals (max lifetime = signal_ticks=4).
    Worst case = pop * 4 if every prey emits every tick. Use pop * 6 for headroom.
    Event ring buffer: for input MI statistics. pop * 20 is generous.
    Death ring buffer: worst case all prey die in one generation.
    """
    max_signals = max(50_000, pop_size * 6)
    max_events = max(100_000, pop_size * 20)
    max_deaths = max(1_000, pop_size)
    return max_signals, max_events, max_deaths


# input_mi.csv: 40 columns (1 + 39 input dims)
INPUT_MI_COLUMNS = ["generation"] + [
    f"mi_{name}" for name in [
        "zone_damage", "energy_delta", "freeze_pressure",
        "food_dx", "food_dy", "food_dist",
        "ally_dx", "ally_dy", "ally_dist",
        "sig0_str", "sig0_dx", "sig0_dy",
        "sig1_str", "sig1_dx", "sig1_dy",
        "sig2_str", "sig2_dx", "sig2_dy",
        "sig3_str", "sig3_dx", "sig3_dy",
        "sig4_str", "sig4_dx", "sig4_dy",
        "sig5_str", "sig5_dx", "sig5_dy",
        "mem0", "mem1", "mem2", "mem3", "mem4", "mem5", "mem6", "mem7",
        "energy",
        "death_nearby", "death_dx", "death_dy",
    ]
]


def run_seed(
    seed: int, generations: int, params: SimParams,
) -> dict:
    """Run a single seed for the given number of generations.

    Returns dict with final_matrix (6x4 int), avg_fitness, max_fitness, mutual_info.
    """
    key = jax.random.key(seed)

    N = params.pop_size
    max_signals, max_events, max_deaths = _buffer_sizes(N)
    k1, k2, k3, key = jax.random.split(key, 4)

    # Initialize population
    prey_x = jax.random.randint(k1, (N,), 0, params.grid_size)
    prey_y = jax.random.randint(k2, (N,), 0, params.grid_size)
    weights = jax.random.uniform(k3, (N, MAX_GENOME_LEN), minval=-1.0, maxval=1.0)
    base_hidden = jnp.full(N, DEFAULT_BASE_HIDDEN, dtype=jnp.int32)
    signal_hidden = jnp.full(N, DEFAULT_SIGNAL_HIDDEN, dtype=jnp.int32)
    parent_indices = jnp.full((N, 2), -1, dtype=jnp.int32)
    grandparent_indices = jnp.full((N, 4), -1, dtype=jnp.int32)

    # MI bin edges: [zone_radius, signal_range, signal_range * 1.375]
    mi_bins = (params.zone_radius, params.signal_range, params.signal_range * 1.375)

    # Trajectory state
    prev_norm_matrix = None
    traj_jsd_history: list[float] = []

    # Track last-generation values for return
    last_mi_counts = np.zeros((6, 4), dtype=np.int32)
    last_avg_fit = 0.0
    last_max_fit = 0.0
    last_mutual_info = 0.0

    with (
        open("output.csv", "w", newline="") as f,
        open("trajectory.csv", "w", newline="") as tf,
        open("input_mi.csv", "w", newline="") as imf,
    ):
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        traj_writer = csv.writer(tf)
        traj_writer.writerow(TRAJ_COLUMNS)
        imi_writer = csv.writer(imf)
        imi_writer.writerow(INPUT_MI_COLUMNS)

        print(
            f"Config: pop={params.pop_size} grid={params.grid_size} "
            f"zones={params.num_zones} radius={params.zone_radius:.1f} "
            f"speed={params.zone_speed:.1f} drain={params.zone_drain_rate:.3f} "
            f"food={params.food_count} ticks={params.ticks_per_eval} "
            f"patches={params.patch_ratio*100:.0f}% kin_bonus={params.kin_bonus:.2f} "
            f"sig_cost={params.signal_cost:.4f} sig_range={params.signal_range:.1f}"
        )
        print(
            f"Backend: {jax.default_backend()} | "
            f"Buffers: signals={max_signals} events={max_events} deaths={max_deaths}"
        )

        for gen in range(generations):
            gen_start = time.time()
            k_eval, k_evo, key = jax.random.split(key, 3)

            # Create world and evaluate
            ws = init_world(
                prey_x=prey_x,
                prey_y=prey_y,
                weights=weights,
                base_hidden=base_hidden,
                signal_hidden=signal_hidden,
                num_zones=params.num_zones,
                food_count=params.food_count,
                grid_size=params.grid_size,
                zone_radius=params.zone_radius,
                zone_speed=params.zone_speed,
                patch_ratio=params.patch_ratio,
                max_signals=max_signals,
                ticks_per_eval=params.ticks_per_eval,
                max_events=max_events,
                max_deaths=max_deaths,
                key=k_eval,
                num_freeze=params.freeze_zones,
                poison_ratio=params.poison_ratio,
            )

            result = evaluate_generation(
                ws,
                grid_size=params.grid_size,
                signal_range=params.signal_range,
                base_drain=params.base_drain,
                signal_cost=params.signal_cost,
                zone_drain_rate=params.zone_drain_rate,
                patch_ratio=params.patch_ratio,
                food_count=params.food_count,
                signal_ticks=params.signal_ticks,
                no_signals=params.no_signals,
                max_signals=max_signals,
                ticks_per_eval=params.ticks_per_eval,
                mi_bins=mi_bins,
                zone_radius_scalar=params.zone_radius,
                max_events=max_events,
                poison_ratio=params.poison_ratio,
            )

            fitness = result.fitness

            # Kin bonus
            if params.kin_bonus > 0:
                fitness = compute_kin_bonus(
                    fitness, parent_indices, grandparent_indices, params.kin_bonus
                )

            # Rank-based selection: replace fitness with rank
            rank_order = jnp.argsort(jnp.argsort(fitness))
            ranked_fitness = rank_order.astype(jnp.float32) + 1.0

            # --- Compute metrics only on metrics_interval generations ---
            collect_metrics = (gen % params.metrics_interval == 0) or (gen == generations - 1)

            if collect_metrics:
                fs = result.final_state
                mi_counts_np = np.asarray(fs.mi_counts)
                symbol_counts_np = np.asarray(fs.symbol_counts)
                recv_counts_np = np.asarray(fs.recv_counts)

                avg_fit = float(jnp.mean(result.fitness))
                max_fit = float(jnp.max(result.fitness))
                total_sig = int(result.total_signals)
                zd = int(result.zone_deaths)
                fzd = int(fs.freeze_zone_deaths)
                pe = int(fs.poison_eaten)

                # Sender metrics
                iconicity = compute_iconicity(
                    int(fs.iconicity_in_zone),
                    int(symbol_counts_np.sum()),
                    int(fs.m_ticks_in_zone),
                    int(fs.m_total_prey_ticks),
                )
                mutual_info = compute_mutual_info(mi_counts_np)
                sig_entropy = compute_signal_entropy(symbol_counts_np)

                last_mi_counts = mi_counts_np
                last_avg_fit = avg_fit
                last_max_fit = max_fit
                last_mutual_info = mutual_info

                # Receiver JSD
                jsd_no_pred, jsd_pred = compute_receiver_jsd(recv_counts_np)

                # Silence correlation
                spt = np.asarray(fs.per_tick_signals)
                apt = np.asarray(fs.per_tick_alive)
                mzd = np.asarray(fs.per_tick_min_zdist)
                norm_sig_rate = np.where(apt > 0, spt / apt, 0.0)
                silence_corr = pearson(norm_sig_rate, mzd)

                # Sender fit correlation
                fitness_np = np.asarray(result.fitness)
                ticks_alive_np = np.asarray(result.ticks_alive)
                signals_sent_np = np.asarray(fs.prey_signals_sent)
                signal_rate_per_prey = np.where(
                    ticks_alive_np > 0,
                    signals_sent_np.astype(np.float64) / ticks_alive_np,
                    0.0,
                )
                sender_fit_corr = pearson(signal_rate_per_prey, fitness_np)

                # Trajectory JSD and fluctuation ratio
                curr_norm = normalize_matrix(mi_counts_np)
                if prev_norm_matrix is not None and curr_norm is not None:
                    traj_jsd_val = trajectory_jsd(prev_norm_matrix, curr_norm)
                else:
                    traj_jsd_val = 0.0
                traj_jsd_history.append(traj_jsd_val)
                traj_fluct = rolling_fluctuation_ratio(traj_jsd_history, FLUCT_WINDOW)
                if curr_norm is not None:
                    prev_norm_matrix = curr_norm

                # Three-way coupling
                recv_with_np = np.asarray(fs.prey_recv_with)
                recv_without_np = np.asarray(fs.prey_recv_without)
                total_w = recv_with_np.reshape(N, -1).sum(axis=1)
                total_wo = recv_without_np.reshape(N, -1).sum(axis=1)
                total_all = total_w + total_wo
                reception_rates = np.where(total_all > 0, total_w / total_all, 0.0)
                receiver_fit_corr = pearson(reception_rates, fitness_np)

                per_prey_jsd = np.array([
                    per_prey_receiver_jsd(recv_with_np[i], recv_without_np[i])
                    for i in range(N)
                ])
                response_fit_corr = pearson(per_prey_jsd, fitness_np)

                # Silence onset
                onset_np = np.asarray(fs.onset_actions)
                present_np = np.asarray(fs.present_actions)
                silence_onset_jsd, silence_move_delta = compute_silence_onset_metrics(
                    onset_np, present_np,
                )

                # Brain size stats
                avg_bh = float(jnp.mean(base_hidden))
                min_bh = int(jnp.min(base_hidden))
                max_bh = int(jnp.max(base_hidden))
                avg_sh = float(jnp.mean(signal_hidden))
                min_sh = int(jnp.min(signal_hidden))
                max_sh = int(jnp.max(signal_hidden))

                # Compute input MI (needed for food_mi and energy_delta_mi in output.csv)
                evt_count = int(fs.evt_count)
                if evt_count > 0:
                    evt_sym_np = np.asarray(fs.evt_symbol)[:evt_count]
                    evt_inp_np = np.asarray(fs.evt_inputs)[:evt_count]
                    input_mi_vals = compute_input_mi(evt_sym_np, evt_inp_np)
                else:
                    input_mi_vals = np.zeros(39)
                food_mi = float(input_mi_vals[5])       # I(Signal; food_dist) at dim 5
                energy_delta_mi = float(input_mi_vals[1])  # I(Signal; energy_delta) at dim 1

                # Write output.csv row (27 columns)
                row = [
                    gen, f"{avg_fit:.4f}", f"{max_fit:.4f}", total_sig,
                    f"{iconicity:.6f}", f"{mutual_info:.6f}",
                    f"{jsd_no_pred:.6f}", f"{jsd_pred:.6f}",
                    f"{silence_corr:.6f}", f"{sender_fit_corr:.6f}",
                    f"{traj_fluct:.6f}",
                    f"{receiver_fit_corr:.6f}", f"{response_fit_corr:.6f}",
                    f"{silence_onset_jsd:.6f}", f"{silence_move_delta:.6f}",
                    f"{avg_bh:.1f}", min_bh, max_bh,
                    f"{avg_sh:.1f}", min_sh, max_sh,
                    zd, f"{sig_entropy:.6f}",
                    fzd, f"{food_mi:.6f}", pe, f"{energy_delta_mi:.6f}",
                ]
                writer.writerow(row)

                # Write trajectory.csv row
                n_pairs = 6 * 5 // 2  # 15
                if curr_norm is not None:
                    per_sym_jsd = compute_per_symbol_jsd(recv_counts_np)
                    contrast = inter_symbol_jsd(curr_norm)
                else:
                    per_sym_jsd = np.zeros(6)
                    contrast = [0.0] * n_pairs

                traj_row = [gen]
                for s in range(6):
                    for d in range(4):
                        traj_row.append(int(mi_counts_np[s, d]))
                for s in range(6):
                    traj_row.append(f"{per_sym_jsd[s]:.6f}")
                traj_row.append(f"{traj_jsd_val:.6f}")
                for v in contrast:
                    traj_row.append(f"{v:.6f}")
                traj_writer.writerow(traj_row)

                # Write input_mi.csv row
                imi_row = [gen] + [f"{v:.6f}" for v in input_mi_vals]
                imi_writer.writerow(imi_row)

            # Progress printing (on metrics gens, or every 100 gens as heartbeat)
            print_gen = collect_metrics and (gen % 10 == 0 or gen == generations - 1)
            heartbeat = (not collect_metrics) and (gen % 100 == 0)
            if print_gen:
                elapsed = time.time() - gen_start
                print(
                    f"Gen {gen:>5}: avg={avg_fit:.1f} max={max_fit:.1f} "
                    f"sig={total_sig} MI={mutual_info:.4f} "
                    f"ico={iconicity:.3f} zd={zd} ent={sig_entropy:.3f} "
                    f"({elapsed:.2f}s)"
                )
            elif heartbeat:
                elapsed = time.time() - gen_start
                print(f"Gen {gen:>5}: ({elapsed:.2f}s)")

            # Evolve
            (prey_x, prey_y, weights, base_hidden, signal_hidden,
             parent_indices, grandparent_indices) = evolve_generation(
                prey_x, prey_y, weights, base_hidden, signal_hidden,
                ranked_fitness, parent_indices, grandparent_indices,
                elite_count=params.elite_count,
                tournament_size=params.tournament_size,
                sigma=params.mutation_sigma,
                grid_size=params.grid_size,
                reproduction_radius=params.reproduction_radius,
                fallback_radius=params.fallback_radius,
                key=k_evo,
            )

    print(f"Done. {generations} generations written to output.csv + trajectory.csv + input_mi.csv")

    return {
        "final_matrix": last_mi_counts,
        "avg_fitness": last_avg_fit,
        "max_fitness": last_max_fit,
        "mutual_info": last_mutual_info,
    }


def run_batch(n_seeds: int, generations: int, params: SimParams):
    """Run multiple seeds and compute cross-population divergence."""
    from semgpu.metrics import cross_population_divergence

    print(f"Batch mode: {n_seeds} seeds x {generations} generations")
    results = []
    for seed in range(n_seeds):
        print(f"--- seed {seed} ---")
        results.append(run_seed(seed, generations, params))

    norm_matrices = [normalize_matrix(r["final_matrix"]) for r in results]

    print("\nDivergence matrix (permutation-aware JSD):")
    header = "     " + "".join(f"  s{j:<4}" for j in range(n_seeds))
    print(header)

    with open("divergence.csv", "w", newline="") as df:
        dw = csv.writer(df)
        dw.writerow(["seed"] + [f"s{j}" for j in range(n_seeds)])
        for i in range(n_seeds):
            row_vals = []
            for j in range(n_seeds):
                if norm_matrices[i] is not None and norm_matrices[j] is not None:
                    div = cross_population_divergence(norm_matrices[i], norm_matrices[j])
                else:
                    div = float("nan")
                row_vals.append(div)
            print(f"s{i:<4}" + "".join(f"  {v:.4f}" for v in row_vals))
            dw.writerow([i] + [f"{v:.4f}" for v in row_vals])

    print("\nPer-seed summary:")
    for i, r in enumerate(results):
        print(f"  seed {i}: avg={r['avg_fitness']:.1f} max={r['max_fitness']:.1f} MI={r['mutual_info']:.3f}")
    print("Divergence matrix saved to divergence.csv")


def main():
    # Parse --batch before SimParams to extract args
    argv = sys.argv[1:]
    batch_args = None
    if "--batch" in argv:
        idx = argv.index("--batch")
        batch_args = (int(argv[idx + 1]), int(argv[idx + 2]))
        # Remove --batch N GENS so SimParams doesn't choke on positionals
        argv = argv[:idx] + argv[idx + 3:]

    params = SimParams.from_cli(argv)

    if batch_args is not None:
        n_seeds, generations = batch_args
        run_batch(n_seeds, generations, params)
    else:
        seed = int(sys.argv[1])
        generations = int(sys.argv[2])
        run_seed(seed, generations, params)


if __name__ == "__main__":
    main()
