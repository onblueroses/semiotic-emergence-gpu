"""Simulation parameters mirroring Rust SimParams."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimParams:
    pop_size: int = 384
    grid_size: int = 56
    num_zones: int = 3
    food_count: int = 100
    ticks_per_eval: int = 500
    signal_range: float = 0.0  # sentinel, computed in from_cli
    zone_radius: float = 8.0
    zone_speed: float = 0.5
    reproduction_radius: float = 0.0  # computed
    fallback_radius: float = 0.0  # computed
    mi_bins: tuple[float, float, float] = (0.0, 0.0, 0.0)  # computed
    elite_count: int = 0  # computed
    tournament_size: int = 3
    mutation_sigma: float = 0.1
    base_drain: float = 0.0008
    neuron_cost: float = 0.0
    signal_cost: float = 0.002
    zone_drain_rate: float = 0.02
    no_signals: bool = False
    patch_ratio: float = 0.5
    kin_bonus: float = 0.1
    metrics_interval: int = 1
    fast_fail_tick: int = 0
    signal_ticks: int = 4
    freeze_zones: int = 0
    poison_ratio: float = 0.0
    checkpoint_interval: int = 0
    resume: str | None = None

    @staticmethod
    def from_cli(argv: list[str] | None = None) -> SimParams:
        p = argparse.ArgumentParser()
        p.add_argument("seed", type=int, nargs="?", default=0)
        p.add_argument("generations", type=int, nargs="?", default=200)
        p.add_argument("--pop", type=int, default=384)
        p.add_argument("--grid", type=int, default=56)
        p.add_argument("--pred", type=int, default=3)
        p.add_argument("--food", type=int, default=100)
        p.add_argument("--ticks", type=int, default=500)
        p.add_argument("--zone-radius", type=float, default=8.0)
        p.add_argument("--zone-speed", type=float, default=0.5)
        p.add_argument("--zone-drain", type=float, default=0.02)
        p.add_argument("--signal-cost", type=float, default=0.002)
        p.add_argument("--signal-range", type=float, default=0.0)
        p.add_argument("--signal-ticks", type=int, default=4)
        p.add_argument("--patch-ratio", type=float, default=0.5)
        p.add_argument("--kin-bonus", type=float, default=0.1)
        p.add_argument("--metrics-interval", type=int, default=1)
        p.add_argument("--fast-fail", type=int, default=0)
        p.add_argument("--zone-coverage", type=float, default=None)
        p.add_argument("--no-signals", action="store_true")
        p.add_argument("--freeze-zones", type=int, default=0)
        p.add_argument("--poison-ratio", type=float, default=0.0)
        p.add_argument("--checkpoint-interval", type=int, default=0)
        p.add_argument("--resume", type=str, default=None)
        args = p.parse_args(argv)

        scale = args.grid / 20.0

        num_zones = args.pred
        if args.zone_coverage is not None:
            grid_area = args.grid ** 2
            zone_area = math.pi * args.zone_radius ** 2
            num_zones = math.ceil(args.zone_coverage * grid_area / zone_area)

        signal_range = args.signal_range if args.signal_range > 0 else 8.0 * scale
        reproduction_radius = 6.0 * scale
        fallback_radius = 10.0 * scale
        mi_bins = (args.zone_radius, signal_range, signal_range * 1.375)
        elite_count = max(args.pop // 6, 2)

        return SimParams(
            pop_size=args.pop,
            grid_size=args.grid,
            num_zones=num_zones,
            food_count=args.food,
            ticks_per_eval=args.ticks,
            signal_range=signal_range,
            zone_radius=args.zone_radius,
            zone_speed=args.zone_speed,
            reproduction_radius=reproduction_radius,
            fallback_radius=fallback_radius,
            mi_bins=mi_bins,
            elite_count=elite_count,
            signal_cost=args.signal_cost,
            zone_drain_rate=args.zone_drain,
            no_signals=args.no_signals,
            patch_ratio=args.patch_ratio,
            kin_bonus=args.kin_bonus,
            metrics_interval=max(args.metrics_interval, 1),
            fast_fail_tick=args.fast_fail,
            signal_ticks=args.signal_ticks,
            freeze_zones=args.freeze_zones,
            poison_ratio=args.poison_ratio,
            checkpoint_interval=args.checkpoint_interval,
            resume=args.resume,
        )
