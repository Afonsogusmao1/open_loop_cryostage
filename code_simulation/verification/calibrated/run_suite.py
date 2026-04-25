#!/usr/bin/env python3
from __future__ import annotations

"""Archive legacy calibrated results and run the refreshed rho(T) batch."""

import argparse
import contextlib
import io
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .common import (
    ACTIVE_TARGETS_C,
    ACTIVE_VELOCITY_STUDY_DIR,
    CALIBRATED_FIGURES_DIR,
    CALIBRATED_SIMULATIONS_DIR,
    WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    build_target_plans,
    prepare_calibrated_results_layout,
)
from .experiment_vs_simulation import generate_calibrated_experiment_vs_simulation_figures
from .front_speed_plots import generate_calibrated_front_speed_plots
from code_simulation.verification.run_pre_stabilized_calibrated_simulations import (
    DEFAULT_T_AFTER_FILL_S,
    run_target,
)


STEP_ORDER = ("archive", "simulate", "figures", "velocity")


@dataclass(frozen=True)
class SuiteRunConfig:
    targets_C: tuple[float, ...]
    overwrite: bool
    t_after_fill_s: float
    base_dir: Path
    figures_dir: Path
    simulations_dir: Path
    velocity_output_dir: Path
    dry_run: bool


class _TeeTextIO(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def _parse_targets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one target temperature")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target temperatures must be finite")
    return values


def _parse_steps(raw: str) -> tuple[str, ...]:
    parts = tuple(part.strip().lower() for part in str(raw).split(",") if part.strip())
    if not parts:
        raise ValueError("expected at least one step")
    if "all" in parts:
        return STEP_ORDER
    invalid = [part for part in parts if part not in STEP_ORDER]
    if invalid:
        raise ValueError(f"unknown steps: {invalid}; valid values are {STEP_ORDER} or 'all'")
    ordered = tuple(step for step in STEP_ORDER if step in parts)
    return ordered


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h {int(minutes):02d}m {sec:04.1f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m {sec:04.1f}s"
    return f"{sec:.1f}s"


def _log_path(simulations_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return simulations_dir / "logs" / f"calibrated_suite_{stamp}.log"


def _print_archive_plan(config: SuiteRunConfig) -> None:
    result = prepare_calibrated_results_layout(config.base_dir, dry_run=True)
    print(f"[archive] base_dir={config.base_dir}")
    if result.moves:
        for move in result.moves:
            print(f"  move {move.source.name} -> {move.destination.relative_to(config.base_dir)}")
    else:
        print(f"  no legacy items to move ({result.status})")
    print(f"  manifest: {result.manifest_path}")


def _run_archive(config: SuiteRunConfig) -> None:
    if config.dry_run:
        _print_archive_plan(config)
        return
    result = prepare_calibrated_results_layout(config.base_dir, dry_run=False)
    print(f"[archive] status={result.status}")
    if result.moves:
        for move in result.moves:
            print(f"  moved {move.source.name} -> {move.destination.relative_to(config.base_dir)}")
    else:
        print("  no legacy items moved")
    print(f"  manifest: {result.manifest_path}")


def _print_simulation_plan(config: SuiteRunConfig) -> None:
    print(
        "[simulate] "
        f"write_field_output=True enable_front_curve=False overwrite={config.overwrite} "
        f"t_after_fill_s={config.t_after_fill_s:.1f}"
    )
    for index, plan in enumerate(build_target_plans(config.targets_C, output_root=config.simulations_dir), start=1):
        print(
            f"  [case {index}/{len(config.targets_C)}] "
            f"T_plate={plan.target_C:.1f} C  Tamb={plan.ambient_C:.4f} C  out={plan.output_dir}"
        )


def _run_simulations(config: SuiteRunConfig) -> None:
    if config.dry_run:
        _print_simulation_plan(config)
        return

    config.simulations_dir.mkdir(parents=True, exist_ok=True)
    durations_s: list[float] = []
    plans = build_target_plans(config.targets_C, output_root=config.simulations_dir)

    for index, plan in enumerate(plans, start=1):
        if durations_s:
            avg_case_s = sum(durations_s) / len(durations_s)
            remaining_s = avg_case_s * (len(plans) - index + 1)
            print(f"\n[batch] upcoming case {index}/{len(plans)}  global ETA ~ {_format_duration(remaining_s)}")

        print(
            f"\n[case {index}/{len(plans)}] "
            f"T_plate={plan.target_C:.1f} C  Tamb={plan.ambient_C:.4f} C\n"
            f"  output: {plan.output_dir}"
        )
        t0 = time.perf_counter()
        run_target(
            target_C=plan.target_C,
            output_root=config.simulations_dir,
            t_after_fill_s=config.t_after_fill_s,
            overwrite=config.overwrite,
            write_field_output=True,
            enable_front_curve=False,
            show_progress=True,
        )
        elapsed_s = time.perf_counter() - t0
        durations_s.append(elapsed_s)
        avg_case_s = sum(durations_s) / len(durations_s)
        remaining_cases = len(plans) - index
        eta_s = avg_case_s * remaining_cases
        print(
            f"[case {index}/{len(plans)}] completed in {_format_duration(elapsed_s)}"
            + (f"  batch ETA ~ {_format_duration(eta_s)}" if remaining_cases > 0 else "  batch complete")
        )


def _run_figures(config: SuiteRunConfig) -> None:
    if config.dry_run:
        print(f"[figures] simulations={config.simulations_dir}")
        print(f"  output_dir={config.figures_dir}")
        print("  targets with experimental comparisons: -10, -15, -20 C")
        return
    generate_calibrated_experiment_vs_simulation_figures(
        simulations_dir=config.simulations_dir,
        output_dir=config.figures_dir,
    )


def _run_velocity(config: SuiteRunConfig) -> None:
    if config.dry_run:
        print(f"[velocity] simulations={config.simulations_dir}")
        print(f"  output_dir={config.velocity_output_dir}")
        print("  outputs: front_position/front_velocity/tc_segment_speeds/speed_summary + refreshed README")
        return
    generate_calibrated_front_speed_plots(
        simulations_dir=config.simulations_dir,
        output_dir=config.velocity_output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive legacy calibrated results, run the rho(T) suite, and regenerate active figures."
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated subset of steps: archive, simulate, figures, velocity, or all.",
    )
    parser.add_argument(
        "--targets-c",
        default=",".join(f"{value:g}" for value in ACTIVE_TARGETS_C),
        help="Comma-separated plate temperatures in C used by the simulate step.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=CALIBRATED_SIMULATIONS_DIR,
        help="Root directory containing calibrated simulation data and figures.",
    )
    parser.add_argument("--t-after-fill-s", type=float, default=DEFAULT_T_AFTER_FILL_S)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-case outputs when simulating.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved actions without modifying files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = _parse_steps(args.steps)
    config = SuiteRunConfig(
        targets_C=_parse_targets(args.targets_c),
        overwrite=bool(args.overwrite),
        t_after_fill_s=float(args.t_after_fill_s),
        base_dir=Path(args.base_dir).resolve(),
        figures_dir=(Path(args.base_dir).resolve() / CALIBRATED_FIGURES_DIR.name),
        simulations_dir=(Path(args.base_dir).resolve() / WITH_RHO_TEMPERATURE_DEPENDENT_DIR.name),
        velocity_output_dir=ACTIVE_VELOCITY_STUDY_DIR.resolve(),
        dry_run=bool(args.dry_run),
    )

    if config.dry_run:
        for step in steps:
            {"archive": _run_archive, "simulate": _run_simulations, "figures": _run_figures, "velocity": _run_velocity}[step](config)
        return

    log_path = _log_path(config.simulations_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = _TeeTextIO(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[suite] log={log_path}")
            for step in steps:
                print(f"\n=== step: {step} ===")
                {"archive": _run_archive, "simulate": _run_simulations, "figures": _run_figures, "velocity": _run_velocity}[step](config)


if __name__ == "__main__":
    main()
