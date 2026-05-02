#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


DEFAULT_DIAG_ROOT = Path(
    "code_simulation/results/active/bo_velocity_control/n8/diagnostico/"
    "earlydense_directspeed_w50_tol1_robust_0p007_0p013_5seed"
)
DEFAULT_COARSE_SUMMARY = DEFAULT_DIAG_ROOT / "campaign_summary.csv"
DEFAULT_COARSE_RUNS_ROOT = Path("code_simulation/results/active/bo_velocity_control/n8/coarse")
DEFAULT_OUTPUT_ROOT = Path("code_simulation/results/active/bo_velocity_control")
DEFAULT_FINE_DIAG_ROOT = DEFAULT_DIAG_ROOT / "fine_confirmations_best_per_target_full_process_article"


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _format_speed_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_theta_csv(path: Path) -> tuple[float, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row["temperature_C"]))
    if not values:
        raise ValueError(f"{path} did not contain any theta values")
    return tuple(float(value) for value in values)


def _theta_arg(theta: tuple[float, ...]) -> str:
    return ",".join(f"{value:.15g}" for value in theta)


def _fine_run_complete(run_dir: Path) -> bool:
    return (run_dir / "velocity_tracking_summary.csv").exists() and (run_dir / "plate_tracking_summary.csv").exists()


def _select_best_per_target(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("status", "")).strip() != "completed":
            continue
        target = _finite_float(row.get("target_front_speed_mm_s"))
        if math.isfinite(target):
            grouped.setdefault(target, []).append(row)
    selected: list[dict[str, str]] = []
    for target in sorted(grouped):
        candidates = grouped[target]
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    _finite_float(row.get("direct_speed_relative_error_pct"), math.inf),
                    _finite_float(row.get("tracking_rmse_mm"), math.inf),
                    int(float(row.get("seed", "999999"))),
                ),
            )
        )
    return selected


def _relative_error_pct(target: float, achieved: float) -> float:
    if not (math.isfinite(target) and target > 0.0 and math.isfinite(achieved)):
        return math.nan
    return float(100.0 * abs(achieved - target) / target)


def _load_fine_summary(run_dir: Path, *, target: float) -> dict[str, float]:
    tracking_path = run_dir / "velocity_tracking_summary.csv"
    plate_path = run_dir / "plate_tracking_summary.csv"
    segment_path = run_dir / "segment_speed_summary.csv"
    tracking = next(csv.DictReader(tracking_path.open(newline="")))
    plate = next(csv.DictReader(plate_path.open(newline="")))
    segment = (
        next(csv.DictReader(segment_path.open(newline="")))
        if segment_path.exists()
        else {}
    )
    achieved = _finite_float(tracking.get("actual_interval_speed_mm_s"))
    return {
        "fine_achieved_direct_speed_mm_s": achieved,
        "fine_direct_speed_relative_error_pct": _relative_error_pct(target, achieved),
        "fine_tracking_rmse_mm": _finite_float(tracking.get("tracking_rmse_mm")),
        "fine_tracking_max_abs_error_mm": _finite_float(tracking.get("tracking_max_abs_error_mm")),
        "fine_segment_speed_rmse_pct": _finite_float(segment.get("segment_speed_rmse_pct")),
        "fine_segment_speed_max_abs_error_pct": _finite_float(segment.get("segment_speed_max_abs_error_pct")),
        "fine_segment_speed_spread_mm_s": _finite_float(segment.get("segment_speed_spread_mm_s")),
        "fine_plate_rmse_error_C": _finite_float(plate.get("rmse_plate_error_C")),
        "fine_plate_mean_abs_error_C": _finite_float(plate.get("mean_abs_plate_error_C")),
    }


def _candidate_rows(
    *,
    coarse_summary: Path,
    coarse_runs_root: Path,
    output_root: Path,
    simulation_profile: str,
) -> list[dict[str, object]]:
    selected = _select_best_per_target(_read_rows(coarse_summary))
    candidates: list[dict[str, object]] = []
    for row in selected:
        target = _finite_float(row["target_front_speed_mm_s"])
        seed = int(float(row["seed"]))
        target_tag = _format_speed_tag(target)
        coarse_run_name = str(row["run_name"])
        coarse_run_dir = coarse_runs_root / coarse_run_name
        theta = _load_theta_csv(coarse_run_dir / "best_theta_profile.csv")
        fine_run_name = f"fine_confirm_v{target_tag}_n8_earlydense_directspeed_w50tol1_seed{seed}"
        fine_run_dir = output_root / "n8" / "fine" / fine_run_name
        candidates.append(
            {
                "target_front_speed_mm_s": target,
                "seed": seed,
                "coarse_run_name": coarse_run_name,
                "coarse_achieved_direct_speed_mm_s": _finite_float(row["achieved_direct_speed_mm_s"]),
                "coarse_direct_speed_relative_error_pct": _finite_float(row["direct_speed_relative_error_pct"]),
                "coarse_tracking_rmse_mm": _finite_float(row["tracking_rmse_mm"]),
                "theta_c": _theta_arg(theta),
                "simulation_profile": simulation_profile,
                "fine_run_name": fine_run_name,
                "fine_run_dir": str(fine_run_dir),
            }
        )
    return candidates


def _run_candidate(
    candidate: dict[str, object],
    *,
    output_root: Path,
    overwrite: bool,
    log_dir: Path,
) -> str:
    target = float(candidate["target_front_speed_mm_s"])
    fine_run_name = str(candidate["fine_run_name"])
    fine_run_dir = output_root / "n8" / "fine" / fine_run_name
    if _fine_run_complete(fine_run_dir) and not overwrite:
        return "skipped_completed"

    cmd = [
        sys.executable,
        "-m",
        "code_simulation.verification.run_velocity_control_evaluation",
        "--simulation-profile",
        str(candidate["simulation_profile"]),
        "--target-front-speed-mm-s",
        f"{target:.12g}",
        f"--theta-c={candidate['theta_c']}",
        "--num-knots",
        "8",
        "--knot-time-schedule",
        "early_dense",
        "--run-name",
        fine_run_name,
        "--output-root",
        str(output_root),
    ]
    if overwrite:
        cmd.append("--overwrite")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{fine_run_name}.log"
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    return "completed" if completed.returncode == 0 else f"failed_returncode_{completed.returncode}"


def _write_summary(
    *,
    candidates: list[dict[str, object]],
    output_root: Path,
    summary_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        target = float(candidate["target_front_speed_mm_s"])
        fine_run_dir = output_root / "n8" / "fine" / str(candidate["fine_run_name"])
        status = "completed" if _fine_run_complete(fine_run_dir) else "missing"
        row: dict[str, object] = {
            key: candidate[key]
            for key in (
                "target_front_speed_mm_s",
                "seed",
                "coarse_run_name",
                "coarse_achieved_direct_speed_mm_s",
                "coarse_direct_speed_relative_error_pct",
                "coarse_tracking_rmse_mm",
                "simulation_profile",
                "fine_run_name",
                "fine_run_dir",
            )
        }
        row["fine_status"] = status
        if status == "completed":
            row.update(_load_fine_summary(fine_run_dir, target=target))
        else:
            row.update(
                {
                    "fine_achieved_direct_speed_mm_s": math.nan,
                    "fine_direct_speed_relative_error_pct": math.nan,
                    "fine_tracking_rmse_mm": math.nan,
                    "fine_tracking_max_abs_error_mm": math.nan,
                    "fine_segment_speed_rmse_pct": math.nan,
                    "fine_segment_speed_max_abs_error_pct": math.nan,
                    "fine_segment_speed_spread_mm_s": math.nan,
                    "fine_plate_rmse_error_C": math.nan,
                    "fine_plate_mean_abs_error_C": math.nan,
                }
            )
        rows.append(row)
    _write_rows(
        summary_path,
        rows,
        fieldnames=(
            "target_front_speed_mm_s",
            "seed",
            "coarse_run_name",
            "coarse_achieved_direct_speed_mm_s",
            "coarse_direct_speed_relative_error_pct",
            "coarse_tracking_rmse_mm",
            "simulation_profile",
            "fine_run_name",
            "fine_run_dir",
            "fine_status",
            "fine_achieved_direct_speed_mm_s",
            "fine_direct_speed_relative_error_pct",
            "fine_tracking_rmse_mm",
            "fine_tracking_max_abs_error_mm",
            "fine_segment_speed_rmse_pct",
            "fine_segment_speed_max_abs_error_pct",
            "fine_segment_speed_spread_mm_s",
            "fine_plate_rmse_error_C",
            "fine_plate_mean_abs_error_C",
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representative fine confirmations for the robust n8 BO campaign.")
    parser.add_argument("--coarse-summary", type=Path, default=DEFAULT_COARSE_SUMMARY)
    parser.add_argument("--coarse-runs-root", type=Path, default=DEFAULT_COARSE_RUNS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_FINE_DIAG_ROOT)
    parser.add_argument("--simulation-profile", default="full_process_article")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = _candidate_rows(
        coarse_summary=args.coarse_summary,
        coarse_runs_root=args.coarse_runs_root,
        output_root=args.output_root,
        simulation_profile=str(args.simulation_profile),
    )
    args.diagnostic_root.mkdir(parents=True, exist_ok=True)
    candidate_path = args.diagnostic_root / "selected_fine_confirmation_candidates.csv"
    _write_rows(
        candidate_path,
        candidates,
        fieldnames=(
            "target_front_speed_mm_s",
            "seed",
            "coarse_run_name",
            "coarse_achieved_direct_speed_mm_s",
            "coarse_direct_speed_relative_error_pct",
            "coarse_tracking_rmse_mm",
            "theta_c",
            "simulation_profile",
            "fine_run_name",
            "fine_run_dir",
        ),
    )
    print(f"Selected candidates: {len(candidates)}")
    print(f"Candidate CSV: {candidate_path.resolve()}")
    if args.dry_run:
        return

    log_dir = args.diagnostic_root / "logs"
    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"[{idx}/{len(candidates)}] {candidate['fine_run_name']} "
            f"target={float(candidate['target_front_speed_mm_s']):.3f} seed={candidate['seed']}",
            flush=True,
        )
        status = _run_candidate(candidate, output_root=args.output_root, overwrite=bool(args.overwrite), log_dir=log_dir)
        print(f"  status={status}", flush=True)
        _write_summary(
            candidates=candidates,
            output_root=args.output_root,
            summary_path=args.diagnostic_root / "fine_confirmation_summary.csv",
        )

    _write_summary(
        candidates=candidates,
        output_root=args.output_root,
        summary_path=args.diagnostic_root / "fine_confirmation_summary.csv",
    )
    print(f"Summary CSV: {(args.diagnostic_root / 'fine_confirmation_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
