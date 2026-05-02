#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.plotting import configure_matplotlib


DEFAULT_OUTPUT_ROOT = Path("code_simulation/results/active/bo_velocity_control")
DEFAULT_DIAG_ROOT = Path(
    "code_simulation/results/active/bo_velocity_control/n8/diagnostico/"
    "earlydense_articlefinal_fronttrack_w50_directspeed_w50_seg5_tol2_0p007_0p013_5seed"
)
DEFAULT_TARGETS = (0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013)
DEFAULT_SEEDS = (17, 29, 41, 53, 67)
DEFAULT_RUN_LABEL = "articlefinal"
TARGET_TOL = 5.0e-7
VELOCITY_SMOOTHING_WINDOWS_S = (60.0, 120.0, 180.0)
ACCEPT_DIRECT_ERROR_PCT = 2.0
ACCEPT_TRACKING_RMSE_MM = 0.15
ACCEPT_TRACKING_MAX_ABS_MM = 0.35
ACCEPT_SEGMENT_RMSE_PCT = 5.0
ACCEPT_SEGMENT_MAX_ABS_ERROR_PCT = 10.0

COLOR_BY_SEED = {
    17: "#1f77b4",
    29: "#2ca02c",
    41: "#d62728",
    53: "#9467bd",
    67: "#8c564b",
}

THETA0_BY_TARGET = {
    # Target-conditioned warm starts fixed before the article-final campaign.
    # These are not model parameters; they are initial open-loop reference
    # trajectories for the optimizer, selected from criteria-passing diagnostic
    # candidates and then kept fixed for all five seeds.
    0.007: (
        -1.40741397716821,
        -2.49399434309166,
        -3.89008915385551,
        -5.82656696741871,
        -8.64006013288259,
        -11.5998578481149,
        -15.0344988847249,
        -18.9711625690246,
    ),
    0.008: (
        -2.32476523853316,
        -3.96604570972673,
        -5.72358269644661,
        -8.13592518053237,
        -11.9284823826724,
        -15.0296823894349,
        -18.0904812245364,
        -20.8443692315924,
    ),
    0.009: (
        -3.5589210015801,
        -5.05100963981819,
        -7.36570415315466,
        -10.2645634596104,
        -15.038758176991,
        -15.893276523366,
        -21.0,
        -21.0,
    ),
    0.010: (
        -4.50159920293404,
        -6.05526778641215,
        -8.78878797509,
        -12.2279386728594,
        -18.3044985696,
        -18.567076686233,
        -21.0,
        -21.0,
    ),
    0.011: (
        -5.0834322092348,
        -6.99454712194228,
        -10.1385540668978,
        -14.8370376297733,
        -20.7030151519016,
        -21.0,
        -21.0,
        -21.0,
    ),
    0.012: (
        -0.000131569913183819,
        -0.000133457284196906,
        -0.000225185453746293,
        -0.00796150806842871,
        -0.281481817182868,
        -9.95188508553596,
        -21.0,
        -21.0,
    ),
    0.013: (
        -0.000249463934604382,
        -0.000249463934604382,
        -0.0176397639820237,
        -0.415773224340642,
        -9.79986887890215,
        -21.0,
        -21.0,
        -21.0,
    ),
}


@dataclass(frozen=True)
class CampaignJob:
    target_front_speed_mm_s: float
    seed: int
    run_name: str
    run_dir: Path
    argv: tuple[str, ...]


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer")
    return values


def _target_key(target: float) -> float:
    for known in THETA0_BY_TARGET:
        if abs(float(target) - known) <= TARGET_TOL:
            return known
    raise ValueError(f"no target-specific theta0 is defined for target {target:.12g}")


def _theta0_arg(target: float, override_theta0: tuple[float, ...] | None = None) -> str:
    theta0 = override_theta0 if override_theta0 is not None else THETA0_BY_TARGET[_target_key(target)]
    return ",".join(f"{value:.15g}" for value in theta0)


def _format_speed_tag(value: float) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def _read_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_complete(run_dir: Path) -> bool:
    required = (
        run_dir / "effective_config.toml",
        run_dir / "bo_history.csv",
        run_dir / "best_tracking_summary.csv",
        run_dir / "best_segment_speed_summary.csv",
        run_dir / "best_segment_speeds.csv",
        run_dir / "best_theta_profile.csv",
    )
    return all(path.exists() for path in required)


def _best_front_path(run_dir: Path) -> Path | None:
    best_dir = run_dir / "best"
    matches = sorted(best_dir.glob("*_front.csv"))
    return matches[0] if matches else None


def _read_front_series(path: Path) -> dict[str, np.ndarray]:
    time_s: list[float] = []
    z_front_mm: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            time_s.append(_finite_float(row.get("time_since_fill_s", row.get("time_s"))))
            z_front_mm.append(_finite_float(row.get("z_front_mm")))
    return {
        "time_s": np.asarray(time_s, dtype=np.float64),
        "z_front_mm": np.asarray(z_front_mm, dtype=np.float64),
    }


def _first_time_at_or_above(time_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    previous_t = math.nan
    previous_v = math.nan
    finite = np.isfinite(time_s) & np.isfinite(values)
    for t_i, value_i in zip(time_s[finite], values[finite], strict=False):
        t_i = float(t_i)
        value_i = float(value_i)
        if value_i >= threshold:
            if math.isfinite(previous_t) and math.isfinite(previous_v) and value_i > previous_v:
                alpha = (float(threshold) - previous_v) / (value_i - previous_v)
                return float(previous_t + alpha * (t_i - previous_t))
            return t_i
        previous_t = t_i
        previous_v = value_i
    return math.nan


def _segment_speeds_from_front_path(
    front_path: Path | None,
    *,
    z_min_mm: float = 2.5,
    z_max_mm: float = 11.5,
    num_segments: int = 5,
) -> str:
    if front_path is None or not front_path.exists() or int(num_segments) <= 0:
        return ""
    series = _read_front_series(front_path)
    time_s = series["time_s"]
    z_front_mm = series["z_front_mm"]
    boundaries = np.linspace(float(z_min_mm), float(z_max_mm), int(num_segments) + 1, dtype=np.float64)
    crossing_times = [_first_time_at_or_above(time_s, z_front_mm, float(z_mm)) for z_mm in boundaries]
    speeds: list[str] = []
    for z0_mm, z1_mm, t0_s, t1_s in zip(
        boundaries[:-1],
        boundaries[1:],
        crossing_times[:-1],
        crossing_times[1:],
        strict=True,
    ):
        if math.isfinite(t0_s) and math.isfinite(t1_s) and t1_s > t0_s:
            speeds.append(f"{float((z1_mm - z0_mm) / (t1_s - t0_s)):.15g}")
        else:
            speeds.append("nan")
    return ";".join(speeds)


def _local_linear_velocity_mm_s(time_s: np.ndarray, z_front_mm: np.ndarray, *, window_s: float) -> np.ndarray:
    velocity = np.full_like(time_s, math.nan, dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(z_front_mm)
    if np.count_nonzero(finite) < 3:
        return velocity

    valid_indices = np.flatnonzero(finite)
    valid_time_s = time_s[valid_indices]
    valid_z_front_mm = z_front_mm[valid_indices]
    half_window_s = 0.5 * float(window_s)
    for source_idx, t_i in zip(valid_indices, valid_time_s, strict=True):
        left = np.searchsorted(valid_time_s, t_i - half_window_s, side="left")
        right = np.searchsorted(valid_time_s, t_i + half_window_s, side="right")
        if right - left < 3:
            continue
        window_time_s = valid_time_s[left:right]
        if float(window_time_s[-1] - window_time_s[0]) <= 1.0e-12:
            continue
        slope_mm_s, _ = np.polyfit(window_time_s, valid_z_front_mm[left:right], deg=1)
        velocity[source_idx] = float(slope_mm_s)
    return velocity


def _build_job(
    *,
    target: float,
    seed: int,
    output_root: Path,
    init_points: int,
    n_iter: int,
    local_refinement_points: int,
    tracking_weight: float,
    direct_speed_weight: float,
    direct_speed_tolerance_pct: float,
    segment_speed_weight: float,
    segment_speed_tolerance_pct: float,
    segment_speed_num_segments: int,
    simulation_profile: str,
    theta0_override: tuple[float, ...] | None,
    run_label: str,
) -> CampaignJob:
    target_tag = _format_speed_tag(target)
    label_part = f"_{run_label}" if run_label else ""
    run_name = (
        f"bo_v{target_tag}_n8_earlydense_fronttrack_w{tracking_weight:g}"
        f"_directspeed_w{direct_speed_weight:g}_seg{segment_speed_num_segments:g}"
        f"_tol{segment_speed_tolerance_pct:g}{label_part}_seed{seed}"
    ).replace(".", "p")
    run_dir = output_root / "n8" / "coarse" / run_name
    argv = (
        sys.executable,
        "-m",
        "code_simulation.optimization.run_velocity_control_bo",
        "--target-front-speed-mm-s",
        f"{target:.12g}",
        "--num-knots",
        "8",
        "--knot-time-schedule",
        "early_dense",
        f"--theta0-c={_theta0_arg(target, override_theta0=theta0_override)}",
        "--theta-bounds=-21:0,-21:0,-21:0,-21:0,-21:0,-21:0,-21:0,-21:0",
        "--t-ref-bounds-c=-21,0",
        "--seed",
        str(seed),
        "--simulation-profile",
        simulation_profile,
        "--init-points",
        str(init_points),
        "--n-iter",
        str(n_iter),
        "--local-refinement-points",
        str(local_refinement_points),
        "--local-refinement-sigma",
        "0.04",
        "--tracking-weight",
        f"{tracking_weight:.12g}",
        "--direct-speed-weight",
        f"{direct_speed_weight:.12g}",
        "--direct-speed-tolerance-pct",
        f"{direct_speed_tolerance_pct:.12g}",
        "--segment-speed-weight",
        f"{segment_speed_weight:.12g}",
        "--segment-speed-tolerance-pct",
        f"{segment_speed_tolerance_pct:.12g}",
        "--segment-speed-num-segments",
        str(segment_speed_num_segments),
        "--acquisition-kind",
        "ei",
        "--acquisition-xi",
        "0.01",
        "--parameterization-kind",
        "monotone_unit_box",
        "--init-strategy",
        "feasible_local_deterministic",
        "--init-local-sigma",
        "0.25",
        "--init-max-attempts-per-point",
        "100",
        "--no-characterization-admissibility",
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
    )
    return CampaignJob(
        target_front_speed_mm_s=float(target),
        seed=int(seed),
        run_name=run_name,
        run_dir=run_dir,
        argv=argv,
    )


def _build_jobs(args: argparse.Namespace) -> list[CampaignJob]:
    targets = _parse_float_list(args.targets)
    seeds = _parse_int_list(args.seeds)
    theta0_override = _parse_float_list(args.theta0_c) if args.theta0_c is not None else None
    if theta0_override is not None and len(theta0_override) != 8:
        raise ValueError("--theta0-c must contain exactly 8 comma-separated temperatures for this n8 campaign")
    run_label = str(args.run_label or "").strip().replace(".", "p").replace(" ", "_")
    return [
        _build_job(
            target=target,
            seed=seed,
            output_root=args.output_root,
            init_points=int(args.init_points),
            n_iter=int(args.n_iter),
            local_refinement_points=int(args.local_refinement_points),
            tracking_weight=float(args.tracking_weight),
            direct_speed_weight=float(args.direct_speed_weight),
            direct_speed_tolerance_pct=float(args.direct_speed_tolerance_pct),
            segment_speed_weight=float(args.segment_speed_weight),
            segment_speed_tolerance_pct=float(args.segment_speed_tolerance_pct),
            segment_speed_num_segments=int(args.segment_speed_num_segments),
            simulation_profile=str(args.simulation_profile),
            theta0_override=theta0_override,
            run_label=run_label,
        )
        for target in targets
        for seed in seeds
    ]


def _run_job(job: CampaignJob, *, log_dir: Path, overwrite: bool) -> tuple[str, int | None]:
    if _run_complete(job.run_dir) and not overwrite:
        return "skipped_completed", None

    argv = list(job.argv)
    if overwrite:
        argv.append("--overwrite")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.run_name}.log"
    with log_path.open("w") as log:
        log.write("$ " + " ".join(argv) + "\n\n")
        log.write(f"start_time_utc={_timestamp_utc()}\n\n")
        log.flush()
        completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"\nend_time_utc={_timestamp_utc()}\n")
        log.write(f"returncode={completed.returncode}\n")
    if completed.returncode == 0 and _run_complete(job.run_dir):
        return "completed", int(completed.returncode)
    return f"failed_returncode_{completed.returncode}", int(completed.returncode)


def _history_summary(run_dir: Path) -> dict[str, float | int]:
    history = _read_rows(run_dir / "bo_history.csv")
    objectives = [_finite_float(row.get("objective_value")) for row in history]
    best_after = [_finite_float(row.get("best_objective_value_after_eval")) for row in history]
    finite_objectives = [value for value in objectives if math.isfinite(value)]
    finite_best = [value for value in best_after if math.isfinite(value)]
    last_improved = math.nan
    if finite_best:
        final_best = finite_best[-1]
        for row in history:
            best = _finite_float(row.get("best_objective_value_after_eval"))
            if math.isfinite(best) and abs(best - final_best) <= 1.0e-15:
                last_improved = _finite_float(row.get("evaluation_index"))
                break
    return {
        "n_evaluations": len(history),
        "first_objective": finite_objectives[0] if finite_objectives else math.nan,
        "best_objective": min(finite_objectives) if finite_objectives else math.nan,
        "final_best_objective": finite_best[-1] if finite_best else math.nan,
        "last_improved_evaluation_index": last_improved,
    }


def _criteria_score(
    *,
    direct_error_pct: float,
    tracking_rmse_mm: float,
    tracking_max_abs_error_mm: float,
    segment_speed_rmse_pct: float,
    segment_speed_max_abs_error_pct: float,
) -> float:
    values = (
        abs(float(direct_error_pct)) / ACCEPT_DIRECT_ERROR_PCT,
        float(tracking_rmse_mm) / ACCEPT_TRACKING_RMSE_MM,
        float(tracking_max_abs_error_mm) / ACCEPT_TRACKING_MAX_ABS_MM,
        float(segment_speed_rmse_pct) / ACCEPT_SEGMENT_RMSE_PCT,
        float(segment_speed_max_abs_error_pct) / ACCEPT_SEGMENT_MAX_ABS_ERROR_PCT,
    )
    return float(max(values)) if all(math.isfinite(value) for value in values) else math.inf


def _passes_acceptance(
    *,
    direct_error_pct: float,
    tracking_rmse_mm: float,
    tracking_max_abs_error_mm: float,
    segment_speed_rmse_pct: float,
    segment_speed_max_abs_error_pct: float,
) -> bool:
    return bool(
        math.isfinite(direct_error_pct)
        and abs(float(direct_error_pct)) <= ACCEPT_DIRECT_ERROR_PCT
        and math.isfinite(tracking_rmse_mm)
        and float(tracking_rmse_mm) <= ACCEPT_TRACKING_RMSE_MM
        and math.isfinite(tracking_max_abs_error_mm)
        and float(tracking_max_abs_error_mm) <= ACCEPT_TRACKING_MAX_ABS_MM
        and math.isfinite(segment_speed_rmse_pct)
        and float(segment_speed_rmse_pct) <= ACCEPT_SEGMENT_RMSE_PCT
        and math.isfinite(segment_speed_max_abs_error_pct)
        and float(segment_speed_max_abs_error_pct) <= ACCEPT_SEGMENT_MAX_ABS_ERROR_PCT
    )


def _history_by_case(run_dir: Path) -> dict[str, dict[str, str]]:
    return {
        str(row.get("case_name", "")): row
        for row in _read_rows(run_dir / "bo_history.csv")
        if str(row.get("case_name", ""))
    }


def _candidate_from_summary(
    summary_path: Path,
    *,
    target: float,
    history: dict[str, dict[str, str]],
) -> dict[str, object]:
    summary = _read_first_row(summary_path)
    case_name = summary_path.name.removesuffix("_velocity_objective_summary.csv")
    history_row = history.get(case_name, {})
    achieved = _finite_float(summary.get("actual_interval_speed_mm_s"))
    direct_error_pct = _finite_float(summary.get("direct_speed_relative_error_pct"))
    if not math.isfinite(direct_error_pct) and math.isfinite(achieved) and target > 0.0:
        direct_error_pct = float(100.0 * abs(achieved - target) / target)
    tracking_rmse_mm = _finite_float(summary.get("tracking_rmse_mm"))
    tracking_max_abs_error_mm = _finite_float(summary.get("tracking_max_abs_error_mm"))
    segment_speed_rmse_pct = _finite_float(summary.get("segment_speed_rmse_pct"))
    segment_speed_max_abs_error_pct = _finite_float(summary.get("segment_speed_max_abs_error_pct"))
    score = _criteria_score(
        direct_error_pct=direct_error_pct,
        tracking_rmse_mm=tracking_rmse_mm,
        tracking_max_abs_error_mm=tracking_max_abs_error_mm,
        segment_speed_rmse_pct=segment_speed_rmse_pct,
        segment_speed_max_abs_error_pct=segment_speed_max_abs_error_pct,
    )
    accepted = _passes_acceptance(
        direct_error_pct=direct_error_pct,
        tracking_rmse_mm=tracking_rmse_mm,
        tracking_max_abs_error_mm=tracking_max_abs_error_mm,
        segment_speed_rmse_pct=segment_speed_rmse_pct,
        segment_speed_max_abs_error_pct=segment_speed_max_abs_error_pct,
    )
    front_matches = sorted(summary_path.parent.glob("*_front.csv"))
    front_path = front_matches[0] if front_matches else None
    theta_values = ""
    theta_json = str(history_row.get("theta_json", ""))
    if theta_json:
        try:
            theta_values = ",".join(f"{float(value):.15g}" for value in json.loads(theta_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            theta_values = theta_json
    return {
        "accepted": accepted,
        "criteria_score": score,
        "case_name": case_name,
        "evaluation_index": _finite_float(history_row.get("evaluation_index")),
        "objective_value": _finite_float(summary.get("objective_value")),
        "achieved_direct_speed_mm_s": achieved,
        "direct_speed_relative_error_pct": direct_error_pct,
        "tracking_rmse_mm": tracking_rmse_mm,
        "tracking_max_abs_error_mm": tracking_max_abs_error_mm,
        "segment_speed_rmse_pct": segment_speed_rmse_pct,
        "segment_speed_mean_abs_error_pct": _finite_float(summary.get("segment_speed_mean_abs_error_pct")),
        "segment_speed_max_abs_error_pct": segment_speed_max_abs_error_pct,
        "segment_speed_min_mm_s": _finite_float(summary.get("segment_speed_min_mm_s")),
        "segment_speed_max_mm_s": _finite_float(summary.get("segment_speed_max_mm_s")),
        "segment_speed_spread_mm_s": _finite_float(summary.get("segment_speed_spread_mm_s")),
        "segment_speed_num_valid_segments": _finite_float(summary.get("segment_speed_num_valid_segments")),
        "t_at_control_z_min_s": _finite_float(summary.get("t_at_control_z_min_s")),
        "summary_path": str(summary_path),
        "front_path": "" if front_path is None else str(front_path),
        "theta_c": theta_values,
        "segment_speeds_mm_s": _segment_speeds_from_front_path(
            front_path,
            num_segments=int(_finite_float(summary.get("segment_speed_num_segments"), 5.0)),
        ),
    }


def _select_candidate_by_criteria(run_dir: Path, *, target: float) -> dict[str, object] | None:
    summaries = sorted(run_dir.glob("evaluations/eval_*/*_velocity_objective_summary.csv"))
    if not summaries:
        return None
    history = _history_by_case(run_dir)
    candidates = [
        _candidate_from_summary(summary_path, target=target, history=history)
        for summary_path in summaries
    ]
    candidates = [candidate for candidate in candidates if math.isfinite(float(candidate["criteria_score"]))]
    if not candidates:
        return None
    accepted = [candidate for candidate in candidates if bool(candidate["accepted"])]
    pool = accepted if accepted else candidates
    selected = min(
        pool,
        key=lambda candidate: (
            float(candidate["criteria_score"]),
            _finite_float(candidate.get("objective_value"), math.inf),
            _finite_float(candidate.get("evaluation_index"), math.inf),
        ),
    )
    selected["selection_source"] = "accepted_by_criteria" if accepted else "least_bad_by_criteria"
    return selected


def _aggregate_run(job: CampaignJob, *, status: str) -> dict[str, object]:
    tracking = _read_first_row(job.run_dir / "best_tracking_summary.csv")
    segment = _read_first_row(job.run_dir / "best_segment_speed_summary.csv")
    segment_rows = _read_rows(job.run_dir / "best_segment_speeds.csv")
    history = _history_summary(job.run_dir) if _run_complete(job.run_dir) else {}
    target = float(job.target_front_speed_mm_s)
    selected = (
        _select_candidate_by_criteria(job.run_dir, target=target)
        if status in {"completed", "skipped_completed"} and _run_complete(job.run_dir)
        else None
    )
    achieved = (
        _finite_float(selected.get("achieved_direct_speed_mm_s")) if selected is not None
        else _finite_float(tracking.get("actual_interval_speed_mm_s"))
    )
    direct_error_pct = (
        _finite_float(selected.get("direct_speed_relative_error_pct")) if selected is not None
        else (
            100.0 * abs(achieved - target) / target
            if math.isfinite(achieved) and math.isfinite(target) and target > 0.0
            else math.nan
        )
    )
    tracking_rmse_mm = (
        _finite_float(selected.get("tracking_rmse_mm")) if selected is not None
        else _finite_float(tracking.get("tracking_rmse_mm"))
    )
    tracking_max_abs_error_mm = (
        _finite_float(selected.get("tracking_max_abs_error_mm")) if selected is not None
        else _finite_float(tracking.get("tracking_max_abs_error_mm"))
    )
    segment_rmse_pct = (
        _finite_float(selected.get("segment_speed_rmse_pct")) if selected is not None
        else _finite_float(segment.get("segment_speed_rmse_pct"))
    )
    worst_segment_error_pct = (
        _finite_float(selected.get("segment_speed_max_abs_error_pct")) if selected is not None
        else _finite_float(segment.get("segment_speed_max_abs_error_pct"))
    )
    accept = status in {"completed", "skipped_completed"} and _passes_acceptance(
        direct_error_pct=direct_error_pct,
        tracking_rmse_mm=tracking_rmse_mm,
        tracking_max_abs_error_mm=tracking_max_abs_error_mm,
        segment_speed_rmse_pct=segment_rmse_pct,
        segment_speed_max_abs_error_pct=worst_segment_error_pct,
    )
    return {
        "run_name": job.run_name,
        "target_front_speed_mm_s": target,
        "seed": int(job.seed),
        "status": status,
        **history,
        "achieved_direct_speed_mm_s": achieved,
        "direct_speed_relative_error_pct": direct_error_pct,
        "tracking_rmse_mm": tracking_rmse_mm,
        "tracking_max_abs_error_mm": tracking_max_abs_error_mm,
        "segment_speed_rmse_pct": segment_rmse_pct,
        "segment_speed_mean_abs_error_pct": (
            _finite_float(selected.get("segment_speed_mean_abs_error_pct")) if selected is not None
            else _finite_float(segment.get("segment_speed_mean_abs_error_pct"))
        ),
        "segment_speed_max_abs_error_pct": worst_segment_error_pct,
        "segment_speed_min_mm_s": (
            _finite_float(selected.get("segment_speed_min_mm_s")) if selected is not None
            else _finite_float(segment.get("segment_speed_min_mm_s"))
        ),
        "segment_speed_max_mm_s": (
            _finite_float(selected.get("segment_speed_max_mm_s")) if selected is not None
            else _finite_float(segment.get("segment_speed_max_mm_s"))
        ),
        "segment_speed_spread_mm_s": (
            _finite_float(selected.get("segment_speed_spread_mm_s")) if selected is not None
            else _finite_float(segment.get("segment_speed_spread_mm_s"))
        ),
        "segment_speed_num_valid_segments": (
            _finite_float(selected.get("segment_speed_num_valid_segments")) if selected is not None
            else _finite_float(segment.get("segment_speed_num_valid_segments"))
        ),
        "accept_fronttrack_direct_and_segment_speed": int(bool(accept)),
        "selection_source": "" if selected is None else str(selected.get("selection_source", "")),
        "selected_case_name": "" if selected is None else str(selected.get("case_name", "")),
        "selected_evaluation_index": math.nan if selected is None else _finite_float(selected.get("evaluation_index")),
        "selected_criteria_score": math.nan if selected is None else _finite_float(selected.get("criteria_score")),
        "selected_objective_value": math.nan if selected is None else _finite_float(selected.get("objective_value")),
        "selected_t_at_control_z_min_s": (
            math.nan if selected is None else _finite_float(selected.get("t_at_control_z_min_s"))
        ),
        "selected_summary_path": "" if selected is None else str(selected.get("summary_path", "")),
        "selected_front_path": "" if selected is None else str(selected.get("front_path", "")),
        "selected_theta_c": "" if selected is None else str(selected.get("theta_c", "")),
        "segment_speeds_mm_s": (
            str(selected.get("segment_speeds_mm_s", "")) if selected is not None
            else ";".join(str(row.get("speed_mm_s", "")) for row in segment_rows)
        ),
        "run_dir": str(job.run_dir),
    }


def _summary_fieldnames() -> tuple[str, ...]:
    return (
        "run_name",
        "target_front_speed_mm_s",
        "seed",
        "status",
        "n_evaluations",
        "first_objective",
        "best_objective",
        "final_best_objective",
        "last_improved_evaluation_index",
        "achieved_direct_speed_mm_s",
        "direct_speed_relative_error_pct",
        "tracking_rmse_mm",
        "tracking_max_abs_error_mm",
        "segment_speed_rmse_pct",
        "segment_speed_mean_abs_error_pct",
        "segment_speed_max_abs_error_pct",
        "segment_speed_min_mm_s",
        "segment_speed_max_mm_s",
        "segment_speed_spread_mm_s",
        "segment_speed_num_valid_segments",
        "accept_fronttrack_direct_and_segment_speed",
        "selection_source",
        "selected_case_name",
        "selected_evaluation_index",
        "selected_criteria_score",
        "selected_objective_value",
        "selected_t_at_control_z_min_s",
        "selected_summary_path",
        "selected_front_path",
        "selected_theta_c",
        "segment_speeds_mm_s",
        "run_dir",
    )


def _write_campaign_summary(path: Path, jobs: list[CampaignJob], statuses: dict[str, str]) -> list[dict[str, object]]:
    rows = [_aggregate_run(job, status=statuses.get(job.run_name, "pending")) for job in jobs]
    _write_rows(path, rows, _summary_fieldnames())
    return rows


def _plot_campaign_summary(diag_root: Path, rows: list[dict[str, object]]) -> None:
    completed = [
        row
        for row in rows
        if str(row.get("status")) in {"completed", "skipped_completed"}
        and math.isfinite(_finite_float(row.get("achieved_direct_speed_mm_s")))
    ]
    if not completed:
        return
    configure_matplotlib(plt)
    targets = [_finite_float(row["target_front_speed_mm_s"]) for row in completed]
    achieved = [_finite_float(row["achieved_direct_speed_mm_s"]) for row in completed]
    axis_max = max(max(targets), max(achieved)) * 1.04

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.45", linewidth=1.2, label="target = achieved")
    for seed in sorted({int(row["seed"]) for row in completed}):
        seed_rows = [row for row in completed if int(row["seed"]) == seed]
        ax.scatter(
            [_finite_float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [_finite_float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
            color=COLOR_BY_SEED.get(seed, "0.35"),
            s=38,
            label=f"seed {seed}",
        )
    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Target direct speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title("n8 front-tracking BO: target vs achieved direct speed")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(diag_root / "target_vs_achieved_direct_speed.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for seed in sorted({int(row["seed"]) for row in completed}):
        seed_rows = sorted([row for row in completed if int(row["seed"]) == seed], key=lambda row: float(row["target_front_speed_mm_s"]))
        axes[0].plot(
            [_finite_float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [_finite_float(row["segment_speed_rmse_pct"]) for row in seed_rows],
            marker="o",
            linewidth=1.2,
            color=COLOR_BY_SEED.get(seed, "0.35"),
            label=f"seed {seed}",
        )
        axes[1].plot(
            [_finite_float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [_finite_float(row["segment_speed_max_abs_error_pct"]) for row in seed_rows],
            marker="o",
            linewidth=1.2,
            color=COLOR_BY_SEED.get(seed, "0.35"),
        )
    axes[0].axhline(5.0, color="0.35", linestyle=":", linewidth=1.0, label="5% criterion")
    axes[1].axhline(10.0, color="0.35", linestyle=":", linewidth=1.0, label="10% criterion")
    axes[0].set_ylabel("Segment speed RMSE (%)")
    axes[1].set_ylabel("Worst segment speed error (%)")
    for ax in axes:
        ax.set_xlabel("Target direct speed (mm/s)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("Segment-speed RMSE")
    axes[1].set_title("Worst segment-speed error")
    fig.suptitle("n8 front-tracking BO: constant-speed robustness metrics", fontsize=12)
    fig.tight_layout()
    fig.savefig(diag_root / "segment_speed_robustness_by_seed.png", dpi=250)
    plt.close(fig)

    _plot_front_position_overlays(diag_root, completed)
    _plot_front_position_error_overlays(diag_root, completed)
    for window_s in VELOCITY_SMOOTHING_WINDOWS_S:
        _plot_front_velocity_overlays(diag_root, completed, window_s=window_s)


def _group_completed_by_target(rows: list[dict[str, object]]) -> dict[float, list[dict[str, object]]]:
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        target = _finite_float(row.get("target_front_speed_mm_s"))
        if math.isfinite(target):
            grouped.setdefault(target, []).append(row)
    return grouped


def _plot_front_position_overlays(diag_root: Path, rows: list[dict[str, object]]) -> None:
    for target, target_rows in _group_completed_by_target(rows).items():
        expected_interval_s = (11.5 - 2.5) / float(target)
        plotted = 0
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for row in sorted(target_rows, key=lambda item: int(float(item["seed"]))):
            run_dir = Path(str(row["run_dir"]))
            selected_front_raw = str(row.get("selected_front_path", "")).strip()
            selected_front_path = Path(selected_front_raw) if selected_front_raw else None
            front_path = (
                selected_front_path
                if selected_front_path is not None and selected_front_path.exists()
                else _best_front_path(run_dir)
            )
            if front_path is None:
                continue
            t0_s = _finite_float(row.get("selected_t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                tracking = _read_first_row(run_dir / "best_tracking_summary.csv")
                t0_s = _finite_float(tracking.get("t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                continue
            series = _read_front_series(front_path)
            aligned_time_s = series["time_s"] - t0_s
            z_front_mm = series["z_front_mm"]
            plot_mask = (
                np.isfinite(aligned_time_s)
                & np.isfinite(z_front_mm)
                & (aligned_time_s >= -60.0)
                & (aligned_time_s <= expected_interval_s + 60.0)
            )
            if not np.any(plot_mask):
                continue
            seed = int(float(row["seed"]))
            ax.plot(
                aligned_time_s[plot_mask],
                z_front_mm[plot_mask],
                linewidth=1.5,
                color=COLOR_BY_SEED.get(seed, "0.35"),
                label=f"seed {seed}",
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ref_time_s = np.linspace(0.0, expected_interval_s, 200, dtype=np.float64)
        ax.plot(ref_time_s, 2.5 + float(target) * ref_time_s, "--", color="0.15", linewidth=1.6, label="target")
        ax.axhline(2.5, color="0.45", linestyle=":", linewidth=1.0)
        ax.axhline(11.5, color="0.45", linestyle=":", linewidth=1.0)
        ax.set_xlim(-60.0, expected_interval_s + 60.0)
        ax.set_ylim(0.0, 12.2)
        ax.set_xlabel("Time aligned to z_front = 2.5 mm (s)")
        ax.set_ylabel("Front position (mm)")
        ax.set_title(f"n8 front trajectories vs target: {target:.3f} mm/s")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(diag_root / f"front_position_overlay_v{_format_speed_tag(target)}.png", dpi=250)
        plt.close(fig)


def _plot_front_position_error_overlays(diag_root: Path, rows: list[dict[str, object]]) -> None:
    for target, target_rows in _group_completed_by_target(rows).items():
        expected_interval_s = (11.5 - 2.5) / float(target)
        plotted = 0
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        for row in sorted(target_rows, key=lambda item: int(float(item["seed"]))):
            run_dir = Path(str(row["run_dir"]))
            selected_front_raw = str(row.get("selected_front_path", "")).strip()
            selected_front_path = Path(selected_front_raw) if selected_front_raw else None
            front_path = (
                selected_front_path
                if selected_front_path is not None and selected_front_path.exists()
                else _best_front_path(run_dir)
            )
            if front_path is None:
                continue
            t0_s = _finite_float(row.get("selected_t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                tracking = _read_first_row(run_dir / "best_tracking_summary.csv")
                t0_s = _finite_float(tracking.get("t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                continue
            series = _read_front_series(front_path)
            aligned_time_s = series["time_s"] - t0_s
            z_front_mm = series["z_front_mm"]
            target_z_mm = 2.5 + float(target) * aligned_time_s
            error_mm = z_front_mm - target_z_mm
            plot_mask = (
                np.isfinite(aligned_time_s)
                & np.isfinite(error_mm)
                & (aligned_time_s >= 0.0)
                & (aligned_time_s <= expected_interval_s)
            )
            if not np.any(plot_mask):
                continue
            seed = int(float(row["seed"]))
            ax.plot(
                aligned_time_s[plot_mask],
                error_mm[plot_mask],
                linewidth=1.4,
                color=COLOR_BY_SEED.get(seed, "0.35"),
                label=f"seed {seed}",
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.axhline(0.0, color="0.15", linestyle="-", linewidth=1.1, label="target")
        ax.axhline(ACCEPT_TRACKING_RMSE_MM, color="0.45", linestyle=":", linewidth=1.0)
        ax.axhline(-ACCEPT_TRACKING_RMSE_MM, color="0.45", linestyle=":", linewidth=1.0)
        ax.axhline(ACCEPT_TRACKING_MAX_ABS_MM, color="0.35", linestyle="--", linewidth=1.0)
        ax.axhline(-ACCEPT_TRACKING_MAX_ABS_MM, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, expected_interval_s)
        ax.set_xlabel("Time aligned to z_front = 2.5 mm (s)")
        ax.set_ylabel("Front-position error (mm)")
        ax.set_title(f"n8 front-position error vs target: {target:.3f} mm/s")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(diag_root / f"front_position_error_overlay_v{_format_speed_tag(target)}.png", dpi=250)
        plt.close(fig)


def _plot_front_velocity_overlays(
    diag_root: Path,
    rows: list[dict[str, object]],
    *,
    window_s: float,
) -> None:
    for target, target_rows in _group_completed_by_target(rows).items():
        expected_interval_s = (11.5 - 2.5) / float(target)
        plotted = 0
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for row in sorted(target_rows, key=lambda item: int(float(item["seed"]))):
            run_dir = Path(str(row["run_dir"]))
            selected_front_raw = str(row.get("selected_front_path", "")).strip()
            selected_front_path = Path(selected_front_raw) if selected_front_raw else None
            front_path = (
                selected_front_path
                if selected_front_path is not None and selected_front_path.exists()
                else _best_front_path(run_dir)
            )
            if front_path is None:
                continue
            t0_s = _finite_float(row.get("selected_t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                tracking = _read_first_row(run_dir / "best_tracking_summary.csv")
                t0_s = _finite_float(tracking.get("t_at_control_z_min_s"))
            if not math.isfinite(t0_s):
                continue
            series = _read_front_series(front_path)
            aligned_time_s = series["time_s"] - t0_s
            velocity_mm_s = _local_linear_velocity_mm_s(
                series["time_s"],
                series["z_front_mm"],
                window_s=window_s,
            )
            plot_mask = (
                np.isfinite(aligned_time_s)
                & np.isfinite(velocity_mm_s)
                & (aligned_time_s >= 0.0)
                & (aligned_time_s <= expected_interval_s)
            )
            if not np.any(plot_mask):
                continue
            seed = int(float(row["seed"]))
            ax.plot(
                aligned_time_s[plot_mask],
                velocity_mm_s[plot_mask],
                linewidth=1.4,
                color=COLOR_BY_SEED.get(seed, "0.35"),
                label=f"seed {seed}",
            )
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.axhline(float(target), color="0.15", linestyle="--", linewidth=1.6, label="target")
        ax.set_xlim(0.0, expected_interval_s)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("Time aligned to z_front = 2.5 mm (s)")
        ax.set_ylabel("Smoothed front velocity (mm/s)")
        ax.set_title(
            f"n8 smoothed front velocity vs target: {target:.3f} mm/s "
            f"({window_s:.0f} s local fit)"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output_path = diag_root / f"front_velocity_overlay_v{_format_speed_tag(target)}_w{window_s:.0f}s.png"
        fig.savefig(output_path, dpi=250)
        if abs(float(window_s) - 60.0) <= 1.0e-9:
            fig.savefig(diag_root / f"front_velocity_overlay_v{_format_speed_tag(target)}.png", dpi=250)
        plt.close(fig)


def _write_readme(path: Path, args: argparse.Namespace, jobs: list[CampaignJob]) -> None:
    lines = [
        "# n8 front-tracking robust BO campaign",
        "",
        "Purpose: make the simulated front-position trajectory follow the linear target over the depth window, not only match the global direct speed.",
        "",
        "## Configuration",
        "",
        f"- Targets: `{args.targets}` mm/s",
        f"- Seeds: `{args.seeds}`",
        "- Schedule: `early_dense`",
        "- Knots: `n=8`",
        f"- Front-position tracking penalty: weight `{args.tracking_weight}`",
        f"- Direct-speed penalty: weight `{args.direct_speed_weight}`, tolerance `{args.direct_speed_tolerance_pct}%`",
        f"- Segment-speed penalty: `{args.segment_speed_num_segments}` segments, weight `{args.segment_speed_weight}`, tolerance `{args.segment_speed_tolerance_pct}%`",
        f"- BO budget: theta0 + `{args.init_points}` deterministic init + `{args.n_iter}` BO + `{args.local_refinement_points}` local refinement",
        f"- Theta0 override: `{args.theta0_c or 'target-specific default'}`.",
        f"- Run label: `{args.run_label or ''}`.",
        "- Characterization admissibility: disabled, matching the previous robust direct-speed campaign.",
        "",
        "## Acceptance criteria",
        "",
        f"- Direct-speed relative error <= `{ACCEPT_DIRECT_ERROR_PCT:g}%`.",
        f"- Front-position tracking RMSE <= `{ACCEPT_TRACKING_RMSE_MM:g} mm`.",
        f"- Front-position tracking max absolute error <= `{ACCEPT_TRACKING_MAX_ABS_MM:g} mm`.",
        f"- Segment-speed RMSE <= `{ACCEPT_SEGMENT_RMSE_PCT:g}%`.",
        f"- Worst segment-speed absolute error <= `{ACCEPT_SEGMENT_MAX_ABS_ERROR_PCT:g}%`.",
        "- All five seeds must be reported; a target is article-ready only if all seeds pass or any failure is localized in the overlays.",
        "",
        "## Outputs",
        "",
        "- `campaign_summary.csv` is updated after each run.",
        "- `target_vs_achieved_direct_speed.png` and `segment_speed_robustness_by_seed.png` are regenerated from completed rows.",
        "- `front_position_overlay_v0pXXX.png` compares all seed front-position curves against the linear target for each velocity.",
        "- `front_position_error_overlay_v0pXXX.png` shows `z_front(t) - z_target(t)` for each velocity.",
        "- `front_velocity_overlay_v0pXXX_wNNNs.png` compares all seed velocities against the target after centered local-linear smoothing fits.",
        "- `front_velocity_overlay_v0pXXX.png` is kept as the 60 s smoothing-window compatibility filename.",
        "- The smoothed velocity plots are diagnostics; front-position tracking is the primary objective.",
        "- Completed runs are re-ranked by the acceptance criteria; accepted candidates are preferred over the scalar BO incumbent.",
        "- Per-run BO folders are stored under `n8/coarse/`.",
        "- Per-run logs are stored under `logs/`.",
        "",
        f"Total jobs: `{len(jobs)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the n8 segmented-speed robust BO campaign.")
    parser.add_argument("--targets", default=",".join(f"{value:.3f}" for value in DEFAULT_TARGETS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diagnostic-root", type=Path, default=DEFAULT_DIAG_ROOT)
    parser.add_argument("--simulation-profile", default="optimization")
    parser.add_argument("--theta0-c", default=None, help="Optional comma-separated n8 theta0 override used for every target.")
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL, help="Optional short label inserted into run names before the seed.")
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--local-refinement-points", type=int, default=20)
    parser.add_argument("--tracking-weight", type=float, default=50.0)
    parser.add_argument("--direct-speed-weight", type=float, default=50.0)
    parser.add_argument("--direct-speed-tolerance-pct", type=float, default=1.0)
    parser.add_argument("--segment-speed-weight", type=float, default=25.0)
    parser.add_argument("--segment-speed-tolerance-pct", type=float, default=2.0)
    parser.add_argument("--segment-speed-num-segments", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = _build_jobs(args)
    args.diagnostic_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.diagnostic_root / "logs"
    _write_readme(args.diagnostic_root / "README.md", args, jobs)

    statuses: dict[str, str] = {}
    command_rows = [
        {
            "run_name": job.run_name,
            "target_front_speed_mm_s": job.target_front_speed_mm_s,
            "seed": job.seed,
            "run_dir": str(job.run_dir),
            "command": " ".join(job.argv),
        }
        for job in jobs
    ]
    _write_rows(
        args.diagnostic_root / "commands.csv",
        command_rows,
        ("run_name", "target_front_speed_mm_s", "seed", "run_dir", "command"),
    )
    _write_campaign_summary(args.diagnostic_root / "campaign_summary.csv", jobs, statuses)

    print(f"Jobs: {len(jobs)}")
    print(f"Diagnostic root: {args.diagnostic_root.resolve()}")
    print(f"Commands CSV: {(args.diagnostic_root / 'commands.csv').resolve()}")
    if args.dry_run:
        return

    for idx, job in enumerate(jobs, start=1):
        print(
            f"[{idx}/{len(jobs)}] {job.run_name} "
            f"target={job.target_front_speed_mm_s:.3f} seed={job.seed}",
            flush=True,
        )
        status, returncode = _run_job(job, log_dir=log_dir, overwrite=bool(args.overwrite))
        _ = returncode
        statuses[job.run_name] = status
        rows = _write_campaign_summary(args.diagnostic_root / "campaign_summary.csv", jobs, statuses)
        _plot_campaign_summary(args.diagnostic_root, rows)
        print(f"  status={status}", flush=True)

    rows = _write_campaign_summary(args.diagnostic_root / "campaign_summary.csv", jobs, statuses)
    _plot_campaign_summary(args.diagnostic_root, rows)
    print(f"Summary CSV: {(args.diagnostic_root / 'campaign_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
