#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.config_files import (
    DEFAULT_BO_CONFIG_PATH,
    DEFAULT_SIMULATION_CONFIG_PATH,
    load_bo_config,
)
from code_simulation.core.paths import (
    project_root,
    velocity_control_coarse_dir,
    velocity_control_diagnostic_dir,
)
from code_simulation.core.plate_tracking import plate_tracking_success
from code_simulation.studies.bo_3knot_target_and_spacing_study_impl import (
    INTERVAL_3_TO_11_MM,
    THETA_FAMILY_ROUND_DECIMALS,
    _best_theta_from_csv,
    _configure_matplotlib,
    _ensure_bo_plate_tracking_artifacts,
    _ensure_clean_directory,
    _format_speed_tag,
    _history_stats,
    _interval_speed,
    _matches_target,
    _normalized_support_tau_from_knot_times,
    _parse_float_list,
    _parse_int_list,
    _read_csv_rows,
    _read_toml,
    _single_row_csv,
    _speed_success,
    _write_csv,
    _write_text,
)


@dataclass(frozen=True)
class StudyRun:
    run_dir: Path
    run_name: str
    target_front_speed_mm_s: float
    schedule: str
    num_knots: int
    normalized_support_tau: tuple[float, ...]
    knot_times_s: tuple[float, ...]
    theta_C: tuple[float, ...]
    initial_theta_C: tuple[float, ...]
    theta0_source: str
    continuation_parent_run_dir: str
    continuation_parent_run_name: str
    objective_value: float
    achieved_direct_speed_mm_s: float
    achieved_tc_3to11_mm_s: float
    tracking_rmse_mm: float
    t_at_control_z_min_s: float
    t_at_control_z_max_s: float
    plate_tolerance_C: float
    max_abs_plate_error_C: float
    rmse_plate_error_C: float
    mean_plate_error_C: float
    mean_abs_plate_error_C: float
    fraction_within_tolerance: float
    n_evaluations: int
    best_evaluation_index: int
    late_best_eval: bool
    random_seed: int | None = None


DEFAULT_OUTPUT_ROOT = velocity_control_diagnostic_dir(3, "bo_variability_reduction_admissible_range")
DEFAULT_RUNNER_ROOT = velocity_control_diagnostic_dir(3, "bo_variability_reduction_admissible_range_runner")
DEFAULT_BO_RUNS_ROOT = velocity_control_coarse_dir(3)
DEFAULT_BASELINE_BUNDLE_ROOT = velocity_control_diagnostic_dir(3, "bo_monotone_feasible_admissible_range")
DEFAULT_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.013 + 1.0e-12, 0.001))
DEFAULT_SCHEDULES = ("uniform", "early_dense", "late_dense")
DEFAULT_SEEDS = (17, 29, 41)
GATE_TARGETS_MM_S = (0.004, 0.006, 0.009, 0.013)
UPPER_RANGE_TARGETS_MM_S = (0.010, 0.011, 0.012, 0.013)
STUDY_NUM_KNOTS = 3
RUN_NAME_TAG = "bov4"
BASELINE_NAME = "bo3"
CANONICAL_THETA0_C = (0.0, -10.0, -20.0)
THETA_BOUNDS_C = ((-10.0, 0.0), (-16.0, -6.0), (-21.0, -10.0))
ACQUISITION_KIND = "ei"
ACQUISITION_XI = 0.01
INIT_POINTS = 2
N_ITER = 30
INIT_STRATEGY = "feasible_local_deterministic"
PARAMETERIZATION_KIND = "monotone_unit_box"
INIT_LOCAL_SIGMA = 0.15
INIT_MAX_ATTEMPTS_PER_POINT = 40
LOCAL_REFINEMENT_POINTS = 6
LOCAL_REFINEMENT_SIGMA = 0.08
ROBUST_MAX_MEDIAN_SPEED_ERROR_PCT = 5.0
ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION = 0.35
ROBUST_MAX_LATE_BEST_FRACTION = 0.4
PLATEAU_MIN_ABS_SPREAD_MM_S = 0.001
PLATEAU_MIN_REL_SPREAD = 0.10
GATE_B_MAX_INFEASIBLE_WORSEN_ABS = 0.05
GATE_B_MIN_SPREAD_IMPROVEMENTS = 6
GATE_B_MIN_IMPROVEMENT_HITS = 4
UPPER_PLATEAU_MAX_SLOPE_RATIO = 0.35
UPPER_PLATEAU_MAX_MEDIAN_SPREAD_MM_S = 0.0006


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _target_range_text(targets_mm_s: tuple[float, ...]) -> str:
    if not targets_mm_s:
        return "none"
    return f"{min(targets_mm_s):.3f} -> {max(targets_mm_s):.3f}"


def _theta0_text() -> str:
    return ",".join(f"{float(value):.15g}" for value in CANONICAL_THETA0_C)


def _theta_bounds_text() -> str:
    return ",".join(f"{lower:.15g}:{upper:.15g}" for lower, upper in THETA_BOUNDS_C)


def _baseline_bundle_target_schedule_csv(root: Path) -> Path:
    return root / "target_schedule_aggregate.csv"


def _target_command(
    run_name: str,
    target_mm_s: float,
    schedule: str,
    seed: int,
    *,
    parent_run_dir: Path | None,
) -> str:
    repo_root = project_root()
    if parent_run_dir is None:
        theta_arg = f"  --theta0-c={_theta0_text()} \\\n"
    else:
        theta_arg = f"  --theta0-from-run-dir {parent_run_dir} \\\n"
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.optimization.run_velocity_control_bo \\\n"
        f"  --output-root {repo_root / 'code_simulation' / 'results' / 'active' / 'bo_velocity_control'} \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        "  --num-knots 3 \\\n"
        f"  --knot-time-schedule {schedule} \\\n"
        f"{theta_arg}"
        f"  --theta-bounds={_theta_bounds_text()} \\\n"
        f"  --seed {int(seed)} \\\n"
        "  --simulation-profile optimization \\\n"
        f"  --init-points {INIT_POINTS} \\\n"
        f"  --n-iter {N_ITER} \\\n"
        f"  --acquisition-kind {ACQUISITION_KIND} \\\n"
        f"  --acquisition-xi {ACQUISITION_XI:.15g} \\\n"
        f"  --parameterization-kind {PARAMETERIZATION_KIND} \\\n"
        f"  --init-strategy {INIT_STRATEGY} \\\n"
        f"  --init-local-sigma {INIT_LOCAL_SIGMA:.15g} \\\n"
        f"  --init-max-attempts-per-point {INIT_MAX_ATTEMPTS_PER_POINT} \\\n"
        f"  --local-refinement-points {LOCAL_REFINEMENT_POINTS} \\\n"
        f"  --local-refinement-sigma {LOCAL_REFINEMENT_SIGMA:.15g} \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _representative_dry_run_command() -> str:
    return _target_command(
        run_name=f"bo_v0p008_n3_uniform_{RUN_NAME_TAG}_seed17",
        target_mm_s=0.008,
        schedule="uniform",
        seed=17,
        parent_run_dir=None,
    ).replace("--overwrite", "--dry-run-config")


def _run_name(target_mm_s: float, schedule: str, seed: int) -> str:
    return f"bo_v{_format_speed_tag(target_mm_s)}_n3_{schedule}_{RUN_NAME_TAG}_seed{int(seed)}"


def _history_diagnostics(path: Path) -> dict[str, float | int]:
    rows = _read_csv_rows(path)
    n_total = len(rows)
    n_infeasible = sum(1 for row in rows if str(row.get("is_valid", "")).strip() not in {"1", "true", "True", "TRUE"})
    _, n_evaluations, best_evaluation_index, late_best = _history_stats(path)
    infeasible_fraction = math.nan if n_total == 0 else float(n_infeasible) / float(n_total)
    best_eval_fraction = math.nan if n_evaluations == 0 else float(best_evaluation_index) / float(n_evaluations)
    return {
        "n_total_evaluations": int(n_total),
        "n_infeasible_evaluations": int(n_infeasible),
        "infeasible_fraction": float(infeasible_fraction),
        "best_evaluation_index": int(best_evaluation_index),
        "best_evaluation_fraction": float(best_eval_fraction),
        "late_best_eval": int(late_best),
    }


def _matches_tuned_bounds(raw_bounds) -> bool:
    if not isinstance(raw_bounds, list) or len(raw_bounds) != len(THETA_BOUNDS_C):
        return False
    for raw_pair, tuned_pair in zip(raw_bounds, THETA_BOUNDS_C, strict=True):
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            return False
        if abs(float(raw_pair[0]) - float(tuned_pair[0])) > 1.0e-12:
            return False
        if abs(float(raw_pair[1]) - float(tuned_pair[1])) > 1.0e-12:
            return False
    return True


def _matches_bov4_signature(config: dict) -> bool:
    trajectory_cfg = dict(config.get("trajectory", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    if int(trajectory_cfg.get("num_knots", 0)) != STUDY_NUM_KNOTS:
        return False
    if not _matches_tuned_bounds(trajectory_cfg.get("theta_bounds_C", [])):
        return False
    if str(bo_cfg.get("acquisition_kind", "")).strip().lower() != ACQUISITION_KIND:
        return False
    if abs(float(bo_cfg.get("acquisition_xi", math.nan)) - ACQUISITION_XI) > 1.0e-12:
        return False
    if int(bo_cfg.get("init_points", -1)) != INIT_POINTS:
        return False
    if int(bo_cfg.get("n_iter", -1)) != N_ITER:
        return False
    if not bool(bo_cfg.get("seed_with_theta0", False)):
        return False
    if str(bo_cfg.get("parameterization_kind", "")).strip().lower() != PARAMETERIZATION_KIND:
        return False
    if str(bo_cfg.get("init_strategy", "")).strip().lower() != INIT_STRATEGY:
        return False
    if abs(float(bo_cfg.get("init_local_sigma", math.nan)) - INIT_LOCAL_SIGMA) > 1.0e-12:
        return False
    if int(bo_cfg.get("init_max_attempts_per_point", -1)) != INIT_MAX_ATTEMPTS_PER_POINT:
        return False
    if int(bo_cfg.get("local_refinement_points", -1)) != LOCAL_REFINEMENT_POINTS:
        return False
    if abs(float(bo_cfg.get("local_refinement_sigma", math.nan)) - LOCAL_REFINEMENT_SIGMA) > 1.0e-12:
        return False
    return True


def _parse_theta_tuple(raw_values) -> tuple[float, ...]:
    if not isinstance(raw_values, list):
        return tuple()
    return tuple(float(value) for value in raw_values)


def _theta0_metadata(theta0_source: str) -> tuple[str, str]:
    source = str(theta0_source).strip()
    if not source.startswith("from_run:"):
        return ("", "")
    raw_path = source.split("from_run:", 1)[1].strip()
    if not raw_path:
        return ("", "")
    parent_dir = Path(raw_path)
    return (str(parent_dir), parent_dir.name)


def _load_bov4_run(path: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> StudyRun | None:
    _ensure_bo_plate_tracking_artifacts(path, simulation_config_path=simulation_config_path, schedules=schedules)
    effective_path = path / "effective_config.toml"
    tracking_path = path / "best_tracking_summary.csv"
    theta_path = path / "best_theta_profile.csv"
    history_path = path / "bo_history.csv"
    eval_history_path = path / "evaluation_history.csv"
    plate_path = path / "plate_tracking_summary.csv"
    tc_path = path / "best_thermocouple_interval_speeds.csv"
    if not (
        effective_path.exists()
        and tracking_path.exists()
        and theta_path.exists()
        and history_path.exists()
        and eval_history_path.exists()
        and plate_path.exists()
    ):
        return None
    config = _read_toml(effective_path)
    if not _matches_bov4_signature(config):
        return None
    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    target_cfg = dict(config.get("velocity_target", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    run_name = str(run_cfg.get("run_name", path.name))
    if f"_{RUN_NAME_TAG}_" not in run_name:
        return None
    if str(run_cfg.get("simulation_profile", "")).strip() != "optimization":
        return None
    num_knots = int(trajectory_cfg.get("num_knots", 0))
    if num_knots != STUDY_NUM_KNOTS:
        return None
    knot_times_s = tuple(float(value) for value in trajectory_cfg.get("knot_times_s", []))
    normalized_support_tau = _normalized_support_tau_from_knot_times(
        knot_times_s=knot_times_s,
        horizon_s=float(trajectory_cfg.get("horizon_s", 0.0)),
    )
    schedule = str(trajectory_cfg.get("knot_time_schedule", "")).strip().lower()
    if schedule not in schedules:
        return None
    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    best_objective, n_evaluations, best_evaluation_index, late_best = _history_stats(history_path)
    seed_raw = bo_cfg.get("random_seed", None)
    theta0_source = str(trajectory_cfg.get("theta0_source", "")).strip()
    continuation_parent_run_dir, continuation_parent_run_name = _theta0_metadata(theta0_source)
    return StudyRun(
        run_dir=path,
        run_name=run_name,
        target_front_speed_mm_s=float(target_cfg.get("target_front_speed_mm_s")),
        schedule=schedule,
        num_knots=num_knots,
        normalized_support_tau=normalized_support_tau,
        knot_times_s=knot_times_s,
        theta_C=_best_theta_from_csv(theta_path),
        initial_theta_C=_parse_theta_tuple(trajectory_cfg.get("theta0_C", [])),
        theta0_source=theta0_source,
        continuation_parent_run_dir=continuation_parent_run_dir,
        continuation_parent_run_name=continuation_parent_run_name,
        objective_value=best_objective,
        achieved_direct_speed_mm_s=float(tracking_row.get("actual_interval_speed_mm_s", "nan")),
        achieved_tc_3to11_mm_s=_interval_speed(tc_path, INTERVAL_3_TO_11_MM),
        tracking_rmse_mm=float(tracking_row.get("tracking_rmse_mm", "nan")),
        t_at_control_z_min_s=float(tracking_row.get("t_at_control_z_min_s", "nan")),
        t_at_control_z_max_s=float(tracking_row.get("t_at_control_z_max_s", "nan")),
        plate_tolerance_C=float(plate_row.get("tolerance_C", "nan")),
        max_abs_plate_error_C=float(plate_row.get("max_abs_plate_error_C", "nan")),
        rmse_plate_error_C=float(plate_row.get("rmse_plate_error_C", "nan")),
        mean_plate_error_C=float(plate_row.get("mean_plate_error_C", "nan")),
        mean_abs_plate_error_C=float(plate_row.get("mean_abs_plate_error_C", "nan")),
        fraction_within_tolerance=float(plate_row.get("fraction_within_tolerance", "nan")),
        n_evaluations=n_evaluations,
        best_evaluation_index=best_evaluation_index,
        late_best_eval=late_best,
        random_seed=None if seed_raw is None else int(seed_raw),
    )


def _scan_bov4_runs(
    root: Path,
    *,
    simulation_config_path: Path,
    schedules: tuple[str, ...],
) -> tuple[StudyRun, ...]:
    summaries: list[StudyRun] = []
    if not root.exists():
        return tuple()
    candidates = [path for path in sorted(root.iterdir()) if path.is_dir() and f"_{RUN_NAME_TAG}_" in path.name]
    for candidate in candidates:
        summary = _load_bov4_run(candidate, simulation_config_path=simulation_config_path, schedules=schedules)
        if summary is not None:
            summaries.append(summary)
    return tuple(summaries)


def _find_run(
    runs: tuple[StudyRun, ...],
    *,
    target_mm_s: float,
    schedule: str,
    seed: int,
) -> StudyRun | None:
    matching = [
        run
        for run in runs
        if _matches_target(run.target_front_speed_mm_s, target_mm_s)
        and run.schedule == schedule
        and run.random_seed == int(seed)
    ]
    if not matching:
        return None
    return min(
        matching,
        key=lambda run: (
            math.inf if not math.isfinite(run.objective_value) else run.objective_value,
            run.run_name,
        ),
    )


def _build_run_rows(
    *,
    runs: tuple[StudyRun, ...],
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    family_lookup: dict[tuple[float, ...], str] = {}
    family_counter = 0
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for schedule in schedules:
            for seed in seeds:
                run = _find_run(runs, target_mm_s=target_mm_s, schedule=schedule, seed=seed)
                speed_success = False
                rel_error_pct = math.nan
                temperature_success = False
                family_id = ""
                history = {
                    "n_total_evaluations": 0,
                    "n_infeasible_evaluations": 0,
                    "infeasible_fraction": math.nan,
                    "best_evaluation_index": 0,
                    "best_evaluation_fraction": math.nan,
                    "late_best_eval": 0,
                }
                if run is not None:
                    history = _history_diagnostics(run.run_dir / "bo_history.csv")
                    speed_success, rel_error_pct = _speed_success(target_mm_s, run.achieved_direct_speed_mm_s)
                    plate_summary_like = type(
                        "_Tmp",
                        (),
                        {"mean_abs_plate_error_C": run.mean_abs_plate_error_C, "tolerance_C": run.plate_tolerance_C},
                    )
                    temperature_success = plate_tracking_success(plate_summary_like)
                    signature = _family_signature(run.theta_C)
                    family_id = family_lookup.get(signature, "")
                    if not family_id:
                        family_counter += 1
                        family_id = f"family_{family_counter}"
                        family_lookup[signature] = family_id
                rows.append(
                    {
                        "status": "completed" if run is not None else "missing",
                        "target_front_speed_mm_s": float(target_mm_s),
                        "schedule": schedule,
                        "seed": int(seed),
                        "selected_run_name": "" if run is None else run.run_name,
                        "theta0_source": "" if run is None else run.theta0_source,
                        "continuation_parent_run_name": "" if run is None else run.continuation_parent_run_name,
                        "continuation_parent_run_dir": "" if run is None else run.continuation_parent_run_dir,
                        "initial_theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.initial_theta_C),
                        "objective_value": math.nan if run is None else run.objective_value,
                        "achieved_direct_speed_mm_s": math.nan if run is None else run.achieved_direct_speed_mm_s,
                        "achieved_tc_3to11_mm_s": math.nan if run is None else run.achieved_tc_3to11_mm_s,
                        "tracking_rmse_mm": math.nan if run is None else run.tracking_rmse_mm,
                        "direct_speed_relative_error_pct": rel_error_pct,
                        "speed_success": int(speed_success),
                        "plate_tolerance_C": math.nan if run is None else run.plate_tolerance_C,
                        "max_abs_plate_error_C": math.nan if run is None else run.max_abs_plate_error_C,
                        "rmse_plate_error_C": math.nan if run is None else run.rmse_plate_error_C,
                        "mean_plate_error_C": math.nan if run is None else run.mean_plate_error_C,
                        "mean_abs_plate_error_C": math.nan if run is None else run.mean_abs_plate_error_C,
                        "fraction_within_tolerance": math.nan if run is None else run.fraction_within_tolerance,
                        "temperature_success": int(temperature_success),
                        "overall_success": int(speed_success and temperature_success),
                        "best_evaluation_index": int(history["best_evaluation_index"]),
                        "n_evaluations": int(history["n_total_evaluations"]),
                        "best_evaluation_fraction": float(history["best_evaluation_fraction"]),
                        "n_infeasible_evaluations": int(history["n_infeasible_evaluations"]),
                        "infeasible_fraction": float(history["infeasible_fraction"]),
                        "late_best_eval": int(history["late_best_eval"]),
                        "failure_mode": (
                            "missing"
                            if run is None
                            else (
                                "recovered"
                                if speed_success and temperature_success
                                else (
                                    "optimizer_not_converged_yet"
                                    if (not speed_success and int(history["late_best_eval"]) == 1)
                                    else ("temperature_tracking_not_ok" if not temperature_success else "missed_target_but_temperature_ok")
                                )
                            )
                        ),
                        "family_id": family_id,
                        "theta_0_C": math.nan if run is None else run.theta_C[0],
                        "theta_1_C": math.nan if run is None else run.theta_C[1],
                        "theta_2_C": math.nan if run is None else run.theta_C[2],
                        "theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.theta_C),
                        "knot_times_s": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.knot_times_s),
                        "normalized_support_tau": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.normalized_support_tau),
                        "has_effective_config": int(run is not None),
                        "has_evaluation_history": int(run is not None),
                        "has_best_tracking_summary": int(run is not None),
                    }
                )
    return rows


def _group_rows(
    rows: list[dict[str, object]],
    *,
    group_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for key in sorted(grouped):
        members = grouped[key]
        target_mm_s = float(members[0]["target_front_speed_mm_s"])
        completed = [row for row in members if row.get("status") == "completed"]
        achieved = [
            float(row["achieved_direct_speed_mm_s"])
            for row in completed
            if math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
        ]
        rel_errors = [
            float(row["direct_speed_relative_error_pct"])
            for row in completed
            if math.isfinite(float(row.get("direct_speed_relative_error_pct", math.nan)))
        ]
        infeasible_fractions = [
            float(row["infeasible_fraction"])
            for row in completed
            if math.isfinite(float(row.get("infeasible_fraction", math.nan)))
        ]
        plate_mae = [
            float(row["mean_abs_plate_error_C"])
            for row in completed
            if math.isfinite(float(row.get("mean_abs_plate_error_C", math.nan)))
        ]
        families = {
            str(row["family_id"])
            for row in completed
            if str(row.get("family_id", "")).strip()
        }
        achieved_spread = math.nan if not achieved else float(max(achieved) - min(achieved))
        plateau_threshold = max(PLATEAU_MIN_ABS_SPREAD_MM_S, PLATEAU_MIN_REL_SPREAD * target_mm_s)
        plateau_split_detected = bool(
            len(families) >= 2
            and math.isfinite(achieved_spread)
            and achieved_spread >= plateau_threshold
        )
        n_speed_success = sum(int(row.get("speed_success", 0)) for row in members)
        n_temperature_success = sum(int(row.get("temperature_success", 0)) for row in members)
        n_overall_success = sum(int(row.get("overall_success", 0)) for row in members)
        num_completed = len(completed)
        speed_success_fraction = math.nan if num_completed == 0 else float(n_speed_success) / float(num_completed)
        temperature_success_fraction = math.nan if num_completed == 0 else float(n_temperature_success) / float(num_completed)
        overall_success_fraction = math.nan if num_completed == 0 else float(n_overall_success) / float(num_completed)
        late_best_fraction = math.nan if num_completed == 0 else float(sum(int(row.get("late_best_eval", 0)) for row in completed)) / float(num_completed)
        median_achieved = math.nan if not achieved else float(np.median(np.asarray(achieved, dtype=np.float64)))
        median_speed_error = math.nan if not rel_errors else float(np.median(np.asarray(rel_errors, dtype=np.float64)))
        median_infeasible_fraction = math.nan if not infeasible_fractions else float(np.median(np.asarray(infeasible_fractions, dtype=np.float64)))
        median_plate_mae = math.nan if not plate_mae else float(np.median(np.asarray(plate_mae, dtype=np.float64)))
        robustly_recovered = bool(
            num_completed >= len(seeds)
            and n_speed_success >= len(seeds)
            and math.isfinite(median_speed_error)
            and median_speed_error <= ROBUST_MAX_MEDIAN_SPEED_ERROR_PCT
            and (not plateau_split_detected)
            and math.isfinite(median_infeasible_fraction)
            and median_infeasible_fraction <= ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION
            and math.isfinite(late_best_fraction)
            and late_best_fraction <= ROBUST_MAX_LATE_BEST_FRACTION
        )
        if num_completed == 0:
            conclusion = ""
        elif robustly_recovered:
            conclusion = "Recovered by tuned n3 BO"
        elif n_speed_success > 0:
            conclusion = "Recovered but not robust"
        elif math.isfinite(median_plate_mae) and median_plate_mae > 0.5 and math.isfinite(median_speed_error) and median_speed_error <= 10.0:
            conclusion = "Not recovered; likely plate/inner-response-limited"
        else:
            conclusion = "Not recovered; likely BO-search-limited"
        aggregate = {name: value for name, value in zip(group_keys, key, strict=True)}
        aggregate.update(
            {
                "num_rows": len(members),
                "num_completed": num_completed,
                "num_speed_success": int(n_speed_success),
                "num_temperature_success": int(n_temperature_success),
                "num_overall_success": int(n_overall_success),
                "speed_success_fraction": speed_success_fraction,
                "temperature_success_fraction": temperature_success_fraction,
                "overall_success_fraction": overall_success_fraction,
                "median_achieved_direct_speed_mm_s": median_achieved,
                "median_direct_speed_relative_error_pct": median_speed_error,
                "median_infeasible_fraction": median_infeasible_fraction,
                "late_best_fraction": late_best_fraction,
                "median_plate_mae_C": median_plate_mae,
                "min_achieved_direct_speed_mm_s": math.nan if not achieved else min(achieved),
                "max_achieved_direct_speed_mm_s": math.nan if not achieved else max(achieved),
                "achieved_speed_spread_mm_s": achieved_spread,
                "num_distinct_theta_families": len(families),
                "plateau_split_detected": int(plateau_split_detected),
                "robustly_recovered": int(robustly_recovered),
                "study_conclusion": conclusion,
            }
        )
        aggregates.append(aggregate)
    return aggregates


def _find_group_aggregate(
    rows: list[dict[str, object]],
    *,
    target_mm_s: float,
    schedule: str,
) -> dict[str, object] | None:
    for row in rows:
        if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s) and str(row.get("schedule")) == schedule:
            return row
    return None


def _load_baseline_aggregates(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline bundle aggregate not found at {path}. "
            "The frozen n3 bo3 bundle is required as the comparison baseline for bov4."
        )
    rows = _read_csv_rows(path)
    baseline_rows: list[dict[str, object]] = []
    for row in rows:
        baseline_rows.append(
            {
                "target_front_speed_mm_s": float(row["target_front_speed_mm_s"]),
                "schedule": str(row["schedule"]),
                "num_completed": int(row["num_completed"]),
                "num_speed_success": int(row["num_speed_success"]),
                "speed_success_fraction": float(row["speed_success_fraction"]),
                "temperature_success_fraction": float(row.get("temperature_success_fraction", math.nan)),
                "overall_success_fraction": float(row.get("overall_success_fraction", math.nan)),
                "median_direct_speed_relative_error_pct": float(row["median_direct_speed_relative_error_pct"]),
                "median_infeasible_fraction": float(row["median_infeasible_fraction"]),
                "late_best_fraction": float(row["late_best_fraction"]),
                "median_plate_mae_C": float(row["median_plate_mae_C"]),
                "median_achieved_direct_speed_mm_s": float(row.get("median_achieved_direct_speed_mm_s", math.nan)),
                "min_achieved_direct_speed_mm_s": float(row["min_achieved_direct_speed_mm_s"]),
                "max_achieved_direct_speed_mm_s": float(row["max_achieved_direct_speed_mm_s"]),
                "achieved_speed_spread_mm_s": float(row["achieved_speed_spread_mm_s"]),
                "study_conclusion": str(row.get("study_conclusion", "")),
            }
        )
    return baseline_rows


def _baseline_comparison_rows(
    *,
    bov4_aggregates: list[dict[str, object]],
    baseline_aggregates: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for schedule in schedules:
            bov4 = _find_group_aggregate(bov4_aggregates, target_mm_s=target_mm_s, schedule=schedule)
            baseline = _find_group_aggregate(baseline_aggregates, target_mm_s=target_mm_s, schedule=schedule)
            baseline_num_speed_success = math.nan if baseline is None else float(baseline["num_speed_success"])
            bov4_num_speed_success = math.nan if bov4 is None else float(bov4["num_speed_success"])
            baseline_error = math.nan if baseline is None else float(baseline["median_direct_speed_relative_error_pct"])
            bov4_error = math.nan if bov4 is None else float(bov4["median_direct_speed_relative_error_pct"])
            baseline_infeasible = math.nan if baseline is None else float(baseline["median_infeasible_fraction"])
            bov4_infeasible = math.nan if bov4 is None else float(bov4["median_infeasible_fraction"])
            baseline_spread = math.nan if baseline is None else float(baseline["achieved_speed_spread_mm_s"])
            bov4_spread = math.nan if bov4 is None else float(bov4["achieved_speed_spread_mm_s"])
            plus_one_success = int(
                math.isfinite(baseline_num_speed_success)
                and math.isfinite(bov4_num_speed_success)
                and bov4_num_speed_success >= baseline_num_speed_success + 1.0
            )
            error_improved_20pct = int(
                math.isfinite(baseline_error)
                and baseline_error > 0.0
                and math.isfinite(bov4_error)
                and bov4_error <= 0.8 * baseline_error
            )
            spread_improved = int(
                math.isfinite(baseline_spread)
                and math.isfinite(bov4_spread)
                and bov4_spread < baseline_spread - 1.0e-12
            )
            infeasible_not_worse = int(
                math.isfinite(baseline_infeasible)
                and math.isfinite(bov4_infeasible)
                and bov4_infeasible <= baseline_infeasible + GATE_B_MAX_INFEASIBLE_WORSEN_ABS + 1.0e-12
            )
            rows.append(
                {
                    "target_front_speed_mm_s": float(target_mm_s),
                    "schedule": schedule,
                    "baseline_available": int(baseline is not None),
                    "baseline_num_completed": 0 if baseline is None else baseline["num_completed"],
                    "baseline_num_speed_success": 0 if baseline is None else baseline["num_speed_success"],
                    "baseline_speed_success_fraction": math.nan if baseline is None else baseline["speed_success_fraction"],
                    "baseline_median_direct_speed_relative_error_pct": baseline_error,
                    "baseline_median_infeasible_fraction": baseline_infeasible,
                    "baseline_late_best_fraction": math.nan if baseline is None else baseline["late_best_fraction"],
                    "baseline_median_plate_mae_C": math.nan if baseline is None else baseline["median_plate_mae_C"],
                    "baseline_median_achieved_direct_speed_mm_s": math.nan if baseline is None else baseline["median_achieved_direct_speed_mm_s"],
                    "baseline_achieved_speed_spread_mm_s": baseline_spread,
                    "bov4_num_completed": 0 if bov4 is None else bov4["num_completed"],
                    "bov4_num_speed_success": 0 if bov4 is None else bov4["num_speed_success"],
                    "bov4_speed_success_fraction": math.nan if bov4 is None else bov4["speed_success_fraction"],
                    "bov4_median_direct_speed_relative_error_pct": bov4_error,
                    "bov4_median_infeasible_fraction": bov4_infeasible,
                    "bov4_late_best_fraction": math.nan if bov4 is None else bov4["late_best_fraction"],
                    "bov4_median_plate_mae_C": math.nan if bov4 is None else bov4["median_plate_mae_C"],
                    "bov4_median_achieved_direct_speed_mm_s": math.nan if bov4 is None else bov4["median_achieved_direct_speed_mm_s"],
                    "bov4_achieved_speed_spread_mm_s": bov4_spread,
                    "delta_num_speed_success": math.nan if (not math.isfinite(baseline_num_speed_success) or not math.isfinite(bov4_num_speed_success)) else bov4_num_speed_success - baseline_num_speed_success,
                    "delta_speed_success_fraction": math.nan if baseline is None or bov4 is None else float(bov4["speed_success_fraction"]) - float(baseline["speed_success_fraction"]),
                    "delta_median_direct_speed_relative_error_pct": math.nan if not math.isfinite(baseline_error) or not math.isfinite(bov4_error) else bov4_error - baseline_error,
                    "delta_median_infeasible_fraction": math.nan if not math.isfinite(baseline_infeasible) or not math.isfinite(bov4_infeasible) else bov4_infeasible - baseline_infeasible,
                    "delta_achieved_speed_spread_mm_s": math.nan if not math.isfinite(baseline_spread) or not math.isfinite(bov4_spread) else bov4_spread - baseline_spread,
                    "infeasible_not_worse_abs_0p05": infeasible_not_worse,
                    "spread_improved_vs_bo3": spread_improved,
                    "plus_one_speed_success_vs_bo3": plus_one_success,
                    "speed_error_improved_20pct_vs_bo3": error_improved_20pct,
                    "gate_b_improvement_hit": int(bool(plus_one_success or error_improved_20pct)),
                }
            )
    return rows


def _plot_target_vs_achieved_by_schedule(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    schedules: tuple[str, ...],
) -> None:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, len(schedules), figsize=(5.4 * len(schedules), 4.8), sharex=True, sharey=True)
    if len(schedules) == 1:
        axes = [axes]
    color_by_seed = {17: "#1f77b4", 29: "#2ca02c", 41: "#d62728"}
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
    ]
    if not completed:
        for ax in axes:
            ax.axis("off")
            ax.text(0.02, 0.95, "No completed bov4 runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    x_values = [float(row["target_front_speed_mm_s"]) for row in completed]
    y_values = [float(row["achieved_direct_speed_mm_s"]) for row in completed]
    axis_max = max(max(x_values), max(y_values)) * 1.03
    for idx, schedule in enumerate(schedules):
        ax = axes[idx]
        schedule_rows = [row for row in completed if str(row.get("schedule")) == schedule]
        diagonal = ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")[0]
        handles = [diagonal]
        labels = ["target = achieved"]
        for seed in sorted({int(row["seed"]) for row in schedule_rows}):
            seed_rows = [row for row in schedule_rows if int(row["seed"]) == seed]
            scatter = ax.scatter(
                [float(row["target_front_speed_mm_s"]) for row in seed_rows],
                [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
                color=color_by_seed.get(seed, "#4c4c4c"),
                s=38,
                label=f"seed {seed}",
            )
            handles.append(scatter)
            labels.append(f"seed {seed}")
        ax.set_title(schedule)
        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.set_xlabel("Target speed (mm/s)")
        if idx == 0:
            ax.set_ylabel("Achieved direct speed (mm/s)")
            ax.legend(handles, labels, loc="upper left")
    fig.suptitle("n3 bov4: target vs achieved direct speed by schedule", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_comparison_by_schedule(
    out_path: Path,
    *,
    comparison_rows: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    baseline_key: str,
    bov4_key: str,
    title: str,
    y_label: str,
) -> None:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, len(schedules), figsize=(5.4 * len(schedules), 4.6), sharex=True, sharey=True)
    if len(schedules) == 1:
        axes = [axes]
    all_values: list[float] = []
    for row in comparison_rows:
        for key in (baseline_key, bov4_key):
            value = float(row.get(key, math.nan))
            if math.isfinite(value):
                all_values.append(value)
    y_max = max(all_values) * 1.08 if all_values else 1.0
    for idx, schedule in enumerate(schedules):
        ax = axes[idx]
        baseline_values = []
        bov4_values = []
        for target_mm_s in targets_mm_s:
            row = next(
                (
                    item
                    for item in comparison_rows
                    if _matches_target(float(item["target_front_speed_mm_s"]), target_mm_s)
                    and str(item.get("schedule")) == schedule
                ),
                None,
            )
            baseline_values.append(math.nan if row is None else float(row.get(baseline_key, math.nan)))
            bov4_values.append(math.nan if row is None else float(row.get(bov4_key, math.nan)))
        ax.plot(targets_mm_s, baseline_values, marker="o", linewidth=2.0, color="#1f77b4", label=BASELINE_NAME)
        ax.plot(targets_mm_s, bov4_values, marker="o", linewidth=2.0, color="#d62728", label=RUN_NAME_TAG)
        ax.set_title(schedule)
        ax.set_xlabel("Target speed (mm/s)")
        ax.set_xlim(min(targets_mm_s), max(targets_mm_s))
        ax.set_ylim(0.0, y_max)
        if idx == 0:
            ax.set_ylabel(y_label)
            ax.legend(loc="upper left")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _upper_plateau_metrics(rows: list[dict[str, object]], *, schedule: str) -> dict[str, float | bool]:
    upper_rows = [
        row
        for row in rows
        if str(row.get("schedule")) == schedule
        and any(_matches_target(float(row["target_front_speed_mm_s"]), target_mm_s) for target_mm_s in UPPER_RANGE_TARGETS_MM_S)
    ]
    target_to_median: dict[float, float] = {}
    spreads: list[float] = []
    for target_mm_s in UPPER_RANGE_TARGETS_MM_S:
        row = next((item for item in upper_rows if _matches_target(float(item["target_front_speed_mm_s"]), target_mm_s)), None)
        if row is None:
            continue
        median_achieved = float(row.get("median_achieved_direct_speed_mm_s", math.nan))
        if math.isfinite(median_achieved):
            target_to_median[float(target_mm_s)] = median_achieved
        spread = float(row.get("achieved_speed_spread_mm_s", math.nan))
        if math.isfinite(spread):
            spreads.append(spread)
    slope_ratio = math.nan
    if 0.010 in target_to_median and 0.013 in target_to_median:
        slope_ratio = (target_to_median[0.013] - target_to_median[0.010]) / (0.013 - 0.010)
    median_spread = math.nan if not spreads else float(np.median(np.asarray(spreads, dtype=np.float64)))
    persistent_plateau = bool(
        math.isfinite(slope_ratio)
        and slope_ratio < UPPER_PLATEAU_MAX_SLOPE_RATIO
        and math.isfinite(median_spread)
        and median_spread <= UPPER_PLATEAU_MAX_MEDIAN_SPREAD_MM_S
    )
    return {
        "upper_slope_ratio": slope_ratio,
        "upper_median_spread_mm_s": median_spread,
        "persistent_upper_plateau": persistent_plateau,
    }


def _plot_upper_range_plateau_assessment(
    out_path: Path,
    *,
    bov4_aggregates: list[dict[str, object]],
    baseline_aggregates: list[dict[str, object]],
    schedules: tuple[str, ...],
) -> None:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, len(schedules), figsize=(5.4 * len(schedules), 4.8), sharex=True, sharey=True)
    if len(schedules) == 1:
        axes = [axes]
    upper_targets = tuple(float(value) for value in UPPER_RANGE_TARGETS_MM_S)
    finite_values: list[float] = list(upper_targets)
    for row in bov4_aggregates + baseline_aggregates:
        value = float(row.get("median_achieved_direct_speed_mm_s", math.nan))
        if math.isfinite(value):
            finite_values.append(value)
    axis_max = max(finite_values) * 1.03 if finite_values else 0.0135
    for idx, schedule in enumerate(schedules):
        ax = axes[idx]
        baseline_values = []
        bov4_values = []
        for target_mm_s in upper_targets:
            baseline_row = _find_group_aggregate(baseline_aggregates, target_mm_s=target_mm_s, schedule=schedule)
            bov4_row = _find_group_aggregate(bov4_aggregates, target_mm_s=target_mm_s, schedule=schedule)
            baseline_values.append(math.nan if baseline_row is None else float(baseline_row.get("median_achieved_direct_speed_mm_s", math.nan)))
            bov4_values.append(math.nan if bov4_row is None else float(bov4_row.get("median_achieved_direct_speed_mm_s", math.nan)))
        ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
        ax.plot(upper_targets, baseline_values, marker="o", linewidth=2.0, color="#1f77b4", label=BASELINE_NAME)
        ax.plot(upper_targets, bov4_values, marker="o", linewidth=2.0, color="#d62728", label=RUN_NAME_TAG)
        baseline_metrics = _upper_plateau_metrics(baseline_aggregates, schedule=schedule)
        bov4_metrics = _upper_plateau_metrics(bov4_aggregates, schedule=schedule)
        annotation = (
            f"{BASELINE_NAME}: slope={baseline_metrics['upper_slope_ratio']:.3f}, "
            f"spread={baseline_metrics['upper_median_spread_mm_s']:.4f}, "
            f"plateau={'yes' if baseline_metrics['persistent_upper_plateau'] else 'no'}\n"
            f"{RUN_NAME_TAG}: slope={bov4_metrics['upper_slope_ratio']:.3f}, "
            f"spread={bov4_metrics['upper_median_spread_mm_s']:.4f}, "
            f"plateau={'yes' if bov4_metrics['persistent_upper_plateau'] else 'no'}"
        )
        ax.text(
            0.03,
            0.97,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        ax.set_title(schedule)
        ax.set_xlim(min(upper_targets) - 0.0002, max(upper_targets) + 0.0002)
        ax.set_ylim(0.0, axis_max)
        ax.set_xlabel("Target speed (mm/s)")
        if idx == 0:
            ax.set_ylabel("Median achieved direct speed (mm/s)")
            ax.legend(loc="upper left")
    fig.suptitle("n3 upper-range plateau assessment: bo3 vs bov4", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _readme_text(*, targets_mm_s: tuple[float, ...], schedules: tuple[str, ...], seeds: tuple[int, ...]) -> str:
    return "\n".join(
        [
            "# n3 bov4 Variability-Reduction Study",
            "",
            "This bundle re-runs the active admissible `n3` target range with a continuation-aware BO meant to reduce seed variability before any further parameterization change.",
            "",
            "## Locked study policy",
            "",
            "- Historical `bo3` in `bo_monotone_feasible_admissible_range` remains frozen and is used only as the comparison baseline.",
            "- `n5` remains frozen and is not part of this bundle.",
            "- Objective unchanged: direct front tracking over `2.5-11.5 mm`.",
            f"- Target-speed range for this bundle: `{_target_range_text(targets_mm_s)} mm/s`.",
            f"- Explicit target grid: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Schedules: `{', '.join(schedules)}`.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            f"- BO settings: `EI`, `xi={ACQUISITION_XI}`, `init_points={INIT_POINTS}`, `n_iter={N_ITER}`, `seed_with_theta0=true`.",
            f"- Parameterization: `{PARAMETERIZATION_KIND}`.",
            f"- Init strategy: `{INIT_STRATEGY}` with `sigma={INIT_LOCAL_SIGMA}` and `max_attempts={INIT_MAX_ATTEMPTS_PER_POINT}`.",
            f"- Local refinement: `{LOCAL_REFINEMENT_POINTS}` points with `sigma={LOCAL_REFINEMENT_SIGMA}`.",
            f"- Canonical first-target theta0: `{CANONICAL_THETA0_C}`.",
            f"- Theta bounds: `{THETA_BOUNDS_C}`.",
            "",
            "## Continuation rule",
            "",
            "- Each `schedule x seed` chain runs low-to-high in target speed.",
            "- `0.004 mm/s` starts from the canonical theta0.",
            "- Each later target tries to start from the previous target run in the same chain.",
            "- If the previous run is unavailable, the runner falls back to the canonical theta0.",
            "",
            "## Metric definitions",
            "",
            "- `speed_success`: relative error of direct front speed versus target, over `2.5-11.5 mm`, threshold `<= 5%`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref|` over the same front-control interval in time, threshold `<= 0.5 C`.",
            "- `overall_success`: both conditions true.",
            "",
            "## Key outputs",
            "",
            "- `study_summary.csv`: one row per `target x schedule x seed` for `_bov4_` runs only.",
            "- `target_schedule_aggregate.csv`: one row per `target x schedule`.",
            "- `target_aggregate.csv`: one row per target aggregated across schedules.",
            "- `baseline_comparison.csv`: `bo3` versus `bov4` deltas and gate checks.",
            "- `decision_summary.csv`: compact decision-oriented target-by-schedule summary.",
            "- `scientific_assessment.md`: short scientific reading of variability and plateau evidence.",
            "- `run_commands.sh`: sequential continuation-aware queue for the current study matrix.",
            "",
        ]
    )


def _scope_note_text(*, targets_mm_s: tuple[float, ...]) -> str:
    return "\n".join(
        [
            "# Scope Note",
            "",
            "- Official admissible velocity interval: `0.0035 -> 0.013 mm/s`.",
            f"- Active BO target grid for this bundle: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            "- This study ingests only `n3` coarse runs whose run name contains `_bov4_` and whose effective config matches the locked bov4 signature.",
            "- The frozen `bo3` bundle is used only as a comparison baseline.",
            "- No `n5`, no custom middle-knot sweep, and no objective change are part of this bundle.",
            "",
        ]
    )


def _run_fieldnames() -> tuple[str, ...]:
    return (
        "status",
        "target_front_speed_mm_s",
        "schedule",
        "seed",
        "selected_run_name",
        "theta0_source",
        "continuation_parent_run_name",
        "continuation_parent_run_dir",
        "initial_theta_C",
        "objective_value",
        "achieved_direct_speed_mm_s",
        "achieved_tc_3to11_mm_s",
        "tracking_rmse_mm",
        "direct_speed_relative_error_pct",
        "speed_success",
        "plate_tolerance_C",
        "max_abs_plate_error_C",
        "rmse_plate_error_C",
        "mean_plate_error_C",
        "mean_abs_plate_error_C",
        "fraction_within_tolerance",
        "temperature_success",
        "overall_success",
        "best_evaluation_index",
        "n_evaluations",
        "best_evaluation_fraction",
        "n_infeasible_evaluations",
        "infeasible_fraction",
        "late_best_eval",
        "failure_mode",
        "family_id",
        "theta_0_C",
        "theta_1_C",
        "theta_2_C",
        "theta_C",
        "knot_times_s",
        "normalized_support_tau",
        "has_effective_config",
        "has_evaluation_history",
        "has_best_tracking_summary",
    )


def _target_schedule_aggregate_fieldnames() -> tuple[str, ...]:
    return (
        "target_front_speed_mm_s",
        "schedule",
        "num_rows",
        "num_completed",
        "num_speed_success",
        "num_temperature_success",
        "num_overall_success",
        "speed_success_fraction",
        "temperature_success_fraction",
        "overall_success_fraction",
        "median_achieved_direct_speed_mm_s",
        "median_direct_speed_relative_error_pct",
        "median_infeasible_fraction",
        "late_best_fraction",
        "median_plate_mae_C",
        "min_achieved_direct_speed_mm_s",
        "max_achieved_direct_speed_mm_s",
        "achieved_speed_spread_mm_s",
        "num_distinct_theta_families",
        "plateau_split_detected",
        "robustly_recovered",
        "study_conclusion",
    )


def _target_aggregate_fieldnames() -> tuple[str, ...]:
    return (
        "target_front_speed_mm_s",
        "num_rows",
        "num_completed",
        "num_speed_success",
        "num_temperature_success",
        "num_overall_success",
        "speed_success_fraction",
        "temperature_success_fraction",
        "overall_success_fraction",
        "median_achieved_direct_speed_mm_s",
        "median_direct_speed_relative_error_pct",
        "median_infeasible_fraction",
        "late_best_fraction",
        "median_plate_mae_C",
        "min_achieved_direct_speed_mm_s",
        "max_achieved_direct_speed_mm_s",
        "achieved_speed_spread_mm_s",
        "num_distinct_theta_families",
        "plateau_split_detected",
        "robustly_recovered",
        "study_conclusion",
    )


def _baseline_comparison_fieldnames() -> tuple[str, ...]:
    return (
        "target_front_speed_mm_s",
        "schedule",
        "baseline_available",
        "baseline_num_completed",
        "baseline_num_speed_success",
        "baseline_speed_success_fraction",
        "baseline_median_direct_speed_relative_error_pct",
        "baseline_median_infeasible_fraction",
        "baseline_late_best_fraction",
        "baseline_median_plate_mae_C",
        "baseline_median_achieved_direct_speed_mm_s",
        "baseline_achieved_speed_spread_mm_s",
        "bov4_num_completed",
        "bov4_num_speed_success",
        "bov4_speed_success_fraction",
        "bov4_median_direct_speed_relative_error_pct",
        "bov4_median_infeasible_fraction",
        "bov4_late_best_fraction",
        "bov4_median_plate_mae_C",
        "bov4_median_achieved_direct_speed_mm_s",
        "bov4_achieved_speed_spread_mm_s",
        "delta_num_speed_success",
        "delta_speed_success_fraction",
        "delta_median_direct_speed_relative_error_pct",
        "delta_median_infeasible_fraction",
        "delta_achieved_speed_spread_mm_s",
        "infeasible_not_worse_abs_0p05",
        "spread_improved_vs_bo3",
        "plus_one_speed_success_vs_bo3",
        "speed_error_improved_20pct_vs_bo3",
        "gate_b_improvement_hit",
    )


def _select_rows(
    rows: list[dict[str, object]],
    *,
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if any(_matches_target(float(row["target_front_speed_mm_s"]), target) for target in targets_mm_s)
        and str(row.get("schedule")) in schedules
    ]
    if seeds is not None:
        seed_set = {int(seed) for seed in seeds}
        selected = [row for row in selected if int(row.get("seed", -1)) in seed_set]
    return selected


def _gate_a_status(rows: list[dict[str, object]], *, targets_mm_s: tuple[float, ...], schedules: tuple[str, ...]) -> dict[str, object]:
    selected = _select_rows(rows, targets_mm_s=targets_mm_s, schedules=schedules, seeds=(17,))
    expected = len(targets_mm_s) * len(schedules)
    completed = [row for row in selected if row.get("status") == "completed"]
    structural_ok = all(
        int(row.get("has_effective_config", 0)) == 1
        and int(row.get("has_evaluation_history", 0)) == 1
        and int(row.get("has_best_tracking_summary", 0)) == 1
        for row in completed
    )
    parent_resolution_ok = all(
        str(row.get("theta0_source", "")).strip() in {"cli", ""} or str(row.get("theta0_source", "")).strip().startswith("from_run:")
        for row in completed
    )
    return {
        "expected": expected,
        "completed": len(completed),
        "passed": bool(len(completed) == expected and structural_ok and parent_resolution_ok),
        "structural_ok": bool(structural_ok),
        "parent_resolution_ok": bool(parent_resolution_ok),
    }


def _gate_b_status(
    comparison_rows: list[dict[str, object]],
    rows: list[dict[str, object]],
    *,
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, object]:
    selected_runs = _select_rows(rows, targets_mm_s=targets_mm_s, schedules=schedules, seeds=seeds)
    expected = len(targets_mm_s) * len(schedules) * len(seeds)
    completed = [row for row in selected_runs if row.get("status") == "completed"]
    selected_comparisons = [
        row
        for row in comparison_rows
        if any(_matches_target(float(row["target_front_speed_mm_s"]), target) for target in targets_mm_s)
        and str(row.get("schedule")) in schedules
    ]
    comparable = [
        row
        for row in selected_comparisons
        if int(row.get("baseline_available", 0)) == 1
        and int(row.get("bov4_num_completed", 0)) >= len(seeds)
    ]
    infeasible_ok = all(int(row.get("infeasible_not_worse_abs_0p05", 0)) == 1 for row in comparable)
    spread_improved_count = sum(int(row.get("spread_improved_vs_bo3", 0)) for row in comparable)
    improvement_hit_count = sum(int(row.get("gate_b_improvement_hit", 0)) for row in comparable)
    return {
        "expected": expected,
        "completed": len(completed),
        "comparable": len(comparable),
        "spread_improved_count": int(spread_improved_count),
        "improvement_hit_count": int(improvement_hit_count),
        "infeasible_ok": bool(infeasible_ok),
        "passed": bool(
            len(completed) == expected
            and len(comparable) == len(targets_mm_s) * len(schedules)
            and infeasible_ok
            and spread_improved_count >= GATE_B_MIN_SPREAD_IMPROVEMENTS
            and improvement_hit_count >= GATE_B_MIN_IMPROVEMENT_HITS
        ),
    }


def _overall_classification(
    *,
    gate_b: dict[str, object],
    bov4_aggregates: list[dict[str, object]],
    schedules: tuple[str, ...],
) -> str:
    plateau_count = sum(
        int(bool(_upper_plateau_metrics(bov4_aggregates, schedule=schedule)["persistent_upper_plateau"]))
        for schedule in schedules
    )
    plate_limited_count = sum(
        1
        for row in bov4_aggregates
        if str(row.get("study_conclusion", "")) == "Not recovered; likely plate/inner-response-limited"
    )
    if not bool(gate_b.get("passed", False)):
        return "still BO-limited"
    if plateau_count >= 2:
        return "likely parameterization-limited (n3 fixed-time)"
    if plate_limited_count >= len(schedules):
        return "likely plate/inner-response-limited"
    return "still BO-limited"


def _scientific_assessment_text(
    *,
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    seeds: tuple[int, ...],
    study_rows: list[dict[str, object]],
    target_schedule_aggregate: list[dict[str, object]],
    baseline_comparison: list[dict[str, object]],
) -> str:
    total_expected = len(targets_mm_s) * len(schedules) * len(seeds)
    total_completed = sum(1 for row in study_rows if row.get("status") == "completed")
    gate_a = _gate_a_status(study_rows, targets_mm_s=GATE_TARGETS_MM_S, schedules=schedules)
    gate_b = _gate_b_status(
        baseline_comparison,
        study_rows,
        targets_mm_s=GATE_TARGETS_MM_S,
        schedules=schedules,
        seeds=seeds,
    )
    overall_class = _overall_classification(
        gate_b=gate_b,
        bov4_aggregates=target_schedule_aggregate,
        schedules=schedules,
    )
    comparable_rows = [
        row
        for row in baseline_comparison
        if int(row.get("baseline_available", 0)) == 1
        and int(row.get("bov4_num_completed", 0)) >= len(seeds)
    ]
    spread_improved_count = sum(int(row.get("spread_improved_vs_bo3", 0)) for row in comparable_rows)
    plus_one_success_count = sum(int(row.get("plus_one_speed_success_vs_bo3", 0)) for row in comparable_rows)
    error20pct_count = sum(int(row.get("speed_error_improved_20pct_vs_bo3", 0)) for row in comparable_rows)
    upper_lines: list[str] = []
    for schedule in schedules:
        metrics = _upper_plateau_metrics(target_schedule_aggregate, schedule=schedule)
        upper_lines.append(
            f"- `{schedule}`: `upper_slope_ratio={metrics['upper_slope_ratio']:.3f}`, "
            f"`upper_median_spread_mm_s={metrics['upper_median_spread_mm_s']:.6f}`, "
            f"`persistent_upper_plateau={'yes' if metrics['persistent_upper_plateau'] else 'no'}`."
        )
    return "\n".join(
        [
            "# Scientific Assessment",
            "",
            "## Current campaign state",
            "",
            f"- Completed `_bov4_` rows: `{total_completed}/{total_expected}`.",
            f"- Gate A status: `{'PASS' if gate_a['passed'] else 'FAIL'}` with `{gate_a['completed']}/{gate_a['expected']}` completed rows.",
            f"- Gate B status: `{'PASS' if gate_b['passed'] else 'FAIL'}` with `{gate_b['completed']}/{gate_b['expected']}` completed rows.",
            "",
            "## 1. Did seed variability go down?",
            "",
            f"- Comparable `target x schedule` groups versus frozen `bo3`: `{len(comparable_rows)}`.",
            f"- Groups where achieved-speed spread decreased: `{spread_improved_count}`.",
            f"- Groups with `+1` or more speed-success seeds versus `bo3`: `{plus_one_success_count}`.",
            f"- Groups with at least `20%` lower median direct-speed error versus `bo3`: `{error20pct_count}`.",
            (
                f"- Answer: `{'yes, materially' if bool(gate_b['passed']) else 'not yet materially'}`."
            ),
            "",
            "## 2. Does the upper plateau persist in `0.010 -> 0.013 mm/s`?",
            "",
            *upper_lines,
            "",
            "## 3. What does the current evidence point to?",
            "",
            f"- Overall class: `{overall_class}`.",
            "- This class is restricted to the allowed study vocabulary and is not a stronger system-limit claim.",
            "",
            "## 4. Why is a stronger system-limit claim still not legitimate?",
            "",
            "- The current evidence still comes from `n3` with fixed knot times.",
            "- We have not yet shown the same upper-range plateau under a richer timing parameterization on top of the reduced-variability BO.",
            "- Therefore the current plateau can support `likely parameterization-limited (n3 fixed-time)`, but not a stronger system-limit claim.",
            "",
            "## Gate reading",
            "",
            f"- Gate A requires `12/12` structural completions; current status is `{'PASS' if gate_a['passed'] else 'FAIL'}`.",
            (
                f"- Gate B requires no infeasible worsening beyond `{GATE_B_MAX_INFEASIBLE_WORSEN_ABS:.2f}`, "
                f"`{GATE_B_MIN_SPREAD_IMPROVEMENTS}` spread improvements, and "
                f"`{GATE_B_MIN_IMPROVEMENT_HITS}` improvement hits; current status is `{'PASS' if gate_b['passed'] else 'FAIL'}`."
            ),
            "",
        ]
    )


def _render_campaign_script(
    *,
    title: str,
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    seeds: tuple[int, ...],
    bo_runs_root: Path,
    output_root: Path,
    baseline_bundle_root: Path,
    runner_root: Path,
    max_jobs: int,
    summary_log_name: str,
    progress_log_name: str,
    summarize: bool,
) -> str:
    repo_root = project_root()
    targets_text = " ".join(f"{value:.3f}" for value in targets_mm_s)
    seeds_text = " ".join(str(seed) for seed in seeds)
    schedules_text = " ".join(schedules)
    summarize_cmd = ""
    if summarize:
        summarize_cmd = (
            f"cd \"{repo_root}\"\n"
            "python -m code_simulation.studies.run_n3_bo4_variability_admissible_range \\\n"
            f"  --bo-runs-root \"{bo_runs_root}\" \\\n"
            f"  --baseline-bundle-root \"{baseline_bundle_root}\" \\\n"
            f"  --output-root \"{output_root}\" \\\n"
            f"  --runner-root \"{runner_root}\" \\\n"
            f"  --targets-mm-s \"{','.join(f'{value:.3f}' for value in targets_mm_s)}\" \\\n"
            f"  --schedules \"{','.join(schedules)}\" \\\n"
            f"  --seeds \"{','.join(str(seed) for seed in seeds)}\" \\\n"
            "  --overwrite > \"$SUMMARY_LOG\" 2>&1\n"
            "echo \"study_complete $(date -u +%FT%TZ)\" >> \"$PROGRESS_LOG\"\n"
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"# {title}",
            f"REPO=\"{repo_root}\"",
            f"COARSE=\"{bo_runs_root}\"",
            f"DIAG=\"{output_root}\"",
            f"RUNNER=\"{runner_root}\"",
            "LOGDIR=\"$RUNNER/logs\"",
            "mkdir -p \"$LOGDIR\"",
            f"PROGRESS_LOG=\"$LOGDIR/{progress_log_name}\"",
            f"SUMMARY_LOG=\"$LOGDIR/{summary_log_name}\"",
            "",
            "run_one() {",
            "  local target=\"$1\"",
            "  local schedule=\"$2\"",
            "  local seed=\"$3\"",
            "  local prev_run_dir=\"$4\"",
            "  local tag=${target/./p}",
            f"  local run_name=\"bo_v${{tag}}_n3_${{schedule}}_{RUN_NAME_TAG}_seed${{seed}}\"",
            "  local run_dir=\"$COARSE/$run_name\"",
            "  local log=\"$LOGDIR/${run_name}.log\"",
            "",
            "  if [ -f \"$run_dir/best_tracking_summary.csv\" ]; then",
            "    echo \"skip_existing $(date -u +%FT%TZ) $run_name\" >> \"$PROGRESS_LOG\"",
            "    return 0",
            "  fi",
            "",
            "  local theta_args=(--theta0-c=" + _theta0_text() + ")",
            "  local parent_note=\"theta0\"",
            "  if [ -n \"$prev_run_dir\" ] && [ -f \"$prev_run_dir/best_tracking_summary.csv\" ]; then",
            "    theta_args=(--theta0-from-run-dir \"$prev_run_dir\")",
            "    parent_note=\"$(basename \"$prev_run_dir\")\"",
            "  fi",
            "",
            "  echo \"start $(date -u +%FT%TZ) $run_name parent=$parent_note\" >> \"$PROGRESS_LOG\"",
            "  cd \"$REPO\"",
            "  python -m code_simulation.optimization.run_velocity_control_bo \\",
            "    --output-root \"$REPO/code_simulation/results/active\" \\",
            "    --target-front-speed-mm-s \"$target\" \\",
            "    --num-knots 3 \\",
            "    --knot-time-schedule \"$schedule\" \\",
            "    \"${theta_args[@]}\" \\",
            f"    --theta-bounds={_theta_bounds_text()} \\",
            "    --seed \"$seed\" \\",
            "    --simulation-profile optimization \\",
            f"    --init-points {INIT_POINTS} \\",
            f"    --n-iter {N_ITER} \\",
            f"    --acquisition-kind {ACQUISITION_KIND} \\",
            f"    --acquisition-xi {ACQUISITION_XI:.15g} \\",
            f"    --parameterization-kind {PARAMETERIZATION_KIND} \\",
            f"    --init-strategy {INIT_STRATEGY} \\",
            f"    --init-local-sigma {INIT_LOCAL_SIGMA:.15g} \\",
            f"    --init-max-attempts-per-point {INIT_MAX_ATTEMPTS_PER_POINT} \\",
            f"    --local-refinement-points {LOCAL_REFINEMENT_POINTS} \\",
            f"    --local-refinement-sigma {LOCAL_REFINEMENT_SIGMA:.15g} \\",
            "    --run-name \"$run_name\" \\",
            "    --overwrite > \"$log\" 2>&1",
            "  local status=$?",
            "  if [ \"$status\" -eq 0 ]; then",
            "    echo \"done $(date -u +%FT%TZ) $run_name parent=$parent_note\" >> \"$PROGRESS_LOG\"",
            "  else",
            "    echo \"failed $(date -u +%FT%TZ) $run_name exit=$status parent=$parent_note\" >> \"$PROGRESS_LOG\"",
            "  fi",
            "  return 0",
            "}",
            "",
            "run_chain() {",
            "  local schedule=\"$1\"",
            "  local seed=\"$2\"",
            "  shift 2",
            "  local prev_run_dir=\"\"",
            "  local target=\"\"",
            "  for target in \"$@\"; do",
            "    run_one \"$target\" \"$schedule\" \"$seed\" \"$prev_run_dir\"",
            "    local tag=${target/./p}",
            f"    local run_dir=\"$COARSE/bo_v${{tag}}_n3_${{schedule}}_{RUN_NAME_TAG}_seed${{seed}}\"",
            "    if [ -f \"$run_dir/best_tracking_summary.csv\" ]; then",
            "      prev_run_dir=\"$run_dir\"",
            "    else",
            "      prev_run_dir=\"\"",
            "    fi",
            "  done",
            "}",
            "",
            f"max_jobs={int(max_jobs)}",
            f"targets=({targets_text})",
            f"schedules=({schedules_text})",
            f"seeds=({seeds_text})",
            "",
            "for schedule in \"${schedules[@]}\"; do",
            "  for seed in \"${seeds[@]}\"; do",
            "    run_chain \"$schedule\" \"$seed\" \"${targets[@]}\" &",
            "    while [ \"$(jobs -rp | wc -l)\" -ge \"$max_jobs\" ]; do",
            "      wait -n || true",
            "    done",
            "  done",
            "done",
            "wait || true",
            "",
            summarize_cmd.rstrip(),
            "",
        ]
    ).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the n3 bov4 variability-reduction study.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--baseline-bundle-root", type=Path, default=DEFAULT_BASELINE_BUNDLE_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--bo-config", type=Path, default=DEFAULT_BO_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-root", type=Path, default=DEFAULT_RUNNER_ROOT)
    parser.add_argument("--targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_TARGETS_MM_S))
    parser.add_argument("--schedules", default=",".join(DEFAULT_SCHEDULES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_mm_s = _parse_float_list(args.targets_mm_s)
    schedules = tuple(part.strip() for part in str(args.schedules).split(",") if part.strip())
    seeds = _parse_int_list(args.seeds)
    if schedules != DEFAULT_SCHEDULES:
        raise ValueError(f"this study is locked to schedules={DEFAULT_SCHEDULES!r}")
    bo_config = load_bo_config(args.bo_config)
    if not bool(bo_config.seed_with_theta0):
        raise ValueError(
            "The active bo.toml has seed_with_theta0=false, but bov4 is locked to seed_with_theta0=true. "
            "Update bo.toml or the study assumptions first."
        )

    bov4_runs = _scan_bov4_runs(
        Path(args.bo_runs_root),
        simulation_config_path=Path(args.simulation_config),
        schedules=schedules,
    )
    baseline_aggregates = _load_baseline_aggregates(
        _baseline_bundle_target_schedule_csv(Path(args.baseline_bundle_root))
    )
    study_rows = _build_run_rows(
        runs=bov4_runs,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        seeds=seeds,
    )
    target_schedule_aggregate = _group_rows(
        study_rows,
        group_keys=("target_front_speed_mm_s", "schedule"),
        seeds=seeds,
    )
    target_aggregate = _group_rows(
        study_rows,
        group_keys=("target_front_speed_mm_s",),
        seeds=seeds,
    )
    baseline_comparison = _baseline_comparison_rows(
        bov4_aggregates=target_schedule_aggregate,
        baseline_aggregates=baseline_aggregates,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
    )

    if args.dry_run:
        completed_rows = sum(1 for row in study_rows if row["status"] == "completed")
        expected_rows = len(targets_mm_s) * len(schedules) * len(seeds)
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Runner root: {Path(args.runner_root).resolve()}")
        print(f"Found bov4 runs: {len(bov4_runs)}")
        print(f"Completed bov4 rows: {completed_rows}/{expected_rows}")
        print(f"Baseline comparison rows: {len(baseline_comparison)}")
        print("")
        print("Representative dry-run command:")
        print(_representative_dry_run_command())
        return

    output_root = Path(args.output_root)
    runner_root = Path(args.runner_root)
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))
    runner_root.mkdir(parents=True, exist_ok=True)
    (runner_root / "logs").mkdir(parents=True, exist_ok=True)

    _write_text(output_root / "README.md", _readme_text(targets_mm_s=targets_mm_s, schedules=schedules, seeds=seeds))
    _write_text(output_root / "scope_note.md", _scope_note_text(targets_mm_s=targets_mm_s))
    _write_csv(output_root / "study_summary.csv", _run_fieldnames(), study_rows)
    _write_csv(output_root / "target_schedule_aggregate.csv", _target_schedule_aggregate_fieldnames(), target_schedule_aggregate)
    _write_csv(output_root / "target_aggregate.csv", _target_aggregate_fieldnames(), target_aggregate)
    _write_csv(output_root / "baseline_comparison.csv", _baseline_comparison_fieldnames(), baseline_comparison)
    _write_csv(
        output_root / "decision_summary.csv",
        (
            "target_front_speed_mm_s",
            "schedule",
            "num_completed",
            "num_speed_success",
            "speed_success_fraction",
            "median_direct_speed_relative_error_pct",
            "median_infeasible_fraction",
            "late_best_fraction",
            "median_plate_mae_C",
            "achieved_speed_spread_mm_s",
            "baseline_num_speed_success",
            "baseline_speed_success_fraction",
            "baseline_median_direct_speed_relative_error_pct",
            "baseline_median_infeasible_fraction",
            "baseline_achieved_speed_spread_mm_s",
            "infeasible_not_worse_abs_0p05",
            "spread_improved_vs_bo3",
            "plus_one_speed_success_vs_bo3",
            "speed_error_improved_20pct_vs_bo3",
            "gate_b_improvement_hit",
            "study_conclusion",
        ),
        [
            {
                "target_front_speed_mm_s": row["target_front_speed_mm_s"],
                "schedule": row["schedule"],
                "num_completed": row["bov4_num_completed"],
                "num_speed_success": row["bov4_num_speed_success"],
                "speed_success_fraction": row["bov4_speed_success_fraction"],
                "median_direct_speed_relative_error_pct": row["bov4_median_direct_speed_relative_error_pct"],
                "median_infeasible_fraction": row["bov4_median_infeasible_fraction"],
                "late_best_fraction": row["bov4_late_best_fraction"],
                "median_plate_mae_C": row["bov4_median_plate_mae_C"],
                "achieved_speed_spread_mm_s": row["bov4_achieved_speed_spread_mm_s"],
                "baseline_num_speed_success": row["baseline_num_speed_success"],
                "baseline_speed_success_fraction": row["baseline_speed_success_fraction"],
                "baseline_median_direct_speed_relative_error_pct": row["baseline_median_direct_speed_relative_error_pct"],
                "baseline_median_infeasible_fraction": row["baseline_median_infeasible_fraction"],
                "baseline_achieved_speed_spread_mm_s": row["baseline_achieved_speed_spread_mm_s"],
                "infeasible_not_worse_abs_0p05": row["infeasible_not_worse_abs_0p05"],
                "spread_improved_vs_bo3": row["spread_improved_vs_bo3"],
                "plus_one_speed_success_vs_bo3": row["plus_one_speed_success_vs_bo3"],
                "speed_error_improved_20pct_vs_bo3": row["speed_error_improved_20pct_vs_bo3"],
                "gate_b_improvement_hit": row["gate_b_improvement_hit"],
                "study_conclusion": (
                    _find_group_aggregate(
                        target_schedule_aggregate,
                        target_mm_s=float(row["target_front_speed_mm_s"]),
                        schedule=str(row["schedule"]),
                    ) or {}
                ).get("study_conclusion", ""),
            }
            for row in baseline_comparison
        ],
    )
    _write_text(
        output_root / "scientific_assessment.md",
        _scientific_assessment_text(
            targets_mm_s=targets_mm_s,
            schedules=schedules,
            seeds=seeds,
            study_rows=study_rows,
            target_schedule_aggregate=target_schedule_aggregate,
            baseline_comparison=baseline_comparison,
        ),
    )

    _plot_target_vs_achieved_by_schedule(
        output_root / "target_vs_achieved_direct_speed_by_schedule.png",
        rows=study_rows,
        schedules=schedules,
    )
    _plot_metric_comparison_by_schedule(
        output_root / "bov3_vs_bov4_success_fraction.png",
        comparison_rows=baseline_comparison,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        baseline_key="baseline_speed_success_fraction",
        bov4_key="bov4_speed_success_fraction",
        title="n3 bo3 vs bov4: speed-success fraction by target and schedule",
        y_label="Speed-success fraction across seeds",
    )
    _plot_metric_comparison_by_schedule(
        output_root / "bov3_vs_bov4_seed_spread.png",
        comparison_rows=baseline_comparison,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        baseline_key="baseline_achieved_speed_spread_mm_s",
        bov4_key="bov4_achieved_speed_spread_mm_s",
        title="n3 bo3 vs bov4: achieved-speed spread by target and schedule",
        y_label="Achieved direct-speed spread across seeds (mm/s)",
    )
    _plot_metric_comparison_by_schedule(
        output_root / "bov3_vs_bov4_median_speed_error.png",
        comparison_rows=baseline_comparison,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        baseline_key="baseline_median_direct_speed_relative_error_pct",
        bov4_key="bov4_median_direct_speed_relative_error_pct",
        title="n3 bo3 vs bov4: median direct-speed error by target and schedule",
        y_label="Median direct-speed relative error (%)",
    )
    _plot_upper_range_plateau_assessment(
        output_root / "upper_range_plateau_assessment.png",
        bov4_aggregates=target_schedule_aggregate,
        baseline_aggregates=baseline_aggregates,
        schedules=schedules,
    )

    run_commands_text = _render_campaign_script(
        title="Sequential continuation-aware queue for the current bov4 study matrix.",
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        seeds=seeds,
        bo_runs_root=Path(args.bo_runs_root),
        output_root=output_root,
        baseline_bundle_root=Path(args.baseline_bundle_root),
        runner_root=runner_root,
        max_jobs=1,
        summary_log_name="summary_current.log",
        progress_log_name="progress_current.log",
        summarize=False,
    )
    _write_text(output_root / "run_commands.sh", run_commands_text)

    _write_text(
        runner_root / "run_n3_bov4_gate_a.sh",
        _render_campaign_script(
            title="Gate A structural pass for n3 bov4.",
            targets_mm_s=GATE_TARGETS_MM_S,
            schedules=DEFAULT_SCHEDULES,
            seeds=(17,),
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            baseline_bundle_root=Path(args.baseline_bundle_root),
            runner_root=runner_root,
            max_jobs=2,
            summary_log_name="summary_gate_a.log",
            progress_log_name="progress_gate_a.log",
            summarize=True,
        ),
    )
    _write_text(
        runner_root / "run_n3_bov4_gate_b.sh",
        _render_campaign_script(
            title="Gate B mini scientific pass for n3 bov4.",
            targets_mm_s=GATE_TARGETS_MM_S,
            schedules=DEFAULT_SCHEDULES,
            seeds=DEFAULT_SEEDS,
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            baseline_bundle_root=Path(args.baseline_bundle_root),
            runner_root=runner_root,
            max_jobs=2,
            summary_log_name="summary_gate_b.log",
            progress_log_name="progress_gate_b.log",
            summarize=True,
        ),
    )
    _write_text(
        runner_root / "run_n3_bov4_full_campaign.sh",
        _render_campaign_script(
            title="Full admissible-range n3 bov4 campaign.",
            targets_mm_s=DEFAULT_TARGETS_MM_S,
            schedules=DEFAULT_SCHEDULES,
            seeds=DEFAULT_SEEDS,
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            baseline_bundle_root=Path(args.baseline_bundle_root),
            runner_root=runner_root,
            max_jobs=2,
            summary_log_name="summary_full.log",
            progress_log_name="progress_full.log",
            summarize=True,
        ),
    )
    _write_text(
        runner_root / "representative_dry_run_command.sh",
        "#!/usr/bin/env bash\nset -e\n\n" + _representative_dry_run_command() + "\n",
    )

    print(f"n3 bov4 variability study written to {output_root.resolve()}")
    print(f"  per-run summary    : {(output_root / 'study_summary.csv').resolve()}")
    print(f"  target x schedule  : {(output_root / 'target_schedule_aggregate.csv').resolve()}")
    print(f"  baseline comparison: {(output_root / 'baseline_comparison.csv').resolve()}")
    print(f"  runner root        : {runner_root.resolve()}")


if __name__ == "__main__":
    main()
