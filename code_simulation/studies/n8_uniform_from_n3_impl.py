#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from code_simulation.core.config_files import DEFAULT_BO_CONFIG_PATH, DEFAULT_SIMULATION_CONFIG_PATH, load_bo_config
from code_simulation.core.paths import active_knot_dir, project_root, velocity_control_results_dir
from code_simulation.core.plate_tracking import plate_tracking_success
from code_simulation.studies.bo_3knot_target_and_spacing_study_impl import (
    INTERVAL_3_TO_11_MM,
    RunMetrics,
    THETA_FAMILY_ROUND_DECIMALS,
    _best_theta_from_csv,
    _ensure_bo_plate_tracking_artifacts,
    _ensure_clean_directory,
    _format_speed_tag,
    _history_stats,
    _infer_schedule_name,
    _interval_speed,
    _matches_target,
    _normalized_support_tau_from_knot_times,
    _parse_float_list,
    _parse_int_list,
    _read_csv_rows,
    _read_toml,
    _scan_runs,
    _single_row_csv,
    _speed_success,
    _write_csv,
    _write_shell_script,
    _write_text,
)


DEFAULT_OUTPUT_ROOT = active_knot_dir(8, "diagnostico", "uniform_n3_anchor_admissible_range")
DEFAULT_BO_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.013 + 1.0e-12, 0.001))
DEFAULT_SCHEDULES = ("uniform",)
DEFAULT_SEEDS = (17, 29, 41, 53, 67)
STUDY_NUM_KNOTS = 8
RUN_NAME_TAG = "n3anchor"
N3_ANCHOR_TARGET_MM_S = 0.008
N3_ANCHOR_THETA_C = (
    -4.339973144034226,
    -11.688371309895278,
    -18.04033737488003,
)
N3_ANCHOR_THETA_BOUNDS_C = (
    (-10.0, 0.0),
    (-16.0, -6.0),
    (-21.0, -10.0),
)
TUNED_THETA0_C = (
    -4.339973144034226,
    -6.439515477137384,
    -8.53905781024054,
    -10.638600143343697,
    -12.595795033464528,
    -14.410642480603028,
    -16.225489927741527,
    -18.04033737488003,
)
TUNED_THETA_BOUNDS_C = (
    (-10.0, 0.0),
    (-11.714285714285714, -1.7142857142857142),
    (-13.428571428571429, -3.428571428571429),
    (-15.142857142857142, -5.142857142857142),
    (-16.714285714285715, -6.571428571428571),
    (-18.142857142857142, -7.7142857142857135),
    (-19.57142857142857, -8.857142857142858),
    (-21.0, -10.0),
)
TUNED_ACQUISITION_KIND = "ei"
TUNED_ACQUISITION_XI = 0.01
TUNED_INIT_POINTS = 2
TUNED_N_ITER = 30
ROBUST_MAX_MEDIAN_SPEED_ERROR_PCT = 5.0
ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION = 0.35
ROBUST_MAX_LATE_BEST_FRACTION = 0.4
PLATEAU_MIN_ABS_SPREAD_MM_S = 0.001
PLATEAU_MIN_REL_SPREAD = 0.10


def _theta0_text() -> str:
    return ",".join(f"{float(value):.15g}" for value in TUNED_THETA0_C)


def _theta_bounds_text() -> str:
    return ",".join(f"{lower:.15g}:{upper:.15g}" for lower, upper in TUNED_THETA_BOUNDS_C)


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _target_command(run_name: str, target_mm_s: float, seed: int) -> str:
    repo_root = project_root()
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.optimization.run_velocity_control_bo \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        f"  --num-knots {STUDY_NUM_KNOTS} \\\n"
        "  --knot-time-schedule uniform \\\n"
        f"  --theta0-c={_theta0_text()} \\\n"
        f"  --theta-bounds={_theta_bounds_text()} \\\n"
        f"  --seed {int(seed)} \\\n"
        "  --simulation-profile optimization \\\n"
        f"  --init-points {TUNED_INIT_POINTS} \\\n"
        f"  --n-iter {TUNED_N_ITER} \\\n"
        f"  --acquisition-kind {TUNED_ACQUISITION_KIND} \\\n"
        f"  --acquisition-xi {TUNED_ACQUISITION_XI:.15g} \\\n"
        "  --parameterization-kind monotone_unit_box \\\n"
        "  --init-strategy feasible_local \\\n"
        "  --init-local-sigma 0.15 \\\n"
        "  --init-max-attempts-per-point 40 \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _fine_confirmation_command(
    *,
    target_mm_s: float,
    theta_C: tuple[float, ...],
    run_name: str,
) -> str:
    repo_root = project_root()
    theta_text = ",".join(f"{float(value):.15g}" for value in theta_C)
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.verification.run_velocity_control_evaluation \\\n"
        "  --simulation-profile full_process_article \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        f"  --theta-c={theta_text} \\\n"
        f"  --num-knots {STUDY_NUM_KNOTS} \\\n"
        "  --knot-time-schedule uniform \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _representative_dry_run_command() -> str:
    return _target_command(
        run_name=f"bo_v{_format_speed_tag(N3_ANCHOR_TARGET_MM_S)}_n8_uniform_{RUN_NAME_TAG}_seed17",
        target_mm_s=N3_ANCHOR_TARGET_MM_S,
        seed=17,
    ).replace("--overwrite", "--dry-run-config")


def _matches_tuned_bounds(raw_bounds) -> bool:
    if not isinstance(raw_bounds, list) or len(raw_bounds) != len(TUNED_THETA_BOUNDS_C):
        return False
    for raw_pair, tuned_pair in zip(raw_bounds, TUNED_THETA_BOUNDS_C, strict=True):
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            return False
        if abs(float(raw_pair[0]) - float(tuned_pair[0])) > 1.0e-12:
            return False
        if abs(float(raw_pair[1]) - float(tuned_pair[1])) > 1.0e-12:
            return False
    return True


def _matches_tuned_signature(config: dict) -> bool:
    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    if int(trajectory_cfg.get("num_knots", 0)) != STUDY_NUM_KNOTS:
        return False
    if str(trajectory_cfg.get("knot_time_schedule", "")).strip().lower() != "uniform":
        return False
    if not _matches_tuned_bounds(trajectory_cfg.get("theta_bounds_C", [])):
        return False
    if str(run_cfg.get("simulation_profile", "")).strip() != "optimization":
        return False
    if str(bo_cfg.get("acquisition_kind", "")).strip().lower() != TUNED_ACQUISITION_KIND:
        return False
    if abs(float(bo_cfg.get("acquisition_xi", math.nan)) - TUNED_ACQUISITION_XI) > 1.0e-12:
        return False
    if int(bo_cfg.get("init_points", -1)) != TUNED_INIT_POINTS:
        return False
    if int(bo_cfg.get("n_iter", -1)) != TUNED_N_ITER:
        return False
    if not bool(bo_cfg.get("seed_with_theta0", False)):
        return False
    if str(bo_cfg.get("parameterization_kind", "")).strip().lower() != "monotone_unit_box":
        return False
    if str(bo_cfg.get("init_strategy", "")).strip().lower() != "feasible_local":
        return False
    if abs(float(bo_cfg.get("init_local_sigma", math.nan)) - 0.15) > 1.0e-12:
        return False
    if int(bo_cfg.get("init_max_attempts_per_point", -1)) != 40:
        return False
    return True


def _history_diagnostics(path: Path) -> dict[str, float | int]:
    rows = _read_csv_rows(path)
    n_total = len(rows)
    n_infeasible = sum(
        1
        for row in rows
        if str(row.get("is_valid", "")).strip() not in {"1", "true", "True", "TRUE"}
    )
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


def _load_bo_run(path: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> RunMetrics | None:
    _ensure_bo_plate_tracking_artifacts(path, simulation_config_path=simulation_config_path, schedules=schedules)
    effective_path = path / "effective_config.toml"
    tracking_path = path / "best_tracking_summary.csv"
    theta_path = path / "best_theta_profile.csv"
    history_path = path / "bo_history.csv"
    plate_path = path / "plate_tracking_summary.csv"
    tc_path = path / "best_thermocouple_interval_speeds.csv"
    if not (
        effective_path.exists()
        and tracking_path.exists()
        and theta_path.exists()
        and history_path.exists()
        and plate_path.exists()
    ):
        return None

    config = _read_toml(effective_path)
    if not _matches_tuned_signature(config):
        return None

    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    target_cfg = dict(config.get("velocity_target", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    run_name = str(run_cfg.get("run_name", path.name))
    if f"_{RUN_NAME_TAG}_" not in run_name:
        return None

    knot_times_s = tuple(float(value) for value in trajectory_cfg.get("knot_times_s", []))
    normalized_support_tau = _normalized_support_tau_from_knot_times(
        knot_times_s=knot_times_s,
        horizon_s=float(trajectory_cfg.get("horizon_s", 0.0)),
    )
    schedule = str(trajectory_cfg.get("knot_time_schedule", "")).strip().lower()
    if not schedule:
        schedule = _infer_schedule_name(
            normalized_support_tau=normalized_support_tau,
            num_knots=STUDY_NUM_KNOTS,
            schedules=schedules,
        )
    if schedule != "uniform":
        return None

    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    best_objective, n_evaluations, best_evaluation_index, late_best = _history_stats(history_path)
    seed_raw = bo_cfg.get("random_seed", None)
    return RunMetrics(
        run_dir=path,
        run_name=run_name,
        simulation_profile="optimization",
        target_front_speed_mm_s=float(target_cfg.get("target_front_speed_mm_s")),
        schedule=schedule,
        num_knots=STUDY_NUM_KNOTS,
        normalized_support_tau=normalized_support_tau,
        knot_times_s=knot_times_s,
        theta_C=_best_theta_from_csv(theta_path),
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


def _find_bo_run(
    runs: tuple[RunMetrics, ...],
    *,
    target_mm_s: float,
    schedule: str,
    seed: int,
) -> RunMetrics | None:
    matching = [
        run
        for run in runs
        if run.schedule == schedule
        and run.random_seed == int(seed)
        and _matches_target(run.target_front_speed_mm_s, target_mm_s)
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


def _run_fieldnames() -> tuple[str, ...]:
    base = (
        "status",
        "target_front_speed_mm_s",
        "schedule",
        "seed",
        "selected_run_name",
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
    )
    theta_fields = tuple(f"theta_{idx}_C" for idx in range(STUDY_NUM_KNOTS))
    return base + theta_fields + ("theta_C", "knot_times_s")


def _build_run_rows(
    *,
    runs: tuple[RunMetrics, ...],
    targets_mm_s: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    family_lookup: dict[tuple[float, ...], str] = {}
    family_counter = 0
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for seed in seeds:
            run = _find_bo_run(runs, target_mm_s=target_mm_s, schedule="uniform", seed=seed)
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
            status = "completed" if run is not None else "missing"
            failure_mode = "missing"
            if run is not None:
                if speed_success and temperature_success:
                    failure_mode = "recovered"
                elif not speed_success and int(history["late_best_eval"]) == 1:
                    failure_mode = "optimizer_not_converged_yet"
                elif not temperature_success:
                    failure_mode = "temperature_tracking_not_ok"
                else:
                    failure_mode = "missed_target_but_temperature_ok"
            theta_values = list(run.theta_C) if run is not None else [math.nan] * STUDY_NUM_KNOTS
            row = {
                "status": status,
                "target_front_speed_mm_s": float(target_mm_s),
                "schedule": "uniform",
                "seed": int(seed),
                "selected_run_name": "" if run is None else run.run_name,
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
                "failure_mode": failure_mode,
                "family_id": family_id,
                "theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.theta_C),
                "knot_times_s": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.knot_times_s),
            }
            for idx in range(STUDY_NUM_KNOTS):
                row[f"theta_{idx}_C"] = theta_values[idx]
            rows.append(row)
    return rows


def _aggregate_by_target(rows: list[dict[str, object]], *, seeds: tuple[int, ...]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["target_front_speed_mm_s"]), []).append(row)

    aggregates: list[dict[str, object]] = []
    for target_mm_s in sorted(grouped):
        members = grouped[target_mm_s]
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
        families = {str(row["family_id"]) for row in completed if str(row.get("family_id", "")).strip()}
        achieved_spread = math.nan if not achieved else float(max(achieved) - min(achieved))
        plateau_threshold = max(PLATEAU_MIN_ABS_SPREAD_MM_S, PLATEAU_MIN_REL_SPREAD * target_mm_s)
        plateau_split_detected = bool(
            len(families) >= 2 and math.isfinite(achieved_spread) and achieved_spread >= plateau_threshold
        )
        n_speed_success = sum(int(row.get("speed_success", 0)) for row in members)
        n_temperature_success = sum(int(row.get("temperature_success", 0)) for row in members)
        n_overall_success = sum(int(row.get("overall_success", 0)) for row in members)
        num_completed = len(completed)
        speed_success_fraction = math.nan if num_completed == 0 else float(n_speed_success) / float(num_completed)
        temperature_success_fraction = (
            math.nan if num_completed == 0 else float(n_temperature_success) / float(num_completed)
        )
        overall_success_fraction = math.nan if num_completed == 0 else float(n_overall_success) / float(num_completed)
        late_best_fraction = (
            math.nan
            if num_completed == 0
            else float(sum(int(row.get("late_best_eval", 0)) for row in completed)) / float(num_completed)
        )
        median_speed_error = math.nan if not rel_errors else float(np.median(np.asarray(rel_errors, dtype=np.float64)))
        median_infeasible_fraction = (
            math.nan
            if not infeasible_fractions
            else float(np.median(np.asarray(infeasible_fractions, dtype=np.float64)))
        )
        median_plate_mae = math.nan if not plate_mae else float(np.median(np.asarray(plate_mae, dtype=np.float64)))
        robustly_recovered = bool(
            num_completed >= len(seeds)
            and n_speed_success >= len(seeds)
            and math.isfinite(median_speed_error)
            and median_speed_error <= ROBUST_MAX_MEDIAN_SPEED_ERROR_PCT
            and not plateau_split_detected
            and math.isfinite(median_infeasible_fraction)
            and median_infeasible_fraction <= ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION
            and math.isfinite(late_best_fraction)
            and late_best_fraction <= ROBUST_MAX_LATE_BEST_FRACTION
        )
        if num_completed == 0:
            conclusion = ""
        elif robustly_recovered:
            conclusion = "Recovered by n8 uniform"
        elif n_overall_success > 0:
            conclusion = "Recovered but not robust"
        elif (
            math.isfinite(late_best_fraction)
            and late_best_fraction > ROBUST_MAX_LATE_BEST_FRACTION
        ) or (
            math.isfinite(median_infeasible_fraction)
            and median_infeasible_fraction > ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION
        ) or plateau_split_detected:
            conclusion = "Not recovered; likely BO-search-limited"
        else:
            conclusion = "Not recovered; review best coarse candidate"
        aggregates.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "schedule": "uniform",
                "num_rows": len(members),
                "num_completed": num_completed,
                "num_speed_success": int(n_speed_success),
                "num_temperature_success": int(n_temperature_success),
                "num_overall_success": int(n_overall_success),
                "speed_success_fraction": speed_success_fraction,
                "temperature_success_fraction": temperature_success_fraction,
                "overall_success_fraction": overall_success_fraction,
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
    return aggregates


def _build_fine_confirmation_candidates(
    *,
    run_rows: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        target_rows = [
            row
            for row in run_rows
            if row.get("status") == "completed"
            and _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
        ]
        if not target_rows:
            continue
        best_row = min(
            target_rows,
            key=lambda row: (
                math.inf if not math.isfinite(float(row.get("objective_value", math.nan))) else float(row["objective_value"]),
                str(row.get("selected_run_name", "")),
            ),
        )
        theta_C = tuple(float(best_row[f"theta_{idx}_C"]) for idx in range(STUDY_NUM_KNOTS))
        run_name = f"fine_confirm_v{_format_speed_tag(target_mm_s)}_n8_uniform_{RUN_NAME_TAG}"
        candidates.append(
            {
                "status": "candidate" if int(best_row.get("overall_success", 0)) else "review_only",
                "target_front_speed_mm_s": float(target_mm_s),
                "schedule": "uniform",
                "source_run_name": str(best_row.get("selected_run_name", "")),
                "source_seed": int(best_row.get("seed", 0)),
                "objective_value": float(best_row.get("objective_value", math.nan)),
                "achieved_direct_speed_mm_s": float(best_row.get("achieved_direct_speed_mm_s", math.nan)),
                "direct_speed_relative_error_pct": float(best_row.get("direct_speed_relative_error_pct", math.nan)),
                "mean_abs_plate_error_C": float(best_row.get("mean_abs_plate_error_C", math.nan)),
                "family_id": str(best_row.get("family_id", "")),
                "recommended_confirmation_run_name": run_name,
                "confirmation_command": _fine_confirmation_command(
                    target_mm_s=float(target_mm_s),
                    theta_C=theta_C,
                    run_name=run_name,
                ),
                **{f"theta_{idx}_C": float(theta_C[idx]) for idx in range(STUDY_NUM_KNOTS)},
            }
        )
    return candidates


def _readme_text(*, targets_mm_s: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    return "\n".join(
        [
            "# N8 Uniform From N3 Study",
            "",
            "This bundle summarizes the active 8-knot coarse BO campaign anchored on the defended uniform n3 result.",
            "",
            "## Locked study policy",
            "",
            f"- Target-speed range: `{min(targets_mm_s):.3f} -> {max(targets_mm_s):.3f} mm/s`.",
            f"- Explicit target grid: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            "- Schedule: `uniform` only.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            "- BO settings: `monotone_unit_box`, `feasible_local`, `EI`, `xi=0.01`, `init_points=2`, `n_iter=30`, `seed_with_theta0=true`.",
            f"- n3 anchor target: `{N3_ANCHOR_TARGET_MM_S:.3f} mm/s` with theta `{N3_ANCHOR_THETA_C}`.",
            f"- Interpolated n8 theta0: `{TUNED_THETA0_C}`.",
            f"- Interpolated n8 theta bounds: `{TUNED_THETA_BOUNDS_C}`.",
            "",
        ]
    )


def _scope_note_text() -> str:
    return "\n".join(
        [
            "# Scope Note",
            "",
            "- The active n8 study is intentionally uniform-only.",
            "- The prior and bounds are derived by piecewise-linear interpolation of the selected n3 uniform anchor.",
            "- Old n3/n4/n5 raw runs remain in their legacy folders; new n8 runs write under `results/active/n8/`.",
            "",
        ]
    )


def _summary_markdown(target_aggregate: list[dict[str, object]]) -> str:
    recovered = [
        f"{float(row['target_front_speed_mm_s']):.3f}"
        for row in target_aggregate
        if int(row.get("robustly_recovered", 0)) == 1
    ]
    coarse_hits = [
        f"{float(row['target_front_speed_mm_s']):.3f}"
        for row in target_aggregate
        if int(row.get("num_overall_success", 0)) > 0
    ]
    lines = [
        "# N8 Uniform Study Summary",
        "",
        f"- Robustly recovered targets: `{', '.join(recovered) if recovered else 'none'}` mm/s.",
        f"- Targets with at least one coarse overall success: `{', '.join(coarse_hits) if coarse_hits else 'none'}` mm/s.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the active n8 uniform-from-n3 BO study.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--bo-config", type=Path, default=DEFAULT_BO_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_TARGETS_MM_S))
    parser.add_argument("--schedules", default="uniform")
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
    if int(bo_config.num_knots) != STUDY_NUM_KNOTS:
        raise ValueError(f"this study expects bo.toml num_knots={STUDY_NUM_KNOTS}")
    if not bool(bo_config.seed_with_theta0):
        raise ValueError("this study is locked to seed_with_theta0=true")

    runs = _scan_runs(
        Path(args.bo_runs_root),
        simulation_config_path=Path(args.simulation_config),
        schedules=schedules,
        loader=_load_bo_run,
    )
    run_rows = _build_run_rows(runs=runs, targets_mm_s=targets_mm_s, seeds=seeds)
    target_aggregate = _aggregate_by_target(run_rows, seeds=seeds)
    fine_candidates = _build_fine_confirmation_candidates(run_rows=run_rows, targets_mm_s=targets_mm_s)

    if args.dry_run:
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Found n8 runs: {len(runs)}")
        print(f"Completed rows: {sum(1 for row in run_rows if row['status'] == 'completed')}/{len(run_rows)}")
        print(f"Expected coarse matrix size: {len(targets_mm_s) * len(seeds) * len(schedules)}")
        print(f"Anchored n3 target: {N3_ANCHOR_TARGET_MM_S:.3f} mm/s")
        print(f"Anchored n3 theta: {N3_ANCHOR_THETA_C}")
        print(f"Interpolated n8 theta0: {TUNED_THETA0_C}")
        print(f"Interpolated n8 theta bounds: {TUNED_THETA_BOUNDS_C}")
        print("")
        print("Representative dry-run command:")
        print(_representative_dry_run_command())
        return

    output_root = Path(args.output_root)
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))

    _write_text(output_root / "README.md", _readme_text(targets_mm_s=targets_mm_s, seeds=seeds))
    _write_text(output_root / "scope_note.md", _scope_note_text())
    _write_text(output_root / "final_study_summary.md", _summary_markdown(target_aggregate))
    _write_csv(output_root / "study_summary.csv", _run_fieldnames(), run_rows)
    _write_csv(
        output_root / "decision_summary.csv",
        (
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
        ),
        target_aggregate,
    )
    _write_csv(
        output_root / "fine_confirmation_candidates.csv",
        (
            "status",
            "target_front_speed_mm_s",
            "schedule",
            "source_run_name",
            "source_seed",
            "objective_value",
            "achieved_direct_speed_mm_s",
            "direct_speed_relative_error_pct",
            "mean_abs_plate_error_C",
            "family_id",
            *(f"theta_{idx}_C" for idx in range(STUDY_NUM_KNOTS)),
            "recommended_confirmation_run_name",
            "confirmation_command",
        ),
        fine_candidates,
    )

    run_commands: list[str] = []
    for target_mm_s in targets_mm_s:
        run_commands.append(f"# Target {target_mm_s:.3f} mm/s")
        for seed in seeds:
            matching = [
                row
                for row in run_rows
                if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
                and int(row["seed"]) == seed
            ]
            row = matching[0]
            run_name = f"bo_v{_format_speed_tag(target_mm_s)}_n8_uniform_{RUN_NAME_TAG}_seed{seed}"
            if row["status"] == "completed":
                run_commands.append(f"# Completed seed {seed}: {row['selected_run_name']}")
            else:
                run_commands.extend([_target_command(run_name, target_mm_s, seed), ""])
        run_commands.append("")
    _write_shell_script(
        output_root / "run_commands.sh",
        header_lines=[
            "# Active n8 uniform-from-n3 BO campaign.",
            "# Completed runs are listed as comments; missing runs remain executable blocks.",
        ],
        command_blocks=run_commands,
    )
    _write_shell_script(
        output_root / "fine_confirmation_commands.sh",
        header_lines=[
            "# Candidate fine confirmations for the active n8 uniform-from-n3 campaign.",
        ],
        command_blocks=[row["confirmation_command"] for row in fine_candidates],
    )
    _write_text(
        output_root / "representative_dry_run_command.sh",
        "#!/usr/bin/env bash\nset -e\n\n" + _representative_dry_run_command() + "\n",
    )

    print(f"N8 uniform-from-n3 study written to {output_root.resolve()}")
    print(f"  per-run summary    : {(output_root / 'study_summary.csv').resolve()}")
    print(f"  decision summary   : {(output_root / 'decision_summary.csv').resolve()}")
    print(f"  fine candidates    : {(output_root / 'fine_confirmation_candidates.csv').resolve()}")
    print(f"  run queue          : {(output_root / 'run_commands.sh').resolve()}")


if __name__ == "__main__":
    main()
