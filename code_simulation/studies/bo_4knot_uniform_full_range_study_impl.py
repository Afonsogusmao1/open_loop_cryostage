#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.config_files import DEFAULT_SIMULATION_CONFIG_PATH
from code_simulation.core.paths import (
    project_root,
    velocity_control_diagnostic_dir,
    velocity_control_results_dir,
)
from code_simulation.core.plate_tracking import plate_tracking_success
from code_simulation.studies.bo_3knot_target_and_spacing_study_impl import (
    INTERVAL_3_TO_11_MM,
    RunMetrics,
    SPEED_SUCCESS_REL_TOL,
    THETA_FAMILY_ROUND_DECIMALS,
    _best_theta_from_csv,
    _configure_matplotlib,
    _ensure_bo_plate_tracking_artifacts,
    _ensure_clean_directory,
    _ensure_fine_plate_tracking_artifacts,
    _failure_mode,
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


DEFAULT_OUTPUT_ROOT = velocity_control_diagnostic_dir(4, "admissible_range")
DEFAULT_BO_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_FINE_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_THREE_KNOT_STUDY_ROOT = velocity_control_diagnostic_dir(3, "admissible_range")
DEFAULT_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.013 + 1.0e-12, 0.001))
DEFAULT_SEEDS = (17, 29, 41)
DEFAULT_THETA0_C = (0.0, -7.0, -14.0, -21.0)
STUDY_NUM_KNOTS = 4
STUDY_SCHEDULE = "uniform"
TARGET_TOL_MM_S = 5.0e-7


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _target_range_text(targets_mm_s: tuple[float, ...]) -> str:
    if not targets_mm_s:
        return "none"
    return f"{min(targets_mm_s):.3f} -> {max(targets_mm_s):.3f}"


def _load_bo_run(path: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> RunMetrics | None:
    _ensure_bo_plate_tracking_artifacts(path, simulation_config_path=simulation_config_path, schedules=schedules)
    effective_path = path / "effective_config.toml"
    tracking_path = path / "best_tracking_summary.csv"
    theta_path = path / "best_theta_profile.csv"
    history_path = path / "bo_history.csv"
    plate_path = path / "plate_tracking_summary.csv"
    tc_path = path / "best_thermocouple_interval_speeds.csv"
    if not (effective_path.exists() and tracking_path.exists() and theta_path.exists() and history_path.exists() and plate_path.exists()):
        return None
    config = _read_toml(effective_path)
    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    target_cfg = dict(config.get("velocity_target", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    num_knots = int(trajectory_cfg.get("num_knots", 0))
    if num_knots != STUDY_NUM_KNOTS:
        return None
    if str(run_cfg.get("simulation_profile", "")).strip() != "optimization":
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
            num_knots=num_knots,
            schedules=schedules,
        )
    if schedule != STUDY_SCHEDULE:
        return None
    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    best_objective, n_evaluations, best_evaluation_index, late_best = _history_stats(history_path)
    seed_raw = bo_cfg.get("random_seed", None)
    return RunMetrics(
        run_dir=path,
        run_name=str(run_cfg.get("run_name", path.name)),
        simulation_profile="optimization",
        target_front_speed_mm_s=float(target_cfg.get("target_front_speed_mm_s")),
        schedule=schedule,
        num_knots=num_knots,
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


def _load_fine_run(path: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> RunMetrics | None:
    _ensure_fine_plate_tracking_artifacts(path, simulation_config_path=simulation_config_path)
    effective_path = path / "effective_config.toml"
    tracking_path = path / "velocity_tracking_summary.csv"
    plate_path = path / "plate_tracking_summary.csv"
    tc_path = path / "thermocouple_interval_speeds.csv"
    if not (effective_path.exists() and tracking_path.exists() and plate_path.exists() and tc_path.exists()):
        return None
    config = _read_toml(effective_path)
    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("manual_trajectory", {}))
    target_cfg = dict(config.get("velocity_target", {}))
    num_knots = int(trajectory_cfg.get("num_knots", 0))
    if num_knots != STUDY_NUM_KNOTS:
        return None
    if str(run_cfg.get("simulation_profile", "")).strip() != "full_process_article":
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
            num_knots=num_knots,
            schedules=schedules,
        )
    if schedule != STUDY_SCHEDULE:
        return None
    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    return RunMetrics(
        run_dir=path,
        run_name=str(run_cfg.get("run_name", path.name)),
        simulation_profile="full_process_article",
        target_front_speed_mm_s=float(target_cfg.get("target_front_speed_mm_s")),
        schedule=schedule,
        num_knots=num_knots,
        normalized_support_tau=normalized_support_tau,
        knot_times_s=knot_times_s,
        theta_C=tuple(float(value) for value in trajectory_cfg.get("theta_C", [])),
        objective_value=math.nan,
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
        n_evaluations=0,
        best_evaluation_index=0,
        late_best_eval=False,
        random_seed=None,
    )


def _find_bo_run(
    runs: tuple[RunMetrics, ...],
    *,
    target_mm_s: float,
    seed: int,
) -> RunMetrics | None:
    matching = [
        run
        for run in runs
        if run.simulation_profile == "optimization"
        and _matches_target(run.target_front_speed_mm_s, target_mm_s)
        and run.schedule == STUDY_SCHEDULE
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


def _find_matching_fine_run(
    runs: tuple[RunMetrics, ...],
    *,
    target_mm_s: float,
    theta_signature: tuple[float, ...] | None = None,
) -> RunMetrics | None:
    matching = [
        run
        for run in runs
        if run.simulation_profile == "full_process_article"
        and _matches_target(run.target_front_speed_mm_s, target_mm_s)
        and run.schedule == STUDY_SCHEDULE
    ]
    if theta_signature is not None:
        exact = [run for run in matching if _family_signature(run.theta_C) == theta_signature]
        if exact:
            matching = exact
    if not matching:
        return None
    return min(
        matching,
        key=lambda run: (
            math.inf if not math.isfinite(run.tracking_rmse_mm) else run.tracking_rmse_mm,
            run.run_name,
        ),
    )


def _target_command(run_name: str, target_mm_s: float, seed: int) -> str:
    repo_root = project_root()
    theta0_arg = ",".join(f"{float(value):.15g}" for value in DEFAULT_THETA0_C)
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.optimization.run_velocity_control_bo \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        "  --num-knots 4 \\\n"
        "  --knot-time-schedule uniform \\\n"
        f"  --theta0-c={theta0_arg} \\\n"
        f"  --seed {int(seed)} \\\n"
        "  --simulation-profile optimization \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _fine_command(run_name: str, target_mm_s: float, theta_C: tuple[float, ...]) -> str:
    repo_root = project_root()
    theta_arg = ",".join(f"{float(value):.15g}" for value in theta_C)
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.verification.run_velocity_control_evaluation \\\n"
        "  --simulation-profile full_process_article \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        f"  --theta-c={theta_arg} \\\n"
        "  --num-knots 4 \\\n"
        "  --knot-time-schedule uniform \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _coarse_rows(
    *,
    bo_runs: tuple[RunMetrics, ...],
    targets_mm_s: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    family_lookup: dict[tuple[float, ...], str] = {}
    family_counter = 0
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for seed in seeds:
            run = _find_bo_run(bo_runs, target_mm_s=target_mm_s, seed=seed)
            speed_success = False
            rel_error_pct = math.nan
            temperature_success = False
            family_id = ""
            if run is not None:
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
            rows.append(
                {
                    "status": status,
                    "target_front_speed_mm_s": float(target_mm_s),
                    "schedule": STUDY_SCHEDULE,
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
                    "best_evaluation_index": 0 if run is None else run.best_evaluation_index,
                    "n_evaluations": 0 if run is None else run.n_evaluations,
                    "late_best_eval": 0 if run is None else int(run.late_best_eval),
                    "failure_mode": _failure_mode(
                        status=status,
                        speed_success=speed_success,
                        temperature_success=temperature_success,
                        late_best_eval=False if run is None else run.late_best_eval,
                    ),
                    "family_id": family_id,
                    "theta_0_C": math.nan if run is None else run.theta_C[0],
                    "theta_1_C": math.nan if run is None else run.theta_C[1],
                    "theta_2_C": math.nan if run is None else run.theta_C[2],
                    "theta_3_C": math.nan if run is None else run.theta_C[3],
                    "theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.theta_C),
                    "knot_times_s": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.knot_times_s),
                }
            )
    return rows


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
        families = {str(row["family_id"]) for row in completed if str(row.get("family_id", "")).strip()}
        best = _best_target_row(completed)
        aggregates.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "num_rows": len(members),
                "num_completed": len(completed),
                "num_speed_success": sum(int(row.get("speed_success", 0)) for row in members),
                "num_temperature_success": sum(int(row.get("temperature_success", 0)) for row in members),
                "num_overall_success": sum(int(row.get("overall_success", 0)) for row in members),
                "num_late_best_eval": sum(int(row.get("late_best_eval", 0)) for row in members),
                "num_distinct_families": len(families),
                "min_achieved_direct_speed_mm_s": math.nan if not achieved else min(achieved),
                "max_achieved_direct_speed_mm_s": math.nan if not achieved else max(achieved),
                "best_run_name": "" if best is None else best["selected_run_name"],
                "best_run_seed": "" if best is None else best["seed"],
                "best_run_objective_value": math.nan if best is None else best["objective_value"],
                "best_run_achieved_direct_speed_mm_s": math.nan if best is None else best["achieved_direct_speed_mm_s"],
                "best_run_direct_speed_relative_error_pct": math.nan if best is None else best["direct_speed_relative_error_pct"],
                "best_run_overall_success": 0 if best is None else best["overall_success"],
            }
        )
    return aggregates


def _best_target_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -int(row.get("overall_success", 0)),
            math.inf if not math.isfinite(float(row.get("direct_speed_relative_error_pct", math.nan))) else float(row["direct_speed_relative_error_pct"]),
            math.inf if not math.isfinite(float(row.get("objective_value", math.nan))) else float(row["objective_value"]),
            math.inf if not math.isfinite(float(row.get("rmse_plate_error_C", math.nan))) else float(row["rmse_plate_error_C"]),
            str(row.get("selected_run_name", "")),
        ),
    )


def _best_rows_by_target(rows: list[dict[str, str]], *, targets_mm_s: tuple[float, ...]) -> dict[float, dict[str, str]]:
    best_by_target: dict[float, dict[str, str]] = {}
    for target_mm_s in targets_mm_s:
        matching = [
            row
            for row in rows
            if row.get("status") == "completed"
            and _matches_target(float(row.get("target_front_speed_mm_s", "nan")), target_mm_s)
        ]
        if not matching:
            continue
        matching.sort(
            key=lambda row: (
                -int(row.get("overall_success", "0")),
                math.inf if not math.isfinite(float(row.get("direct_speed_relative_error_pct", "nan"))) else float(row["direct_speed_relative_error_pct"]),
                math.inf if not math.isfinite(float(row.get("objective_value", "nan"))) else float(row["objective_value"]),
                math.inf if not math.isfinite(float(row.get("rmse_plate_error_C", "nan"))) else float(row["rmse_plate_error_C"]),
                str(row.get("selected_run_name", "")),
            )
        )
        best_by_target[target_mm_s] = matching[0]
    return best_by_target


def _load_three_knot_best_rows(study_root: Path, *, targets_mm_s: tuple[float, ...]) -> dict[float, dict[str, str]]:
    admissible_summary_path = study_root / "study_summary.csv"
    if admissible_summary_path.exists():
        admissible_rows = _read_csv_rows(admissible_summary_path)
        if admissible_rows:
            return _best_rows_by_target(admissible_rows, targets_mm_s=targets_mm_s)

    schedule_path = study_root / "schedule_comparison" / "schedule_summary.csv"
    uniform_path = study_root / "target_vs_achived" / "study_summary.csv"
    schedule_rows = _read_csv_rows(schedule_path) if schedule_path.exists() else []
    uniform_rows = _read_csv_rows(uniform_path) if uniform_path.exists() else []
    return _best_rows_by_target(schedule_rows + uniform_rows, targets_mm_s=targets_mm_s)


def _comparison_rows(
    *,
    coarse_rows: list[dict[str, object]],
    three_knot_best_by_target: dict[float, dict[str, str]],
    targets_mm_s: tuple[float, ...],
) -> list[dict[str, object]]:
    rows_by_target: dict[float, list[dict[str, object]]] = {}
    for row in coarse_rows:
        rows_by_target.setdefault(float(row["target_front_speed_mm_s"]), []).append(row)
    comparison_rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        four_best = _best_target_row([row for row in rows_by_target.get(target_mm_s, []) if row.get("status") == "completed"])
        three_best = three_knot_best_by_target.get(target_mm_s)
        three_success = 0 if three_best is None else int(three_best.get("overall_success", "0"))
        three_rel_error = math.nan if three_best is None else float(three_best.get("direct_speed_relative_error_pct", "nan"))
        four_success = 0 if four_best is None else int(four_best.get("overall_success", 0))
        four_rel_error = math.nan if four_best is None else float(four_best.get("direct_speed_relative_error_pct", math.nan))
        improved = 0
        if four_best is not None and three_best is not None:
            if four_success > three_success:
                improved = 1
            elif four_success == three_success:
                if math.isfinite(four_rel_error) and math.isfinite(three_rel_error) and four_rel_error < three_rel_error - 1.0e-12:
                    improved = 1
        comparison_rows.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "best_3k_run_name": "" if three_best is None else three_best.get("selected_run_name", ""),
                "best_3k_schedule": "" if three_best is None else three_best.get("schedule", ""),
                "best_3k_seed": "" if three_best is None else three_best.get("seed", ""),
                "best_3k_overall_success": three_success,
                "best_3k_achieved_direct_speed_mm_s": math.nan if three_best is None else float(three_best.get("achieved_direct_speed_mm_s", "nan")),
                "best_3k_direct_speed_relative_error_pct": three_rel_error,
                "best_3k_objective_value": math.nan if three_best is None else float(three_best.get("objective_value", "nan")),
                "best_3k_rmse_plate_error_C": math.nan if three_best is None else float(three_best.get("rmse_plate_error_C", "nan")),
                "best_4k_run_name": "" if four_best is None else four_best["selected_run_name"],
                "best_4k_seed": "" if four_best is None else four_best["seed"],
                "best_4k_overall_success": four_success,
                "best_4k_achieved_direct_speed_mm_s": math.nan if four_best is None else four_best["achieved_direct_speed_mm_s"],
                "best_4k_direct_speed_relative_error_pct": four_rel_error,
                "best_4k_objective_value": math.nan if four_best is None else four_best["objective_value"],
                "best_4k_rmse_plate_error_C": math.nan if four_best is None else four_best["rmse_plate_error_C"],
                "four_knot_improves_on_three_knot": int(improved),
            }
        )
    return comparison_rows


def _candidate_row_from_coarse(row: dict[str, object]) -> dict[str, object]:
    target_mm_s = float(row["target_front_speed_mm_s"])
    fine_run_name = f"fine_confirm_v{_format_speed_tag(target_mm_s)}_n4_uniform_best"
    return {
        "target_front_speed_mm_s": target_mm_s,
        "source_run_name": row["selected_run_name"],
        "source_seed": row["seed"],
        "family_id": row["family_id"],
        "theta_0_C": row["theta_0_C"],
        "theta_1_C": row["theta_1_C"],
        "theta_2_C": row["theta_2_C"],
        "theta_3_C": row["theta_3_C"],
        "theta_C": row["theta_C"],
        "recommended_confirmation_run_name": fine_run_name,
        "confirmation_command": _fine_command(
            fine_run_name,
            target_mm_s,
            tuple(float(value) for value in str(row["theta_C"]).split(",") if value),
        ),
    }


def _representative_fine_candidates(
    *,
    coarse_rows: list[dict[str, object]],
    fine_runs: tuple[RunMetrics, ...],
) -> list[dict[str, object]]:
    best_success_by_target: dict[float, dict[str, object]] = {}
    for row in coarse_rows:
        if row.get("status") != "completed" or int(row.get("overall_success", 0)) != 1:
            continue
        target_mm_s = float(row["target_front_speed_mm_s"])
        current = best_success_by_target.get(target_mm_s)
        if current is None or _best_target_row([current, row]) == row:
            best_success_by_target[target_mm_s] = row

    if not best_success_by_target:
        return []

    family_best: dict[tuple[float, ...], dict[str, object]] = {}
    for row in best_success_by_target.values():
        signature = _family_signature(tuple(float(value) for value in str(row["theta_C"]).split(",") if value))
        current = family_best.get(signature)
        if current is None:
            family_best[signature] = row
            continue
        current_rel = float(current["direct_speed_relative_error_pct"])
        new_rel = float(row["direct_speed_relative_error_pct"])
        current_obj = float(current["objective_value"])
        new_obj = float(row["objective_value"])
        if (new_rel, new_obj, float(row["target_front_speed_mm_s"])) < (current_rel, current_obj, float(current["target_front_speed_mm_s"])):
            family_best[signature] = row

    family_rows = sorted(family_best.values(), key=lambda row: float(row["target_front_speed_mm_s"]))
    if len(family_rows) <= 4:
        selected = family_rows
    else:
        indices = [0, round((len(family_rows) - 1) / 3), round(2 * (len(family_rows) - 1) / 3), len(family_rows) - 1]
        selected = []
        seen_targets: set[float] = set()
        for idx in indices:
            row = family_rows[int(idx)]
            target_mm_s = float(row["target_front_speed_mm_s"])
            if target_mm_s in seen_targets:
                continue
            selected.append(row)
            seen_targets.add(target_mm_s)

    candidates: list[dict[str, object]] = []
    for row in selected:
        theta_signature = _family_signature(tuple(float(value) for value in str(row["theta_C"]).split(",") if value))
        fine_run = _find_matching_fine_run(
            fine_runs,
            target_mm_s=float(row["target_front_speed_mm_s"]),
            theta_signature=theta_signature,
        )
        speed_success = False
        temperature_success = False
        if fine_run is not None:
            speed_success, _ = _speed_success(float(row["target_front_speed_mm_s"]), fine_run.achieved_direct_speed_mm_s)
            plate_summary_like = type(
                "_Tmp",
                (),
                {"mean_abs_plate_error_C": fine_run.mean_abs_plate_error_C, "tolerance_C": fine_run.plate_tolerance_C},
            )
            temperature_success = plate_tracking_success(plate_summary_like)
        candidate = _candidate_row_from_coarse(row)
        candidate.update(
            {
                "status": "completed" if fine_run is not None else "pending",
                "matched_fine_run_name": "" if fine_run is None else fine_run.run_name,
                "fine_achieved_direct_speed_mm_s": math.nan if fine_run is None else fine_run.achieved_direct_speed_mm_s,
                "fine_tracking_rmse_mm": math.nan if fine_run is None else fine_run.tracking_rmse_mm,
                "fine_rmse_plate_error_C": math.nan if fine_run is None else fine_run.rmse_plate_error_C,
                "fine_speed_success": int(speed_success),
                "fine_temperature_success": int(temperature_success),
                "fine_overall_success": int(speed_success and temperature_success),
            }
        )
        candidates.append(candidate)
    return candidates


def _decision_rows(
    *,
    coarse_rows: list[dict[str, object]],
    fine_runs: tuple[RunMetrics, ...],
    targets_mm_s: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        target_rows = [
            row
            for row in coarse_rows
            if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s) and row.get("status") == "completed"
        ]
        fine_target_runs = [
            run for run in fine_runs if _matches_target(run.target_front_speed_mm_s, target_mm_s) and run.schedule == STUDY_SCHEDULE
        ]
        fine_success_count = 0
        for run in fine_target_runs:
            speed_success, _ = _speed_success(target_mm_s, run.achieved_direct_speed_mm_s)
            plate_summary_like = type(
                "_Tmp",
                (),
                {"mean_abs_plate_error_C": run.mean_abs_plate_error_C, "tolerance_C": run.plate_tolerance_C},
            )
            if speed_success and plate_tracking_success(plate_summary_like):
                fine_success_count += 1
        recovered = fine_success_count > 0
        any_late = any(int(row.get("late_best_eval", 0)) for row in target_rows)
        family_count = len({str(row.get("family_id", "")) for row in target_rows if str(row.get("family_id", "")).strip()})
        achieved = [
            float(row["achieved_direct_speed_mm_s"])
            for row in target_rows
            if math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
        ]
        clustered_away = bool(
            achieved
            and (max(achieved) - min(achieved) <= max(0.0005, 0.05 * float(target_mm_s)))
            and not any(int(row.get("overall_success", 0)) for row in target_rows)
        )
        all_temperature_fail = bool(target_rows) and all(int(row.get("temperature_success", 0)) == 0 for row in target_rows)
        likely_bo_search_limited = int((not recovered) and bool(target_rows) and (any_late or family_count > 1))
        likely_plant_inner_response_limited = int((not recovered) and all_temperature_fail)
        likely_4k_parameterization_limited = int(
            (not recovered)
            and bool(target_rows)
            and not any(int(row.get("overall_success", 0)) for row in target_rows)
            and clustered_away
            and not likely_bo_search_limited
            and not likely_plant_inner_response_limited
        )
        rows.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "num_completed_coarse_runs": len(target_rows),
                "num_coarse_overall_successes": sum(int(row.get("overall_success", 0)) for row in target_rows),
                "num_fine_confirmations": len(fine_target_runs),
                "num_fine_successes": int(fine_success_count),
                "recovered_by_4knots_uniform": int(recovered),
                "likely_bo_search_limited": int(likely_bo_search_limited),
                "likely_4k_parameterization_limited": int(likely_4k_parameterization_limited),
                "likely_plant_inner_response_limited": int(likely_plant_inner_response_limited),
                "min_achieved_direct_speed_mm_s": math.nan if not achieved else min(achieved),
                "max_achieved_direct_speed_mm_s": math.nan if not achieved else max(achieved),
                "num_distinct_theta_families": int(family_count),
            }
        )
    return rows


def _plot_target_vs_achieved(out_path: Path, *, rows: list[dict[str, object]]) -> None:
    completed = [
        row
        for row in rows
        if row.get("status") == "completed" and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
    ]
    _configure_matplotlib()
    fig, ax = plt.subplots()
    if not completed:
        ax.axis("off")
        ax.set_title("4-knot uniform admissible-range sweep: target vs achieved direct speed")
        ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    color_by_seed = {17: "#1f77b4", 29: "#2ca02c", 41: "#d62728"}
    x_all = np.asarray([float(row["target_front_speed_mm_s"]) for row in completed], dtype=np.float64)
    y_all = np.asarray([float(row["achieved_direct_speed_mm_s"]) for row in completed], dtype=np.float64)
    line_min = float(np.min(x_all))
    line_max = float(np.max(x_all))
    ax.plot([line_min, line_max], [line_min, line_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
    for seed in sorted({int(row["seed"]) for row in completed}):
        seed_rows = [row for row in completed if int(row["seed"]) == seed]
        ax.scatter(
            [float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
            color=color_by_seed.get(seed),
            label=f"seed {seed}",
            s=36,
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title("4-knot uniform admissible-range sweep: target vs achieved direct speed")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_success_rate(out_path: Path, *, aggregate_rows: list[dict[str, object]]) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots()
    if not aggregate_rows:
        ax.axis("off")
        ax.set_title("4-knot uniform admissible-range sweep: coarse success rate by target")
        ax.text(0.02, 0.95, "No aggregate rows available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    targets = np.asarray([float(row["target_front_speed_mm_s"]) for row in aggregate_rows], dtype=np.float64)
    success_rate = np.asarray(
        [
            float(row["num_overall_success"]) / float(row["num_completed"])
            if float(row["num_completed"]) > 0
            else math.nan
            for row in aggregate_rows
        ],
        dtype=np.float64,
    )
    ax.plot(targets, success_rate, marker="o", linewidth=1.5, color="#1f77b4")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Coarse overall-success rate")
    ax.set_title("4-knot uniform admissible-range sweep: coarse success rate by target")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_best_comparison(out_path: Path, *, comparison_rows: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in comparison_rows
        if math.isfinite(float(row.get("best_3k_achieved_direct_speed_mm_s", math.nan)))
        or math.isfinite(float(row.get("best_4k_achieved_direct_speed_mm_s", math.nan)))
    ]
    _configure_matplotlib()
    fig, ax = plt.subplots()
    if not rows:
        ax.axis("off")
        ax.set_title("Best 3-knot vs best 4-knot uniform coarse speed")
        ax.text(0.02, 0.95, "No comparison rows available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    x = np.asarray([float(row["target_front_speed_mm_s"]) for row in rows], dtype=np.float64)
    y3 = np.asarray([float(row.get("best_3k_achieved_direct_speed_mm_s", math.nan)) for row in rows], dtype=np.float64)
    y4 = np.asarray([float(row.get("best_4k_achieved_direct_speed_mm_s", math.nan)) for row in rows], dtype=np.float64)
    ax.plot(x, x, "--", color="0.5", linewidth=1.2, label="target")
    ax.plot(x, y3, marker="o", linewidth=1.5, color="#d62728", label="best 3-knot")
    ax.plot(x, y4, marker="o", linewidth=1.5, color="#1f77b4", label="best 4-knot uniform")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title("Best 3-knot vs best 4-knot uniform coarse speed")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _coarse_note(rows: list[dict[str, object]], *, targets_mm_s: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    successes = [row for row in completed if int(row.get("overall_success", 0)) == 1]
    return "\n".join(
        [
            "# 4-Knot Uniform Coarse Sweep Note",
            "",
            "This stage isolates knot count by holding the time schedule fixed at `uniform` and sweeping the configured admissible target-speed range.",
            "",
            "## Configuration",
            "",
            f"- Targets: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            f"- Initial theta: `{DEFAULT_THETA0_C}`.",
            f"- Completed runs: `{len(completed)}/{len(rows)}`.",
            f"- Coarse runs meeting the dual success rule: `{len(successes)}`.",
            "",
            "## Dual success rule",
            "",
            "- `speed_success`: direct interval-speed error within `5%` over `2.5-11.5 mm`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref|` within the `+-0.5 C` tolerance over the same front-control interval.",
            "",
        ]
    )


def _readme_text(*, targets_mm_s: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    return "\n".join(
        [
            "# 4-Knot Uniform BO Study",
            "",
            "This folder maps what the fixed `4-knot + uniform-time` BO setup can recover across the configured admissible target-speed range.",
            "",
            "## Policy",
            "",
            "- BO objective window: `2.5-11.5 mm`.",
            "- Experimental / thermocouple span: `3.0-11.0 mm`.",
            "- Knot count: `4`.",
            "- Knot-time schedule: `uniform` only for this round.",
            "- Dual coarse success rule: direct-speed error within `5%` and mean absolute `|T_plate - T_ref|` within the `+-0.5 C` tolerance over the same front-control interval.",
            "- Official admissible velocity interval: `0.0035-0.013 mm/s`.",
            "",
            "## Coverage",
            "",
            f"- Target-speed range for this bundle: `{_target_range_text(targets_mm_s)} mm/s`.",
            f"- Targets: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            "- Coverage is limited to the currently available raw `n4` runs inside this admissible target grid; missing targets remain pending.",
            "",
            "## Key outputs",
            "",
            "- `coarse_sweep/study_summary.csv`: one row per `target x seed`.",
            "- `coarse_sweep/target_aggregate.csv`: one aggregate row per target.",
            "- `comparison_with_3k/best_by_target_comparison.csv`: best `3-knot` versus best `4-knot uniform` coarse comparison at each target.",
            "- `decision_summary.csv`: target-level interpretation after the current coarse and fine evidence.",
            "- `fine_confirmation_candidates.csv`: representative recovered families to confirm in `full_process_article`.",
            "- `final_study_summary.md`: compact narrative summary for the current state of the study.",
            "",
        ]
    )


def _final_summary_text(
    *,
    decision_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    fine_candidates: list[dict[str, object]],
) -> str:
    coarse_recovered = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("num_coarse_overall_successes", 0)) > 0
    ]
    fine_recovered = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("recovered_by_4knots_uniform", 0)) == 1
    ]
    unresolved = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("num_completed_coarse_runs", 0)) > 0 and int(row.get("num_coarse_overall_successes", 0)) == 0
    ]
    improved_targets = [
        float(row["target_front_speed_mm_s"])
        for row in comparison_rows
        if int(row.get("four_knot_improves_on_three_knot", 0)) == 1
    ]
    bo_limited_targets = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("likely_bo_search_limited", 0)) == 1
    ]
    parameterization_limited_targets = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("likely_4k_parameterization_limited", 0)) == 1
    ]
    plant_limited_targets = [
        float(row["target_front_speed_mm_s"])
        for row in decision_rows
        if int(row.get("likely_plant_inner_response_limited", 0)) == 1
    ]
    completed_fine = [row for row in fine_candidates if row.get("status") == "completed"]
    return "\n".join(
        [
            "# Final 4-Knot Uniform Study Summary",
            "",
            "## 1. Recovered speeds",
            "",
            f"- Coarse overall-success targets: `{', '.join(f'{value:.3f}' for value in coarse_recovered) if coarse_recovered else 'none yet'}` mm/s.",
            f"- Fine-confirmed recovered targets: `{', '.join(f'{value:.3f}' for value in fine_recovered) if fine_recovered else 'none yet'}` mm/s.",
            "",
            "## 2. Not yet recovered",
            "",
            f"- Targets with completed coarse evidence but no coarse overall success: `{', '.join(f'{value:.3f}' for value in unresolved) if unresolved else 'none'}` mm/s.",
            "",
            "## 3. Comparison against 3 knots",
            "",
            f"- Targets where best `4-knot uniform` coarse evidence improves on the best `3-knot` coarse evidence: `{', '.join(f'{value:.3f}' for value in improved_targets) if improved_targets else 'none yet'}` mm/s.",
            "",
            "## 4. Current failure interpretation",
            "",
            f"- BO-search-limited signals: `{', '.join(f'{value:.3f}' for value in bo_limited_targets) if bo_limited_targets else 'none'}` mm/s.",
            f"- Possible `4-knot` parameterization limits: `{', '.join(f'{value:.3f}' for value in parameterization_limited_targets) if parameterization_limited_targets else 'none'}` mm/s.",
            f"- Possible plant / inner-response limits: `{', '.join(f'{value:.3f}' for value in plant_limited_targets) if plant_limited_targets else 'none'}` mm/s.",
            "",
            "## Representative fine status",
            "",
            f"- Completed representative fine confirmations: `{len(completed_fine)}`.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 4-knot uniform BO study.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--fine-runs-root", type=Path, default=DEFAULT_FINE_RUNS_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--three-knot-study-root", type=Path, default=DEFAULT_THREE_KNOT_STUDY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_TARGETS_MM_S))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_mm_s = _parse_float_list(args.targets_mm_s)
    seeds = _parse_int_list(args.seeds)
    schedules = (STUDY_SCHEDULE,)

    bo_runs = _scan_runs(
        Path(args.bo_runs_root),
        simulation_config_path=Path(args.simulation_config),
        schedules=schedules,
        loader=_load_bo_run,
    )
    fine_runs = _scan_runs(
        Path(args.fine_runs_root),
        simulation_config_path=Path(args.simulation_config),
        schedules=schedules,
        loader=_load_fine_run,
    )
    coarse_rows = _coarse_rows(bo_runs=bo_runs, targets_mm_s=targets_mm_s, seeds=seeds)
    aggregate_rows = _aggregate_rows(coarse_rows)
    three_knot_best_by_target = _load_three_knot_best_rows(
        Path(args.three_knot_study_root),
        targets_mm_s=targets_mm_s,
    )
    comparison_rows = _comparison_rows(
        coarse_rows=coarse_rows,
        three_knot_best_by_target=three_knot_best_by_target,
        targets_mm_s=targets_mm_s,
    )
    fine_candidates = _representative_fine_candidates(coarse_rows=coarse_rows, fine_runs=fine_runs)
    decision_rows = _decision_rows(coarse_rows=coarse_rows, fine_runs=fine_runs, targets_mm_s=targets_mm_s)

    if args.dry_run:
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Found BO runs: {len(bo_runs)}")
        print(f"Found fine runs: {len(fine_runs)}")
        print(f"Completed coarse rows: {sum(1 for row in coarse_rows if row['status'] == 'completed')}/{len(coarse_rows)}")
        print(f"Coarse overall successes: {sum(int(row.get('overall_success', 0)) for row in coarse_rows)}")
        print(f"Representative fine candidates: {len(fine_candidates)}")
        return

    output_root = Path(args.output_root)
    coarse_dir = output_root / "coarse_sweep"
    comparison_dir = output_root / "comparison_with_3k"
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))
    coarse_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    _write_text(output_root / "README.md", _readme_text(targets_mm_s=targets_mm_s, seeds=seeds))
    _write_csv(
        coarse_dir / "study_summary.csv",
        (
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
            "late_best_eval",
            "failure_mode",
            "family_id",
            "theta_0_C",
            "theta_1_C",
            "theta_2_C",
            "theta_3_C",
            "theta_C",
            "knot_times_s",
        ),
        coarse_rows,
    )
    _write_csv(
        coarse_dir / "target_aggregate.csv",
        (
            "target_front_speed_mm_s",
            "num_rows",
            "num_completed",
            "num_speed_success",
            "num_temperature_success",
            "num_overall_success",
            "num_late_best_eval",
            "num_distinct_families",
            "min_achieved_direct_speed_mm_s",
            "max_achieved_direct_speed_mm_s",
            "best_run_name",
            "best_run_seed",
            "best_run_objective_value",
            "best_run_achieved_direct_speed_mm_s",
            "best_run_direct_speed_relative_error_pct",
            "best_run_overall_success",
        ),
        aggregate_rows,
    )
    _write_text(coarse_dir / "study_note.md", _coarse_note(coarse_rows, targets_mm_s=targets_mm_s, seeds=seeds))
    _plot_target_vs_achieved(coarse_dir / "target_vs_achieved_direct_speed.png", rows=coarse_rows)
    _plot_success_rate(coarse_dir / "success_rate_by_target.png", aggregate_rows=aggregate_rows)

    coarse_commands: list[str] = []
    for target_mm_s in targets_mm_s:
        for seed in seeds:
            matching = [
                row
                for row in coarse_rows
                if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s) and int(row["seed"]) == seed
            ]
            row = matching[0]
            run_name = f"bo_v{_format_speed_tag(target_mm_s)}_n4_uniform_seed{seed}"
            if row["status"] == "completed":
                coarse_commands.append(f"# Completed target {target_mm_s:.3f} mm/s, seed {seed}: {row['selected_run_name']}")
            else:
                coarse_commands.extend([_target_command(run_name, target_mm_s, seed), ""])
        coarse_commands.append("")
    _write_shell_script(
        coarse_dir / "run_commands.sh",
        header_lines=[
            "# 4-knot uniform admissible-range sweep commands across the configured seed set.",
            "# Completed runs are listed as comments; missing runs remain executable blocks.",
        ],
        command_blocks=coarse_commands,
    )

    _write_csv(
        comparison_dir / "best_by_target_comparison.csv",
        (
            "target_front_speed_mm_s",
            "best_3k_run_name",
            "best_3k_schedule",
            "best_3k_seed",
            "best_3k_overall_success",
            "best_3k_achieved_direct_speed_mm_s",
            "best_3k_direct_speed_relative_error_pct",
            "best_3k_objective_value",
            "best_3k_rmse_plate_error_C",
            "best_4k_run_name",
            "best_4k_seed",
            "best_4k_overall_success",
            "best_4k_achieved_direct_speed_mm_s",
            "best_4k_direct_speed_relative_error_pct",
            "best_4k_objective_value",
            "best_4k_rmse_plate_error_C",
            "four_knot_improves_on_three_knot",
        ),
        comparison_rows,
    )
    _plot_best_comparison(comparison_dir / "best_3k_vs_best_4k.png", comparison_rows=comparison_rows)

    _write_csv(
        output_root / "decision_summary.csv",
        (
            "target_front_speed_mm_s",
            "num_completed_coarse_runs",
            "num_coarse_overall_successes",
            "num_fine_confirmations",
            "num_fine_successes",
            "recovered_by_4knots_uniform",
            "likely_bo_search_limited",
            "likely_4k_parameterization_limited",
            "likely_plant_inner_response_limited",
            "min_achieved_direct_speed_mm_s",
            "max_achieved_direct_speed_mm_s",
            "num_distinct_theta_families",
        ),
        decision_rows,
    )

    _write_csv(
        output_root / "fine_confirmation_candidates.csv",
        (
            "status",
            "target_front_speed_mm_s",
            "source_run_name",
            "source_seed",
            "family_id",
            "theta_0_C",
            "theta_1_C",
            "theta_2_C",
            "theta_3_C",
            "theta_C",
            "recommended_confirmation_run_name",
            "matched_fine_run_name",
            "fine_achieved_direct_speed_mm_s",
            "fine_tracking_rmse_mm",
            "fine_rmse_plate_error_C",
            "fine_speed_success",
            "fine_temperature_success",
            "fine_overall_success",
            "confirmation_command",
        ),
        fine_candidates,
    )
    fine_commands: list[str] = []
    for row in fine_candidates:
        if row["status"] == "completed":
            fine_commands.append(
                f"# Completed: {row['recommended_confirmation_run_name']} -> {row['matched_fine_run_name']}"
            )
        else:
            fine_commands.extend([str(row["confirmation_command"]), ""])
    _write_shell_script(
        output_root / "fine_confirmation_commands.sh",
        header_lines=[
            "# Fine-confirmation commands for representative 4-knot uniform families.",
            "# Completed confirmations are listed as comments.",
        ],
        command_blocks=fine_commands,
    )

    _write_csv(
        output_root / "representative_fine_summary.csv",
        (
            "status",
            "target_front_speed_mm_s",
            "source_run_name",
            "source_seed",
            "family_id",
            "matched_fine_run_name",
            "fine_achieved_direct_speed_mm_s",
            "fine_tracking_rmse_mm",
            "fine_rmse_plate_error_C",
            "fine_speed_success",
            "fine_temperature_success",
            "fine_overall_success",
        ),
        [
            {
                "status": row["status"],
                "target_front_speed_mm_s": row["target_front_speed_mm_s"],
                "source_run_name": row["source_run_name"],
                "source_seed": row["source_seed"],
                "family_id": row["family_id"],
                "matched_fine_run_name": row["matched_fine_run_name"],
                "fine_achieved_direct_speed_mm_s": row["fine_achieved_direct_speed_mm_s"],
                "fine_tracking_rmse_mm": row["fine_tracking_rmse_mm"],
                "fine_rmse_plate_error_C": row["fine_rmse_plate_error_C"],
                "fine_speed_success": row["fine_speed_success"],
                "fine_temperature_success": row["fine_temperature_success"],
                "fine_overall_success": row["fine_overall_success"],
            }
            for row in fine_candidates
        ],
    )

    _write_text(
        output_root / "final_study_summary.md",
        _final_summary_text(
            decision_rows=decision_rows,
            comparison_rows=comparison_rows,
            fine_candidates=fine_candidates,
        ),
    )

    print(f"4-knot uniform admissible-range study written to {output_root.resolve()}")
    print(f"  coarse summary : {(coarse_dir / 'study_summary.csv').resolve()}")
    print(f"  target aggregate: {(coarse_dir / 'target_aggregate.csv').resolve()}")
    print(f"  comparison      : {(comparison_dir / 'best_by_target_comparison.csv').resolve()}")
    print(f"  decision summary: {(output_root / 'decision_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
