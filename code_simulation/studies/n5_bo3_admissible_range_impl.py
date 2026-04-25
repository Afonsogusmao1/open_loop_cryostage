#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.config_files import DEFAULT_BO_CONFIG_PATH, DEFAULT_SIMULATION_CONFIG_PATH, load_bo_config
from code_simulation.core.paths import project_root, velocity_control_diagnostic_dir, velocity_control_results_dir
from code_simulation.core.plate_tracking import plate_tracking_success
from code_simulation.studies.bo_3knot_target_and_spacing_study_impl import (
    INTERVAL_3_TO_11_MM,
    RunMetrics,
    THETA_FAMILY_ROUND_DECIMALS,
    _best_theta_from_csv,
    _configure_matplotlib,
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


DEFAULT_OUTPUT_ROOT = velocity_control_diagnostic_dir(5, "bo_monotone_feasible_admissible_range")
DEFAULT_BO_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.013 + 1.0e-12, 0.001))
DEFAULT_SCHEDULES = ("uniform", "early_dense", "late_dense")
DEFAULT_SEEDS = (17, 29, 41)
STUDY_NUM_KNOTS = 5
RUN_NAME_TAG = "bov3"
TUNED_THETA0_C = (0.0, -5.0, -10.0, -15.0, -20.0)
TUNED_THETA_BOUNDS_C = (
    (-10.0, 0.0),
    (-13.0, -3.0),
    (-16.0, -6.0),
    (-18.5, -8.0),
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


def _target_range_text(targets_mm_s: tuple[float, ...]) -> str:
    if not targets_mm_s:
        return "none"
    return f"{min(targets_mm_s):.3f} -> {max(targets_mm_s):.3f}"


def _tuned_theta_bounds_text() -> str:
    return ",".join(f"{lower:.15g}:{upper:.15g}" for lower, upper in TUNED_THETA_BOUNDS_C)


def _theta0_text() -> str:
    return ",".join(f"{float(value):.15g}" for value in TUNED_THETA0_C)


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _target_command(run_name: str, target_mm_s: float, schedule: str, seed: int) -> str:
    repo_root = project_root()
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.optimization.run_velocity_control_bo \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        "  --num-knots 5 \\\n"
        f"  --knot-time-schedule {schedule} \\\n"
        f"  --theta0-c={_theta0_text()} \\\n"
        f"  --theta-bounds={_tuned_theta_bounds_text()} \\\n"
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


def _representative_dry_run_command() -> str:
    return _target_command(
        run_name=f"bo_v0p008_n5_uniform_{RUN_NAME_TAG}_seed17",
        target_mm_s=0.008,
        schedule="uniform",
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
    trajectory_cfg = dict(config.get("trajectory", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    if int(trajectory_cfg.get("num_knots", 0)) != STUDY_NUM_KNOTS:
        return False
    if not _matches_tuned_bounds(trajectory_cfg.get("theta_bounds_C", [])):
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
    if not _matches_tuned_signature(config):
        return None
    run_cfg = dict(config.get("run", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    target_cfg = dict(config.get("velocity_target", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    run_name = str(run_cfg.get("run_name", path.name))
    if f"_{RUN_NAME_TAG}_" not in run_name:
        return None
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
    if schedule not in schedules:
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
        if run.simulation_profile == "optimization"
        and _matches_target(run.target_front_speed_mm_s, target_mm_s)
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
    runs: tuple[RunMetrics, ...],
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
                run = _find_bo_run(runs, target_mm_s=target_mm_s, schedule=schedule, seed=seed)
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
                    "schedule": schedule,
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


def _group_rows(rows: list[dict[str, object]], *, group_keys: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, object]] = []
    for key in sorted(grouped):
        members = grouped[key]
        completed = [row for row in members if row.get("status") == "completed"]
        target_mm_s = float(members[0]["target_front_speed_mm_s"])
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
        plateau_split_detected = bool(len(families) >= 2 and math.isfinite(achieved_spread) and achieved_spread >= plateau_threshold)
        n_speed_success = sum(int(row.get("speed_success", 0)) for row in members)
        n_temperature_success = sum(int(row.get("temperature_success", 0)) for row in members)
        n_overall_success = sum(int(row.get("overall_success", 0)) for row in members)
        num_completed = len(completed)
        speed_success_fraction = math.nan if num_completed == 0 else float(n_speed_success) / float(num_completed)
        temperature_success_fraction = math.nan if num_completed == 0 else float(n_temperature_success) / float(num_completed)
        overall_success_fraction = math.nan if num_completed == 0 else float(n_overall_success) / float(num_completed)
        late_best_fraction = math.nan if num_completed == 0 else float(sum(int(row.get("late_best_eval", 0)) for row in completed)) / float(num_completed)
        median_speed_error = math.nan if not rel_errors else float(np.median(np.asarray(rel_errors, dtype=np.float64)))
        median_infeasible_fraction = math.nan if not infeasible_fractions else float(np.median(np.asarray(infeasible_fractions, dtype=np.float64)))
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
            conclusion = "Recovered by BO3 n5"
        elif n_speed_success > 0:
            conclusion = "Recovered but not robust"
        elif math.isfinite(median_plate_mae) and median_plate_mae > 0.5 and math.isfinite(median_speed_error) and median_speed_error <= 10.0:
            conclusion = "Not recovered; likely plate/inner-response-limited"
        elif (
            (math.isfinite(late_best_fraction) and late_best_fraction > ROBUST_MAX_LATE_BEST_FRACTION)
            or (math.isfinite(median_infeasible_fraction) and median_infeasible_fraction > ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION)
            or plateau_split_detected
            or len(families) >= 3
        ):
            conclusion = "Not recovered; likely BO-search-limited"
        else:
            conclusion = "Not recovered; likely parameterization-limited"
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
            ax.text(0.02, 0.95, "No completed n5 BO3 runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    x_values = [float(row["target_front_speed_mm_s"]) for row in completed]
    y_values = [float(row["achieved_direct_speed_mm_s"]) for row in completed]
    axis_max = max(max(x_values), max(y_values)) * 1.03
    for ax, schedule in zip(axes, schedules, strict=True):
        schedule_rows = [row for row in completed if str(row.get("schedule")) == schedule]
        ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
        for seed in sorted({int(row["seed"]) for row in schedule_rows}):
            seed_rows = [row for row in schedule_rows if int(row["seed"]) == seed]
            ax.scatter(
                [float(row["target_front_speed_mm_s"]) for row in seed_rows],
                [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
                color=color_by_seed.get(seed),
                s=38,
                label=f"seed {seed}",
            )
        ax.set_title(schedule)
        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.set_xlabel("Target speed (mm/s)")
        if ax is axes[0]:
            ax.set_ylabel("Achieved direct speed (mm/s)")
        ax.legend(loc="upper left")
    fig.suptitle("BO3 n5: target vs achieved direct speed by schedule", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_heatmap(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
    schedules: tuple[str, ...],
    value_key: str,
    title: str,
    colorbar_label: str,
    value_format: str = ".2f",
) -> None:
    matrix = np.full((len(schedules), len(targets_mm_s)), np.nan, dtype=np.float64)
    for i, schedule in enumerate(schedules):
        for j, target_mm_s in enumerate(targets_mm_s):
            row = _find_group_aggregate(rows, target_mm_s=target_mm_s, schedule=schedule)
            if row is None:
                continue
            value = float(row.get(value_key, math.nan))
            if math.isfinite(value):
                matrix[i, j] = value

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8.5, 0.6 * len(targets_mm_s) + 3.0), 3.8))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Schedule")
    ax.set_xticks(np.arange(len(targets_mm_s)))
    ax.set_xticklabels([f"{value:.3f}" for value in targets_mm_s], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(schedules)))
    ax.set_yticklabels(list(schedules))
    for i in range(len(schedules)):
        for j in range(len(targets_mm_s)):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, format(value, value_format), ha="center", va="center", color="white", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _readme_text(*, targets_mm_s: tuple[float, ...], schedules: tuple[str, ...], seeds: tuple[int, ...]) -> str:
    return "\n".join(
        [
            "# N5 BO3 Admissible-Range Study",
            "",
            "This folder contains the fixed-time 5-knot BO3 study over the active admissible target-speed range.",
            "",
            "## Locked study policy",
            "",
            "- Objective unchanged: direct front tracking over `2.5-11.5 mm`.",
            f"- Target-speed range for this bundle: `{_target_range_text(targets_mm_s)} mm/s`.",
            f"- Explicit target grid: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Schedules: `{', '.join(schedules)}`.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            "- BO3 settings: `monotone_unit_box`, `feasible_local`, `EI`, `xi=0.01`, `init_points=2`, `n_iter=30`, `seed_with_theta0=true`.",
            f"- Theta0: `{TUNED_THETA0_C}`.",
            f"- Theta bounds: `{TUNED_THETA_BOUNDS_C}`.",
            "",
            "## Metric definitions",
            "",
            "- `speed_success`: relative error of direct front speed versus target, over `2.5-11.5 mm`, threshold `<= 5%`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref|` over the same front-control interval in time, threshold `<= 0.5 C`.",
            "- `overall_success`: both conditions true.",
            "- `robustly_recovered`: all three seeds satisfy `speed_success`, median speed error is `<= 5%`, and robustness diagnostics do not indicate a plateau split or persistent late best.",
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
            "- This study ingests only `n5` coarse runs whose run name contains `_bov3_` and whose effective config matches the BO3 signature.",
            "- No `n3` middle-time-movable runs, `n4` runs, or objective changes are part of this study.",
            "",
        ]
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


def _aggregate_fieldnames(prefix_keys: tuple[str, ...]) -> tuple[str, ...]:
    return prefix_keys + (
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the BO3 admissible-range study for n5.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--bo-config", type=Path, default=DEFAULT_BO_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
        raise ValueError("This study is locked to seed_with_theta0=true.")

    runs = _scan_runs(
        Path(args.bo_runs_root),
        simulation_config_path=Path(args.simulation_config),
        schedules=schedules,
        loader=_load_bo_run,
    )
    run_rows = _build_run_rows(runs=runs, targets_mm_s=targets_mm_s, schedules=schedules, seeds=seeds)
    target_schedule_aggregate = _group_rows(run_rows, group_keys=("target_front_speed_mm_s", "schedule"), seeds=seeds)
    target_aggregate = _group_rows(run_rows, group_keys=("target_front_speed_mm_s",), seeds=seeds)

    if args.dry_run:
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Found n5 BO3 runs: {len(runs)}")
        print(f"Completed rows: {sum(1 for row in run_rows if row['status'] == 'completed')}/{len(run_rows)}")
        print(f"Expected coarse matrix size: {len(targets_mm_s) * len(schedules) * len(seeds)}")
        print("")
        print("Representative dry-run command:")
        print(_representative_dry_run_command())
        return

    output_root = Path(args.output_root)
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))

    _write_text(output_root / "README.md", _readme_text(targets_mm_s=targets_mm_s, schedules=schedules, seeds=seeds))
    _write_text(output_root / "scope_note.md", _scope_note_text(targets_mm_s=targets_mm_s))
    _write_csv(output_root / "study_summary.csv", _run_fieldnames(), run_rows)
    _write_csv(output_root / "target_schedule_aggregate.csv", _aggregate_fieldnames(("target_front_speed_mm_s", "schedule")), target_schedule_aggregate)
    _write_csv(output_root / "target_aggregate.csv", _aggregate_fieldnames(("target_front_speed_mm_s",)), target_aggregate)
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
            "num_distinct_theta_families",
            "plateau_split_detected",
            "robustly_recovered",
            "study_conclusion",
        ),
        [
            {
                "target_front_speed_mm_s": row["target_front_speed_mm_s"],
                "schedule": row["schedule"],
                "num_completed": row["num_completed"],
                "num_speed_success": row["num_speed_success"],
                "speed_success_fraction": row["speed_success_fraction"],
                "median_direct_speed_relative_error_pct": row["median_direct_speed_relative_error_pct"],
                "median_infeasible_fraction": row["median_infeasible_fraction"],
                "late_best_fraction": row["late_best_fraction"],
                "median_plate_mae_C": row["median_plate_mae_C"],
                "achieved_speed_spread_mm_s": row["achieved_speed_spread_mm_s"],
                "num_distinct_theta_families": row["num_distinct_theta_families"],
                "plateau_split_detected": row["plateau_split_detected"],
                "robustly_recovered": row["robustly_recovered"],
                "study_conclusion": row["study_conclusion"],
            }
            for row in target_schedule_aggregate
        ],
    )
    _plot_target_vs_achieved_by_schedule(
        output_root / "target_vs_achieved_direct_speed_by_schedule.png",
        rows=run_rows,
        schedules=schedules,
    )
    _plot_metric_heatmap(
        output_root / "success_fraction_by_target_and_schedule.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        value_key="speed_success_fraction",
        title="BO3 n5: speed-success fraction by target and schedule",
        colorbar_label="Speed-success fraction across seeds",
    )
    _plot_metric_heatmap(
        output_root / "infeasible_fraction_by_target_and_schedule.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        value_key="median_infeasible_fraction",
        title="BO3 n5: median infeasible fraction by target and schedule",
        colorbar_label="Median infeasible fraction across seeds",
    )
    _plot_metric_heatmap(
        output_root / "late_best_fraction_by_target_and_schedule.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        value_key="late_best_fraction",
        title="BO3 n5: late-best fraction by target and schedule",
        colorbar_label="Late-best fraction across seeds",
    )
    _plot_metric_heatmap(
        output_root / "median_plate_mae_by_target_and_schedule.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedules=schedules,
        value_key="median_plate_mae_C",
        title="BO3 n5: median plate MAE by target and schedule",
        colorbar_label="Median plate MAE (C)",
    )

    run_commands: list[str] = []
    for target_mm_s in targets_mm_s:
        run_commands.append(f"# Target {target_mm_s:.3f} mm/s")
        for schedule in schedules:
            run_commands.append(f"# Schedule {schedule}")
            for seed in seeds:
                matching = [
                    row
                    for row in run_rows
                    if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
                    and str(row["schedule"]) == schedule
                    and int(row["seed"]) == seed
                ]
                row = matching[0]
                run_name = f"bo_v{_format_speed_tag(target_mm_s)}_n5_{schedule}_{RUN_NAME_TAG}_seed{seed}"
                if row["status"] == "completed":
                    run_commands.append(f"# Completed {schedule}, seed {seed}: {row['selected_run_name']}")
                else:
                    run_commands.extend([_target_command(run_name, target_mm_s, schedule, seed), ""])
            run_commands.append("")
        run_commands.append("")
    _write_shell_script(
        output_root / "run_commands.sh",
        header_lines=[
            "# BO3 admissible-range study for n5.",
            "# Completed runs are listed as comments; missing runs remain executable blocks.",
        ],
        command_blocks=run_commands,
    )
    _write_text(
        output_root / "representative_dry_run_command.sh",
        "#!/usr/bin/env bash\nset -e\n\n" + _representative_dry_run_command() + "\n",
    )

    print(f"BO3 n5 admissible-range study written to {output_root.resolve()}")
    print(f"  per-run summary    : {(output_root / 'study_summary.csv').resolve()}")
    print(f"  decision summary   : {(output_root / 'decision_summary.csv').resolve()}")
    print(f"  run queue          : {(output_root / 'run_commands.sh').resolve()}")


if __name__ == "__main__":
    main()
