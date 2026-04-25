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
    _write_shell_script,
    _write_text,
)


@dataclass(frozen=True)
class StudyRun:
    run_dir: Path
    run_name: str
    campaign_group: str
    schedule_family: str
    base_schedule_name: str
    custom_tau_mid: float | None
    target_front_speed_mm_s: float
    num_knots: int
    normalized_support_tau: tuple[float, ...]
    knot_times_s: tuple[float, ...]
    theta_C: tuple[float, ...]
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


DEFAULT_OUTPUT_ROOT = velocity_control_diagnostic_dir(3, "bo4_middle_knot_custom_admissible_range")
DEFAULT_RUNNER_ROOT = velocity_control_diagnostic_dir(3, "bo4_middle_knot_custom_admissible_range_runner")
DEFAULT_BO_RUNS_ROOT = velocity_control_coarse_dir(3)
DEFAULT_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.013 + 1.0e-12, 0.001))
DEFAULT_SEEDS = (17, 29, 41)
BASELINE_SCHEDULES = ("uniform", "early_dense", "late_dense")
CUSTOM_TAU_MID_VALUES = (0.25, 0.40, 0.60, 0.75)
GATE_TARGETS_MM_S = (0.006, 0.009, 0.012)
STUDY_NUM_KNOTS = 3
BASELINE_RUN_TAG = "bov3"
CUSTOM_RUN_TAG = "bov4mk"
TUNED_THETA0_C = (0.0, -10.0, -20.0)
TUNED_THETA_BOUNDS_C = ((-10.0, 0.0), (-16.0, -6.0), (-21.0, -10.0))
TUNED_ACQUISITION_KIND = "ei"
TUNED_ACQUISITION_XI = 0.01
TUNED_INIT_POINTS = 2
TUNED_N_ITER = 30
TUNED_PARAMETERIZATION_KIND = "monotone_unit_box"
TUNED_INIT_STRATEGY = "feasible_local_deterministic"
TUNED_INIT_LOCAL_SIGMA = 0.15
TUNED_INIT_MAX_ATTEMPTS_PER_POINT = 40
TUNED_LOCAL_REFINEMENT_POINTS = 6
TUNED_LOCAL_REFINEMENT_SIGMA = 0.08
ROBUST_MAX_MEDIAN_SPEED_ERROR_PCT = 5.0
ROBUST_MAX_MEDIAN_INFEASIBLE_FRACTION = 0.35
ROBUST_MAX_LATE_BEST_FRACTION = 0.4
PLATEAU_MIN_ABS_SPREAD_MM_S = 0.001
PLATEAU_MIN_REL_SPREAD = 0.10
CAMPAIGN_GROUP_BASELINE = "baseline_historical"
CAMPAIGN_GROUP_CUSTOM = "custom_middle_knot"


def _tau_tag(tau_mid: float) -> str:
    return f"{float(tau_mid):0.2f}".replace(".", "p")


def _custom_family_label(tau_mid: float) -> str:
    return f"custom_tau{_tau_tag(tau_mid)}"


CUSTOM_FAMILY_LABELS = tuple(_custom_family_label(value) for value in CUSTOM_TAU_MID_VALUES)
DEFAULT_SCHEDULE_FAMILIES = BASELINE_SCHEDULES + CUSTOM_FAMILY_LABELS
CUSTOM_TAU_BY_FAMILY = {
    _custom_family_label(value): float(value)
    for value in CUSTOM_TAU_MID_VALUES
}
CUSTOM_SUPPORT_BY_FAMILY = {
    label: (0.0, float(tau_mid), 1.0)
    for label, tau_mid in CUSTOM_TAU_BY_FAMILY.items()
}


def _target_range_text(targets_mm_s: tuple[float, ...]) -> str:
    if not targets_mm_s:
        return "none"
    return f"{min(targets_mm_s):.3f} -> {max(targets_mm_s):.3f}"


def _theta0_text() -> str:
    return ",".join(f"{float(value):.15g}" for value in TUNED_THETA0_C)


def _tuned_theta_bounds_text() -> str:
    return ",".join(f"{lower:.15g}:{upper:.15g}" for lower, upper in TUNED_THETA_BOUNDS_C)


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _custom_support_text(tau_mid: float) -> str:
    return f"0,{float(tau_mid):0.2f},1"


def _family_metadata(schedule_family: str) -> tuple[str, str, float | None]:
    if schedule_family in BASELINE_SCHEDULES:
        return (CAMPAIGN_GROUP_BASELINE, schedule_family, None)
    tau_mid = CUSTOM_TAU_BY_FAMILY.get(schedule_family)
    if tau_mid is None:
        raise ValueError(f"Unsupported schedule family {schedule_family!r}")
    return (CAMPAIGN_GROUP_CUSTOM, "custom", float(tau_mid))


def _target_command(
    run_name: str,
    target_mm_s: float,
    tau_mid: float,
    seed: int,
    *,
    parent_run_dir: Path | None,
) -> str:
    repo_root = project_root()
    custom_support_text = _custom_support_text(tau_mid)
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
        "  --knot-time-schedule custom \\\n"
        f"  --knot-time-custom-support-tau {custom_support_text} \\\n"
        f"{theta_arg}"
        f"  --theta-bounds={_tuned_theta_bounds_text()} \\\n"
        f"  --seed {int(seed)} \\\n"
        "  --simulation-profile optimization \\\n"
        f"  --init-points {TUNED_INIT_POINTS} \\\n"
        f"  --n-iter {TUNED_N_ITER} \\\n"
        f"  --acquisition-kind {TUNED_ACQUISITION_KIND} \\\n"
        f"  --acquisition-xi {TUNED_ACQUISITION_XI:.15g} \\\n"
        f"  --parameterization-kind {TUNED_PARAMETERIZATION_KIND} \\\n"
        f"  --init-strategy {TUNED_INIT_STRATEGY} \\\n"
        f"  --init-local-sigma {TUNED_INIT_LOCAL_SIGMA:.15g} \\\n"
        f"  --init-max-attempts-per-point {TUNED_INIT_MAX_ATTEMPTS_PER_POINT} \\\n"
        f"  --local-refinement-points {TUNED_LOCAL_REFINEMENT_POINTS} \\\n"
        f"  --local-refinement-sigma {TUNED_LOCAL_REFINEMENT_SIGMA:.15g} \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _representative_dry_run_command() -> str:
    return _target_command(
        run_name=f"bo_v0p009_n3_{_custom_family_label(0.40)}_{CUSTOM_RUN_TAG}_seed17",
        target_mm_s=0.009,
        tau_mid=0.40,
        seed=17,
        parent_run_dir=None,
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


def _matches_theta0(raw_theta0) -> bool:
    if not isinstance(raw_theta0, list) or len(raw_theta0) != len(TUNED_THETA0_C):
        return False
    for raw_value, tuned_value in zip(raw_theta0, TUNED_THETA0_C, strict=True):
        if abs(float(raw_value) - float(tuned_value)) > 1.0e-12:
            return False
    return True


def _matches_bov3_baseline_signature(config: dict) -> bool:
    trajectory_cfg = dict(config.get("trajectory", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    if int(trajectory_cfg.get("num_knots", 0)) != STUDY_NUM_KNOTS:
        return False
    if not _matches_theta0(trajectory_cfg.get("theta0_C", [])):
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


def _matches_bov4mk_custom_signature(config: dict) -> bool:
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
    if str(bo_cfg.get("parameterization_kind", "")).strip().lower() != TUNED_PARAMETERIZATION_KIND:
        return False
    if str(bo_cfg.get("init_strategy", "")).strip().lower() != TUNED_INIT_STRATEGY:
        return False
    if abs(float(bo_cfg.get("init_local_sigma", math.nan)) - TUNED_INIT_LOCAL_SIGMA) > 1.0e-12:
        return False
    if int(bo_cfg.get("init_max_attempts_per_point", -1)) != TUNED_INIT_MAX_ATTEMPTS_PER_POINT:
        return False
    if int(bo_cfg.get("local_refinement_points", -1)) != TUNED_LOCAL_REFINEMENT_POINTS:
        return False
    if abs(float(bo_cfg.get("local_refinement_sigma", math.nan)) - TUNED_LOCAL_REFINEMENT_SIGMA) > 1.0e-12:
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


def _resolve_campaign_metadata(
    *,
    run_name: str,
    base_schedule_name: str,
    normalized_support_tau: tuple[float, ...],
) -> tuple[str, str, float | None] | None:
    if f"_{CUSTOM_RUN_TAG}_" in run_name:
        if base_schedule_name != "custom":
            return None
        for schedule_family, support_tau in CUSTOM_SUPPORT_BY_FAMILY.items():
            if np.allclose(
                np.asarray(normalized_support_tau, dtype=np.float64),
                np.asarray(support_tau, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            ):
                if f"_{schedule_family}_" not in run_name:
                    return None
                return (CAMPAIGN_GROUP_CUSTOM, schedule_family, CUSTOM_TAU_BY_FAMILY[schedule_family])
        return None
    if f"_{BASELINE_RUN_TAG}_" in run_name:
        if base_schedule_name not in BASELINE_SCHEDULES:
            return None
        if f"_{base_schedule_name}_" not in run_name:
            return None
        return (CAMPAIGN_GROUP_BASELINE, base_schedule_name, None)
    return None


def _load_study_run(path: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> StudyRun | None:
    _ensure_bo_plate_tracking_artifacts(path, simulation_config_path=simulation_config_path, schedules=BASELINE_SCHEDULES)
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
    run_name = str(run_cfg.get("run_name", path.name))
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
    base_schedule_name = str(trajectory_cfg.get("knot_time_schedule", "")).strip().lower()
    metadata = _resolve_campaign_metadata(
        run_name=run_name,
        base_schedule_name=base_schedule_name,
        normalized_support_tau=normalized_support_tau,
    )
    if metadata is None:
        return None
    campaign_group, schedule_family, custom_tau_mid = metadata
    if campaign_group == CAMPAIGN_GROUP_BASELINE:
        if not _matches_bov3_baseline_signature(config):
            return None
    elif campaign_group == CAMPAIGN_GROUP_CUSTOM:
        if not _matches_bov4mk_custom_signature(config):
            return None
    else:
        return None

    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    best_objective, n_evaluations, best_evaluation_index, late_best = _history_stats(history_path)
    seed_raw = bo_cfg.get("random_seed", None)
    return StudyRun(
        run_dir=path,
        run_name=run_name,
        campaign_group=campaign_group,
        schedule_family=schedule_family,
        base_schedule_name=base_schedule_name,
        custom_tau_mid=custom_tau_mid,
        target_front_speed_mm_s=float(target_cfg.get("target_front_speed_mm_s")),
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


def _scan_study_runs(
    root: Path,
    *,
    simulation_config_path: Path,
) -> tuple[StudyRun, ...]:
    summaries: list[StudyRun] = []
    if not root.exists():
        return tuple()
    candidates = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and ("_bov3_" in path.name or "_bov4mk_" in path.name)
    ]
    for candidate in candidates:
        summary = _load_study_run(
            candidate,
            simulation_config_path=simulation_config_path,
            schedules=BASELINE_SCHEDULES,
        )
        if summary is not None:
            summaries.append(summary)
    return tuple(summaries)


def _find_run(
    runs: tuple[StudyRun, ...],
    *,
    target_mm_s: float,
    schedule_family: str,
    seed: int,
) -> StudyRun | None:
    matching = [
        run
        for run in runs
        if _matches_target(run.target_front_speed_mm_s, target_mm_s)
        and run.schedule_family == schedule_family
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
    schedule_families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    family_lookup: dict[tuple[float, ...], str] = {}
    family_counter = 0
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for schedule_family in schedule_families:
            expected_campaign_group, expected_base_schedule_name, expected_tau_mid = _family_metadata(schedule_family)
            for seed in seeds:
                run = _find_run(runs, target_mm_s=target_mm_s, schedule_family=schedule_family, seed=seed)
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
                normalized_support_tau = (
                    "" if run is None else ",".join(f"{float(value):.15g}" for value in run.normalized_support_tau)
                )
                row = {
                    "status": status,
                    "campaign_group": expected_campaign_group if run is None else run.campaign_group,
                    "schedule_family": schedule_family if run is None else run.schedule_family,
                    "base_schedule_name": expected_base_schedule_name if run is None else run.base_schedule_name,
                    "custom_tau_mid": math.nan if expected_tau_mid is None else expected_tau_mid,
                    "target_front_speed_mm_s": float(target_mm_s),
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
                    "normalized_support_tau": normalized_support_tau,
                }
                for idx in range(STUDY_NUM_KNOTS):
                    row[f"theta_{idx}_C"] = theta_values[idx]
                rows.append(row)
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
        median_achieved = math.nan if not achieved else float(np.median(np.asarray(achieved, dtype=np.float64)))
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
            conclusion = "Recovered by n3 BO3"
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
                "base_schedule_name": str(members[0]["base_schedule_name"]),
                "custom_tau_mid": float(members[0]["custom_tau_mid"]) if math.isfinite(float(members[0]["custom_tau_mid"])) else math.nan,
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
    schedule_family: str,
) -> dict[str, object] | None:
    for row in rows:
        if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s) and str(row.get("schedule_family")) == schedule_family:
            return row
    return None


def _best_family_row(
    rows: list[dict[str, object]],
    *,
    target_mm_s: float,
    campaign_group: str,
) -> dict[str, object] | None:
    candidates = [
        row
        for row in rows
        if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
        and str(row.get("campaign_group")) == campaign_group
        and int(row.get("num_completed", 0)) > 0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            math.inf if not math.isfinite(float(row.get("median_direct_speed_relative_error_pct", math.nan))) else float(row["median_direct_speed_relative_error_pct"]),
            -float(row.get("speed_success_fraction", 0.0)),
            math.inf if not math.isfinite(float(row.get("median_plate_mae_C", math.nan))) else float(row["median_plate_mae_C"]),
            str(row.get("schedule_family", "")),
        ),
    )


def _build_target_aggregate(
    *,
    target_schedule_rows: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        baseline = _best_family_row(target_schedule_rows, target_mm_s=target_mm_s, campaign_group=CAMPAIGN_GROUP_BASELINE)
        custom = _best_family_row(target_schedule_rows, target_mm_s=target_mm_s, campaign_group=CAMPAIGN_GROUP_CUSTOM)

        baseline_achieved = math.nan if baseline is None else float(baseline.get("median_achieved_direct_speed_mm_s", math.nan))
        custom_achieved = math.nan if custom is None else float(custom.get("median_achieved_direct_speed_mm_s", math.nan))
        baseline_error = math.nan if baseline is None else float(baseline.get("median_direct_speed_relative_error_pct", math.nan))
        custom_error = math.nan if custom is None else float(custom.get("median_direct_speed_relative_error_pct", math.nan))
        achieved_delta = math.nan if not (math.isfinite(baseline_achieved) and math.isfinite(custom_achieved)) else float(custom_achieved - baseline_achieved)
        error_delta = math.nan if not (math.isfinite(baseline_error) and math.isfinite(custom_error)) else float(custom_error - baseline_error)

        rows.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "baseline_best_schedule_family": "" if baseline is None else baseline["schedule_family"],
                "baseline_best_num_completed": 0 if baseline is None else int(baseline["num_completed"]),
                "baseline_best_speed_success_fraction": math.nan if baseline is None else float(baseline["speed_success_fraction"]),
                "baseline_best_median_achieved_direct_speed_mm_s": baseline_achieved,
                "baseline_best_median_direct_speed_relative_error_pct": baseline_error,
                "baseline_best_median_plate_mae_C": math.nan if baseline is None else float(baseline["median_plate_mae_C"]),
                "baseline_best_study_conclusion": "" if baseline is None else baseline["study_conclusion"],
                "custom_best_schedule_family": "" if custom is None else custom["schedule_family"],
                "custom_best_num_completed": 0 if custom is None else int(custom["num_completed"]),
                "custom_best_speed_success_fraction": math.nan if custom is None else float(custom["speed_success_fraction"]),
                "custom_best_median_achieved_direct_speed_mm_s": custom_achieved,
                "custom_best_median_direct_speed_relative_error_pct": custom_error,
                "custom_best_median_plate_mae_C": math.nan if custom is None else float(custom["median_plate_mae_C"]),
                "custom_best_study_conclusion": "" if custom is None else custom["study_conclusion"],
                "custom_minus_baseline_median_achieved_direct_speed_mm_s": achieved_delta,
                "custom_minus_baseline_median_direct_speed_error_pct": error_delta,
                "custom_beats_baseline_on_median_speed_error": int(
                    math.isfinite(baseline_error) and math.isfinite(custom_error) and custom_error < baseline_error - 1.0e-12
                ),
                "custom_beats_baseline_on_median_achieved_speed": int(
                    math.isfinite(baseline_achieved) and math.isfinite(custom_achieved) and custom_achieved > baseline_achieved + 1.0e-12
                ),
            }
        )
    return rows


def _plot_target_vs_achieved_by_schedule_family(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    schedule_families: tuple[str, ...],
) -> None:
    _configure_matplotlib()
    ncols = min(4, max(1, len(schedule_families)))
    nrows = int(math.ceil(float(len(schedule_families)) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).ravel()
    color_by_seed = {17: "#1f77b4", 29: "#2ca02c", 41: "#d62728"}
    completed = [
        row
        for row in rows
        if row.get("status") == "completed"
        and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
    ]
    if not completed:
        for ax in axes_array:
            ax.axis("off")
            ax.text(0.02, 0.95, "No completed n3 custom study runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return

    x_values = [float(row["target_front_speed_mm_s"]) for row in completed]
    y_values = [float(row["achieved_direct_speed_mm_s"]) for row in completed]
    axis_max = max(max(x_values), max(y_values)) * 1.03
    legend_handles = None
    legend_labels = None
    for idx, schedule_family in enumerate(schedule_families):
        ax = axes_array[idx]
        schedule_rows = [row for row in completed if str(row.get("schedule_family")) == schedule_family]
        diagonal = ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")[0]
        handles = [diagonal]
        labels = ["target = achieved"]
        for seed in sorted({int(row["seed"]) for row in schedule_rows}):
            seed_rows = [row for row in schedule_rows if int(row["seed"]) == seed]
            scatter = ax.scatter(
                [float(row["target_front_speed_mm_s"]) for row in seed_rows],
                [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
                color=color_by_seed.get(seed),
                s=38,
                label=f"seed {seed}",
            )
            handles.append(scatter)
            labels.append(f"seed {seed}")
        ax.set_title(schedule_family)
        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.set_xlabel("Target speed (mm/s)")
        if idx % ncols == 0:
            ax.set_ylabel("Achieved direct speed (mm/s)")
        if legend_handles is None:
            legend_handles = handles
            legend_labels = labels

    for ax in axes_array[len(schedule_families):]:
        ax.axis("off")
    if legend_handles is not None and legend_labels is not None:
        axes_array[0].legend(legend_handles, legend_labels, loc="upper left")
    fig.suptitle("n3 BO3: target vs achieved direct speed by schedule family", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_metric_heatmap(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    targets_mm_s: tuple[float, ...],
    schedule_families: tuple[str, ...],
    value_key: str,
    title: str,
    colorbar_label: str,
    value_format: str = ".2f",
) -> None:
    matrix = np.full((len(schedule_families), len(targets_mm_s)), np.nan, dtype=np.float64)
    for i, schedule_family in enumerate(schedule_families):
        for j, target_mm_s in enumerate(targets_mm_s):
            row = _find_group_aggregate(rows, target_mm_s=target_mm_s, schedule_family=schedule_family)
            if row is None:
                continue
            value = float(row.get(value_key, math.nan))
            if math.isfinite(value):
                matrix[i, j] = value

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(9.0, 0.6 * len(targets_mm_s) + 3.0), max(3.8, 0.52 * len(schedule_families) + 1.6)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Schedule family")
    ax.set_xticks(np.arange(len(targets_mm_s)))
    ax.set_xticklabels([f"{value:.3f}" for value in targets_mm_s], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(schedule_families)))
    ax.set_yticklabels(list(schedule_families))
    for i in range(len(schedule_families)):
        for j in range(len(targets_mm_s)):
            value = matrix[i, j]
            if math.isfinite(value):
                ax.text(j, i, format(value, value_format), ha="center", va="center", color="white", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_baseline_vs_custom_envelope(
    out_path: Path,
    *,
    target_rows: list[dict[str, object]],
) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    targets = [float(row["target_front_speed_mm_s"]) for row in target_rows]
    baseline_values = [float(row.get("baseline_best_median_achieved_direct_speed_mm_s", math.nan)) for row in target_rows]
    custom_values = [float(row.get("custom_best_median_achieved_direct_speed_mm_s", math.nan)) for row in target_rows]
    finite_values = [
        value
        for value in baseline_values + custom_values + targets
        if math.isfinite(float(value))
    ]
    axis_max = max(finite_values) * 1.03 if finite_values else 1.0
    ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
    ax.plot(targets, baseline_values, marker="o", color="#1f77b4", linewidth=2.0, label="best historical baseline")
    ax.plot(targets, custom_values, marker="o", color="#d62728", linewidth=2.0, label="best custom middle-knot")
    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Median achieved direct speed (mm/s)")
    ax.set_title("n3 bov4mk: baseline vs custom middle-knot envelope")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _readme_text(
    *,
    targets_mm_s: tuple[float, ...],
    schedule_families: tuple[str, ...],
    seeds: tuple[int, ...],
) -> str:
    custom_lines = [
        f"- `{label}` -> normalized support `{CUSTOM_SUPPORT_BY_FAMILY[label]}`"
        for label in schedule_families
        if label in CUSTOM_SUPPORT_BY_FAMILY
    ]
    return "\n".join(
        [
            "# n3 bov4 Middle-Knot Custom Study",
            "",
            "This folder compares the frozen historical `n3` fixed-time BO3 baselines against a new custom middle-knot sweep run with bov4 mechanics.",
            "",
            "## Locked study policy",
            "",
            "- Objective unchanged: direct front tracking over `2.5-11.5 mm`.",
            f"- Target-speed range in this bundle: `{_target_range_text(targets_mm_s)} mm/s`.",
            f"- Explicit target grid: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Schedule families in this bundle: `{', '.join(schedule_families)}`.",
            f"- Seeds in this bundle: `{', '.join(str(seed) for seed in seeds)}`.",
            "- Historical baseline families are ingested only for comparison: `uniform`, `early_dense`, `late_dense`.",
            "- Historical fixed-time baselines are ingested only from `_bov3_` runs.",
            "- New custom study runs are ingested only when the run name contains `_bov4mk_` and the effective config matches the locked bov4 custom signature.",
            "- bov4 custom settings are locked: `monotone_unit_box`, `feasible_local_deterministic`, `EI`, `xi=0.01`, `init_points=2`, `n_iter=30`, `seed_with_theta0=true`, `local_refinement_points=6`, `local_refinement_sigma=0.08`.",
            f"- Theta0: `{TUNED_THETA0_C}`.",
            f"- Theta bounds: `{TUNED_THETA_BOUNDS_C}`.",
            "",
            "## Custom middle-knot grid",
            "",
            *custom_lines,
            "",
            "## Metric definitions",
            "",
            "- `speed_success`: relative error of direct front speed versus target, over `2.5-11.5 mm`, threshold `<= 5%`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref|` over the same front-control interval in time, threshold `<= 0.5 C`.",
            "- `overall_success`: both conditions true.",
            "- `robustly_recovered`: all requested seeds satisfy `speed_success`, median speed error is `<= 5%`, and robustness diagnostics do not indicate a plateau split or persistent late best.",
            "",
            "## Interpretation rule",
            "",
            "- `n5` is frozen in this phase; the decision question here is whether moving the middle knot in `n3`, with the improved bov4 search mechanics, is already enough to improve the high-speed envelope before revisiting a larger parameterization.",
            "",
        ]
    )


def _scope_note_text(*, targets_mm_s: tuple[float, ...], schedule_families: tuple[str, ...]) -> str:
    return "\n".join(
        [
            "# Scope Note",
            "",
            "- Official admissible velocity interval: `0.0035 -> 0.013 mm/s`.",
            f"- Active BO target grid for this bundle: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Schedule families in this bundle: `{', '.join(schedule_families)}`.",
            "- Historical `n3` BO3 fixed-time baseline families are shown only for comparison.",
            "- New `custom_tau...` families with `_bov4mk_` are the only executable part of this study.",
            "- No `n4`, no fresh `n5`, and no objective changes are part of this bundle.",
            "",
        ]
    )


def _run_fieldnames() -> tuple[str, ...]:
    base = (
        "status",
        "campaign_group",
        "schedule_family",
        "base_schedule_name",
        "custom_tau_mid",
        "target_front_speed_mm_s",
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
    return base + theta_fields + ("theta_C", "knot_times_s", "normalized_support_tau")


def _target_schedule_aggregate_fieldnames() -> tuple[str, ...]:
    return (
        "target_front_speed_mm_s",
        "campaign_group",
        "schedule_family",
        "base_schedule_name",
        "custom_tau_mid",
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
        "baseline_best_schedule_family",
        "baseline_best_num_completed",
        "baseline_best_speed_success_fraction",
        "baseline_best_median_achieved_direct_speed_mm_s",
        "baseline_best_median_direct_speed_relative_error_pct",
        "baseline_best_median_plate_mae_C",
        "baseline_best_study_conclusion",
        "custom_best_schedule_family",
        "custom_best_num_completed",
        "custom_best_speed_success_fraction",
        "custom_best_median_achieved_direct_speed_mm_s",
        "custom_best_median_direct_speed_relative_error_pct",
        "custom_best_median_plate_mae_C",
        "custom_best_study_conclusion",
        "custom_minus_baseline_median_achieved_direct_speed_mm_s",
        "custom_minus_baseline_median_direct_speed_error_pct",
        "custom_beats_baseline_on_median_speed_error",
        "custom_beats_baseline_on_median_achieved_speed",
    )


def _render_campaign_script(
    *,
    title: str,
    targets_mm_s: tuple[float, ...],
    schedule_families: tuple[str, ...],
    seeds: tuple[int, ...],
    bo_runs_root: Path,
    output_root: Path,
    runner_root: Path,
    max_jobs: int,
    summary_log_name: str,
    progress_log_name: str,
    summarize: bool,
) -> str:
    repo_root = project_root()
    targets_text = " ".join(f"{value:.3f}" for value in targets_mm_s)
    seeds_text = " ".join(str(seed) for seed in seeds)
    families_text = " ".join(schedule_families)
    summarize_cmd = ""
    if summarize:
        summarize_cmd = (
            f"cd \"{repo_root}\"\n"
            "python -m code_simulation.studies.run_n3_bo4_middle_knot_custom_admissible_range \\\n"
            f"  --bo-runs-root \"{bo_runs_root}\" \\\n"
            f"  --output-root \"{output_root}\" \\\n"
            f"  --runner-root \"{runner_root}\" \\\n"
            f"  --targets-mm-s \"{','.join(f'{value:.3f}' for value in targets_mm_s)}\" \\\n"
            f"  --schedule-families \"{','.join(DEFAULT_SCHEDULE_FAMILIES)}\" \\\n"
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
            "  local schedule_family=\"$2\"",
            "  local seed=\"$3\"",
            "  local prev_run_dir=\"$4\"",
            "  local tau_mid=\"${schedule_family#custom_tau}\"",
            "  tau_mid=\"${tau_mid/p/.}\"",
            "  local tag=${target/./p}",
            f"  local run_name=\"bo_v${{tag}}_n3_${{schedule_family}}_{CUSTOM_RUN_TAG}_seed${{seed}}\"",
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
            "    --knot-time-schedule custom \\",
            "    --knot-time-custom-support-tau \"0,${tau_mid},1\" \\",
            "    \"${theta_args[@]}\" \\",
            f"    --theta-bounds={_tuned_theta_bounds_text()} \\",
            "    --seed \"$seed\" \\",
            "    --simulation-profile optimization \\",
            f"    --init-points {TUNED_INIT_POINTS} \\",
            f"    --n-iter {TUNED_N_ITER} \\",
            f"    --acquisition-kind {TUNED_ACQUISITION_KIND} \\",
            f"    --acquisition-xi {TUNED_ACQUISITION_XI:.15g} \\",
            f"    --parameterization-kind {TUNED_PARAMETERIZATION_KIND} \\",
            f"    --init-strategy {TUNED_INIT_STRATEGY} \\",
            f"    --init-local-sigma {TUNED_INIT_LOCAL_SIGMA:.15g} \\",
            f"    --init-max-attempts-per-point {TUNED_INIT_MAX_ATTEMPTS_PER_POINT} \\",
            f"    --local-refinement-points {TUNED_LOCAL_REFINEMENT_POINTS} \\",
            f"    --local-refinement-sigma {TUNED_LOCAL_REFINEMENT_SIGMA:.15g} \\",
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
            "  local schedule_family=\"$1\"",
            "  local seed=\"$2\"",
            "  shift 2",
            "  local prev_run_dir=\"\"",
            "  local target=\"\"",
            "  for target in \"$@\"; do",
            "    run_one \"$target\" \"$schedule_family\" \"$seed\" \"$prev_run_dir\"",
            "    local tag=${target/./p}",
            f"    local run_dir=\"$COARSE/bo_v${{tag}}_n3_${{schedule_family}}_{CUSTOM_RUN_TAG}_seed${{seed}}\"",
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
            f"schedule_families=({families_text})",
            f"seeds=({seeds_text})",
            "",
            "for schedule_family in \"${schedule_families[@]}\"; do",
            "  for seed in \"${seeds[@]}\"; do",
            "    run_chain \"$schedule_family\" \"$seed\" \"${targets[@]}\" &",
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
    parser = argparse.ArgumentParser(description="Summarize the n3 bov4 custom middle-knot admissible-range study.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--bo-config", type=Path, default=DEFAULT_BO_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-root", type=Path, default=DEFAULT_RUNNER_ROOT)
    parser.add_argument("--targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_TARGETS_MM_S))
    parser.add_argument("--schedule-families", default=",".join(DEFAULT_SCHEDULE_FAMILIES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_mm_s = _parse_float_list(args.targets_mm_s)
    schedule_families = tuple(part.strip() for part in str(args.schedule_families).split(",") if part.strip())
    seeds = _parse_int_list(args.seeds)
    unsupported = [label for label in schedule_families if label not in DEFAULT_SCHEDULE_FAMILIES]
    if unsupported:
        raise ValueError(f"Unsupported schedule families: {unsupported!r}")

    bo_config = load_bo_config(args.bo_config)
    if not bool(bo_config.seed_with_theta0):
        raise ValueError("This study is locked to seed_with_theta0=true.")

    runs = _scan_study_runs(
        Path(args.bo_runs_root),
        simulation_config_path=Path(args.simulation_config),
    )
    run_rows = _build_run_rows(runs=runs, targets_mm_s=targets_mm_s, schedule_families=schedule_families, seeds=seeds)
    target_schedule_aggregate = _group_rows(
        run_rows,
        group_keys=("target_front_speed_mm_s", "campaign_group", "schedule_family"),
        seeds=seeds,
    )
    target_aggregate = _build_target_aggregate(
        target_schedule_rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
    )

    if args.dry_run:
        baseline_count = sum(1 for run in runs if run.campaign_group == CAMPAIGN_GROUP_BASELINE)
        custom_count = sum(1 for run in runs if run.campaign_group == CAMPAIGN_GROUP_CUSTOM)
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Runner root: {Path(args.runner_root).resolve()}")
        print(f"Found compatible n3 baseline runs: {baseline_count}")
        print(f"Found compatible n3 custom runs   : {custom_count}")
        print(f"Completed rows in requested matrix: {sum(1 for row in run_rows if row['status'] == 'completed')}/{len(run_rows)}")
        print(f"Requested schedule families       : {', '.join(schedule_families)}")
        print("")
        print("Representative dry-run command:")
        print(_representative_dry_run_command())
        return

    output_root = Path(args.output_root)
    runner_root = Path(args.runner_root)
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))
    runner_root.mkdir(parents=True, exist_ok=True)
    (runner_root / "logs").mkdir(parents=True, exist_ok=True)

    _write_text(
        output_root / "README.md",
        _readme_text(targets_mm_s=targets_mm_s, schedule_families=schedule_families, seeds=seeds),
    )
    _write_text(
        output_root / "scope_note.md",
        _scope_note_text(targets_mm_s=targets_mm_s, schedule_families=schedule_families),
    )
    _write_csv(output_root / "study_summary.csv", _run_fieldnames(), run_rows)
    _write_csv(output_root / "target_schedule_aggregate.csv", _target_schedule_aggregate_fieldnames(), target_schedule_aggregate)
    _write_csv(output_root / "target_aggregate.csv", _target_aggregate_fieldnames(), target_aggregate)
    _write_csv(
        output_root / "decision_summary.csv",
        (
            "target_front_speed_mm_s",
            "campaign_group",
            "schedule_family",
            "num_completed",
            "num_speed_success",
            "speed_success_fraction",
            "median_achieved_direct_speed_mm_s",
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
                "campaign_group": row["campaign_group"],
                "schedule_family": row["schedule_family"],
                "num_completed": row["num_completed"],
                "num_speed_success": row["num_speed_success"],
                "speed_success_fraction": row["speed_success_fraction"],
                "median_achieved_direct_speed_mm_s": row["median_achieved_direct_speed_mm_s"],
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

    _plot_target_vs_achieved_by_schedule_family(
        output_root / "target_vs_achieved_direct_speed_by_schedule_family.png",
        rows=run_rows,
        schedule_families=schedule_families,
    )
    _plot_metric_heatmap(
        output_root / "success_fraction_by_target_and_schedule_family.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedule_families=schedule_families,
        value_key="speed_success_fraction",
        title="n3 bov4mk: speed-success fraction by target and schedule family",
        colorbar_label="Speed-success fraction across seeds",
    )
    _plot_metric_heatmap(
        output_root / "median_direct_speed_error_by_target_and_schedule_family.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedule_families=schedule_families,
        value_key="median_direct_speed_relative_error_pct",
        title="n3 bov4mk: median direct-speed error by target and schedule family",
        colorbar_label="Median direct-speed relative error (%)",
    )
    _plot_metric_heatmap(
        output_root / "median_plate_mae_by_target_and_schedule_family.png",
        rows=target_schedule_aggregate,
        targets_mm_s=targets_mm_s,
        schedule_families=schedule_families,
        value_key="median_plate_mae_C",
        title="n3 bov4mk: median plate MAE by target and schedule family",
        colorbar_label="Median plate MAE (C)",
    )
    _plot_baseline_vs_custom_envelope(
        output_root / "baseline_vs_custom_envelope.png",
        target_rows=target_aggregate,
    )

    custom_only_families = tuple(label for label in schedule_families if label in CUSTOM_TAU_BY_FAMILY)
    _write_text(
        output_root / "run_commands.sh",
        _render_campaign_script(
            title="Sequential continuation-aware queue for the current bov4 custom middle-knot study matrix.",
            targets_mm_s=targets_mm_s,
            schedule_families=custom_only_families,
            seeds=seeds,
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            runner_root=runner_root,
            max_jobs=1,
            summary_log_name="summary_current.log",
            progress_log_name="progress_current.log",
            summarize=False,
        ),
    )
    _write_text(
        output_root / "representative_dry_run_command.sh",
        "#!/usr/bin/env bash\nset -e\n\n" + _representative_dry_run_command() + "\n",
    )
    _write_text(
        runner_root / "run_n3_bov4mk_gate.sh",
        _render_campaign_script(
            title="Gate pass for n3 bov4 custom middle-knot study.",
            targets_mm_s=GATE_TARGETS_MM_S,
            schedule_families=CUSTOM_FAMILY_LABELS,
            seeds=(17,),
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            runner_root=runner_root,
            max_jobs=2,
            summary_log_name="summary_gate.log",
            progress_log_name="progress_gate.log",
            summarize=True,
        ),
    )
    _write_text(
        runner_root / "run_n3_bov4mk_full_campaign.sh",
        _render_campaign_script(
            title="Full admissible-range campaign for n3 bov4 custom middle-knot study.",
            targets_mm_s=targets_mm_s,
            schedule_families=CUSTOM_FAMILY_LABELS,
            seeds=seeds,
            bo_runs_root=Path(args.bo_runs_root),
            output_root=output_root,
            runner_root=runner_root,
            max_jobs=2,
            summary_log_name="summary_full.log",
            progress_log_name="progress_full.log",
            summarize=True,
        ),
    )

    print(f"n3 bov4 custom middle-knot study written to {output_root.resolve()}")
    print(f"  per-run summary    : {(output_root / 'study_summary.csv').resolve()}")
    print(f"  decision summary   : {(output_root / 'decision_summary.csv').resolve()}")
    print(f"  run queue          : {(output_root / 'run_commands.sh').resolve()}")


if __name__ == "__main__":
    main()
