#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.config_files import DEFAULT_SIMULATION_CONFIG_PATH, load_simulation_profile
from code_simulation.core.paths import (
    project_root,
    velocity_control_diagnostic_dir,
    velocity_control_results_dir,
)
from code_simulation.core.plate_tracking import (
    plate_tracking_success,
    summarize_plate_tracking,
    write_plate_tracking_summary_csv,
    write_plate_tracking_timeseries_csv,
)
from code_simulation.core.plotting import configure_matplotlib
from code_simulation.core.trajectory_profiles import PiecewiseLinearTemperatureProfile
from code_simulation.optimization.open_loop_workflow_config import build_knot_time_normalized_support_tau
from code_simulation.simulation.cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
from code_simulation.simulation.open_loop_cascade import build_plate_temperature_response


DEFAULT_OUTPUT_ROOT = velocity_control_diagnostic_dir(3)
DEFAULT_BO_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_FINE_RUNS_ROOT = velocity_control_results_dir()
DEFAULT_UNIFORM_TARGETS_MM_S = tuple(float(value) for value in np.arange(0.004, 0.018 + 1.0e-12, 0.001))
DEFAULT_SCHEDULE_TARGETS_MM_S = (0.006, 0.008, 0.010)
DEFAULT_SCHEDULES = ("uniform", "early_dense", "mid_dense", "late_dense")
DEFAULT_SEEDS = (17, 29, 41)
TARGET_TOL_MM_S = 5.0e-7
INTERVAL_3_TO_11_MM = "3_to_11_mm"
THETA_FAMILY_ROUND_DECIMALS = 3
SPEED_SUCCESS_REL_TOL = 0.05


@dataclass(frozen=True)
class ScheduleDefinition:
    name: str
    normalized_support_tau: tuple[float, ...]
    equivalent_to_schedule: str | None


@dataclass(frozen=True)
class RunMetrics:
    run_dir: Path
    run_name: str
    simulation_profile: str
    target_front_speed_mm_s: float
    schedule: str
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


def _configure_matplotlib() -> None:
    configure_matplotlib(plt)


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated numeric value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all values must be finite")
    return values


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated integer value")
    return values


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _single_row_csv(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    return rows[0] if rows else {}


def _best_theta_from_csv(path: Path) -> tuple[float, ...]:
    return tuple(float(row["temperature_C"]) for row in _read_csv_rows(path))


def _format_speed_tag(value_mm_s: float) -> str:
    return f"{float(value_mm_s):0.3f}".replace(".", "p")


def _matches_target(value: float, target: float) -> bool:
    return abs(float(value) - float(target)) <= TARGET_TOL_MM_S


def _family_signature(theta_C: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), THETA_FAMILY_ROUND_DECIMALS) for value in theta_C)


def _time_grid_s(horizon_s: float, dt_s: float) -> np.ndarray:
    time_s = np.arange(0.0, float(horizon_s), float(dt_s), dtype=np.float64)
    if time_s.size == 0 or time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
    if time_s[-1] < float(horizon_s) - 1.0e-12:
        time_s = np.append(time_s, float(horizon_s))
    return time_s


def _interval_speed(path: Path, interval_name: str) -> float:
    for row in _read_csv_rows(path):
        if row.get("interval") == interval_name:
            try:
                return float(row["speed_mm_s"])
            except (TypeError, ValueError, KeyError):
                return math.nan
    return math.nan


def _history_stats(path: Path) -> tuple[float, int, int, bool]:
    rows = _read_csv_rows(path)
    if not rows:
        return (math.nan, 0, 0, False)
    feasible_values: list[tuple[int, float]] = []
    for row in rows:
        is_valid = str(row.get("is_valid", "")).strip() in {"1", "true", "True", "TRUE"}
        if not is_valid:
            continue
        try:
            feasible_values.append((int(row["evaluation_index"]), float(row["objective_value"])))
        except (TypeError, ValueError, KeyError):
            continue
    n_evaluations = len(rows)
    if not feasible_values:
        return (math.nan, n_evaluations, 0, False)
    best_idx, best_objective = min(feasible_values, key=lambda item: item[1])
    late_best = bool(best_idx >= max(n_evaluations - 2, math.ceil(0.75 * n_evaluations)))
    return (float(best_objective), n_evaluations, int(best_idx), late_best)


def _normalized_support_tau_from_knot_times(
    *,
    knot_times_s: tuple[float, ...],
    horizon_s: float,
) -> tuple[float, ...]:
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("horizon_s must be finite and positive")
    return tuple(float(value) / float(horizon_s) for value in knot_times_s)


def _schedule_definitions(*, num_knots: int, schedules: tuple[str, ...]) -> tuple[ScheduleDefinition, ...]:
    definitions: list[ScheduleDefinition] = []
    for schedule_name in schedules:
        support_tau = build_knot_time_normalized_support_tau(
            num_knots=int(num_knots),
            knot_time_schedule=schedule_name,
        )
        equivalent_to = None
        for previous in definitions:
            if np.allclose(
                np.asarray(previous.normalized_support_tau, dtype=np.float64),
                np.asarray(support_tau, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            ):
                equivalent_to = previous.name
                break
        definitions.append(
            ScheduleDefinition(
                name=schedule_name,
                normalized_support_tau=tuple(float(value) for value in support_tau),
                equivalent_to_schedule=equivalent_to,
            )
        )
    return tuple(definitions)


def _infer_schedule_name(
    *,
    normalized_support_tau: tuple[float, ...],
    num_knots: int,
    schedules: tuple[str, ...],
) -> str:
    for schedule_name in schedules:
        candidate_tau = build_knot_time_normalized_support_tau(
            num_knots=int(num_knots),
            knot_time_schedule=schedule_name,
        )
        if np.allclose(
            np.asarray(normalized_support_tau, dtype=np.float64),
            np.asarray(candidate_tau, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return schedule_name
    return "custom"


def _ensure_bo_plate_tracking_artifacts(run_dir: Path, *, simulation_config_path: Path, schedules: tuple[str, ...]) -> None:
    summary_path = run_dir / "plate_tracking_summary.csv"
    timeseries_path = run_dir / "T_ref_T_plate_timeseries.csv"
    if summary_path.exists() and timeseries_path.exists():
        return
    effective_path = run_dir / "effective_config.toml"
    tracking_path = run_dir / "best_tracking_summary.csv"
    theta_path = run_dir / "best_theta_profile.csv"
    if not (effective_path.exists() and tracking_path.exists() and theta_path.exists()):
        return
    config = _read_toml(effective_path)
    run_cfg = dict(config.get("run", {}))
    initial_cfg = dict(config.get("initial_conditions", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    uncertainty_cfg = dict(config.get("temperature_uncertainty", {}))
    simulation_profile = load_simulation_profile(
        simulation_config_path,
        profile_name=str(run_cfg.get("simulation_profile", "")).strip(),
    )
    horizon_s = float(trajectory_cfg.get("horizon_s", 0.0))
    knot_times_s = tuple(float(value) for value in trajectory_cfg.get("knot_times_s", []))
    theta_C = _best_theta_from_csv(theta_path)
    if not knot_times_s or not theta_C:
        return
    profile = PiecewiseLinearTemperatureProfile(knot_times_s=knot_times_s, knot_temperatures_C=theta_C)
    response = build_plate_temperature_response(
        time_s=_time_grid_s(horizon_s, simulation_profile.cryostage_dt_s),
        T_ref_profile_C=profile,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        T_plate0_C=float(initial_cfg.get("initial_plate_temperature_C", 2.5)),
        bcs=None,
    )
    tracking_row = _single_row_csv(tracking_path)
    t_start = float(tracking_row.get("t_at_control_z_min_s", "nan"))
    t_end_raw = tracking_row.get("t_at_control_z_max_s", "nan")
    t_end = None if str(t_end_raw).strip().lower() == "nan" else float(t_end_raw)
    plate_summary, plate_series = summarize_plate_tracking(
        time_s=response.cryostage_time_s,
        T_ref_C=response.T_ref_C,
        T_plate_C=response.T_plate_C,
        tolerance_C=float(uncertainty_cfg.get("characterization_temperature_margin_C", 0.5)),
        evaluation_window_start_s=t_start,
        evaluation_window_end_s=t_end,
    )
    write_plate_tracking_summary_csv(summary_path, plate_summary)
    write_plate_tracking_timeseries_csv(timeseries_path, plate_series)


def _ensure_fine_plate_tracking_artifacts(run_dir: Path, *, simulation_config_path: Path) -> None:
    summary_path = run_dir / "plate_tracking_summary.csv"
    timeseries_path = run_dir / "T_ref_T_plate_timeseries.csv"
    if summary_path.exists() and timeseries_path.exists():
        return
    effective_path = run_dir / "effective_config.toml"
    tracking_path = run_dir / "velocity_tracking_summary.csv"
    if not (effective_path.exists() and tracking_path.exists()):
        return
    config = _read_toml(effective_path)
    run_cfg = dict(config.get("run", {}))
    initial_cfg = dict(config.get("initial_conditions", {}))
    uncertainty_cfg = dict(config.get("temperature_uncertainty", {}))
    trajectory_cfg = dict(config.get("manual_trajectory", {}))
    simulation_profile = load_simulation_profile(
        simulation_config_path,
        profile_name=str(run_cfg.get("simulation_profile", "")).strip(),
    )
    horizon_s = float(trajectory_cfg.get("horizon_s", 0.0))
    knot_times_s = tuple(float(value) for value in trajectory_cfg.get("knot_times_s", []))
    theta_C = tuple(float(value) for value in trajectory_cfg.get("theta_C", []))
    if not knot_times_s or not theta_C:
        return
    profile = PiecewiseLinearTemperatureProfile(knot_times_s=knot_times_s, knot_temperatures_C=theta_C)
    response = build_plate_temperature_response(
        time_s=_time_grid_s(horizon_s, simulation_profile.cryostage_dt_s),
        T_ref_profile_C=profile,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        T_plate0_C=float(initial_cfg.get("initial_plate_temperature_C", 2.5)),
        bcs=None,
    )
    tracking_row = _single_row_csv(tracking_path)
    t_start = float(tracking_row.get("t_at_control_z_min_s", "nan"))
    t_end_raw = tracking_row.get("t_at_control_z_max_s", "nan")
    t_end = None if str(t_end_raw).strip().lower() == "nan" else float(t_end_raw)
    plate_summary, plate_series = summarize_plate_tracking(
        time_s=response.cryostage_time_s,
        T_ref_C=response.T_ref_C,
        T_plate_C=response.T_plate_C,
        tolerance_C=float(uncertainty_cfg.get("characterization_temperature_margin_C", 0.5)),
        evaluation_window_start_s=t_start,
        evaluation_window_end_s=t_end,
    )
    write_plate_tracking_summary_csv(summary_path, plate_summary)
    write_plate_tracking_timeseries_csv(timeseries_path, plate_series)


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
    if num_knots != 3:
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
    tracking_row = _single_row_csv(tracking_path)
    plate_row = _single_row_csv(plate_path)
    best_objective, n_evaluations, best_evaluation_index, late_best = _history_stats(history_path)
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
        random_seed=(
            None if bo_cfg.get("random_seed") is None else int(bo_cfg.get("random_seed"))
        ),
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
    if num_knots != 3:
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


def _scan_runs(
    root: Path,
    *,
    simulation_config_path: Path,
    schedules: tuple[str, ...],
    loader,
) -> tuple[RunMetrics, ...]:
    summaries: list[RunMetrics] = []
    if not root.exists():
        return tuple()
    for candidate in sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and 1 <= len(path.relative_to(root).parts) <= 3
    ):
        summary = loader(candidate, simulation_config_path=simulation_config_path, schedules=schedules)
        if summary is not None:
            summaries.append(summary)
    return tuple(summaries)


def _find_bo_run(
    runs: tuple[RunMetrics, ...],
    *,
    target_mm_s: float,
    schedule: str,
    seed: int,
) -> RunMetrics | None:
    matching = [
        run for run in runs
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


def _find_fine_run(
    runs: tuple[RunMetrics, ...],
    *,
    target_mm_s: float,
    schedule: str,
) -> RunMetrics | None:
    matching = [
        run for run in runs
        if run.simulation_profile == "full_process_article"
        and _matches_target(run.target_front_speed_mm_s, target_mm_s)
        and run.schedule == schedule
    ]
    if not matching:
        return None
    return min(
        matching,
        key=lambda run: (
            math.inf if not math.isfinite(run.tracking_rmse_mm) else run.tracking_rmse_mm,
            run.run_name,
        ),
    )


def _speed_success(target_mm_s: float, achieved_mm_s: float) -> tuple[bool, float]:
    if not (math.isfinite(target_mm_s) and target_mm_s > 0.0 and math.isfinite(achieved_mm_s)):
        return (False, math.nan)
    rel_error = abs(float(achieved_mm_s) - float(target_mm_s)) / float(target_mm_s)
    return (bool(rel_error <= SPEED_SUCCESS_REL_TOL + 1.0e-12), 100.0 * rel_error)


def _failure_mode(*, status: str, speed_success: bool, temperature_success: bool, late_best_eval: bool) -> str:
    if status != "completed":
        return "infeasible_or_admissibility_limited"
    if speed_success and temperature_success:
        return "recovered"
    if late_best_eval and not speed_success:
        return "optimizer_not_converged_yet"
    if not temperature_success:
        return "temperature_tracking_not_ok"
    return "missed_target_but_temperature_ok"


def _uniform_run_name(target_mm_s: float, seed: int) -> str:
    return f"bo_v{_format_speed_tag(target_mm_s)}_n3_uniform_seed{int(seed)}"


def _schedule_run_name(target_mm_s: float, schedule: str, seed: int) -> str:
    return f"bo_v{_format_speed_tag(target_mm_s)}_n3_{schedule}_seed{int(seed)}"


def _fine_run_name(target_mm_s: float, schedule: str) -> str:
    return f"fine_confirm_v{_format_speed_tag(target_mm_s)}_n3_{schedule}"


def _target_command(run_name: str, target_mm_s: float, schedule: str, seed: int) -> str:
    repo_root = project_root()
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.optimization.run_velocity_control_bo \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        "  --num-knots 3 \\\n"
        f"  --knot-time-schedule {schedule} \\\n"
        f"  --seed {int(seed)} \\\n"
        "  --simulation-profile optimization \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _fine_command(run_name: str, target_mm_s: float, schedule: str, theta_C: tuple[float, ...]) -> str:
    repo_root = project_root()
    theta_arg = ",".join(f"{float(value):.15g}" for value in theta_C)
    return (
        f"cd {repo_root}\n"
        "python -m code_simulation.verification.run_velocity_control_evaluation \\\n"
        "  --simulation-profile full_process_article \\\n"
        f"  --target-front-speed-mm-s {target_mm_s:.3f} \\\n"
        f"  --theta-c={theta_arg} \\\n"
        "  --num-knots 3 \\\n"
        f"  --knot-time-schedule {schedule} \\\n"
        f"  --run-name {run_name} \\\n"
        "  --overwrite"
    )


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_shell_script(path: Path, header_lines: list[str], command_blocks: list[str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -e", ""]
    lines.extend(header_lines)
    if header_lines:
        lines.append("")
    lines.extend(command_blocks)
    _write_text(path, "\n".join(lines).rstrip() + "\n")


def _ensure_clean_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _uniform_rows(
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
            run = _find_bo_run(bo_runs, target_mm_s=target_mm_s, schedule="uniform", seed=seed)
            speed_success = False
            rel_error_pct = math.nan
            temperature_success = False
            family_id = ""
            if run is not None:
                speed_success, rel_error_pct = _speed_success(target_mm_s, run.achieved_direct_speed_mm_s)
                plate_summary_like = type(
                    "_Tmp",
                    (),
                    {
                        "mean_abs_plate_error_C": run.mean_abs_plate_error_C,
                        "tolerance_C": run.plate_tolerance_C,
                    },
                )
                temperature_success = plate_tracking_success(plate_summary_like)
                signature = _family_signature(run.theta_C)
                family_id = family_lookup.get(signature, "")
                if not family_id:
                    family_counter += 1
                    family_id = f"family_{family_counter}"
                    family_lookup[signature] = family_id
            status = "completed" if run is not None else "missing"
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
                "theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.theta_C),
                "knot_times_s": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.knot_times_s),
            }
            rows.append(row)
    return rows


def _schedule_rows(
    *,
    bo_runs: tuple[RunMetrics, ...],
    schedule_definitions: tuple[ScheduleDefinition, ...],
    targets_mm_s: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_mm_s in targets_mm_s:
        for seed in seeds:
            uniform_run = _find_bo_run(bo_runs, target_mm_s=target_mm_s, schedule="uniform", seed=seed)
            uniform_objective = math.nan if uniform_run is None else uniform_run.objective_value
            for definition in schedule_definitions:
                if definition.equivalent_to_schedule is not None:
                    rows.append(
                        {
                            "status": "equivalent_to_existing_schedule",
                            "target_front_speed_mm_s": float(target_mm_s),
                            "schedule": definition.name,
                            "seed": int(seed),
                            "equivalent_to_schedule": definition.equivalent_to_schedule,
                            "selected_run_name": "",
                            "objective_value": math.nan,
                            "achieved_direct_speed_mm_s": math.nan,
                            "achieved_tc_3to11_mm_s": math.nan,
                            "tracking_rmse_mm": math.nan,
                            "direct_speed_relative_error_pct": math.nan,
                            "speed_success": 0,
                            "plate_tolerance_C": math.nan,
                            "max_abs_plate_error_C": math.nan,
                            "rmse_plate_error_C": math.nan,
                            "mean_plate_error_C": math.nan,
                            "mean_abs_plate_error_C": math.nan,
                            "fraction_within_tolerance": math.nan,
                            "temperature_success": 0,
                            "overall_success": 0,
                            "best_evaluation_index": 0,
                            "n_evaluations": 0,
                            "late_best_eval": 0,
                            "failure_mode": "equivalent_to_uniform",
                            "theta_0_C": math.nan,
                            "theta_1_C": math.nan,
                            "theta_2_C": math.nan,
                            "theta_C": "",
                            "knot_times_s": "",
                            "uniform_selected_run_name": "" if uniform_run is None else uniform_run.run_name,
                            "uniform_objective_value": uniform_objective,
                            "objective_delta_vs_uniform": math.nan,
                            "improves_on_uniform": 0,
                        }
                    )
                    continue

                run = _find_bo_run(bo_runs, target_mm_s=target_mm_s, schedule=definition.name, seed=seed)
                speed_success = False
                rel_error_pct = math.nan
                temperature_success = False
                if run is not None:
                    speed_success, rel_error_pct = _speed_success(target_mm_s, run.achieved_direct_speed_mm_s)
                    plate_summary_like = type(
                        "_Tmp",
                        (),
                        {
                            "mean_abs_plate_error_C": run.mean_abs_plate_error_C,
                            "tolerance_C": run.plate_tolerance_C,
                        },
                    )
                    temperature_success = plate_tracking_success(plate_summary_like)
                status = "completed" if run is not None else "missing"
                objective_value = math.nan if run is None else run.objective_value
                rows.append(
                    {
                        "status": status,
                        "target_front_speed_mm_s": float(target_mm_s),
                        "schedule": definition.name,
                        "seed": int(seed),
                        "equivalent_to_schedule": "",
                        "selected_run_name": "" if run is None else run.run_name,
                        "objective_value": objective_value,
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
                        "theta_0_C": math.nan if run is None else run.theta_C[0],
                        "theta_1_C": math.nan if run is None else run.theta_C[1],
                        "theta_2_C": math.nan if run is None else run.theta_C[2],
                        "theta_C": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.theta_C),
                        "knot_times_s": "" if run is None else ",".join(f"{float(v):.15g}" for v in run.knot_times_s),
                        "uniform_selected_run_name": "" if uniform_run is None else uniform_run.run_name,
                        "uniform_objective_value": uniform_objective,
                        "objective_delta_vs_uniform": (
                            math.nan
                            if run is None or not math.isfinite(uniform_objective)
                            else float(uniform_objective) - float(run.objective_value)
                        ),
                        "improves_on_uniform": int(
                            run is not None
                            and math.isfinite(uniform_objective)
                            and math.isfinite(run.objective_value)
                            and float(run.objective_value) < float(uniform_objective) - 1.0e-12
                        ),
                    }
                )
    return rows


def _aggregate_rows(rows: list[dict[str, object]], *, group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, object]] = []
    for key in sorted(grouped):
        members = grouped[key]
        completed = [row for row in members if row.get("status") == "completed"]
        achieved = [float(row["achieved_direct_speed_mm_s"]) for row in completed if math.isfinite(float(row["achieved_direct_speed_mm_s"]))]
        families = {str(row["family_id"]) for row in completed if str(row.get("family_id", "")).strip()}
        aggregate = {name: value for name, value in zip(group_keys, key, strict=True)}
        aggregate.update(
            {
                "num_rows": len(members),
                "num_completed": len(completed),
                "num_speed_success": sum(int(row.get("speed_success", 0)) for row in members),
                "num_temperature_success": sum(int(row.get("temperature_success", 0)) for row in members),
                "num_overall_success": sum(int(row.get("overall_success", 0)) for row in members),
                "num_late_best_eval": sum(int(row.get("late_best_eval", 0)) for row in members),
                "num_distinct_families": len(families),
                "min_achieved_direct_speed_mm_s": math.nan if not achieved else min(achieved),
                "max_achieved_direct_speed_mm_s": math.nan if not achieved else max(achieved),
            }
        )
        aggregates.append(aggregate)
    return aggregates


def _build_decision_rows(
    *,
    schedule_rows: list[dict[str, object]],
    fine_runs: tuple[RunMetrics, ...],
    schedules: tuple[ScheduleDefinition, ...],
    targets_mm_s: tuple[float, ...],
) -> list[dict[str, object]]:
    decision_rows: list[dict[str, object]] = []
    distinct_schedules = tuple(defn.name for defn in schedules if defn.equivalent_to_schedule is None)
    for target_mm_s in targets_mm_s:
        target_rows = [
            row for row in schedule_rows
            if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
            and str(row.get("schedule")) in distinct_schedules
            and row.get("status") == "completed"
        ]
        fine_target_runs = [
            run for run in fine_runs
            if _matches_target(run.target_front_speed_mm_s, target_mm_s)
            and run.schedule in distinct_schedules
        ]
        fine_success_count = 0
        fine_success_schedules: list[str] = []
        for run in fine_target_runs:
            speed_success, _ = _speed_success(target_mm_s, run.achieved_direct_speed_mm_s)
            plate_summary_like = type(
                "_Tmp",
                (),
                {
                    "mean_abs_plate_error_C": run.mean_abs_plate_error_C,
                    "tolerance_C": run.plate_tolerance_C,
                },
            )
            if speed_success and plate_tracking_success(plate_summary_like):
                fine_success_count += 1
                fine_success_schedules.append(run.schedule)
        recovered_by_3knots = fine_success_count > 0
        all_completed_fail = bool(target_rows) and not any(int(row.get("overall_success", 0)) for row in target_rows)
        all_schedules_fail = all(
            not any(
                int(row.get("overall_success", 0))
                for row in target_rows
                if row.get("schedule") == schedule_name
            )
            for schedule_name in distinct_schedules
        )
        any_late = any(int(row.get("late_best_eval", 0)) for row in target_rows)
        family_count = len({str(row.get("theta_C", "")) for row in target_rows if str(row.get("theta_C", "")).strip()})
        achieved = [float(row["achieved_direct_speed_mm_s"]) for row in target_rows if math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))]
        clustered_away = bool(
            achieved
            and (max(achieved) - min(achieved) <= max(0.0005, 0.05 * float(target_mm_s)))
            and not any(int(row.get("overall_success", 0)) for row in target_rows)
        )
        all_temperature_fail = bool(target_rows) and all(int(row.get("temperature_success", 0)) == 0 for row in target_rows)
        likely_bo_search_limited = int((not recovered_by_3knots) and bool(target_rows) and (any_late or family_count > 1))
        likely_plant_inner_response_limited = int((not recovered_by_3knots) and all_temperature_fail)
        likely_3knot_parameterization_limited = int(
            (not recovered_by_3knots)
            and all_completed_fail
            and all_schedules_fail
            and clustered_away
            and not likely_bo_search_limited
            and not likely_plant_inner_response_limited
        )
        decision_rows.append(
            {
                "target_front_speed_mm_s": float(target_mm_s),
                "num_completed_coarse_runs": len(target_rows),
                "num_coarse_overall_successes": sum(int(row.get("overall_success", 0)) for row in target_rows),
                "num_fine_confirmations": len(fine_target_runs),
                "num_fine_successes": int(fine_success_count),
                "fine_success_schedules": ",".join(sorted(set(fine_success_schedules))),
                "recovered_by_3knots": int(recovered_by_3knots),
                "likely_bo_search_limited": int(likely_bo_search_limited),
                "likely_3knot_parameterization_limited": int(likely_3knot_parameterization_limited),
                "likely_plant_inner_response_limited": int(likely_plant_inner_response_limited),
                "min_achieved_direct_speed_mm_s": math.nan if not achieved else min(achieved),
                "max_achieved_direct_speed_mm_s": math.nan if not achieved else max(achieved),
                "num_distinct_theta_families": int(family_count),
            }
        )
    return decision_rows


def _plot_uniform_target_vs_achieved(out_path: Path, *, rows: list[dict[str, object]], y_key: str, title: str, y_label: str) -> None:
    completed = [row for row in rows if row.get("status") == "completed" and math.isfinite(float(row.get(y_key, math.nan)))]
    if not completed:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    color_by_seed = {17: "#1f77b4", 29: "#2ca02c", 41: "#d62728"}
    _configure_matplotlib()
    fig, ax = plt.subplots()
    x_all = np.asarray([float(row["target_front_speed_mm_s"]) for row in completed], dtype=np.float64)
    y_all = np.asarray([float(row[y_key]) for row in completed], dtype=np.float64)
    line_min = float(np.min(x_all))
    line_max = float(np.max(x_all))
    ax.plot([line_min, line_max], [line_min, line_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
    for seed in sorted({int(row["seed"]) for row in completed}):
        seed_rows = [row for row in completed if int(row["seed"]) == seed]
        ax.scatter(
            [float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [float(row[y_key]) for row in seed_rows],
            color=color_by_seed.get(seed, None),
            label=f"seed {seed}",
            s=36,
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_target_vs_achieved_by_schedule(
    out_path: Path,
    *,
    uniform_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> None:
    _configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharex=True, sharey=True)

    color_by_seed = {17: "#1f77b4", 29: "#2ca02c", 41: "#d62728"}
    schedule_titles = {
        "uniform": "uniform\n(full range tested)",
        "early_dense": "early_dense\n(tested: 0.006, 0.008, 0.010)",
        "late_dense": "late_dense\n(tested: 0.006, 0.008, 0.010)",
    }
    row_lookup = {
        "uniform": [
            row for row in uniform_rows
            if row.get("status") == "completed"
            and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
        ],
        "early_dense": [
            row for row in schedule_rows
            if row.get("status") == "completed"
            and str(row.get("schedule")) == "early_dense"
            and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
        ],
        "late_dense": [
            row for row in schedule_rows
            if row.get("status") == "completed"
            and str(row.get("schedule")) == "late_dense"
            and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
        ],
    }

    x_values: list[float] = []
    y_values: list[float] = []
    for rows in row_lookup.values():
        x_values.extend(float(row["target_front_speed_mm_s"]) for row in rows)
        y_values.extend(float(row["achieved_direct_speed_mm_s"]) for row in rows)
    if not x_values or not y_values:
        for ax in axes:
            ax.axis("off")
            ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return

    x_max = max(max(x_values), max(y_values)) * 1.03
    for ax, schedule in zip(axes, ("uniform", "early_dense", "late_dense"), strict=True):
        rows = row_lookup[schedule]
        ax.plot([0.0, x_max], [0.0, x_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
        for seed in sorted({int(row["seed"]) for row in rows}):
            seed_rows = [row for row in rows if int(row["seed"]) == seed]
            ax.scatter(
                [float(row["target_front_speed_mm_s"]) for row in seed_rows],
                [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
                color=color_by_seed.get(seed, None),
                label=f"seed {seed}",
                s=42,
            )
        ax.set_title(schedule_titles[schedule])
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, x_max)
        ax.set_xlabel("Target speed (mm/s)")
        if ax is axes[0]:
            ax.set_ylabel("Achieved direct speed (mm/s)")
        ax.legend(loc="upper left")

    fig.suptitle("3-knot target vs achieved direct speed by knot-time schedule", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_schedule_comparison(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> None:
    plotted_schedules = [definition.name for definition in schedule_definitions if definition.equivalent_to_schedule is None]
    completed = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("schedule") in plotted_schedules
        and math.isfinite(float(row.get("achieved_direct_speed_mm_s", math.nan)))
    ]
    if not completed:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title("3-knot spacing comparison for tested targets only (0.006, 0.008, 0.010 mm/s)")
        ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    color_by_schedule = {
        "uniform": "#1f77b4",
        "early_dense": "#2ca02c",
        "late_dense": "#d62728",
    }
    best_by_target_schedule: dict[tuple[float, str], dict[str, object]] = {}
    for row in completed:
        key = (float(row["target_front_speed_mm_s"]), str(row["schedule"]))
        current = best_by_target_schedule.get(key)
        if current is None or float(row["objective_value"]) < float(current["objective_value"]):
            best_by_target_schedule[key] = row
    _configure_matplotlib()
    fig, ax = plt.subplots()
    x_targets = sorted({float(row["target_front_speed_mm_s"]) for row in completed})
    ax.plot(x_targets, x_targets, "--", color="0.5", linewidth=1.2, label="target = achieved")
    for schedule_name in plotted_schedules:
        schedule_rows = [
            row for (target, schedule), row in sorted(best_by_target_schedule.items())
            if schedule == schedule_name
        ]
        if not schedule_rows:
            continue
        x = np.asarray([float(row["target_front_speed_mm_s"]) for row in schedule_rows], dtype=np.float64)
        y = np.asarray([float(row["achieved_direct_speed_mm_s"]) for row in schedule_rows], dtype=np.float64)
        order = np.argsort(x)
        ax.plot(
            x[order],
            y[order],
            marker="o",
            linewidth=1.5,
            color=color_by_schedule.get(schedule_name, None),
            label=schedule_name,
        )
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title("3-knot spacing comparison for tested targets only (0.006, 0.008, 0.010 mm/s)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_theta_profiles(
    out_path: Path,
    *,
    rows: list[dict[str, object]],
    schedule_targets_mm_s: tuple[float, ...],
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> None:
    plotted_schedules = [definition.name for definition in schedule_definitions if definition.equivalent_to_schedule is None]
    completed = [
        row for row in rows
        if row.get("status") == "completed"
        and row.get("schedule") in plotted_schedules
    ]
    if not completed:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title("Best schedule temperature profiles for tested targets only")
        ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return
    color_by_schedule = {
        "uniform": "#1f77b4",
        "early_dense": "#2ca02c",
        "late_dense": "#d62728",
    }
    best_rows: dict[tuple[float, str], dict[str, object]] = {}
    for row in completed:
        key = (float(row["target_front_speed_mm_s"]), str(row["schedule"]))
        current = best_rows.get(key)
        if current is None or float(row["objective_value"]) < float(current["objective_value"]):
            best_rows[key] = row
    _configure_matplotlib()
    fig, axes = plt.subplots(len(schedule_targets_mm_s), 1, figsize=(8.0, 3.2 * len(schedule_targets_mm_s)), sharex=True)
    if len(schedule_targets_mm_s) == 1:
        axes = [axes]
    for ax, target_mm_s in zip(axes, schedule_targets_mm_s, strict=True):
        for schedule_name in plotted_schedules:
            row = best_rows.get((float(target_mm_s), schedule_name))
            if row is None:
                continue
            knot_times = [float(value) for value in str(row["knot_times_s"]).split(",") if value]
            theta = [float(value) for value in str(row["theta_C"]).split(",") if value]
            ax.plot(knot_times, theta, marker="o", linewidth=1.6, color=color_by_schedule.get(schedule_name, None), label=schedule_name)
        ax.set_title(f"Tested target {target_mm_s:.3f} mm/s")
        ax.set_ylabel("T_ref knot temperature (C)")
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time since cooling start (s)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _completed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("status") == "completed"]


def _uniform_error_summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in _completed_rows(rows):
        target = _finite_float(row.get("target_front_speed_mm_s"))
        if target is None:
            continue
        grouped.setdefault(target, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for target in sorted(grouped):
        members = grouped[target]
        rel_errors = [
            value
            for value in (_finite_float(row.get("direct_speed_relative_error_pct")) for row in members)
            if value is not None
        ]
        overall_success_fraction = float(
            sum(int(row.get("overall_success", 0)) for row in members) / float(len(members))
        )
        late_best_fraction = float(
            sum(int(row.get("late_best_eval", 0)) for row in members) / float(len(members))
        )
        summary_rows.append(
            {
                "target_front_speed_mm_s": target,
                "n_completed": len(members),
                "best_relative_speed_error_pct": math.nan if not rel_errors else min(rel_errors),
                "worst_relative_speed_error_pct": math.nan if not rel_errors else max(rel_errors),
                "overall_success_fraction": overall_success_fraction,
                "late_best_fraction": late_best_fraction,
            }
        )
    return summary_rows


def _schedule_error_summary_rows(
    rows: list[dict[str, object]],
    *,
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> list[dict[str, object]]:
    plotted_schedules = tuple(
        definition.name for definition in schedule_definitions if definition.equivalent_to_schedule is None
    )
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in _completed_rows(rows):
        schedule = str(row.get("schedule", "")).strip()
        if schedule not in plotted_schedules:
            continue
        target = _finite_float(row.get("target_front_speed_mm_s"))
        if target is None:
            continue
        grouped.setdefault((target, schedule), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for target, schedule in sorted(grouped):
        members = grouped[(target, schedule)]
        rel_errors = [
            value
            for value in (_finite_float(row.get("direct_speed_relative_error_pct")) for row in members)
            if value is not None
        ]
        summary_rows.append(
            {
                "target_front_speed_mm_s": target,
                "schedule": schedule,
                "n_completed": len(members),
                "best_relative_speed_error_pct": math.nan if not rel_errors else min(rel_errors),
                "median_relative_speed_error_pct": math.nan if not rel_errors else float(np.median(rel_errors)),
                "worst_relative_speed_error_pct": math.nan if not rel_errors else max(rel_errors),
                "overall_success_fraction": float(
                    sum(int(row.get("overall_success", 0)) for row in members) / float(len(members))
                ),
                "late_best_fraction": float(
                    sum(int(row.get("late_best_eval", 0)) for row in members) / float(len(members))
                ),
            }
        )
    return summary_rows


def _read_front_series(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = sorted(
        path
        for path in run_dir.glob("*_front.csv")
        if path.is_file() and not path.name.endswith("_front_curve.csv")
    )
    if not candidates:
        return None
    rows = _read_csv_rows(candidates[0])
    if not rows:
        return None
    time_s: list[float] = []
    z_front_mm: list[float] = []
    for row in rows:
        try:
            time_value = float(row.get("time_since_fill_s", row.get("time_s", "nan")))
            z_value = float(row.get("z_front_mm", "nan"))
        except (TypeError, ValueError):
            continue
        time_s.append(time_value)
        z_front_mm.append(z_value)
    if not time_s:
        return None
    return (np.asarray(time_s, dtype=np.float64), np.asarray(z_front_mm, dtype=np.float64))


def _plot_uniform_full_range_diagnostic(
    out_path: Path,
    *,
    uniform_summary: list[dict[str, object]],
) -> None:
    if not uniform_summary:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title("N3 uniform full-range diagnostic")
        ax.text(0.02, 0.95, "No completed uniform runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return

    x = np.asarray([float(row["target_front_speed_mm_s"]) for row in uniform_summary], dtype=np.float64)
    best_error = np.asarray([float(row["best_relative_speed_error_pct"]) for row in uniform_summary], dtype=np.float64)
    worst_error = np.asarray([float(row["worst_relative_speed_error_pct"]) for row in uniform_summary], dtype=np.float64)
    success_fraction = np.asarray([100.0 * float(row["overall_success_fraction"]) for row in uniform_summary], dtype=np.float64)
    late_fraction = np.asarray([100.0 * float(row["late_best_fraction"]) for row in uniform_summary], dtype=np.float64)

    _configure_matplotlib()
    fig, (ax_error, ax_fraction) = plt.subplots(2, 1, figsize=(10.5, 7.6), sharex=True)

    ax_error.axhspan(0.0, 5.0, color="#dff0d8", alpha=0.65, zorder=0)
    ax_error.fill_between(x, best_error, worst_error, color="0.82", alpha=0.9, label="seed error range")
    ax_error.plot(x, best_error, color="#1f77b4", linewidth=1.8, marker="o", label="best seed error")
    ax_error.axhline(5.0, color="0.35", linestyle="--", linewidth=1.0)
    ax_error.set_ylabel("Direct speed error (%)")
    ax_error.set_title("N3 uniform sweep across the full target range")
    ax_error.legend(loc="upper left")

    ax_fraction.plot(x, success_fraction, color="#2ca02c", linewidth=1.8, marker="o", label="overall success fraction")
    ax_fraction.plot(x, late_fraction, color="#7f3c8d", linewidth=1.8, marker="s", label="late-best fraction")
    ax_fraction.set_xlabel("Target speed (mm/s)")
    ax_fraction.set_ylabel("Fraction across seeds (%)")
    ax_fraction.set_ylim(0.0, 100.0)
    ax_fraction.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_schedule_error_heatmap(
    out_path: Path,
    *,
    schedule_summary: list[dict[str, object]],
    schedule_definitions: tuple[ScheduleDefinition, ...],
    value_key: str,
    title: str,
    colorbar_label: str,
    value_format: str,
) -> None:
    plotted_schedules = [
        definition.name for definition in schedule_definitions if definition.equivalent_to_schedule is None
    ]
    targets = sorted({float(row["target_front_speed_mm_s"]) for row in schedule_summary})
    if not plotted_schedules or not targets:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.02, 0.95, "No completed spacing-comparison runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return

    matrix = np.full((len(plotted_schedules), len(targets)), np.nan, dtype=np.float64)
    for i, schedule in enumerate(plotted_schedules):
        for j, target in enumerate(targets):
            matches = [
                row for row in schedule_summary
                if str(row.get("schedule")) == schedule
                and _matches_target(float(row["target_front_speed_mm_s"]), target)
            ]
            if not matches:
                continue
            matrix[i, j] = float(matches[0][value_key])

    _configure_matplotlib()
    fig, ax = plt.subplots(figsize=(1.8 * len(targets) + 2.8, 1.6 * len(plotted_schedules) + 2.2))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{target:.3f}" for target in targets])
    ax.set_yticks(np.arange(len(plotted_schedules)))
    ax.set_yticklabels(plotted_schedules)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Knot-time schedule")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "NA" if not math.isfinite(float(value)) else format(float(value), value_format)
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_z_front_overlays(
    out_dir: Path,
    *,
    fine_runs: tuple[RunMetrics, ...],
) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[float, list[RunMetrics]] = {}
    for run in fine_runs:
        grouped.setdefault(float(run.target_front_speed_mm_s), []).append(run)

    generated_rows: list[dict[str, object]] = []
    color_by_schedule = {
        "uniform": "#1f77b4",
        "early_dense": "#2ca02c",
        "late_dense": "#d62728",
    }

    for target in sorted(grouped):
        runs = sorted(grouped[target], key=lambda run: (run.schedule, run.run_name))
        series_by_run: list[tuple[RunMetrics, np.ndarray, np.ndarray]] = []
        for run in runs:
            series = _read_front_series(run.run_dir)
            if series is None:
                continue
            time_s, z_front_mm = series
            finite = np.isfinite(time_s) & np.isfinite(z_front_mm)
            if not np.any(finite):
                continue
            series_by_run.append((run, time_s[finite], z_front_mm[finite]))
        if not series_by_run:
            continue

        _configure_matplotlib()
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        for run, time_s, z_front_mm in series_by_run:
            label = run.schedule if len(series_by_run) == 1 else f"{run.schedule}: {run.run_name}"
            ax.plot(
                time_s,
                z_front_mm,
                linewidth=1.7,
                color=color_by_schedule.get(run.schedule, None),
                label=label,
            )
            expected_end_s = float(run.t_at_control_z_min_s) + (11.5 - 2.5) / float(target)
            ref_mask = (time_s >= float(run.t_at_control_z_min_s)) & (time_s <= expected_end_s + 1.0e-12)
            if np.any(ref_mask):
                z_ref_mm = 2.5 + float(target) * (time_s[ref_mask] - float(run.t_at_control_z_min_s))
                ax.plot(
                    time_s[ref_mask],
                    z_ref_mm,
                    linestyle="--",
                    linewidth=1.4,
                    color="0.20",
                    alpha=0.9,
                    label="target reference" if run is series_by_run[0][0] else None,
                )

        ax.set_xlabel("Time since fill (s)")
        ax.set_ylabel("z_front (mm)")
        ax.set_title(f"z_front simulated vs target reference according to imposed velocity: {target:.3f} mm/s")
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = out_dir / f"z_front_simulated_vs_target_reference_v{_format_speed_tag(target)}.png"
        fig.savefig(out_path)
        plt.close(fig)
        generated_rows.append(
            {
                "target_front_speed_mm_s": target,
                "num_curves": len(series_by_run),
                "output_plot": out_path.name,
                "run_names": ",".join(run.run_name for run, _, _ in series_by_run),
                "schedules": ",".join(run.schedule for run, _, _ in series_by_run),
            }
        )

    return generated_rows


def _plot_n3_error_diagnostic(
    out_path: Path,
    *,
    uniform_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> None:
    uniform_summary = _uniform_error_summary_rows(uniform_rows)
    schedule_summary = _schedule_error_summary_rows(schedule_rows, schedule_definitions=schedule_definitions)
    if not uniform_summary and not schedule_summary:
        _configure_matplotlib()
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.set_title("3-knot error diagnostic")
        ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")
        fig.savefig(out_path)
        plt.close(fig)
        return

    color_by_schedule = {
        "uniform": "#1f77b4",
        "early_dense": "#2ca02c",
        "late_dense": "#d62728",
    }

    _configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    ax_uniform_error, ax_uniform_late, ax_schedule_error, ax_schedule_late = axes.flat

    if uniform_summary:
        x = np.asarray([float(row["target_front_speed_mm_s"]) for row in uniform_summary], dtype=np.float64)
        best_error = np.asarray([float(row["best_relative_speed_error_pct"]) for row in uniform_summary], dtype=np.float64)
        worst_error = np.asarray([float(row["worst_relative_speed_error_pct"]) for row in uniform_summary], dtype=np.float64)
        success_fraction = np.asarray([100.0 * float(row["overall_success_fraction"]) for row in uniform_summary], dtype=np.float64)
        late_fraction = np.asarray([100.0 * float(row["late_best_fraction"]) for row in uniform_summary], dtype=np.float64)

        ax_uniform_error.axhspan(0.0, 5.0, color="#dff0d8", alpha=0.65, zorder=0)
        ax_uniform_error.vlines(x, best_error, worst_error, color="0.70", linewidth=2.0, label="seed range")
        scatter = ax_uniform_error.scatter(
            x,
            best_error,
            c=success_fraction,
            cmap="viridis",
            vmin=0.0,
            vmax=100.0,
            s=56,
            edgecolors="black",
            linewidths=0.4,
            label="best seed",
            zorder=3,
        )
        cbar = fig.colorbar(scatter, ax=ax_uniform_error, fraction=0.046, pad=0.04)
        cbar.set_label("Overall-success rate across seeds (%)")
        ax_uniform_error.axhline(5.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_uniform_error.set_title("Uniform sweep: speed error across the full target range")
        ax_uniform_error.set_xlabel("Target speed (mm/s)")
        ax_uniform_error.set_ylabel("Direct speed error (%)")
        ax_uniform_error.set_ylim(bottom=0.0)
        ax_uniform_error.legend(loc="upper left")

        ax_uniform_late.plot(x, late_fraction, marker="o", color="#7f3c8d", linewidth=1.6)
        ax_uniform_late.set_title("Uniform sweep: how often the best point arrived late")
        ax_uniform_late.set_xlabel("Target speed (mm/s)")
        ax_uniform_late.set_ylabel("Late-best fraction across seeds (%)")
        ax_uniform_late.set_ylim(0.0, 100.0)
    else:
        for ax, title in (
            (ax_uniform_error, "Uniform sweep: speed error across the full target range"),
            (ax_uniform_late, "Uniform sweep: how often the best point arrived late"),
        ):
            ax.axis("off")
            ax.set_title(title)
            ax.text(0.02, 0.95, "No completed runs available yet.", va="top", ha="left")

    if schedule_summary:
        plotted_schedules = tuple(
            definition.name for definition in schedule_definitions if definition.equivalent_to_schedule is None
        )
        for schedule in plotted_schedules:
            members = [row for row in schedule_summary if str(row.get("schedule")) == schedule]
            if not members:
                continue
            x = np.asarray([float(row["target_front_speed_mm_s"]) for row in members], dtype=np.float64)
            best_error = np.asarray([float(row["best_relative_speed_error_pct"]) for row in members], dtype=np.float64)
            late_fraction = np.asarray([100.0 * float(row["late_best_fraction"]) for row in members], dtype=np.float64)
            success_fraction = np.asarray([100.0 * float(row["overall_success_fraction"]) for row in members], dtype=np.float64)
            order = np.argsort(x)
            ax_schedule_error.plot(
                x[order],
                best_error[order],
                marker="o",
                linewidth=1.6,
                color=color_by_schedule.get(schedule, None),
                label=schedule,
            )
            ax_schedule_late.scatter(
                best_error,
                late_fraction,
                s=np.clip(36.0 + success_fraction, 36.0, 136.0),
                color=color_by_schedule.get(schedule, None),
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
                label=schedule,
            )
            for row in members:
                x_err = float(row["best_relative_speed_error_pct"])
                y_late = 100.0 * float(row["late_best_fraction"])
                target = float(row["target_front_speed_mm_s"])
                ax_schedule_late.annotate(
                    f"{target:.3f}",
                    (x_err, y_late),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

        ax_schedule_error.axhspan(0.0, 5.0, color="#dff0d8", alpha=0.65, zorder=0)
        ax_schedule_error.axhline(5.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_schedule_error.set_title("Spacing comparison: best speed error by tested target")
        ax_schedule_error.set_xlabel("Target speed (mm/s)")
        ax_schedule_error.set_ylabel("Best direct speed error (%)")
        ax_schedule_error.set_ylim(bottom=0.0)
        ax_schedule_error.legend(loc="upper left")

        ax_schedule_late.axvline(5.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_schedule_late.axhline(50.0, color="0.55", linestyle=":", linewidth=1.0)
        ax_schedule_late.set_title("Spacing comparison: error vs optimizer saturation proxy")
        ax_schedule_late.set_xlabel("Best direct speed error (%)")
        ax_schedule_late.set_ylabel("Late-best fraction across seeds (%)")
        ax_schedule_late.set_xlim(left=0.0)
        ax_schedule_late.set_ylim(0.0, 100.0)
        ax_schedule_late.legend(loc="best")
    else:
        for ax, title in (
            (ax_schedule_error, "Spacing comparison: best speed error by tested target"),
            (ax_schedule_late, "Spacing comparison: error vs optimizer saturation proxy"),
        ):
            ax.axis("off")
            ax.set_title(title)
            ax.text(0.02, 0.95, "No completed schedule-comparison runs available yet.", va="top", ha="left")

    fig.suptitle("3-knot error diagnostic for the tested BO setup", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _error_diagnostic_note(
    *,
    uniform_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, object]],
    schedule_definitions: tuple[ScheduleDefinition, ...],
) -> str:
    uniform_summary = _uniform_error_summary_rows(uniform_rows)
    schedule_summary = _schedule_error_summary_rows(schedule_rows, schedule_definitions=schedule_definitions)
    uniform_best_successes = sum(
        1
        for row in uniform_summary
        if _finite_float(row.get("best_relative_speed_error_pct")) is not None
        and float(row["best_relative_speed_error_pct"]) <= 5.0 + 1.0e-12
    )
    schedule_best_successes = sum(
        1
        for row in schedule_summary
        if _finite_float(row.get("best_relative_speed_error_pct")) is not None
        and float(row["best_relative_speed_error_pct"]) <= 5.0 + 1.0e-12
    )
    return "\n".join(
        [
            "# N3 Error Diagnostic",
            "",
            "This figure turns the Oliveira discussion into a compact visual diagnostic for the tested `3-knot` workflow.",
            "",
            "## How to read it",
            "",
            "- The green band marks the direct speed-error acceptance threshold (`<= 5%`).",
            "- In the uniform full-range panel, each vertical segment spans the seed-to-seed error range at one target, and the colored dot marks the best seed.",
            "- The `late-best fraction` panels use the BO history as a simple proxy for whether the optimizer was still improving near the end of the run.",
            "- In the bottom-right panel, points to the right of `5%` with low late-best fraction are the strongest candidates for a real limitation of the tested `3-knot` setup.",
            "",
            "## Snapshot",
            "",
            f"- Uniform full-range targets whose best seed reached the `5%` speed-error band: `{uniform_best_successes}/{len(uniform_summary)}`.",
            f"- Tested target/schedule combinations in the spacing comparison whose best seed reached the `5%` band: `{schedule_best_successes}/{len(schedule_summary)}`.",
            "",
            "## Caveat",
            "",
            "This is evidence about the tested `3-knot` setup, not a universal proof that every possible 3-knot formulation must fail.",
            "",
        ]
    )


def _uniform_note(rows: list[dict[str, object]], *, targets_mm_s: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    completed = [row for row in rows if row.get("status") == "completed"]
    successes = [row for row in completed if int(row.get("overall_success", 0)) == 1]
    return "\n".join(
        [
            "# Uniform Target Sweep Note",
            "",
            "This stage tests only the fixed `3-knot + uniform-time` setup across the configured seed set and across the full target-speed range.",
            "",
            "## Completion",
            "",
            f"- Targets: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            f"- Completed runs: `{len(completed)}/{len(rows)}`.",
            f"- Coarse runs meeting the dual success rule: `{len(successes)}`.",
            "",
            "## Dual success rule",
            "",
            "- `speed_success`: relative error on direct front speed within `5%` over `2.5-11.5 mm`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref| <= 0.5 C` over the same front-control interval in time.",
            "",
        ]
    )


def _schedule_note(rows: list[dict[str, object]], *, schedule_definitions: tuple[ScheduleDefinition, ...], targets_mm_s: tuple[float, ...], seeds: tuple[int, ...]) -> str:
    improved = [row for row in rows if int(row.get("improves_on_uniform", 0)) == 1]
    return "\n".join(
        [
            "# Schedule Comparison Note",
            "",
            "This stage keeps `3 knots` fixed and asks whether time spacing alone changes what can be recovered, but only for the explicitly tested targets.",
            "",
            "## Compared schedules",
            "",
            *[
                (
                    f"- `{definition.name}` support: `{definition.normalized_support_tau}`."
                    if definition.equivalent_to_schedule is None
                    else (
                        f"- `{definition.name}` support: `{definition.normalized_support_tau}`; "
                        f"identical to `{definition.equivalent_to_schedule}` for `3` knots."
                    )
                )
                for definition in schedule_definitions
            ],
            "",
            f"- Targets: `{', '.join(f'{value:.3f}' for value in targets_mm_s)}` mm/s.",
            f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
            f"- Completed non-uniform improvements over uniform: `{len(improved)}`.",
            "",
            "## Dual success rule",
            "",
            "- `speed_success`: relative error on direct front speed within `5%` over `2.5-11.5 mm`.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref| <= 0.5 C` over the same front-control interval in time.",
            "",
        ]
    )


def _root_readme(
    *,
    uniform_targets_mm_s: tuple[float, ...],
    schedule_targets_mm_s: tuple[float, ...],
    seeds: tuple[int, ...],
) -> str:
    return "\n".join(
        [
            "# N3 Diagnostico",
            "",
            "This folder contains the derived diagnostics for all `3-knot` BO work.",
            "",
            "## Policy",
            "",
            "- BO objective window: `2.5-11.5 mm`.",
            "- Experimental / thermocouple span: `3.0-11.0 mm`.",
            "- `speed_success`: relative error on direct front speed over `2.5-11.5 mm`, with a `5%` threshold.",
            "- `temperature_success`: mean absolute `|T_plate - T_ref| <= 0.5 C` over the same front-control interval in time.",
            "- `tracking_rmse_mm` and thermocouple interval speeds remain diagnostic outputs, not the primary pass/fail rule.",
            "",
            "## Stages",
            "",
            f"- Uniform full-range targets: `{', '.join(f'{value:.3f}' for value in uniform_targets_mm_s)}` mm/s.",
            f"- Spacing-comparison targets actually tested: `{', '.join(f'{value:.3f}' for value in schedule_targets_mm_s)}` mm/s.",
            f"- Seed set: `{', '.join(str(seed) for seed in seeds)}`.",
            "",
            "## Key outputs",
            "",
            "- `target_vs_achived/study_summary.csv`: per-target, per-seed `uniform` results and failure modes.",
            "- `schedule_comparison/schedule_summary.csv`: per-target, per-seed spacing-comparison results only for `0.006`, `0.008`, and `0.010 mm/s`.",
            "- `decision_summary.csv`: target-level interpretation flags before any move to `4 knots`.",
            "- `uniform_full_range_diagnostic.png`: full-range `uniform` summary of direct-speed error, success rate, and late-best fraction.",
            "- `schedule_comparison/best_speed_error_heatmap.png`: spacing-comparison best error for the tested targets only.",
            "- `schedule_comparison/success_fraction_heatmap.png`: spacing-comparison success fraction for the tested targets only.",
            "- `z_front_overlays/`: `z_front(t)` overlays built only from fine runs that really have saved `front.csv` files.",
            "- `fine_confirmation_candidates.csv`: representative cases to confirm in `full_process_article`.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the 3-knot BO target and spacing decision study.")
    parser.add_argument("--bo-runs-root", type=Path, default=DEFAULT_BO_RUNS_ROOT)
    parser.add_argument("--fine-runs-root", type=Path, default=DEFAULT_FINE_RUNS_ROOT)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--uniform-targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_UNIFORM_TARGETS_MM_S))
    parser.add_argument("--schedule-targets-mm-s", default=",".join(f"{value:.3f}" for value in DEFAULT_SCHEDULE_TARGETS_MM_S))
    parser.add_argument("--schedules", default=",".join(DEFAULT_SCHEDULES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uniform_targets_mm_s = _parse_float_list(args.uniform_targets_mm_s)
    schedule_targets_mm_s = _parse_float_list(args.schedule_targets_mm_s)
    schedules = tuple(part.strip() for part in str(args.schedules).split(",") if part.strip())
    if not schedules:
        raise ValueError("--schedules must contain at least one schedule name")
    seeds = _parse_int_list(args.seeds)
    schedule_definitions = _schedule_definitions(num_knots=3, schedules=schedules)

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
    uniform_rows = _uniform_rows(bo_runs=bo_runs, targets_mm_s=uniform_targets_mm_s, seeds=seeds)
    schedule_rows = _schedule_rows(
        bo_runs=bo_runs,
        schedule_definitions=schedule_definitions,
        targets_mm_s=schedule_targets_mm_s,
        seeds=seeds,
    )
    uniform_aggregate = _aggregate_rows(uniform_rows, group_keys=("target_front_speed_mm_s",))
    schedule_aggregate = _aggregate_rows(schedule_rows, group_keys=("target_front_speed_mm_s", "schedule"))
    decision_rows = _build_decision_rows(
        schedule_rows=schedule_rows,
        fine_runs=fine_runs,
        schedules=schedule_definitions,
        targets_mm_s=schedule_targets_mm_s,
    )

    fine_candidates: list[dict[str, object]] = []
    candidate_keys: set[tuple[float, str]] = set()
    for row in schedule_rows:
        if row.get("status") != "completed":
            continue
        target = float(row["target_front_speed_mm_s"])
        schedule = str(row["schedule"])
        if (target, schedule) in candidate_keys:
            continue
        if int(row.get("overall_success", 0)) == 1 or int(row.get("improves_on_uniform", 0)) == 1:
            run = _find_bo_run(bo_runs, target_mm_s=target, schedule=schedule, seed=int(row["seed"]))
            if run is None:
                continue
            fine = _find_fine_run(fine_runs, target_mm_s=target, schedule=schedule)
            fine_candidates.append(
                {
                    "status": "completed" if fine is not None else "pending",
                    "target_front_speed_mm_s": target,
                    "schedule": schedule,
                    "source_run_name": run.run_name,
                    "source_seed": int(row["seed"]),
                    "theta_0_C": run.theta_C[0],
                    "theta_1_C": run.theta_C[1],
                    "theta_2_C": run.theta_C[2],
                    "recommended_confirmation_run_name": _fine_run_name(target, schedule),
                    "matched_fine_run_name": "" if fine is None else fine.run_name,
                    "fine_achieved_direct_speed_mm_s": math.nan if fine is None else fine.achieved_direct_speed_mm_s,
                    "fine_tracking_rmse_mm": math.nan if fine is None else fine.tracking_rmse_mm,
                    "confirmation_command": _fine_command(_fine_run_name(target, schedule), target, schedule, run.theta_C),
                }
            )
            candidate_keys.add((target, schedule))

    if args.dry_run:
        print(f"Output root: {Path(args.output_root).resolve()}")
        print(f"Found BO runs: {len(bo_runs)}")
        print(f"Found fine runs: {len(fine_runs)}")
        print(f"Uniform completed: {sum(1 for row in uniform_rows if row['status'] == 'completed')}/{len(uniform_rows)}")
        print(f"Schedule completed: {sum(1 for row in schedule_rows if row['status'] == 'completed')}/{len(schedule_rows)}")
        return

    output_root = Path(args.output_root)
    uniform_dir = output_root / "target_vs_achived"
    schedule_dir = output_root / "schedule_comparison"
    z_front_overlay_dir = output_root / "z_front_overlays"
    _ensure_clean_directory(output_root, overwrite=bool(args.overwrite))
    uniform_dir.mkdir(parents=True, exist_ok=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)
    z_front_overlay_dir.mkdir(parents=True, exist_ok=True)

    _write_text(
        output_root / "README.md",
        _root_readme(
            uniform_targets_mm_s=uniform_targets_mm_s,
            schedule_targets_mm_s=schedule_targets_mm_s,
            seeds=seeds,
        ),
    )

    _write_csv(
        uniform_dir / "study_summary.csv",
        fieldnames=(
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
            "theta_C",
            "knot_times_s",
        ),
        rows=uniform_rows,
    )
    _write_csv(
        uniform_dir / "target_aggregate.csv",
        fieldnames=(
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
        ),
        rows=uniform_aggregate,
    )
    _write_text(uniform_dir / "study_note.md", _uniform_note(uniform_rows, targets_mm_s=uniform_targets_mm_s, seeds=seeds))
    _plot_uniform_target_vs_achieved(
        uniform_dir / "target_vs_achieved_direct_speed.png",
        rows=uniform_rows,
        y_key="achieved_direct_speed_mm_s",
        title="3-knot uniform sweep: target vs achieved direct speed",
        y_label="Achieved direct speed (mm/s)",
    )
    _plot_target_vs_achieved_by_schedule(
        uniform_dir / "target_vs_achieved_direct_speed_by_schedule.png",
        uniform_rows=uniform_rows,
        schedule_rows=schedule_rows,
        schedule_definitions=schedule_definitions,
    )
    _plot_uniform_target_vs_achieved(
        uniform_dir / "target_vs_achieved_tc_speed.png",
        rows=uniform_rows,
        y_key="achieved_tc_3to11_mm_s",
        title="3-knot uniform sweep: target vs achieved TC 3.0->11.0 mm speed",
        y_label="Achieved TC 3.0->11.0 mm speed (mm/s)",
    )
    _plot_uniform_full_range_diagnostic(
        output_root / "uniform_full_range_diagnostic.png",
        uniform_summary=_uniform_error_summary_rows(uniform_rows),
    )
    uniform_commands: list[str] = []
    for target_mm_s in uniform_targets_mm_s:
        for seed in seeds:
            matching = [
                row for row in uniform_rows
                if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
                and int(row["seed"]) == seed
            ]
            row = matching[0]
            if row["status"] == "completed":
                uniform_commands.append(f"# Completed target {target_mm_s:.3f} mm/s, seed {seed}: {row['selected_run_name']}")
            else:
                uniform_commands.extend([_target_command(_uniform_run_name(target_mm_s, seed), target_mm_s, "uniform", seed), ""])
        uniform_commands.append("")
    _write_shell_script(
        uniform_dir / "run_commands.sh",
        header_lines=[
            "# Stage 1 uniform-target sweep commands across the configured seed set.",
            "# Completed runs are listed as comments; missing runs remain executable blocks.",
        ],
        command_blocks=uniform_commands,
    )

    _write_csv(
        schedule_dir / "schedule_support_tau.csv",
        fieldnames=("schedule", "normalized_support_tau", "equivalent_to_schedule"),
        rows=[
            {
                "schedule": definition.name,
                "normalized_support_tau": ",".join(f"{value:.15g}" for value in definition.normalized_support_tau),
                "equivalent_to_schedule": "" if definition.equivalent_to_schedule is None else definition.equivalent_to_schedule,
            }
            for definition in schedule_definitions
        ],
    )
    _write_csv(
        schedule_dir / "schedule_summary.csv",
        fieldnames=(
            "status",
            "target_front_speed_mm_s",
            "schedule",
            "seed",
            "equivalent_to_schedule",
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
            "theta_0_C",
            "theta_1_C",
            "theta_2_C",
            "theta_C",
            "knot_times_s",
            "uniform_selected_run_name",
            "uniform_objective_value",
            "objective_delta_vs_uniform",
            "improves_on_uniform",
        ),
        rows=schedule_rows,
    )
    _write_csv(
        schedule_dir / "schedule_aggregate.csv",
        fieldnames=(
            "target_front_speed_mm_s",
            "schedule",
            "num_rows",
            "num_completed",
            "num_speed_success",
            "num_temperature_success",
            "num_overall_success",
            "num_late_best_eval",
            "num_distinct_families",
            "min_achieved_direct_speed_mm_s",
            "max_achieved_direct_speed_mm_s",
        ),
        rows=schedule_aggregate,
    )
    _write_text(
        schedule_dir / "study_note.md",
        _schedule_note(
            schedule_rows,
            schedule_definitions=schedule_definitions,
            targets_mm_s=schedule_targets_mm_s,
            seeds=seeds,
        ),
    )
    _plot_schedule_comparison(
        schedule_dir / "schedule_comparison_by_target.png",
        rows=schedule_rows,
        schedule_definitions=schedule_definitions,
    )
    _plot_schedule_error_heatmap(
        schedule_dir / "best_speed_error_heatmap.png",
        schedule_summary=_schedule_error_summary_rows(schedule_rows, schedule_definitions=schedule_definitions),
        schedule_definitions=schedule_definitions,
        value_key="best_relative_speed_error_pct",
        title="3-knot spacing comparison: best speed error for tested targets only",
        colorbar_label="Best direct speed error (%)",
        value_format=".2f",
    )
    _plot_schedule_error_heatmap(
        schedule_dir / "success_fraction_heatmap.png",
        schedule_summary=_schedule_error_summary_rows(schedule_rows, schedule_definitions=schedule_definitions),
        schedule_definitions=schedule_definitions,
        value_key="overall_success_fraction",
        title="3-knot spacing comparison: success fraction for tested targets only",
        colorbar_label="Overall success fraction across seeds",
        value_format=".2f",
    )
    _plot_theta_profiles(
        schedule_dir / "best_schedule_theta_profiles.png",
        rows=schedule_rows,
        schedule_targets_mm_s=schedule_targets_mm_s,
        schedule_definitions=schedule_definitions,
    )
    schedule_commands: list[str] = []
    for target_mm_s in schedule_targets_mm_s:
        schedule_commands.append(f"# Target {target_mm_s:.3f} mm/s")
        for seed in seeds:
            schedule_commands.append(f"# Seed {seed}")
            for definition in schedule_definitions:
                if definition.equivalent_to_schedule is not None:
                    schedule_commands.append(
                        f"# Skip {definition.name}: analytically identical to {definition.equivalent_to_schedule} for 3 knots."
                    )
                    continue
                matching = [
                    row for row in schedule_rows
                    if _matches_target(float(row["target_front_speed_mm_s"]), target_mm_s)
                    and str(row["schedule"]) == definition.name
                    and int(row["seed"]) == seed
                ]
                row = matching[0]
                if row["status"] == "completed":
                    schedule_commands.append(f"# Completed {definition.name}, seed {seed}: {row['selected_run_name']}")
                else:
                    schedule_commands.extend([
                        _target_command(_schedule_run_name(target_mm_s, definition.name, seed), target_mm_s, definition.name, seed),
                        "",
                    ])
            schedule_commands.append("")
    _write_shell_script(
        schedule_dir / "run_commands.sh",
        header_lines=[
            "# Stage 2 3-knot schedule-comparison commands across the configured seed set.",
            "# `mid_dense` is analytically identical to `uniform` for 3 knots and is therefore listed as a skip comment.",
        ],
        command_blocks=schedule_commands,
    )

    _write_csv(
        output_root / "decision_summary.csv",
        fieldnames=(
            "target_front_speed_mm_s",
            "num_completed_coarse_runs",
            "num_coarse_overall_successes",
            "num_fine_confirmations",
            "num_fine_successes",
            "fine_success_schedules",
            "recovered_by_3knots",
            "likely_bo_search_limited",
            "likely_3knot_parameterization_limited",
            "likely_plant_inner_response_limited",
            "min_achieved_direct_speed_mm_s",
            "max_achieved_direct_speed_mm_s",
            "num_distinct_theta_families",
        ),
        rows=decision_rows,
    )
    _write_csv(
        output_root / "uniform_error_summary.csv",
        fieldnames=(
            "target_front_speed_mm_s",
            "n_completed",
            "best_relative_speed_error_pct",
            "worst_relative_speed_error_pct",
            "overall_success_fraction",
            "late_best_fraction",
        ),
        rows=_uniform_error_summary_rows(uniform_rows),
    )
    _write_csv(
        output_root / "schedule_error_summary.csv",
        fieldnames=(
            "target_front_speed_mm_s",
            "schedule",
            "n_completed",
            "best_relative_speed_error_pct",
            "median_relative_speed_error_pct",
            "worst_relative_speed_error_pct",
            "overall_success_fraction",
            "late_best_fraction",
        ),
        rows=_schedule_error_summary_rows(schedule_rows, schedule_definitions=schedule_definitions),
    )
    _write_csv(
        output_root / "z_front_overlay_index.csv",
        fieldnames=(
            "target_front_speed_mm_s",
            "num_curves",
            "output_plot",
            "run_names",
            "schedules",
        ),
        rows=_plot_z_front_overlays(
            z_front_overlay_dir,
            fine_runs=fine_runs,
        ),
    )
    _write_text(
        output_root / "scope_note.md",
        "\n".join(
            [
                "# N3 Scope Note",
                "",
                "- Full-range coverage currently exists only for `uniform`.",
                "- Spacing comparison currently exists only for `0.006`, `0.008`, and `0.010 mm/s`.",
                "- `z_front(t)` overlay plots are generated only for fine runs that actually have saved `front.csv` files.",
                "- No coarse-only run is given a synthetic or reconstructed `z_front(t)` curve.",
                "",
            ]
        ),
    )

    _write_csv(
        output_root / "fine_confirmation_candidates.csv",
        fieldnames=(
            "status",
            "target_front_speed_mm_s",
            "schedule",
            "source_run_name",
            "source_seed",
            "theta_0_C",
            "theta_1_C",
            "theta_2_C",
            "recommended_confirmation_run_name",
            "matched_fine_run_name",
            "fine_achieved_direct_speed_mm_s",
            "fine_tracking_rmse_mm",
            "confirmation_command",
        ),
        rows=fine_candidates,
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
            "# Fine-confirmation commands for representative 3-knot cases.",
            "# Completed confirmations are listed as comments.",
        ],
        command_blocks=fine_commands,
    )

    print(f"3-knot BO decision study written to {output_root.resolve()}")
    print(f"  uniform sweep   : {(uniform_dir / 'study_summary.csv').resolve()}")
    print(f"  schedule compare: {(schedule_dir / 'schedule_summary.csv').resolve()}")
    print(f"  decision summary: {(output_root / 'decision_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
