from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code_simulation.optimization.open_loop_problem import (
    FrontTrajectory,
    OpenLoopObjectiveResult,
    PreparedOpenLoopCandidate,
    load_active_front_observable,
    prepare_open_loop_candidate,
)
from code_simulation.optimization.open_loop_workflow_config import OpenLoopProblemConfig
from code_simulation.simulation.cryostage_model import CryostageModelParams
from code_simulation.core.trajectory_profiles import PiecewiseLinearTemperatureProfile


PROBE_Z_MM = (3.0, 6.2, 11.0)
THERMOCOUPLE_THRESHOLD_C = 0.0


@dataclass(frozen=True)
class ConstantVelocityObjectiveConfig:
    target_front_speed_mm_s: float
    control_z_min_mm: float
    control_z_max_mm: float

    def __post_init__(self) -> None:
        target_front_speed_mm_s = float(self.target_front_speed_mm_s)
        control_z_min_mm = float(self.control_z_min_mm)
        control_z_max_mm = float(self.control_z_max_mm)
        if not math.isfinite(target_front_speed_mm_s) or target_front_speed_mm_s <= 0.0:
            raise ValueError("target_front_speed_mm_s must be finite and positive")
        if not math.isfinite(control_z_min_mm) or not math.isfinite(control_z_max_mm):
            raise ValueError("control z limits must be finite")
        if control_z_min_mm < 0.0 or control_z_max_mm <= control_z_min_mm:
            raise ValueError("control z limits must satisfy 0 <= z_min < z_max")
        object.__setattr__(self, "target_front_speed_mm_s", target_front_speed_mm_s)
        object.__setattr__(self, "control_z_min_mm", control_z_min_mm)
        object.__setattr__(self, "control_z_max_mm", control_z_max_mm)


@dataclass(frozen=True)
class VelocityTrackingSeries:
    control_time_s: np.ndarray
    z_front_m: np.ndarray
    z_ref_m: np.ndarray
    objective_mask: np.ndarray


@dataclass(frozen=True)
class VelocityTrackingSummary:
    target_front_speed_mm_s: float
    control_z_min_mm: float
    control_z_max_mm: float
    t_at_control_z_min_s: float
    t_at_control_z_max_s: float
    expected_t_at_control_z_max_s: float
    actual_interval_speed_mm_s: float
    regression_speed_mm_s: float
    tracking_mse: float
    tracking_rmse_mm: float
    tracking_mean_error_mm: float
    tracking_max_abs_error_mm: float
    completion_penalty: float
    reached_control_z_min: bool
    reached_control_z_max: bool
    num_tracking_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "target_front_speed_mm_s": float(self.target_front_speed_mm_s),
            "control_z_min_mm": float(self.control_z_min_mm),
            "control_z_max_mm": float(self.control_z_max_mm),
            "t_at_control_z_min_s": float(self.t_at_control_z_min_s),
            "t_at_control_z_max_s": float(self.t_at_control_z_max_s),
            "expected_t_at_control_z_max_s": float(self.expected_t_at_control_z_max_s),
            "actual_interval_speed_mm_s": float(self.actual_interval_speed_mm_s),
            "regression_speed_mm_s": float(self.regression_speed_mm_s),
            "tracking_mse": float(self.tracking_mse),
            "tracking_rmse_mm": float(self.tracking_rmse_mm),
            "tracking_mean_error_mm": float(self.tracking_mean_error_mm),
            "tracking_max_abs_error_mm": float(self.tracking_max_abs_error_mm),
            "completion_penalty": float(self.completion_penalty),
            "reached_control_z_min": int(self.reached_control_z_min),
            "reached_control_z_max": int(self.reached_control_z_max),
            "num_tracking_samples": int(self.num_tracking_samples),
        }


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def first_time_at_or_above(time_s, values, threshold: float) -> float:
    time = _as_float_array(time_s, name="time_s")
    values = _as_float_array(values, name="values")
    previous_t = math.nan
    previous_v = math.nan
    for t_i, value_i in zip(time, values, strict=False):
        t_i = float(t_i)
        value_i = float(value_i)
        if not (math.isfinite(t_i) and math.isfinite(value_i)):
            continue
        if value_i >= threshold:
            if math.isfinite(previous_t) and math.isfinite(previous_v) and value_i > previous_v:
                alpha = (float(threshold) - previous_v) / (value_i - previous_v)
                return float(previous_t + alpha * (t_i - previous_t))
            return t_i
        previous_t = t_i
        previous_v = value_i
    return math.nan


def _first_downward_crossing_time_s(time_s, values_C, threshold_C: float) -> float:
    time = _as_float_array(time_s, name="time_s")
    values = _as_float_array(values_C, name="values_C")
    previous_t = math.nan
    previous_v = math.nan
    for t_i, value_i in zip(time, values, strict=False):
        t_i = float(t_i)
        value_i = float(value_i)
        if not (math.isfinite(t_i) and math.isfinite(value_i)):
            continue
        if value_i <= threshold_C:
            if math.isfinite(previous_t) and math.isfinite(previous_v) and value_i < previous_v:
                alpha = (float(threshold_C) - previous_v) / (value_i - previous_v)
                return float(previous_t + alpha * (t_i - previous_t))
            return t_i
        previous_t = t_i
        previous_v = value_i
    return math.nan


def constant_velocity_tracking(
    front_trajectory: FrontTrajectory,
    objective_config: ConstantVelocityObjectiveConfig,
    *,
    incomplete_penalty_value: float,
) -> tuple[VelocityTrackingSummary, VelocityTrackingSeries]:
    control_time_s = np.asarray(front_trajectory.time_since_fill_s, dtype=np.float64)
    z_front_m = np.asarray(front_trajectory.z_front_m, dtype=np.float64)
    z_min_m = float(objective_config.control_z_min_mm) * 1.0e-3
    z_max_m = float(objective_config.control_z_max_mm) * 1.0e-3
    span_m = z_max_m - z_min_m
    target_speed_m_s = float(objective_config.target_front_speed_mm_s) * 1.0e-3

    finite = np.isfinite(control_time_s) & np.isfinite(z_front_m) & (control_time_s >= 0.0)
    t_z_min_s = first_time_at_or_above(control_time_s[finite], z_front_m[finite], z_min_m)
    t_z_max_s = first_time_at_or_above(control_time_s[finite], z_front_m[finite], z_max_m)
    expected_interval_s = span_m / target_speed_m_s
    expected_t_z_max_s = t_z_min_s + expected_interval_s if math.isfinite(t_z_min_s) else math.nan

    z_ref_m = np.full_like(control_time_s, math.nan, dtype=np.float64)
    if math.isfinite(t_z_min_s):
        z_ref_all = z_min_m + target_speed_m_s * (control_time_s - t_z_min_s)
        in_reference_window = (
            np.isfinite(z_ref_all)
            & (z_ref_all >= z_min_m - 1.0e-12)
            & (z_ref_all <= z_max_m + 1.0e-12)
        )
        z_ref_m[in_reference_window] = z_ref_all[in_reference_window]

    objective_mask = finite & np.isfinite(z_ref_m)
    if np.any(objective_mask):
        error_m = z_front_m[objective_mask] - z_ref_m[objective_mask]
        error_norm = error_m / span_m
        tracking_mse = float(np.mean(error_norm * error_norm))
        tracking_rmse_mm = float(np.sqrt(np.mean(error_m * error_m)) * 1000.0)
        tracking_mean_error_mm = float(np.mean(error_m) * 1000.0)
        tracking_max_abs_error_mm = float(np.max(np.abs(error_m)) * 1000.0)
    else:
        tracking_mse = float(incomplete_penalty_value)
        tracking_rmse_mm = math.nan
        tracking_mean_error_mm = math.nan
        tracking_max_abs_error_mm = math.nan

    reached_z_min = math.isfinite(t_z_min_s)
    reached_z_max = math.isfinite(t_z_max_s)
    if reached_z_min and reached_z_max and t_z_max_s > t_z_min_s:
        actual_interval_speed_mm_s = float((z_max_m - z_min_m) * 1000.0 / (t_z_max_s - t_z_min_s))
        completion_penalty = 0.0
    else:
        actual_interval_speed_mm_s = math.nan
        last_front_m = float(np.nanmax(z_front_m[finite])) if np.any(finite) else math.nan
        if math.isfinite(last_front_m):
            missing_fraction = max(z_max_m - min(last_front_m, z_max_m), 0.0) / span_m
        else:
            missing_fraction = 1.0
        completion_penalty = float(incomplete_penalty_value + missing_fraction * missing_fraction)

    regression_mask = finite & (z_front_m >= z_min_m - 1.0e-12) & (z_front_m <= z_max_m + 1.0e-12)
    if np.count_nonzero(regression_mask) >= 2:
        slope_m_s, _ = np.polyfit(control_time_s[regression_mask], z_front_m[regression_mask], deg=1)
        regression_speed_mm_s = float(slope_m_s * 1000.0)
    else:
        regression_speed_mm_s = math.nan

    summary = VelocityTrackingSummary(
        target_front_speed_mm_s=float(objective_config.target_front_speed_mm_s),
        control_z_min_mm=float(objective_config.control_z_min_mm),
        control_z_max_mm=float(objective_config.control_z_max_mm),
        t_at_control_z_min_s=float(t_z_min_s),
        t_at_control_z_max_s=float(t_z_max_s),
        expected_t_at_control_z_max_s=float(expected_t_z_max_s),
        actual_interval_speed_mm_s=float(actual_interval_speed_mm_s),
        regression_speed_mm_s=float(regression_speed_mm_s),
        tracking_mse=float(tracking_mse),
        tracking_rmse_mm=float(tracking_rmse_mm),
        tracking_mean_error_mm=float(tracking_mean_error_mm),
        tracking_max_abs_error_mm=float(tracking_max_abs_error_mm),
        completion_penalty=float(completion_penalty),
        reached_control_z_min=bool(reached_z_min),
        reached_control_z_max=bool(reached_z_max),
        num_tracking_samples=int(np.count_nonzero(objective_mask)),
    )
    series = VelocityTrackingSeries(
        control_time_s=control_time_s,
        z_front_m=z_front_m,
        z_ref_m=z_ref_m,
        objective_mask=objective_mask,
    )
    return summary, series


def temperature_smoothness_penalty(
    profile: PiecewiseLinearTemperatureProfile,
    config: OpenLoopProblemConfig,
) -> float:
    knot_times_s = np.asarray(profile.knot_times_s, dtype=np.float64)
    knot_temperatures_C = np.asarray(profile.knot_temperatures_C, dtype=np.float64)
    if knot_times_s.size < 2:
        return 0.0
    slopes_C_per_s = np.diff(knot_temperatures_C) / np.diff(knot_times_s)
    temp_span_C = max(float(config.T_ref_bounds_C[1] - config.T_ref_bounds_C[0]), 1.0)
    control_window_s = max(float(knot_times_s[-1] - knot_times_s[0]), config.cryostage_dt_s)
    scaled_slopes = slopes_C_per_s * (control_window_s / temp_span_C)
    return float(np.mean(scaled_slopes * scaled_slopes))


def write_tracking_summary_csv(path: Path, summary: VelocityTrackingSummary) -> None:
    row = summary.to_dict()
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_objective_summary_csv(
    path: Path,
    *,
    objective_value: float,
    tracking_summary: VelocityTrackingSummary,
    smoothness_penalty: float,
    completion_penalty: float,
    num_objective_samples: int,
) -> Path:
    row = {
        "objective_value": float(objective_value),
        "tracking_mse": float(tracking_summary.tracking_mse),
        "smoothness_penalty": float(smoothness_penalty),
        "completion_penalty": float(completion_penalty),
        "num_objective_samples": int(num_objective_samples),
        **tracking_summary.to_dict(),
    }
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def _read_numeric_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    data = {name: [] for name in columns}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in columns:
                try:
                    data[name].append(float(row.get(name, "nan")))
                except (TypeError, ValueError):
                    data[name].append(math.nan)
    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def thermocouple_interval_speeds(
    probes_path: str | Path,
    *,
    threshold_C: float = THERMOCOUPLE_THRESHOLD_C,
) -> list[dict[str, float | str]]:
    probes_path = Path(probes_path)
    column_by_z = {
        3.0: "T_z3p0mm_C",
        6.2: "T_z6p2mm_C",
        11.0: "T_z11p0mm_C",
    }
    cols = _read_numeric_csv_columns(probes_path, ("time_since_fill_s", *tuple(column_by_z.values())))
    valid_time = np.isfinite(cols["time_since_fill_s"]) & (cols["time_since_fill_s"] >= 0.0)
    crossing_by_z: dict[float, float] = {}
    for z_mm, column in column_by_z.items():
        crossing_by_z[z_mm] = _first_downward_crossing_time_s(
            cols["time_since_fill_s"][valid_time],
            cols[column][valid_time],
            threshold_C,
        )

    rows: list[dict[str, float | str]] = []
    for z0, z1 in ((3.0, 6.2), (6.2, 11.0), (3.0, 11.0)):
        t0 = crossing_by_z[z0]
        t1 = crossing_by_z[z1]
        if math.isfinite(t0) and math.isfinite(t1) and t1 > t0:
            speed = (z1 - z0) / (t1 - t0)
        else:
            speed = math.nan
        rows.append(
            {
                "interval": f"{z0:g}_to_{z1:g}_mm",
                "z_start_mm": float(z0),
                "z_end_mm": float(z1),
                "t_start_crossing_s": float(t0),
                "t_end_crossing_s": float(t1),
                "speed_mm_s": float(speed),
                "threshold_C": float(threshold_C),
            }
        )
    return rows


def write_rows_csv(path: str | Path, rows: list[dict[str, float | str]]) -> None:
    path = Path(path)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_velocity_control_objective(
    theta,
    config: OpenLoopProblemConfig,
    cryostage_params: CryostageModelParams,
    out_dir: str | Path,
    case_name: str,
    *,
    objective_config: ConstantVelocityObjectiveConfig,
    T_plate0_C: float | None = None,
    prepared_candidate: PreparedOpenLoopCandidate | None = None,
) -> OpenLoopObjectiveResult:
    from code_simulation.simulation.open_loop_cascade import run_open_loop_case

    if prepared_candidate is None:
        prepared_candidate = prepare_open_loop_candidate(theta, config)
    T_ref_profile_C = prepared_candidate.T_ref_profile_C

    cascade_result = run_open_loop_case(
        time_s=config.cryostage_time_grid_s(),
        T_ref_profile_C=T_ref_profile_C,
        cryostage_params=cryostage_params,
        out_dir=out_dir,
        prefix=case_name,
        T_plate0_C=T_plate0_C,
        **config.cascade_run_kwargs(),
    )

    front_trajectory = load_active_front_observable(cascade_result)
    tracking_summary, tracking_series = constant_velocity_tracking(
        front_trajectory,
        objective_config,
        incomplete_penalty_value=config.incomplete_penalty_value,
    )
    smoothness_penalty = temperature_smoothness_penalty(T_ref_profile_C, config)
    completion_penalty = float(tracking_summary.completion_penalty)
    objective_value = (
        config.tracking_weight * float(tracking_summary.tracking_mse)
        + config.completion_weight * completion_penalty
        + config.smoothness_weight * float(smoothness_penalty)
    )
    summary_path = _write_objective_summary_csv(
        Path(out_dir) / f"{case_name}_velocity_objective_summary.csv",
        objective_value=float(objective_value),
        tracking_summary=tracking_summary,
        smoothness_penalty=float(smoothness_penalty),
        completion_penalty=float(completion_penalty),
        num_objective_samples=int(tracking_summary.num_tracking_samples),
    )

    return OpenLoopObjectiveResult(
        objective_value=float(objective_value),
        tracking_mse=float(tracking_summary.tracking_mse),
        smoothness_penalty=float(smoothness_penalty),
        terminal_penalty=0.0,
        num_objective_samples=int(tracking_summary.num_tracking_samples),
        cascade_result=cascade_result,
        front_trajectory=front_trajectory,
        z_front_reference_m=tracking_series.z_ref_m,
        completion_penalty=float(completion_penalty),
        freeze_completion_time_s=front_trajectory.first_freeze_completion_time_s(),
        freeze_completion_reached=bool(math.isfinite(front_trajectory.first_freeze_completion_time_s())),
        safety_cap_s=float(config.safety_cap_s),
        objective_summary_path=summary_path,
    )


__all__ = [
    "ConstantVelocityObjectiveConfig",
    "VelocityTrackingSeries",
    "VelocityTrackingSummary",
    "constant_velocity_tracking",
    "evaluate_velocity_control_objective",
    "first_time_at_or_above",
    "temperature_smoothness_penalty",
    "thermocouple_interval_speeds",
    "write_rows_csv",
    "write_tracking_summary_csv",
]
