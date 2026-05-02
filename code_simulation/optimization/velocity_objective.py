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
    direct_speed_weight: float = 0.0
    direct_speed_tolerance_pct: float = 0.0
    segment_speed_weight: float = 0.0
    segment_speed_tolerance_pct: float = 0.0
    segment_speed_num_segments: int = 0

    def __post_init__(self) -> None:
        target_front_speed_mm_s = float(self.target_front_speed_mm_s)
        control_z_min_mm = float(self.control_z_min_mm)
        control_z_max_mm = float(self.control_z_max_mm)
        direct_speed_weight = float(self.direct_speed_weight)
        direct_speed_tolerance_pct = float(self.direct_speed_tolerance_pct)
        segment_speed_weight = float(self.segment_speed_weight)
        segment_speed_tolerance_pct = float(self.segment_speed_tolerance_pct)
        segment_speed_num_segments = int(self.segment_speed_num_segments)
        if not math.isfinite(target_front_speed_mm_s) or target_front_speed_mm_s <= 0.0:
            raise ValueError("target_front_speed_mm_s must be finite and positive")
        if not math.isfinite(control_z_min_mm) or not math.isfinite(control_z_max_mm):
            raise ValueError("control z limits must be finite")
        if control_z_min_mm < 0.0 or control_z_max_mm <= control_z_min_mm:
            raise ValueError("control z limits must satisfy 0 <= z_min < z_max")
        if not math.isfinite(direct_speed_weight) or direct_speed_weight < 0.0:
            raise ValueError("direct_speed_weight must be finite and non-negative")
        if not math.isfinite(direct_speed_tolerance_pct) or direct_speed_tolerance_pct < 0.0:
            raise ValueError("direct_speed_tolerance_pct must be finite and non-negative")
        if not math.isfinite(segment_speed_weight) or segment_speed_weight < 0.0:
            raise ValueError("segment_speed_weight must be finite and non-negative")
        if not math.isfinite(segment_speed_tolerance_pct) or segment_speed_tolerance_pct < 0.0:
            raise ValueError("segment_speed_tolerance_pct must be finite and non-negative")
        if segment_speed_num_segments < 0:
            raise ValueError("segment_speed_num_segments must be non-negative")
        if segment_speed_weight > 0.0 and segment_speed_num_segments < 2:
            raise ValueError("segment_speed_num_segments must be at least 2 when segment_speed_weight is positive")
        object.__setattr__(self, "target_front_speed_mm_s", target_front_speed_mm_s)
        object.__setattr__(self, "control_z_min_mm", control_z_min_mm)
        object.__setattr__(self, "control_z_max_mm", control_z_max_mm)
        object.__setattr__(self, "direct_speed_weight", direct_speed_weight)
        object.__setattr__(self, "direct_speed_tolerance_pct", direct_speed_tolerance_pct)
        object.__setattr__(self, "segment_speed_weight", segment_speed_weight)
        object.__setattr__(self, "segment_speed_tolerance_pct", segment_speed_tolerance_pct)
        object.__setattr__(self, "segment_speed_num_segments", segment_speed_num_segments)


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


@dataclass(frozen=True)
class SegmentSpeedSummary:
    target_front_speed_mm_s: float
    segment_speed_num_segments: int
    segment_speed_num_valid_segments: int
    segment_speed_penalty: float
    segment_speed_rmse_pct: float
    segment_speed_mean_abs_error_pct: float
    segment_speed_max_abs_error_pct: float
    segment_speed_min_mm_s: float
    segment_speed_max_mm_s: float
    segment_speed_spread_mm_s: float
    segment_speed_weight: float
    segment_speed_tolerance_pct: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "target_front_speed_mm_s": float(self.target_front_speed_mm_s),
            "segment_speed_num_segments": int(self.segment_speed_num_segments),
            "segment_speed_num_valid_segments": int(self.segment_speed_num_valid_segments),
            "segment_speed_penalty": float(self.segment_speed_penalty),
            "segment_speed_rmse_pct": float(self.segment_speed_rmse_pct),
            "segment_speed_mean_abs_error_pct": float(self.segment_speed_mean_abs_error_pct),
            "segment_speed_max_abs_error_pct": float(self.segment_speed_max_abs_error_pct),
            "segment_speed_min_mm_s": float(self.segment_speed_min_mm_s),
            "segment_speed_max_mm_s": float(self.segment_speed_max_mm_s),
            "segment_speed_spread_mm_s": float(self.segment_speed_spread_mm_s),
            "segment_speed_weight": float(self.segment_speed_weight),
            "segment_speed_tolerance_pct": float(self.segment_speed_tolerance_pct),
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


def direct_speed_error_penalty(
    tracking_summary: VelocityTrackingSummary,
    objective_config: ConstantVelocityObjectiveConfig,
    *,
    incomplete_penalty_value: float,
) -> tuple[float, float, float]:
    target_speed = float(objective_config.target_front_speed_mm_s)
    achieved_speed = float(tracking_summary.actual_interval_speed_mm_s)
    if not (math.isfinite(target_speed) and target_speed > 0.0 and math.isfinite(achieved_speed)):
        return float(incomplete_penalty_value), math.nan, math.nan
    signed_relative_error = float((achieved_speed - target_speed) / target_speed)
    abs_relative_error = abs(signed_relative_error)
    tolerance = float(objective_config.direct_speed_tolerance_pct) / 100.0
    excess_error = max(abs_relative_error - tolerance, 0.0)
    return float(excess_error * excess_error), float(signed_relative_error), float(abs_relative_error * 100.0)


def segment_speed_rows(
    front_trajectory: FrontTrajectory,
    objective_config: ConstantVelocityObjectiveConfig,
) -> list[dict[str, float | int]]:
    num_segments = int(objective_config.segment_speed_num_segments)
    if num_segments <= 0:
        return []

    control_time_s = np.asarray(front_trajectory.time_since_fill_s, dtype=np.float64)
    z_front_m = np.asarray(front_trajectory.z_front_m, dtype=np.float64)
    finite = np.isfinite(control_time_s) & np.isfinite(z_front_m) & (control_time_s >= 0.0)
    z_boundaries_mm = np.linspace(
        float(objective_config.control_z_min_mm),
        float(objective_config.control_z_max_mm),
        num_segments + 1,
        dtype=np.float64,
    )
    crossing_times_s = [
        first_time_at_or_above(control_time_s[finite], z_front_m[finite], float(z_mm) * 1.0e-3)
        for z_mm in z_boundaries_mm
    ]

    target_speed = float(objective_config.target_front_speed_mm_s)
    rows: list[dict[str, float | int]] = []
    for idx in range(num_segments):
        z0_mm = float(z_boundaries_mm[idx])
        z1_mm = float(z_boundaries_mm[idx + 1])
        t0_s = float(crossing_times_s[idx])
        t1_s = float(crossing_times_s[idx + 1])
        if math.isfinite(t0_s) and math.isfinite(t1_s) and t1_s > t0_s:
            speed_mm_s = float((z1_mm - z0_mm) / (t1_s - t0_s))
            signed_relative_error = float((speed_mm_s - target_speed) / target_speed)
            abs_relative_error_pct = float(abs(signed_relative_error) * 100.0)
        else:
            speed_mm_s = math.nan
            signed_relative_error = math.nan
            abs_relative_error_pct = math.nan
        rows.append(
            {
                "segment_index": int(idx),
                "z_start_mm": z0_mm,
                "z_end_mm": z1_mm,
                "t_start_crossing_s": t0_s,
                "t_end_crossing_s": t1_s,
                "speed_mm_s": float(speed_mm_s),
                "signed_relative_error": float(signed_relative_error),
                "abs_relative_error_pct": float(abs_relative_error_pct),
            }
        )
    return rows


def segment_speed_error_penalty(
    front_trajectory: FrontTrajectory,
    objective_config: ConstantVelocityObjectiveConfig,
    *,
    incomplete_penalty_value: float,
) -> SegmentSpeedSummary:
    rows = segment_speed_rows(front_trajectory, objective_config)
    num_segments = int(objective_config.segment_speed_num_segments)
    if num_segments <= 0:
        return SegmentSpeedSummary(
            target_front_speed_mm_s=float(objective_config.target_front_speed_mm_s),
            segment_speed_num_segments=0,
            segment_speed_num_valid_segments=0,
            segment_speed_penalty=0.0,
            segment_speed_rmse_pct=math.nan,
            segment_speed_mean_abs_error_pct=math.nan,
            segment_speed_max_abs_error_pct=math.nan,
            segment_speed_min_mm_s=math.nan,
            segment_speed_max_mm_s=math.nan,
            segment_speed_spread_mm_s=math.nan,
            segment_speed_weight=float(objective_config.segment_speed_weight),
            segment_speed_tolerance_pct=float(objective_config.segment_speed_tolerance_pct),
        )

    speeds = np.asarray([float(row["speed_mm_s"]) for row in rows], dtype=np.float64)
    valid = np.isfinite(speeds)
    target_speed = float(objective_config.target_front_speed_mm_s)
    tolerance = float(objective_config.segment_speed_tolerance_pct) / 100.0
    if np.count_nonzero(valid) != num_segments:
        penalty = float(incomplete_penalty_value)
    else:
        relative_errors = (speeds[valid] - target_speed) / target_speed
        excess_errors = np.maximum(np.abs(relative_errors) - tolerance, 0.0)
        penalty = float(np.mean(excess_errors * excess_errors))

    if np.any(valid):
        abs_error_pct = np.abs((speeds[valid] - target_speed) / target_speed) * 100.0
        rmse_pct = float(np.sqrt(np.mean(abs_error_pct * abs_error_pct)))
        mean_abs_pct = float(np.mean(abs_error_pct))
        max_abs_pct = float(np.max(abs_error_pct))
        min_speed = float(np.min(speeds[valid]))
        max_speed = float(np.max(speeds[valid]))
        spread = float(max_speed - min_speed)
    else:
        rmse_pct = math.nan
        mean_abs_pct = math.nan
        max_abs_pct = math.nan
        min_speed = math.nan
        max_speed = math.nan
        spread = math.nan

    return SegmentSpeedSummary(
        target_front_speed_mm_s=target_speed,
        segment_speed_num_segments=num_segments,
        segment_speed_num_valid_segments=int(np.count_nonzero(valid)),
        segment_speed_penalty=float(penalty),
        segment_speed_rmse_pct=float(rmse_pct),
        segment_speed_mean_abs_error_pct=float(mean_abs_pct),
        segment_speed_max_abs_error_pct=float(max_abs_pct),
        segment_speed_min_mm_s=float(min_speed),
        segment_speed_max_mm_s=float(max_speed),
        segment_speed_spread_mm_s=float(spread),
        segment_speed_weight=float(objective_config.segment_speed_weight),
        segment_speed_tolerance_pct=float(objective_config.segment_speed_tolerance_pct),
    )


def write_segment_speed_summary_csv(path: Path, summary: SegmentSpeedSummary) -> None:
    row = summary.to_dict()
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


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
    segment_speed_summary: SegmentSpeedSummary,
    smoothness_penalty: float,
    completion_penalty: float,
    direct_speed_penalty: float,
    direct_speed_relative_error: float,
    direct_speed_relative_error_pct: float,
    direct_speed_weight: float,
    direct_speed_tolerance_pct: float,
    num_objective_samples: int,
) -> Path:
    row = {
        "objective_value": float(objective_value),
        "tracking_mse": float(tracking_summary.tracking_mse),
        "smoothness_penalty": float(smoothness_penalty),
        "completion_penalty": float(completion_penalty),
        "direct_speed_penalty": float(direct_speed_penalty),
        "direct_speed_relative_error": float(direct_speed_relative_error),
        "direct_speed_relative_error_pct": float(direct_speed_relative_error_pct),
        "direct_speed_weight": float(direct_speed_weight),
        "direct_speed_tolerance_pct": float(direct_speed_tolerance_pct),
        **segment_speed_summary.to_dict(),
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
    direct_speed_penalty, direct_speed_relative_error, direct_speed_relative_error_pct = direct_speed_error_penalty(
        tracking_summary,
        objective_config,
        incomplete_penalty_value=config.incomplete_penalty_value,
    )
    segment_speed_summary = segment_speed_error_penalty(
        front_trajectory,
        objective_config,
        incomplete_penalty_value=config.incomplete_penalty_value,
    )
    objective_value = (
        config.tracking_weight * float(tracking_summary.tracking_mse)
        + config.completion_weight * completion_penalty
        + config.smoothness_weight * float(smoothness_penalty)
        + float(objective_config.direct_speed_weight) * float(direct_speed_penalty)
        + float(objective_config.segment_speed_weight) * float(segment_speed_summary.segment_speed_penalty)
    )
    summary_path = _write_objective_summary_csv(
        Path(out_dir) / f"{case_name}_velocity_objective_summary.csv",
        objective_value=float(objective_value),
        tracking_summary=tracking_summary,
        segment_speed_summary=segment_speed_summary,
        smoothness_penalty=float(smoothness_penalty),
        completion_penalty=float(completion_penalty),
        direct_speed_penalty=float(direct_speed_penalty),
        direct_speed_relative_error=float(direct_speed_relative_error),
        direct_speed_relative_error_pct=float(direct_speed_relative_error_pct),
        direct_speed_weight=float(objective_config.direct_speed_weight),
        direct_speed_tolerance_pct=float(objective_config.direct_speed_tolerance_pct),
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
    "SegmentSpeedSummary",
    "VelocityTrackingSeries",
    "VelocityTrackingSummary",
    "constant_velocity_tracking",
    "direct_speed_error_penalty",
    "evaluate_velocity_control_objective",
    "first_time_at_or_above",
    "segment_speed_error_penalty",
    "segment_speed_rows",
    "temperature_smoothness_penalty",
    "thermocouple_interval_speeds",
    "write_segment_speed_summary_csv",
    "write_rows_csv",
    "write_tracking_summary_csv",
]
