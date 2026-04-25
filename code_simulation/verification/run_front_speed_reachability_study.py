#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.config_files import DEFAULT_SIMULATION_CONFIG_PATH, load_simulation_profiles
from code_simulation.core.paths import results_dir
from code_simulation.core.plotting import configure_matplotlib
from code_simulation.core.trajectory_profiles import PiecewiseLinearTemperatureProfile
from code_simulation.optimization.open_loop_workflow_config import (
    LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT,
    apply_simulation_profile,
    build_problem_config,
)
from code_simulation.optimization.reachability_constraints import (
    ReachabilityConstraints,
    check_piecewise_linear_trajectory_admissibility,
    load_reachability_constraints,
)
from code_simulation.simulation.ambient import describe_ambient_temperature_model
from code_simulation.simulation.cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
from code_simulation.simulation.open_loop_cascade import build_plate_temperature_response, run_open_loop_case
from code_simulation.verification.front_summary import (
    first_freeze_complete_time,
    first_time_at_or_above,
    load_front_columns,
    nan_to_str,
)


DEFAULT_RUN_NAME = "front_speed_reachability"
DEFAULT_OUT_ROOT_DIR = results_dir("front_speed_reachability")
DEFAULT_TARGETS_C = (0.0, -5.0, -10.0, -15.0, -20.0, -21.0)
DEFAULT_TARGET_RANGE_C = (0.0, -21.0)
DEFAULT_WINDOW_S = 60.0
DEFAULT_SENSITIVITY_WINDOWS_S = (30.0, 120.0)
DEFAULT_MAX_AFTER_FILL_S = 3600.0
DEFAULT_CONFIRM_EXTREMES_COUNT = 6
DEFAULT_MIN_MEASURABLE_FRONT_MM = 1.0
DEFAULT_TC_PROBE_Z_M = (3.0e-3, 6.2e-3, 11.0e-3)
DEFAULT_TC_PROBE_WALL_INSET_M = 1.0e-3
DEFAULT_TC_THRESHOLD_C = 0.0
POSITIVE_SPEED_EPS_MM_S = 1.0e-9


@dataclass(frozen=True)
class FrontSpeedCandidate:
    name: str
    family: str
    target_C: float
    support_status: str
    support_note: str
    knot_times_s: tuple[float, ...]
    knot_temperatures_C: tuple[float, ...]
    admissible: bool
    admissibility_reasons: tuple[str, ...]
    description: str

    def profile(self) -> PiecewiseLinearTemperatureProfile:
        return PiecewiseLinearTemperatureProfile(
            knot_times_s=self.knot_times_s,
            knot_temperatures_C=self.knot_temperatures_C,
        )


@dataclass(frozen=True)
class SpeedSummary:
    stage: str
    candidate_name: str
    family: str
    target_C: float
    support_status: str
    admissible: bool
    front_csv_path: Path
    max_z_front_mm: float
    centerline_top_arrival_s: float
    wall_top_arrival_s: float
    full_freezing_s: float
    max_sustained_front_speed_mm_s: float
    min_positive_sustained_front_speed_mm_s: float
    mean_speed_10_90_mm_s: float
    full_depth_average_speed_mm_s: float
    max_instant_front_speed_mm_s: float
    min_positive_instant_front_speed_mm_s: float
    no_front_or_stalled: bool


@dataclass(frozen=True)
class ThermocoupleSpeedSummary:
    stage: str
    candidate_name: str
    family: str
    target_C: float
    support_status: str
    admissible: bool
    probes_csv_path: Path
    threshold_C: float
    crossing_3p0mm_s: float
    crossing_6p2mm_s: float
    crossing_11p0mm_s: float
    v_tc_3p0_to_6p2_mm_s: float
    v_tc_6p2_to_11p0_mm_s: float
    v_tc_3p0_to_11p0_mm_s: float
    thermocouple_speed_complete: bool


def _configure_matplotlib() -> None:
    configure_matplotlib(plt)


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all values must be finite")
    return values


def _parse_target_range(raw: str) -> tuple[float, float]:
    values = _parse_float_list(raw)
    if len(values) != 2:
        raise ValueError("--target-range-c must contain exactly two comma-separated values")
    warm_C, cold_C = values
    if warm_C < cold_C:
        raise ValueError("--target-range-c must be written as warm,cold, for example 0,-21")
    return float(warm_C), float(cold_C)


def _parse_probe_z_m(raw: str) -> tuple[float, float, float]:
    values_mm = _parse_float_list(raw)
    if len(values_mm) != 3:
        raise ValueError("--probe-z-mm must contain exactly three comma-separated heights")
    values_m = tuple(1.0e-3 * float(value) for value in values_mm)
    if any(value <= 0.0 for value in values_m):
        raise ValueError("--probe-z-mm values must be positive")
    if any(b <= a for a, b in zip(values_m, values_m[1:])):
        raise ValueError("--probe-z-mm values must be strictly increasing")
    return (float(values_m[0]), float(values_m[1]), float(values_m[2]))


def _temperature_tag(value_C: float) -> str:
    prefix = "m" if value_C < 0.0 else "p"
    return prefix + f"{abs(float(value_C)):.3g}".replace(".", "p")


def _format_probe_label_mm(z_m: float) -> str:
    return f"{1000.0 * float(z_m):.1f}".replace(".", "p") + "mm"


def _probe_column_name(z_m: float) -> str:
    return f"T_z{_format_probe_label_mm(z_m)}_C"


def _build_time_grid(horizon_s: float, dt_s: float) -> np.ndarray:
    time_s = np.arange(0.0, float(horizon_s), float(dt_s), dtype=np.float64)
    if time_s.size == 0 or time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
    if time_s[-1] < float(horizon_s) - 1.0e-12:
        time_s = np.append(time_s, float(horizon_s))
    return time_s


def _support_status(target_C: float, constraints: ReachabilityConstraints) -> tuple[str, str]:
    target_C = float(target_C)
    characterized = np.asarray(constraints.characterized_targets_C, dtype=np.float64)
    if np.any(np.isclose(characterized, target_C, atol=1.0e-12)):
        return "direct", "Target has direct cryostage characterization support."
    min_direct = float(np.min(characterized))
    max_direct = float(np.max(characterized))
    if min_direct <= target_C <= max_direct:
        return "interpolated", "Target is inside the characterized range but not directly measured."
    if target_C < min_direct:
        return "extrapolated_cold_limit", "Target is colder than the coldest directly characterized target."
    return "operational_limit", "Target is warmer than the warmest cooling target; cooling demand may be zero or small."


def _duration_for_cooling_drop_s(drop_C: float, constraints: ReachabilityConstraints) -> float:
    drop_C = max(float(drop_C), 0.0)
    if drop_C <= 1.0e-12:
        return 0.0
    windows = np.asarray(constraints.overall_window_s, dtype=np.float64)
    drops = np.asarray(constraints.overall_window_drop_C, dtype=np.float64)
    if windows.size == 0 or drops.size == 0:
        return 600.0
    order = np.argsort(windows)
    windows = windows[order]
    drops = drops[order]
    if drop_C <= float(np.max(drops)) + 1.0e-12:
        return float(np.interp(drop_C, drops, windows))
    max_window = float(windows[-1])
    max_drop = max(float(drops[-1]), 1.0e-12)
    return float(math.ceil(drop_C / max_drop) * max_window)


def _candidate_from_knots(
    *,
    family: str,
    target_C: float,
    knot_times_s: tuple[float, ...],
    knot_temperatures_C: tuple[float, ...],
    constraints: ReachabilityConstraints,
    description: str,
) -> FrontSpeedCandidate:
    status, note = _support_status(target_C, constraints)
    name = f"{family}_{_temperature_tag(target_C)}"
    try:
        report = check_piecewise_linear_trajectory_admissibility(
            knot_times_s,
            knot_temperatures_C,
            constraints=constraints,
            require_monotone_nonincreasing=True,
        )
        admissible = bool(report.is_admissible)
        reasons = tuple(str(reason) for reason in report.reasons)
    except Exception as exc:
        admissible = False
        reasons = (str(exc),)
    return FrontSpeedCandidate(
        name=name,
        family=family,
        target_C=float(target_C),
        support_status=status,
        support_note=note,
        knot_times_s=tuple(float(value) for value in knot_times_s),
        knot_temperatures_C=tuple(float(value) for value in knot_temperatures_C),
        admissible=admissible,
        admissibility_reasons=reasons,
        description=description,
    )


def _ramp_knots(
    *,
    target_C: float,
    duration_s: float,
    max_after_fill_s: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    duration_s = min(float(duration_s), float(max_after_fill_s))
    if duration_s >= float(max_after_fill_s) - 1.0e-12:
        return (0.0, float(max_after_fill_s)), (0.0, float(target_C))
    return (0.0, float(duration_s), float(max_after_fill_s)), (0.0, float(target_C), float(target_C))


def build_candidates(
    *,
    targets_C: tuple[float, ...],
    target_range_C: tuple[float, float],
    max_after_fill_s: float,
    constraints: ReachabilityConstraints,
) -> tuple[FrontSpeedCandidate, ...]:
    warm_limit_C, cold_limit_C = target_range_C
    candidates: list[FrontSpeedCandidate] = []
    for target_C in targets_C:
        target_C = float(target_C)
        if target_C > warm_limit_C + 1.0e-12 or target_C < cold_limit_C - 1.0e-12:
            raise ValueError(
                f"target {target_C:.6g} C is outside requested operational range "
                f"[{cold_limit_C:.6g}, {warm_limit_C:.6g}] C"
            )

        candidates.append(
            _candidate_from_knots(
                family="hold",
                target_C=target_C,
                knot_times_s=(0.0, float(max_after_fill_s)),
                knot_temperatures_C=(target_C, target_C),
                constraints=constraints,
                description="Constant T_ref hold at the requested target.",
            )
        )
        if target_C < 0.0:
            fastest_duration_s = min(
                float(max_after_fill_s),
                max(10.0, _duration_for_cooling_drop_s(abs(target_C), constraints)),
            )
            for family, duration_s, description in (
                (
                    "ramp_fastest_envelope",
                    fastest_duration_s,
                    "Monotone ramp using the conservative finite-window cooling envelope duration.",
                ),
                (
                    "ramp_medium",
                    min(float(max_after_fill_s), max(900.0, fastest_duration_s)),
                    "Moderate monotone ramp to the requested target.",
                ),
                (
                    "ramp_slow",
                    min(float(max_after_fill_s), max(1800.0, fastest_duration_s)),
                    "Slow monotone ramp to the requested target.",
                ),
            ):
                knot_times_s, knot_temperatures_C = _ramp_knots(
                    target_C=target_C,
                    duration_s=duration_s,
                    max_after_fill_s=float(max_after_fill_s),
                )
                candidates.append(
                    _candidate_from_knots(
                        family=family,
                        target_C=target_C,
                        knot_times_s=knot_times_s,
                        knot_temperatures_C=knot_temperatures_C,
                        constraints=constraints,
                        description=description,
                    )
                )
    return tuple(candidates)


def _load_front_csv_with_instant_speed(front_csv_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cols = load_front_columns(front_csv_path)
    v_front_mm_s: list[float] = []
    with front_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                value = float(row.get("v_front_mm_per_s", "nan"))
            except (TypeError, ValueError):
                value = math.nan
            v_front_mm_s.append(value)
    return cols, np.asarray(v_front_mm_s, dtype=np.float64)


def robust_speed_windows(
    *,
    time_s: np.ndarray,
    z_front_m: np.ndarray,
    window_s: float,
    stride_s: float | None = None,
    min_samples: int = 3,
) -> list[dict[str, float]]:
    time_s = np.asarray(time_s, dtype=np.float64)
    z_front_mm = 1000.0 * np.asarray(z_front_m, dtype=np.float64)
    mask = np.isfinite(time_s) & np.isfinite(z_front_mm)
    time_s = time_s[mask]
    z_front_mm = z_front_mm[mask]
    if time_s.size < min_samples:
        return []

    window_s = float(window_s)
    stride_s = 0.5 * window_s if stride_s is None else float(stride_s)
    if window_s <= 0.0 or stride_s <= 0.0:
        raise ValueError("window_s and stride_s must be positive")

    out: list[dict[str, float]] = []
    start = float(time_s[0])
    final_start = float(time_s[-1] - window_s)
    while start <= final_start + 1.0e-12:
        end = start + window_s
        wmask = (time_s >= start - 1.0e-12) & (time_s <= end + 1.0e-12)
        if int(np.count_nonzero(wmask)) >= int(min_samples):
            t = time_s[wmask]
            z = z_front_mm[wmask]
            if float(t[-1] - t[0]) >= 0.5 * window_s:
                slope, intercept = np.polyfit(t, z, deg=1)
                predicted = slope * t + intercept
                residual = z - predicted
                ss_res = float(np.sum(residual * residual))
                ss_tot = float(np.sum((z - float(np.mean(z))) ** 2))
                r2 = math.nan if ss_tot <= 1.0e-18 else 1.0 - ss_res / ss_tot
                out.append(
                    {
                        "window_start_s": float(start),
                        "window_end_s": float(end),
                        "window_center_s": float(0.5 * (start + end)),
                        "window_s": float(window_s),
                        "speed_mm_s": float(slope),
                        "r2": float(r2),
                        "n_samples": float(t.size),
                    }
                )
        start += stride_s
    return out


def _first_time_at_fraction(time_s: np.ndarray, z_front_m: np.ndarray, threshold_m: float) -> float:
    return first_time_at_or_above(time_s, z_front_m, float(threshold_m))


def summarize_front_speed(
    *,
    stage: str,
    candidate: FrontSpeedCandidate,
    front_csv_path: Path,
    H_fill_m: float,
    window_s: float,
    sensitivity_windows_s: tuple[float, ...],
    min_measurable_front_mm: float = DEFAULT_MIN_MEASURABLE_FRONT_MM,
) -> tuple[SpeedSummary, list[dict[str, float]]]:
    cols, v_front_mm_s = _load_front_csv_with_instant_speed(front_csv_path)
    time_since_fill_s = cols["time_since_fill_s"]
    z_front_m = cols["z_front_m"]
    valid_front = np.isfinite(time_since_fill_s) & np.isfinite(z_front_m)
    finite_z_mm = 1000.0 * z_front_m[valid_front]
    max_z_front_mm = math.nan if finite_z_mm.size == 0 else float(np.max(finite_z_mm))

    all_windows: list[dict[str, float]] = []
    for current_window_s in (float(window_s), *tuple(float(value) for value in sensitivity_windows_s)):
        for row in robust_speed_windows(
            time_s=time_since_fill_s,
            z_front_m=z_front_m,
            window_s=current_window_s,
        ):
            row.update(
                {
                    "stage": stage,
                    "candidate_name": candidate.name,
                    "family": candidate.family,
                    "target_C": float(candidate.target_C),
                    "is_primary_window": bool(abs(current_window_s - float(window_s)) <= 1.0e-12),
                }
            )
            all_windows.append(row)

    primary_speeds = np.asarray(
        [row["speed_mm_s"] for row in all_windows if bool(row["is_primary_window"])],
        dtype=np.float64,
    )
    primary_speeds = primary_speeds[np.isfinite(primary_speeds)]
    positive_primary = primary_speeds[primary_speeds > POSITIVE_SPEED_EPS_MM_S]
    max_sustained = math.nan if primary_speeds.size == 0 else float(np.max(primary_speeds))
    min_positive = math.nan if positive_primary.size == 0 else float(np.min(positive_primary))

    t10 = _first_time_at_fraction(time_since_fill_s, z_front_m, 0.1 * float(H_fill_m))
    t90 = _first_time_at_fraction(time_since_fill_s, z_front_m, 0.9 * float(H_fill_m))
    if math.isfinite(t10) and math.isfinite(t90) and t90 > t10:
        mean_10_90 = float(0.8 * float(H_fill_m) * 1000.0 / (t90 - t10))
    else:
        mean_10_90 = math.nan

    full_freezing_s = first_freeze_complete_time(time_since_fill_s, cols["freeze_complete_flag"])
    full_depth_avg = (
        float(float(H_fill_m) * 1000.0 / full_freezing_s)
        if math.isfinite(full_freezing_s) and full_freezing_s > 0.0
        else math.nan
    )
    finite_instant = v_front_mm_s[np.isfinite(v_front_mm_s)]
    positive_instant = finite_instant[finite_instant > POSITIVE_SPEED_EPS_MM_S]
    max_instant = math.nan if finite_instant.size == 0 else float(np.max(finite_instant))
    min_positive_instant = math.nan if positive_instant.size == 0 else float(np.min(positive_instant))
    no_front_or_stalled = (
        not math.isfinite(max_z_front_mm)
        or max_z_front_mm < float(min_measurable_front_mm)
        or not (math.isfinite(max_sustained) and max_sustained > POSITIVE_SPEED_EPS_MM_S)
    )

    summary = SpeedSummary(
        stage=stage,
        candidate_name=candidate.name,
        family=candidate.family,
        target_C=float(candidate.target_C),
        support_status=candidate.support_status,
        admissible=bool(candidate.admissible),
        front_csv_path=front_csv_path,
        max_z_front_mm=max_z_front_mm,
        centerline_top_arrival_s=first_time_at_or_above(time_since_fill_s, z_front_m, float(H_fill_m) - 1.0e-12),
        wall_top_arrival_s=first_time_at_or_above(time_since_fill_s, cols["z_front_wall_m"], float(H_fill_m) - 1.0e-12),
        full_freezing_s=full_freezing_s,
        max_sustained_front_speed_mm_s=max_sustained,
        min_positive_sustained_front_speed_mm_s=min_positive,
        mean_speed_10_90_mm_s=mean_10_90,
        full_depth_average_speed_mm_s=full_depth_avg,
        max_instant_front_speed_mm_s=max_instant,
        min_positive_instant_front_speed_mm_s=min_positive_instant,
        no_front_or_stalled=no_front_or_stalled,
    )
    return summary, all_windows


def _first_downward_crossing_time_s(time_s: np.ndarray, temperature_C: np.ndarray, threshold_C: float) -> float:
    time_s = np.asarray(time_s, dtype=np.float64)
    temperature_C = np.asarray(temperature_C, dtype=np.float64)
    valid = np.isfinite(time_s) & np.isfinite(temperature_C)
    t = time_s[valid]
    y = temperature_C[valid]
    if t.size == 0:
        return math.nan

    threshold_C = float(threshold_C)
    if y[0] <= threshold_C:
        return float(t[0])
    for i in range(t.size - 1):
        t0 = float(t[i])
        t1 = float(t[i + 1])
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if y0 == threshold_C:
            return t0
        if y0 > threshold_C and y1 <= threshold_C:
            if y1 == y0:
                return t1
            frac = (threshold_C - y0) / (y1 - y0)
            return float(t0 + frac * (t1 - t0))
    if y[-1] == threshold_C:
        return float(t[-1])
    return math.nan


def _load_probe_columns(
    probes_csv_path: Path,
    *,
    probe_z_m: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    columns = {
        "time_s": [],
        "time_since_fill_s": [],
    }
    probe_columns = [_probe_column_name(value) for value in probe_z_m]
    for column in probe_columns:
        columns[column] = []

    with probes_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{probes_csv_path} is missing a header row")
        missing = [column for column in ("time_s", "time_since_fill_s", *probe_columns) if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{probes_csv_path} is missing probe columns: {missing}")
        for row in reader:
            columns["time_s"].append(float(row["time_s"]))
            columns["time_since_fill_s"].append(float(row["time_since_fill_s"]))
            for column in probe_columns:
                columns[column].append(float(row[column]))

    return {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}


def _segment_speed_mm_s(z0_m: float, z1_m: float, t0_s: float, t1_s: float) -> float:
    if not (math.isfinite(t0_s) and math.isfinite(t1_s)) or t1_s <= t0_s:
        return math.nan
    return float((float(z1_m) - float(z0_m)) * 1000.0 / (t1_s - t0_s))


def summarize_thermocouple_speed(
    *,
    stage: str,
    candidate: FrontSpeedCandidate,
    probes_csv_path: Path,
    probe_z_m: tuple[float, float, float],
    threshold_C: float,
) -> tuple[ThermocoupleSpeedSummary, list[dict[str, object]]]:
    cols = _load_probe_columns(probes_csv_path, probe_z_m=probe_z_m)
    time_since_fill_s = cols["time_since_fill_s"]
    post_fill = np.isfinite(time_since_fill_s) & (time_since_fill_s >= 0.0)
    probe_columns = [_probe_column_name(value) for value in probe_z_m]

    crossing_times = []
    crossing_rows: list[dict[str, object]] = []
    for z_m, column in zip(probe_z_m, probe_columns, strict=True):
        crossing_s = _first_downward_crossing_time_s(
            time_since_fill_s[post_fill],
            cols[column][post_fill],
            float(threshold_C),
        )
        crossing_times.append(crossing_s)
        crossing_rows.append(
            {
                "stage": stage,
                "candidate_name": candidate.name,
                "family": candidate.family,
                "target_C": f"{candidate.target_C:.6f}",
                "support_status": candidate.support_status,
                "admissible": str(int(candidate.admissible)),
                "probe_label": _format_probe_label_mm(z_m),
                "probe_z_mm": f"{1000.0 * float(z_m):.6f}",
                "threshold_C": f"{float(threshold_C):.6f}",
                "crossing_time_since_fill_s": nan_to_str(crossing_s),
                "crossed": str(int(math.isfinite(crossing_s))),
                "probes_csv_path": str(probes_csv_path),
            }
        )

    t0_s, t1_s, t2_s = crossing_times
    v01 = _segment_speed_mm_s(probe_z_m[0], probe_z_m[1], t0_s, t1_s)
    v12 = _segment_speed_mm_s(probe_z_m[1], probe_z_m[2], t1_s, t2_s)
    v02 = _segment_speed_mm_s(probe_z_m[0], probe_z_m[2], t0_s, t2_s)
    complete = bool(math.isfinite(v01) and math.isfinite(v12) and math.isfinite(v02))
    summary = ThermocoupleSpeedSummary(
        stage=stage,
        candidate_name=candidate.name,
        family=candidate.family,
        target_C=float(candidate.target_C),
        support_status=candidate.support_status,
        admissible=bool(candidate.admissible),
        probes_csv_path=probes_csv_path,
        threshold_C=float(threshold_C),
        crossing_3p0mm_s=t0_s,
        crossing_6p2mm_s=t1_s,
        crossing_11p0mm_s=t2_s,
        v_tc_3p0_to_6p2_mm_s=v01,
        v_tc_6p2_to_11p0_mm_s=v12,
        v_tc_3p0_to_11p0_mm_s=v02,
        thermocouple_speed_complete=complete,
    )
    return summary, crossing_rows


def _summary_to_row(summary: SpeedSummary) -> dict[str, str]:
    return {
        "stage": summary.stage,
        "candidate_name": summary.candidate_name,
        "family": summary.family,
        "target_C": f"{summary.target_C:.6f}",
        "support_status": summary.support_status,
        "admissible": str(int(summary.admissible)),
        "front_csv_path": str(summary.front_csv_path),
        "max_z_front_mm": nan_to_str(summary.max_z_front_mm),
        "centerline_top_arrival_s": nan_to_str(summary.centerline_top_arrival_s),
        "wall_top_arrival_s": nan_to_str(summary.wall_top_arrival_s),
        "full_freezing_s": nan_to_str(summary.full_freezing_s),
        "max_sustained_front_speed_mm_s": nan_to_str(summary.max_sustained_front_speed_mm_s),
        "min_positive_sustained_front_speed_mm_s": nan_to_str(summary.min_positive_sustained_front_speed_mm_s),
        "mean_speed_10_90_mm_s": nan_to_str(summary.mean_speed_10_90_mm_s),
        "full_depth_average_speed_mm_s": nan_to_str(summary.full_depth_average_speed_mm_s),
        "max_instant_front_speed_mm_s": nan_to_str(summary.max_instant_front_speed_mm_s),
        "min_positive_instant_front_speed_mm_s": nan_to_str(summary.min_positive_instant_front_speed_mm_s),
        "no_front_or_stalled": str(int(summary.no_front_or_stalled)),
    }


def _tc_summary_to_row(summary: ThermocoupleSpeedSummary) -> dict[str, str]:
    return {
        "stage": summary.stage,
        "candidate_name": summary.candidate_name,
        "family": summary.family,
        "target_C": f"{summary.target_C:.6f}",
        "support_status": summary.support_status,
        "admissible": str(int(summary.admissible)),
        "probes_csv_path": str(summary.probes_csv_path),
        "threshold_C": f"{summary.threshold_C:.6f}",
        "crossing_3p0mm_s": nan_to_str(summary.crossing_3p0mm_s),
        "crossing_6p2mm_s": nan_to_str(summary.crossing_6p2mm_s),
        "crossing_11p0mm_s": nan_to_str(summary.crossing_11p0mm_s),
        "v_tc_3p0_to_6p2_mm_s": nan_to_str(summary.v_tc_3p0_to_6p2_mm_s),
        "v_tc_6p2_to_11p0_mm_s": nan_to_str(summary.v_tc_6p2_to_11p0_mm_s),
        "v_tc_3p0_to_11p0_mm_s": nan_to_str(summary.v_tc_3p0_to_11p0_mm_s),
        "thermocouple_speed_complete": str(int(summary.thermocouple_speed_complete)),
    }


def _combined_summary_rows(
    front_summaries: list[SpeedSummary],
    tc_summaries: list[ThermocoupleSpeedSummary],
) -> list[dict[str, str]]:
    tc_by_key = {(item.stage, item.candidate_name): item for item in tc_summaries}
    rows: list[dict[str, str]] = []
    for front in front_summaries:
        tc = tc_by_key.get((front.stage, front.candidate_name))
        rows.append(
            {
                "stage": front.stage,
                "candidate_name": front.candidate_name,
                "family": front.family,
                "target_C": f"{front.target_C:.6f}",
                "support_status": front.support_status,
                "admissible": str(int(front.admissible)),
                "max_sustained_front_speed_mm_s": nan_to_str(front.max_sustained_front_speed_mm_s),
                "mean_speed_10_90_mm_s": nan_to_str(front.mean_speed_10_90_mm_s),
                "full_depth_average_speed_mm_s": nan_to_str(front.full_depth_average_speed_mm_s),
                "full_freezing_s": nan_to_str(front.full_freezing_s),
                "no_front_or_stalled": str(int(front.no_front_or_stalled)),
                "tc_threshold_C": "" if tc is None else f"{tc.threshold_C:.6f}",
                "tc_crossing_3p0mm_s": "" if tc is None else nan_to_str(tc.crossing_3p0mm_s),
                "tc_crossing_6p2mm_s": "" if tc is None else nan_to_str(tc.crossing_6p2mm_s),
                "tc_crossing_11p0mm_s": "" if tc is None else nan_to_str(tc.crossing_11p0mm_s),
                "v_tc_3p0_to_6p2_mm_s": "" if tc is None else nan_to_str(tc.v_tc_3p0_to_6p2_mm_s),
                "v_tc_6p2_to_11p0_mm_s": "" if tc is None else nan_to_str(tc.v_tc_6p2_to_11p0_mm_s),
                "v_tc_3p0_to_11p0_mm_s": "" if tc is None else nan_to_str(tc.v_tc_3p0_to_11p0_mm_s),
                "thermocouple_speed_complete": "" if tc is None else str(int(tc.thermocouple_speed_complete)),
                "front_csv_path": str(front.front_csv_path),
                "probes_csv_path": "" if tc is None else str(tc.probes_csv_path),
            }
        )
    return rows


def _parse_csv_float(raw: str | None) -> float:
    if raw is None or str(raw).strip() == "":
        return math.nan
    return float(raw)


def _parse_csv_bool(raw: str | None) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _read_candidate_profiles(path: Path) -> dict[str, FrontSpeedCandidate]:
    candidates: dict[str, FrontSpeedCandidate] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = str(row["candidate_name"])
            reasons_raw = str(row.get("admissibility_reasons", "")).strip()
            candidates[name] = FrontSpeedCandidate(
                name=name,
                family=str(row["family"]),
                target_C=float(row["target_C"]),
                support_status=str(row["support_status"]),
                support_note=str(row.get("support_note", "")),
                knot_times_s=tuple(float(value) for value in json.loads(row["knot_times_s"])),
                knot_temperatures_C=tuple(float(value) for value in json.loads(row["knot_temperatures_C"])),
                admissible=_parse_csv_bool(row.get("admissible")),
                admissibility_reasons=tuple(reason for reason in reasons_raw.split("; ") if reason),
                description=str(row.get("description", "")),
            )
    return candidates


def _read_speed_summary_csv(path: Path, *, min_measurable_front_mm: float) -> list[SpeedSummary]:
    summaries: list[SpeedSummary] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            max_z_front_mm = _parse_csv_float(row.get("max_z_front_mm"))
            max_sustained = _parse_csv_float(row.get("max_sustained_front_speed_mm_s"))
            no_front_or_stalled = (
                not math.isfinite(max_z_front_mm)
                or max_z_front_mm < float(min_measurable_front_mm)
                or not (math.isfinite(max_sustained) and max_sustained > POSITIVE_SPEED_EPS_MM_S)
            )
            summaries.append(
                SpeedSummary(
                    stage=str(row["stage"]),
                    candidate_name=str(row["candidate_name"]),
                    family=str(row["family"]),
                    target_C=float(row["target_C"]),
                    support_status=str(row["support_status"]),
                    admissible=_parse_csv_bool(row.get("admissible")),
                    front_csv_path=Path(row["front_csv_path"]),
                    max_z_front_mm=max_z_front_mm,
                    centerline_top_arrival_s=_parse_csv_float(row.get("centerline_top_arrival_s")),
                    wall_top_arrival_s=_parse_csv_float(row.get("wall_top_arrival_s")),
                    full_freezing_s=_parse_csv_float(row.get("full_freezing_s")),
                    max_sustained_front_speed_mm_s=max_sustained,
                    min_positive_sustained_front_speed_mm_s=_parse_csv_float(
                        row.get("min_positive_sustained_front_speed_mm_s")
                    ),
                    mean_speed_10_90_mm_s=_parse_csv_float(row.get("mean_speed_10_90_mm_s")),
                    full_depth_average_speed_mm_s=_parse_csv_float(row.get("full_depth_average_speed_mm_s")),
                    max_instant_front_speed_mm_s=_parse_csv_float(row.get("max_instant_front_speed_mm_s")),
                    min_positive_instant_front_speed_mm_s=_parse_csv_float(
                        row.get("min_positive_instant_front_speed_mm_s")
                    ),
                    no_front_or_stalled=no_front_or_stalled,
                )
            )
    return summaries


def _read_tc_summary_csv(path: Path) -> list[ThermocoupleSpeedSummary]:
    if not path.exists():
        return []
    summaries: list[ThermocoupleSpeedSummary] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            summaries.append(
                ThermocoupleSpeedSummary(
                    stage=str(row["stage"]),
                    candidate_name=str(row["candidate_name"]),
                    family=str(row["family"]),
                    target_C=float(row["target_C"]),
                    support_status=str(row["support_status"]),
                    admissible=_parse_csv_bool(row.get("admissible")),
                    probes_csv_path=Path(row["probes_csv_path"]),
                    threshold_C=_parse_csv_float(row.get("threshold_C")),
                    crossing_3p0mm_s=_parse_csv_float(row.get("crossing_3p0mm_s")),
                    crossing_6p2mm_s=_parse_csv_float(row.get("crossing_6p2mm_s")),
                    crossing_11p0mm_s=_parse_csv_float(row.get("crossing_11p0mm_s")),
                    v_tc_3p0_to_6p2_mm_s=_parse_csv_float(row.get("v_tc_3p0_to_6p2_mm_s")),
                    v_tc_6p2_to_11p0_mm_s=_parse_csv_float(row.get("v_tc_6p2_to_11p0_mm_s")),
                    v_tc_3p0_to_11p0_mm_s=_parse_csv_float(row.get("v_tc_3p0_to_11p0_mm_s")),
                    thermocouple_speed_complete=_parse_csv_bool(row.get("thermocouple_speed_complete")),
                )
            )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_candidate_profiles(path: Path, candidates: tuple[FrontSpeedCandidate, ...]) -> None:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "target_C": f"{candidate.target_C:.6f}",
                "support_status": candidate.support_status,
                "support_note": candidate.support_note,
                "admissible": str(int(candidate.admissible)),
                "admissibility_reasons": "; ".join(candidate.admissibility_reasons),
                "knot_times_s": json.dumps([float(value) for value in candidate.knot_times_s]),
                "knot_temperatures_C": json.dumps([float(value) for value in candidate.knot_temperatures_C]),
                "description": candidate.description,
            }
        )
    _write_csv(
        path,
        rows,
        [
            "candidate_name",
            "family",
            "target_C",
            "support_status",
            "support_note",
            "admissible",
            "admissibility_reasons",
            "knot_times_s",
            "knot_temperatures_C",
            "description",
        ],
    )


def _write_plate_profile_csv(path: Path, *, time_s: np.ndarray, T_ref_C: np.ndarray, T_plate_C: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "T_ref_C", "T_plate_C"])
        for row in zip(time_s, T_ref_C, T_plate_C, strict=True):
            writer.writerow([f"{float(row[0]):.9f}", f"{float(row[1]):.9f}", f"{float(row[2]):.9f}"])


def _select_fine_candidates(
    summaries: list[SpeedSummary],
    candidates_by_name: dict[str, FrontSpeedCandidate],
    *,
    confirm_extremes_count: int,
) -> tuple[FrontSpeedCandidate, ...]:
    if confirm_extremes_count <= 0:
        return ()
    usable = [
        summary
        for summary in summaries
        if not summary.no_front_or_stalled
        and math.isfinite(summary.max_sustained_front_speed_mm_s)
        and math.isfinite(summary.min_positive_sustained_front_speed_mm_s)
    ]
    if not usable:
        return ()
    half = max(1, int(math.ceil(confirm_extremes_count / 2)))
    fastest = sorted(usable, key=lambda item: item.max_sustained_front_speed_mm_s, reverse=True)[:half]
    slowest = sorted(usable, key=lambda item: item.min_positive_sustained_front_speed_mm_s)[:half]
    selected: list[FrontSpeedCandidate] = []
    seen: set[str] = set()
    for summary in [*fastest, *slowest]:
        if summary.candidate_name in seen:
            continue
        selected.append(candidates_by_name[summary.candidate_name])
        seen.add(summary.candidate_name)
        if len(selected) >= confirm_extremes_count:
            break
    return tuple(selected)


def _stage_config(
    *,
    profile_name: str,
    simulation_config_path: str | Path,
    max_after_fill_s: float,
    probe_z_m: tuple[float, float, float] = DEFAULT_TC_PROBE_Z_M,
    probe_wall_inset_m: float = DEFAULT_TC_PROBE_WALL_INSET_M,
):
    profiles = load_simulation_profiles(simulation_config_path)
    profile = profiles.get_profile(profile_name)
    config = build_problem_config(num_knots=LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT)
    config = apply_simulation_profile(config, profile)
    solver_kwargs = config.cascade_run_kwargs()
    solver_kwargs["t_after_fill_s"] = float(max_after_fill_s)
    solver_kwargs["write_probe_csv"] = True
    solver_kwargs["write_field_output"] = False
    solver_kwargs["enable_front_curve"] = False
    solver_kwargs["show_progress"] = False
    solver_kwargs["probe_z_m"] = tuple(float(value) for value in probe_z_m)
    solver_kwargs["probe_wall_inset_m"] = float(probe_wall_inset_m)
    return config, solver_kwargs


def _run_stage(
    *,
    stage: str,
    profile_name: str,
    simulation_config_path: str | Path,
    max_after_fill_s: float,
    candidates: tuple[FrontSpeedCandidate, ...],
    stage_dir: Path,
    window_s: float,
    sensitivity_windows_s: tuple[float, ...],
    min_measurable_front_mm: float,
    probe_z_m: tuple[float, float, float],
    probe_wall_inset_m: float,
    tc_threshold_C: float,
) -> tuple[list[SpeedSummary], list[dict[str, float]], list[ThermocoupleSpeedSummary], list[dict[str, object]]]:
    config, solver_kwargs = _stage_config(
        profile_name=profile_name,
        simulation_config_path=simulation_config_path,
        max_after_fill_s=max_after_fill_s,
        probe_z_m=probe_z_m,
        probe_wall_inset_m=probe_wall_inset_m,
    )
    geom = solver_kwargs["geom"]
    time_s = _build_time_grid(max_after_fill_s, config.cryostage_dt_s)

    summaries: list[SpeedSummary] = []
    speed_windows: list[dict[str, float]] = []
    tc_summaries: list[ThermocoupleSpeedSummary] = []
    tc_crossing_rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_dir = stage_dir / candidate.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{stage}_{candidate.name}"
        plate_response = build_plate_temperature_response(
            time_s=time_s,
            T_ref_profile_C=candidate.profile(),
            cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
            bcs=solver_kwargs.get("bcs"),
        )
        _write_plate_profile_csv(
            candidate_dir / f"{prefix}_plate_profile.csv",
            time_s=plate_response.cryostage_time_s,
            T_ref_C=plate_response.T_ref_C,
            T_plate_C=plate_response.T_plate_C,
        )
        result = run_open_loop_case(
            time_s=time_s,
            T_ref_profile_C=candidate.profile(),
            cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
            out_dir=candidate_dir,
            prefix=prefix,
            **solver_kwargs,
        )
        summary, windows = summarize_front_speed(
            stage=stage,
            candidate=candidate,
            front_csv_path=result.front_path,
            H_fill_m=float(geom.H_fill),
            window_s=window_s,
            sensitivity_windows_s=sensitivity_windows_s,
            min_measurable_front_mm=min_measurable_front_mm,
        )
        summaries.append(summary)
        speed_windows.extend(windows)
        tc_summary, crossing_rows = summarize_thermocouple_speed(
            stage=stage,
            candidate=candidate,
            probes_csv_path=result.probes_path,
            probe_z_m=probe_z_m,
            threshold_C=float(tc_threshold_C),
        )
        tc_summaries.append(tc_summary)
        tc_crossing_rows.extend(crossing_rows)
    return summaries, speed_windows, tc_summaries, tc_crossing_rows


def _plot_speed_summary(path: Path, summaries: list[SpeedSummary], *, title: str) -> None:
    valid = [summary for summary in summaries if math.isfinite(summary.max_sustained_front_speed_mm_s)]
    if not valid:
        return
    x = np.asarray([summary.target_C for summary in valid], dtype=np.float64)
    y = np.asarray([summary.max_sustained_front_speed_mm_s for summary in valid], dtype=np.float64)
    colors = ["#2A9D8F" if not summary.no_front_or_stalled else "#C44536" for summary in valid]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors, s=55)
    for summary in valid:
        ax.annotate(summary.family, (summary.target_C, summary.max_sustained_front_speed_mm_s), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Max robust front speed (mm/s)")
    fig.savefig(path)
    plt.close(fig)


def _plot_front_vs_tc_speed_by_target(
    path: Path,
    *,
    front_summaries: list[SpeedSummary],
    tc_summaries: list[ThermocoupleSpeedSummary],
    title: str,
) -> None:
    front_holds = [item for item in front_summaries if item.family == "hold"]
    tc_by_key = {(item.stage, item.candidate_name): item for item in tc_summaries}
    x_front: list[float] = []
    y_front: list[float] = []
    x_tc: list[float] = []
    y_tc: list[float] = []
    for front in front_holds:
        if math.isfinite(front.mean_speed_10_90_mm_s):
            x_front.append(front.target_C)
            y_front.append(front.mean_speed_10_90_mm_s)
        tc = tc_by_key.get((front.stage, front.candidate_name))
        if tc is not None and math.isfinite(tc.v_tc_3p0_to_11p0_mm_s):
            x_tc.append(front.target_C)
            y_tc.append(tc.v_tc_3p0_to_11p0_mm_s)
    if not x_front and not x_tc:
        return
    fig, ax = plt.subplots()
    if x_front:
        ax.plot(x_front, y_front, "o-", label="direct front mean 10-90")
    if x_tc:
        ax.plot(x_tc, y_tc, "s--", label="thermocouple 3.0-11.0 mm")
    ax.set_title(title)
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Speed (mm/s)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _plot_tc_segment_speeds(
    path: Path,
    *,
    tc_summaries: list[ThermocoupleSpeedSummary],
    title: str,
) -> None:
    holds = sorted(
        [item for item in tc_summaries if item.family == "hold"],
        key=lambda item: item.target_C,
    )
    if not holds:
        return
    fig, ax = plt.subplots()
    series = (
        ("3.0-6.2 mm", [item.v_tc_3p0_to_6p2_mm_s for item in holds]),
        ("6.2-11.0 mm", [item.v_tc_6p2_to_11p0_mm_s for item in holds]),
        ("3.0-11.0 mm", [item.v_tc_3p0_to_11p0_mm_s for item in holds]),
    )
    x = np.asarray([item.target_C for item in holds], dtype=np.float64)
    for label, values in series:
        y = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Thermocouple speed (mm/s)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _plot_tc_crossing_times(
    path: Path,
    *,
    tc_summaries: list[ThermocoupleSpeedSummary],
    title: str,
) -> None:
    holds = sorted(
        [item for item in tc_summaries if item.family == "hold"],
        key=lambda item: item.target_C,
    )
    if not holds:
        return
    fig, ax = plt.subplots()
    series = (
        ("3.0 mm", [item.crossing_3p0mm_s for item in holds]),
        ("6.2 mm", [item.crossing_6p2mm_s for item in holds]),
        ("11.0 mm", [item.crossing_11p0mm_s for item in holds]),
    )
    x = np.asarray([item.target_C for item in holds], dtype=np.float64)
    for label, values in series:
        y = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.plot(x[mask], y[mask], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Crossing time since fill (s)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _plot_speed_envelope_combined(
    path: Path,
    *,
    front_summaries: list[SpeedSummary],
    tc_summaries: list[ThermocoupleSpeedSummary],
    title: str,
) -> None:
    tc_by_key = {(item.stage, item.candidate_name): item for item in tc_summaries}
    fig, ax = plt.subplots()
    for family, marker in (("hold", "o"), ("ramp_fastest_envelope", "^"), ("ramp_medium", "s"), ("ramp_slow", "D")):
        selected = [item for item in front_summaries if item.family == family]
        if not selected:
            continue
        x = np.asarray([item.target_C for item in selected], dtype=np.float64)
        y = np.asarray([item.max_sustained_front_speed_mm_s for item in selected], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], marker=marker, alpha=0.75, label=f"front {family}")
        y_tc = np.asarray(
            [
                tc_by_key.get((item.stage, item.candidate_name)).v_tc_3p0_to_11p0_mm_s
                if tc_by_key.get((item.stage, item.candidate_name)) is not None
                else math.nan
                for item in selected
            ],
            dtype=np.float64,
        )
        mask_tc = np.isfinite(x) & np.isfinite(y_tc)
        if np.any(mask_tc):
            ax.scatter(x[mask_tc], y_tc[mask_tc], marker=marker, facecolors="none", edgecolors="black", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Speed (mm/s)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def _plot_probe_temperatures_with_crossings(
    path: Path,
    *,
    tc_summary: ThermocoupleSpeedSummary,
    probe_z_m: tuple[float, float, float],
) -> None:
    if not tc_summary.probes_csv_path.exists():
        return
    cols = _load_probe_columns(tc_summary.probes_csv_path, probe_z_m=probe_z_m)
    time_since_fill_s = cols["time_since_fill_s"]
    mask = np.isfinite(time_since_fill_s) & (time_since_fill_s >= 0.0)
    if not np.any(mask):
        return
    fig, ax = plt.subplots()
    crossing_times = (tc_summary.crossing_3p0mm_s, tc_summary.crossing_6p2mm_s, tc_summary.crossing_11p0mm_s)
    for z_m, crossing_s in zip(probe_z_m, crossing_times, strict=True):
        column = _probe_column_name(z_m)
        label = _format_probe_label_mm(z_m).replace("p", ".").replace("mm", " mm")
        ax.plot(time_since_fill_s[mask], cols[column][mask], linewidth=1.8, label=label)
        if math.isfinite(crossing_s):
            ax.scatter([crossing_s], [tc_summary.threshold_C], s=35)
    ax.axhline(tc_summary.threshold_C, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Probe Temperatures: {tc_summary.stage}/{tc_summary.candidate_name}")
    ax.set_xlabel("Time since fill (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _plot_front_position_with_tc_arrivals(
    path: Path,
    *,
    front_summary: SpeedSummary,
    tc_summary: ThermocoupleSpeedSummary,
    probe_z_m: tuple[float, float, float],
) -> None:
    if not front_summary.front_csv_path.exists():
        return
    cols, _ = _load_front_csv_with_instant_speed(front_summary.front_csv_path)
    time_since_fill_s = cols["time_since_fill_s"]
    z_front_mm = 1000.0 * cols["z_front_m"]
    mask = np.isfinite(time_since_fill_s) & np.isfinite(z_front_mm)
    if not np.any(mask):
        return
    fig, ax = plt.subplots()
    ax.plot(time_since_fill_s[mask], z_front_mm[mask], linewidth=2.0, label="direct z_front")
    crossing_times = (tc_summary.crossing_3p0mm_s, tc_summary.crossing_6p2mm_s, tc_summary.crossing_11p0mm_s)
    for z_m, crossing_s in zip(probe_z_m, crossing_times, strict=True):
        z_mm = 1000.0 * float(z_m)
        ax.axhline(z_mm, color="gray", linestyle="--", linewidth=0.9)
        if math.isfinite(crossing_s):
            ax.scatter([crossing_s], [z_mm], s=40)
    ax.set_title(f"Front and TC Arrivals: {front_summary.stage}/{front_summary.candidate_name}")
    ax.set_xlabel("Time since fill (s)")
    ax.set_ylabel("z (mm)")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _read_csv_flexible(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for experimental thermocouple comparison") from exc
    try:
        return pd.read_csv(path, comment="#")
    except Exception:
        return pd.read_csv(path, comment="#", sep=";")


def _smooth_series(values: np.ndarray, window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _detect_experimental_insertion_time_s(time_s: np.ndarray, probe_values: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.size < 2:
        return 0.0
    dt = np.diff(time_s)
    dt[dt == 0.0] = np.nan
    slopes = []
    for values in probe_values:
        smoothed = _smooth_series(np.asarray(values, dtype=np.float64), window=9)
        slopes.append(np.diff(smoothed) / dt)
    score = np.nanmax(np.vstack(slopes), axis=0)
    if not np.any(np.isfinite(score)):
        return 0.0
    idx = int(np.nanargmax(score))
    return float(0.5 * (time_s[idx] + time_s[idx + 1]))


def _target_from_experimental_path(path: Path) -> float:
    text = str(path.parent.name).lower()
    for token, target in (("min5", -5.0), ("min10", -10.0), ("min15", -15.0), ("min20", -20.0), ("min21", -21.0)):
        if token in text:
            return target
    name = path.name.lower()
    for token, target in (("min5", -5.0), ("min10", -10.0), ("min15", -15.0), ("min20", -20.0), ("min21", -21.0)):
        if token in name:
            return target
    return math.nan


def _experimental_tc_rows(*, experimental_data_dir: Path, threshold_C: float) -> list[dict[str, object]]:
    if not experimental_data_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(experimental_data_dir.glob("min*/cryostage_log_*.csv")):
        df = _read_csv_flexible(path)
        required = ("t_rec_s", "T3", "T7", "T12")
        if any(column not in df.columns for column in required):
            continue
        time_s = df["t_rec_s"].to_numpy(dtype=np.float64)
        if np.nanmin(time_s) > 100.0:
            time_s = time_s - time_s[0]
        T3 = df["T3"].to_numpy(dtype=np.float64)
        T7 = df["T7"].to_numpy(dtype=np.float64)
        T12 = df["T12"].to_numpy(dtype=np.float64)
        insertion_s = _detect_experimental_insertion_time_s(time_s, (T3, T7, T12))
        time_since_fill_s = time_s - insertion_s
        post_fill = np.isfinite(time_since_fill_s) & (time_since_fill_s >= 0.0)
        t3 = _first_downward_crossing_time_s(time_since_fill_s[post_fill], T3[post_fill], threshold_C)
        t7 = _first_downward_crossing_time_s(time_since_fill_s[post_fill], T7[post_fill], threshold_C)
        t12 = _first_downward_crossing_time_s(time_since_fill_s[post_fill], T12[post_fill], threshold_C)
        v37 = _segment_speed_mm_s(3.0e-3, 6.2e-3, t3, t7)
        v712 = _segment_speed_mm_s(6.2e-3, 11.0e-3, t7, t12)
        v312 = _segment_speed_mm_s(3.0e-3, 11.0e-3, t3, t12)
        rows.append(
            {
                "source": "experiment",
                "file": str(path),
                "target_C": f"{_target_from_experimental_path(path):.6f}",
                "threshold_C": f"{float(threshold_C):.6f}",
                "insertion_time_s": nan_to_str(insertion_s),
                "crossing_T3_s": nan_to_str(t3),
                "crossing_T7_s": nan_to_str(t7),
                "crossing_T12_s": nan_to_str(t12),
                "v_tc_3p0_to_6p2_mm_s": nan_to_str(v37),
                "v_tc_6p2_to_11p0_mm_s": nan_to_str(v712),
                "v_tc_3p0_to_11p0_mm_s": nan_to_str(v312),
                "thermocouple_speed_complete": str(int(math.isfinite(v37) and math.isfinite(v712) and math.isfinite(v312))),
            }
        )
    return rows


def _write_experimental_comparison(
    *,
    run_dir: Path,
    experimental_data_dir: Path,
    tc_summaries: list[ThermocoupleSpeedSummary],
    threshold_C: float,
) -> None:
    rows = _experimental_tc_rows(experimental_data_dir=experimental_data_dir, threshold_C=threshold_C)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    _write_csv(run_dir / "experimental_tc_speed_summary.csv", rows, fieldnames)

    exp_target = np.asarray([float(row["target_C"]) for row in rows], dtype=np.float64)
    exp_speed = np.asarray([_parse_csv_float(str(row["v_tc_3p0_to_11p0_mm_s"])) for row in rows], dtype=np.float64)
    sim_holds = [item for item in tc_summaries if item.family == "hold" and item.stage in {"fine", "coarse"}]
    fig, ax = plt.subplots()
    mask_exp = np.isfinite(exp_target) & np.isfinite(exp_speed)
    if np.any(mask_exp):
        ax.scatter(exp_target[mask_exp], exp_speed[mask_exp], alpha=0.6, label="experiment runs")
    if sim_holds:
        sim_target = np.asarray([item.target_C for item in sim_holds], dtype=np.float64)
        sim_speed = np.asarray([item.v_tc_3p0_to_11p0_mm_s for item in sim_holds], dtype=np.float64)
        mask_sim = np.isfinite(sim_target) & np.isfinite(sim_speed)
        if np.any(mask_sim):
            ax.plot(sim_target[mask_sim], sim_speed[mask_sim], "ko--", label="simulation holds")
    ax.set_title("Simulation vs Experiment TC Speed")
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Thermocouple speed 3.0-11.0 mm (mm/s)")
    ax.legend(loc="best")
    fig.savefig(run_dir / "figures" / "sim_vs_exp_tc_speed_by_target.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, column, marker in (
        ("T3 / 3.0 mm", "crossing_T3_s", "o"),
        ("T7 / 6.2 mm", "crossing_T7_s", "s"),
        ("T12 / 11.0 mm", "crossing_T12_s", "^"),
    ):
        y = np.asarray([_parse_csv_float(str(row[column])) for row in rows], dtype=np.float64)
        mask = np.isfinite(exp_target) & np.isfinite(y)
        if np.any(mask):
            ax.scatter(exp_target[mask], y[mask], marker=marker, alpha=0.6, label=f"experiment {label}")
    ax.set_title("Experimental TC Crossing Times")
    ax.set_xlabel("Target T_ref (C)")
    ax.set_ylabel("Crossing time since fill (s)")
    ax.legend(loc="best")
    fig.savefig(run_dir / "figures" / "sim_vs_exp_crossing_times_by_target.png")
    plt.close(fig)


def _write_speed_analysis_outputs(
    *,
    run_dir: Path,
    front_summaries: list[SpeedSummary],
    tc_summaries: list[ThermocoupleSpeedSummary],
    tc_crossing_rows: list[dict[str, object]],
    probe_z_m: tuple[float, float, float],
    experimental_data_dir: Path,
    tc_threshold_C: float,
    skip_experimental_comparison: bool,
) -> None:
    tc_rows = [_tc_summary_to_row(item) for item in tc_summaries]
    if tc_rows:
        _write_csv(run_dir / "thermocouple_speed_summary.csv", tc_rows, list(tc_rows[0].keys()))
    if tc_crossing_rows:
        _write_csv(run_dir / "thermocouple_crossings.csv", tc_crossing_rows, list(tc_crossing_rows[0].keys()))

    combined_rows = _combined_summary_rows(front_summaries, tc_summaries)
    if combined_rows:
        _write_csv(run_dir / "combined_speed_summary.csv", combined_rows, list(combined_rows[0].keys()))

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    _plot_front_vs_tc_speed_by_target(
        figures_dir / "front_vs_tc_speed_by_target.png",
        front_summaries=front_summaries,
        tc_summaries=tc_summaries,
        title="Direct Front Speed vs Thermocouple-Equivalent Speed",
    )
    _plot_tc_segment_speeds(
        figures_dir / "tc_segment_speeds_by_target.png",
        tc_summaries=tc_summaries,
        title="Thermocouple Segment Speeds",
    )
    _plot_tc_crossing_times(
        figures_dir / "tc_crossing_times_by_target.png",
        tc_summaries=tc_summaries,
        title="Thermocouple Crossing Times",
    )
    _plot_speed_envelope_combined(
        figures_dir / "speed_envelope_combined.png",
        front_summaries=front_summaries,
        tc_summaries=tc_summaries,
        title="Direct and Thermocouple-Equivalent Speed Envelope",
    )

    front_by_key = {(item.stage, item.candidate_name): item for item in front_summaries}
    for tc_summary in tc_summaries:
        tag = f"{tc_summary.stage}_{tc_summary.candidate_name}"
        _plot_probe_temperatures_with_crossings(
            figures_dir / f"probe_temperatures_with_crossings_{tag}.png",
            tc_summary=tc_summary,
            probe_z_m=probe_z_m,
        )
        front_summary = front_by_key.get((tc_summary.stage, tc_summary.candidate_name))
        if front_summary is not None:
            _plot_front_position_with_tc_arrivals(
                figures_dir / f"front_position_with_tc_arrivals_{tag}.png",
                front_summary=front_summary,
                tc_summary=tc_summary,
                probe_z_m=probe_z_m,
            )

    if not skip_experimental_comparison:
        _write_experimental_comparison(
            run_dir=run_dir,
            experimental_data_dir=experimental_data_dir,
            tc_summaries=tc_summaries,
            threshold_C=tc_threshold_C,
        )


def _write_report(
    path: Path,
    *,
    run_name: str,
    target_range_C: tuple[float, float],
    constraints: ReachabilityConstraints,
    coarse_summaries: list[SpeedSummary],
    fine_summaries: list[SpeedSummary],
    window_s: float,
    sensitivity_windows_s: tuple[float, ...],
    min_measurable_front_mm: float,
    tc_summaries: list[ThermocoupleSpeedSummary],
    tc_threshold_C: float,
    probe_z_m: tuple[float, float, float],
) -> None:
    envelope_summaries = fine_summaries if fine_summaries else coarse_summaries
    envelope_label = "Fine Confirmed Envelope" if fine_summaries else "Coarse Screening Envelope"
    non_stalled = [summary for summary in envelope_summaries if not summary.no_front_or_stalled]
    fastest = (
        max(non_stalled, key=lambda item: item.max_sustained_front_speed_mm_s)
        if non_stalled
        else None
    )
    slowest = (
        min(non_stalled, key=lambda item: item.min_positive_sustained_front_speed_mm_s)
        if non_stalled
        else None
    )
    tc_complete = [summary for summary in tc_summaries if summary.thermocouple_speed_complete]
    fastest_tc = (
        max(tc_complete, key=lambda item: item.v_tc_3p0_to_11p0_mm_s)
        if tc_complete
        else None
    )
    warm_C, cold_C = target_range_C
    probe_z_mm = [1000.0 * float(value) for value in probe_z_m]
    lines = [
        "# Front-Speed Reachability Study",
        "",
        f"- Run name: `{run_name}`",
        f"- Requested operational target range: `{warm_C:.3f} C` to `{cold_C:.3f} C`.",
        f"- Directly characterized cryostage targets: `{list(constraints.characterized_targets_C)}`.",
        "- Targets outside the direct characterization range are marked as extrapolated/operational limits.",
        f"- Primary speed metric: robust linear-regression speed over `{float(window_s):.1f} s` windows.",
        f"- Sensitivity windows: `{list(float(value) for value in sensitivity_windows_s)}` s.",
        f"- Cases with maximum centerline front depth below `{float(min_measurable_front_mm):.3f} mm` are reported as no-front/stalled.",
        f"- Thermocouple-equivalent probes: `{probe_z_mm}` mm.",
        f"- Thermocouple-equivalent crossing threshold: `{float(tc_threshold_C):.3f} C`.",
        "- Existing `v_front_mm_per_s` remains a diagnostic finite-difference speed.",
        "",
        "## Current Solver Speed Definition",
        "",
        "- The solver samples the centerline temperature from the cold plate upward.",
        "- `z_front_m` is the first crossing from `T <= threshold` to `T > threshold`, with linear interpolation.",
        "- The CSV instantaneous speed is `(z_front_current - z_front_previous) * 1000 / dt`.",
        "- This study reports robust windowed speeds as the main result because per-step differences are dt- and sampling-sensitive.",
        "- Thermocouple-equivalent speed uses only the simulated probe temperatures and is the primary metric for experimental comparison.",
        "",
        f"## {envelope_label}",
        "",
    ]
    if not fine_summaries:
        lines.append("- Fine high-fidelity confirmation has not been run for this result set.")
    if fastest is None or slowest is None:
        lines.append("- No non-stalled front-speed envelope was identified in the selected runs.")
    else:
        lines.extend(
            [
                (
                    "- Fastest robust speed: "
                    f"`{fastest.max_sustained_front_speed_mm_s:.6f} mm/s` "
                    f"from `{fastest.candidate_name}` in `{fastest.stage}`."
                ),
                (
                    "- Slowest positive robust speed: "
                    f"`{slowest.min_positive_sustained_front_speed_mm_s:.6f} mm/s` "
                    f"from `{slowest.candidate_name}` in `{slowest.stage}`."
                ),
            ]
        )
    lines.append("")
    lines.append("## Thermocouple-Equivalent Envelope")
    lines.append("")
    if fastest_tc is None:
        lines.append("- No complete thermocouple-equivalent speed was identified in the selected runs.")
    else:
        lines.append(
            "- Fastest thermocouple-equivalent speed: "
            f"`{fastest_tc.v_tc_3p0_to_11p0_mm_s:.6f} mm/s` "
            f"from `{fastest_tc.candidate_name}` in `{fastest_tc.stage}`."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_profiles.csv`: candidates, supports, and admissibility notes.",
            "- `coarse_speed_summary.csv`: coarse screening results.",
            "- `fine_speed_summary.csv`: high-fidelity confirmation of selected extremes.",
            "- `speed_windows.csv`: windowed speed measurements.",
            "- `thermocouple_crossings.csv`: simulated probe crossing times.",
            "- `thermocouple_speed_summary.csv`: thermocouple-equivalent segment speeds.",
            "- `combined_speed_summary.csv`: direct front and thermocouple-equivalent speeds side by side.",
            "- `experimental_tc_speed_summary.csv`: real thermocouple-derived speeds when experimental CSVs are available.",
            "- `figures/`: speed-envelope summary plots.",
            "",
            "## Caveats",
            "",
            "- `0 C` is a warm operational boundary, not a cold characterized target.",
            "- `-21 C` is colder than the directly characterized `-20 C` target.",
            "- Warming is not treated as admissible because dedicated warming-step characterization is still missing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study physically reachable freezing-front speed ranges.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--out-root-dir", default=str(DEFAULT_OUT_ROOT_DIR))
    parser.add_argument("--simulation-config", default=str(DEFAULT_SIMULATION_CONFIG_PATH))
    parser.add_argument("--coarse-profile", default="optimization")
    parser.add_argument("--fine-profile", default="full_process_article")
    parser.add_argument("--target-range-c", default="0,-21")
    parser.add_argument("--targets-c", default=",".join(str(value).rstrip("0").rstrip(".") for value in DEFAULT_TARGETS_C))
    parser.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument(
        "--sensitivity-windows-s",
        default=",".join(str(value).rstrip("0").rstrip(".") for value in DEFAULT_SENSITIVITY_WINDOWS_S),
    )
    parser.add_argument("--max-after-fill-s", type=float, default=DEFAULT_MAX_AFTER_FILL_S)
    parser.add_argument("--confirm-extremes-count", type=int, default=DEFAULT_CONFIRM_EXTREMES_COUNT)
    parser.add_argument(
        "--min-measurable-front-mm",
        type=float,
        default=DEFAULT_MIN_MEASURABLE_FRONT_MM,
        help="Classify cases below this maximum centerline front depth as no-front/stalled.",
    )
    parser.add_argument(
        "--probe-z-mm",
        default=",".join(f"{1000.0 * value:.1f}" for value in DEFAULT_TC_PROBE_Z_M),
        help="Comma-separated thermocouple-equivalent probe heights in mm.",
    )
    parser.add_argument(
        "--probe-wall-inset-mm",
        type=float,
        default=1000.0 * DEFAULT_TC_PROBE_WALL_INSET_M,
        help="Radial inset from the inner wall for thermocouple-equivalent probes.",
    )
    parser.add_argument("--tc-threshold-c", type=float, default=DEFAULT_TC_THRESHOLD_C)
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional limit for smoke-test runs.")
    parser.add_argument("--constraints-dir", default=None)
    parser.add_argument("--analyze-front-csv", default=None, help="Analyze one existing front CSV and print metrics as JSON.")
    parser.add_argument("--analyze-probes-csv", default=None, help="Analyze one existing probes CSV and print thermocouple metrics as JSON.")
    parser.add_argument(
        "--experimental-data-dir",
        default="data/constant_plateT_water_ICT_readings",
        help="Directory containing real constant-plate thermocouple CSVs for optional comparison.",
    )
    parser.add_argument("--skip-experimental-comparison", action="store_true")
    parser.add_argument(
        "--fine-only-from-run-dir",
        default=None,
        help="Run only fine confirmation using candidate_profiles.csv and coarse_speed_summary.csv from this run directory.",
    )
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_matplotlib()

    target_range_C = _parse_target_range(args.target_range_c)
    targets_C = _parse_float_list(args.targets_c)
    sensitivity_windows_s = _parse_float_list(args.sensitivity_windows_s)
    probe_z_m = _parse_probe_z_m(args.probe_z_mm)
    probe_wall_inset_m = 1.0e-3 * float(args.probe_wall_inset_mm)
    tc_threshold_C = float(args.tc_threshold_c)
    constraints = load_reachability_constraints(args.constraints_dir)

    candidates = build_candidates(
        targets_C=targets_C,
        target_range_C=target_range_C,
        max_after_fill_s=float(args.max_after_fill_s),
        constraints=constraints,
    )
    if args.max_candidates is not None:
        candidates = candidates[: max(0, int(args.max_candidates))]

    coarse_config, coarse_solver_kwargs = _stage_config(
        profile_name=args.coarse_profile,
        simulation_config_path=args.simulation_config,
        max_after_fill_s=float(args.max_after_fill_s),
        probe_z_m=probe_z_m,
        probe_wall_inset_m=probe_wall_inset_m,
    )
    fine_config, fine_solver_kwargs = _stage_config(
        profile_name=args.fine_profile,
        simulation_config_path=args.simulation_config,
        max_after_fill_s=float(args.max_after_fill_s),
        probe_z_m=probe_z_m,
        probe_wall_inset_m=probe_wall_inset_m,
    )

    if args.analyze_front_csv is not None:
        dummy_candidate = FrontSpeedCandidate(
            name="external_front_csv",
            family="external",
            target_C=math.nan,
            support_status="unknown",
            support_note="Existing front CSV analysis.",
            knot_times_s=(0.0, float(args.max_after_fill_s)),
            knot_temperatures_C=(0.0, 0.0),
            admissible=False,
            admissibility_reasons=(),
            description="Existing front CSV analysis.",
        )
        summary, windows = summarize_front_speed(
            stage="external",
            candidate=dummy_candidate,
            front_csv_path=Path(args.analyze_front_csv),
            H_fill_m=float(coarse_solver_kwargs["geom"].H_fill),
            window_s=float(args.window_s),
            sensitivity_windows_s=sensitivity_windows_s,
            min_measurable_front_mm=float(args.min_measurable_front_mm),
        )
        print(
            json.dumps(
                {
                    "summary": _summary_to_row(summary),
                    "num_speed_windows": len(windows),
                    "primary_window_s": float(args.window_s),
                },
                indent=2,
            )
        )
        return

    if args.analyze_probes_csv is not None:
        dummy_candidate = FrontSpeedCandidate(
            name="external_probes_csv",
            family="external",
            target_C=math.nan,
            support_status="unknown",
            support_note="Existing probes CSV analysis.",
            knot_times_s=(0.0, float(args.max_after_fill_s)),
            knot_temperatures_C=(0.0, 0.0),
            admissible=False,
            admissibility_reasons=(),
            description="Existing probes CSV analysis.",
        )
        summary, crossing_rows = summarize_thermocouple_speed(
            stage="external",
            candidate=dummy_candidate,
            probes_csv_path=Path(args.analyze_probes_csv),
            probe_z_m=probe_z_m,
            threshold_C=tc_threshold_C,
        )
        print(
            json.dumps(
                {
                    "summary": _tc_summary_to_row(summary),
                    "crossings": crossing_rows,
                },
                indent=2,
            )
        )
        return

    dry_payload = {
        "run_name": str(args.run_name),
        "target_range_C": [float(value) for value in target_range_C],
        "targets_C": [float(value) for value in targets_C],
        "constraints_dir": str(constraints.constraints_dir.resolve()),
        "directly_characterized_targets_C": [float(value) for value in constraints.characterized_targets_C],
        "coarse_profile": {
            "name": str(args.coarse_profile),
            "cryostage_dt_s": float(coarse_config.cryostage_dt_s),
            "Nr": int(coarse_solver_kwargs["Nr"]),
            "Nz": int(coarse_solver_kwargs["Nz"]),
            "dt": float(coarse_solver_kwargs["dt"]),
            "Nz_front": int(coarse_solver_kwargs["Nz_front"]),
            "write_probe_csv": bool(coarse_solver_kwargs.get("write_probe_csv")),
            "write_field_output": bool(coarse_solver_kwargs.get("write_field_output")),
            "enable_front_curve": bool(coarse_solver_kwargs.get("enable_front_curve")),
            "ambient": describe_ambient_temperature_model(coarse_solver_kwargs.get("ambient_temperature_from_plate_C")),
        },
        "fine_profile": {
            "name": str(args.fine_profile),
            "cryostage_dt_s": float(fine_config.cryostage_dt_s),
            "Nr": int(fine_solver_kwargs["Nr"]),
            "Nz": int(fine_solver_kwargs["Nz"]),
            "dt": float(fine_solver_kwargs["dt"]),
            "Nz_front": int(fine_solver_kwargs["Nz_front"]),
            "write_probe_csv": bool(fine_solver_kwargs.get("write_probe_csv")),
            "write_field_output": bool(fine_solver_kwargs.get("write_field_output")),
            "enable_front_curve": bool(fine_solver_kwargs.get("enable_front_curve")),
            "ambient": describe_ambient_temperature_model(fine_solver_kwargs.get("ambient_temperature_from_plate_C")),
        },
        "speed_metric": {
            "primary": "linear regression slope of z_front_mm(t) over sliding windows",
            "window_s": float(args.window_s),
            "sensitivity_windows_s": [float(value) for value in sensitivity_windows_s],
            "min_measurable_front_mm": float(args.min_measurable_front_mm),
        },
        "thermocouple_metric": {
            "probe_z_mm": [1000.0 * float(value) for value in probe_z_m],
            "probe_wall_inset_mm": 1000.0 * float(probe_wall_inset_m),
            "threshold_C": float(tc_threshold_C),
            "primary": "mean speed between 3.0 mm and 11.0 mm downward crossings",
        },
        "candidates": [
            {
                "candidate_name": candidate.name,
                "family": candidate.family,
                "target_C": float(candidate.target_C),
                "support_status": candidate.support_status,
                "support_note": candidate.support_note,
                "admissible": bool(candidate.admissible),
                "admissibility_reasons": list(candidate.admissibility_reasons),
                "knot_times_s": [float(value) for value in candidate.knot_times_s],
                "knot_temperatures_C": [float(value) for value in candidate.knot_temperatures_C],
            }
            for candidate in candidates
        ],
    }

    if args.dry_run_config:
        if args.fine_only_from_run_dir is not None:
            run_dir = Path(args.fine_only_from_run_dir).resolve()
            existing_candidates_by_name = _read_candidate_profiles(run_dir / "candidate_profiles.csv")
            existing_coarse_summaries = _read_speed_summary_csv(
                run_dir / "coarse_speed_summary.csv",
                min_measurable_front_mm=float(args.min_measurable_front_mm),
            )
            selected = _select_fine_candidates(
                existing_coarse_summaries,
                existing_candidates_by_name,
                confirm_extremes_count=int(args.confirm_extremes_count),
            )
            dry_payload["fine_only_from_run_dir"] = str(run_dir)
            dry_payload["selected_fine_candidates"] = [
                {
                    "candidate_name": candidate.name,
                    "family": candidate.family,
                    "target_C": float(candidate.target_C),
                    "support_status": candidate.support_status,
                    "admissible": bool(candidate.admissible),
                }
                for candidate in selected
            ]
        print(json.dumps(dry_payload, indent=2))
        return

    window_fieldnames = [
        "stage",
        "candidate_name",
        "family",
        "target_C",
        "is_primary_window",
        "window_s",
        "window_start_s",
        "window_end_s",
        "window_center_s",
        "speed_mm_s",
        "r2",
        "n_samples",
    ]

    if args.fine_only_from_run_dir is not None:
        run_dir = Path(args.fine_only_from_run_dir).resolve()
        (run_dir / "figures").mkdir(exist_ok=True)
        candidates_by_name = _read_candidate_profiles(run_dir / "candidate_profiles.csv")
        coarse_summaries = _read_speed_summary_csv(
            run_dir / "coarse_speed_summary.csv",
            min_measurable_front_mm=float(args.min_measurable_front_mm),
        )
        fine_candidates = _select_fine_candidates(
            coarse_summaries,
            candidates_by_name,
            confirm_extremes_count=int(args.confirm_extremes_count),
        )
        if not fine_candidates:
            raise RuntimeError("No fine candidates were selected from the existing coarse summary.")

        fine_dir = run_dir / "fine"
        if fine_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{fine_dir} already exists. Re-run with --overwrite to replace fine outputs.")
            shutil.rmtree(fine_dir)

        fine_summaries, fine_windows, fine_tc_summaries, fine_tc_crossings = _run_stage(
            stage="fine",
            profile_name=args.fine_profile,
            simulation_config_path=args.simulation_config,
            max_after_fill_s=float(args.max_after_fill_s),
            candidates=fine_candidates,
            stage_dir=fine_dir,
            window_s=float(args.window_s),
            sensitivity_windows_s=sensitivity_windows_s,
            min_measurable_front_mm=float(args.min_measurable_front_mm),
            probe_z_m=probe_z_m,
            probe_wall_inset_m=probe_wall_inset_m,
            tc_threshold_C=tc_threshold_C,
        )
        summary_fieldnames = list(_summary_to_row(fine_summaries[0]).keys()) if fine_summaries else []
        _write_csv(run_dir / "fine_speed_summary.csv", [_summary_to_row(item) for item in fine_summaries], summary_fieldnames)
        _plot_speed_summary(run_dir / "figures" / "fine_speed_envelope.png", fine_summaries, title="Fine Front-Speed Envelope")

        existing_window_rows: list[dict[str, object]] = []
        speed_windows_path = run_dir / "speed_windows.csv"
        if speed_windows_path.exists():
            with speed_windows_path.open(newline="") as f:
                existing_window_rows = [
                    dict(row)
                    for row in csv.DictReader(f)
                    if str(row.get("stage", "")) != "fine"
                ]
        _write_csv(speed_windows_path, [*existing_window_rows, *fine_windows], window_fieldnames)

        existing_tc_summaries = [
            item for item in _read_tc_summary_csv(run_dir / "thermocouple_speed_summary.csv")
            if item.stage != "fine"
        ]
        existing_crossing_rows: list[dict[str, object]] = []
        crossings_path = run_dir / "thermocouple_crossings.csv"
        if crossings_path.exists():
            with crossings_path.open(newline="") as f:
                existing_crossing_rows = [
                    dict(row)
                    for row in csv.DictReader(f)
                    if str(row.get("stage", "")) != "fine"
                ]
        all_front_summaries = [*coarse_summaries, *fine_summaries]
        all_tc_summaries = [*existing_tc_summaries, *fine_tc_summaries]
        _write_speed_analysis_outputs(
            run_dir=run_dir,
            front_summaries=all_front_summaries,
            tc_summaries=all_tc_summaries,
            tc_crossing_rows=[*existing_crossing_rows, *fine_tc_crossings],
            probe_z_m=probe_z_m,
            experimental_data_dir=Path(args.experimental_data_dir),
            tc_threshold_C=tc_threshold_C,
            skip_experimental_comparison=bool(args.skip_experimental_comparison),
        )
        _write_report(
            run_dir / "front_speed_reachability_report.md",
            run_name=str(args.run_name),
            target_range_C=target_range_C,
            constraints=constraints,
            coarse_summaries=coarse_summaries,
            fine_summaries=fine_summaries,
            window_s=float(args.window_s),
            sensitivity_windows_s=sensitivity_windows_s,
            min_measurable_front_mm=float(args.min_measurable_front_mm),
            tc_summaries=all_tc_summaries,
            tc_threshold_C=tc_threshold_C,
            probe_z_m=probe_z_m,
        )
        print(f"Fine confirmation written to {run_dir}")
        print(f"Fine summary: {run_dir / 'fine_speed_summary.csv'}")
        print(f"Report: {run_dir / 'front_speed_reachability_report.md'}")
        return

    out_root_dir = Path(args.out_root_dir).resolve()
    run_dir = out_root_dir / str(args.run_name)
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{run_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    (run_dir / "dry_run_config.json").write_text(json.dumps(dry_payload, indent=2) + "\n", encoding="utf-8")
    _write_candidate_profiles(run_dir / "candidate_profiles.csv", candidates)

    coarse_summaries, coarse_windows, coarse_tc_summaries, coarse_tc_crossings = _run_stage(
        stage="coarse",
        profile_name=args.coarse_profile,
        simulation_config_path=args.simulation_config,
        max_after_fill_s=float(args.max_after_fill_s),
        candidates=candidates,
        stage_dir=run_dir / "coarse",
        window_s=float(args.window_s),
        sensitivity_windows_s=sensitivity_windows_s,
        min_measurable_front_mm=float(args.min_measurable_front_mm),
        probe_z_m=probe_z_m,
        probe_wall_inset_m=probe_wall_inset_m,
        tc_threshold_C=tc_threshold_C,
    )
    summary_fieldnames = list(_summary_to_row(coarse_summaries[0]).keys()) if coarse_summaries else []
    if summary_fieldnames:
        _write_csv(run_dir / "coarse_speed_summary.csv", [_summary_to_row(item) for item in coarse_summaries], summary_fieldnames)
    _plot_speed_summary(run_dir / "figures" / "coarse_speed_envelope.png", coarse_summaries, title="Coarse Front-Speed Envelope")

    candidates_by_name = {candidate.name: candidate for candidate in candidates}
    fine_candidates = _select_fine_candidates(
        coarse_summaries,
        candidates_by_name,
        confirm_extremes_count=int(args.confirm_extremes_count),
    )
    fine_summaries: list[SpeedSummary] = []
    fine_windows: list[dict[str, float]] = []
    fine_tc_summaries: list[ThermocoupleSpeedSummary] = []
    fine_tc_crossings: list[dict[str, object]] = []
    if fine_candidates:
        fine_summaries, fine_windows, fine_tc_summaries, fine_tc_crossings = _run_stage(
            stage="fine",
            profile_name=args.fine_profile,
            simulation_config_path=args.simulation_config,
            max_after_fill_s=float(args.max_after_fill_s),
            candidates=fine_candidates,
            stage_dir=run_dir / "fine",
            window_s=float(args.window_s),
            sensitivity_windows_s=sensitivity_windows_s,
            min_measurable_front_mm=float(args.min_measurable_front_mm),
            probe_z_m=probe_z_m,
            probe_wall_inset_m=probe_wall_inset_m,
            tc_threshold_C=tc_threshold_C,
        )
        _write_csv(run_dir / "fine_speed_summary.csv", [_summary_to_row(item) for item in fine_summaries], summary_fieldnames)
        _plot_speed_summary(run_dir / "figures" / "fine_speed_envelope.png", fine_summaries, title="Fine Front-Speed Envelope")
    else:
        _write_csv(run_dir / "fine_speed_summary.csv", [], summary_fieldnames)

    window_rows = coarse_windows + fine_windows
    _write_csv(run_dir / "speed_windows.csv", window_rows, window_fieldnames)
    all_front_summaries = [*coarse_summaries, *fine_summaries]
    all_tc_summaries = [*coarse_tc_summaries, *fine_tc_summaries]
    _write_speed_analysis_outputs(
        run_dir=run_dir,
        front_summaries=all_front_summaries,
        tc_summaries=all_tc_summaries,
        tc_crossing_rows=[*coarse_tc_crossings, *fine_tc_crossings],
        probe_z_m=probe_z_m,
        experimental_data_dir=Path(args.experimental_data_dir),
        tc_threshold_C=tc_threshold_C,
        skip_experimental_comparison=bool(args.skip_experimental_comparison),
    )
    _write_report(
        run_dir / "front_speed_reachability_report.md",
        run_name=str(args.run_name),
        target_range_C=target_range_C,
        constraints=constraints,
        coarse_summaries=coarse_summaries,
        fine_summaries=fine_summaries,
        window_s=float(args.window_s),
        sensitivity_windows_s=sensitivity_windows_s,
        min_measurable_front_mm=float(args.min_measurable_front_mm),
        tc_summaries=all_tc_summaries,
        tc_threshold_C=tc_threshold_C,
        probe_z_m=probe_z_m,
    )
    print(f"Front-speed reachability study written to {run_dir}")
    print(f"Candidate profiles: {run_dir / 'candidate_profiles.csv'}")
    print(f"Coarse summary: {run_dir / 'coarse_speed_summary.csv'}")
    print(f"Fine summary: {run_dir / 'fine_speed_summary.csv'}")
    print(f"Report: {run_dir / 'front_speed_reachability_report.md'}")


if __name__ == "__main__":
    main()
