#!/usr/bin/env python3
"""Evaluate one manual cryostage trajectory against a target front speed."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from code_simulation.core.config_files import (
    DEFAULT_SIMULATION_CONFIG_PATH,
    DEFAULT_VELOCITY_CONTROL_CONFIG_PATH,
    ManualTrajectoryConfig,
    VelocityControlFileConfig,
    load_simulation_profile,
    load_velocity_control_config,
)
from code_simulation.core.paths import velocity_control_results_dir
from code_simulation.core.plotting import configure_matplotlib
from code_simulation.core.plate_tracking import (
    PlateTrackingSummary,
    summarize_plate_tracking,
    write_plate_tracking_summary_csv,
    write_plate_tracking_timeseries_csv,
)
from code_simulation.core.trajectory_profiles import PiecewiseLinearTemperatureProfile


PROBE_Z_MM = (3.0, 6.2, 11.0)
PROBE_Z_M = tuple(value * 1.0e-3 for value in PROBE_Z_MM)
PROBE_WALL_INSET_M = 1.0e-3
H_OUT_W_M2K = 2.0
SEGMENT_SPEED_NUM_SEGMENTS = 3


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all comma-separated values must be finite")
    return values


def _parse_optional_float_tuple(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = _parse_float_tuple(raw)
    return None if len(values) == 0 else values


def _validate_support_tau(support_tau: np.ndarray, *, num_knots: int, name: str) -> tuple[float, ...]:
    if support_tau.ndim != 1 or support_tau.size != int(num_knots):
        raise ValueError(f"{name} must contain exactly {int(num_knots)} values")
    if not np.all(np.isfinite(support_tau)):
        raise ValueError(f"{name} must contain finite values")
    if abs(float(support_tau[0])) > 1.0e-12 or abs(float(support_tau[-1]) - 1.0) > 1.0e-12:
        raise ValueError(f"{name} must start at 0.0 and end at 1.0")
    if np.any(np.diff(support_tau) <= 1.0e-12):
        raise ValueError(f"{name} must be strictly increasing")
    if np.any(support_tau < -1.0e-12) or np.any(support_tau > 1.0 + 1.0e-12):
        raise ValueError(f"{name} must stay within [0, 1]")
    return tuple(float(value) for value in np.clip(support_tau, 0.0, 1.0))


def build_knot_times_s(config: ManualTrajectoryConfig) -> tuple[float, ...]:
    schedule = config.knot_time_schedule
    num_knots = int(config.num_knots)
    uniform_tau = np.linspace(0.0, 1.0, num_knots, dtype=np.float64)
    if schedule == "uniform":
        support_tau = uniform_tau
    elif schedule == "early_dense":
        support_tau = np.power(uniform_tau, 1.5)
    elif schedule == "late_dense":
        support_tau = 1.0 - np.power(1.0 - uniform_tau, 1.5)
    elif schedule == "mid_dense":
        centered = 2.0 * uniform_tau - 1.0
        support_tau = 0.5 * (1.0 + np.sign(centered) * np.power(np.abs(centered), 2.0))
        support_tau[0] = 0.0
        support_tau[-1] = 1.0
    elif schedule == "custom":
        if config.knot_time_custom_support_tau is None:
            raise ValueError("custom knot schedule requires knot_time_custom_support_tau")
        support_tau = np.asarray(config.knot_time_custom_support_tau, dtype=np.float64)
    else:
        raise ValueError(
            "manual_trajectory.knot_time_schedule must be one of "
            "'uniform', 'early_dense', 'mid_dense', 'late_dense', or 'custom'"
        )
    support = _validate_support_tau(support_tau, num_knots=num_knots, name="knot support")
    return tuple(float(config.horizon_s * tau) for tau in support)


def _time_grid_s(horizon_s: float, dt_s: float) -> np.ndarray:
    time_s = np.arange(0.0, float(horizon_s), float(dt_s), dtype=np.float64)
    if time_s.size == 0 or time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
    if time_s[-1] < float(horizon_s) - 1.0e-12:
        time_s = np.append(time_s, float(horizon_s))
    return time_s


def _read_key_value_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["parameter"]: row["value"] for row in csv.DictReader(f)}


def _read_numeric_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    data = {name: [] for name in columns}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in columns:
                try:
                    data[name].append(float(row.get(name, "nan")))
                except (TypeError, ValueError):
                    data[name].append(math.nan)
    return {name: np.asarray(values, dtype=np.float64) for name, values in data.items()}


def _first_time_at_or_above(time_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    previous_t = math.nan
    previous_v = math.nan
    for t_i, value_i in zip(time_s, values, strict=False):
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


def _first_downward_crossing_time_s(time_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    previous_t = math.nan
    previous_v = math.nan
    for t_i, value_i in zip(time_s, values, strict=False):
        t_i = float(t_i)
        value_i = float(value_i)
        if not (math.isfinite(t_i) and math.isfinite(value_i)):
            continue
        if value_i <= threshold:
            if math.isfinite(previous_t) and math.isfinite(previous_v) and previous_v > value_i:
                alpha = (previous_v - float(threshold)) / (previous_v - value_i)
                return float(previous_t + alpha * (t_i - previous_t))
            return t_i
        previous_t = t_i
        previous_v = value_i
    return math.nan


def _tracking_summary(
    *,
    front_path: Path,
    cooling_start_time_s: float,
    target_speed_mm_s: float,
    z_min_mm: float,
    z_max_mm: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    cols = _read_numeric_csv_columns(front_path, ("time_s", "z_front_mm", "v_front_mm_per_s"))
    control_time_s = cols["time_s"] - float(cooling_start_time_s)
    z_front_mm = cols["z_front_mm"]
    finite = np.isfinite(control_time_s) & np.isfinite(z_front_mm) & (control_time_s >= 0.0)

    t_z_min_s = _first_time_at_or_above(control_time_s[finite], z_front_mm[finite], z_min_mm)
    t_z_max_s = _first_time_at_or_above(control_time_s[finite], z_front_mm[finite], z_max_mm)
    expected_interval_s = (float(z_max_mm) - float(z_min_mm)) / float(target_speed_mm_s)
    expected_t_z_max_s = t_z_min_s + expected_interval_s if math.isfinite(t_z_min_s) else math.nan

    z_ref_mm = np.full_like(control_time_s, math.nan, dtype=np.float64)
    if math.isfinite(t_z_min_s):
        z_ref_mm = float(z_min_mm) + float(target_speed_mm_s) * (control_time_s - t_z_min_s)

    objective_mask = (
        np.isfinite(control_time_s)
        & np.isfinite(z_front_mm)
        & np.isfinite(z_ref_mm)
        & (control_time_s >= t_z_min_s)
        & (z_ref_mm >= float(z_min_mm) - 1.0e-12)
        & (z_ref_mm <= float(z_max_mm) + 1.0e-12)
    )
    if np.any(objective_mask):
        error_mm = z_front_mm[objective_mask] - z_ref_mm[objective_mask]
        rmse_mm = float(np.sqrt(np.mean(error_mm * error_mm)))
        mean_error_mm = float(np.mean(error_mm))
        max_abs_error_mm = float(np.max(np.abs(error_mm)))
    else:
        rmse_mm = math.nan
        mean_error_mm = math.nan
        max_abs_error_mm = math.nan

    if math.isfinite(t_z_min_s) and math.isfinite(t_z_max_s) and t_z_max_s > t_z_min_s:
        actual_interval_speed_mm_s = (float(z_max_mm) - float(z_min_mm)) / (t_z_max_s - t_z_min_s)
    else:
        actual_interval_speed_mm_s = math.nan

    summary = {
        "target_front_speed_mm_s": float(target_speed_mm_s),
        "control_z_min_mm": float(z_min_mm),
        "control_z_max_mm": float(z_max_mm),
        "t_at_control_z_min_s": float(t_z_min_s),
        "t_at_control_z_max_s": float(t_z_max_s),
        "expected_t_at_control_z_max_s": float(expected_t_z_max_s),
        "actual_interval_speed_mm_s": float(actual_interval_speed_mm_s),
        "tracking_rmse_mm": float(rmse_mm),
        "tracking_mean_error_mm": float(mean_error_mm),
        "tracking_max_abs_error_mm": float(max_abs_error_mm),
        "num_tracking_samples": float(np.count_nonzero(objective_mask)),
    }
    series = {
        "control_time_s": control_time_s,
        "z_front_mm": z_front_mm,
        "z_ref_mm": z_ref_mm,
        "v_front_mm_per_s": cols["v_front_mm_per_s"],
        "objective_mask": objective_mask.astype(np.float64),
    }
    return summary, series


def _segment_speed_summary(
    *,
    series: dict[str, np.ndarray],
    target_speed_mm_s: float,
    z_min_mm: float,
    z_max_mm: float,
    num_segments: int = SEGMENT_SPEED_NUM_SEGMENTS,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    control_time_s = series["control_time_s"]
    z_front_mm = series["z_front_mm"]
    finite = np.isfinite(control_time_s) & np.isfinite(z_front_mm) & (control_time_s >= 0.0)
    boundaries = np.linspace(float(z_min_mm), float(z_max_mm), int(num_segments) + 1, dtype=np.float64)
    crossing_times = [
        _first_time_at_or_above(control_time_s[finite], z_front_mm[finite], float(boundary))
        for boundary in boundaries
    ]

    rows: list[dict[str, float | int]] = []
    speeds: list[float] = []
    target = float(target_speed_mm_s)
    for idx in range(int(num_segments)):
        z0 = float(boundaries[idx])
        z1 = float(boundaries[idx + 1])
        t0 = float(crossing_times[idx])
        t1 = float(crossing_times[idx + 1])
        if math.isfinite(t0) and math.isfinite(t1) and t1 > t0:
            speed = float((z1 - z0) / (t1 - t0))
            signed_relative_error = float((speed - target) / target)
            abs_relative_error_pct = float(abs(signed_relative_error) * 100.0)
            speeds.append(speed)
        else:
            speed = math.nan
            signed_relative_error = math.nan
            abs_relative_error_pct = math.nan
        rows.append(
            {
                "segment_index": int(idx),
                "z_start_mm": z0,
                "z_end_mm": z1,
                "t_start_crossing_s": t0,
                "t_end_crossing_s": t1,
                "speed_mm_s": float(speed),
                "signed_relative_error": float(signed_relative_error),
                "abs_relative_error_pct": float(abs_relative_error_pct),
            }
        )

    speed_arr = np.asarray(speeds, dtype=np.float64)
    if speed_arr.size:
        abs_error_pct = np.abs((speed_arr - target) / target) * 100.0
        rmse_pct = float(np.sqrt(np.mean(abs_error_pct * abs_error_pct)))
        mean_abs_pct = float(np.mean(abs_error_pct))
        max_abs_pct = float(np.max(abs_error_pct))
        min_speed = float(np.min(speed_arr))
        max_speed = float(np.max(speed_arr))
        spread = float(max_speed - min_speed)
    else:
        rmse_pct = math.nan
        mean_abs_pct = math.nan
        max_abs_pct = math.nan
        min_speed = math.nan
        max_speed = math.nan
        spread = math.nan

    summary = {
        "target_front_speed_mm_s": target,
        "segment_speed_num_segments": int(num_segments),
        "segment_speed_num_valid_segments": int(speed_arr.size),
        "segment_speed_rmse_pct": float(rmse_pct),
        "segment_speed_mean_abs_error_pct": float(mean_abs_pct),
        "segment_speed_max_abs_error_pct": float(max_abs_pct),
        "segment_speed_min_mm_s": float(min_speed),
        "segment_speed_max_mm_s": float(max_speed),
        "segment_speed_spread_mm_s": float(spread),
    }
    return summary, rows


def _thermocouple_speeds(
    *,
    probes_path: Path,
    cooling_start_time_s: float,
    threshold_C: float = 0.0,
) -> list[dict[str, float | str]]:
    column_by_z = {
        3.0: "T_z3p0mm_C",
        6.2: "T_z6p2mm_C",
        11.0: "T_z11p0mm_C",
    }
    cols = _read_numeric_csv_columns(probes_path, ("time_s", *tuple(column_by_z.values())))
    control_time_s = cols["time_s"] - float(cooling_start_time_s)
    valid_time = np.isfinite(control_time_s) & (control_time_s >= 0.0)
    crossing_by_z: dict[float, float] = {}
    for z_mm, column in column_by_z.items():
        crossing_by_z[z_mm] = _first_downward_crossing_time_s(
            control_time_s[valid_time],
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


def _write_single_row_csv(path: Path, row: dict[str, float | str | int]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_rows_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_effective_config(
    *,
    config: VelocityControlFileConfig,
    simulation_profile,
    knot_times_s: tuple[float, ...],
    output_dir: Path,
) -> str:
    trajectory = config.manual_trajectory
    target = config.velocity_target
    initial = config.initial_conditions
    uncertainty = config.temperature_uncertainty
    lines = [
        "# Effective configuration used by run_velocity_control_evaluation.",
        "",
        "[run]",
        f'run_name = "{config.run_name}"',
        f'simulation_profile = "{simulation_profile.name}"',
        f'output_dir = "{output_dir}"',
        "",
        "[simulation_profile]",
        f'name = "{simulation_profile.name}"',
        f"cryostage_dt_s = {simulation_profile.cryostage_dt_s:.12g}",
        f"solver_dt_s = {simulation_profile.solver_dt_s:.12g}",
        f"Nr = {int(simulation_profile.Nr)}",
        f"Nz = {int(simulation_profile.Nz)}",
        f"Nz_front = {int(simulation_profile.Nz_front)}",
        f"Nr_front_curve = {int(simulation_profile.Nr_front_curve)}",
        f"Nz_front_curve = {int(simulation_profile.Nz_front_curve)}",
        f"write_field_output = {str(bool(simulation_profile.write_field_output)).lower()}",
        f"write_probe_csv = {str(bool(simulation_profile.write_probe_csv)).lower()}",
        f"enable_front_curve = {str(bool(simulation_profile.enable_front_curve)).lower()}",
        f"use_tabulated_water_ice = {str(bool(simulation_profile.use_tabulated_water_ice)).lower()}",
        "",
        "[velocity_target]",
        f"target_front_speed_mm_s = {target.target_front_speed_mm_s:.12g}",
        f"control_z_min_mm = {target.control_z_min_mm:.12g}",
        f"control_z_max_mm = {target.control_z_max_mm:.12g}",
        "",
        "[initial_conditions]",
        f"initial_water_temperature_C = {initial.initial_water_temperature_C:.12g}",
        f"initial_plate_temperature_C = {initial.initial_plate_temperature_C:.12g}",
        "no_warm_hold = true",
        "",
        "[temperature_uncertainty]",
        (
            "characterization_temperature_margin_C = "
            f"{uncertainty.characterization_temperature_margin_C:.12g}"
        ),
        "",
        "[manual_trajectory]",
        f"horizon_s = {trajectory.horizon_s:.12g}",
        f"num_knots = {trajectory.num_knots}",
        f'knot_time_schedule = "{trajectory.knot_time_schedule}"',
        f"knot_time_custom_support_tau = {list(trajectory.knot_time_custom_support_tau or [])}",
        f"knot_times_s = {list(knot_times_s)}",
        f"theta_C = {list(trajectory.theta_C)}",
        f"T_ref_bounds_C = {list(trajectory.T_ref_bounds_C)}",
        f"require_monotone_nonincreasing = {str(trajectory.require_monotone_nonincreasing).lower()}",
        "",
    ]
    return "\n".join(lines)


def _write_effective_config(
    path: Path,
    *,
    config: VelocityControlFileConfig,
    simulation_profile,
    knot_times_s: tuple[float, ...],
    output_dir: Path,
) -> None:
    path.write_text(
        _render_effective_config(
            config=config,
            simulation_profile=simulation_profile,
            knot_times_s=knot_times_s,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )


def _plot_temperature_profiles(
    path: Path,
    *,
    time_s: np.ndarray,
    T_ref_C: np.ndarray,
    T_plate_C: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    fig, ax = plt.subplots()
    ax.plot(time_s, T_ref_C, label="T_ref")
    ax.plot(time_s, T_plate_C, label="T_plate")
    ax.set_xlabel("Time since cooling start (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Cryostage reference and modeled plate temperature")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def _plot_front_position(path: Path, *, series: dict[str, np.ndarray], z_min_mm: float, z_max_mm: float) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    fig, ax = plt.subplots()
    ax.plot(series["control_time_s"], series["z_front_mm"], label="Simulated front")
    ax.plot(series["control_time_s"], series["z_ref_mm"], linestyle="--", label="Target reference")
    ax.axhline(z_min_mm, color="0.35", linestyle=":", linewidth=1.0, label="Control window")
    ax.axhline(z_max_mm, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time since cooling start (s)")
    ax.set_ylabel("Front position (mm)")
    ax.set_title("Freezing-front position against target-speed reference")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def _plot_front_velocity(path: Path, *, series: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib(plt)
    mask = np.isfinite(series["control_time_s"]) & np.isfinite(series["v_front_mm_per_s"])
    fig, ax = plt.subplots()
    ax.plot(series["control_time_s"][mask], series["v_front_mm_per_s"][mask], linewidth=1.0)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time since cooling start (s)")
    ax.set_ylabel("Direct front velocity (mm/s)")
    ax.set_title("Direct front velocity diagnostic")
    fig.savefig(path)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    config: VelocityControlFileConfig,
    timing: dict[str, str],
    summary: dict[str, float],
    segment_summary: dict[str, float | int],
    segment_rows: list[dict[str, float | int]],
    plate_summary: PlateTrackingSummary,
    tc_rows: list[dict[str, float | str]],
) -> None:
    segment_lines = []
    for row in segment_rows:
        speed = row["speed_mm_s"]
        speed_text = f"{float(speed):.6g} mm/s" if math.isfinite(float(speed)) else "not reached"
        error = row["abs_relative_error_pct"]
        error_text = f"{float(error):.6g}%" if math.isfinite(float(error)) else "nan"
        segment_lines.append(
            f"- segment {int(row['segment_index'])}: "
            f"{row['z_start_mm']:.3g}-{row['z_end_mm']:.3g} mm, "
            f"{speed_text}, abs. error {error_text}"
        )
    tc_lines = []
    for row in tc_rows:
        speed = row["speed_mm_s"]
        speed_text = f"{float(speed):.6g} mm/s" if math.isfinite(float(speed)) else "not reached"
        tc_lines.append(f"- {row['interval']}: {speed_text}")
    path.write_text(
        "\n".join(
            [
                "# Velocity Control Evaluation",
                "",
                "This run evaluates one manual cryostage temperature trajectory. It does not run BO.",
                "",
                f"- Target speed: `{config.velocity_target.target_front_speed_mm_s:.6g} mm/s`",
                (
                    "- Control region: "
                    f"`{config.velocity_target.control_z_min_mm:.3g}-{config.velocity_target.control_z_max_mm:.3g} mm`"
                ),
                "- Warm hold: `disabled`",
                f"- Initial water temperature: `{config.initial_conditions.initial_water_temperature_C:.6g} C`",
                f"- Initial plate temperature: `{config.initial_conditions.initial_plate_temperature_C:.6g} C`",
                (
                    "- Characterization temperature margin: "
                    f"`+-{config.temperature_uncertainty.characterization_temperature_margin_C:.6g} C`"
                ),
                f"- Fill time: `{timing.get('fill_time_s', 'nan')} s`",
                f"- Cooling start time: `{timing.get('cooling_start_time_s', 'nan')} s`",
                "",
                "## Front-Tracking Summary",
                "",
                f"- Actual interval speed: `{summary['actual_interval_speed_mm_s']:.6g} mm/s`",
                f"- Tracking RMSE: `{summary['tracking_rmse_mm']:.6g} mm`",
                f"- Number of tracking samples: `{int(summary['num_tracking_samples'])}`",
                "",
                "## Segment-Speed Summary",
                "",
                (
                    "- Segment-speed RMSE: "
                    f"`{float(segment_summary['segment_speed_rmse_pct']):.6g}%`"
                ),
                (
                    "- Segment-speed spread: "
                    f"`{float(segment_summary['segment_speed_spread_mm_s']):.6g} mm/s`"
                ),
                *segment_lines,
                "",
                "## Plate Tracking Summary",
                "",
                (
                    "- Mean absolute plate error over the control-window evaluation interval: "
                    f"`{plate_summary.mean_abs_plate_error_C:.6g} C`"
                ),
                f"- Plate RMSE over the same interval: `{plate_summary.rmse_plate_error_C:.6g} C`",
                (
                    "- Fraction of samples within "
                    f"`+-{plate_summary.tolerance_C:.6g} C`: "
                    f"`{plate_summary.fraction_within_tolerance:.6g}`"
                ),
                "",
                "## Thermocouple-Equivalent Speeds",
                "",
                *tc_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _apply_cli_overrides(config: VelocityControlFileConfig, args: argparse.Namespace) -> VelocityControlFileConfig:
    target_updates = {}
    if args.target_front_speed_mm_s is not None:
        target_updates["target_front_speed_mm_s"] = args.target_front_speed_mm_s
    velocity_target = replace(config.velocity_target, **target_updates) if target_updates else config.velocity_target

    initial_updates = {}
    if args.initial_water_temperature_c is not None:
        initial_updates["initial_water_temperature_C"] = args.initial_water_temperature_c
    if args.initial_plate_temperature_c is not None:
        initial_updates["initial_plate_temperature_C"] = args.initial_plate_temperature_c
    initial_conditions = (
        replace(config.initial_conditions, **initial_updates)
        if initial_updates
        else config.initial_conditions
    )

    trajectory_updates = {}
    if args.t_after_fill_s is not None:
        trajectory_updates["horizon_s"] = args.t_after_fill_s
    if args.num_knots is not None:
        trajectory_updates["num_knots"] = args.num_knots
    if args.knot_time_schedule is not None:
        trajectory_updates["knot_time_schedule"] = args.knot_time_schedule
    custom_support = _parse_optional_float_tuple(args.knot_time_custom_support_tau)
    if custom_support is not None:
        trajectory_updates["knot_time_custom_support_tau"] = custom_support
    if args.theta_c is not None:
        trajectory_updates["theta_C"] = _parse_float_tuple(args.theta_c)
    manual_trajectory = (
        replace(config.manual_trajectory, **trajectory_updates)
        if trajectory_updates
        else config.manual_trajectory
    )

    run_updates = {}
    if args.run_name is not None:
        run_updates["run_name"] = args.run_name
    if args.simulation_profile is not None:
        run_updates["simulation_profile"] = args.simulation_profile
    return replace(
        config,
        velocity_target=velocity_target,
        initial_conditions=initial_conditions,
        manual_trajectory=manual_trajectory,
        **run_updates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one manual cryostage T_ref trajectory against a target front speed."
    )
    parser.add_argument("--velocity-control-config", type=Path, default=DEFAULT_VELOCITY_CONTROL_CONFIG_PATH)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--simulation-profile", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--target-front-speed-mm-s", type=float, default=None)
    parser.add_argument("--theta-c", default=None, help="Comma-separated manual T_ref knot temperatures in C.")
    parser.add_argument("--num-knots", type=int, default=None)
    parser.add_argument("--knot-time-schedule", default=None)
    parser.add_argument("--knot-time-custom-support-tau", default=None)
    parser.add_argument("--t-after-fill-s", type=float, default=None, help="Cooling-trajectory horizon after fill.")
    parser.add_argument("--initial-water-temperature-c", type=float, default=None)
    parser.add_argument("--initial-plate-temperature-c", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=velocity_control_results_dir())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _apply_cli_overrides(load_velocity_control_config(args.velocity_control_config), args)
    simulation_profile = load_simulation_profile(args.simulation_config, profile_name=config.simulation_profile)
    knot_times_s = build_knot_times_s(config.manual_trajectory)
    output_dir = Path(args.output_root) / f"n{int(config.manual_trajectory.num_knots)}" / "fine" / config.run_name
    effective_config_path = output_dir / "effective_config.toml"

    if args.dry_run_config:
        print(
            _render_effective_config(
                config=config,
                simulation_profile=simulation_profile,
                knot_times_s=knot_times_s,
                output_dir=output_dir,
            )
        )
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_effective_config(
        effective_config_path,
        config=config,
        simulation_profile=simulation_profile,
        knot_times_s=knot_times_s,
        output_dir=output_dir,
    )

    from code_simulation.simulation.cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
    from code_simulation.simulation.geometry import GeometryParams
    from code_simulation.simulation.open_loop_cascade import run_open_loop_case
    from code_simulation.simulation.solver import FreezeStopOptions, PhaseChangeParams, PrefillOptions, ThermalBCs

    T_ref_profile = PiecewiseLinearTemperatureProfile(
        knot_times_s=knot_times_s,
        knot_temperatures_C=config.manual_trajectory.theta_C,
    )
    time_s = _time_grid_s(config.manual_trajectory.horizon_s, simulation_profile.cryostage_dt_s)
    geom = GeometryParams(
        R_in=7.5e-3,
        t_wall=2.0e-3,
        t_base=0.0,
        H_fill=15.0e-3,
        H_total=17.0e-3,
    )
    phase = PhaseChangeParams(Tf=0.0, L_latent=334000.0, dT_mushy=0.5)
    freeze_stop = FreezeStopOptions(mode="fillable_region", extra_subcooling_C=0.0)
    prefill = PrefillOptions(mode="steady")
    bcs = ThermalBCs(T_room_C=5.75, h_top=H_OUT_W_M2K, h_side=H_OUT_W_M2K)
    prefix = config.run_name

    result = run_open_loop_case(
        time_s=time_s,
        T_ref_profile_C=T_ref_profile,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        out_dir=output_dir,
        prefix=prefix,
        T_plate0_C=config.initial_conditions.initial_plate_temperature_C,
        geom=geom,
        Nr=simulation_profile.Nr,
        Nz=simulation_profile.Nz,
        dt=simulation_profile.solver_dt_s,
        pre_cool_s=0.0,
        t_after_fill_s=config.manual_trajectory.horizon_s,
        write_every=simulation_profile.write_every_s,
        write_field_output=simulation_profile.write_field_output,
        write_probe_csv=True,
        show_progress=bool(simulation_profile.show_progress or args.show_progress),
        T_fill_C=config.initial_conditions.initial_water_temperature_C,
        T_plate_profile_time_origin="absolute",
        ambient_temperature_from_plate_C=simulation_profile.ambient_temperature_from_plate_C,
        bcs=bcs,
        phase=phase,
        prefill=prefill,
        freeze_stop=freeze_stop,
        front_definition_mode="isotherm_Tf",
        probe_z_m=PROBE_Z_M,
        probe_wall_inset_m=PROBE_WALL_INSET_M,
        Nz_front=simulation_profile.Nz_front,
        enable_front_curve=simulation_profile.enable_front_curve,
        Nr_front_curve=simulation_profile.Nr_front_curve,
        Nz_front_curve=simulation_profile.Nz_front_curve,
        stop_when_wall_frozen=False,
        use_tabulated_water_ice=simulation_profile.use_tabulated_water_ice,
    )

    timing = _read_key_value_csv(result.timing_path)
    cooling_start_time_s = float(timing.get("cooling_start_time_s", "0.0"))
    if not math.isfinite(cooling_start_time_s):
        cooling_start_time_s = 0.0
    summary, series = _tracking_summary(
        front_path=result.front_path,
        cooling_start_time_s=cooling_start_time_s,
        target_speed_mm_s=config.velocity_target.target_front_speed_mm_s,
        z_min_mm=config.velocity_target.control_z_min_mm,
        z_max_mm=config.velocity_target.control_z_max_mm,
    )
    segment_summary, segment_rows = _segment_speed_summary(
        series=series,
        target_speed_mm_s=config.velocity_target.target_front_speed_mm_s,
        z_min_mm=config.velocity_target.control_z_min_mm,
        z_max_mm=config.velocity_target.control_z_max_mm,
    )
    tc_rows = _thermocouple_speeds(
        probes_path=result.probes_path,
        cooling_start_time_s=cooling_start_time_s,
    )
    plate_summary, plate_series = summarize_plate_tracking(
        time_s=result.cryostage_time_s,
        T_ref_C=result.T_ref_C,
        T_plate_C=result.T_plate_C,
        tolerance_C=config.temperature_uncertainty.characterization_temperature_margin_C,
        evaluation_window_start_s=float(summary["t_at_control_z_min_s"]),
        evaluation_window_end_s=(
            None
            if not math.isfinite(float(summary["t_at_control_z_max_s"]))
            else float(summary["t_at_control_z_max_s"])
        ),
    )

    _write_single_row_csv(output_dir / "velocity_tracking_summary.csv", summary)
    _write_single_row_csv(output_dir / "segment_speed_summary.csv", segment_summary)
    _write_rows_csv(output_dir / "segment_interval_speeds.csv", segment_rows)
    _write_rows_csv(output_dir / "thermocouple_interval_speeds.csv", tc_rows)
    write_plate_tracking_summary_csv(output_dir / "plate_tracking_summary.csv", plate_summary)
    write_plate_tracking_timeseries_csv(output_dir / "T_ref_T_plate_timeseries.csv", plate_series)
    _plot_temperature_profiles(
        output_dir / "T_ref_and_T_plate_vs_time.png",
        time_s=result.cryostage_time_s,
        T_ref_C=result.T_ref_C,
        T_plate_C=result.T_plate_C,
    )
    _plot_front_position(
        output_dir / "front_position_vs_reference.png",
        series=series,
        z_min_mm=config.velocity_target.control_z_min_mm,
        z_max_mm=config.velocity_target.control_z_max_mm,
    )
    _plot_front_velocity(output_dir / "front_velocity_diagnostic_vs_time.png", series=series)
    _write_report(
        output_dir / "velocity_control_report.md",
        config=config,
        timing=timing,
        summary=summary,
        segment_summary=segment_summary,
        segment_rows=segment_rows,
        plate_summary=plate_summary,
        tc_rows=tc_rows,
    )

    print(f"Velocity-control evaluation written to {output_dir.resolve()}")
    print(f"  effective config: {effective_config_path.resolve()}")
    print(f"  tracking summary: {(output_dir / 'velocity_tracking_summary.csv').resolve()}")
    print(f"  segment summary : {(output_dir / 'segment_speed_summary.csv').resolve()}")
    print(f"  plate summary   : {(output_dir / 'plate_tracking_summary.csv').resolve()}")
    print(f"  report          : {(output_dir / 'velocity_control_report.md').resolve()}")


if __name__ == "__main__":
    main()
