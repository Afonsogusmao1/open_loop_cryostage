from __future__ import annotations

"""Generate active front-speed plots from rho(T) calibrated simulations."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import (
    ACTIVE_TARGETS_C,
    ACTIVE_VELOCITY_STUDY_DIR,
    WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    discover_simulation_outputs,
)


THRESHOLD_C = 0.0
FRONT_VELOCITY_MOVING_AVERAGE_WINDOW_S = 30.0
SMOOTHED_FRONT_POSITION_FRACTION_CUTOFF = 0.99
PROBE_Z_MM = (3.0, 6.2, 11.0)
PROBE_COLUMNS = ("T_z3p0mm_C", "T_z6p2mm_C", "T_z11p0mm_C")
COLORS = {
    -5.0: "#cc79a7",
    -10.0: "#2563eb",
    -15.0: "#059669",
    -20.0: "#d97706",
    -21.0: "#111827",
}
STALE_OUTPUT_NAMES = (
    "front_velocity_raw_vs_time.png",
    "front_velocity_no_moving_average_vs_time.png",
    "front_velocity_vs_time.png",
    "front_position_vs_time.png",
    "tc_segment_speeds_by_temperature.png",
    "speed_summary.csv",
    "discussion.md",
    "calibrated_front_position_vs_time.png",
    "calibrated_front_velocity_no_moving_average_vs_time.png",
    "calibrated_front_velocity_raw_vs_time.png",
    "calibrated_front_velocity_vs_time.png",
    "calibrated_speed_summary.csv",
    "calibrated_tc_main_speed_by_temperature.png",
    "calibrated_tc_segment_speeds_by_temperature.png",
    "calibrated_thermocouple_velocity_vs_time.png",
)


@dataclass(frozen=True)
class SimulationCase:
    target_C: float
    label: str
    directory: Path
    front_csv: Path
    probes_csv: Path


@dataclass(frozen=True)
class ThermocoupleSummary:
    target_C: float
    crossing_3p0mm_s: float
    crossing_6p2mm_s: float
    crossing_11p0mm_s: float
    v_3p0_to_6p2_mm_s: float
    v_6p2_to_11p0_mm_s: float
    v_3p0_to_11p0_mm_s: float


def _as_float(raw: str | None) -> float:
    if raw is None:
        return math.nan
    text = str(raw).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _nan_to_str(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.9f}"


def _read_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        data = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(_as_float(row.get(column)))
    return {column: np.asarray(values, dtype=np.float64) for column, values in data.items()}


def _discover_cases(simulations_dir: Path, *, expected_targets: tuple[float, ...]) -> list[SimulationCase]:
    outputs = discover_simulation_outputs(simulations_dir, expected_targets=expected_targets)
    cases = [
        SimulationCase(
            target_C=target_C,
            label=f"{target_C:g} C",
            directory=output.directory,
            front_csv=output.front_csv,
            probes_csv=output.probes_csv,
        )
        for target_C, output in outputs.items()
    ]
    return list(sorted(cases, key=lambda item: item.target_C, reverse=True))


def _first_downward_crossing_time_s(time_s: np.ndarray, temperature_C: np.ndarray, threshold_C: float) -> float:
    valid = np.isfinite(time_s) & np.isfinite(temperature_C)
    t = time_s[valid]
    y = temperature_C[valid]
    if t.size < 2:
        return math.nan

    above_indices = np.flatnonzero(y > threshold_C)
    if above_indices.size == 0:
        return math.nan
    start = int(above_indices[0])
    if start >= t.size - 1:
        return math.nan

    y0 = y[start:-1]
    y1 = y[start + 1 :]
    transition_indices = np.flatnonzero((y0 > threshold_C) & (y1 <= threshold_C))
    if transition_indices.size == 0:
        exact_indices = np.flatnonzero(y[start:] == threshold_C)
        if exact_indices.size == 0:
            return math.nan
        return float(t[start + int(exact_indices[0])])

    idx = start + int(transition_indices[0])
    t0 = float(t[idx])
    t1 = float(t[idx + 1])
    temp0 = float(y[idx])
    temp1 = float(y[idx + 1])
    if temp1 == temp0:
        return t1
    fraction = (threshold_C - temp0) / (temp1 - temp0)
    return float(t0 + fraction * (t1 - t0))


def _segment_speed_mm_s(z0_mm: float, z1_mm: float, t0_s: float, t1_s: float) -> float:
    if not (math.isfinite(t0_s) and math.isfinite(t1_s)) or t1_s <= t0_s:
        return math.nan
    return float((z1_mm - z0_mm) / (t1_s - t0_s))


def _centered_moving_average(time_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    time_s = np.asarray(time_s, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    valid_values = np.where(np.isfinite(values), values, 0.0)
    valid_counts = np.isfinite(values).astype(np.float64)
    cumulative_values = np.concatenate(([0.0], np.cumsum(valid_values)))
    cumulative_counts = np.concatenate(([0.0], np.cumsum(valid_counts)))
    half_window_s = 0.5 * float(window_s)
    averaged = np.full_like(values, np.nan, dtype=np.float64)
    for idx, t_s in enumerate(time_s):
        left = int(np.searchsorted(time_s, t_s - half_window_s, side="left"))
        right = int(np.searchsorted(time_s, t_s + half_window_s, side="right"))
        count = cumulative_counts[right] - cumulative_counts[left]
        if count > 0.0:
            averaged[idx] = (cumulative_values[right] - cumulative_values[left]) / count
    return averaged


def _force_nonnegative_axes(ax) -> None:
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)


def _remove_stale_outputs(output_dir: Path) -> None:
    for name in STALE_OUTPUT_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _thermocouple_summary(case: SimulationCase) -> ThermocoupleSummary:
    cols = _read_columns(case.probes_csv, ("time_since_fill_s", *PROBE_COLUMNS))
    time_since_fill_s = cols["time_since_fill_s"]
    post_fill = np.isfinite(time_since_fill_s) & (time_since_fill_s >= 0.0)
    crossings = tuple(
        _first_downward_crossing_time_s(time_since_fill_s[post_fill], cols[column][post_fill], THRESHOLD_C)
        for column in PROBE_COLUMNS
    )
    return ThermocoupleSummary(
        target_C=case.target_C,
        crossing_3p0mm_s=crossings[0],
        crossing_6p2mm_s=crossings[1],
        crossing_11p0mm_s=crossings[2],
        v_3p0_to_6p2_mm_s=_segment_speed_mm_s(PROBE_Z_MM[0], PROBE_Z_MM[1], crossings[0], crossings[1]),
        v_6p2_to_11p0_mm_s=_segment_speed_mm_s(PROBE_Z_MM[1], PROBE_Z_MM[2], crossings[1], crossings[2]),
        v_3p0_to_11p0_mm_s=_segment_speed_mm_s(PROBE_Z_MM[0], PROBE_Z_MM[2], crossings[0], crossings[2]),
    )


def _plot_front_velocity_raw(cases: list[SimulationCase], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for case in cases:
        cols = _read_columns(case.front_csv, ("time_since_fill_s", "v_front_mm_per_s"))
        time_s = cols["time_since_fill_s"]
        speed = cols["v_front_mm_per_s"]
        mask = np.isfinite(time_s) & (time_s >= 0.0) & np.isfinite(speed)
        ax.plot(time_s[mask], speed[mask], lw=0.9, alpha=0.85, color=COLORS.get(case.target_C), label=case.label)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_title("Raw direct front velocity from constant-plate simulations")
    ax.set_xlabel("Time since fill (s)")
    ax.set_ylabel("Front velocity, finite difference (mm/s)")
    ax.legend(title="Plate temperature")
    ax.grid(True, alpha=0.25)
    _force_nonnegative_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_front_velocity_smoothed(cases: list[SimulationCase], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for case in cases:
        cols = _read_columns(case.front_csv, ("time_since_fill_s", "z_front_mm", "v_front_mm_per_s"))
        time_s = cols["time_since_fill_s"]
        position = cols["z_front_mm"]
        speed = cols["v_front_mm_per_s"]
        finite_position = position[np.isfinite(position)]
        if finite_position.size == 0:
            continue
        position_cutoff_mm = SMOOTHED_FRONT_POSITION_FRACTION_CUTOFF * float(np.nanmax(finite_position))
        mask = (
            np.isfinite(time_s)
            & (time_s >= 0.0)
            & np.isfinite(position)
            & np.isfinite(speed)
            & (position < position_cutoff_mm)
        )
        plot_time_s = time_s[mask]
        smoothed_speed = _centered_moving_average(plot_time_s, speed[mask], FRONT_VELOCITY_MOVING_AVERAGE_WINDOW_S)
        ax.plot(plot_time_s, smoothed_speed, lw=1.8, color=COLORS.get(case.target_C), label=case.label)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_title("Smoothed direct front velocity from constant-plate simulations")
    ax.set_xlabel("Time since fill (s)")
    ax.set_ylabel("Front velocity, 30 s moving average (mm/s)")
    ax.legend(title="Plate temperature")
    ax.grid(True, alpha=0.25)
    _force_nonnegative_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_front_position(cases: list[SimulationCase], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for case in cases:
        cols = _read_columns(case.front_csv, ("time_since_fill_s", "z_front_mm"))
        time_s = cols["time_since_fill_s"]
        position = cols["z_front_mm"]
        mask = np.isfinite(time_s) & (time_s >= 0.0) & np.isfinite(position)
        ax.plot(time_s[mask], position[mask], lw=1.6, color=COLORS.get(case.target_C), label=case.label)
    for probe_z_mm in PROBE_Z_MM:
        ax.axhline(probe_z_mm, color="0.25", lw=0.8, ls=":")
        ax.text(0.99, probe_z_mm, f"{probe_z_mm:g} mm", transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_title("Front position from constant-plate simulations")
    ax.set_xlabel("Time since fill (s)")
    ax.set_ylabel("Centerline front position (mm)")
    ax.legend(title="Plate temperature")
    ax.grid(True, alpha=0.25)
    _force_nonnegative_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_thermocouple_segment_speeds_by_temperature(
    cases: list[SimulationCase],
    summaries: dict[float, ThermocoupleSummary],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(3, dtype=np.float64)
    series = (
        ("3.0 to 6.2 mm", "v_3p0_to_6p2_mm_s"),
        ("6.2 to 11.0 mm", "v_6p2_to_11p0_mm_s"),
        ("3.0 to 11.0 mm", "v_3p0_to_11p0_mm_s"),
    )
    offsets = np.linspace(-0.26, 0.26, len(cases))
    for case, offset in zip(cases, offsets, strict=True):
        summary = summaries[case.target_C]
        speeds = np.asarray([getattr(summary, attr_name) for _, attr_name in series], dtype=np.float64)
        mask = np.isfinite(speeds)
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask] + float(offset),
            speeds[mask],
            s=62,
            color=COLORS.get(case.target_C),
            edgecolors="white",
            linewidths=0.6,
            label=case.label,
            zorder=3,
        )
    ax.set_title("Thermocouple interval speeds")
    ax.set_xlabel("Thermocouple interval")
    ax.set_ylabel("Velocity from 0 C crossings (mm/s)")
    ax.set_xticks(x, [label for label, _ in series])
    ax.legend(title="Plate temperature", ncols=2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xlim(-0.55, len(series) - 0.45)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_summary_csv(
    cases: list[SimulationCase],
    summaries: dict[float, ThermocoupleSummary],
    path: Path,
    *,
    simulations_dir: Path,
) -> None:
    fieldnames = [
        "target_C",
        "front_csv",
        "probes_csv",
        "threshold_C",
        "crossing_3p0mm_s",
        "crossing_6p2mm_s",
        "crossing_11p0mm_s",
        "v_3p0_to_6p2_mm_s",
        "v_6p2_to_11p0_mm_s",
        "v_3p0_to_11p0_mm_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            summary = summaries[case.target_C]
            writer.writerow(
                {
                    "target_C": f"{case.target_C:.6f}",
                    "front_csv": str(case.front_csv.relative_to(simulations_dir)),
                    "probes_csv": str(case.probes_csv.relative_to(simulations_dir)),
                    "threshold_C": f"{THRESHOLD_C:.6f}",
                    "crossing_3p0mm_s": _nan_to_str(summary.crossing_3p0mm_s),
                    "crossing_6p2mm_s": _nan_to_str(summary.crossing_6p2mm_s),
                    "crossing_11p0mm_s": _nan_to_str(summary.crossing_11p0mm_s),
                    "v_3p0_to_6p2_mm_s": _nan_to_str(summary.v_3p0_to_6p2_mm_s),
                    "v_6p2_to_11p0_mm_s": _nan_to_str(summary.v_6p2_to_11p0_mm_s),
                    "v_3p0_to_11p0_mm_s": _nan_to_str(summary.v_3p0_to_11p0_mm_s),
                }
            )


def _write_readme(output_dir: Path, *, simulations_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    text = f"""# Velocity Study

This folder holds the active constant-plate front-velocity study regenerated
from the rho(T) calibrated simulations in:

- `{simulations_dir}`

The plots are generated from the existing `*_front.csv` and `*_probes.csv`
files. No new freezing simulations are run when refreshing this study.

## Main Outputs

- `front_velocity_raw_vs_time.png`
- `front_velocity_no_moving_average_vs_time.png`
- `front_velocity_vs_time.png`
- `front_position_vs_time.png`
- `tc_segment_speeds_by_temperature.png`
- `speed_summary.csv`

## Regeneration Command

```bash
cd /home/fenics/shared/Open_loop
python -m code_simulation.verification.generate_calibrated_front_speed_plots \\
  --simulations-dir /home/fenics/shared/Open_loop/data/simulations_calibrated/with_rho_temperature_dependent \\
  --output-dir /home/fenics/shared/Open_loop/code_simulation/results/active/velocity_study
```
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def generate_calibrated_front_speed_plots(
    *,
    simulations_dir: Path = WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    output_dir: Path = ACTIVE_VELOCITY_STUDY_DIR,
    expected_targets: tuple[float, ...] = ACTIVE_TARGETS_C,
) -> tuple[Path, ...]:
    simulations_dir = Path(simulations_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)

    cases = _discover_cases(simulations_dir, expected_targets=expected_targets)
    summaries = {case.target_C: _thermocouple_summary(case) for case in cases}

    written_paths = [
        output_dir / "front_velocity_raw_vs_time.png",
        output_dir / "front_velocity_no_moving_average_vs_time.png",
        output_dir / "front_velocity_vs_time.png",
        output_dir / "front_position_vs_time.png",
        output_dir / "tc_segment_speeds_by_temperature.png",
        output_dir / "speed_summary.csv",
    ]
    _plot_front_velocity_raw(cases, written_paths[0])
    _plot_front_velocity_raw(cases, written_paths[1])
    _plot_front_velocity_smoothed(cases, written_paths[2])
    _plot_front_position(cases, written_paths[3])
    _plot_thermocouple_segment_speeds_by_temperature(cases, summaries, written_paths[4])
    _write_summary_csv(cases, summaries, written_paths[5], simulations_dir=simulations_dir)
    readme_path = _write_readme(output_dir, simulations_dir=simulations_dir)

    print(f"[velocity] wrote analysis to {output_dir}")
    return tuple((*written_paths, readme_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate direct-front and thermocouple-equivalent velocity plots from rho(T) calibrated simulations."
    )
    parser.add_argument("--simulations-dir", type=Path, default=WITH_RHO_TEMPERATURE_DEPENDENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=ACTIVE_VELOCITY_STUDY_DIR)
    parser.add_argument(
        "--targets-c",
        default=",".join(f"{value:g}" for value in ACTIVE_TARGETS_C),
        help="Comma-separated targets expected in the simulations directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = tuple(float(part.strip()) for part in str(args.targets_c).split(",") if part.strip())
    generate_calibrated_front_speed_plots(
        simulations_dir=Path(args.simulations_dir),
        output_dir=Path(args.output_dir),
        expected_targets=values,
    )


if __name__ == "__main__":
    main()
