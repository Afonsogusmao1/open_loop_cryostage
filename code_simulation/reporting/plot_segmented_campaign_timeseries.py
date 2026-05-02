#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.plotting import configure_matplotlib


COLOR_BY_SEED = {
    17: "#1f77b4",
    29: "#2ca02c",
    41: "#d62728",
    53: "#9467bd",
    67: "#8c564b",
}


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_numeric_columns(path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    data = {column: [] for column in columns}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in columns:
                data[column].append(_finite_float(row.get(column)))
    return {column: np.asarray(values, dtype=np.float64) for column, values in data.items()}


def _best_front_csv(run_dir: Path) -> Path:
    matches = sorted(path for path in (run_dir / "best").glob("*_front.csv") if "curve" not in path.name)
    if not matches:
        raise FileNotFoundError(f"no best/*_front.csv found in {run_dir}")
    return matches[0]


def _moving_average(values: np.ndarray, *, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    kernel = np.ones(int(window_samples), dtype=np.float64)
    numerator = np.convolve(clean, kernel, mode="same")
    denominator = np.convolve(finite.astype(np.float64), kernel, mode="same")
    out = numerator / np.maximum(denominator, 1.0)
    out[denominator <= 0.0] = math.nan
    return out


def _smooth_window_samples(*, window_samples: int = 30) -> int:
    samples = max(1, int(window_samples))
    return samples + 1 if samples % 2 == 0 else samples


def _window(summary: dict[str, str], *, margin_s: float = 90.0) -> tuple[float, float]:
    start = _finite_float(summary.get("t_at_control_z_min_s"))
    end = _finite_float(summary.get("t_at_control_z_max_s"))
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        return 0.0, math.nan
    return max(0.0, start - margin_s), end + margin_s


def _shade_control_window(ax: plt.Axes, summary: dict[str, str]) -> None:
    start = _finite_float(summary.get("t_at_control_z_min_s"))
    end = _finite_float(summary.get("t_at_control_z_max_s"))
    if math.isfinite(start) and math.isfinite(end) and end > start:
        ax.axvspan(start, end, color="0.88", alpha=0.55, linewidth=0, zorder=0)


def _apply_window(ax: plt.Axes, summary: dict[str, str], time_s: np.ndarray) -> None:
    start, end = _window(summary)
    if math.isfinite(end):
        finite_time = time_s[np.isfinite(time_s)]
        if finite_time.size:
            ax.set_xlim(start, min(end, float(np.max(finite_time))))


def _selected_best_per_target(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("status")) != "completed":
            continue
        target = _finite_float(row.get("target_front_speed_mm_s"))
        if math.isfinite(target):
            grouped.setdefault(target, []).append(row)

    selected: list[dict[str, str]] = []
    for target in sorted(grouped):
        candidates = grouped[target]
        selected.append(
            min(
                candidates,
                key=lambda row: (
                    int(row.get("accept_direct_and_segment_speed", "0")) * -1,
                    _finite_float(row.get("segment_speed_rmse_pct"), math.inf),
                    _finite_float(row.get("direct_speed_relative_error_pct"), math.inf),
                    _finite_float(row.get("tracking_rmse_mm"), math.inf),
                    int(float(row.get("seed", "999999"))),
                ),
            )
        )
    return selected


def _rows_for_target(rows: list[dict[str, str]], target: float) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if str(row.get("status")) == "completed"
        and abs(_finite_float(row.get("target_front_speed_mm_s")) - float(target)) <= 5.0e-7
    ]


def _plot_temperature(ax: plt.Axes, row: dict[str, str], *, title: str) -> np.ndarray:
    run_dir = Path(str(row["run_dir"]))
    summary = _read_rows(run_dir / "best_tracking_summary.csv")[0]
    cols = _read_numeric_columns(run_dir / "T_ref_T_plate_timeseries.csv", ("time_s", "T_ref_C", "T_plate_C"))
    _shade_control_window(ax, summary)
    ax.plot(cols["time_s"], cols["T_ref_C"], color="black", linewidth=1.2, label="T_ref")
    ax.plot(cols["time_s"], cols["T_plate_C"], color="#1f77b4", linewidth=1.1, label="T_plate")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    _apply_window(ax, summary, cols["time_s"])
    return cols["time_s"]


def _plot_velocity(ax: plt.Axes, row: dict[str, str], *, title: str) -> np.ndarray:
    run_dir = Path(str(row["run_dir"]))
    summary = _read_rows(run_dir / "best_tracking_summary.csv")[0]
    target = _finite_float(row.get("target_front_speed_mm_s"))
    seed = int(float(row.get("seed", "0")))
    cols = _read_numeric_columns(_best_front_csv(run_dir), ("time_s", "v_front_mm_per_s"))
    smooth = _moving_average(cols["v_front_mm_per_s"], window_samples=_smooth_window_samples())
    _shade_control_window(ax, summary)
    ax.plot(
        cols["time_s"],
        smooth,
        color=COLOR_BY_SEED.get(seed, "#2ca02c"),
        linewidth=1.2,
        label="v_front, 30-sample mean",
    )
    ax.axhline(target, color="black", linestyle="--", linewidth=1.1, label="target speed")
    ax.set_ylim(0.0, max(0.016, 1.45 * target))
    ax.set_ylabel("Front speed (mm/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    _apply_window(ax, summary, cols["time_s"])
    return cols["time_s"]


def plot_best_per_target(rows: list[dict[str, str]], output: Path) -> None:
    selected = _selected_best_per_target(rows)
    configure_matplotlib(plt)
    fig, axes = plt.subplots(len(selected), 2, figsize=(12.0, 2.45 * len(selected)), sharex=False)
    for idx, row in enumerate(selected):
        target = _finite_float(row.get("target_front_speed_mm_s"))
        seed = int(float(row.get("seed", "0")))
        accepted = "accepted" if str(row.get("accept_direct_and_segment_speed")) == "1" else "not accepted"
        title = f"target {target:.3f} mm/s, seed {seed} ({accepted})"
        _plot_temperature(axes[idx, 0], row, title=title)
        _plot_velocity(axes[idx, 1], row, title=title)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Segmented n8 BO: temperature and front-speed time series, best seed per target", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def plot_target_all_seeds(rows: list[dict[str, str]], *, target: float, output: Path) -> None:
    target_rows = sorted(_rows_for_target(rows, target), key=lambda row: int(float(row["seed"])))
    configure_matplotlib(plt)
    fig, axes = plt.subplots(len(target_rows), 2, figsize=(12.0, 2.6 * len(target_rows)), sharex=False)
    for idx, row in enumerate(target_rows):
        seed = int(float(row.get("seed", "0")))
        accepted = "accepted" if str(row.get("accept_direct_and_segment_speed")) == "1" else "not accepted"
        title = f"target {target:.3f} mm/s, seed {seed} ({accepted})"
        _plot_temperature(axes[idx, 0], row, title=title)
        _plot_velocity(axes[idx, 1], row, title=title)
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[0, 1].legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Segmented n8 BO: temperature and front-speed time series, target {target:.3f} mm/s", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot temperature and front-speed time series for a segmented n8 BO campaign.")
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--critical-target", type=float, default=0.013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.campaign_summary)
    plot_best_per_target(
        rows,
        args.output_dir / "temperature_and_front_velocity_best_per_target.png",
    )
    plot_target_all_seeds(
        rows,
        target=float(args.critical_target),
        output=args.output_dir
        / f"temperature_and_front_velocity_target_{f'{float(args.critical_target):.3f}'.replace('.', 'p')}.png",
    )
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
