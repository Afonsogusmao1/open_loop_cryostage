#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOR_BY_SEED = {
    17: "#1f77b4",
    29: "#2ca02c",
    41: "#d62728",
    53: "#9467bd",
    67: "#8c564b",
}
TARGET_TOL = 1.0e-12


def _finite_float(value: object) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _parse_schedules(raw: str) -> tuple[str, ...]:
    schedules = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not schedules:
        raise ValueError("at least one schedule must be provided")
    return schedules


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_rows(
    rows: list[dict[str, str]],
    *,
    schedules: tuple[str, ...],
    target_min_mm_s: float | None,
    target_max_mm_s: float | None,
) -> list[dict[str, str]]:
    schedule_set = set(schedules)
    completed: list[dict[str, str]] = []
    for row in rows:
        target = _finite_float(row.get("target_front_speed_mm_s"))
        achieved = _finite_float(row.get("achieved_direct_speed_mm_s"))
        if row.get("status") != "completed":
            continue
        if str(row.get("schedule", "")).strip() not in schedule_set:
            continue
        if not (math.isfinite(target) and math.isfinite(achieved)):
            continue
        if target_min_mm_s is not None and target < target_min_mm_s - TARGET_TOL:
            continue
        if target_max_mm_s is not None and target > target_max_mm_s + TARGET_TOL:
            continue
        completed.append(row)
    return completed


def plot_target_vs_achieved(
    *,
    rows: list[dict[str, str]],
    output: Path,
    title: str,
    schedules: tuple[str, ...],
    target_min_mm_s: float | None = None,
    target_max_mm_s: float | None = None,
) -> int:
    completed = _completed_rows(
        rows,
        schedules=schedules,
        target_min_mm_s=target_min_mm_s,
        target_max_mm_s=target_max_mm_s,
    )

    fig, axes = plt.subplots(
        1,
        len(schedules),
        figsize=(5.4 * len(schedules), 4.8),
        sharex=True,
        sharey=True,
    )
    if len(schedules) == 1:
        axes = [axes]

    if completed:
        x_values = [_finite_float(row["target_front_speed_mm_s"]) for row in completed]
        y_values = [_finite_float(row["achieved_direct_speed_mm_s"]) for row in completed]
        axis_max = max(max(x_values), max(y_values)) * 1.03
    else:
        axis_max = max(
            value
            for value in (target_max_mm_s, target_min_mm_s, 0.013)
            if value is not None
        )

    for ax, schedule in zip(axes, schedules, strict=True):
        schedule_rows = [
            row
            for row in completed
            if str(row.get("schedule", "")).strip() == schedule
        ]
        ax.plot(
            [0.0, axis_max],
            [0.0, axis_max],
            "--",
            color="0.5",
            linewidth=1.2,
            label="target = achieved",
        )
        for seed in sorted({int(row["seed"]) for row in schedule_rows}):
            seed_rows = [row for row in schedule_rows if int(row["seed"]) == seed]
            ax.scatter(
                [_finite_float(row["target_front_speed_mm_s"]) for row in seed_rows],
                [_finite_float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
                color=COLOR_BY_SEED.get(seed),
                s=38,
                label=f"seed {seed}",
            )
        ax.set_title(schedule)
        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.set_xlabel("Target speed (mm/s)")
        if ax is axes[0]:
            ax.set_ylabel("Achieved direct speed (mm/s)")
        if schedule_rows:
            ax.legend(loc="upper left")
        else:
            ax.text(0.03, 0.95, "No completed runs", transform=ax.transAxes, va="top")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return len(completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target vs achieved direct speed from a BO study_summary.csv."
    )
    parser.add_argument("--study-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--schedules", required=True)
    parser.add_argument("--target-min-mm-s", type=float)
    parser.add_argument("--target-max-mm-s", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.study_summary)
    schedules = _parse_schedules(args.schedules)
    plotted = plot_target_vs_achieved(
        rows=rows,
        output=args.output,
        title=str(args.title),
        schedules=schedules,
        target_min_mm_s=args.target_min_mm_s,
        target_max_mm_s=args.target_max_mm_s,
    )
    print(f"Plotted completed rows: {plotted}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
