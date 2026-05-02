#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from code_simulation.core.plotting import configure_matplotlib


COLOR_BY_SEED = {
    17: "#1f77b4",
    29: "#2ca02c",
    41: "#d62728",
    53: "#9467bd",
    67: "#8c564b",
}

PHASE_LABELS = {
    "seed": "theta0",
    "init": "deterministic init",
    "bo": "BO acquisition",
    "refinement": "local refinement",
}

PHASE_COLORS = {
    "seed": "#bdbdbd",
    "init": "#9ecae1",
    "bo": "#a1d99b",
    "refinement": "#fdae6b",
}

PHASE_ORDER = ("seed", "init", "bo", "refinement")
TARGET_TOL = 5.0e-7
DEFAULT_FORMATS = ("png", "pdf")


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(part.strip().lower().lstrip(".") for part in raw.split(",") if part.strip())
    if not formats:
        return DEFAULT_FORMATS
    unsupported = sorted(set(formats) - {"png", "pdf"})
    if unsupported:
        raise ValueError(f"unsupported figure format(s): {', '.join(unsupported)}")
    return formats


def _parse_targets(raw: str) -> tuple[float, ...]:
    targets = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not targets:
        raise ValueError("at least one example target must be provided")
    return targets


def _completed_campaign_rows(campaign_summary: Path) -> list[dict[str, str]]:
    rows = _read_rows(campaign_summary)
    return [
        row
        for row in rows
        if str(row.get("status", "")).strip().lower() == "completed"
        and math.isfinite(_finite_float(row.get("target_front_speed_mm_s")))
        and math.isfinite(_finite_float(row.get("achieved_direct_speed_mm_s")))
    ]


def _completed_fine_rows(fine_summary: Path) -> list[dict[str, str]]:
    rows = _read_rows(fine_summary)
    return [
        row
        for row in rows
        if str(row.get("fine_status", "")).strip().lower() == "completed"
        and math.isfinite(_finite_float(row.get("target_front_speed_mm_s")))
        and math.isfinite(_finite_float(row.get("fine_achieved_direct_speed_mm_s")))
    ]


def _target_key(row: dict[str, str]) -> float:
    return _finite_float(row.get("target_front_speed_mm_s"))


def _seed_key(row: dict[str, str]) -> int:
    return _safe_int(row.get("seed"))


def _resolve_path(path: Path, *, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return repo_root / path


def _infer_repo_root(article_dir: Path) -> Path:
    article_dir = article_dir.resolve()
    if article_dir.parent.name == "code_simulation":
        return article_dir.parent.parent
    for parent in article_dir.parents:
        if (parent / "code_simulation").is_dir():
            return parent
    return Path.cwd().resolve()


def _save_figure(
    fig: plt.Figure,
    *,
    figures_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    overwrite: bool,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / f"{stem}.{fmt}" for fmt in formats]
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"refusing to overwrite existing figure(s): {joined}")
    for path in paths:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return paths


def _axis_max_from_xy(x_values: list[float], y_values: list[float]) -> float:
    finite = [value for value in [*x_values, *y_values] if math.isfinite(value)]
    if not finite:
        return 0.014
    return max(0.014, max(finite) * 1.05)


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    midpoint = len(finite) // 2
    if len(finite) % 2:
        return float(finite[midpoint])
    return float(0.5 * (finite[midpoint - 1] + finite[midpoint]))


def _phase_ranges(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for phase in PHASE_ORDER:
        indices = [
            _safe_int(row.get("evaluation_index"))
            for row in rows
            if str(row.get("phase", "")).strip() == phase
        ]
        indices = [index for index in indices if index >= 0]
        if indices:
            ranges.append((phase, min(indices), max(indices)))
    return ranges


def _shade_phases(
    ax: plt.Axes,
    phase_ranges: list[tuple[str, int, int]],
    *,
    add_labels: bool,
) -> None:
    for phase, start, end in phase_ranges:
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=PHASE_COLORS.get(phase, "#cccccc"),
            alpha=0.17,
            linewidth=0,
            zorder=0,
        )
        if start > 1:
            ax.axvline(start - 0.5, color="0.55", linewidth=0.8, linestyle=":", zorder=1)
        if add_labels:
            ax.text(
                (start + end) / 2.0,
                0.98,
                PHASE_LABELS.get(phase, phase),
                ha="center",
                va="top",
                fontsize=7.5,
                transform=ax.get_xaxis_transform(),
            )


def _group_by_target(rows: list[dict[str, str]]) -> dict[float, list[dict[str, str]]]:
    grouped: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        target = _target_key(row)
        if math.isfinite(target):
            grouped[target].append(row)
    return dict(sorted(grouped.items()))


def plot_bo_target_vs_achieved(rows: list[dict[str, str]]) -> plt.Figure:
    configure_matplotlib(plt, figure_size=(5.8, 5.0))
    fig, ax = plt.subplots()
    targets = [_finite_float(row.get("target_front_speed_mm_s")) for row in rows]
    achieved = [_finite_float(row.get("achieved_direct_speed_mm_s")) for row in rows]
    axis_max = _axis_max_from_xy(targets, achieved)

    ax.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        "--",
        color="0.45",
        linewidth=1.2,
        label="target = achieved",
    )
    for seed in sorted({_seed_key(row) for row in rows}):
        seed_rows = [row for row in rows if _seed_key(row) == seed]
        ax.scatter(
            [_finite_float(row.get("target_front_speed_mm_s")) for row in seed_rows],
            [_finite_float(row.get("achieved_direct_speed_mm_s")) for row in seed_rows],
            color=COLOR_BY_SEED.get(seed, "0.35"),
            s=42,
            label=f"seed {seed}",
        )
    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Target direct speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title("n8 robust BO: target vs achieved direct speed")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_error_vs_target_seed_variability(rows: list[dict[str, str]]) -> plt.Figure:
    configure_matplotlib(plt, figure_size=(9.2, 4.2))
    fig, axes = plt.subplots(1, 2)
    ax_error, ax_spread = axes
    grouped = _group_by_target(rows)

    for seed in sorted({_seed_key(row) for row in rows}):
        seed_rows = sorted([row for row in rows if _seed_key(row) == seed], key=_target_key)
        ax_error.scatter(
            [_target_key(row) for row in seed_rows],
            [_finite_float(row.get("direct_speed_relative_error_pct")) for row in seed_rows],
            color=COLOR_BY_SEED.get(seed, "0.35"),
            s=34,
            label=f"seed {seed}",
        )

    targets = sorted(grouped)
    median_errors = [
        _median([_finite_float(row.get("direct_speed_relative_error_pct")) for row in grouped[target]])
        for target in targets
    ]
    spreads = [
        max(_finite_float(row.get("achieved_direct_speed_mm_s")) for row in grouped[target])
        - min(_finite_float(row.get("achieved_direct_speed_mm_s")) for row in grouped[target])
        for target in targets
    ]

    ax_error.plot(targets, median_errors, color="black", linewidth=1.6, marker="o", label="median")
    ax_error.axhline(1.0, color="0.35", linewidth=1.0, linestyle=":", label="1% tolerance")
    ax_error.set_xlabel("Target direct speed (mm/s)")
    ax_error.set_ylabel("Absolute speed error (%)")
    ax_error.set_title("Seed-level direct-speed error")
    ax_error.legend(loc="upper left", fontsize=8)

    ax_spread.bar(targets, spreads, width=0.00055, color="#6baed6", edgecolor="0.25")
    ax_spread.set_xlabel("Target direct speed (mm/s)")
    ax_spread.set_ylabel("Seed spread in achieved speed (mm/s)")
    ax_spread.set_title("Across-seed spread")

    fig.suptitle("n8 robust BO: error and seed variability", fontsize=12)
    fig.tight_layout()
    return fig


def _read_objective_rows(objective_csv: Path) -> list[dict[str, str]]:
    rows = _read_rows(objective_csv)
    clean_rows: list[dict[str, str]] = []
    for row in rows:
        target = _finite_float(row.get("target_front_speed_mm_s"))
        seed = _safe_int(row.get("seed"))
        objective = _finite_float(row.get("objective_value"))
        if math.isfinite(target) and seed > 0 and math.isfinite(objective):
            clean_rows.append(row)
    return clean_rows


def _best_series(rows: list[dict[str, str]]) -> list[float]:
    best_so_far = math.inf
    result: list[float] = []
    for row in rows:
        candidate = _finite_float(row.get("best_objective_value_after_eval"))
        if not math.isfinite(candidate):
            candidate = _finite_float(row.get("objective_value"))
        best_so_far = min(best_so_far, candidate)
        result.append(best_so_far)
    return result


def plot_objective_phases(rows: list[dict[str, str]]) -> plt.Figure:
    configure_matplotlib(plt)
    grouped = _group_by_target(rows)
    targets = sorted(grouped)
    ncols = 2
    nrows = math.ceil(len(targets) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols, 3.5 * nrows),
        sharex=True,
    )
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])

    for target, ax in zip(targets, axes_list, strict=False):
        target_rows = grouped[target]
        _shade_phases(ax, _phase_ranges(target_rows), add_labels=(target == targets[0]))
        for seed in sorted({_seed_key(row) for row in target_rows}):
            seed_rows = sorted(
                [row for row in target_rows if _seed_key(row) == seed],
                key=lambda row: _safe_int(row.get("evaluation_index")),
            )
            x = [_safe_int(row.get("evaluation_index")) for row in seed_rows]
            objective = [_finite_float(row.get("objective_value")) for row in seed_rows]
            best = _best_series(seed_rows)
            color = COLOR_BY_SEED.get(seed, "0.35")
            ax.scatter(x, objective, s=7, color=color, alpha=0.20, linewidths=0)
            ax.plot(x, best, color=color, linewidth=1.25, label=f"seed {seed}")
        ax.set_yscale("log")
        ax.set_title(f"target {target:.3f} mm/s")
        ax.set_ylabel("Objective value J")
        ax.grid(True, which="both", alpha=0.24)

    for ax in axes_list[len(targets) :]:
        ax.axis("off")
    for ax in axes_list[-ncols:]:
        ax.set_xlabel("Evaluation index")

    seed_handles = [
        plt.Line2D([0], [0], color=COLOR_BY_SEED[seed], linewidth=1.8, label=f"seed {seed}")
        for seed in sorted(COLOR_BY_SEED)
    ]
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[phase], alpha=0.30, label=PHASE_LABELS[phase])
        for phase in PHASE_ORDER
    ]
    axes_list[0].legend(handles=[*seed_handles, *phase_handles], loc="upper right", fontsize=8)
    fig.suptitle("n8 robust BO: objective convergence by phase", fontsize=13)
    fig.tight_layout()
    return fig


def plot_fine_target_vs_achieved(rows: list[dict[str, str]]) -> plt.Figure:
    configure_matplotlib(plt, figure_size=(5.8, 5.0))
    fig, ax = plt.subplots()
    targets = [_finite_float(row.get("target_front_speed_mm_s")) for row in rows]
    achieved = [_finite_float(row.get("fine_achieved_direct_speed_mm_s")) for row in rows]
    axis_max = _axis_max_from_xy(targets, achieved)

    ax.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        "--",
        color="0.45",
        linewidth=1.2,
        label="target = achieved",
    )
    for row in sorted(rows, key=_target_key):
        seed = _seed_key(row)
        ax.scatter(
            _target_key(row),
            _finite_float(row.get("fine_achieved_direct_speed_mm_s")),
            color=COLOR_BY_SEED.get(seed, "0.35"),
            s=48,
            label=f"seed {seed}",
        )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)
    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Target direct speed (mm/s)")
    ax.set_ylabel("Fine achieved direct speed (mm/s)")
    ax.set_title("Fine confirmations: target vs achieved direct speed")
    fig.tight_layout()
    return fig


def _row_for_target(rows: list[dict[str, str]], target: float) -> dict[str, str]:
    for row in rows:
        if abs(_target_key(row) - target) <= TARGET_TOL:
            return row
    available = ", ".join(f"{_target_key(row):.3f}" for row in sorted(rows, key=_target_key))
    raise ValueError(f"target {target:.3f} not found in fine summary; available: {available}")


def _fine_run_dir(row: dict[str, str], *, repo_root: Path) -> Path:
    raw = str(row.get("fine_run_dir", "")).strip()
    if not raw:
        raise ValueError(f"fine_run_dir missing for target {_target_key(row):.3f}")
    run_dir = _resolve_path(Path(raw), repo_root=repo_root)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    return run_dir


def _front_csv(row: dict[str, str], *, run_dir: Path) -> Path:
    fine_run_name = str(row.get("fine_run_name", "")).strip()
    if fine_run_name:
        candidate = run_dir / f"{fine_run_name}_front.csv"
        if candidate.exists():
            return candidate
    candidates = sorted(path for path in run_dir.glob("*_front.csv") if "curve" not in path.name)
    if not candidates:
        raise FileNotFoundError(f"no *_front.csv found in {run_dir}")
    return candidates[0]


def _tracking_summary(run_dir: Path) -> dict[str, str]:
    path = run_dir / "velocity_tracking_summary.csv"
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0]


def _plate_summary(run_dir: Path) -> dict[str, str]:
    path = run_dir / "plate_tracking_summary.csv"
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows[0]


def _window_from_tracking(summary: dict[str, str], *, margin_s: float = 120.0) -> tuple[float, float]:
    start = _finite_float(summary.get("t_at_control_z_min_s"))
    end = _finite_float(summary.get("t_at_control_z_max_s"))
    if not (math.isfinite(start) and math.isfinite(end)):
        return (0.0, math.nan)
    return (max(0.0, start - margin_s), end + margin_s)


def _window_from_plate(summary: dict[str, str], *, margin_s: float = 120.0) -> tuple[float, float]:
    start = _finite_float(summary.get("evaluation_window_start_s"))
    end = _finite_float(summary.get("evaluation_window_end_s"))
    if not (math.isfinite(start) and math.isfinite(end)):
        return (0.0, math.nan)
    return (max(0.0, start - margin_s), end + margin_s)


def _apply_xlim(ax: plt.Axes, xlim: tuple[float, float], max_time: float) -> None:
    start, end = xlim
    if math.isfinite(end):
        ax.set_xlim(start, min(end, max_time))


def _plot_window(ax: plt.Axes, start: float, end: float) -> None:
    if math.isfinite(start) and math.isfinite(end) and end > start:
        ax.axvspan(start, end, color="0.86", alpha=0.45, linewidth=0, zorder=0)


def _target_examples(
    fine_rows: list[dict[str, str]],
    example_targets: tuple[float, ...],
) -> list[dict[str, str]]:
    return [_row_for_target(fine_rows, target) for target in example_targets]


def plot_temperature_tracking_examples(
    example_rows: list[dict[str, str]],
    *,
    repo_root: Path,
) -> plt.Figure:
    configure_matplotlib(plt)
    fig, axes = plt.subplots(
        len(example_rows),
        1,
        figsize=(7.2, 2.7 * len(example_rows)),
        sharex=False,
    )
    axes_list = list(axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes]))
    for row, ax in zip(example_rows, axes_list, strict=True):
        target = _target_key(row)
        run_dir = _fine_run_dir(row, repo_root=repo_root)
        series = _read_rows(run_dir / "T_ref_T_plate_timeseries.csv")
        plate = _plate_summary(run_dir)
        window_start = _finite_float(plate.get("evaluation_window_start_s"))
        window_end = _finite_float(plate.get("evaluation_window_end_s"))
        x = [_finite_float(item.get("time_s")) for item in series]
        t_ref = [_finite_float(item.get("T_ref_C")) for item in series]
        t_plate = [_finite_float(item.get("T_plate_C")) for item in series]
        _plot_window(ax, window_start, window_end)
        ax.plot(x, t_ref, color="black", linewidth=1.4, label="T_ref")
        ax.plot(x, t_plate, color="#1f77b4", linewidth=1.1, label="T_plate")
        ax.set_title(f"target {target:.3f} mm/s, seed {_seed_key(row)}")
        ax.set_ylabel("Temperature (deg C)")
        _apply_xlim(ax, _window_from_plate(plate), max(x))
        ax.legend(loc="best", fontsize=8)
    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle("Reference and plate temperature tracking", fontsize=13)
    fig.tight_layout()
    return fig


def _reference_front(summary: dict[str, str], target: float) -> tuple[list[float], list[float]]:
    z_min = _finite_float(summary.get("control_z_min_mm"))
    z_max = _finite_float(summary.get("control_z_max_mm"))
    t_min = _finite_float(summary.get("t_at_control_z_min_s"))
    t_expected = _finite_float(summary.get("expected_t_at_control_z_max_s"))
    if not all(math.isfinite(value) for value in (z_min, z_max, t_min, t_expected)):
        t_expected = t_min + (z_max - z_min) / target
    return [t_min, t_expected], [z_min, z_max]


def plot_front_position_tracking_examples(
    example_rows: list[dict[str, str]],
    *,
    repo_root: Path,
) -> plt.Figure:
    configure_matplotlib(plt)
    fig, axes = plt.subplots(
        len(example_rows),
        1,
        figsize=(7.2, 2.7 * len(example_rows)),
        sharex=False,
    )
    axes_list = list(axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes]))
    for row, ax in zip(example_rows, axes_list, strict=True):
        target = _target_key(row)
        run_dir = _fine_run_dir(row, repo_root=repo_root)
        summary = _tracking_summary(run_dir)
        front = _read_rows(_front_csv(row, run_dir=run_dir))
        x = [_finite_float(item.get("time_s")) for item in front]
        z = [_finite_float(item.get("z_front_rel_mm")) for item in front]
        window_start = _finite_float(summary.get("t_at_control_z_min_s"))
        window_end = _finite_float(summary.get("t_at_control_z_max_s"))
        _plot_window(ax, window_start, window_end)
        ax.plot(x, z, color="#1f77b4", linewidth=1.1, label="simulated front")
        ref_x, ref_y = _reference_front(summary, target)
        ax.plot(ref_x, ref_y, "--", color="black", linewidth=1.2, label="target reference")
        ax.set_title(f"target {target:.3f} mm/s, seed {_seed_key(row)}")
        ax.set_ylabel("Front position z_f (mm)")
        _apply_xlim(ax, _window_from_tracking(summary), max(x))
        ax.legend(loc="best", fontsize=8)
    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle("Freezing-front position tracking", fontsize=13)
    fig.tight_layout()
    return fig


def _moving_average(values: list[float], *, window: int) -> list[float]:
    if window <= 1:
        return values
    half = window // 2
    result: list[float] = []
    for index in range(len(values)):
        lo = max(0, index - half)
        hi = min(len(values), index + half + 1)
        finite = [value for value in values[lo:hi] if math.isfinite(value)]
        result.append(float(sum(finite) / len(finite)) if finite else math.nan)
    return result


def _smooth_window_samples(times: list[float], *, window_s: float = 30.0) -> int:
    deltas = [
        times[index + 1] - times[index]
        for index in range(len(times) - 1)
        if math.isfinite(times[index]) and math.isfinite(times[index + 1]) and times[index + 1] > times[index]
    ]
    dt = _median(deltas)
    if not math.isfinite(dt) or dt <= 0.0:
        return 1
    samples = max(1, int(round(window_s / dt)))
    return samples + 1 if samples % 2 == 0 else samples


def plot_front_velocity_tracking_examples(
    example_rows: list[dict[str, str]],
    *,
    repo_root: Path,
) -> plt.Figure:
    configure_matplotlib(plt)
    fig, axes = plt.subplots(
        len(example_rows),
        1,
        figsize=(7.2, 2.7 * len(example_rows)),
        sharex=False,
    )
    axes_list = list(axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes]))
    max_target = max(_target_key(row) for row in example_rows)
    for row, ax in zip(example_rows, axes_list, strict=True):
        target = _target_key(row)
        run_dir = _fine_run_dir(row, repo_root=repo_root)
        summary = _tracking_summary(run_dir)
        front = _read_rows(_front_csv(row, run_dir=run_dir))
        x = [_finite_float(item.get("time_s")) for item in front]
        v = [_finite_float(item.get("v_front_mm_per_s")) for item in front]
        v_smooth = _moving_average(v, window=_smooth_window_samples(x))
        window_start = _finite_float(summary.get("t_at_control_z_min_s"))
        window_end = _finite_float(summary.get("t_at_control_z_max_s"))
        _plot_window(ax, window_start, window_end)
        ax.plot(x, v_smooth, color="#2ca02c", linewidth=1.2, label="simulated v_f, 30 s mean")
        ax.axhline(target, color="black", linewidth=1.1, linestyle="--", label="target speed")
        ax.set_ylim(0.0, max(0.016, 1.45 * max_target))
        ax.set_title(f"target {target:.3f} mm/s, seed {_seed_key(row)}")
        ax.set_ylabel("Front speed v_f (mm/s)")
        _apply_xlim(ax, _window_from_tracking(summary), max(x))
        ax.legend(loc="best", fontsize=8)
    axes_list[-1].set_xlabel("Time (s)")
    fig.suptitle("Freezing-front velocity tracking", fontsize=13)
    fig.tight_layout()
    return fig


def _relative(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _caption_entries() -> list[tuple[str, str]]:
    return [
        (
            "fig_bo_target_vs_achieved_n8",
            "Coarse n8 robust Bayesian optimization results: target direct speed against achieved direct speed for all five seeds. The dashed line marks perfect agreement.",
        ),
        (
            "fig_bo_objective_phases_n8",
            "Objective-function convergence during the n8 robust BO campaign. Raw objective evaluations are shown as transparent points, best-so-far curves as solid lines, and shaded bands mark theta0, deterministic initialization, BO acquisition, and local refinement.",
        ),
        (
            "fig_bo_error_vs_target_seed_variability",
            "Direct-speed error and across-seed spread after robust n8 BO. This is the main diagnostic for seed variability in the coarse optimization stage.",
        ),
        (
            "fig_fine_target_vs_achieved_best_per_target",
            "Fine-grid confirmation of the selected best seed for each target. These data validate whether the direct-speed result survives the full-process article simulation profile.",
        ),
        (
            "fig_temperature_tracking_examples",
            "Representative examples of reference and achieved plate temperature during fine confirmations. Shaded regions indicate the front-tracking evaluation window.",
        ),
        (
            "fig_front_position_tracking_examples",
            "Representative fine-grid freezing-front position trajectories compared with the linear target reference over the controlled depth interval.",
        ),
        (
            "fig_front_velocity_tracking_examples",
            "Representative fine-grid freezing-front velocity trajectories. The simulated velocity is shown as a 30 s moving mean to expose the control trend rather than numerical differentiation noise.",
        ),
    ]


def write_manifest(
    *,
    path: Path,
    repo_root: Path,
    campaign_dir: Path,
    fine_summary: Path,
    generated: dict[str, list[Path]],
    campaign_rows: list[dict[str, str]],
    fine_rows: list[dict[str, str]],
    example_targets: tuple[float, ...],
) -> None:
    target_count = len({_target_key(row) for row in campaign_rows})
    seed_count = len({_seed_key(row) for row in campaign_rows})
    fine_target_count = len({_target_key(row) for row in fine_rows})
    lines = [
        "# Article figure manifest",
        "",
        f"Generated on: {date.today().isoformat()}",
        "",
        "## Sources",
        "",
        f"- BO campaign: `{_relative(campaign_dir, root=repo_root)}`",
        f"- Campaign summary: `{_relative(campaign_dir / 'campaign_summary.csv', root=repo_root)}`",
        f"- Objective evaluations: `{_relative(campaign_dir / 'objective_by_evaluation.csv', root=repo_root)}`",
        f"- Fine summary: `{_relative(fine_summary, root=repo_root)}`",
        "",
        "## Counts",
        "",
        f"- Completed coarse BO rows: `{len(campaign_rows)}`",
        f"- Coarse targets: `{target_count}`",
        f"- Coarse seeds: `{seed_count}`",
        f"- Completed fine representative rows: `{len(fine_rows)}`",
        f"- Fine representative targets: `{fine_target_count}`",
        f"- Example targets for time-series figures: `{', '.join(f'{target:.3f}' for target in example_targets)} mm/s`",
        "",
        "## Figures and captions",
        "",
    ]
    captions = dict(_caption_entries())
    for stem, paths in generated.items():
        lines.append(f"### `{stem}`")
        lines.append("")
        for figure_path in paths:
            lines.append(f"- File: `{_relative(figure_path, root=repo_root)}`")
        lines.append(f"- Caption: {captions[stem]}")
        lines.append("")

    lines.extend(
        [
            "## Notes for Oliveira",
            "",
            "- The objective plot directly addresses whether BO was still improving when each phase ended.",
            "- The seed-variability figure separates direct-speed error from across-seed achieved-speed spread.",
            "- The fine figures are representative best-per-target confirmations, not yet the full all-seed fine robustness set.",
            "- The Word draft was not edited in this pass.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _list_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_dir())


def _list_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file())


def write_cleanup_inventory(
    *,
    path: Path,
    repo_root: Path,
    article_dir: Path,
    campaign_dir: Path,
    fine_rows: list[dict[str, str]],
) -> None:
    n8_diag_root = repo_root / "code_simulation/results/active/bo_velocity_control/n8/diagnostico"
    n8_fine_root = repo_root / "code_simulation/results/active/bo_velocity_control/n8/fine"
    n3_root = repo_root / "code_simulation/results/active/bo_velocity_control/n3"
    original_n8_root = repo_root / "code_simulation/results/active/n8"

    selected_fine_dirs = {
        _fine_run_dir(row, repo_root=repo_root).resolve()
        for row in fine_rows
        if str(row.get("fine_run_dir", "")).strip()
    }
    diag_dirs = _list_dirs(n8_diag_root)
    fine_dirs = _list_dirs(n8_fine_root)
    article_files = _list_files(article_dir)

    article_ready = [
        campaign_dir,
        campaign_dir / "campaign_summary.csv",
        campaign_dir / "objective_by_evaluation.csv",
        article_dir / "figures",
    ]
    article_ready.extend(sorted(selected_fine_dirs))

    archive_candidates = [
        path
        for path in diag_dirs
        if path.resolve() != campaign_dir.resolve()
        and any(token in path.name for token in ("probe", "unanchored", "n3_anchor"))
    ]

    lines = [
        "# Open_loop cleanup inventory",
        "",
        f"Generated on: {date.today().isoformat()}",
        "",
        "No files were moved or deleted. This is a review manifest only.",
        "",
        "## Article-ready / keep active",
        "",
    ]
    for item in article_ready:
        lines.append(f"- `{_relative(item, root=repo_root)}`")

    lines.extend(["", "## Existing Article_drafts files", ""])
    for item in article_files:
        lines.append(f"- `{_relative(item, root=repo_root)}`")

    lines.extend(["", "## Old diagnostic or probe folders to review for archiving", ""])
    if archive_candidates:
        for item in archive_candidates:
            lines.append(f"- `{_relative(item, root=repo_root)}`")
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Other n8 diagnostic folders", ""])
    other_diag_dirs = [
        item
        for item in diag_dirs
        if item not in archive_candidates and item.resolve() != campaign_dir.resolve()
    ]
    if other_diag_dirs:
        for item in other_diag_dirs:
            lines.append(f"- `{_relative(item, root=repo_root)}`")
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Fine runs currently present", ""])
    if fine_dirs:
        for item in fine_dirs:
            marker = "article representative" if item.resolve() in selected_fine_dirs else "not in current fine summary"
            lines.append(f"- `{_relative(item, root=repo_root)}` - {marker}")
    else:
        lines.append("- None detected.")

    lines.extend(
        [
            "",
            "## Unsafe to delete without explicit review",
            "",
            f"- `{_relative(article_dir, root=repo_root)}`",
            f"- `{_relative(campaign_dir, root=repo_root)}`",
            f"- `{_relative(n8_fine_root, root=repo_root)}`",
            f"- `{_relative(n3_root, root=repo_root)}`",
            f"- `{_relative(original_n8_root, root=repo_root)}`",
            "- Any folder containing unpublished logs, summaries, HDF5/XDMF fields, or article figures.",
            "",
            "## Suggested next cleanup action",
            "",
            "- Review this manifest with the generated figures open.",
            "- Archive old probe/diagnostic folders only after confirming they are not cited by the article or SI.",
            "- Prefer moving to an archive folder over deleting.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_article_figures(
    *,
    campaign_dir: Path,
    fine_summary: Path,
    article_dir: Path,
    example_targets: tuple[float, ...],
    formats: tuple[str, ...],
    overwrite: bool,
) -> dict[str, list[Path]]:
    campaign_dir = campaign_dir.resolve()
    fine_summary = fine_summary.resolve()
    article_dir = article_dir.resolve()
    repo_root = _infer_repo_root(article_dir)
    figures_dir = article_dir / "figures"

    campaign_rows = _completed_campaign_rows(campaign_dir / "campaign_summary.csv")
    fine_rows = _completed_fine_rows(fine_summary)
    objective_rows = _read_objective_rows(campaign_dir / "objective_by_evaluation.csv")
    example_rows = _target_examples(fine_rows, example_targets)

    generated: dict[str, list[Path]] = {}
    figure_builders = [
        ("fig_bo_target_vs_achieved_n8", lambda: plot_bo_target_vs_achieved(campaign_rows)),
        ("fig_bo_objective_phases_n8", lambda: plot_objective_phases(objective_rows)),
        ("fig_bo_error_vs_target_seed_variability", lambda: plot_error_vs_target_seed_variability(campaign_rows)),
        ("fig_fine_target_vs_achieved_best_per_target", lambda: plot_fine_target_vs_achieved(fine_rows)),
        (
            "fig_temperature_tracking_examples",
            lambda: plot_temperature_tracking_examples(example_rows, repo_root=repo_root),
        ),
        (
            "fig_front_position_tracking_examples",
            lambda: plot_front_position_tracking_examples(example_rows, repo_root=repo_root),
        ),
        (
            "fig_front_velocity_tracking_examples",
            lambda: plot_front_velocity_tracking_examples(example_rows, repo_root=repo_root),
        ),
    ]

    for stem, builder in figure_builders:
        generated[stem] = _save_figure(
            builder(),
            figures_dir=figures_dir,
            stem=stem,
            formats=formats,
            overwrite=overwrite,
        )

    write_manifest(
        path=figures_dir / "figure_manifest.md",
        repo_root=repo_root,
        campaign_dir=campaign_dir,
        fine_summary=fine_summary,
        generated=generated,
        campaign_rows=campaign_rows,
        fine_rows=fine_rows,
        example_targets=example_targets,
    )
    write_cleanup_inventory(
        path=article_dir / "open_loop_cleanup_inventory.md",
        repo_root=repo_root,
        article_dir=article_dir,
        campaign_dir=campaign_dir,
        fine_rows=fine_rows,
    )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate article-ready figures from n8 BO/fine CSV outputs.")
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--fine-summary", type=Path, required=True)
    parser.add_argument("--article-dir", type=Path, required=True)
    parser.add_argument("--example-targets", default="0.007,0.010,0.013")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formats = _parse_formats(args.formats)
    example_targets = _parse_targets(args.example_targets)
    generated = generate_article_figures(
        campaign_dir=args.campaign_dir,
        fine_summary=args.fine_summary,
        article_dir=args.article_dir,
        example_targets=example_targets,
        formats=formats,
        overwrite=args.overwrite,
    )
    print("Generated article figures:")
    for stem, paths in generated.items():
        for path in paths:
            print(f"- {stem}: {path.resolve()}")
    print(f"Manifest: {(args.article_dir / 'figures' / 'figure_manifest.md').resolve()}")
    print(f"Cleanup inventory: {(args.article_dir / 'open_loop_cleanup_inventory.md').resolve()}")


if __name__ == "__main__":
    main()
