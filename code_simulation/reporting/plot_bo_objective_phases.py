#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _target_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def _phase_category(raw_phase: str) -> str:
    phase = str(raw_phase).strip().lower()
    if phase == "seed":
        return "seed"
    if phase in {"init_local", "init_global", "random"}:
        return "init"
    if phase == "bayes":
        return "bo"
    if phase == "refine_local":
        return "refinement"
    return phase or "unknown"


def _read_history(run_dir: Path, *, target: float, seed: int) -> list[dict[str, object]]:
    history_path = run_dir / "bo_history.csv"
    if not history_path.exists():
        history_path = run_dir / "evaluation_history.csv"
    if not history_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with history_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            objective = _finite_float(row.get("objective_value"))
            best_after = _finite_float(row.get("best_objective_value_after_eval"))
            if not math.isfinite(objective):
                continue
            eval_index = int(float(row["evaluation_index"]))
            phase = _phase_category(str(row.get("phase", "")))
            rows.append(
                {
                    "target_front_speed_mm_s": float(target),
                    "seed": int(seed),
                    "run_name": run_dir.name,
                    "evaluation_index": eval_index,
                    "phase": phase,
                    "raw_phase": str(row.get("phase", "")),
                    "objective_value": objective,
                    "best_objective_value_after_eval": best_after,
                }
            )
    rows.sort(key=lambda item: int(item["evaluation_index"]))
    return rows


def collect_evaluations(runs_root: Path, run_re: re.Pattern[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        match = run_re.match(run_dir.name)
        if match is None:
            continue
        target = _target_from_tag(match.group("tag"))
        seed = int(match.group("seed"))
        rows.extend(_read_history(run_dir, target=target, seed=seed))
    return rows


def _write_evaluation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "run_name",
        "target_front_speed_mm_s",
        "seed",
        "evaluation_index",
        "phase",
        "raw_phase",
        "objective_value",
        "best_objective_value_after_eval",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _phase_best(rows: list[dict[str, object]]) -> dict[str, float]:
    best_by_phase: dict[str, float] = {}
    best_so_far = math.inf
    for row in sorted(rows, key=lambda item: int(item["evaluation_index"])):
        objective = float(row["objective_value"])
        if objective < best_so_far:
            best_so_far = objective
        phase = str(row["phase"])
        if phase in PHASE_ORDER:
            best_by_phase[phase] = best_so_far
    return best_by_phase


def _pct_improvement(before: float, after: float) -> float:
    if not (math.isfinite(before) and math.isfinite(after) and before > 0.0):
        return math.nan
    return float(100.0 * (before - after) / before)


def _write_phase_summary(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_name"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for run_name, run_rows in sorted(grouped.items()):
        target = float(run_rows[0]["target_front_speed_mm_s"])
        seed = int(run_rows[0]["seed"])
        phase_best = _phase_best(run_rows)
        seed_best = phase_best.get("seed", math.nan)
        init_best = phase_best.get("init", seed_best)
        bo_best = phase_best.get("bo", init_best)
        refinement_best = phase_best.get("refinement", bo_best)
        summary_rows.append(
            {
                "run_name": run_name,
                "target_front_speed_mm_s": target,
                "seed": seed,
                "n_evaluations": len(run_rows),
                "best_after_theta0": seed_best,
                "best_after_init": init_best,
                "best_after_bo": bo_best,
                "best_after_refinement": refinement_best,
                "theta0_to_init_improvement_pct": _pct_improvement(seed_best, init_best),
                "init_to_bo_improvement_pct": _pct_improvement(init_best, bo_best),
                "bo_to_refinement_improvement_pct": _pct_improvement(bo_best, refinement_best),
                "theta0_to_final_improvement_pct": _pct_improvement(seed_best, refinement_best),
            }
        )

    fieldnames = (
        "run_name",
        "target_front_speed_mm_s",
        "seed",
        "n_evaluations",
        "best_after_theta0",
        "best_after_init",
        "best_after_bo",
        "best_after_refinement",
        "theta0_to_init_improvement_pct",
        "init_to_bo_improvement_pct",
        "bo_to_refinement_improvement_pct",
        "theta0_to_final_improvement_pct",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def _phase_ranges(rows: list[dict[str, object]]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for phase in PHASE_ORDER:
        indices = [int(row["evaluation_index"]) for row in rows if str(row["phase"]) == phase]
        if indices:
            ranges.append((phase, min(indices), max(indices)))
    return ranges


def _shade_phases(ax, phase_ranges: list[tuple[str, int, int]], *, add_labels: bool) -> None:
    for phase, start, end in phase_ranges:
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=PHASE_COLORS.get(phase, "#cccccc"),
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )
        if add_labels:
            ax.text(
                (start + end) / 2.0,
                0.98,
                PHASE_LABELS.get(phase, phase),
                ha="center",
                va="top",
                fontsize=8,
                rotation=0,
                transform=ax.get_xaxis_transform(),
            )
        if start > 1:
            ax.axvline(start - 0.5, color="0.55", linewidth=0.8, linestyle=":", zorder=1)


def plot_objective_by_target(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    targets = sorted({float(row["target_front_speed_mm_s"]) for row in rows})
    if not targets:
        return
    grouped_by_target: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_by_target[float(row["target_front_speed_mm_s"])].append(row)

    ncols = 2
    nrows = math.ceil(len(targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 3.8 * nrows), sharex=True)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])

    for target, ax in zip(targets, axes_list, strict=False):
        target_rows = grouped_by_target[target]
        phase_ranges = _phase_ranges(target_rows)
        _shade_phases(ax, phase_ranges, add_labels=(target == targets[0]))

        for seed in sorted({int(row["seed"]) for row in target_rows}):
            seed_rows = sorted(
                [row for row in target_rows if int(row["seed"]) == seed],
                key=lambda item: int(item["evaluation_index"]),
            )
            color = COLOR_BY_SEED.get(seed)
            x = [int(row["evaluation_index"]) for row in seed_rows]
            y = [float(row["objective_value"]) for row in seed_rows]
            best = [float(row["best_objective_value_after_eval"]) for row in seed_rows]
            ax.scatter(x, y, s=8, color=color, alpha=0.22, linewidths=0)
            ax.plot(x, best, color=color, linewidth=1.35, label=f"seed {seed}")

        ax.set_title(f"target {target:.3f} mm/s")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.23)
        ax.set_ylabel("Objective value J")

    for ax in axes_list[len(targets):]:
        ax.axis("off")
    for ax in axes_list[-ncols:]:
        ax.set_xlabel("Evaluation index")

    seed_handles = [
        plt.Line2D([0], [0], color=COLOR_BY_SEED[seed], linewidth=1.8, label=f"seed {seed}")
        for seed in sorted(COLOR_BY_SEED)
    ]
    phase_handles = [
        Patch(facecolor=PHASE_COLORS[phase], alpha=0.25, label=PHASE_LABELS[phase])
        for phase in PHASE_ORDER
    ]
    axes_list[0].legend(handles=[*seed_handles, *phase_handles], loc="upper right", fontsize=8, ncols=1)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    midpoint = len(finite) // 2
    if len(finite) % 2:
        return float(finite[midpoint])
    return float(0.5 * (finite[midpoint - 1] + finite[midpoint]))


def plot_phase_summary(path: Path, summary_rows: list[dict[str, object]], *, title: str) -> None:
    targets = sorted({float(row["target_front_speed_mm_s"]) for row in summary_rows})
    if not targets:
        return
    x = list(range(len(PHASE_ORDER)))
    labels = ["theta0", "init", "BO", "refinement"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for target in targets:
        target_rows = [row for row in summary_rows if abs(float(row["target_front_speed_mm_s"]) - target) < 1e-12]
        y = [
            _median([float(row["best_after_theta0"]) for row in target_rows]),
            _median([float(row["best_after_init"]) for row in target_rows]),
            _median([float(row["best_after_bo"]) for row in target_rows]),
            _median([float(row["best_after_refinement"]) for row in target_rows]),
        ]
        ax.plot(x, y, marker="o", linewidth=1.7, label=f"{target:.3f} mm/s")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("Median best objective J")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BO objective convergence with optimization phases marked.")
    parser.add_argument("--runs-root", type=Path, default=Path("code_simulation/results/active/bo_velocity_control/n8/coarse"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name-regex", required=True)
    parser.add_argument("--title", default="BO objective value by evaluation and phase")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_re = re.compile(args.run_name_regex)
    rows = collect_evaluations(args.runs_root, run_re)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_evaluation_csv(args.output_root / "objective_by_evaluation.csv", rows)
    summary_rows = _write_phase_summary(args.output_root / "objective_phase_summary.csv", rows)
    plot_objective_by_target(
        args.output_root / "objective_by_iteration_with_phases.png",
        rows,
        title=args.title,
    )
    plot_phase_summary(
        args.output_root / "objective_phase_best_summary.png",
        summary_rows,
        title="Median best objective after each optimization phase",
    )
    print(f"Found evaluations: {len(rows)}")
    print(f"Found runs: {len({row['run_name'] for row in rows})}")
    print(f"Evaluation CSV: {(args.output_root / 'objective_by_evaluation.csv').resolve()}")
    print(f"Phase summary CSV: {(args.output_root / 'objective_phase_summary.csv').resolve()}")
    print(f"Phase plot: {(args.output_root / 'objective_by_iteration_with_phases.png').resolve()}")


if __name__ == "__main__":
    main()
