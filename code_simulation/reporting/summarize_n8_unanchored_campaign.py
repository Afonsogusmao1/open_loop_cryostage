#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import tomllib
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
DEFAULT_RUN_RE = r"bo_v(?P<tag>\d+p\d+)_n8_earlydense_unanchored_seed(?P<seed>\d+)$"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _target_from_tag(tag: str) -> float:
    return float(tag.replace("p", "."))


def _run_metadata(run_dir: Path, run_re: re.Pattern[str]) -> tuple[float, int]:
    match = run_re.match(run_dir.name)
    fallback_target = _target_from_tag(match.group("tag")) if match else math.nan
    fallback_seed = int(match.group("seed")) if match else -1
    effective_config = run_dir / "effective_config.toml"
    if not effective_config.exists():
        return fallback_target, fallback_seed
    with effective_config.open("rb") as handle:
        config = tomllib.load(handle)
    target = _finite_float(
        dict(config.get("velocity_target", {})).get("target_front_speed_mm_s"),
        fallback_target,
    )
    seed = int(dict(config.get("bayesian_optimization", {})).get("random_seed", fallback_seed))
    return target, seed


def _history_metrics(history_rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    feasible = [
        row
        for row in history_rows
        if str(row.get("is_valid", "")).strip() in {"1", "true", "True", "TRUE"}
        and math.isfinite(_finite_float(row.get("objective_value")))
    ]
    if not feasible:
        return {
            "n_evaluations": len(history_rows),
            "best_objective": math.nan,
            "best_evaluation_index": -1,
            "last_improved_evaluation_index": -1,
            "objective_improved_in_last_10": "",
        }

    best_so_far = math.inf
    best_idx = -1
    last_improved_idx = -1
    first_obj = _finite_float(feasible[0].get("objective_value"))
    final_obj = math.nan
    for row in feasible:
        idx = int(float(row["evaluation_index"]))
        objective = _finite_float(row.get("objective_value"))
        if objective < best_so_far - 1.0e-15:
            best_so_far = objective
            best_idx = idx
            last_improved_idx = idx
        final_obj = min(best_so_far, objective)
    last_eval = max(int(float(row["evaluation_index"])) for row in feasible)
    improved_last_10 = bool(last_improved_idx >= max(1, last_eval - 9))
    return {
        "n_evaluations": len(history_rows),
        "first_objective": first_obj,
        "best_objective": final_obj,
        "best_evaluation_index": best_idx,
        "last_improved_evaluation_index": last_improved_idx,
        "objective_improved_in_last_10": int(improved_last_10),
    }


def _tracking_metrics(run_dir: Path) -> dict[str, float]:
    rows = _read_csv_rows(run_dir / "best_tracking_summary.csv")
    if not rows:
        return {
            "achieved_direct_speed_mm_s": math.nan,
            "direct_speed_relative_error_pct": math.nan,
            "tracking_rmse_mm": math.nan,
        }
    row = rows[0]
    target = _finite_float(row.get("target_front_speed_mm_s"))
    achieved = _finite_float(row.get("actual_interval_speed_mm_s"))
    rel_error = math.nan
    if math.isfinite(target) and target > 0.0 and math.isfinite(achieved):
        rel_error = 100.0 * abs(achieved - target) / target
    return {
        "achieved_direct_speed_mm_s": achieved,
        "direct_speed_relative_error_pct": rel_error,
        "tracking_rmse_mm": _finite_float(row.get("tracking_rmse_mm")),
    }


def summarize_runs(runs_root: Path, run_re: re.Pattern[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_re.match(run_dir.name) is None:
            continue
        target, seed = _run_metadata(run_dir, run_re)
        history_rows = _read_csv_rows(run_dir / "bo_history.csv")
        status = "completed" if history_rows and (run_dir / "best_tracking_summary.csv").exists() else "incomplete"
        row: dict[str, object] = {
            "run_name": run_dir.name,
            "target_front_speed_mm_s": target,
            "seed": seed,
            "status": status,
        }
        row.update(_history_metrics(history_rows))
        row.update(_tracking_metrics(run_dir))
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "run_name",
        "target_front_speed_mm_s",
        "seed",
        "status",
        "n_evaluations",
        "first_objective",
        "best_objective",
        "best_evaluation_index",
        "last_improved_evaluation_index",
        "objective_improved_in_last_10",
        "achieved_direct_speed_mm_s",
        "direct_speed_relative_error_pct",
        "tracking_rmse_mm",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_target_vs_achieved(path: Path, rows: list[dict[str, object]], title: str) -> None:
    completed = [
        row
        for row in rows
        if row["status"] == "completed"
        and math.isfinite(float(row["achieved_direct_speed_mm_s"]))
    ]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if completed:
        axis_max = 1.03 * max(
            max(float(row["target_front_speed_mm_s"]) for row in completed),
            max(float(row["achieved_direct_speed_mm_s"]) for row in completed),
        )
    else:
        axis_max = 0.0145
    ax.plot([0.0, axis_max], [0.0, axis_max], "--", color="0.5", linewidth=1.2, label="target = achieved")
    for seed in sorted({int(row["seed"]) for row in completed}):
        seed_rows = [row for row in completed if int(row["seed"]) == seed]
        ax.scatter(
            [float(row["target_front_speed_mm_s"]) for row in seed_rows],
            [float(row["achieved_direct_speed_mm_s"]) for row in seed_rows],
            color=COLOR_BY_SEED.get(seed),
            s=38,
            label=f"seed {seed}",
        )
    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Target speed (mm/s)")
    ax.set_ylabel("Achieved direct speed (mm/s)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_objective_convergence(path: Path, runs_root: Path, rows: list[dict[str, object]], title: str) -> None:
    targets = sorted({float(row["target_front_speed_mm_s"]) for row in rows if math.isfinite(float(row["target_front_speed_mm_s"]))})
    if not targets:
        return
    ncols = 2
    nrows = math.ceil(len(targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), sharex=True)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
    for ax, target in zip(axes_list, targets, strict=False):
        target_rows = [row for row in rows if abs(float(row["target_front_speed_mm_s"]) - target) < 1.0e-12]
        for row in sorted(target_rows, key=lambda item: int(item["seed"])):
            history_rows = _read_csv_rows(runs_root / str(row["run_name"]) / "bo_history.csv")
            x_values: list[int] = []
            y_values: list[float] = []
            for hist in history_rows:
                value = _finite_float(hist.get("best_objective_value_after_eval"))
                if not math.isfinite(value) or value <= 0.0:
                    continue
                x_values.append(int(float(hist["evaluation_index"])))
                y_values.append(value)
            if x_values:
                seed = int(row["seed"])
                ax.plot(x_values, y_values, color=COLOR_BY_SEED.get(seed), linewidth=1.2, label=f"seed {seed}")
        ax.set_title(f"target {target:.3f} mm/s")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylabel("Best objective")
    for ax in axes_list[len(targets):]:
        ax.axis("off")
    for ax in axes_list[-ncols:]:
        ax.set_xlabel("Evaluation")
    axes_list[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the n8 early-dense unanchored BO campaign.")
    parser.add_argument("--runs-root", type=Path, default=Path("code_simulation/results/active/bo_velocity_control/n8/coarse"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name-regex", default=DEFAULT_RUN_RE)
    parser.add_argument("--target-plot-title", default="n8 early-dense unanchored BO: target vs achieved")
    parser.add_argument("--convergence-plot-title", default="n8 early-dense unanchored BO objective convergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_re = re.compile(args.run_name_regex)
    rows = summarize_runs(args.runs_root, run_re)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(args.output_root / "campaign_summary.csv", rows)
    _plot_target_vs_achieved(args.output_root / "target_vs_achieved_direct_speed.png", rows, args.target_plot_title)
    _plot_objective_convergence(args.output_root / "objective_convergence_by_target.png", args.runs_root, rows, args.convergence_plot_title)
    completed = sum(1 for row in rows if row["status"] == "completed")
    print(f"Found runs: {len(rows)}")
    print(f"Completed runs: {completed}/{len(rows)}")
    print(f"Summary: {(args.output_root / 'campaign_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
