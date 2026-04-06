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

from open_loop_bayesian_optimizer import bayes_opt_runtime_details
from reachability_constraints import (
    conservative_settling_time_s,
    default_constraints_dir,
    load_reachability_constraints,
)
from run_open_loop_optimization import (
    _default_theta0_for_config,
    build_problem_config,
    main as run_open_loop_optimization_main,
)


DEFAULT_STUDY_NAME = "fixed_n_bo_comparison_full_process_seed17_init4_iter8"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"
DEFAULT_N_VALUES = (3, 4, 5, 6, 7)
DEFAULT_FORMULATION = "full_process_article"
DEFAULT_RANDOM_SEED = 17
DEFAULT_INIT_POINTS = 4
DEFAULT_N_ITER = 8
DEFAULT_ACQ_KIND = "ucb"
DEFAULT_KAPPA = 2.0
DEFAULT_XI = 0.0
DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY = 1.0e6
DEFAULT_BOUND_LOWER_ANCHORS_C = (-0.5, -9.0, -15.0, -18.0, -20.0)
DEFAULT_BOUND_UPPER_ANCHORS_C = (0.0, -3.0, -8.0, -12.0, -14.0)
POST_INITIAL_WARM_LIMIT_C = -5.0
HISTORY_FIELDNAMES = [
    "evaluation_index",
    "phase",
    "case_name",
    "objective_value",
    "optimizer_target_value",
    "is_valid",
    "expensive_simulation_executed",
    "incumbent_after_eval",
    "best_objective_value_after_eval",
    "runtime_s",
    "out_dir",
    "feasibility_status",
    "error_message",
    "raw_candidate_json",
    "theta_json",
]


@dataclass(frozen=True)
class FixedNRunSummary:
    num_knots: int
    status: str
    run_name: str
    run_dir: Path
    theta_bounds_C: tuple[tuple[float, float], ...]
    seed_theta_C: tuple[float, ...]
    seeded_baseline_feasible: bool
    seed_objective_value: float
    best_objective_value: float
    best_theta_C: tuple[float, ...]
    total_evaluations: int
    feasible_evaluations: int
    infeasible_evaluations: int
    evaluation_errors: int
    expensive_simulation_runs: int
    total_runtime_s: float
    median_feasible_runtime_s: float
    best_evaluation_runtime_s: float
    improved_beyond_seeded_baseline: bool
    dominated_by_infeasible_suggestions: bool
    warning: str
    key_message: str


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 4.5),
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _parse_n_values(raw_n_values: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_n_values.split(",") if part.strip())
    if not values:
        raise ValueError("--n-values must contain at least one knot count")
    if any(value < 2 for value in values):
        raise ValueError("all knot counts must be at least 2")
    if len(set(values)) != len(values):
        raise ValueError("--n-values must not contain duplicates")
    return values


def _format_float_tuple(values: tuple[float, ...], *, precision: int = 3) -> str:
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in values) + ")"


def _interpolated_theta_bounds_C(num_knots: int) -> tuple[tuple[float, float], ...]:
    base_tau = np.linspace(0.0, 1.0, len(DEFAULT_BOUND_LOWER_ANCHORS_C), dtype=np.float64)
    target_tau = np.linspace(0.0, 1.0, int(num_knots), dtype=np.float64)
    lower = np.interp(target_tau, base_tau, np.asarray(DEFAULT_BOUND_LOWER_ANCHORS_C, dtype=np.float64))
    upper = np.interp(target_tau, base_tau, np.asarray(DEFAULT_BOUND_UPPER_ANCHORS_C, dtype=np.float64))
    if int(num_knots) > 1:
        upper[1:] = np.minimum(upper[1:], POST_INITIAL_WARM_LIMIT_C)
    bounds = []
    for idx, (lower_C, upper_C) in enumerate(zip(lower, upper, strict=True)):
        if not lower_C < upper_C:
            raise ValueError(f"empty BO bound interval at knot index {idx}: [{lower_C}, {upper_C}]")
        bounds.append((float(lower_C), float(upper_C)))
    return tuple(bounds)


def _theta_bounds_arg(theta_bounds_C: tuple[tuple[float, float], ...]) -> str:
    return ",".join(f"{lower_C:.6f}:{upper_C:.6f}" for lower_C, upper_C in theta_bounds_C)


def _seed_theta_for_run(num_knots: int, theta_bounds_C: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    config = build_problem_config(formulation=DEFAULT_FORMULATION, num_knots=int(num_knots))
    theta0 = np.asarray(_default_theta0_for_config(config), dtype=np.float64)
    lower = np.asarray([pair[0] for pair in theta_bounds_C], dtype=np.float64)
    upper = np.asarray([pair[1] for pair in theta_bounds_C], dtype=np.float64)
    seeded = np.clip(theta0, lower, upper)
    return tuple(float(value) for value in seeded)


def _theta_arg(theta_C: tuple[float, ...]) -> str:
    return ",".join(f"{float(value):.6f}" for value in theta_C)


def _structural_prescreen_message(*, num_knots: int, theta_bounds_C: tuple[tuple[float, float], ...]) -> str | None:
    config = build_problem_config(formulation=DEFAULT_FORMULATION, num_knots=int(num_knots))
    if len(config.knot_times_s) <= 2:
        return None

    constraints = load_reachability_constraints(default_constraints_dir(Path(__file__).resolve().parent))
    warmest_characterized_target_C = float(max(constraints.characterized_targets_C))
    first_interval_s = float(config.knot_times_s[1] - config.knot_times_s[0])
    first_upper_bound_C = float(theta_bounds_C[1][1])
    if first_upper_bound_C > warmest_characterized_target_C + 1e-12:
        return None

    settling_time_s = float(conservative_settling_time_s(warmest_characterized_target_C, constraints))
    if first_interval_s + 1e-12 >= settling_time_s:
        return None

    return (
        "pre-screened out before BO because the first uniform knot interval is shorter than the conservative "
        f"settling time to the warmest characterized cold target: Δt1={first_interval_s:.1f} s, "
        f"settling(-5.0 C)={settling_time_s:.1f} s, first-knot upper bound={first_upper_bound_C:.2f} C"
    )


def _write_placeholder_figure(out_path: Path, *, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.92, "\n".join(lines), va="top", ha="left", fontsize=10.5, family="monospace")
    fig.savefig(out_path)
    plt.close(fig)


def _write_empty_history_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()


def _write_empty_best_theta_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["knot_index", "time_s", "temperature_C"])


def _write_prescreen_placeholder_run(
    *,
    run_dir: Path,
    run_name: str,
    num_knots: int,
    theta_bounds_C: tuple[tuple[float, float], ...],
    seed_theta_C: tuple[float, ...],
    reason: str,
    bo_settings: dict[str, float | int | str | bool],
) -> FixedNRunSummary:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluations").mkdir(exist_ok=True)

    history_csv_path = run_dir / "evaluation_history.csv"
    best_theta_csv_path = run_dir / "best_theta_profile.csv"
    best_summary_path = run_dir / "best_solution_summary.md"
    run_note_path = run_dir / "study_run_note.md"
    run_settings_path = run_dir / "run_settings.json"

    _write_empty_history_csv(history_csv_path)
    _write_empty_best_theta_csv(best_theta_csv_path)
    _write_placeholder_figure(
        analysis_dir / "objective_history.png",
        title=f"Objective History (k={num_knots})",
        lines=["No BO evaluations were launched.", reason],
    )
    _write_placeholder_figure(
        analysis_dir / "feasible_vs_infeasible.png",
        title=f"Feasible vs Infeasible Suggestions (k={num_knots})",
        lines=["No BO evaluations were launched.", "The model order was eliminated by the deterministic pre-screen."],
    )
    _write_placeholder_figure(
        analysis_dir / "best_front_tracking.png",
        title=f"Best Front Tracking (k={num_knots})",
        lines=["No admissible candidate was evaluated.", "There is no best front-tracking trace for this knot count."],
    )

    best_summary_path.write_text(
        "\n".join(
            [
                "# Open-Loop Optimization Summary",
                "",
                "## Study status",
                f"- Knot count: `{num_knots}`",
                "- Status: `pre-screened out before BO launch`",
                f"- Reason: `{reason}`",
                "",
                "## Fixed study policy",
                "- Objective: unchanged front-position tracking objective.",
                "- Admissibility: unchanged transient plus empirical hold gating before expensive simulation.",
                f"- BO bounds: `{theta_bounds_C}`",
                f"- Seed theta after applying the shared bound policy: `{seed_theta_C}`",
                "- No expensive simulation was executed for this knot count.",
                "",
                "## Artifacts",
                f"- Evaluation history CSV: `{history_csv_path}`",
                f"- Best theta profile CSV: `{best_theta_csv_path}`",
                f"- Objective history plot: `{analysis_dir / 'objective_history.png'}`",
                f"- Feasibility plot: `{analysis_dir / 'feasible_vs_infeasible.png'}`",
                f"- Best front tracking plot: `{analysis_dir / 'best_front_tracking.png'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_note_path.write_text(
        "\n".join(
            [
                "# Fixed-N BO Study Run Note",
                "",
                f"- Knot count: `{num_knots}`",
                "- Run status: `pre-screened out before BO launch`",
                f"- Shared BO settings: `{bo_settings}`",
                f"- Shared bound policy output: `{theta_bounds_C}`",
                f"- Seed theta after shared clipping: `{seed_theta_C}`",
                f"- Deterministic elimination reason: `{reason}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_settings_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "num_knots": int(num_knots),
                "status": "prescreen_eliminated",
                "reason": reason,
                "theta_bounds_C": theta_bounds_C,
                "seed_theta_C": seed_theta_C,
                "bo": bo_settings,
                "artifacts": {
                    "history_csv_path": str(history_csv_path.resolve()),
                    "best_theta_profile_csv": str(best_theta_csv_path.resolve()),
                    "best_summary_md": str(best_summary_path.resolve()),
                    "objective_history_plot": str((analysis_dir / "objective_history.png").resolve()),
                    "feasibility_plot": str((analysis_dir / "feasible_vs_infeasible.png").resolve()),
                    "front_tracking_plot": str((analysis_dir / "best_front_tracking.png").resolve()),
                    "study_run_note_md": str(run_note_path.resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FixedNRunSummary(
        num_knots=int(num_knots),
        status="prescreen_eliminated",
        run_name=run_name,
        run_dir=run_dir,
        theta_bounds_C=theta_bounds_C,
        seed_theta_C=seed_theta_C,
        seeded_baseline_feasible=False,
        seed_objective_value=math.nan,
        best_objective_value=math.nan,
        best_theta_C=tuple(),
        total_evaluations=0,
        feasible_evaluations=0,
        infeasible_evaluations=0,
        evaluation_errors=0,
        expensive_simulation_runs=0,
        total_runtime_s=0.0,
        median_feasible_runtime_s=math.nan,
        best_evaluation_runtime_s=math.nan,
        improved_beyond_seeded_baseline=False,
        dominated_by_infeasible_suggestions=False,
        warning="",
        key_message=reason,
    )


def _load_best_theta_profile(path: Path) -> tuple[float, ...]:
    theta_C: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta_C.append(float(row["temperature_C"]))
    return tuple(theta_C)


def _bool_from_csv(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True", "TRUE"}


def _load_completed_run_summary(*, num_knots: int, run_name: str, run_dir: Path, theta_bounds_C, seed_theta_C) -> FixedNRunSummary:
    history_csv_path = run_dir / "evaluation_history.csv"
    best_theta_path = run_dir / "best_theta_profile.csv"
    rows: list[dict[str, str]] = []
    with history_csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    feasible_rows = [row for row in rows if _bool_from_csv(row["is_valid"])]
    infeasible_rows = [row for row in rows if row["feasibility_status"] == "infeasible"]
    error_rows = [row for row in rows if row["feasibility_status"] == "evaluation_error"]
    seed_rows = [row for row in rows if row["phase"] == "seed"]
    runtime_values = np.asarray([float(row["runtime_s"]) for row in rows], dtype=np.float64) if rows else np.asarray([], dtype=np.float64)
    feasible_runtime_values = (
        np.asarray([float(row["runtime_s"]) for row in feasible_rows], dtype=np.float64)
        if feasible_rows
        else np.asarray([], dtype=np.float64)
    )

    best_theta_C = _load_best_theta_profile(best_theta_path) if best_theta_path.exists() else tuple()
    if feasible_rows:
        best_row = min(feasible_rows, key=lambda row: float(row["objective_value"]))
        best_objective_value = float(best_row["objective_value"])
        best_runtime_s = float(best_row["runtime_s"])
    else:
        best_objective_value = math.nan
        best_runtime_s = math.nan

    seeded_baseline_feasible = bool(seed_rows and _bool_from_csv(seed_rows[0]["is_valid"]))
    seed_objective_value = float(seed_rows[0]["objective_value"]) if seed_rows else math.nan
    improved_beyond_seeded_baseline = bool(
        feasible_rows
        and seed_rows
        and math.isfinite(seed_objective_value)
        and best_objective_value < seed_objective_value - 1.0e-12
    )

    total_evaluations = len(rows)
    feasible_evaluations = len(feasible_rows)
    infeasible_evaluations = len(infeasible_rows)
    evaluation_errors = len(error_rows)
    expensive_runs = sum(1 for row in rows if _bool_from_csv(row["expensive_simulation_executed"]))
    dominated = infeasible_evaluations > feasible_evaluations
    warning = "dominated by infeasible BO suggestions" if dominated else ""
    key_message = rows[-1]["error_message"] if rows and rows[-1]["error_message"] else "completed"

    return FixedNRunSummary(
        num_knots=int(num_knots),
        status="completed",
        run_name=run_name,
        run_dir=run_dir,
        theta_bounds_C=tuple((float(lo), float(hi)) for lo, hi in theta_bounds_C),
        seed_theta_C=seed_theta_C,
        seeded_baseline_feasible=seeded_baseline_feasible,
        seed_objective_value=seed_objective_value,
        best_objective_value=best_objective_value,
        best_theta_C=best_theta_C,
        total_evaluations=total_evaluations,
        feasible_evaluations=feasible_evaluations,
        infeasible_evaluations=infeasible_evaluations,
        evaluation_errors=evaluation_errors,
        expensive_simulation_runs=expensive_runs,
        total_runtime_s=float(np.sum(runtime_values)) if runtime_values.size else 0.0,
        median_feasible_runtime_s=float(np.median(feasible_runtime_values)) if feasible_runtime_values.size else math.nan,
        best_evaluation_runtime_s=best_runtime_s,
        improved_beyond_seeded_baseline=improved_beyond_seeded_baseline,
        dominated_by_infeasible_suggestions=dominated,
        warning=warning,
        key_message=key_message,
    )


def _write_per_run_note(summary: FixedNRunSummary, *, bo_settings: dict[str, float | int | str | bool]) -> None:
    note_path = summary.run_dir / "study_run_note.md"
    lines = [
        "# Fixed-N BO Study Run Note",
        "",
        f"- Knot count: `{summary.num_knots}`",
        f"- Run status: `{summary.status}`",
        f"- Shared BO settings: `{bo_settings}`",
        f"- Shared bound policy output: `{summary.theta_bounds_C}`",
        f"- Seed theta after shared clipping: `{summary.seed_theta_C}`",
        f"- Seeded baseline feasible: `{summary.seeded_baseline_feasible}`",
        f"- Total evaluations: `{summary.total_evaluations}`",
        f"- Feasible evaluations: `{summary.feasible_evaluations}`",
        f"- Infeasible evaluations: `{summary.infeasible_evaluations}`",
        f"- Evaluation errors: `{summary.evaluation_errors}`",
        f"- Expensive simulations executed: `{summary.expensive_simulation_runs}`",
        f"- Best objective: `{summary.best_objective_value:.9e}`" if math.isfinite(summary.best_objective_value) else "- Best objective: `n/a`",
        f"- Best theta: `{summary.best_theta_C}`" if summary.best_theta_C else "- Best theta: `n/a`",
        f"- Improved beyond seeded baseline: `{summary.improved_beyond_seeded_baseline}`",
    ]
    if summary.warning:
        lines.append(f"- Warning: `{summary.warning}`")
    lines.append(f"- Key message: `{summary.key_message}`")
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_study_summary_csv(path: Path, summaries: tuple[FixedNRunSummary, ...]) -> None:
    fieldnames = [
        "num_knots",
        "status",
        "run_name",
        "run_dir",
        "best_objective_value",
        "best_theta_json",
        "seeded_baseline_feasible",
        "seed_objective_value",
        "total_evaluations",
        "feasible_evaluations",
        "infeasible_evaluations",
        "evaluation_errors",
        "expensive_simulation_runs",
        "total_runtime_s",
        "median_feasible_runtime_s",
        "best_evaluation_runtime_s",
        "improved_beyond_seeded_baseline",
        "dominated_by_infeasible_suggestions",
        "warning",
        "theta_bounds_json",
        "seed_theta_json",
        "key_message",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "num_knots": summary.num_knots,
                    "status": summary.status,
                    "run_name": summary.run_name,
                    "run_dir": str(summary.run_dir.resolve()),
                    "best_objective_value": summary.best_objective_value,
                    "best_theta_json": json.dumps(list(summary.best_theta_C)),
                    "seeded_baseline_feasible": int(summary.seeded_baseline_feasible),
                    "seed_objective_value": summary.seed_objective_value,
                    "total_evaluations": summary.total_evaluations,
                    "feasible_evaluations": summary.feasible_evaluations,
                    "infeasible_evaluations": summary.infeasible_evaluations,
                    "evaluation_errors": summary.evaluation_errors,
                    "expensive_simulation_runs": summary.expensive_simulation_runs,
                    "total_runtime_s": summary.total_runtime_s,
                    "median_feasible_runtime_s": summary.median_feasible_runtime_s,
                    "best_evaluation_runtime_s": summary.best_evaluation_runtime_s,
                    "improved_beyond_seeded_baseline": int(summary.improved_beyond_seeded_baseline),
                    "dominated_by_infeasible_suggestions": int(summary.dominated_by_infeasible_suggestions),
                    "warning": summary.warning,
                    "theta_bounds_json": json.dumps([list(pair) for pair in summary.theta_bounds_C]),
                    "seed_theta_json": json.dumps(list(summary.seed_theta_C)),
                    "key_message": summary.key_message,
                }
            )


def _write_study_summary_md(
    path: Path,
    *,
    study_name: str,
    summaries: tuple[FixedNRunSummary, ...],
    n_values: tuple[int, ...],
    bo_settings: dict[str, float | int | str | bool],
    bound_policy_description: str,
    bo_runtime: dict[str, str],
) -> None:
    lines = [
        "# Fixed-N Bayesian-Optimization Comparison Study",
        "",
        "## Shared comparison policy",
        f"- Study name: `{study_name}`",
        f"- Candidate knot counts: `{n_values}`",
        "- Objective: unchanged front-position tracking objective from the existing expensive evaluator.",
        "- Admissibility: unchanged transient characterization plus empirical long-duration hold gating before expensive simulation.",
        "- Optimized variables: knot temperatures only; knot times remain the fixed uniform grid implied by the chosen knot count.",
        f"- BO package: `bayesian-optimization {bo_runtime['package_version']}` from `{bo_runtime['package_path']}`",
        f"- Shared BO settings: `{bo_settings}`",
        f"- Shared bound policy: {bound_policy_description}",
        "",
        "## Results",
        "",
        "| N | Status | Best objective | Feasible | Infeasible | Improved vs seed | Warning |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for summary in summaries:
        best_text = f"{summary.best_objective_value:.6e}" if math.isfinite(summary.best_objective_value) else "n/a"
        lines.append(
            f"| {summary.num_knots} | {summary.status} | {best_text} | {summary.feasible_evaluations} | {summary.infeasible_evaluations} | {summary.improved_beyond_seeded_baseline} | {summary.warning or ' '} |"
        )

    completed = [summary for summary in summaries if summary.status == "completed" and math.isfinite(summary.best_objective_value)]
    if completed:
        best_completed = min(completed, key=lambda summary: summary.best_objective_value)
        recommendation = (
            f"Carry forward `N={best_completed.num_knots}` because it achieved the lowest best objective "
            f"({best_completed.best_objective_value:.6e}) under the shared BO and admissibility policy."
        )
    else:
        recommendation = "No completed BO run produced a feasible incumbent."
    lines.extend(
        [
            "",
            "## Recommendation",
            f"- {recommendation}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_study_comparison(path: Path, summaries: tuple[FixedNRunSummary, ...]) -> None:
    n_values = np.asarray([summary.num_knots for summary in summaries], dtype=np.float64)
    best_values = np.asarray(
        [summary.best_objective_value if math.isfinite(summary.best_objective_value) else np.nan for summary in summaries],
        dtype=np.float64,
    )
    feasible_fraction = np.asarray(
        [
            (summary.feasible_evaluations / summary.total_evaluations) if summary.total_evaluations > 0 else np.nan
            for summary in summaries
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)
    axes[0].plot(n_values, best_values, marker="o", linewidth=2.0, color="#2A6F97")
    for summary, x_value, y_value in zip(summaries, n_values, best_values, strict=True):
        if math.isfinite(y_value):
            axes[0].annotate(summary.status, (x_value, y_value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        else:
            axes[0].annotate(summary.status, (x_value, 0.02), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_title("Best Objective Across Fixed Knot Counts")
    axes[0].set_ylabel("Best objective value")
    finite_best = best_values[np.isfinite(best_values)]
    if finite_best.size >= 2 and np.min(finite_best) > 0.0 and (np.max(finite_best) / np.min(finite_best)) >= 10.0:
        axes[0].set_yscale("log")

    axes[1].bar(n_values, feasible_fraction, width=0.6, color="#2A9D8F", alpha=0.85)
    axes[1].set_title("Feasible Fraction Across Fixed Knot Counts")
    axes[1].set_xlabel("Number of knot temperatures (N)")
    axes[1].set_ylabel("feasible eval fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(n_values)
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-N Bayesian-optimization comparison study.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Deterministic study folder name.")
    parser.add_argument(
        "--out-root-dir",
        default=str(DEFAULT_OUT_ROOT_DIR),
        help="Root folder where the study folder is written.",
    )
    parser.add_argument(
        "--n-values",
        default=",".join(str(value) for value in DEFAULT_N_VALUES),
        help="Comma-separated knot counts to compare.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Shared BO random seed.")
    parser.add_argument("--init-points", type=int, default=DEFAULT_INIT_POINTS, help="Shared BO random suggestions.")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER, help="Shared BO acquisition-guided suggestions.")
    parser.add_argument(
        "--acq-kind",
        default=DEFAULT_ACQ_KIND,
        choices=("ucb", "ei", "poi"),
        help="BO acquisition function kind.",
    )
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA, help="BO acquisition kappa parameter.")
    parser.add_argument("--xi", type=float, default=DEFAULT_XI, help="BO acquisition xi parameter.")
    parser.add_argument(
        "--infeasible-objective-penalty",
        type=float,
        default=DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY,
        help="Deterministic objective penalty used when admissibility fails before simulation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing deterministic study folder before running.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_matplotlib()

    n_values = _parse_n_values(args.n_values)
    bo_runtime = bayes_opt_runtime_details()
    bo_settings: dict[str, float | int | str | bool] = {
        "method": "bayesian-optimization",
        "random_seed": int(args.seed),
        "init_points": int(args.init_points),
        "n_iter": int(args.n_iter),
        "acquisition_kind": str(args.acq_kind),
        "acquisition_kappa": float(args.kappa),
        "acquisition_xi": float(args.xi),
        "seed_with_theta0": True,
        "infeasible_objective_penalty": float(args.infeasible_objective_penalty),
    }
    bound_policy_description = (
        "Interpolate the canonical 5-knot BO envelope anchors "
        f"lower={DEFAULT_BOUND_LOWER_ANCHORS_C}, upper={DEFAULT_BOUND_UPPER_ANCHORS_C} over normalized knot time, "
        f"then cap every post-initial knot upper bound at {POST_INITIAL_WARM_LIMIT_C:.1f} C so BO never searches warmer "
        "targets than the characterization-supported cold-transition range."
    )

    out_root_dir = Path(args.out_root_dir)
    study_dir = out_root_dir / args.study_name
    runs_root_dir = study_dir / "runs"
    if study_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{study_dir} already exists. Re-run with --overwrite to replace this deterministic study folder."
            )
        shutil.rmtree(study_dir)
    runs_root_dir.mkdir(parents=True, exist_ok=True)

    study_settings_path = study_dir / "study_settings.json"
    summaries: list[FixedNRunSummary] = []

    for num_knots in n_values:
        run_name = f"bo_compare_full_process_k{num_knots}_seed{int(args.seed)}_init{int(args.init_points)}_iter{int(args.n_iter)}"
        run_dir = runs_root_dir / run_name
        theta_bounds_C = _interpolated_theta_bounds_C(int(num_knots))
        seed_theta_C = _seed_theta_for_run(int(num_knots), theta_bounds_C)

        prescreen_reason = _structural_prescreen_message(num_knots=int(num_knots), theta_bounds_C=theta_bounds_C)
        if prescreen_reason is not None:
            summary = _write_prescreen_placeholder_run(
                run_dir=run_dir,
                run_name=run_name,
                num_knots=int(num_knots),
                theta_bounds_C=theta_bounds_C,
                seed_theta_C=seed_theta_C,
                reason=prescreen_reason,
                bo_settings=bo_settings,
            )
            summaries.append(summary)
            continue

        run_open_loop_optimization_main(
            [
                "--run-name",
                run_name,
                "--out-root-dir",
                str(runs_root_dir),
                "--formulation",
                DEFAULT_FORMULATION,
                "--num-knots",
                str(int(num_knots)),
                f"--theta0={_theta_arg(seed_theta_C)}",
                "--method",
                "bayesian-optimization",
                "--seed",
                str(int(args.seed)),
                "--init-points",
                str(int(args.init_points)),
                "--n-iter",
                str(int(args.n_iter)),
                "--acq-kind",
                str(args.acq_kind),
                "--kappa",
                str(float(args.kappa)),
                "--xi",
                str(float(args.xi)),
                f"--theta-bounds={_theta_bounds_arg(theta_bounds_C)}",
                "--infeasible-objective-penalty",
                str(float(args.infeasible_objective_penalty)),
            ]
        )
        summary = _load_completed_run_summary(
            num_knots=int(num_knots),
            run_name=run_name,
            run_dir=run_dir,
            theta_bounds_C=theta_bounds_C,
            seed_theta_C=seed_theta_C,
        )
        _write_per_run_note(summary, bo_settings=bo_settings)
        summaries.append(summary)

    summary_tuple = tuple(sorted(summaries, key=lambda item: item.num_knots))
    study_settings_path.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "n_values": list(n_values),
                "shared_bo_settings": bo_settings,
                "bound_policy_description": bound_policy_description,
                "bo_runtime": bo_runtime,
                "runs_root_dir": str(runs_root_dir.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_study_summary_csv(study_dir / "study_summary.csv", summary_tuple)
    _write_study_summary_md(
        study_dir / "study_summary.md",
        study_name=args.study_name,
        summaries=summary_tuple,
        n_values=n_values,
        bo_settings=bo_settings,
        bound_policy_description=bound_policy_description,
        bo_runtime=bo_runtime,
    )
    _plot_study_comparison(study_dir / "comparison_across_knot_count.png", summary_tuple)

    print(f"Fixed-N BO study completed: {study_dir}")
    for summary in summary_tuple:
        best_text = f"{summary.best_objective_value:.9e}" if math.isfinite(summary.best_objective_value) else "n/a"
        print(
            f"N={summary.num_knots}: status={summary.status}, best={best_text}, "
            f"feasible={summary.feasible_evaluations}, infeasible={summary.infeasible_evaluations}, "
            f"warning={summary.warning or 'none'}"
        )


if __name__ == "__main__":
    main()
