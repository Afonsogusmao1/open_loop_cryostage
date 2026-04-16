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

from open_loop_workflow_config import (
    SUPPORTED_KNOT_TIME_SCHEDULES,
    build_problem_config,
    default_theta0_for_config,
    parse_normalized_support_tau_by_n_arg,
)
from open_loop_bayesian_optimizer import bayes_opt_runtime_details
from reachability_constraints import (
    conservative_settling_time_s,
    default_constraints_dir,
    load_reachability_constraints,
)
from run_open_loop_optimization import main as run_open_loop_optimization_main


DEFAULT_STUDY_NAME = "schedule_sensitivity_full_process_seed17_init4_iter8"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"
DEFAULT_SCHEDULE_FAMILIES = ("uniform", "early_dense", "mid_dense", "late_dense")
DEFAULT_FORMULATION = "full_process_article"
DEFAULT_RANDOM_SEED = 17
DEFAULT_INIT_POINTS = 4
DEFAULT_N_ITER = 8
DEFAULT_ACQ_KIND = "ucb"
DEFAULT_KAPPA = 2.0
DEFAULT_XI = 0.0
DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY = 1.0e6
BOUND_POLICY_TEMPLATE_SUPPORT_TAU = (0.0, 0.25, 0.5, 0.75, 1.0)
BOUND_POLICY_LOWER_TEMPERATURES_C = (-0.5, -9.0, -15.0, -18.0, -20.0)
BOUND_POLICY_UPPER_TEMPERATURES_C = (0.0, -3.0, -8.0, -12.0, -14.0)
POST_INITIAL_WARM_LIMIT_C = -5.0
PLACEHOLDER_FIGURES = (
    "objective_history.png",
    "feasible_vs_infeasible.png",
    "best_front_tracking.png",
    "plate_reference_trajectory.png",
)


@dataclass(frozen=True)
class ScheduleSensitivityRunSummary:
    knot_time_schedule: str
    knot_time_normalized_support_tau: tuple[float, ...]
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
        raise ValueError("--n-values must contain at least one trajectory parameter count")
    if any(value < 2 for value in values):
        raise ValueError("all trajectory parameter counts must be at least 2")
    if len(set(values)) != len(values):
        raise ValueError("--n-values must not contain duplicates")
    return values


def _parse_schedule_families(raw_schedule_families: str) -> tuple[str, ...]:
    families = tuple(part.strip() for part in raw_schedule_families.split(",") if part.strip())
    if not families:
        raise ValueError("--schedule-families must contain at least one schedule name")
    invalid = [family for family in families if family not in SUPPORTED_KNOT_TIME_SCHEDULES]
    if invalid:
        raise ValueError(
            f"Unsupported schedule families {invalid!r}; expected members of {SUPPORTED_KNOT_TIME_SCHEDULES!r}"
        )
    if len(set(families)) != len(families):
        raise ValueError("--schedule-families must not contain duplicates")
    return families


def _normalized_support_tau_from_config(config) -> tuple[float, ...]:
    return tuple(float(value) / float(config.horizon_s) for value in config.knot_times_s)


def _format_float_tuple(values: tuple[float, ...], *, precision: int = 3) -> str:
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in values) + ")"


def _interpolated_theta_bounds_C(normalized_support_tau: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    base_tau = np.asarray(BOUND_POLICY_TEMPLATE_SUPPORT_TAU, dtype=np.float64)
    target_tau = np.asarray(normalized_support_tau, dtype=np.float64)
    lower = np.interp(target_tau, base_tau, np.asarray(BOUND_POLICY_LOWER_TEMPERATURES_C, dtype=np.float64))
    upper = np.interp(target_tau, base_tau, np.asarray(BOUND_POLICY_UPPER_TEMPERATURES_C, dtype=np.float64))
    if target_tau.size > 1:
        upper[1:] = np.minimum(upper[1:], POST_INITIAL_WARM_LIMIT_C)
    bounds: list[tuple[float, float]] = []
    for idx, (lower_C, upper_C) in enumerate(zip(lower, upper, strict=True)):
        if not lower_C < upper_C:
            raise ValueError(f"empty BO bound interval at knot index {idx}: [{lower_C}, {upper_C}]")
        bounds.append((float(lower_C), float(upper_C)))
    return tuple(bounds)


def _theta_bounds_arg(theta_bounds_C: tuple[tuple[float, float], ...]) -> str:
    return ",".join(f"{lower_C:.6f}:{upper_C:.6f}" for lower_C, upper_C in theta_bounds_C)


def _seed_theta_for_run(config, theta_bounds_C: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    theta0 = np.asarray(default_theta0_for_config(config), dtype=np.float64)
    lower = np.asarray([pair[0] for pair in theta_bounds_C], dtype=np.float64)
    upper = np.asarray([pair[1] for pair in theta_bounds_C], dtype=np.float64)
    seeded = np.clip(theta0, lower, upper)
    return tuple(float(value) for value in seeded)


def _theta_arg(theta_C: tuple[float, ...]) -> str:
    return ",".join(f"{float(value):.6f}" for value in theta_C)


def _structural_prescreen_message(config, theta_bounds_C: tuple[tuple[float, float], ...]) -> str | None:
    if len(config.knot_times_s) <= 2:
        return None

    constraints = load_reachability_constraints(default_constraints_dir(Path(__file__).resolve().parent))
    warmest_characterized_target_C = float(max(constraints.characterized_targets_C))
    first_interval_s = float(config.knot_times_s[1] - config.knot_times_s[0])
    first_upper_bound_C = float(theta_bounds_C[1][1])
    if first_upper_bound_C > warmest_characterized_target_C + 1.0e-12:
        return None

    settling_time_s = float(conservative_settling_time_s(warmest_characterized_target_C, constraints))
    if first_interval_s + 1.0e-12 >= settling_time_s:
        return None

    return (
        "pre-screened out before BO because the first knot interval is shorter than the conservative settling time "
        f"to the warmest characterized cold target: Δt1={first_interval_s:.1f} s, "
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
    fieldnames = [
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _write_empty_best_theta_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["knot_index", "time_s", "temperature_C"])


def _write_prescreen_placeholder_run(
    *,
    run_dir: Path,
    run_name: str,
    schedule_name: str,
    normalized_support_tau: tuple[float, ...],
    theta_bounds_C: tuple[tuple[float, float], ...],
    seed_theta_C: tuple[float, ...],
    reason: str,
    bo_settings: dict[str, float | int | str | bool],
) -> ScheduleSensitivityRunSummary:
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
        title=f"Objective History ({schedule_name}, params={len(normalized_support_tau)})",
        lines=["No BO evaluations were launched.", reason],
    )
    _write_placeholder_figure(
        analysis_dir / "feasible_vs_infeasible.png",
        title=f"Feasible vs Infeasible Suggestions ({schedule_name}, params={len(normalized_support_tau)})",
        lines=["No BO evaluations were launched.", "The run was eliminated by the deterministic pre-screen."],
    )
    _write_placeholder_figure(
        analysis_dir / "best_front_tracking.png",
        title=f"Best Front Tracking ({schedule_name}, params={len(normalized_support_tau)})",
        lines=["No admissible candidate was evaluated.", "There is no best front-tracking trace for this run."],
    )
    _write_placeholder_figure(
        analysis_dir / "plate_reference_trajectory.png",
        title=f"Plate and Reference Trajectories ({schedule_name}, params={len(normalized_support_tau)})",
        lines=["No admissible candidate was evaluated.", "There is no best plate/reference trace for this run."],
    )

    best_summary_path.write_text(
        "\n".join(
            [
                "# Open-Loop Optimization Summary",
                "",
                "## Study status",
                f"- Knot-time schedule: `{schedule_name}`",
                f"- Normalized support: `{normalized_support_tau}`",
                f"- Trajectory parameter count: `{len(normalized_support_tau)}`",
                "- Status: `pre-screened out before BO launch`",
                f"- Reason: `{reason}`",
                "",
                "## Fixed study policy",
                "- Objective: unchanged front-position tracking objective.",
                "- Admissibility: unchanged transient plus empirical hold gating before expensive simulation.",
                f"- BO bounds: `{theta_bounds_C}`",
                f"- Seed theta after applying the shared bound policy: `{seed_theta_C}`",
                "- No expensive simulation was executed for this run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_note_path.write_text(
        "\n".join(
            [
                "# Schedule Sensitivity Run Note",
                "",
                f"- Schedule family: `{schedule_name}`",
                f"- Normalized support: `{normalized_support_tau}`",
                f"- Trajectory parameter count: `{len(normalized_support_tau)}`",
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
                "method": "bayesian-optimization",
                "formulation": DEFAULT_FORMULATION,
                "status": "prescreen_eliminated",
                "reason": reason,
                "num_knots": len(normalized_support_tau),
                "knot_time_schedule": schedule_name,
                "knot_time_custom_support_tau": None,
                "knot_time_normalized_support_tau": list(normalized_support_tau),
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
                    "plate_reference_trajectory_plot": str((analysis_dir / "plate_reference_trajectory.png").resolve()),
                    "study_run_note_md": str(run_note_path.resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ScheduleSensitivityRunSummary(
        knot_time_schedule=schedule_name,
        knot_time_normalized_support_tau=normalized_support_tau,
        num_knots=len(normalized_support_tau),
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


def _load_completed_run_summary(
    *,
    schedule_name: str,
    normalized_support_tau: tuple[float, ...],
    num_knots: int,
    run_name: str,
    run_dir: Path,
    theta_bounds_C: tuple[tuple[float, float], ...],
    seed_theta_C: tuple[float, ...],
) -> ScheduleSensitivityRunSummary:
    history_csv_path = run_dir / "evaluation_history.csv"
    best_theta_path = run_dir / "best_theta_profile.csv"
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

    return ScheduleSensitivityRunSummary(
        knot_time_schedule=schedule_name,
        knot_time_normalized_support_tau=normalized_support_tau,
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


def _write_per_run_note(summary: ScheduleSensitivityRunSummary, *, bo_settings: dict[str, float | int | str | bool]) -> None:
    note_path = summary.run_dir / "study_run_note.md"
    lines = [
        "# Schedule Sensitivity Run Note",
        "",
        f"- Schedule family: `{summary.knot_time_schedule}`",
        f"- Normalized support: `{summary.knot_time_normalized_support_tau}`",
        f"- Trajectory parameter count: `{summary.num_knots}`",
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


def _best_summary_by_schedule(summaries: tuple[ScheduleSensitivityRunSummary, ...]) -> dict[str, ScheduleSensitivityRunSummary | None]:
    winners: dict[str, ScheduleSensitivityRunSummary | None] = {}
    for schedule_name in sorted(set(summary.knot_time_schedule for summary in summaries)):
        eligible = [
            summary
            for summary in summaries
            if summary.knot_time_schedule == schedule_name and math.isfinite(summary.best_objective_value)
        ]
        winners[schedule_name] = min(eligible, key=lambda item: item.best_objective_value) if eligible else None
    return winners


def _write_study_summary_csv(path: Path, summaries: tuple[ScheduleSensitivityRunSummary, ...]) -> None:
    winners = _best_summary_by_schedule(summaries)
    fieldnames = [
        "knot_time_schedule",
        "knot_time_normalized_support_tau_json",
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
        "schedule_winner_num_knots",
        "schedule_winner_best_objective_value",
        "objective_gap_vs_schedule_winner",
        "key_message",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            winner = winners.get(summary.knot_time_schedule)
            winner_objective = winner.best_objective_value if winner is not None else math.nan
            objective_gap = (
                summary.best_objective_value - winner_objective
                if winner is not None and math.isfinite(summary.best_objective_value)
                else math.nan
            )
            writer.writerow(
                {
                    "knot_time_schedule": summary.knot_time_schedule,
                    "knot_time_normalized_support_tau_json": json.dumps(list(summary.knot_time_normalized_support_tau)),
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
                    "schedule_winner_num_knots": winner.num_knots if winner is not None else "",
                    "schedule_winner_best_objective_value": winner_objective,
                    "objective_gap_vs_schedule_winner": objective_gap,
                    "key_message": summary.key_message,
                }
            )


def _write_schedule_family_summary(
    *,
    schedule_dir: Path,
    schedule_name: str,
    summaries: tuple[ScheduleSensitivityRunSummary, ...],
) -> None:
    schedule_dir.mkdir(parents=True, exist_ok=True)
    completed = [summary for summary in summaries if math.isfinite(summary.best_objective_value)]
    completed_sorted = sorted(completed, key=lambda item: item.best_objective_value)
    winner = completed_sorted[0] if completed_sorted else None

    lines = [
        "# Schedule Family Summary",
        "",
        f"Schedule family: {schedule_name}",
        "",
        "Results by trajectory parameter count:",
    ]
    for summary in sorted(summaries, key=lambda item: item.num_knots):
        best_text = f"{summary.best_objective_value:.9e}" if math.isfinite(summary.best_objective_value) else "n/a"
        lines.extend(
            [
                f"- theta_count={summary.num_knots}: status={summary.status}, best={best_text}, feasible={summary.feasible_evaluations}, infeasible={summary.infeasible_evaluations}",
                f"  normalized_support_tau={summary.knot_time_normalized_support_tau}",
                f"  theta_bounds_C={summary.theta_bounds_C}",
                f"  run_dir={summary.run_dir}",
            ]
        )
    if winner is None:
        lines.extend(["", "No completed feasible run is available for this schedule family."])
    else:
        lines.extend(
            [
                "",
                f"Winner: theta_count={winner.num_knots} with best objective {winner.best_objective_value:.9e}",
                f"Winning theta: {winner.best_theta_C}",
                f"Winning run: {winner.run_dir}",
            ]
        )
        run_settings = json.loads((winner.run_dir / "run_settings.json").read_text(encoding="utf-8"))
        artifacts = run_settings.get("artifacts", {})
        artifact_copies = {
            "objective_history_plot": schedule_dir / "best_objective_history.png",
            "plate_reference_trajectory_plot": schedule_dir / "best_plate_reference_trajectory.png",
            "front_tracking_plot": schedule_dir / "best_front_tracking.png",
        }
        for key, dest_path in artifact_copies.items():
            src_text = artifacts.get(key)
            if src_text:
                src_path = Path(src_text)
                if src_path.exists():
                    shutil.copy2(src_path, dest_path)
        best_summary_src = winner.run_dir / "best_solution_summary.md"
        if best_summary_src.exists():
            shutil.copy2(best_summary_src, schedule_dir / "best_solution_summary.md")
    (schedule_dir / "schedule_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_study_summary_txt(
    path: Path,
    *,
    study_name: str,
    schedule_families: tuple[str, ...],
    n_values: tuple[int, ...],
    summaries: tuple[ScheduleSensitivityRunSummary, ...],
    bo_settings: dict[str, float | int | str | bool],
    bound_policy_description: str,
    bo_runtime: dict[str, str],
) -> None:
    winners = _best_summary_by_schedule(summaries)
    lines = [
        "Open-loop external knot-time schedule sensitivity study",
        "",
        f"Study name: {study_name}",
        f"Schedule families: {schedule_families}",
        f"Trajectory parameter counts in this controlled comparison: {n_values}",
        "Scientific question: under explicitly selected trajectory parameter counts, does external time-schedule choice materially change BO outcomes under the same admissibility policy?",
        "",
        "Shared policy:",
        "- Objective: unchanged front-position tracking objective from the existing expensive evaluator.",
        "- Admissibility: unchanged characterization-derived transient gating plus empirical long-duration hold support before expensive simulation.",
        "- Optimized variables: theta temperature parameters only; support times remain externally fixed per run.",
        f"- BO package: bayesian-optimization {bo_runtime['package_version']} from {bo_runtime['package_path']}",
        f"- Shared BO settings: {bo_settings}",
        f"- Shared bound policy: {bound_policy_description}",
        "",
        "Per-schedule ranking:",
    ]
    winning_counts: dict[str, int] = {}
    for schedule_name in schedule_families:
        schedule_rows = sorted(
            [summary for summary in summaries if summary.knot_time_schedule == schedule_name],
            key=lambda item: item.num_knots,
        )
        winner = winners.get(schedule_name)
        for summary in schedule_rows:
            best_text = f"{summary.best_objective_value:.9e}" if math.isfinite(summary.best_objective_value) else "n/a"
            lines.append(
                f"- {schedule_name}, theta_count={summary.num_knots}: status={summary.status}, best={best_text}, normalized_support_tau={summary.knot_time_normalized_support_tau}"
            )
        if winner is None:
            lines.append(f"- {schedule_name}: no completed feasible run is available")
        else:
            winning_counts[schedule_name] = winner.num_knots
            lines.append(
                f"- {schedule_name}: winner uses theta_count={winner.num_knots} with objective {winner.best_objective_value:.9e}"
            )
        lines.append("")

    unique_winners = sorted(set(winning_counts.values()))
    if not winning_counts:
        overall_line = "No schedule family produced a completed feasible comparison."
    elif len(unique_winners) == 1:
        overall_line = f"Ranking did not flip across the tested schedule families: every completed schedule family preferred theta_count={unique_winners[0]}."
    else:
        overall_line = (
            "Ranking changed across the tested schedule families: "
            + ", ".join(f"{schedule} -> theta_count={n_value}" for schedule, n_value in winning_counts.items())
            + "."
        )
    lines.extend(["Overall reading:", f"- {overall_line}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_best_objective_by_schedule(path: Path, summaries: tuple[ScheduleSensitivityRunSummary, ...], schedule_families: tuple[str, ...], n_values: tuple[int, ...]) -> None:
    x_values = np.arange(len(schedule_families), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for num_knots in n_values:
        y_values = []
        for schedule_name in schedule_families:
            matching = [
                summary.best_objective_value
                for summary in summaries
                if summary.knot_time_schedule == schedule_name and summary.num_knots == num_knots and math.isfinite(summary.best_objective_value)
            ]
            y_values.append(matching[0] if matching else math.nan)
        ax.plot(x_values, np.asarray(y_values, dtype=np.float64), marker="o", linewidth=2.0, label=f"theta_count={num_knots}")
    ax.set_xticks(x_values, labels=list(schedule_families))
    ax.set_title("Best Objective by External Knot-Time Schedule")
    ax.set_xlabel("Schedule family")
    ax.set_ylabel("Best objective value")
    finite_values = [
        summary.best_objective_value
        for summary in summaries
        if math.isfinite(summary.best_objective_value) and summary.best_objective_value > 0.0
    ]
    if len(finite_values) >= 2 and (max(finite_values) / min(finite_values)) >= 10.0:
        ax.set_yscale("log")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled external time-schedule sensitivity support for the BO open-loop workflow."
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Deterministic study folder name.")
    parser.add_argument(
        "--out-root-dir",
        default=str(DEFAULT_OUT_ROOT_DIR),
        help="Root folder where the study folder is written.",
    )
    parser.add_argument(
        "--n-values",
        required=True,
        help="Required comma-separated trajectory parameter counts to compare in this controlled support runner.",
    )
    parser.add_argument(
        "--schedule-families",
        default=",".join(DEFAULT_SCHEDULE_FAMILIES),
        help="Comma-separated external knot-time schedule families to compare.",
    )
    parser.add_argument(
        "--custom-support-by-n",
        default=None,
        help=(
            "Optional semicolon-separated custom normalized support map used when schedule_families includes custom. "
            "Format: 'count:t0,t1,...;count:t0,t1,...'."
        ),
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
    schedule_families = _parse_schedule_families(args.schedule_families)
    custom_support_by_n = parse_normalized_support_tau_by_n_arg(args.custom_support_by_n)
    if "custom" in schedule_families:
        missing = [num_knots for num_knots in n_values if num_knots not in custom_support_by_n]
        if missing:
            raise ValueError(
                "schedule_families includes custom, but --custom-support-by-n does not define support times for "
                f"trajectory parameter counts {missing}"
            )

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
        "Interpolate the BO envelope seed template "
        f"support={BOUND_POLICY_TEMPLATE_SUPPORT_TAU}, lower={BOUND_POLICY_LOWER_TEMPERATURES_C}, "
        f"upper={BOUND_POLICY_UPPER_TEMPERATURES_C} over the actual normalized support times, "
        f"then cap every post-initial parameter upper bound at {POST_INITIAL_WARM_LIMIT_C:.1f} C so BO never searches warmer "
        "targets than the characterization-supported cold-transition range."
    )

    out_root_dir = Path(args.out_root_dir)
    study_dir = out_root_dir / args.study_name
    runs_root_dir = study_dir / "runs"
    per_schedule_root_dir = study_dir / "per_schedule"
    if study_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{study_dir} already exists. Re-run with --overwrite to replace this deterministic study folder."
            )
        shutil.rmtree(study_dir)
    runs_root_dir.mkdir(parents=True, exist_ok=True)
    per_schedule_root_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[ScheduleSensitivityRunSummary] = []
    for schedule_name in schedule_families:
        schedule_runs_root_dir = runs_root_dir / f"schedule_{schedule_name}"
        schedule_runs_root_dir.mkdir(parents=True, exist_ok=True)
        for num_knots in n_values:
            custom_support_tau = custom_support_by_n.get(num_knots) if schedule_name == "custom" else None
            config = build_problem_config(
                formulation=DEFAULT_FORMULATION,
                num_knots=int(num_knots),
                knot_time_schedule=schedule_name,
                knot_time_custom_support_tau=custom_support_tau,
            )
            normalized_support_tau = _normalized_support_tau_from_config(config)
            theta_bounds_C = _interpolated_theta_bounds_C(normalized_support_tau)
            seed_theta_C = _seed_theta_for_run(config, theta_bounds_C)
            run_name = (
                f"bo_schedule_sensitivity_{schedule_name}_params{int(num_knots)}_seed{int(args.seed)}_"
                f"init{int(args.init_points)}_iter{int(args.n_iter)}"
            )
            run_dir = schedule_runs_root_dir / run_name

            prescreen_reason = _structural_prescreen_message(config, theta_bounds_C)
            if prescreen_reason is not None:
                summary = _write_prescreen_placeholder_run(
                    run_dir=run_dir,
                    run_name=run_name,
                    schedule_name=schedule_name,
                    normalized_support_tau=normalized_support_tau,
                    theta_bounds_C=theta_bounds_C,
                    seed_theta_C=seed_theta_C,
                    reason=prescreen_reason,
                    bo_settings=bo_settings,
                )
                summaries.append(summary)
                continue

            optimization_args = [
                "--run-name",
                run_name,
                "--out-root-dir",
                str(schedule_runs_root_dir),
                "--formulation",
                DEFAULT_FORMULATION,
                "--num-knots",
                str(int(num_knots)),
                "--knot-time-schedule",
                schedule_name,
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
            if custom_support_tau is not None:
                optimization_args.extend(
                    [
                        "--knot-time-custom-support-tau",
                        ",".join(f"{float(value):.12g}" for value in custom_support_tau),
                    ]
                )
            run_open_loop_optimization_main(optimization_args)
            summary = _load_completed_run_summary(
                schedule_name=schedule_name,
                normalized_support_tau=normalized_support_tau,
                num_knots=int(num_knots),
                run_name=run_name,
                run_dir=run_dir,
                theta_bounds_C=theta_bounds_C,
                seed_theta_C=seed_theta_C,
            )
            _write_per_run_note(summary, bo_settings=bo_settings)
            summaries.append(summary)

    summary_tuple = tuple(sorted(summaries, key=lambda item: (item.knot_time_schedule, item.num_knots)))
    study_settings = {
        "study_name": args.study_name,
        "schedule_families": list(schedule_families),
        "n_values": list(n_values),
        "scope_note": "comparison support only; the active single-run workflow is run_open_loop_optimization.py",
        "custom_support_by_n": {str(key): list(value) for key, value in custom_support_by_n.items()},
        "shared_bo_settings": bo_settings,
        "bound_policy_description": bound_policy_description,
        "bo_runtime": bo_runtime,
        "runs_root_dir": str(runs_root_dir.resolve()),
        "per_schedule_root_dir": str(per_schedule_root_dir.resolve()),
    }
    (study_dir / "study_settings.json").write_text(json.dumps(study_settings, indent=2), encoding="utf-8")
    _write_study_summary_csv(study_dir / "study_summary.csv", summary_tuple)
    _write_study_summary_txt(
        study_dir / "study_summary.txt",
        study_name=args.study_name,
        schedule_families=schedule_families,
        n_values=n_values,
        summaries=summary_tuple,
        bo_settings=bo_settings,
        bound_policy_description=bound_policy_description,
        bo_runtime=bo_runtime,
    )
    _plot_best_objective_by_schedule(
        study_dir / "best_objective_by_schedule.png",
        summary_tuple,
        schedule_families=schedule_families,
        n_values=n_values,
    )
    for schedule_name in schedule_families:
        schedule_rows = tuple(summary for summary in summary_tuple if summary.knot_time_schedule == schedule_name)
        _write_schedule_family_summary(
            schedule_dir=per_schedule_root_dir / schedule_name,
            schedule_name=schedule_name,
            summaries=schedule_rows,
        )

    print(f"Schedule-sensitivity study completed: {study_dir}")
    winners = _best_summary_by_schedule(summary_tuple)
    for schedule_name in schedule_families:
        winner = winners.get(schedule_name)
        if winner is None:
            print(f"{schedule_name}: no completed feasible run")
            continue
        print(
            f"{schedule_name}: winner_theta_count={winner.num_knots}, best={winner.best_objective_value:.9e}, "
            f"support_tau={winner.knot_time_normalized_support_tau}"
        )


if __name__ == "__main__":
    main()
