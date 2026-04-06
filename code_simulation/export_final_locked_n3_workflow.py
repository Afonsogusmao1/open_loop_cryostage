#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryostage_model import DEFAULT_CRYOSTAGE_PARAMS, simulate_plate_temperature
from open_loop_problem import build_front_reference, build_reference_profile_from_theta, load_front_csv
from reachability_constraints import check_piecewise_linear_trajectory_admissibility
from run_open_loop_optimization import build_problem_config


DEFAULT_SELECTED_RUN_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "open_loop_bayesian_optimization"
    / "fixed_n_bo_confirmation_k3_k4_seed29_init4_iter8"
    / "runs"
    / "bo_compare_full_process_k3_seed29_init4_iter8"
)
DEFAULT_INITIAL_STUDY_SUMMARY = (
    Path(__file__).resolve().parent
    / "results"
    / "open_loop_bayesian_optimization"
    / "fixed_n_bo_comparison_full_process_seed17_init4_iter8"
    / "study_summary.csv"
)
DEFAULT_CONFIRMATION_STUDY_SUMMARY = (
    Path(__file__).resolve().parent
    / "results"
    / "open_loop_bayesian_optimization"
    / "fixed_n_bo_confirmation_k3_k4_seed29_init4_iter8"
    / "study_summary.csv"
)
DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "open_loop_final_workflow"
    / "locked_n3_seed29_init4_iter8"
)


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the locked final N=3 workflow artifacts.")
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SELECTED_RUN_DIR), help="Selected N=3 run to lock as final.")
    parser.add_argument(
        "--initial-study-summary-csv",
        default=str(DEFAULT_INITIAL_STUDY_SUMMARY),
        help="CSV summary from the first fixed-N BO study.",
    )
    parser.add_argument(
        "--confirmation-study-summary-csv",
        default=str(DEFAULT_CONFIRMATION_STUDY_SUMMARY),
        help="CSV summary from the focused N=3 vs N=4 confirmation rerun.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Final locked workflow output folder.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing final workflow folder.")
    return parser.parse_args(argv)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_key_value_text(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _load_theta_profile(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    knot_times_s: list[float] = []
    theta_C: list[float] = []
    for row in _read_csv_rows(path):
        knot_times_s.append(float(row["time_s"]))
        theta_C.append(float(row["temperature_C"]))
    return tuple(knot_times_s), tuple(theta_C)


def _load_study_rows_by_n(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["num_knots"]): row for row in _read_csv_rows(path)}


def _plot_front_tracking(*, time_s: np.ndarray, z_front_m: np.ndarray, z_ref_m: np.ndarray, out_path: Path) -> None:
    valid_mask = np.isfinite(z_ref_m)
    fig, ax = plt.subplots()
    ax.plot(time_s, 1000.0 * z_front_m, linewidth=2.0, label=r"$z_{front}$")
    ax.plot(time_s[valid_mask], 1000.0 * z_ref_m[valid_mask], "--", linewidth=2.0, label=r"$z_{ref}$")
    ax.set_title("Locked N=3 Front Tracking")
    ax.set_xlabel("Time Since Fill (s)")
    ax.set_ylabel("Front Position (mm)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_plate_reference(*, time_s: np.ndarray, T_ref_C: np.ndarray, T_plate_C: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, T_ref_C, linewidth=2.0, label=r"$T_{ref}$")
    ax.plot(time_s, T_plate_C, linewidth=2.0, label=r"$T_{plate,model}$")
    ax.set_title("Locked N=3 Plate and Reference Trajectories")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _write_reference_profile_csv(*, out_path: Path, time_s: np.ndarray, T_ref_C: np.ndarray, T_plate_C: np.ndarray) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "T_ref_C", "T_plate_model_C"])
        for ti, tref, tplate in zip(time_s, T_ref_C, T_plate_C, strict=True):
            writer.writerow([float(ti), float(tref), float(tplate)])


def _write_admissibility_summary_md(out_path: Path, report) -> None:
    lines = [
        "# Final Admissibility Summary",
        "",
        f"- Admissible: `{report.is_admissible}`",
        f"- Monotone cooling hard enforced: `{report.monotone_cooling_hard_enforced}`",
        f"- Warming supported: `{report.warming_supported}`",
        f"- Constraints directory: `{report.constraints_dir}`",
        f"- Top-level reasons: `{list(report.reasons)}`",
        "",
        "## Segment checks",
    ]
    for segment in report.segment_results:
        lines.extend(
            [
                f"### Segment {segment.segment_index}",
                f"- Time window: `{segment.t_start_s:.1f} -> {segment.t_end_s:.1f} s`",
                f"- Temperature window: `{segment.T_start_C:.2f} -> {segment.T_end_C:.2f} C`",
                f"- Admissible: `{segment.is_admissible}`",
                f"- Cooling drop requested: `{segment.requested_cooling_drop_C:.2f} C`",
                f"- Hold-like after segment: `{segment.hold_like_after_segment}` with requested hold `{segment.requested_hold_duration_s:.1f} s`",
                f"- Conservative first-entry time: `{segment.conservative_first_entry_time_s}`",
                f"- Conservative settling time: `{segment.conservative_settling_time_s}`",
                f"- Empirical hold basis target: `{segment.empirical_hold_basis_target_C}`",
                f"- Empirical supported hold duration: `{segment.conservative_supported_hold_duration_s}`",
                f"- Reasons: `{list(segment.reasons)}`",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_solution_summary(
    *,
    out_path: Path,
    source_run_dir: Path,
    selected_theta_C: tuple[float, ...],
    knot_times_s: tuple[float, ...],
    objective_summary: dict[str, str],
    run_settings: dict,
    evaluation_metadata: dict,
    initial_study_rows: dict[int, dict[str, str]],
    confirmation_study_rows: dict[int, dict[str, str]],
) -> None:
    lines = [
        "# Final Selected Open-Loop Solution",
        "",
        "## Locked source",
        f"- Locked source run: `{source_run_dir}`",
        f"- Selected evaluation: `{evaluation_metadata['case_name']}` (evaluation `{evaluation_metadata['evaluation_index']}`)",
        f"- Selected knot count: `{run_settings['num_knots']}`",
        f"- Knot times: `{knot_times_s}`",
        f"- Selected theta: `{selected_theta_C}`",
        f"- Best objective value: `{float(evaluation_metadata['objective_value']):.9e}`",
        f"- Freeze completion reached: `{objective_summary['freeze_completion_reached']}` at `{objective_summary['freeze_completion_time_s']} s`",
        "",
        "## Selection rationale",
        f"- First fixed-N study best objective at `N=3`: `{float(initial_study_rows[3]['best_objective_value']):.9e}`",
        f"- First fixed-N study best objective at `N=4`: `{float(initial_study_rows[4]['best_objective_value']):.9e}`",
        f"- Confirmation rerun best objective at `N=3`: `{float(confirmation_study_rows[3]['best_objective_value']):.9e}`",
        f"- Confirmation rerun best objective at `N=4`: `{float(confirmation_study_rows[4]['best_objective_value']):.9e}`",
        "- Final choice: carry forward `N=3` because the confirmation rerun removed the earlier small edge for `N=4`, while `N=3` reproduced the same best objective across two explicit BO seeds and keeps the simpler model order.",
        "",
        "## Locked optimization settings",
        f"- Method: `{run_settings['method']}`",
        f"- BO settings: `{run_settings['bo']}`",
        f"- Bounds policy output for the locked run: `{run_settings['theta_bounds_C']}`",
        f"- Infeasible objective penalty: `{run_settings['infeasible_objective_penalty']}`",
        f"- Raw candidate equals selected theta: `{tuple(evaluation_metadata['raw_candidate'])}`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_workflow_summary(
    *,
    out_path: Path,
    out_dir: Path,
    source_run_dir: Path,
    run_settings: dict,
) -> None:
    bo = run_settings["bo"]
    lines = [
        "# Final Workflow Summary",
        "",
        "## Chosen workflow",
        "- Final chosen knot count: `N = 3`.",
        f"- Locked source run: `{source_run_dir}`.",
        f"- Final artifact bundle: `{out_dir}`.",
        "- Scientific objective: unchanged front-position tracking objective from the existing expensive evaluator.",
        "- Admissibility: unchanged transient reachability from cryostage characterization plus long-duration hold feasibility from freezing-run plate telemetry.",
        "",
        "## Why N = 3",
        "- The broad fixed-N study showed `N=3` and `N=4` were very close, with worse outcomes for higher counts and `N=7` screened out.",
        "- The focused confirmation rerun between `N=3` and `N=4` with a new seed selected `N=3`.",
        "- `N=3` reproduced the same best objective across two explicit seeds, while `N=4` did not reproduce its earlier advantage.",
        "",
        "## Locked BO settings",
        f"- Method: `{run_settings['method']}`",
        f"- Seed: `{bo['random_seed']}`",
        f"- Init points: `{bo['init_points']}`",
        f"- BO iterations: `{bo['n_iter']}`",
        f"- Acquisition: `{bo['acquisition_kind']}` with `kappa={bo['acquisition_kappa']}` and `xi={bo['acquisition_xi']}`",
        f"- Seed with theta0: `{bo['seed_with_theta0']}`",
        f"- Bound policy output for `N=3`: `{run_settings['theta_bounds_C']}`",
        f"- Infeasible-candidate policy: explicit pre-simulation admissibility check with deterministic objective penalty `{run_settings['infeasible_objective_penalty']}` and no expensive simulation on failure.",
        "",
        "## Scripts to run",
        "- Single `N=3` optimization run: `/home/fenics/shared/Open_loop/code_simulation/run_open_loop_optimization.py`",
        "  Example settings: `--formulation full_process_article --num-knots 3 --method bayesian-optimization --seed 29 --init-points 4 --n-iter 8 --theta-bounds=-0.5:0.0,-15.0:-8.0,-20.0:-14.0`.",
        "- Fixed-N BO comparison study: `/home/fenics/shared/Open_loop/code_simulation/run_open_loop_fixed_n_bo_study.py`",
        "- Reachability/admissibility diagnostics: `/home/fenics/shared/Open_loop/code_simulation/run_reachability_diagnostics.py`",
        "- Final workflow export: `/home/fenics/shared/Open_loop/code_simulation/export_final_locked_n3_workflow.py`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_matplotlib()

    source_run_dir = Path(args.source_run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    initial_study_summary_csv = Path(args.initial_study_summary_csv).resolve()
    confirmation_study_summary_csv = Path(args.confirmation_study_summary_csv).resolve()

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_settings = json.loads((source_run_dir / "run_settings.json").read_text(encoding="utf-8"))
    if int(run_settings["num_knots"]) != 3:
        raise ValueError("The locked final workflow export expects a selected N=3 run.")

    knot_times_s, theta_C = _load_theta_profile(source_run_dir / "best_theta_profile.csv")
    selected_eval_dir = Path(json.loads((source_run_dir / "best" / "evaluation_metadata.json").read_text(encoding="utf-8"))["out_dir"])
    evaluation_metadata = json.loads((source_run_dir / "best" / "evaluation_metadata.json").read_text(encoding="utf-8"))
    objective_summary_src = source_run_dir / "best" / f"{evaluation_metadata['case_name']}_objective_summary.txt"
    front_csv_src = source_run_dir / "best" / f"{evaluation_metadata['case_name']}_front.csv"
    objective_summary = _read_key_value_text(objective_summary_src)

    config = build_problem_config(formulation=str(run_settings["formulation"]), num_knots=int(run_settings["num_knots"]))
    reference_profile = build_reference_profile_from_theta(theta_C, config)
    admissibility_report = check_piecewise_linear_trajectory_admissibility(
        knot_times_s,
        theta_C,
        constraints_dir=config.characterization_constraints_dir,
        require_monotone_nonincreasing=config.require_monotone_nonincreasing,
    )
    time_s = config.cryostage_time_grid_s()
    T_ref_C = np.asarray([float(reference_profile(float(ti))) for ti in time_s], dtype=np.float64)
    T_plate0_C = float(config.solver_kwargs["bcs"].T_room_C)
    T_plate_C = simulate_plate_temperature(time_s, reference_profile, DEFAULT_CRYOSTAGE_PARAMS, T_plate0_C)
    front = load_front_csv(front_csv_src)
    z_ref_m = build_front_reference(front.time_since_fill_s, front.z_front_m, config)

    initial_study_rows = _load_study_rows_by_n(initial_study_summary_csv)
    confirmation_study_rows = _load_study_rows_by_n(confirmation_study_summary_csv)

    final_theta_csv = out_dir / "final_selected_theta_profile.csv"
    shutil.copy2(source_run_dir / "best_theta_profile.csv", final_theta_csv)
    final_objective_summary_txt = out_dir / "final_objective_summary.txt"
    shutil.copy2(objective_summary_src, final_objective_summary_txt)
    final_front_csv = out_dir / "final_front_trajectory.csv"
    shutil.copy2(front_csv_src, final_front_csv)
    shutil.copy2(source_run_dir / "evaluation_history.csv", out_dir / "locked_run_evaluation_history.csv")
    shutil.copy2(source_run_dir / "analysis" / "objective_history.png", out_dir / "optimization_objective_history.png")
    shutil.copy2(source_run_dir / "analysis" / "feasible_vs_infeasible.png", out_dir / "optimization_feasibility_history.png")
    _write_reference_profile_csv(out_path=out_dir / "final_reference_profile.csv", time_s=time_s, T_ref_C=T_ref_C, T_plate_C=T_plate_C)
    _plot_front_tracking(time_s=front.time_since_fill_s, z_front_m=front.z_front_m, z_ref_m=z_ref_m, out_path=out_dir / "front_tracking_plot.png")
    _plot_plate_reference(time_s=time_s, T_ref_C=T_ref_C, T_plate_C=T_plate_C, out_path=out_dir / "plate_reference_trajectory.png")
    (out_dir / "final_admissibility_report.json").write_text(json.dumps(admissibility_report.to_dict(), indent=2), encoding="utf-8")
    _write_admissibility_summary_md(out_dir / "final_admissibility_summary.md", admissibility_report)
    _write_final_solution_summary(
        out_path=out_dir / "final_selected_solution_summary.md",
        source_run_dir=source_run_dir,
        selected_theta_C=theta_C,
        knot_times_s=knot_times_s,
        objective_summary=objective_summary,
        run_settings=run_settings,
        evaluation_metadata=evaluation_metadata,
        initial_study_rows=initial_study_rows,
        confirmation_study_rows=confirmation_study_rows,
    )
    _write_workflow_summary(
        out_path=out_dir / "final_workflow_summary.md",
        out_dir=out_dir,
        source_run_dir=source_run_dir,
        run_settings=run_settings,
    )
    (out_dir / "locked_run_manifest.json").write_text(
        json.dumps(
            {
                "locked_source_run_dir": str(source_run_dir),
                "locked_source_evaluation_dir": str(selected_eval_dir),
                "selected_theta_C": list(theta_C),
                "knot_times_s": list(knot_times_s),
                "run_settings": run_settings,
                "objective_summary_source": str(objective_summary_src),
                "front_csv_source": str(front_csv_src),
                "initial_study_summary_csv": str(initial_study_summary_csv),
                "confirmation_study_summary_csv": str(confirmation_study_summary_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Locked final N=3 workflow exported to: {out_dir}")
    print(f"Selected theta: {theta_C}")
    print(f"Selected objective: {float(evaluation_metadata['objective_value']):.9e}")


if __name__ == "__main__":
    main()
