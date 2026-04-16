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

from cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
from open_loop_cascade import build_plate_temperature_response
from open_loop_bayesian_optimizer import BayesianOptimizationConfig, bayes_opt_runtime_details
from open_loop_optimizer import OpenLoopOptimizationResult, optimize_open_loop_theta
from open_loop_problem import (
    build_front_reference,
    build_reference_profile_from_theta,
    front_definition_contract_summary,
    front_reference_contract_summary,
    load_front_csv,
)
from solver import SUPPORTED_FRONT_DEFINITION_MODES
from open_loop_workflow_config import (
    DEFAULT_FORMULATION,
    OpenLoopProblemConfig,
    DEFAULT_KNOT_TIME_SCHEDULE,
    SUPPORTED_KNOT_TIME_SCHEDULES,
    build_external_knot_times_s,
    build_knot_time_normalized_support_tau,
    build_problem_config,
    default_theta0_for_config,
    parse_normalized_support_tau_arg,
    parse_normalized_support_tau_by_n_arg,
)


DEFAULT_METHOD = "bayesian-optimization"
DEFAULT_RUN_NAME = "bo_full_process_active"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"
ACTIVE_WORKFLOW_CHAIN = (
    "theta -> T_ref(t) -> cryostage model / inner PID response -> "
    "T_plate(t) -> freezing solver -> z_front(t) -> J(theta)"
)
DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY = 1.0e6


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


_default_theta0_for_config = default_theta0_for_config
_parse_normalized_support_tau_arg = parse_normalized_support_tau_arg
_parse_normalized_support_tau_by_n_arg = parse_normalized_support_tau_by_n_arg


def _parse_theta_arg(raw_theta: str | None, *, num_knots: int) -> tuple[float, ...] | None:
    if raw_theta is None:
        return None
    parts = [part.strip() for part in raw_theta.split(",")]
    if len(parts) != int(num_knots):
        raise ValueError(
            "--theta0 must contain exactly one comma-separated value per trajectory parameter "
            f"({num_knots} expected, got {len(parts)})"
        )
    values = np.asarray([float(part) for part in parts], dtype=np.float64)
    if values.ndim != 1 or values.size != int(num_knots) or not np.all(np.isfinite(values)):
        raise ValueError("--theta0 must contain only finite 1D values")
    return tuple(float(value) for value in values)


def _resolved_run_name(
    *,
    requested_run_name: str,
    config: OpenLoopProblemConfig,
    formulation: str,
    knot_time_schedule: str,
) -> str:
    if formulation == DEFAULT_FORMULATION and requested_run_name == DEFAULT_RUN_NAME:
        resolved = DEFAULT_RUN_NAME
        if str(knot_time_schedule) != DEFAULT_KNOT_TIME_SCHEDULE:
            resolved += f"_sched{str(knot_time_schedule).replace('_', '')}"
        return resolved
    return requested_run_name


def _theta_satisfies_active_constraints(theta, config: OpenLoopProblemConfig) -> bool:
    try:
        build_reference_profile_from_theta(theta, config)
    except Exception:
        return False
    return True


def _parse_theta_bounds_arg(raw_theta_bounds: str | None, *, num_knots: int) -> tuple[tuple[float, float], ...] | None:
    if raw_theta_bounds is None:
        return None
    raw_pairs = [part.strip() for part in raw_theta_bounds.split(",")]
    if len(raw_pairs) != int(num_knots):
        raise ValueError(
            "--theta-bounds must contain exactly one lower:upper pair per trajectory parameter "
            f"({num_knots} expected, got {len(raw_pairs)})"
        )
    bounds: list[tuple[float, float]] = []
    for idx, raw_pair in enumerate(raw_pairs):
        if ":" not in raw_pair:
            raise ValueError(f"theta bound {idx} must be written as lower:upper")
        lower_text, upper_text = (piece.strip() for piece in raw_pair.split(":", 1))
        lower_C = float(lower_text)
        upper_C = float(upper_text)
        if not (math.isfinite(lower_C) and math.isfinite(upper_C) and lower_C < upper_C):
            raise ValueError(f"theta bound {idx} must satisfy lower < upper with finite values")
        bounds.append((lower_C, upper_C))
    return tuple(bounds)


def _plot_objective_history(result: OpenLoopOptimizationResult, out_path: Path) -> None:
    indices = np.asarray([entry.evaluation_index for entry in result.history], dtype=np.float64)
    objective_values = np.asarray([entry.objective_value for entry in result.history], dtype=np.float64)
    feasible_mask = np.asarray([entry.is_valid for entry in result.history], dtype=bool)

    fig, ax = plt.subplots()
    if np.any(feasible_mask):
        ax.plot(
            indices[feasible_mask],
            objective_values[feasible_mask],
            marker="o",
            linewidth=1.5,
            markersize=4.0,
            color="#2A6F97",
            label="feasible evaluations",
        )
    if np.any(~feasible_mask):
        ax.scatter(
            indices[~feasible_mask],
            objective_values[~feasible_mask],
            marker="x",
            color="#C44536",
            label="infeasible / failed",
        )
    ax.scatter(
        [result.best_evaluation_index],
        [result.best_objective_value],
        color="#2A9D8F",
        s=60,
        zorder=3,
        label="best feasible",
    )
    finite_values = objective_values[np.isfinite(objective_values) & (objective_values > 0.0)]
    if finite_values.size >= 2 and float(np.max(finite_values) / np.min(finite_values)) >= 10.0:
        ax.set_yscale("log")
    ax.set_title("Objective History")
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Returned Objective Value")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_feasibility_timeline(result: OpenLoopOptimizationResult, out_path: Path) -> None:
    status_to_y = {"evaluation_error": -1.0, "infeasible": 0.0, "feasible": 1.0}
    status_to_color = {"evaluation_error": "#6D597A", "infeasible": "#C44536", "feasible": "#2A9D8F"}

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for status in ("evaluation_error", "infeasible", "feasible"):
        x_values = [entry.evaluation_index for entry in result.history if entry.feasibility_status == status]
        y_values = [status_to_y[status]] * len(x_values)
        if x_values:
            ax.scatter(x_values, y_values, color=status_to_color[status], label=status, s=45)
    ax.set_title("Feasible vs Infeasible Suggestions")
    ax.set_xlabel("Evaluation Index")
    ax.set_yticks([-1.0, 0.0, 1.0], labels=["evaluation_error", "infeasible", "feasible"])
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_best_front_tracking(*, best_front_path: Path, config: OpenLoopProblemConfig, out_path: Path) -> None:
    front = load_front_csv(best_front_path)
    z_ref_m = build_front_reference(front.time_since_fill_s, front.z_front_m, config)
    valid_mask = np.isfinite(z_ref_m)

    fig, ax = plt.subplots()
    ax.plot(front.time_since_fill_s, 1000.0 * front.z_front_m, linewidth=2.0, label=r"$z_{front}$")
    ax.plot(front.time_since_fill_s[valid_mask], 1000.0 * z_ref_m[valid_mask], "--", linewidth=2.0, label=r"$z_{ref}$")
    ax.set_title("Best-Run Front Tracking")
    ax.set_xlabel("Time Since Fill (s)")
    ax.set_ylabel("Front Position (mm)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_best_plate_reference_trajectory(
    *,
    theta_C: tuple[float, ...],
    config: OpenLoopProblemConfig,
    out_path: Path,
) -> None:
    reference_profile = build_reference_profile_from_theta(theta_C, config)
    plate_response = build_plate_temperature_response(
        time_s=config.cryostage_time_grid_s(),
        T_ref_profile_C=reference_profile,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        bcs=config.solver_kwargs.get("bcs"),
    )

    fig, ax = plt.subplots()
    ax.plot(plate_response.cryostage_time_s, plate_response.T_ref_C, linewidth=2.0, label=r"$T_{ref}$")
    ax.plot(plate_response.cryostage_time_s, plate_response.T_plate_C, linewidth=2.0, label=r"$T_{plate,model}$")
    ax.set_title("Best-Run Plate and Reference Trajectories")
    ax.set_xlabel("Time Since Fill (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _write_best_theta_profile_csv(out_path: Path, *, knot_times_s: tuple[float, ...], theta_C: tuple[float, ...]) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["knot_index", "time_s", "temperature_C"])
        for idx, (time_s, temperature_C) in enumerate(zip(knot_times_s, theta_C, strict=True)):
            writer.writerow([idx, float(time_s), float(temperature_C)])


def _write_best_solution_summary(
    *,
    out_path: Path,
    result: OpenLoopOptimizationResult,
    config: OpenLoopProblemConfig,
    theta0: tuple[float, ...],
    theta_bounds_C: tuple[tuple[float, float], ...] | None,
    knot_time_schedule: str,
    knot_time_normalized_support_tau: tuple[float, ...],
    infeasible_objective_penalty: float,
    best_theta_profile_csv: Path,
    history_plot_path: Path,
    feasibility_plot_path: Path,
    front_plot_path: Path,
    plate_reference_plot_path: Path,
    bo_runtime: dict[str, str] | None,
    smoke_test_note: str | None,
) -> None:
    total_evaluations = len(result.history)
    feasible_evaluations = sum(1 for entry in result.history if entry.is_valid)
    infeasible_evaluations = sum(1 for entry in result.history if entry.feasibility_status == "infeasible")
    evaluation_errors = sum(1 for entry in result.history if entry.feasibility_status == "evaluation_error")
    expensive_runs = sum(1 for entry in result.history if entry.expensive_simulation_executed)
    best_theta_is_valid = _theta_satisfies_active_constraints(result.best_theta, config)
    reference_contract = front_reference_contract_summary(config)
    front_definition_contract = front_definition_contract_summary(config)

    lines = [
        "# Open-Loop Optimization Summary",
        "",
        "## Optimizer role",
        f"- Main optimizer path: `{result.method}`.",
        f"- Active workflow chain: `{ACTIVE_WORKFLOW_CHAIN}`.",
        "- Scientific objective: unchanged front-position tracking objective from `evaluate_open_loop_objective`.",
        f"- Trajectory parameterization: `{len(config.knot_times_s)}` temperature parameters on externally fixed support times `{list(config.knot_times_s)}`.",
        f"- External time schedule: `{knot_time_schedule}` with normalized support `{list(knot_time_normalized_support_tau)}`.",
        f"- Optimized variables: `theta` temperature parameters only; support times are run settings, not optimized in this runner.",
        "",
        "## Infeasible-candidate policy",
        "- Every candidate is first checked through `build_reference_profile_from_theta`, which enforces bounds, monotone cooling, and the active admissibility layer before any expensive simulation starts.",
        f"- If that pre-check fails, the evaluation is logged as `infeasible`, the expensive simulation is skipped, and a deterministic penalty objective of `{float(infeasible_objective_penalty):.6g}` is returned.",
        "- Feasible candidates proceed into the expensive cascade and keep the unchanged scientific objective value.",
        "- Any post-admissibility evaluation failure is logged separately as `evaluation_error`.",
        "",
        "## Run settings",
        f"- Initial theta: `{theta0}`",
        f"- Best theta: `{result.best_theta}`",
        f"- Best theta satisfies active constraints: `{best_theta_is_valid}`",
        f"- Active T_ref bounds: `{config.T_ref_bounds_C}`",
        f"- Active support times (s): `{list(config.knot_times_s)}`",
        f"- Front definition mode: `{front_definition_contract['front_definition_mode']}`",
        f"- Front threshold: `{front_definition_contract['front_threshold_C']:.9e}` C",
        f"- Front reference mode: `{reference_contract['front_reference_mode']}`",
        f"- H_fill_m: `{reference_contract['H_fill_m']:.9e}`",
        f"- Target start/end/duration (s): `{reference_contract['target_start_s']:.6f}` / `{reference_contract['target_end_s']:.6f}` / `{reference_contract['target_duration_s']:.6f}`",
        f"- Implied target front speed: `{reference_contract['implied_target_front_speed_m_per_s']:.9e}` m/s (`{1.0e3 * reference_contract['implied_target_front_speed_m_per_s']:.9e}` mm/s)",
        f"- Objective weights: tracking=`{config.tracking_weight:.6f}`, smoothness=`{config.smoothness_weight:.6f}`, completion=`{config.completion_weight:.6f}`, terminal=`{config.terminal_weight:.6f}`",
        f"- BO theta bounds: `{theta_bounds_C}`",
        f"- Evaluations: total=`{total_evaluations}`, feasible=`{feasible_evaluations}`, infeasible=`{infeasible_evaluations}`, evaluation_error=`{evaluation_errors}`, expensive_simulation_executed=`{expensive_runs}`",
        f"- Best objective value: `{result.best_objective_value:.9e}` at evaluation `{result.best_evaluation_index}`",
        f"- Optimizer termination: success=`{result.success}`, status=`{result.status}`, message=`{result.message}`",
    ]
    if bo_runtime is not None:
        lines.extend(
            [
                f"- BO package: `bayesian-optimization {bo_runtime['package_version']}`",
                f"- BO package path: `{bo_runtime['package_path']}`",
            ]
        )
    if smoke_test_note:
        lines.extend(["", "## Smoke test note", f"- {smoke_test_note}"])
    lines.extend(
        [
            "",
            "## Artifacts",
            f"- Evaluation history CSV: `{result.history_csv_path}`",
            f"- Best evaluation folder copy: `{result.best_dir}`",
            f"- Best theta profile CSV: `{best_theta_profile_csv}`",
            f"- Objective history plot: `{history_plot_path}`",
            f"- Feasibility timeline plot: `{feasibility_plot_path}`",
            f"- Best front tracking plot: `{front_plot_path}`",
            f"- Best plate/reference trajectory plot: `{plate_reference_plot_path}`",
            "",
            "## Methodological status",
            "- This run records one executable BO trajectory-design setting. A single run does not close the trajectory-parameterization question; use the Phase 5 binary parameterization study runner for carry-forward evidence.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the active BO-based open-loop trajectory design workflow.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder stem inside the output root.")
    parser.add_argument(
        "--out-root-dir",
        default=str(DEFAULT_OUT_ROOT_DIR),
        help="Root folder where optimization runs are written.",
    )
    parser.add_argument(
        "--formulation",
        default=DEFAULT_FORMULATION,
        choices=("full_process_article", "legacy_exploratory"),
        help="Problem formulation to run.",
    )
    parser.add_argument(
        "--front-definition-mode",
        default=None,
        choices=SUPPORTED_FRONT_DEFINITION_MODES,
        help="Temperature-threshold rule used to extract z_front(t); defaults to the workflow config.",
    )
    parser.add_argument(
        "--num-knots",
        type=int,
        default=None,
        help="Runtime trajectory parameter count for T_ref on the externally fixed support; required for full_process_article.",
    )
    parser.add_argument(
        "--knot-time-schedule",
        default=DEFAULT_KNOT_TIME_SCHEDULE,
        choices=SUPPORTED_KNOT_TIME_SCHEDULES,
        help="Externally fixed knot-time schedule family for full_process_article.",
    )
    parser.add_argument(
        "--knot-time-custom-support-tau",
        default=None,
        help="Optional comma-separated normalized support times in [0,1] used when --knot-time-schedule=custom.",
    )
    parser.add_argument(
        "--theta0",
        default=None,
        help="Optional comma-separated initial theta temperatures. Defaults to the canonical seed profile resampled to the active support.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="Optimizer backend. The active default is 'bayesian-optimization'; scipy.optimize.minimize method names remain for legacy compatibility.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing run folder before running.")
    parser.add_argument(
        "--infeasible-objective-penalty",
        type=float,
        default=DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY,
        help="Deterministic objective value returned when the active admissibility pre-check fails.",
    )

    parser.add_argument("--maxiter", type=int, default=15, help="Maximum optimizer iterations for the legacy scipy path.")
    parser.add_argument("--maxfev", type=int, default=40, help="Maximum objective evaluations for the legacy scipy path.")
    parser.add_argument("--xatol", type=float, default=0.25, help="Absolute theta convergence tolerance for the legacy scipy path.")
    parser.add_argument("--fatol", type=float, default=1.0e-9, help="Absolute objective convergence tolerance for the legacy scipy path.")

    parser.add_argument("--seed", type=int, default=17, help="Random seed for the BO backend.")
    parser.add_argument("--init-points", type=int, default=4, help="Number of initial random BO suggestions.")
    parser.add_argument("--n-iter", type=int, default=12, help="Number of BO acquisition-guided suggestions.")
    parser.add_argument(
        "--acq-kind",
        default="ucb",
        choices=("ucb", "ei", "poi"),
        help="BO acquisition function kind.",
    )
    parser.add_argument("--kappa", type=float, default=2.576, help="BO acquisition kappa parameter.")
    parser.add_argument("--xi", type=float, default=0.0, help="BO acquisition xi parameter.")
    parser.add_argument(
        "--theta-bounds",
        default=None,
        help="Optional comma-separated lower:upper bounds with one pair per trajectory parameter.",
    )
    parser.add_argument(
        "--no-seed-theta0",
        action="store_true",
        help="Do not register the canonical initial theta as the first BO observation.",
    )
    parser.add_argument(
        "--smoke-test-note",
        default=None,
        help="Optional line included in the markdown summary to document the smoke-test scope.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_matplotlib()

    if args.formulation == DEFAULT_FORMULATION and args.num_knots is None:
        raise ValueError("--num-knots is required for the active full_process_article workflow")
    if args.knot_time_custom_support_tau is not None and args.num_knots is None:
        raise ValueError("--knot-time-custom-support-tau requires an explicit --num-knots value")

    knot_time_custom_support_tau = None
    if args.knot_time_custom_support_tau is not None:
        knot_time_custom_support_tau = _parse_normalized_support_tau_arg(
            args.knot_time_custom_support_tau,
            num_knots=int(args.num_knots),
        )
    config = build_problem_config(
        formulation=args.formulation,
        num_knots=args.num_knots,
        knot_time_schedule=args.knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
        front_definition_mode=args.front_definition_mode,
    )
    knot_time_normalized_support_tau = tuple(
        float(value) / float(config.horizon_s) for value in config.knot_times_s
    )
    theta0 = _parse_theta_arg(args.theta0, num_knots=len(config.knot_times_s))
    if theta0 is None:
        theta0 = _default_theta0_for_config(config)
    run_name = _resolved_run_name(
        requested_run_name=args.run_name,
        config=config,
        formulation=args.formulation,
        knot_time_schedule=args.knot_time_schedule,
    )
    out_root_dir = Path(args.out_root_dir)
    run_dir = out_root_dir / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{run_dir} already exists. Re-run with --overwrite to replace this deterministic run folder."
            )
        shutil.rmtree(run_dir)

    theta_bounds_C = _parse_theta_bounds_arg(args.theta_bounds, num_knots=len(config.knot_times_s))
    bayesopt_config = BayesianOptimizationConfig(
        random_seed=int(args.seed),
        init_points=int(args.init_points),
        n_iter=int(args.n_iter),
        acquisition_kind=str(args.acq_kind),
        acquisition_kappa=float(args.kappa),
        acquisition_xi=float(args.xi),
        theta_bounds_C=theta_bounds_C,
        seed_with_theta0=not bool(args.no_seed_theta0),
    )

    bcs = config.solver_kwargs["bcs"]
    T_fill_C = float(config.solver_kwargs["T_fill_C"])

    print(
        "Using canonical cryostage parameters "
        f"(tau_s={DEFAULT_CRYOSTAGE_PARAMS.tau_s:.6f}, "
        f"gain={DEFAULT_CRYOSTAGE_PARAMS.gain:.6f}, "
        f"offset_C={DEFAULT_CRYOSTAGE_PARAMS.offset_C:.6f})"
    )
    print(f"Formulation = {args.formulation}")
    print(f"Optimizer method = {args.method}")
    print(f"Active workflow chain = {ACTIVE_WORKFLOW_CHAIN}")
    print(f"Active T_ref bounds = {config.T_ref_bounds_C}")
    print(f"Monotonicity required = {config.require_monotone_nonincreasing}")
    print(f"Characterization admissibility enforced = {config.enforce_characterization_admissibility}")
    if config.enforce_characterization_admissibility:
        print(f"Characterization constraints dir = {config.characterization_constraints_dir}")
    print(f"Safety cap (horizon_s) = {config.safety_cap_s:.1f} s")
    print(f"Trajectory parameter count = {len(config.knot_times_s)}")
    print(f"External time schedule = {args.knot_time_schedule}")
    print(f"Normalized support = {knot_time_normalized_support_tau}")
    print(f"Control support times = {config.knot_times_s}")
    print(f"Initial theta = {theta0}")
    print(f"BO theta bounds = {bayesopt_config.theta_bounds_C}")
    print(
        "BO settings "
        f"(seed={bayesopt_config.random_seed}, init_points={bayesopt_config.init_points}, "
        f"n_iter={bayesopt_config.n_iter}, acq={bayesopt_config.acquisition_kind}, "
        f"kappa={bayesopt_config.acquisition_kappa:.6f}, xi={bayesopt_config.acquisition_xi:.6f}, "
        f"seed_with_theta0={bayesopt_config.seed_with_theta0})"
    )
    print(f"Infeasible objective penalty = {float(args.infeasible_objective_penalty):.6g}")
    reference_contract = front_reference_contract_summary(config)
    front_definition_contract = front_definition_contract_summary(config)
    print(f"Front definition mode = {front_definition_contract['front_definition_mode']}")
    print(f"Front threshold = {front_definition_contract['front_threshold_C']:.9e} C")
    print(f"Front reference mode = {config.front_reference_mode}")
    print(
        "Front target "
        f"(H_fill={reference_contract['H_fill_m']:.9e} m, "
        f"target_end={reference_contract['target_end_s']:.3f} s, "
        f"speed={reference_contract['implied_target_front_speed_m_per_s']:.9e} m/s)"
    )
    print(
        "Objective weights "
        f"(tracking={config.tracking_weight:.6f}, completion={config.completion_weight:.6f}, "
        f"smoothness={config.smoothness_weight:.6f}, terminal={config.terminal_weight:.6f})"
    )
    print(
        "Runner thermal settings "
        f"(T_fill_C={T_fill_C:.6f}, h_top={bcs.h_top:.6f}, h_side={bcs.h_side:.6f})"
    )
    print(
        "Runtime settings "
        f"(Nr={config.solver_kwargs['Nr']}, Nz={config.solver_kwargs['Nz']}, "
        f"dt={config.solver_kwargs['dt']:.3f}, write_every={config.solver_kwargs['write_every']:.1f})"
    )

    result = optimize_open_loop_theta(
        theta0=theta0,
        config=config,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        out_root_dir=out_root_dir,
        run_name=run_name,
        method=args.method,
        options={
            "maxiter": args.maxiter,
            "maxfev": args.maxfev,
            "xatol": args.xatol,
            "fatol": args.fatol,
        },
        bayesopt_config=bayesopt_config,
        infeasible_objective_penalty=float(args.infeasible_objective_penalty),
    )

    analysis_dir = result.run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    best_front_path = result.best_dir / f"{result.best_case_name}_front.csv"
    history_plot_path = analysis_dir / "objective_history.png"
    feasibility_plot_path = analysis_dir / "feasible_vs_infeasible.png"
    front_plot_path = analysis_dir / "best_front_tracking.png"
    plate_reference_plot_path = analysis_dir / "plate_reference_trajectory.png"
    best_theta_profile_csv = result.run_dir / "best_theta_profile.csv"
    best_summary_md = result.run_dir / "best_solution_summary.md"
    run_settings_json = result.run_dir / "run_settings.json"

    _plot_objective_history(result, history_plot_path)
    _plot_feasibility_timeline(result, feasibility_plot_path)
    _plot_best_front_tracking(best_front_path=best_front_path, config=config, out_path=front_plot_path)
    _plot_best_plate_reference_trajectory(theta_C=result.best_theta, config=config, out_path=plate_reference_plot_path)
    _write_best_theta_profile_csv(best_theta_profile_csv, knot_times_s=config.knot_times_s, theta_C=result.best_theta)

    bo_runtime = bayes_opt_runtime_details() if str(result.method) == "bayesian-optimization" else None
    reference_contract = front_reference_contract_summary(config)
    front_definition_contract = front_definition_contract_summary(config)

    _write_best_solution_summary(
        out_path=best_summary_md,
        result=result,
        config=config,
        theta0=theta0,
        theta_bounds_C=bayesopt_config.theta_bounds_C,
        knot_time_schedule=args.knot_time_schedule,
        knot_time_normalized_support_tau=knot_time_normalized_support_tau,
        infeasible_objective_penalty=float(args.infeasible_objective_penalty),
        best_theta_profile_csv=best_theta_profile_csv,
        history_plot_path=history_plot_path,
        feasibility_plot_path=feasibility_plot_path,
        front_plot_path=front_plot_path,
        plate_reference_plot_path=plate_reference_plot_path,
        bo_runtime=bo_runtime,
        smoke_test_note=args.smoke_test_note,
    )
    run_settings_json.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "method": result.method,
                "formulation": args.formulation,
                "workflow_chain": ACTIVE_WORKFLOW_CHAIN,
                "trajectory_parameter_count": len(config.knot_times_s),
                "num_knots": len(config.knot_times_s),
                "knot_time_schedule": str(args.knot_time_schedule),
                "knot_time_custom_support_tau": None if knot_time_custom_support_tau is None else [float(value) for value in knot_time_custom_support_tau],
                "knot_time_normalized_support_tau": [float(value) for value in knot_time_normalized_support_tau],
                "support_times_s": [float(value) for value in config.knot_times_s],
                "knot_times_s": [float(value) for value in config.knot_times_s],
                "initial_theta": [float(value) for value in theta0],
                "theta_bounds_C": bayesopt_config.theta_bounds_C,
                "infeasible_objective_penalty": float(args.infeasible_objective_penalty),
                "front_definition_mode": front_definition_contract['front_definition_mode'],
                "front_threshold_C": front_definition_contract['front_threshold_C'],
                "front_definition_contract": front_definition_contract,
                "front_reference_mode": config.front_reference_mode,
                "H_fill_m": reference_contract["H_fill_m"],
                "target_end_s": reference_contract["target_end_s"],
                "implied_target_front_speed_m_per_s": reference_contract["implied_target_front_speed_m_per_s"],
                "front_reference_contract": reference_contract,
                "objective_weights": {
                    "tracking_weight": config.tracking_weight,
                    "smoothness_weight": config.smoothness_weight,
                    "completion_weight": config.completion_weight,
                    "terminal_weight": config.terminal_weight,
                },
                "bo": {
                    "random_seed": bayesopt_config.random_seed,
                    "init_points": bayesopt_config.init_points,
                    "n_iter": bayesopt_config.n_iter,
                    "acquisition_kind": bayesopt_config.acquisition_kind,
                    "acquisition_kappa": bayesopt_config.acquisition_kappa,
                    "acquisition_xi": bayesopt_config.acquisition_xi,
                    "seed_with_theta0": bayesopt_config.seed_with_theta0,
                },
                "artifacts": {
                    "history_csv_path": str(result.history_csv_path.resolve()),
                    "best_dir": str(result.best_dir.resolve()),
                    "best_theta_profile_csv": str(best_theta_profile_csv.resolve()),
                    "best_summary_md": str(best_summary_md.resolve()),
                    "objective_history_plot": str(history_plot_path.resolve()),
                    "feasibility_plot": str(feasibility_plot_path.resolve()),
                    "front_tracking_plot": str(front_plot_path.resolve()),
                    "plate_reference_trajectory_plot": str(plate_reference_plot_path.resolve()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_evaluations = len(result.history)
    feasible_evaluations = sum(1 for entry in result.history if entry.is_valid)
    infeasible_evaluations = sum(1 for entry in result.history if entry.feasibility_status == "infeasible")
    expensive_runs = sum(1 for entry in result.history if entry.expensive_simulation_executed)
    best_theta_is_valid = _theta_satisfies_active_constraints(result.best_theta, config)

    print("Optimization summary")
    print(f"Best theta: {result.best_theta}")
    print(f"Best theta satisfies active constraints: {best_theta_is_valid}")
    print(f"Best J: {result.best_objective_value:.9e}")
    print(
        f"Evaluations: total={total_evaluations}, feasible={feasible_evaluations}, "
        f"infeasible={infeasible_evaluations}, expensive_simulation_executed={expensive_runs}, nfev={result.nfev}"
    )
    print(
        f"Optimizer termination: success={result.success}, status={result.status}, "
        f"message={result.message}"
    )
    print(f"Evaluation history: {result.history_csv_path}")
    print(f"Best run folder: {result.best_dir}")
    print(f"Best theta profile CSV: {best_theta_profile_csv}")
    print(f"Best plate/reference plot: {plate_reference_plot_path}")
    print(f"Best solution summary: {best_summary_md}")
    print(f"Run settings JSON: {run_settings_json}")
    print(f"Analysis folder: {analysis_dir}")


if __name__ == "__main__":
    main()
