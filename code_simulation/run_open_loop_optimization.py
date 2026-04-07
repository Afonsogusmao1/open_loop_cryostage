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
from geometry import GeometryParams
from open_loop_bayesian_optimizer import BayesianOptimizationConfig, bayes_opt_runtime_details
from open_loop_optimizer import OpenLoopOptimizationResult, optimize_open_loop_theta
from open_loop_problem import (
    OpenLoopProblemConfig,
    build_front_reference,
    build_reference_profile_from_theta,
    load_front_csv,
)
from reachability_constraints import default_constraints_dir
from solver import FreezeStopOptions, PrefillOptions, ThermalBCs


DEFAULT_FORMULATION = "full_process_article"
DEFAULT_METHOD = "bayesian-optimization"
DEFAULT_RUN_NAME = "bo_full_process_k5"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"
DEFAULT_T_REF_BOUNDS_C = (-20.0, 0.0)
DEFAULT_REQUIRE_MONOTONE_NONINCREASING = True
DEFAULT_THETA0 = (-0.1, -5.5, -11.9, -17.0, -17.2)
DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR = default_constraints_dir(Path(__file__).resolve().parent)
DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY = 1.0e6
DEFAULT_KNOT_TIME_SCHEDULE = "uniform"
SUPPORTED_KNOT_TIME_SCHEDULES = ("uniform", "early_dense", "mid_dense", "late_dense", "custom")
EARLY_LATE_DENSE_POWER = 1.5
MID_DENSE_POWER = 2.0


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


def _uniform_knot_times_s(*, horizon_s: float, num_knots: int) -> tuple[float, ...]:
    if num_knots < 2:
        raise ValueError("num_knots must be at least 2")
    horizon_s = float(horizon_s)
    step_s = horizon_s / float(num_knots - 1)
    knot_times_s = [float(i * step_s) for i in range(num_knots - 1)]
    knot_times_s.append(horizon_s)
    return tuple(knot_times_s)


def _validated_normalized_support_tau(
    values,
    *,
    num_knots: int,
    name: str,
) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size != int(num_knots):
        raise ValueError(f"{name} must contain exactly {int(num_knots)} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < -1.0e-12) or np.any(arr > 1.0 + 1.0e-12):
        raise ValueError(f"{name} must stay within [0, 1]")
    arr = np.clip(arr, 0.0, 1.0)
    if abs(float(arr[0])) > 1.0e-12 or abs(float(arr[-1]) - 1.0) > 1.0e-12:
        raise ValueError(f"{name} must start at 0 and end at 1")
    if np.any(np.diff(arr) <= 1.0e-12):
        raise ValueError(f"{name} must be strictly increasing")
    return tuple(float(value) for value in arr)


def _parse_normalized_support_tau_arg(raw_support_tau: str | None, *, num_knots: int) -> tuple[float, ...] | None:
    if raw_support_tau is None:
        return None
    parts = [part.strip() for part in str(raw_support_tau).split(",")]
    if len(parts) != int(num_knots):
        raise ValueError(
            "--knot-time-custom-support-tau must contain exactly one normalized support time per knot "
            f"({int(num_knots)} expected, got {len(parts)})"
        )
    return _validated_normalized_support_tau(
        [float(part) for part in parts],
        num_knots=int(num_knots),
        name="knot_time_custom_support_tau",
    )


def _parse_normalized_support_tau_by_n_arg(raw_support_map: str | None) -> dict[int, tuple[float, ...]]:
    if raw_support_map is None:
        return {}
    resolved: dict[int, tuple[float, ...]] = {}
    entries = [entry.strip() for entry in str(raw_support_map).split(";") if entry.strip()]
    if not entries:
        raise ValueError("--custom-support-by-n must contain at least one N:support-list entry when provided")
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                "Each --custom-support-by-n entry must be written as 'N:t0,t1,...,tN', for example '3:0,0.35,1'"
            )
        n_text, support_text = entry.split(":", 1)
        num_knots = int(n_text.strip())
        support_parts = [part.strip() for part in support_text.split(",") if part.strip()]
        support_tau = _validated_normalized_support_tau(
            [float(part) for part in support_parts],
            num_knots=num_knots,
            name=f"custom support for N={num_knots}",
        )
        resolved[num_knots] = support_tau
    return resolved


def build_knot_time_normalized_support_tau(
    *,
    num_knots: int,
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    num_knots = int(num_knots)
    schedule = str(knot_time_schedule).strip().lower()
    if schedule not in SUPPORTED_KNOT_TIME_SCHEDULES:
        raise ValueError(
            f"Unsupported knot_time_schedule={knot_time_schedule!r}; expected one of {SUPPORTED_KNOT_TIME_SCHEDULES!r}"
        )

    if schedule != "custom" and knot_time_custom_support_tau is not None:
        raise ValueError("knot_time_custom_support_tau is only valid when knot_time_schedule='custom'")
    if schedule == "custom":
        if knot_time_custom_support_tau is None:
            raise ValueError("knot_time_schedule='custom' requires knot_time_custom_support_tau")
        return _validated_normalized_support_tau(
            knot_time_custom_support_tau,
            num_knots=num_knots,
            name="knot_time_custom_support_tau",
        )

    uniform_tau = np.linspace(0.0, 1.0, num_knots, dtype=np.float64)
    if schedule == "uniform":
        support_tau = uniform_tau
    elif schedule == "early_dense":
        support_tau = np.power(uniform_tau, EARLY_LATE_DENSE_POWER)
    elif schedule == "late_dense":
        support_tau = 1.0 - np.power(1.0 - uniform_tau, EARLY_LATE_DENSE_POWER)
    elif schedule == "mid_dense":
        centered = 2.0 * uniform_tau - 1.0
        support_tau = 0.5 * (1.0 + np.sign(centered) * np.power(np.abs(centered), MID_DENSE_POWER))
        support_tau[0] = 0.0
        support_tau[-1] = 1.0
    else:
        raise ValueError(f"Unsupported knot_time_schedule={knot_time_schedule!r}")

    return _validated_normalized_support_tau(
        support_tau,
        num_knots=num_knots,
        name=f"{schedule} normalized support",
    )


def build_external_knot_times_s(
    *,
    horizon_s: float,
    num_knots: int,
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    support_tau = build_knot_time_normalized_support_tau(
        num_knots=int(num_knots),
        knot_time_schedule=knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
    )
    horizon_s = float(horizon_s)
    return tuple(float(horizon_s * tau_i) for tau_i in support_tau)


def _build_legacy_problem_config() -> OpenLoopProblemConfig:
    horizon_s = 180.0
    return OpenLoopProblemConfig(
        horizon_s=horizon_s,
        cryostage_dt_s=2.0,
        knot_times_s=(0.0, 45.0, 90.0, 135.0, 180.0),
        front_target_speed_m_per_s=2.0e-5,
        tracking_weight=1.0,
        t_ignore_s=20.0,
        T_ref_bounds_C=DEFAULT_T_REF_BOUNDS_C,
        require_monotone_nonincreasing=DEFAULT_REQUIRE_MONOTONE_NONINCREASING,
        enforce_characterization_admissibility=True,
        characterization_constraints_dir=DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR,
        solver_kwargs={
            "geom": GeometryParams(
                R_in=7.5e-3,
                t_wall=2.0e-3,
                t_base=0.0,
                H_fill=15.0e-3,
                H_total=17.0e-3,
            ),
            "Nr": 36,
            "Nz": 72,
            "dt": 2.0,
            "pre_cool_s": 0.0,
            "write_every": 30.0,
            "T_fill_C": 12.5,
            "bcs": ThermalBCs(T_room_C=5.75, h_top=2.0, h_side=2.0),
            "prefill": PrefillOptions(mode="steady"),
            "freeze_stop": FreezeStopOptions(mode="fillable_region"),
            "probe_z_m": (3.0e-3, 6.2e-3, 11.0e-3),
            "probe_wall_inset_m": 1.0e-3,
            "Nz_front": 200,
            "enable_front_curve": False,
            "stop_when_wall_frozen": False,
            "use_tabulated_water_ice": True,
        },
        front_reference_mode="legacy_linear_speed",
    )


def _build_full_process_problem_config(
    *,
    num_knots: int = len(DEFAULT_THETA0),
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
) -> OpenLoopProblemConfig:
    safety_cap_s = 2400.0
    knot_times_s = build_external_knot_times_s(
        horizon_s=safety_cap_s,
        num_knots=num_knots,
        knot_time_schedule=knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
    )
    return OpenLoopProblemConfig(
        horizon_s=safety_cap_s,
        cryostage_dt_s=4.0,
        knot_times_s=knot_times_s,
        front_target_speed_m_per_s=2.0e-5,
        tracking_weight=1.0,
        smoothness_weight=0.02,
        completion_weight=0.25,
        t_ignore_s=0.0,
        T_ref_bounds_C=DEFAULT_T_REF_BOUNDS_C,
        require_monotone_nonincreasing=DEFAULT_REQUIRE_MONOTONE_NONINCREASING,
        enforce_characterization_admissibility=True,
        characterization_constraints_dir=DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR,
        solver_kwargs={
            "geom": GeometryParams(
                R_in=7.5e-3,
                t_wall=2.0e-3,
                t_base=0.0,
                H_fill=15.0e-3,
                H_total=17.0e-3,
            ),
            "Nr": 36,
            "Nz": 72,
            "dt": 4.0,
            "pre_cool_s": 0.0,
            "write_every": 1.0e9,
            "write_field_output": False,
            "write_probe_csv": False,
            "show_progress": False,
            "T_fill_C": 12.5,
            "bcs": ThermalBCs(T_room_C=5.75, h_top=2.0, h_side=2.0),
            "prefill": PrefillOptions(mode="steady"),
            "freeze_stop": FreezeStopOptions(mode="fillable_region"),
            "probe_z_m": (3.0e-3, 6.2e-3, 11.0e-3),
            "probe_wall_inset_m": 1.0e-3,
            "Nz_front": 200,
            "enable_front_curve": False,
            "stop_when_wall_frozen": False,
            "use_tabulated_water_ice": True,
        },
        front_reference_mode="saturating_full_process",
        front_reference_alpha=4.0,
        incomplete_penalty_value=2.0,
    )


def build_problem_config(
    *,
    formulation: str = DEFAULT_FORMULATION,
    num_knots: int | None = None,
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
) -> OpenLoopProblemConfig:
    if formulation == "legacy_exploratory":
        if num_knots not in (None, len(DEFAULT_THETA0)):
            raise ValueError("num_knots is only configurable for full_process_article")
        if knot_time_schedule != DEFAULT_KNOT_TIME_SCHEDULE or knot_time_custom_support_tau is not None:
            raise ValueError("knot-time schedules are only configurable for full_process_article")
        return _build_legacy_problem_config()
    if formulation == "full_process_article":
        resolved_num_knots = len(DEFAULT_THETA0) if num_knots is None else int(num_knots)
        return _build_full_process_problem_config(
            num_knots=resolved_num_knots,
            knot_time_schedule=knot_time_schedule,
            knot_time_custom_support_tau=knot_time_custom_support_tau,
        )
    raise ValueError(f"Unknown formulation={formulation!r}")


def _default_theta0_for_config(config: OpenLoopProblemConfig) -> tuple[float, ...]:
    if len(config.knot_times_s) == len(DEFAULT_THETA0):
        return DEFAULT_THETA0
    base_tau = np.linspace(0.0, 1.0, len(DEFAULT_THETA0), dtype=np.float64)
    target_tau = np.asarray(config.knot_times_s, dtype=np.float64) / float(config.horizon_s)
    theta0 = np.interp(target_tau, base_tau, np.asarray(DEFAULT_THETA0, dtype=np.float64))
    theta0 = np.clip(theta0, config.T_ref_bounds_C[0], config.T_ref_bounds_C[1])
    return tuple(float(value) for value in theta0)


def _parse_theta_arg(raw_theta: str | None, *, num_knots: int) -> tuple[float, ...] | None:
    if raw_theta is None:
        return None
    parts = [part.strip() for part in raw_theta.split(",")]
    if len(parts) != int(num_knots):
        raise ValueError(
            "--theta0 must contain exactly one comma-separated value per knot temperature "
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
    if formulation == "full_process_article" and requested_run_name == DEFAULT_RUN_NAME:
        resolved = DEFAULT_RUN_NAME.replace(f"_k{len(DEFAULT_THETA0)}", f"_k{len(config.knot_times_s)}")
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
            "--theta-bounds must contain exactly one lower:upper pair per knot temperature "
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
    time_s = config.cryostage_time_grid_s()
    T_ref_C = np.asarray([float(reference_profile(float(ti))) for ti in time_s], dtype=np.float64)
    bcs = config.solver_kwargs.get("bcs")
    T_plate0_C = float(bcs.T_room_C) if bcs is not None and hasattr(bcs, "T_room_C") else float(T_ref_C[0])
    T_plate_C = simulate_plate_temperature(time_s, reference_profile, DEFAULT_CRYOSTAGE_PARAMS, T_plate0_C)

    fig, ax = plt.subplots()
    ax.plot(time_s, T_ref_C, linewidth=2.0, label=r"$T_{ref}$")
    ax.plot(time_s, T_plate_C, linewidth=2.0, label=r"$T_{plate,model}$")
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

    lines = [
        "# Open-Loop Optimization Summary",
        "",
        "## Optimizer role",
        f"- Main optimizer path: `{result.method}`.",
        "- Scientific objective: unchanged front-position tracking objective from `evaluate_open_loop_objective`.",
        f"- Fixed knot count: `{len(config.knot_times_s)}` knot temperatures at externally fixed times `{list(config.knot_times_s)}`.",
        f"- Fixed knot-time schedule: `{knot_time_schedule}` with normalized support `{list(knot_time_normalized_support_tau)}`.",
        f"- Optimized variables: knot temperatures only, with no knot-time or knot-count optimization.",
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
        f"- Active knot times (s): `{list(config.knot_times_s)}`",
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
            "## Next step before the fixed-N comparison study",
            "- Keep this BO path fixed at one externally chosen knot count, then compare budgets and settings across candidate fixed-N studies only after confirming the per-run settings are scientifically acceptable.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible open-loop optimization.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder name inside the output root.")
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
        "--num-knots",
        type=int,
        default=len(DEFAULT_THETA0),
        help="Number of externally fixed T_ref knots over [0, horizon_s] for full_process_article.",
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
        help="Optional comma-separated initial knot temperatures. Defaults to the canonical profile resampled to the active knot grid.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="Optimizer backend. Use 'bayesian-optimization' for the new BO path or a scipy.optimize.minimize method name for the legacy path.",
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
        help="Optional comma-separated lower:upper bounds for each knot temperature, for example '-0.5:0,-7:-4,-13:-10,-17:-14.5,-20:-17'.",
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

    knot_time_custom_support_tau = _parse_normalized_support_tau_arg(
        args.knot_time_custom_support_tau,
        num_knots=int(args.num_knots),
    )
    config = build_problem_config(
        formulation=args.formulation,
        num_knots=args.num_knots,
        knot_time_schedule=args.knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
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
    print(f"Active T_ref bounds = {config.T_ref_bounds_C}")
    print(f"Monotonicity required = {config.require_monotone_nonincreasing}")
    print(f"Characterization admissibility enforced = {config.enforce_characterization_admissibility}")
    if config.enforce_characterization_admissibility:
        print(f"Characterization constraints dir = {config.characterization_constraints_dir}")
    print(f"Safety cap (horizon_s) = {config.safety_cap_s:.1f} s")
    print(f"Knot-time schedule = {args.knot_time_schedule}")
    print(f"Normalized knot support = {knot_time_normalized_support_tau}")
    print(f"Control knot times = {config.knot_times_s}")
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
    print(f"Front reference mode = {config.front_reference_mode}")
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
                "num_knots": len(config.knot_times_s),
                "knot_time_schedule": str(args.knot_time_schedule),
                "knot_time_custom_support_tau": None if knot_time_custom_support_tau is None else [float(value) for value in knot_time_custom_support_tau],
                "knot_time_normalized_support_tau": [float(value) for value in knot_time_normalized_support_tau],
                "knot_times_s": [float(value) for value in config.knot_times_s],
                "initial_theta": [float(value) for value in theta0],
                "theta_bounds_C": bayesopt_config.theta_bounds_C,
                "infeasible_objective_penalty": float(args.infeasible_objective_penalty),
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
