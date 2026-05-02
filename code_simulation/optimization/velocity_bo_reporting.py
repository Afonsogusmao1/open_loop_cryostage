from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_simulation.core.plotting import configure_matplotlib
from code_simulation.core.plate_tracking import (
    summarize_plate_tracking,
    write_plate_tracking_summary_csv,
    write_plate_tracking_timeseries_csv,
)
from code_simulation.core.trajectory_profiles import PiecewiseLinearTemperatureProfile
from code_simulation.optimization.open_loop_optimizer import OpenLoopOptimizationResult
from code_simulation.optimization.open_loop_problem import load_front_csv
from code_simulation.optimization.reachability_constraints import (
    direct_characterization_support_band_C,
    load_reachability_constraints,
)
from code_simulation.optimization.open_loop_workflow_config import OpenLoopProblemConfig
from code_simulation.optimization.velocity_objective import (
    ConstantVelocityObjectiveConfig,
    constant_velocity_tracking,
    segment_speed_error_penalty,
    segment_speed_rows,
    thermocouple_interval_speeds,
    write_segment_speed_summary_csv,
    write_rows_csv,
    write_tracking_summary_csv,
)
from code_simulation.simulation.cryostage_model import CryostageModelParams
from code_simulation.simulation.open_loop_cascade import OpenLoopPlateResponse, build_plate_temperature_response


def _configure_matplotlib() -> None:
    configure_matplotlib(plt)


def _single_match(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def write_best_theta_profile_csv(
    out_path: Path,
    *,
    knot_times_s: tuple[float, ...],
    theta_C: tuple[float, ...],
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("knot_index", "time_s", "temperature_C"))
        writer.writeheader()
        for idx, (time_s, temperature_C) in enumerate(zip(knot_times_s, theta_C, strict=True)):
            writer.writerow(
                {
                    "knot_index": int(idx),
                    "time_s": float(time_s),
                    "temperature_C": float(temperature_C),
                }
            )


def plot_bo_convergence(result: OpenLoopOptimizationResult, out_path: Path) -> None:
    _configure_matplotlib()
    indices = np.asarray([entry.evaluation_index for entry in result.history], dtype=np.float64)
    objective_values = np.asarray([entry.objective_value for entry in result.history], dtype=np.float64)
    feasible_mask = np.asarray([entry.is_valid for entry in result.history], dtype=bool)

    fig, ax = plt.subplots()
    if np.any(feasible_mask):
        ax.plot(
            indices[feasible_mask],
            objective_values[feasible_mask],
            marker="o",
            linewidth=1.4,
            markersize=4.0,
            color="#1f77b4",
            label="feasible",
        )
    if np.any(~feasible_mask):
        ax.scatter(
            indices[~feasible_mask],
            objective_values[~feasible_mask],
            marker="x",
            color="#d62728",
            label="infeasible/failed",
        )
    ax.scatter(
        [result.best_evaluation_index],
        [result.best_objective_value],
        color="#2ca02c",
        s=60,
        label="best",
        zorder=3,
    )
    finite_positive = objective_values[np.isfinite(objective_values) & (objective_values > 0.0)]
    if finite_positive.size >= 2 and float(np.max(finite_positive) / np.min(finite_positive)) >= 10.0:
        ax.set_yscale("log")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Objective value")
    ax.set_title("Velocity-control BO convergence")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_temperature_profiles(out_path: Path, *, plate_response: OpenLoopPlateResponse) -> None:
    _configure_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(plate_response.cryostage_time_s, plate_response.T_ref_C, label="T_ref")
    ax.plot(plate_response.cryostage_time_s, plate_response.T_plate_C, label="T_plate")
    ax.set_xlabel("Time since fill/cooling start (s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Best cryostage reference and modeled plate temperature")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _build_plate_response(
    *,
    theta_C: tuple[float, ...],
    config: OpenLoopProblemConfig,
    cryostage_params: CryostageModelParams,
    initial_plate_temperature_C: float,
) -> OpenLoopPlateResponse:
    T_ref_profile = PiecewiseLinearTemperatureProfile(
        knot_times_s=config.knot_times_s,
        knot_temperatures_C=theta_C,
    )
    return build_plate_temperature_response(
        time_s=config.cryostage_time_grid_s(),
        T_ref_profile_C=T_ref_profile,
        cryostage_params=cryostage_params,
        T_plate0_C=initial_plate_temperature_C,
        bcs=config.solver_kwargs.get("bcs"),
    )


def plot_best_front_position(
    out_path: Path,
    *,
    front_path: Path,
    objective_config: ConstantVelocityObjectiveConfig,
    incomplete_penalty_value: float,
) -> None:
    _configure_matplotlib()
    front = load_front_csv(front_path)
    summary, series = constant_velocity_tracking(
        front,
        objective_config,
        incomplete_penalty_value=incomplete_penalty_value,
    )
    _ = summary

    fig, ax = plt.subplots()
    ax.plot(series.control_time_s, 1000.0 * series.z_front_m, label="z_front")
    mask = np.isfinite(series.z_ref_m)
    ax.plot(series.control_time_s[mask], 1000.0 * series.z_ref_m[mask], "--", label="target-speed reference")
    ax.axhline(objective_config.control_z_min_mm, color="0.35", linestyle=":", linewidth=1.0, label="control window")
    ax.axhline(objective_config.control_z_max_mm, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Time since fill/cooling start (s)")
    ax.set_ylabel("Front position (mm)")
    ax.set_title("Best front position against velocity target")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def write_velocity_bo_report(
    out_path: Path,
    *,
    result: OpenLoopOptimizationResult,
    objective_config: ConstantVelocityObjectiveConfig,
    theta_bounds_C: tuple[tuple[float, float], ...],
    initial_water_temperature_C: float,
    initial_plate_temperature_C: float,
    simulation_profile_name: str,
    use_tabulated_water_ice: bool,
    uses_temperatures_outside_direct_support_band: bool,
    direct_support_band_C: tuple[float, float],
    characterization_temperature_margin_C: float,
) -> None:
    backend = result.bayesian_backend_result
    lines = [
        "# Velocity-Control BO Report",
        "",
        "This run optimized cryostage reference temperatures for a constant direct freezing-front velocity.",
        "",
        "## Target",
        "",
        f"- Target front speed: `{objective_config.target_front_speed_mm_s:.6g} mm/s`.",
        (
            "- Control window: "
            f"`{objective_config.control_z_min_mm:.3g}-{objective_config.control_z_max_mm:.3g} mm`."
        ),
        f"- Initial water temperature: `{initial_water_temperature_C:.6g} C`.",
        f"- Initial plate temperature: `{initial_plate_temperature_C:.6g} C`.",
        f"- Simulation profile: `{simulation_profile_name}`.",
        f"- Tabulated water/ice properties enabled: `{str(bool(use_tabulated_water_ice)).lower()}`.",
        f"- Direct-speed objective weight: `{objective_config.direct_speed_weight:.6g}`.",
        (
            "- Segment-speed objective: "
            f"`{objective_config.segment_speed_num_segments}` segments, "
            f"weight `{objective_config.segment_speed_weight:.6g}`, "
            f"tolerance `{objective_config.segment_speed_tolerance_pct:.6g}%`."
        ),
        "",
        "## BO Result",
        "",
        f"- Success: `{bool(result.success)}`.",
        f"- Message: `{result.message}`.",
        f"- Number of evaluations: `{result.nfev}`.",
        f"- Best evaluation index: `{result.best_evaluation_index}`.",
        f"- Best objective value: `{result.best_objective_value:.9e}`.",
        f"- Best theta: `{tuple(float(value) for value in result.best_theta)}`.",
        f"- BO theta bounds: `{theta_bounds_C}`.",
        (
            "- Direct characterization support band with margin: "
            f"`{direct_support_band_C[0]:.6g} to {direct_support_band_C[1]:.6g} C` "
            f"(using `+-{characterization_temperature_margin_C:.6g} C`)."
        ),
        (
            "- Uses temperatures outside the direct characterization support band: "
            f"`{str(bool(uses_temperatures_outside_direct_support_band)).lower()}`."
        ),
        "",
        "## Metric Notes",
        "",
        "- The BO objective uses the direct simulated front position `z_front(t)`.",
        "- When enabled, the segment-speed term penalizes unequal average speeds across depth intervals.",
        "- The raw finite-difference velocity `v_front_mm_per_s` remains diagnostic because it is a noisy derivative.",
        "- Thermocouple-equivalent speeds are interval averages and are written only for diagnostic comparison with experiments.",
        "",
    ]
    if backend is not None:
        lines.extend(
            [
                "## BO Backend",
                "",
                f"- Method: `{backend.method}`.",
                f"- Acquisition: `{backend.acquisition_kind}` with `kappa={backend.acquisition_kappa:.6g}` and `xi={backend.acquisition_xi:.6g}`.",
                f"- Parameterization: `{backend.parameterization_kind}`.",
                f"- Search bounds: `{backend.search_bounds}`.",
                f"- Init strategy: `{backend.init_strategy}`.",
                f"- Init local sigma: `{backend.init_local_sigma:.6g}`.",
                f"- Init max attempts per point: `{backend.init_max_attempts_per_point}`.",
                f"- Init candidate attempts: `{backend.init_candidate_attempts}`.",
                f"- Init precheck rejections: `{backend.init_precheck_rejections}`.",
                f"- Init accepted candidates: `{backend.init_accepted_candidates}`.",
                f"- Init global fallback accepts: `{backend.init_global_fallback_accepts}`.",
                f"- Local refinement points: `{backend.local_refinement_points}`.",
                f"- Local refinement sigma: `{backend.local_refinement_sigma:.6g}`.",
                f"- Local refinement attempts: `{backend.local_refinement_candidate_attempts}`.",
                f"- Local refinement precheck rejections: `{backend.local_refinement_precheck_rejections}`.",
                f"- Local refinement accepted candidates: `{backend.local_refinement_accepted_candidates}`.",
                f"- Local refinement improved candidates: `{backend.local_refinement_improved_candidates}`.",
                "- `evaluation_history.csv` stores both the BO-space `raw_candidate` and the reconstructed physical `theta`.",
                f"- Backend package: `{backend.package_version}` from `{backend.package_path}`.",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def finalize_velocity_bo_outputs(
    *,
    result: OpenLoopOptimizationResult,
    config: OpenLoopProblemConfig,
    objective_config: ConstantVelocityObjectiveConfig,
    cryostage_params: CryostageModelParams,
    initial_water_temperature_C: float,
    initial_plate_temperature_C: float,
    simulation_profile_name: str,
    theta_bounds_C: tuple[tuple[float, float], ...],
) -> dict[str, Path]:
    run_dir = result.run_dir
    artifacts: dict[str, Path] = {}

    bo_history_path = run_dir / "bo_history.csv"
    shutil.copyfile(result.history_csv_path, bo_history_path)
    artifacts["bo_history"] = bo_history_path

    best_theta_profile_path = run_dir / "best_theta_profile.csv"
    write_best_theta_profile_csv(
        best_theta_profile_path,
        knot_times_s=config.knot_times_s,
        theta_C=result.best_theta,
    )
    artifacts["best_theta_profile"] = best_theta_profile_path

    best_front_path = _single_match(result.best_dir, "*_front.csv")
    tracking_summary = None
    if best_front_path is not None:
        front = load_front_csv(best_front_path)
        tracking_summary, _ = constant_velocity_tracking(
            front,
            objective_config,
            incomplete_penalty_value=config.incomplete_penalty_value,
        )
        best_tracking_summary_path = run_dir / "best_tracking_summary.csv"
        write_tracking_summary_csv(best_tracking_summary_path, tracking_summary)
        artifacts["best_tracking_summary"] = best_tracking_summary_path
        if objective_config.segment_speed_num_segments > 0:
            segment_summary = segment_speed_error_penalty(
                front,
                objective_config,
                incomplete_penalty_value=config.incomplete_penalty_value,
            )
            segment_summary_path = run_dir / "best_segment_speed_summary.csv"
            write_segment_speed_summary_csv(segment_summary_path, segment_summary)
            artifacts["best_segment_speed_summary"] = segment_summary_path

            segment_rows_path = run_dir / "best_segment_speeds.csv"
            write_rows_csv(segment_rows_path, segment_speed_rows(front, objective_config))
            artifacts["best_segment_speeds"] = segment_rows_path

        best_front_plot_path = run_dir / "best_front_position_vs_reference.png"
        plot_best_front_position(
            best_front_plot_path,
            front_path=best_front_path,
            objective_config=objective_config,
            incomplete_penalty_value=config.incomplete_penalty_value,
        )
        artifacts["best_front_position_vs_reference"] = best_front_plot_path

    best_probes_path = _single_match(result.best_dir, "*_probes.csv")
    if best_probes_path is not None:
        tc_rows = thermocouple_interval_speeds(best_probes_path)
        tc_path = run_dir / "best_thermocouple_interval_speeds.csv"
        write_rows_csv(tc_path, tc_rows)
        artifacts["best_thermocouple_interval_speeds"] = tc_path

    convergence_path = run_dir / "bo_convergence.png"
    plot_bo_convergence(result, convergence_path)
    artifacts["bo_convergence"] = convergence_path

    temperature_plot_path = run_dir / "best_T_ref_and_T_plate_vs_time.png"
    plate_response = _build_plate_response(
        theta_C=result.best_theta,
        config=config,
        cryostage_params=cryostage_params,
        initial_plate_temperature_C=initial_plate_temperature_C,
    )
    plot_best_temperature_profiles(
        temperature_plot_path,
        plate_response=plate_response,
    )
    artifacts["best_T_ref_and_T_plate_vs_time"] = temperature_plot_path
    plate_tracking_summary, plate_tracking_series = summarize_plate_tracking(
        time_s=plate_response.cryostage_time_s,
        T_ref_C=plate_response.T_ref_C,
        T_plate_C=plate_response.T_plate_C,
        tolerance_C=config.characterization_temperature_margin_C,
        evaluation_window_start_s=(
            math.nan if tracking_summary is None else tracking_summary.t_at_control_z_min_s
        ),
        evaluation_window_end_s=(
            None
            if tracking_summary is None or not tracking_summary.reached_control_z_max
            else tracking_summary.t_at_control_z_max_s
        ),
    )
    plate_tracking_summary_path = run_dir / "plate_tracking_summary.csv"
    write_plate_tracking_summary_csv(plate_tracking_summary_path, plate_tracking_summary)
    artifacts["plate_tracking_summary"] = plate_tracking_summary_path
    plate_tracking_timeseries_path = run_dir / "T_ref_T_plate_timeseries.csv"
    write_plate_tracking_timeseries_csv(plate_tracking_timeseries_path, plate_tracking_series)
    artifacts["T_ref_T_plate_timeseries"] = plate_tracking_timeseries_path

    constraints = load_reachability_constraints(config.characterization_constraints_dir)
    direct_support_band_C = direct_characterization_support_band_C(
        constraints,
        temperature_margin_C=config.characterization_temperature_margin_C,
    )
    uses_temperatures_outside_direct_support_band = any(
        float(value) < direct_support_band_C[0] - 1.0e-12
        or float(value) > direct_support_band_C[1] + 1.0e-12
        for value in result.best_theta
    )
    report_path = run_dir / "velocity_control_bo_report.md"
    write_velocity_bo_report(
        report_path,
        result=result,
        objective_config=objective_config,
        theta_bounds_C=theta_bounds_C,
        initial_water_temperature_C=initial_water_temperature_C,
        initial_plate_temperature_C=initial_plate_temperature_C,
        simulation_profile_name=simulation_profile_name,
        use_tabulated_water_ice=bool(config.solver_kwargs.get("use_tabulated_water_ice", False)),
        uses_temperatures_outside_direct_support_band=uses_temperatures_outside_direct_support_band,
        direct_support_band_C=direct_support_band_C,
        characterization_temperature_margin_C=config.characterization_temperature_margin_C,
    )
    artifacts["velocity_control_bo_report"] = report_path

    return artifacts


__all__ = [
    "finalize_velocity_bo_outputs",
    "plot_best_front_position",
    "plot_best_temperature_profiles",
    "plot_bo_convergence",
    "write_best_theta_profile_csv",
    "write_velocity_bo_report",
]
