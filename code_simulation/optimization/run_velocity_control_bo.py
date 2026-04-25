from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from code_simulation.core.config_files import (
    BOFileConfig,
    DEFAULT_BO_CONFIG_PATH,
    DEFAULT_SIMULATION_CONFIG_PATH,
    DEFAULT_VELOCITY_CONTROL_CONFIG_PATH,
    VelocityControlFileConfig,
    load_bo_config,
    load_simulation_profile,
    load_velocity_control_config,
)
from code_simulation.core.paths import velocity_control_results_dir
from code_simulation.optimization.open_loop_bayesian_optimizer import (
    BayesianOptimizationConfig,
    build_candidate_parameterization,
    normalize_theta_bounds,
)
from code_simulation.optimization.open_loop_optimizer import optimize_open_loop_theta
from code_simulation.optimization.open_loop_workflow_config import (
    DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR,
    OpenLoopProblemConfig,
    apply_simulation_profile,
    build_external_knot_times_s,
)
from code_simulation.optimization.velocity_bo_reporting import finalize_velocity_bo_outputs
from code_simulation.optimization.velocity_objective import (
    ConstantVelocityObjectiveConfig,
    evaluate_velocity_control_objective,
)
from code_simulation.simulation.cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
from code_simulation.simulation.geometry import GeometryParams
from code_simulation.simulation.solver import FreezeStopOptions, PhaseChangeParams, PrefillOptions, ThermalBCs


DEFAULT_RUN_NAME = "bo_v0p008_n8_uniform_seed17"
DEFAULT_OUT_ROOT_DIR = velocity_control_results_dir()
PROBE_Z_M = (3.0e-3, 6.2e-3, 11.0e-3)
PROBE_WALL_INSET_M = 1.0e-3
H_OUT_W_M2K = 2.0
DIRECT_CHARACTERIZATION_COLD_LIMIT_C = -20.0


def _parse_float_tuple(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{name} must contain at least one comma-separated value")
    values = tuple(float(part) for part in parts)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _parse_theta_bounds(raw: str | None, *, num_knots: int) -> tuple[tuple[float, float], ...] | None:
    if raw is None:
        return None
    pairs = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(pairs) != int(num_knots):
        raise ValueError(f"--theta-bounds must contain exactly {int(num_knots)} lower:upper pairs")
    bounds: list[tuple[float, float]] = []
    for idx, pair in enumerate(pairs):
        if ":" not in pair:
            raise ValueError(f"theta bound {idx} must be written as lower:upper")
        lower_text, upper_text = (piece.strip() for piece in pair.split(":", 1))
        lower = float(lower_text)
        upper = float(upper_text)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"theta bound {idx} must satisfy lower < upper with finite values")
        bounds.append((lower, upper))
    return tuple(bounds)


def _resolve_theta0(
    *,
    args: argparse.Namespace,
    bo_config: BOFileConfig,
    num_knots: int,
) -> tuple[tuple[float, ...], str]:
    if args.theta0_from_run_dir is not None:
        run_dir = Path(args.theta0_from_run_dir).resolve()
        theta_path = run_dir / "best_theta_profile.csv"
        if not theta_path.exists():
            raise FileNotFoundError(f"{theta_path} does not exist")
        theta_rows: list[float] = []
        for raw_line in theta_path.read_text(encoding="utf-8").splitlines()[1:]:
            if not raw_line.strip():
                continue
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"{theta_path} contains a malformed row: {raw_line!r}")
            theta_rows.append(float(parts[2]))
        theta0 = tuple(theta_rows)
        if len(theta0) != int(num_knots):
            raise ValueError(
                f"theta0 loaded from {theta_path} must contain exactly {int(num_knots)} values"
            )
        return theta0, f"from_run:{run_dir}"
    cli_theta0 = _parse_float_tuple(args.theta0_c, name="--theta0-c")
    theta0 = cli_theta0 if cli_theta0 is not None else bo_config.theta0_C
    if theta0 is None:
        theta0 = tuple(np.linspace(0.0, -20.0, int(num_knots), dtype=np.float64))
    if len(theta0) != int(num_knots):
        raise ValueError(f"theta0 must contain exactly {int(num_knots)} values")
    if cli_theta0 is not None:
        theta0_source = "cli"
    elif bo_config.theta0_C is not None:
        theta0_source = "bo_config"
    else:
        theta0_source = "default_linear"
    return tuple(float(value) for value in theta0), theta0_source


def _resolve_num_knots(args: argparse.Namespace, bo_config: BOFileConfig) -> int:
    num_knots = int(bo_config.num_knots if args.num_knots is None else args.num_knots)
    if num_knots < 2:
        raise ValueError("num_knots must be at least 2")
    return num_knots


def _build_problem_config(
    *,
    velocity_config: VelocityControlFileConfig,
    bo_config: BOFileConfig,
    simulation_profile,
    args: argparse.Namespace,
    num_knots: int,
) -> OpenLoopProblemConfig:
    target = velocity_config.velocity_target
    initial = velocity_config.initial_conditions
    trajectory = velocity_config.manual_trajectory
    horizon_s = float(args.horizon_s if args.horizon_s is not None else trajectory.horizon_s)
    knot_time_schedule = str(args.knot_time_schedule or bo_config.knot_time_schedule)
    cli_support = _parse_float_tuple(args.knot_time_custom_support_tau, name="--knot-time-custom-support-tau")
    custom_support = cli_support if cli_support is not None else bo_config.knot_time_custom_support_tau
    T_ref_bounds_C = bo_config.T_ref_bounds_C
    if args.T_ref_bounds_C is not None:
        parsed_bounds = _parse_float_tuple(args.T_ref_bounds_C, name="--T-ref-bounds-C")
        if parsed_bounds is None or len(parsed_bounds) != 2:
            raise ValueError("--T-ref-bounds-C must contain exactly two comma-separated values")
        T_ref_bounds_C = (float(parsed_bounds[0]), float(parsed_bounds[1]))
    if T_ref_bounds_C[0] >= T_ref_bounds_C[1]:
        raise ValueError("T_ref_bounds_C must satisfy lower < upper")

    knot_times_s = build_external_knot_times_s(
        horizon_s=horizon_s,
        num_knots=int(num_knots),
        knot_time_schedule=knot_time_schedule,
        knot_time_custom_support_tau=custom_support,
    )
    allowed_cold_extrapolation_C = max(0.0, DIRECT_CHARACTERIZATION_COLD_LIMIT_C - float(T_ref_bounds_C[0]))

    base_config = OpenLoopProblemConfig(
        horizon_s=horizon_s,
        cryostage_dt_s=float(simulation_profile.cryostage_dt_s),
        knot_times_s=knot_times_s,
        front_target_speed_m_per_s=float(target.target_front_speed_mm_s) * 1.0e-3,
        tracking_weight=1.0,
        smoothness_weight=float(args.smoothness_weight),
        completion_weight=1.0,
        t_ignore_s=0.0,
        T_ref_bounds_C=(float(T_ref_bounds_C[0]), float(T_ref_bounds_C[1])),
        require_monotone_nonincreasing=bool(trajectory.require_monotone_nonincreasing),
        enforce_characterization_admissibility=not bool(args.no_characterization_admissibility),
        characterization_constraints_dir=DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR,
        allowed_cold_extrapolation_C=allowed_cold_extrapolation_C,
        characterization_temperature_margin_C=(
            velocity_config.temperature_uncertainty.characterization_temperature_margin_C
        ),
        solver_kwargs={
            "geom": GeometryParams(
                R_in=7.5e-3,
                t_wall=2.0e-3,
                t_base=0.0,
                H_fill=15.0e-3,
                H_total=17.0e-3,
            ),
            "Nr": int(simulation_profile.Nr),
            "Nz": int(simulation_profile.Nz),
            "dt": float(simulation_profile.solver_dt_s),
            "pre_cool_s": 0.0,
            "write_every": float(simulation_profile.write_every_s),
            "write_field_output": bool(simulation_profile.write_field_output),
            "write_probe_csv": True,
            "show_progress": bool(simulation_profile.show_progress or args.show_progress),
            "T_fill_C": float(initial.initial_water_temperature_C),
            "bcs": ThermalBCs(T_room_C=5.75, h_top=H_OUT_W_M2K, h_side=H_OUT_W_M2K),
            "phase": PhaseChangeParams(Tf=0.0, L_latent=334000.0, dT_mushy=0.5),
            "prefill": PrefillOptions(mode="steady"),
            "freeze_stop": FreezeStopOptions(mode="fillable_region", extra_subcooling_C=0.0),
            "front_definition_mode": "isotherm_Tf",
            "probe_z_m": PROBE_Z_M,
            "probe_wall_inset_m": PROBE_WALL_INSET_M,
            "Nz_front": int(simulation_profile.Nz_front),
            "enable_front_curve": bool(simulation_profile.enable_front_curve),
            "Nr_front_curve": int(simulation_profile.Nr_front_curve),
            "Nz_front_curve": int(simulation_profile.Nz_front_curve),
            "stop_when_wall_frozen": False,
            "use_tabulated_water_ice": bool(simulation_profile.use_tabulated_water_ice),
        },
        front_reference_mode="linear_full_process",
        incomplete_penalty_value=float(args.incomplete_penalty_value),
    )
    profiled_config = apply_simulation_profile(base_config, simulation_profile)
    solver_kwargs = dict(profiled_config.solver_kwargs)
    solver_kwargs["write_probe_csv"] = True
    solver_kwargs["show_progress"] = bool(simulation_profile.show_progress or args.show_progress)
    return replace(profiled_config, solver_kwargs=solver_kwargs)


def _build_objective_config(
    velocity_config: VelocityControlFileConfig,
    args: argparse.Namespace,
) -> ConstantVelocityObjectiveConfig:
    target = velocity_config.velocity_target
    return ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=float(
            target.target_front_speed_mm_s
            if args.target_front_speed_mm_s is None
            else args.target_front_speed_mm_s
        ),
        control_z_min_mm=float(target.control_z_min_mm),
        control_z_max_mm=float(target.control_z_max_mm),
    )


def _apply_velocity_overrides(
    velocity_config: VelocityControlFileConfig,
    args: argparse.Namespace,
) -> VelocityControlFileConfig:
    run_updates = {}
    if args.simulation_profile is not None:
        run_updates["simulation_profile"] = str(args.simulation_profile)
    if args.run_name is not None:
        run_updates["run_name"] = str(args.run_name)

    target_updates = {}
    if args.target_front_speed_mm_s is not None:
        target_updates["target_front_speed_mm_s"] = float(args.target_front_speed_mm_s)
    velocity_target = (
        replace(velocity_config.velocity_target, **target_updates)
        if target_updates
        else velocity_config.velocity_target
    )

    initial_updates = {}
    if args.initial_water_temperature_C is not None:
        initial_updates["initial_water_temperature_C"] = float(args.initial_water_temperature_C)
    if args.initial_plate_temperature_C is not None:
        initial_updates["initial_plate_temperature_C"] = float(args.initial_plate_temperature_C)
    initial_conditions = (
        replace(velocity_config.initial_conditions, **initial_updates)
        if initial_updates
        else velocity_config.initial_conditions
    )
    return replace(
        velocity_config,
        velocity_target=velocity_target,
        initial_conditions=initial_conditions,
        **run_updates,
    )


def _render_effective_config(
    *,
    run_name: str,
    output_dir: Path,
    velocity_config: VelocityControlFileConfig,
    bo_config: BOFileConfig,
    objective_config: ConstantVelocityObjectiveConfig,
    problem_config: OpenLoopProblemConfig,
    simulation_profile,
    knot_time_schedule: str,
    knot_time_custom_support_tau: tuple[float, ...] | None,
    theta0: tuple[float, ...],
    theta0_source: str,
    theta_bounds_C: tuple[tuple[float, float], ...],
    search_bounds: tuple[tuple[float, float], ...],
    bayesopt_config: BayesianOptimizationConfig,
) -> str:
    initial = velocity_config.initial_conditions
    uncertainty = velocity_config.temperature_uncertainty
    lines = [
        "# Effective velocity-control BO configuration.",
        "",
        "[run]",
        f'run_name = "{run_name}"',
        f'output_dir = "{output_dir}"',
        f'simulation_profile = "{simulation_profile.name}"',
        "",
        "[simulation_profile]",
        f'name = "{simulation_profile.name}"',
        f"cryostage_dt_s = {simulation_profile.cryostage_dt_s:.12g}",
        f"solver_dt_s = {simulation_profile.solver_dt_s:.12g}",
        f"Nr = {int(simulation_profile.Nr)}",
        f"Nz = {int(simulation_profile.Nz)}",
        f"Nz_front = {int(simulation_profile.Nz_front)}",
        f"Nr_front_curve = {int(simulation_profile.Nr_front_curve)}",
        f"Nz_front_curve = {int(simulation_profile.Nz_front_curve)}",
        f"write_field_output = {str(bool(simulation_profile.write_field_output)).lower()}",
        f"write_probe_csv = {str(bool(simulation_profile.write_probe_csv)).lower()}",
        f"enable_front_curve = {str(bool(simulation_profile.enable_front_curve)).lower()}",
        f"use_tabulated_water_ice = {str(bool(simulation_profile.use_tabulated_water_ice)).lower()}",
        "",
        "[velocity_target]",
        f"target_front_speed_mm_s = {objective_config.target_front_speed_mm_s:.12g}",
        f"control_z_min_mm = {objective_config.control_z_min_mm:.12g}",
        f"control_z_max_mm = {objective_config.control_z_max_mm:.12g}",
        "",
        "[initial_conditions]",
        f"initial_water_temperature_C = {initial.initial_water_temperature_C:.12g}",
        f"initial_plate_temperature_C = {initial.initial_plate_temperature_C:.12g}",
        "no_warm_hold = true",
        "",
        "[temperature_uncertainty]",
        (
            "characterization_temperature_margin_C = "
            f"{uncertainty.characterization_temperature_margin_C:.12g}"
        ),
        "",
        "[trajectory]",
        f"horizon_s = {problem_config.horizon_s:.12g}",
        f"num_knots = {len(problem_config.knot_times_s)}",
        f'knot_time_schedule = "{knot_time_schedule}"',
        f"knot_time_custom_support_tau = {list(knot_time_custom_support_tau or [])}",
        f"knot_times_s = {list(problem_config.knot_times_s)}",
        f"theta0_C = {list(theta0)}",
        f'theta0_source = "{theta0_source}"',
        f"T_ref_bounds_C = {list(problem_config.T_ref_bounds_C)}",
        f"theta_bounds_C = {[list(pair) for pair in theta_bounds_C]}",
        f"require_monotone_nonincreasing = {str(problem_config.require_monotone_nonincreasing).lower()}",
        f"allowed_cold_extrapolation_C = {problem_config.allowed_cold_extrapolation_C:.12g}",
        (
            "characterization_temperature_margin_C = "
            f"{problem_config.characterization_temperature_margin_C:.12g}"
        ),
        f"optimize_knot_times = {str(bo_config.optimize_knot_times).lower()}",
        "",
        "[bayesian_optimization]",
        f"random_seed = {bayesopt_config.random_seed}",
        f"init_points = {bayesopt_config.init_points}",
        f"n_iter = {bayesopt_config.n_iter}",
        f'acquisition_kind = "{bayesopt_config.acquisition_kind}"',
        f"acquisition_kappa = {bayesopt_config.acquisition_kappa:.12g}",
        f"acquisition_xi = {bayesopt_config.acquisition_xi:.12g}",
        f"seed_with_theta0 = {str(bayesopt_config.seed_with_theta0).lower()}",
        f'parameterization_kind = "{bayesopt_config.parameterization_kind}"',
        f'init_strategy = "{bayesopt_config.init_strategy}"',
        f"init_local_sigma = {bayesopt_config.init_local_sigma:.12g}",
        f"init_max_attempts_per_point = {bayesopt_config.init_max_attempts_per_point}",
        f"local_refinement_points = {bayesopt_config.local_refinement_points}",
        f"local_refinement_sigma = {bayesopt_config.local_refinement_sigma:.12g}",
        f"search_bounds = {[list(pair) for pair in search_bounds]}",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for constant freezing-front velocity control."
    )
    parser.add_argument("--velocity-config", "--velocity-control-config", dest="velocity_config", type=Path, default=DEFAULT_VELOCITY_CONTROL_CONFIG_PATH)
    parser.add_argument("--bo-config", type=Path, default=DEFAULT_BO_CONFIG_PATH)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIMULATION_CONFIG_PATH)
    parser.add_argument("--simulation-profile", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT_DIR)
    parser.add_argument("--target-front-speed-mm-s", type=float, default=None)
    parser.add_argument("--initial-water-temperature-c", "--initial-water-temperature-C", dest="initial_water_temperature_C", type=float, default=None)
    parser.add_argument("--initial-plate-temperature-c", "--initial-plate-temperature-C", dest="initial_plate_temperature_C", type=float, default=None)
    parser.add_argument("--horizon-s", type=float, default=None)
    parser.add_argument("--num-knots", type=int, default=None)
    parser.add_argument("--theta0-c", default=None, help="Comma-separated initial BO theta temperatures in C.")
    parser.add_argument("--theta0-from-run-dir", type=Path, default=None, help="Load theta0 from best_theta_profile.csv inside a previous BO run directory.")
    parser.add_argument("--theta-bounds", default=None, help="Comma-separated lower:upper pairs, one per optimized knot.")
    parser.add_argument("--t-ref-bounds-c", "--T-ref-bounds-C", dest="T_ref_bounds_C", default=None, help="Comma-separated global lower,upper bounds in C.")
    parser.add_argument("--knot-time-schedule", default=None)
    parser.add_argument("--knot-time-custom-support-tau", default=None)
    parser.add_argument("--init-points", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--acquisition-kind", choices=("ucb", "ei", "poi"), default=None)
    parser.add_argument("--acquisition-kappa", type=float, default=None)
    parser.add_argument("--acquisition-xi", type=float, default=None)
    parser.add_argument("--parameterization-kind", choices=("raw_theta", "monotone_unit_box"), default=None)
    parser.add_argument("--init-strategy", choices=("uniform_random", "feasible_local", "feasible_local_deterministic"), default=None)
    parser.add_argument("--init-local-sigma", type=float, default=None)
    parser.add_argument("--init-max-attempts-per-point", type=int, default=None)
    parser.add_argument("--local-refinement-points", type=int, default=None)
    parser.add_argument("--local-refinement-sigma", type=float, default=None)
    parser.add_argument("--no-seed-theta0", action="store_true")
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--incomplete-penalty-value", type=float, default=2.0)
    parser.add_argument("--no-characterization-admissibility", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bo_config = load_bo_config(args.bo_config)
    velocity_config = _apply_velocity_overrides(load_velocity_control_config(args.velocity_config), args)
    simulation_profile_name = args.simulation_profile or velocity_config.simulation_profile
    simulation_profile = load_simulation_profile(args.simulation_config, profile_name=simulation_profile_name)
    num_knots = _resolve_num_knots(args, bo_config)
    objective_config = _build_objective_config(velocity_config, args)
    problem_config = _build_problem_config(
        velocity_config=velocity_config,
        bo_config=bo_config,
        simulation_profile=simulation_profile,
        args=args,
        num_knots=num_knots,
    )
    theta0, theta0_source = _resolve_theta0(
        args=args,
        bo_config=bo_config,
        num_knots=len(problem_config.knot_times_s),
    )
    cli_theta_bounds = _parse_theta_bounds(args.theta_bounds, num_knots=len(problem_config.knot_times_s))
    theta_bounds_C = cli_theta_bounds if cli_theta_bounds is not None else bo_config.theta_bounds_C
    normalized_theta_bounds = normalize_theta_bounds(
        theta_bounds_C,
        num_variables=len(problem_config.knot_times_s),
        default_bounds_C=problem_config.T_ref_bounds_C,
    )
    bayesopt_config = BayesianOptimizationConfig(
        random_seed=int(bo_config.random_seed if args.seed is None else args.seed),
        init_points=int(bo_config.init_points if args.init_points is None else args.init_points),
        n_iter=int(bo_config.n_iter if args.n_iter is None else args.n_iter),
        acquisition_kind=str(bo_config.acquisition_kind if args.acquisition_kind is None else args.acquisition_kind),
        acquisition_kappa=float(bo_config.acquisition_kappa if args.acquisition_kappa is None else args.acquisition_kappa),
        acquisition_xi=float(bo_config.acquisition_xi if args.acquisition_xi is None else args.acquisition_xi),
        theta_bounds_C=normalized_theta_bounds,
        seed_with_theta0=bool(bo_config.seed_with_theta0) and not bool(args.no_seed_theta0),
        parameterization_kind=str(
            bo_config.parameterization_kind if args.parameterization_kind is None else args.parameterization_kind
        ),
        init_strategy=str(bo_config.init_strategy if args.init_strategy is None else args.init_strategy),
        init_local_sigma=float(bo_config.init_local_sigma if args.init_local_sigma is None else args.init_local_sigma),
        init_max_attempts_per_point=int(
            bo_config.init_max_attempts_per_point
            if args.init_max_attempts_per_point is None
            else args.init_max_attempts_per_point
        ),
        local_refinement_points=int(
            bo_config.local_refinement_points
            if args.local_refinement_points is None
            else args.local_refinement_points
        ),
        local_refinement_sigma=float(
            bo_config.local_refinement_sigma
            if args.local_refinement_sigma is None
            else args.local_refinement_sigma
        ),
    )
    search_bounds = build_candidate_parameterization(
        theta_bounds_C=normalized_theta_bounds,
        parameterization_kind=bayesopt_config.parameterization_kind,
    ).search_bounds
    run_name = args.run_name or DEFAULT_RUN_NAME
    run_output_root = Path(args.output_root) / f"n{num_knots}" / "coarse"
    output_dir = run_output_root / run_name
    knot_time_schedule = str(args.knot_time_schedule or bo_config.knot_time_schedule)
    cli_support = _parse_float_tuple(args.knot_time_custom_support_tau, name="--knot-time-custom-support-tau")
    knot_time_custom_support_tau = cli_support if cli_support is not None else bo_config.knot_time_custom_support_tau
    effective_config_text = _render_effective_config(
        run_name=run_name,
        output_dir=output_dir,
        velocity_config=velocity_config,
        bo_config=bo_config,
        objective_config=objective_config,
        problem_config=problem_config,
        simulation_profile=simulation_profile,
        knot_time_schedule=knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
        theta0=theta0,
        theta0_source=theta0_source,
        theta_bounds_C=normalized_theta_bounds,
        search_bounds=search_bounds,
        bayesopt_config=bayesopt_config,
    )

    if args.dry_run_config:
        print(effective_config_text)
        return

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    def objective_evaluator(theta, config, cryostage_params, out_dir, case_name, *, T_plate0_C=None, prepared_candidate=None):
        return evaluate_velocity_control_objective(
            theta,
            config,
            cryostage_params,
            out_dir,
            case_name,
            objective_config=objective_config,
            T_plate0_C=T_plate0_C,
            prepared_candidate=prepared_candidate,
        )

    result = optimize_open_loop_theta(
        theta0=theta0,
        config=problem_config,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        out_root_dir=run_output_root,
        run_name=run_name,
        method="bayesian-optimization",
        bayesopt_config=bayesopt_config,
        infeasible_objective_penalty=bo_config.infeasible_objective_penalty,
        T_plate0_C=velocity_config.initial_conditions.initial_plate_temperature_C,
        objective_evaluator=objective_evaluator,
    )
    effective_config_path = result.run_dir / "effective_config.toml"
    effective_config_path.write_text(effective_config_text, encoding="utf-8")

    artifacts = finalize_velocity_bo_outputs(
        result=result,
        config=problem_config,
        objective_config=objective_config,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        initial_water_temperature_C=velocity_config.initial_conditions.initial_water_temperature_C,
        initial_plate_temperature_C=velocity_config.initial_conditions.initial_plate_temperature_C,
        simulation_profile_name=simulation_profile.name,
        theta_bounds_C=normalized_theta_bounds,
    )

    print(f"Velocity-control BO written to {result.run_dir.resolve()}")
    print(f"  effective config: {effective_config_path.resolve()}")
    print(f"  bo history      : {artifacts['bo_history'].resolve()}")
    print(f"  best theta      : {artifacts['best_theta_profile'].resolve()}")
    if "best_tracking_summary" in artifacts:
        print(f"  tracking summary: {artifacts['best_tracking_summary'].resolve()}")
    if "plate_tracking_summary" in artifacts:
        print(f"  plate summary   : {artifacts['plate_tracking_summary'].resolve()}")
    print(f"  report          : {artifacts['velocity_control_bo_report'].resolve()}")


if __name__ == "__main__":
    main()
