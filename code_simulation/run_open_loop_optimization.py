from __future__ import annotations

import argparse
from pathlib import Path

from cryostage_model import DEFAULT_CRYOSTAGE_PARAMS
from geometry import GeometryParams
from open_loop_optimizer import optimize_open_loop_theta
from open_loop_problem import OpenLoopProblemConfig, build_reference_profile_from_theta
from solver import FreezeStopOptions, PrefillOptions, ThermalBCs


DEFAULT_RUN_NAME = "baseline_h180_dt2_fill12p5_h2"
DEFAULT_T_REF_BOUNDS_C = (-20.0, 0.0)
DEFAULT_REQUIRE_MONOTONE_NONINCREASING = True
DEFAULT_THETA0 = (0.0, 0.0, -6.0, -12.0, -18.0)


def build_problem_config() -> OpenLoopProblemConfig:
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible open-loop optimization.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder name inside the output root.")
    parser.add_argument(
        "--out-root-dir",
        default=str(Path(__file__).resolve().parent / "results" / "open_loop_optimization"),
        help="Root folder where optimization runs are written.",
    )
    parser.add_argument("--method", default="Nelder-Mead", help="scipy.optimize.minimize method.")
    parser.add_argument("--maxiter", type=int, default=15, help="Maximum optimizer iterations.")
    parser.add_argument("--maxfev", type=int, default=40, help="Maximum objective evaluations.")
    parser.add_argument("--xatol", type=float, default=0.25, help="Absolute theta convergence tolerance.")
    parser.add_argument("--fatol", type=float, default=1.0e-9, help="Absolute objective convergence tolerance.")
    return parser.parse_args()


def _theta_satisfies_active_constraints(theta, config: OpenLoopProblemConfig) -> bool:
    try:
        build_reference_profile_from_theta(theta, config)
    except Exception:
        return False
    return True


def main() -> None:
    args = parse_args()
    config = build_problem_config()
    bcs = config.solver_kwargs["bcs"]
    T_fill_C = float(config.solver_kwargs["T_fill_C"])

    print(
        "Using canonical cryostage parameters "
        f"(tau_s={DEFAULT_CRYOSTAGE_PARAMS.tau_s:.6f}, "
        f"gain={DEFAULT_CRYOSTAGE_PARAMS.gain:.6f}, "
        f"offset_C={DEFAULT_CRYOSTAGE_PARAMS.offset_C:.6f})"
    )
    print(f"Active T_ref bounds = {config.T_ref_bounds_C}")
    print(f"Monotonicity required = {config.require_monotone_nonincreasing}")
    print(
        "Runner thermal settings "
        f"(T_fill_C={T_fill_C:.6f}, h_top={bcs.h_top:.6f}, h_side={bcs.h_side:.6f})"
    )
    print(f"Initial theta = {DEFAULT_THETA0}")

    result = optimize_open_loop_theta(
        theta0=DEFAULT_THETA0,
        config=config,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        out_root_dir=Path(args.out_root_dir),
        run_name=args.run_name,
        method=args.method,
        options={
            "maxiter": args.maxiter,
            "maxfev": args.maxfev,
            "xatol": args.xatol,
            "fatol": args.fatol,
        },
    )

    total_evaluations = len(result.history)
    valid_evaluations = sum(1 for entry in result.history if entry.is_valid)
    best_theta_is_valid = _theta_satisfies_active_constraints(result.best_theta, config)

    print("Optimization summary")
    print(f"Best theta: {result.best_theta}")
    print(f"Best theta satisfies active constraints: {best_theta_is_valid}")
    print(f"Best J: {result.best_objective_value:.9e}")
    print(f"Evaluations: total={total_evaluations}, valid={valid_evaluations}, scipy_nfev={result.nfev}")
    print(
        f"Optimizer termination: success={result.success}, status={result.status}, "
        f"message={result.message}"
    )
    print(f"Evaluation history: {result.history_csv_path}")
    print(f"Best run folder: {result.best_dir}")
    print(f"Output folder: {result.run_dir}")


if __name__ == "__main__":
    main()
