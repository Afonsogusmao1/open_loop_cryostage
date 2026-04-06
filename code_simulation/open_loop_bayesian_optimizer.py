from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy  # noqa: F401  # Keep the runtime scipy loaded ahead of the repo-local BO compatibility path.


_BAYESOPT_COMPAT_DIR = Path(__file__).resolve().parent / "_vendor_bayesopt_compat"


@dataclass(frozen=True)
class BayesianOptimizationConfig:
    random_seed: int = 17
    init_points: int = 4
    n_iter: int = 12
    acquisition_kind: str = "ucb"
    acquisition_kappa: float = 2.576
    acquisition_xi: float = 0.0
    theta_bounds_C: tuple[tuple[float, float], ...] | None = None
    seed_with_theta0: bool = True

    def __post_init__(self) -> None:
        if self.init_points < 0:
            raise ValueError("init_points must be non-negative")
        if self.n_iter < 0:
            raise ValueError("n_iter must be non-negative")
        if self.acquisition_kind not in {"ucb", "ei", "poi"}:
            raise ValueError("acquisition_kind must be one of {'ucb', 'ei', 'poi'}")
        if not math.isfinite(self.acquisition_kappa) or self.acquisition_kappa < 0.0:
            raise ValueError("acquisition_kappa must be a finite non-negative value")
        if not math.isfinite(self.acquisition_xi) or self.acquisition_xi < 0.0:
            raise ValueError("acquisition_xi must be a finite non-negative value")


@dataclass(frozen=True)
class BayesianOptimizationBackendResult:
    method: str
    success: bool
    status: int
    message: str
    nfev: int
    nit: int
    random_seed: int
    init_points: int
    n_iter: int
    acquisition_kind: str
    acquisition_kappa: float
    acquisition_xi: float
    theta_bounds_C: tuple[tuple[float, float], ...]
    package_version: str
    package_path: Path


def _import_bayes_opt():
    compat_path = str(_BAYESOPT_COMPAT_DIR)
    if _BAYESOPT_COMPAT_DIR.exists() and compat_path not in sys.path:
        sys.path.append(compat_path)

    try:
        import bayes_opt
        from bayes_opt import BayesianOptimization, UtilityFunction
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "bayesian-optimization is not importable. Install the package or populate "
            f"{_BAYESOPT_COMPAT_DIR} before running the BO backend."
        ) from exc

    return bayes_opt, BayesianOptimization, UtilityFunction


def bayes_opt_runtime_details() -> dict[str, str]:
    bayes_opt, _, _ = _import_bayes_opt()
    return {
        "package_version": str(getattr(bayes_opt, "__version__", "unknown")),
        "package_path": str(Path(bayes_opt.__file__).resolve()),
        "compat_dir": str(_BAYESOPT_COMPAT_DIR.resolve()),
    }


def theta_parameter_names(num_variables: int) -> tuple[str, ...]:
    if num_variables <= 0:
        raise ValueError("num_variables must be positive")
    return tuple(f"theta_{i}" for i in range(int(num_variables)))


def normalize_theta_bounds(
    theta_bounds_C: tuple[tuple[float, float], ...] | None,
    *,
    num_variables: int,
    default_bounds_C: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    lower_default, upper_default = (float(default_bounds_C[0]), float(default_bounds_C[1]))
    if not (math.isfinite(lower_default) and math.isfinite(upper_default) and lower_default < upper_default):
        raise ValueError("default_bounds_C must contain finite values with lower < upper")

    if theta_bounds_C is None:
        return tuple((lower_default, upper_default) for _ in range(int(num_variables)))

    if len(theta_bounds_C) != int(num_variables):
        raise ValueError(
            "theta_bounds_C must contain exactly one (lower, upper) pair per optimized knot "
            f"({num_variables} expected, got {len(theta_bounds_C)})"
        )

    normalized: list[tuple[float, float]] = []
    for idx, pair in enumerate(theta_bounds_C):
        if len(pair) != 2:
            raise ValueError(f"theta_bounds_C[{idx}] must contain exactly two values")
        lower_C = float(pair[0])
        upper_C = float(pair[1])
        if not (math.isfinite(lower_C) and math.isfinite(upper_C) and lower_C < upper_C):
            raise ValueError(f"theta_bounds_C[{idx}] must satisfy lower < upper with finite values")
        if lower_C < lower_default - 1e-12 or upper_C > upper_default + 1e-12:
            raise ValueError(
                f"theta_bounds_C[{idx}]={pair!r} lies outside the active T_ref bounds {default_bounds_C!r}"
            )
        normalized.append((lower_C, upper_C))
    return tuple(normalized)


def theta_to_parameter_dict(theta: tuple[float, ...], parameter_names: tuple[str, ...]) -> dict[str, float]:
    if len(theta) != len(parameter_names):
        raise ValueError("theta and parameter_names must have the same length")
    return {name: float(value) for name, value in zip(parameter_names, theta, strict=True)}


def parameter_dict_to_theta(params: dict[str, float], parameter_names: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(float(params[name]) for name in parameter_names)


def run_bayesian_optimization(
    *,
    theta0: tuple[float, ...],
    default_bounds_C: tuple[float, float],
    config: BayesianOptimizationConfig,
    evaluate_candidate: Callable[[tuple[float, ...], str], float],
) -> BayesianOptimizationBackendResult:
    theta0 = tuple(float(value) for value in theta0)
    parameter_names = theta_parameter_names(len(theta0))
    theta_bounds_C = normalize_theta_bounds(
        config.theta_bounds_C,
        num_variables=len(theta0),
        default_bounds_C=default_bounds_C,
    )

    for idx, (value_C, (lower_C, upper_C)) in enumerate(zip(theta0, theta_bounds_C, strict=True)):
        if value_C < lower_C - 1e-12 or value_C > upper_C + 1e-12:
            raise ValueError(
                f"theta0[{idx}]={value_C:.6g} lies outside the BO bounds [{lower_C:.6g}, {upper_C:.6g}]"
            )

    bayes_opt, BayesianOptimization, UtilityFunction = _import_bayes_opt()
    pbounds = {
        name: (float(lower_C), float(upper_C))
        for name, (lower_C, upper_C) in zip(parameter_names, theta_bounds_C, strict=True)
    }
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=int(config.random_seed),
        verbose=0,
        allow_duplicate_points=True,
    )
    utility = UtilityFunction(
        kind=config.acquisition_kind,
        kappa=float(config.acquisition_kappa),
        xi=float(config.acquisition_xi),
    )

    rng = np.random.RandomState(int(config.random_seed))
    nfev = 0

    if config.seed_with_theta0:
        seed_params = theta_to_parameter_dict(theta0, parameter_names)
        seed_objective_value = float(evaluate_candidate(theta0, "seed"))
        optimizer.register(params=seed_params, target=-seed_objective_value)
        nfev += 1

    for _ in range(int(config.init_points)):
        random_theta = tuple(
            float(rng.uniform(lower_C, upper_C))
            for lower_C, upper_C in theta_bounds_C
        )
        random_params = theta_to_parameter_dict(random_theta, parameter_names)
        objective_value = float(evaluate_candidate(random_theta, "random"))
        optimizer.register(params=random_params, target=-objective_value)
        nfev += 1

    for _ in range(int(config.n_iter)):
        utility.update_params()
        suggestion = optimizer.suggest(utility)
        suggested_theta = parameter_dict_to_theta(suggestion, parameter_names)
        objective_value = float(evaluate_candidate(suggested_theta, "bayes"))
        optimizer.register(params=suggestion, target=-objective_value)
        nfev += 1

    best_target = optimizer.max["target"] if optimizer.max is not None else math.nan
    best_params = optimizer.max["params"] if optimizer.max is not None else {}
    message = (
        f"Completed BO loop with {nfev} evaluations; best registered target={best_target:.9e}; "
        f"best params={best_params}"
    )
    return BayesianOptimizationBackendResult(
        method="bayesian-optimization",
        success=True,
        status=0,
        message=message,
        nfev=int(nfev),
        nit=int(config.n_iter),
        random_seed=int(config.random_seed),
        init_points=int(config.init_points),
        n_iter=int(config.n_iter),
        acquisition_kind=str(config.acquisition_kind),
        acquisition_kappa=float(config.acquisition_kappa),
        acquisition_xi=float(config.acquisition_xi),
        theta_bounds_C=theta_bounds_C,
        package_version=str(getattr(bayes_opt, "__version__", "unknown")),
        package_path=Path(bayes_opt.__file__).resolve(),
    )


__all__ = [
    "BayesianOptimizationBackendResult",
    "BayesianOptimizationConfig",
    "bayes_opt_runtime_details",
    "normalize_theta_bounds",
    "parameter_dict_to_theta",
    "run_bayesian_optimization",
    "theta_parameter_names",
    "theta_to_parameter_dict",
]
