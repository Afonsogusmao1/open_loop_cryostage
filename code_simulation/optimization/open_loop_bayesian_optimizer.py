from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy  # noqa: F401  # Keep the runtime scipy loaded ahead of the repo-local BO compatibility path.

from code_simulation.core.paths import BAYESOPT_COMPAT_DIR


_BAYESOPT_COMPAT_DIR = BAYESOPT_COMPAT_DIR


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
    parameterization_kind: str = "raw_theta"
    init_strategy: str = "uniform_random"
    init_local_sigma: float = 0.15
    init_max_attempts_per_point: int = 40
    local_refinement_points: int = 0
    local_refinement_sigma: float = 0.08

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
        if self.parameterization_kind not in {"raw_theta", "monotone_unit_box"}:
            raise ValueError("parameterization_kind must be 'raw_theta' or 'monotone_unit_box'")
        if self.init_strategy not in {"uniform_random", "feasible_local", "feasible_local_deterministic"}:
            raise ValueError(
                "init_strategy must be 'uniform_random', 'feasible_local', or 'feasible_local_deterministic'"
            )
        if not math.isfinite(self.init_local_sigma) or self.init_local_sigma <= 0.0:
            raise ValueError("init_local_sigma must be a finite positive value")
        if int(self.init_max_attempts_per_point) <= 0:
            raise ValueError("init_max_attempts_per_point must be a positive integer")
        if int(self.local_refinement_points) < 0:
            raise ValueError("local_refinement_points must be non-negative")
        if not math.isfinite(self.local_refinement_sigma) or self.local_refinement_sigma <= 0.0:
            raise ValueError("local_refinement_sigma must be a finite positive value")


@dataclass(frozen=True)
class CandidateParameterization:
    kind: str
    physical_theta_bounds_C: tuple[tuple[float, float], ...]
    search_bounds: tuple[tuple[float, float], ...]
    feasible_lower_C: tuple[float, ...]
    feasible_upper_C: tuple[float, ...]

    def search_to_physical_theta(self, raw_candidate: tuple[float, ...]) -> tuple[float, ...]:
        raw_candidate = _coerce_candidate_tuple(raw_candidate, name="raw_candidate")
        if len(raw_candidate) != len(self.search_bounds):
            raise ValueError("raw_candidate length must match search_bounds")
        _validate_within_bounds(raw_candidate, self.search_bounds, name="raw_candidate")
        if self.kind == "raw_theta":
            return raw_candidate
        theta_C = [0.0] * len(raw_candidate)
        last_idx = len(raw_candidate) - 1
        lower_last = float(self.feasible_lower_C[last_idx])
        upper_last = float(self.feasible_upper_C[last_idx])
        theta_C[last_idx] = _interp_from_unit_interval(raw_candidate[last_idx], lower_last, upper_last)
        for idx in range(last_idx - 1, -1, -1):
            lower_eff = max(float(self.feasible_lower_C[idx]), float(theta_C[idx + 1]))
            upper_eff = float(self.feasible_upper_C[idx])
            if lower_eff > upper_eff + 1.0e-12:
                raise ValueError("monotone parameterization produced an empty feasible interval")
            theta_C[idx] = _interp_from_unit_interval(raw_candidate[idx], lower_eff, upper_eff)
        return tuple(float(value) for value in theta_C)

    def physical_theta_to_search(self, theta_C: tuple[float, ...]) -> tuple[float, ...]:
        theta_C = _coerce_candidate_tuple(theta_C, name="theta_C")
        if len(theta_C) != len(self.physical_theta_bounds_C):
            raise ValueError("theta_C length must match physical_theta_bounds_C")
        _validate_within_bounds(theta_C, self.physical_theta_bounds_C, name="theta_C")
        if self.kind == "raw_theta":
            return theta_C
        raw_candidate = [0.0] * len(theta_C)
        for idx in range(len(theta_C) - 1, -1, -1):
            lower_eff = float(self.feasible_lower_C[idx])
            if idx < len(theta_C) - 1:
                lower_eff = max(lower_eff, float(theta_C[idx + 1]))
            upper_eff = float(self.feasible_upper_C[idx])
            if float(theta_C[idx]) < lower_eff - 1.0e-12 or float(theta_C[idx]) > upper_eff + 1.0e-12:
                raise ValueError(
                    f"theta_C[{idx}]={theta_C[idx]:.6g} is not representable inside the active monotone parameterization"
                )
            raw_candidate[idx] = _unit_interval_coordinate(theta_C[idx], lower_eff, upper_eff)
        return tuple(float(value) for value in raw_candidate)


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
    parameterization_kind: str
    search_bounds: tuple[tuple[float, float], ...]
    init_strategy: str
    init_local_sigma: float
    init_max_attempts_per_point: int
    init_candidate_attempts: int
    init_precheck_rejections: int
    init_accepted_candidates: int
    init_global_fallback_accepts: int
    local_refinement_points: int
    local_refinement_sigma: float
    local_refinement_candidate_attempts: int
    local_refinement_precheck_rejections: int
    local_refinement_accepted_candidates: int
    local_refinement_improved_candidates: int
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


def _coerce_candidate_tuple(values, *, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(value) for value in arr)


def _validate_within_bounds(
    values: tuple[float, ...],
    bounds: tuple[tuple[float, float], ...],
    *,
    name: str,
) -> None:
    if len(values) != len(bounds):
        raise ValueError(f"{name} length must match bounds length")
    for idx, (value, (lower, upper)) in enumerate(zip(values, bounds, strict=True)):
        if float(value) < float(lower) - 1.0e-12 or float(value) > float(upper) + 1.0e-12:
            raise ValueError(f"{name}[{idx}]={value:.6g} lies outside [{lower:.6g}, {upper:.6g}]")


def _interp_from_unit_interval(value: float, lower: float, upper: float) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if upper < lower - 1.0e-12:
        raise ValueError("lower must be <= upper")
    if upper <= lower + 1.0e-12:
        return float(lower)
    return float(lower + min(max(value, 0.0), 1.0) * (upper - lower))


def _unit_interval_coordinate(value: float, lower: float, upper: float) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if upper < lower - 1.0e-12:
        raise ValueError("lower must be <= upper")
    if upper <= lower + 1.0e-12:
        return 0.0
    return float(min(max((value - lower) / (upper - lower), 0.0), 1.0))


def _monotone_feasible_bounds(
    theta_bounds_C: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    lower = [float(pair[0]) for pair in theta_bounds_C]
    upper = [float(pair[1]) for pair in theta_bounds_C]
    feasible_upper = [0.0] * len(theta_bounds_C)
    feasible_lower = [0.0] * len(theta_bounds_C)
    feasible_upper[0] = upper[0]
    for idx in range(1, len(theta_bounds_C)):
        feasible_upper[idx] = min(upper[idx], feasible_upper[idx - 1])
    feasible_lower[-1] = lower[-1]
    for idx in range(len(theta_bounds_C) - 2, -1, -1):
        feasible_lower[idx] = max(lower[idx], feasible_lower[idx + 1])
    for idx, (lower_eff, upper_eff) in enumerate(zip(feasible_lower, feasible_upper, strict=True)):
        if lower_eff > upper_eff + 1.0e-12:
            raise ValueError(
                "theta bounds do not admit any monotone non-increasing trajectory; "
                f"knot {idx} has effective feasible interval [{lower_eff:.6g}, {upper_eff:.6g}]"
            )
    return tuple(feasible_lower), tuple(feasible_upper)


def build_candidate_parameterization(
    *,
    theta_bounds_C: tuple[tuple[float, float], ...],
    parameterization_kind: str,
) -> CandidateParameterization:
    if parameterization_kind == "raw_theta":
        return CandidateParameterization(
            kind="raw_theta",
            physical_theta_bounds_C=theta_bounds_C,
            search_bounds=theta_bounds_C,
            feasible_lower_C=tuple(float(lower) for lower, _ in theta_bounds_C),
            feasible_upper_C=tuple(float(upper) for _, upper in theta_bounds_C),
        )

    feasible_lower_C, feasible_upper_C = _monotone_feasible_bounds(theta_bounds_C)
    return CandidateParameterization(
        kind="monotone_unit_box",
        physical_theta_bounds_C=theta_bounds_C,
        search_bounds=tuple((0.0, 1.0) for _ in theta_bounds_C),
        feasible_lower_C=feasible_lower_C,
        feasible_upper_C=feasible_upper_C,
    )


def _sample_uniform_raw_candidate(
    *,
    rng: np.random.RandomState,
    search_bounds: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    return tuple(float(rng.uniform(lower, upper)) for lower, upper in search_bounds)


def _van_der_corput(index: int, base: int) -> float:
    if int(index) <= 0:
        raise ValueError("index must be positive")
    if int(base) <= 1:
        raise ValueError("base must be greater than one")
    value = 0.0
    denominator = 1.0
    n = int(index)
    while n > 0:
        n, remainder = divmod(n, int(base))
        denominator *= float(base)
        value += float(remainder) / denominator
    return float(value)


def _first_primes(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < int(count):
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _sample_deterministic_uniform_raw_candidate(
    *,
    sample_index: int,
    search_bounds: tuple[tuple[float, float], ...],
) -> tuple[float, ...]:
    primes = _first_primes(len(search_bounds))
    values: list[float] = []
    for dim_idx, (lower, upper) in enumerate(search_bounds):
        unit_value = _van_der_corput(int(sample_index) + 1, primes[dim_idx])
        values.append(float(lower + unit_value * (upper - lower)))
    return tuple(values)


def _sample_local_raw_candidate(
    *,
    center: tuple[float, ...],
    rng: np.random.RandomState,
    search_bounds: tuple[tuple[float, float], ...],
    sigma: float,
) -> tuple[float, ...]:
    values: list[float] = []
    for center_value, (lower, upper) in zip(center, search_bounds, strict=True):
        width = float(upper - lower)
        proposal = float(center_value) + float(sigma) * width * float(rng.normal())
        values.append(float(min(max(proposal, lower), upper)))
    return tuple(values)


def _sample_deterministic_local_raw_candidate(
    *,
    center: tuple[float, ...],
    search_bounds: tuple[tuple[float, float], ...],
    sigma: float,
    sample_index: int,
) -> tuple[float, ...]:
    if int(sample_index) < 0:
        raise ValueError("sample_index must be non-negative")
    num_dims = len(center)
    if num_dims == 0:
        raise ValueError("center must contain at least one value")
    pattern_bits = min(num_dims, 16)
    pattern_period = 1 << pattern_bits
    pattern_index = int(sample_index) % pattern_period
    shrink_level = int(sample_index) // pattern_period
    scale = float(sigma) / float(max(math.sqrt(num_dims), 1.0)) / float(shrink_level + 1)
    values: list[float] = []
    for dim_idx, (center_value, (lower, upper)) in enumerate(zip(center, search_bounds, strict=True)):
        bit_index = dim_idx if dim_idx < pattern_bits else dim_idx % pattern_bits
        sign = 1.0 if ((pattern_index >> bit_index) & 1) else -1.0
        proposal = float(center_value) + sign * scale * float(upper - lower)
        values.append(float(min(max(proposal, lower), upper)))
    return tuple(values)


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
    evaluate_candidate: Callable[..., float],
    precheck_candidate: Callable[[tuple[float, ...]], object] | None = None,
) -> BayesianOptimizationBackendResult:
    theta0 = tuple(float(value) for value in theta0)
    parameter_names = theta_parameter_names(len(theta0))
    theta_bounds_C = normalize_theta_bounds(
        config.theta_bounds_C,
        num_variables=len(theta0),
        default_bounds_C=default_bounds_C,
    )
    parameterization = build_candidate_parameterization(
        theta_bounds_C=theta_bounds_C,
        parameterization_kind=str(config.parameterization_kind),
    )
    seed_raw_candidate = parameterization.physical_theta_to_search(theta0)

    for idx, (value_C, (lower_C, upper_C)) in enumerate(zip(theta0, theta_bounds_C, strict=True)):
        if value_C < lower_C - 1e-12 or value_C > upper_C + 1e-12:
            raise ValueError(
                f"theta0[{idx}]={value_C:.6g} lies outside the BO bounds [{lower_C:.6g}, {upper_C:.6g}]"
            )

    bayes_opt, BayesianOptimization, UtilityFunction = _import_bayes_opt()
    pbounds = {
        name: (float(lower_C), float(upper_C))
        for name, (lower_C, upper_C) in zip(parameter_names, parameterization.search_bounds, strict=True)
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
    init_candidate_attempts = 0
    init_precheck_rejections = 0
    init_accepted_candidates = 0
    init_global_fallback_accepts = 0
    local_refinement_candidate_attempts = 0
    local_refinement_precheck_rejections = 0
    local_refinement_accepted_candidates = 0
    local_refinement_improved_candidates = 0
    deterministic_local_sample_index = 0
    deterministic_global_sample_index = 0

    def _decode_and_precheck(raw_candidate: tuple[float, ...]) -> tuple[float, ...]:
        theta_candidate = parameterization.search_to_physical_theta(raw_candidate)
        if precheck_candidate is not None:
            precheck_candidate(theta_candidate)
        return theta_candidate

    if config.seed_with_theta0:
        seed_params = theta_to_parameter_dict(seed_raw_candidate, parameter_names)
        seed_objective_value = float(evaluate_candidate(theta0, "seed", raw_candidate=seed_raw_candidate))
        optimizer.register(params=seed_params, target=-seed_objective_value)
        nfev += 1

    for _ in range(int(config.init_points)):
        if config.init_strategy == "uniform_random":
            raw_candidate = _sample_uniform_raw_candidate(
                rng=rng,
                search_bounds=parameterization.search_bounds,
            )
            theta_candidate = parameterization.search_to_physical_theta(raw_candidate)
            objective_value = float(
                evaluate_candidate(theta_candidate, "random", raw_candidate=raw_candidate)
            )
            optimizer.register(
                params=theta_to_parameter_dict(raw_candidate, parameter_names),
                target=-objective_value,
            )
            init_candidate_attempts += 1
            init_accepted_candidates += 1
            nfev += 1
            continue

        deterministic_local = config.init_strategy == "feasible_local_deterministic"
        accepted = False
        for _local_attempt in range(int(config.init_max_attempts_per_point)):
            init_candidate_attempts += 1
            if deterministic_local:
                raw_candidate = _sample_deterministic_local_raw_candidate(
                    center=seed_raw_candidate,
                    search_bounds=parameterization.search_bounds,
                    sigma=float(config.init_local_sigma),
                    sample_index=deterministic_local_sample_index,
                )
                deterministic_local_sample_index += 1
            else:
                raw_candidate = _sample_local_raw_candidate(
                    center=seed_raw_candidate,
                    rng=rng,
                    search_bounds=parameterization.search_bounds,
                    sigma=float(config.init_local_sigma),
                )
            try:
                theta_candidate = _decode_and_precheck(raw_candidate)
            except Exception:
                init_precheck_rejections += 1
                continue
            objective_value = float(
                evaluate_candidate(theta_candidate, "init_local", raw_candidate=raw_candidate)
            )
            optimizer.register(
                params=theta_to_parameter_dict(raw_candidate, parameter_names),
                target=-objective_value,
            )
            init_accepted_candidates += 1
            nfev += 1
            accepted = True
            break
        if accepted:
            continue

        for _global_attempt in range(int(config.init_max_attempts_per_point)):
            init_candidate_attempts += 1
            if deterministic_local:
                raw_candidate = _sample_deterministic_uniform_raw_candidate(
                    sample_index=deterministic_global_sample_index,
                    search_bounds=parameterization.search_bounds,
                )
                deterministic_global_sample_index += 1
            else:
                raw_candidate = _sample_uniform_raw_candidate(
                    rng=rng,
                    search_bounds=parameterization.search_bounds,
                )
            try:
                theta_candidate = _decode_and_precheck(raw_candidate)
            except Exception:
                init_precheck_rejections += 1
                continue
            objective_value = float(
                evaluate_candidate(theta_candidate, "init_global", raw_candidate=raw_candidate)
            )
            optimizer.register(
                params=theta_to_parameter_dict(raw_candidate, parameter_names),
                target=-objective_value,
            )
            init_accepted_candidates += 1
            init_global_fallback_accepts += 1
            nfev += 1
            accepted = True
            break

        if not accepted:
            raise RuntimeError(
                "Unable to generate a feasible BO init candidate after "
                f"{config.init_max_attempts_per_point} local attempts and "
                f"{config.init_max_attempts_per_point} global fallback attempts"
            )

    for _ in range(int(config.n_iter)):
        utility.update_params()
        suggestion = optimizer.suggest(utility)
        suggested_raw_candidate = parameter_dict_to_theta(suggestion, parameter_names)
        suggested_theta = parameterization.search_to_physical_theta(suggested_raw_candidate)
        objective_value = float(
            evaluate_candidate(suggested_theta, "bayes", raw_candidate=suggested_raw_candidate)
        )
        optimizer.register(params=suggestion, target=-objective_value)
        nfev += 1

    if int(config.local_refinement_points) > 0 and optimizer.max is not None:
        incumbent_params = optimizer.max.get("params", {})
        if incumbent_params:
            incumbent_raw_candidate = parameter_dict_to_theta(incumbent_params, parameter_names)
            seen_candidates = {
                tuple(round(float(value), 12) for value in incumbent_raw_candidate),
            }
            refinement_sample_index = 0
            max_refinement_attempts = max(
                int(config.local_refinement_points) * 8,
                int(config.local_refinement_points) + 4,
            )
            while (
                local_refinement_accepted_candidates < int(config.local_refinement_points)
                and local_refinement_candidate_attempts < max_refinement_attempts
            ):
                local_refinement_candidate_attempts += 1
                raw_candidate = _sample_deterministic_local_raw_candidate(
                    center=incumbent_raw_candidate,
                    search_bounds=parameterization.search_bounds,
                    sigma=float(config.local_refinement_sigma),
                    sample_index=refinement_sample_index,
                )
                refinement_sample_index += 1
                candidate_key = tuple(round(float(value), 12) for value in raw_candidate)
                if candidate_key in seen_candidates:
                    continue
                seen_candidates.add(candidate_key)
                try:
                    theta_candidate = _decode_and_precheck(raw_candidate)
                except Exception:
                    local_refinement_precheck_rejections += 1
                    continue
                previous_best_target = optimizer.max["target"] if optimizer.max is not None else -math.inf
                objective_value = float(
                    evaluate_candidate(theta_candidate, "refine_local", raw_candidate=raw_candidate)
                )
                optimizer.register(
                    params=theta_to_parameter_dict(raw_candidate, parameter_names),
                    target=-objective_value,
                )
                nfev += 1
                local_refinement_accepted_candidates += 1
                current_best_target = optimizer.max["target"] if optimizer.max is not None else previous_best_target
                if current_best_target > previous_best_target + 1.0e-12:
                    local_refinement_improved_candidates += 1
                    incumbent_raw_candidate = parameter_dict_to_theta(optimizer.max["params"], parameter_names)
                    seen_candidates.add(tuple(round(float(value), 12) for value in incumbent_raw_candidate))

    best_target = optimizer.max["target"] if optimizer.max is not None else math.nan
    best_params = optimizer.max["params"] if optimizer.max is not None else {}
    message = (
        f"Completed BO loop with {nfev} evaluations; best registered target={best_target:.9e}; "
        f"best params={best_params}; init_attempts={init_candidate_attempts}; "
        f"init_precheck_rejections={init_precheck_rejections}; init_accepted={init_accepted_candidates}; "
        f"refine_attempts={local_refinement_candidate_attempts}; "
        f"refine_accepted={local_refinement_accepted_candidates}; "
        f"refine_improved={local_refinement_improved_candidates}"
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
        parameterization_kind=str(config.parameterization_kind),
        search_bounds=parameterization.search_bounds,
        init_strategy=str(config.init_strategy),
        init_local_sigma=float(config.init_local_sigma),
        init_max_attempts_per_point=int(config.init_max_attempts_per_point),
        init_candidate_attempts=int(init_candidate_attempts),
        init_precheck_rejections=int(init_precheck_rejections),
        init_accepted_candidates=int(init_accepted_candidates),
        init_global_fallback_accepts=int(init_global_fallback_accepts),
        local_refinement_points=int(config.local_refinement_points),
        local_refinement_sigma=float(config.local_refinement_sigma),
        local_refinement_candidate_attempts=int(local_refinement_candidate_attempts),
        local_refinement_precheck_rejections=int(local_refinement_precheck_rejections),
        local_refinement_accepted_candidates=int(local_refinement_accepted_candidates),
        local_refinement_improved_candidates=int(local_refinement_improved_candidates),
        package_version=str(getattr(bayes_opt, "__version__", "unknown")),
        package_path=Path(bayes_opt.__file__).resolve(),
    )


__all__ = [
    "BayesianOptimizationBackendResult",
    "BayesianOptimizationConfig",
    "CandidateParameterization",
    "bayes_opt_runtime_details",
    "build_candidate_parameterization",
    "normalize_theta_bounds",
    "parameter_dict_to_theta",
    "run_bayesian_optimization",
    "theta_parameter_names",
    "theta_to_parameter_dict",
]
