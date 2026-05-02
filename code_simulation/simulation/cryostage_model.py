from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from code_simulation.core.arrays import as_float_array as _as_float_array
from code_simulation.core.paths import project_root as default_project_root


def _validate_time_grid(time_s: np.ndarray, *, name: str = "time_s") -> None:
    if time_s.size == 0:
        return
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class CryostageModelParams:
    tau_s: float
    gain: float = 1.0
    offset_C: float = 0.0
    reference_temperatures_C: tuple[float, ...] = ()
    response_tau_s: tuple[float, ...] = ()
    steady_plate_C: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_s) or self.tau_s <= 0.0:
            raise ValueError("tau_s must be a finite positive value")
        if not math.isfinite(self.gain):
            raise ValueError("gain must be finite")
        if not math.isfinite(self.offset_C):
            raise ValueError("offset_C must be finite")

        reference_temperatures_C = tuple(float(value) for value in self.reference_temperatures_C)
        response_tau_s = tuple(float(value) for value in self.response_tau_s)
        steady_plate_C = tuple(float(value) for value in self.steady_plate_C)
        object.__setattr__(self, "reference_temperatures_C", reference_temperatures_C)
        object.__setattr__(self, "response_tau_s", response_tau_s)
        object.__setattr__(self, "steady_plate_C", steady_plate_C)

        lookup_lengths = {
            len(reference_temperatures_C),
            len(response_tau_s),
            len(steady_plate_C),
        }
        if lookup_lengths != {0}:
            if len(lookup_lengths) != 1 or len(reference_temperatures_C) < 2:
                raise ValueError(
                    "reference_temperatures_C, response_tau_s, and steady_plate_C "
                    "must all be empty or have the same length >= 2"
                )
            if any(not math.isfinite(value) for value in reference_temperatures_C):
                raise ValueError("reference_temperatures_C must contain only finite values")
            if any((not math.isfinite(value) or value <= 0.0) for value in response_tau_s):
                raise ValueError("response_tau_s must contain only finite positive values")
            if any(not math.isfinite(value) for value in steady_plate_C):
                raise ValueError("steady_plate_C must contain only finite values")
            if any(
                right <= left
                for left, right in zip(reference_temperatures_C[:-1], reference_temperatures_C[1:])
            ):
                raise ValueError("reference_temperatures_C must be strictly increasing")

    @property
    def uses_temperature_lookup(self) -> bool:
        return bool(self.reference_temperatures_C)

    def response_tau_for_reference_C(self, T_ref_C: float) -> float:
        if not self.uses_temperature_lookup:
            return float(self.tau_s)
        tau_s = _interp1d_with_linear_extrapolation(
            float(T_ref_C),
            self.reference_temperatures_C,
            self.response_tau_s,
        )
        return max(float(tau_s), 1e-9)

    def steady_plate_for_reference_C(self, T_ref_C: float) -> float:
        if not self.uses_temperature_lookup:
            return self.gain * float(T_ref_C) + self.offset_C
        return _interp1d_with_linear_extrapolation(
            float(T_ref_C),
            self.reference_temperatures_C,
            self.steady_plate_C,
        )


def _interp1d_with_linear_extrapolation(
    x: float,
    xp: tuple[float, ...],
    fp: tuple[float, ...],
) -> float:
    if x <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


@dataclass(frozen=True)
class CharacterizationRun:
    name: str
    time_s: np.ndarray
    T_ref_C: np.ndarray
    T_plate_C: np.ndarray

    def __post_init__(self) -> None:
        time_s = _as_float_array(self.time_s, name="time_s")
        T_ref_C = _as_float_array(self.T_ref_C, name="T_ref_C")
        T_plate_C = _as_float_array(self.T_plate_C, name="T_plate_C")

        if time_s.size == 0:
            raise ValueError("time_s must contain at least one sample")
        if T_ref_C.shape != time_s.shape or T_plate_C.shape != time_s.shape:
            raise ValueError("time_s, T_ref_C, and T_plate_C must have the same length")
        _validate_time_grid(time_s)

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "T_ref_C", T_ref_C)
        object.__setattr__(self, "T_plate_C", T_plate_C)


def _simulate_plate_temperature_from_samples(
    time_s: np.ndarray,
    T_ref_samples_C: np.ndarray,
    params: CryostageModelParams,
    T_plate0_C: float,
) -> np.ndarray:
    time_s = _as_float_array(time_s, name="time_s")
    T_ref_samples_C = _as_float_array(T_ref_samples_C, name="T_ref_samples_C")
    if time_s.shape != T_ref_samples_C.shape:
        raise ValueError("time_s and T_ref_samples_C must have the same length")
    _validate_time_grid(time_s)

    T_plate_C = np.empty_like(time_s, dtype=np.float64)
    if time_s.size == 0:
        return T_plate_C

    T_plate_C[0] = float(T_plate0_C)
    for i in range(1, time_s.size):
        dt_s = float(time_s[i] - time_s[i - 1])
        T_ref_C = float(T_ref_samples_C[i - 1])
        alpha = math.exp(-dt_s / params.response_tau_for_reference_C(T_ref_C))
        T_plate_ss_C = params.steady_plate_for_reference_C(T_ref_C)
        T_plate_C[i] = alpha * T_plate_C[i - 1] + (1.0 - alpha) * T_plate_ss_C
    return T_plate_C


def simulate_plate_temperature(
    time_s,
    T_ref_profile_C,
    params: CryostageModelParams,
    T_plate0_C: float,
) -> np.ndarray:
    time_s = _as_float_array(time_s, name="time_s")
    _validate_time_grid(time_s)
    T_ref_samples_C = np.array([float(T_ref_profile_C(float(ti))) for ti in time_s], dtype=np.float64)
    return _simulate_plate_temperature_from_samples(
        time_s=time_s,
        T_ref_samples_C=T_ref_samples_C,
        params=params,
        T_plate0_C=T_plate0_C,
    )


def simulate_characterization_run(
    run: CharacterizationRun,
    params: CryostageModelParams,
    T_plate0_C: float | None = None,
) -> np.ndarray:
    if T_plate0_C is None:
        T_plate0_C = float(run.T_plate_C[0])
    return _simulate_plate_temperature_from_samples(
        time_s=run.time_s,
        T_ref_samples_C=run.T_ref_C,
        params=params,
        T_plate0_C=T_plate0_C,
    )


def load_characterization_run(
    path: str | Path,
    *,
    time_col: str = "panel_t_s",
    T_ref_col: str = "set",
    T_plate_col: str = "T_cal",
    power_col: str = "power",
    row_type: str = "telemetry",
    active_power_threshold: float | None = None,
) -> CharacterizationRun:
    path = Path(path)
    if active_power_threshold is not None:
        active_power_threshold = float(active_power_threshold)
        if not math.isfinite(active_power_threshold):
            raise ValueError("active_power_threshold must be finite")

    rows: list[tuple[float, float, float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if row_type and str(row.get("row_type", "")).lower() != row_type.lower():
                continue
            try:
                t_s = float(row[time_col])
                T_ref_C = float(row[T_ref_col])
                T_plate_C = float(row[T_plate_col])
                power = float(row[power_col]) if active_power_threshold is not None else 0.0
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((t_s, T_ref_C, T_plate_C, power))

    if not rows:
        raise ValueError(f"{path.name} has no valid rows for the requested columns")

    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    unique_mask = np.ones(arr.shape[0], dtype=bool)
    unique_mask[1:] = np.diff(arr[:, 0]) > 0.0
    arr = arr[unique_mask]

    if active_power_threshold is not None:
        active_indices = np.flatnonzero(arr[:, 3] > active_power_threshold)
        if active_indices.size == 0:
            raise ValueError(
                f"{path.name} has no samples with {power_col} > {active_power_threshold:g}"
            )
        arr = arr[int(active_indices[0]) :]

    arr[:, 0] -= arr[0, 0]
    return CharacterizationRun(
        name=path.stem,
        time_s=arr[:, 0],
        T_ref_C=arr[:, 1],
        T_plate_C=arr[:, 2],
    )


def fit_first_order_model(
    runs: Iterable[CharacterizationRun],
    *,
    tau_bounds_s: tuple[float, float] = (5.0, 300.0),
    num_tau: int = 250,
) -> CryostageModelParams:
    runs = list(runs)
    if not runs:
        raise ValueError("runs must contain at least one characterization run")
    if num_tau < 2:
        raise ValueError("num_tau must be at least 2")

    tau_min_s = float(tau_bounds_s[0])
    tau_max_s = float(tau_bounds_s[1])
    if tau_min_s <= 0.0 or tau_max_s <= tau_min_s:
        raise ValueError("tau_bounds_s must satisfy 0 < tau_min_s < tau_max_s")

    tau_grid_s = np.geomspace(tau_min_s, tau_max_s, int(num_tau))
    best_params: CryostageModelParams | None = None
    best_rmse = math.inf

    for tau_s in tau_grid_s:
        phi_rows = []
        target_rows = []
        for run in runs:
            dt_s = np.diff(run.time_s)
            alpha = np.exp(-dt_s / tau_s)
            target_rows.append(run.T_plate_C[1:] - alpha * run.T_plate_C[:-1])
            phi_rows.append(np.column_stack(((1.0 - alpha) * run.T_ref_C[:-1], 1.0 - alpha)))

        Phi = np.vstack(phi_rows)
        target = np.concatenate(target_rows)
        coeffs, *_ = np.linalg.lstsq(Phi, target, rcond=None)
        residual = Phi @ coeffs - target
        rmse = math.sqrt(float(np.mean(residual * residual)))
        params = CryostageModelParams(
            tau_s=float(tau_s),
            gain=float(coeffs[0]),
            offset_C=float(coeffs[1]),
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    if best_params is None:
        raise RuntimeError("first-order fit failed")
    return best_params


def fit_first_order_step_response_model(
    runs: Iterable[CharacterizationRun],
    *,
    tau_bounds_s: tuple[float, float] = (20.0, 250.0),
    num_tau: int = 800,
    refinement_fraction: float = 0.15,
    num_refinement_tau: int = 1200,
) -> CryostageModelParams:
    """Fit the full active step-response trajectory of the cryostage.

    The characterization assays are nominal step responses to fixed cryostage
    setpoints.  For a fixed tau, the full response

        T(t) = h(t) T(0) + (1 - h(t)) (g T_ref + b)

    is linear in gain ``g`` and offset ``b``.  This fit therefore minimizes the
    recursive trajectory error directly, instead of the adjacent one-step
    recurrence error used by ``fit_first_order_model``.
    """

    runs = list(runs)
    if not runs:
        raise ValueError("runs must contain at least one characterization run")
    if num_tau < 2:
        raise ValueError("num_tau must be at least 2")
    if num_refinement_tau < 2:
        raise ValueError("num_refinement_tau must be at least 2")

    tau_min_s = float(tau_bounds_s[0])
    tau_max_s = float(tau_bounds_s[1])
    if tau_min_s <= 0.0 or tau_max_s <= tau_min_s:
        raise ValueError("tau_bounds_s must satisfy 0 < tau_min_s < tau_max_s")
    refinement_fraction = float(refinement_fraction)
    if not math.isfinite(refinement_fraction) or refinement_fraction <= 0.0:
        raise ValueError("refinement_fraction must be a finite positive value")

    packed_runs = []
    for run in runs:
        T_ref_nominal_C = float(np.median(run.T_ref_C))
        packed_runs.append(
            (
                run.time_s,
                T_ref_nominal_C,
                float(run.T_plate_C[0]),
                run.T_plate_C,
            )
        )

    def fit_tau_grid(tau_grid_s: np.ndarray) -> tuple[float, CryostageModelParams]:
        best_params: CryostageModelParams | None = None
        best_rmse = math.inf

        for tau_s in tau_grid_s:
            phi_rows = []
            target_rows = []
            for time_s, T_ref_nominal_C, T_plate0_C, T_plate_C in packed_runs:
                h = np.exp(-time_s / float(tau_s))
                weight = 1.0 - h
                phi_rows.append(
                    np.column_stack(
                        (
                            weight * T_ref_nominal_C,
                            weight,
                        )
                    )
                )
                target_rows.append(T_plate_C - h * T_plate0_C)

            Phi = np.vstack(phi_rows)
            target = np.concatenate(target_rows)
            coeffs, *_ = np.linalg.lstsq(Phi, target, rcond=None)
            residual = Phi @ coeffs - target
            rmse = math.sqrt(float(np.mean(residual * residual)))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = CryostageModelParams(
                    tau_s=float(tau_s),
                    gain=float(coeffs[0]),
                    offset_C=float(coeffs[1]),
                )

        if best_params is None:
            raise RuntimeError("first-order step-response fit failed")
        return best_rmse, best_params

    coarse_grid_s = np.geomspace(tau_min_s, tau_max_s, int(num_tau))
    _, coarse_params = fit_tau_grid(coarse_grid_s)

    refined_min_s = max(tau_min_s, coarse_params.tau_s * (1.0 - refinement_fraction))
    refined_max_s = min(tau_max_s, coarse_params.tau_s * (1.0 + refinement_fraction))
    refined_grid_s = np.linspace(refined_min_s, refined_max_s, int(num_refinement_tau))
    _, refined_params = fit_tau_grid(refined_grid_s)
    return refined_params


def fit_first_order_temperature_lookup_model(
    runs: Iterable[CharacterizationRun],
    *,
    tau_bounds_s: tuple[float, float] = (10.0, 250.0),
    num_tau: int = 1200,
) -> CryostageModelParams:
    """Fit a first-order model with setpoint-dependent tau and steady state.

    The model remains first order during simulation, but the response time and
    steady plate temperature are linearly interpolated as functions of the
    requested reference temperature:

        dT_plate/dt = (T_inf(T_ref) - T_plate) / tau(T_ref).

    The lookup values are fitted independently for each characterized reference
    setpoint and are linearly extrapolated just outside the characterized range.
    """

    runs = list(runs)
    if not runs:
        raise ValueError("runs must contain at least one characterization run")
    if num_tau < 2:
        raise ValueError("num_tau must be at least 2")

    tau_min_s = float(tau_bounds_s[0])
    tau_max_s = float(tau_bounds_s[1])
    if tau_min_s <= 0.0 or tau_max_s <= tau_min_s:
        raise ValueError("tau_bounds_s must satisfy 0 < tau_min_s < tau_max_s")

    runs_by_reference: dict[float, list[CharacterizationRun]] = {}
    for run in runs:
        T_ref_nominal_C = float(np.median(run.T_ref_C))
        runs_by_reference.setdefault(T_ref_nominal_C, []).append(run)

    if len(runs_by_reference) < 2:
        raise ValueError("at least two distinct reference temperatures are required")

    def fit_reference_group(group_runs: list[CharacterizationRun]) -> tuple[float, float]:
        packed_runs = [
            (
                run.time_s,
                float(run.T_plate_C[0]),
                run.T_plate_C,
            )
            for run in group_runs
        ]
        best_rmse = math.inf
        best_tau_s = math.nan
        best_steady_plate_C = math.nan

        for tau_s in np.geomspace(tau_min_s, tau_max_s, int(num_tau)):
            numerator = 0.0
            denominator = 0.0
            cached_terms = []
            for time_s, T_plate0_C, T_plate_C in packed_runs:
                h = np.exp(-time_s / float(tau_s))
                weight = 1.0 - h
                numerator += float(np.dot(weight, T_plate_C - h * T_plate0_C))
                denominator += float(np.dot(weight, weight))
                cached_terms.append((h, weight, T_plate0_C, T_plate_C))
            if denominator <= 0.0:
                continue

            steady_plate_C = numerator / denominator
            residuals = []
            for h, weight, T_plate0_C, T_plate_C in cached_terms:
                predicted = h * T_plate0_C + weight * steady_plate_C
                residuals.append(predicted - T_plate_C)
            residual = np.concatenate(residuals)
            rmse = math.sqrt(float(np.mean(residual * residual)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_tau_s = float(tau_s)
                best_steady_plate_C = float(steady_plate_C)

        if not math.isfinite(best_tau_s) or not math.isfinite(best_steady_plate_C):
            raise RuntimeError("temperature lookup fit failed")
        return best_tau_s, best_steady_plate_C

    references = []
    response_tau_s = []
    steady_plate_C = []
    for reference_C in sorted(runs_by_reference):
        tau_s, steady_C = fit_reference_group(runs_by_reference[reference_C])
        references.append(float(reference_C))
        response_tau_s.append(float(tau_s))
        steady_plate_C.append(float(steady_C))

    Phi = np.column_stack((references, np.ones(len(references), dtype=np.float64)))
    gain, offset_C = np.linalg.lstsq(Phi, np.asarray(steady_plate_C), rcond=None)[0]

    return CryostageModelParams(
        tau_s=float(np.median(np.asarray(response_tau_s, dtype=np.float64))),
        gain=float(gain),
        offset_C=float(offset_C),
        reference_temperatures_C=tuple(references),
        response_tau_s=tuple(response_tau_s),
        steady_plate_C=tuple(steady_plate_C),
    )


def default_characterization_run_paths(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    if repo_root is None:
        repo_root = default_project_root()
    repo_root = Path(repo_root)
    return tuple(
        sorted(
            (
                repo_root / "data" / "characterization_cryostage"
            ).glob("characterization_min*/cryostage_characterization_min*.csv")
        )
    )


def fit_default_cryostage_params(repo_root: str | Path | None = None) -> CryostageModelParams:
    runs = [
        load_characterization_run(path, active_power_threshold=1.0)
        for path in default_characterization_run_paths(repo_root)
    ]
    return fit_first_order_temperature_lookup_model(runs)


def root_mean_square_error(T_true_C, T_pred_C) -> float:
    T_true_C = _as_float_array(T_true_C, name="T_true_C")
    T_pred_C = _as_float_array(T_pred_C, name="T_pred_C")
    if T_true_C.shape != T_pred_C.shape:
        raise ValueError("T_true_C and T_pred_C must have the same length")
    err = T_pred_C - T_true_C
    return float(np.sqrt(np.mean(err * err)))


def cryostage_params_to_dict(params: CryostageModelParams) -> dict[str, object]:
    return {
        "model_kind": (
            "temperature_lookup_first_order"
            if params.uses_temperature_lookup
            else "global_first_order"
        ),
        "tau_s": float(params.tau_s),
        "gain": float(params.gain),
        "offset_C": float(params.offset_C),
        "reference_temperatures_C": [
            float(value) for value in params.reference_temperatures_C
        ],
        "response_tau_s": [float(value) for value in params.response_tau_s],
        "steady_plate_C": [float(value) for value in params.steady_plate_C],
    }


def cryostage_params_from_dict(raw: dict[str, object]) -> CryostageModelParams:
    return CryostageModelParams(
        tau_s=float(raw["tau_s"]),
        gain=float(raw.get("gain", 1.0)),
        offset_C=float(raw.get("offset_C", 0.0)),
        reference_temperatures_C=tuple(
            float(value) for value in raw.get("reference_temperatures_C", [])
        ),
        response_tau_s=tuple(float(value) for value in raw.get("response_tau_s", [])),
        steady_plate_C=tuple(float(value) for value in raw.get("steady_plate_C", [])),
    )


DEFAULT_CRYOSTAGE_PARAMS = CryostageModelParams(
    tau_s=98.088190,
    gain=1.007606,
    offset_C=0.386079,
    reference_temperatures_C=(-20.0, -15.0, -10.0, -5.0),
    response_tau_s=(75.296198, 98.219855, 97.956524, 99.014102),
    steady_plate_C=(-19.769483, -14.761845, -9.612014, -4.692656),
)


__all__ = [
    "CharacterizationRun",
    "CryostageModelParams",
    "DEFAULT_CRYOSTAGE_PARAMS",
    "cryostage_params_from_dict",
    "cryostage_params_to_dict",
    "default_characterization_run_paths",
    "fit_default_cryostage_params",
    "fit_first_order_model",
    "fit_first_order_step_response_model",
    "fit_first_order_temperature_lookup_model",
    "load_characterization_run",
    "root_mean_square_error",
    "simulate_characterization_run",
    "simulate_plate_temperature",
]
