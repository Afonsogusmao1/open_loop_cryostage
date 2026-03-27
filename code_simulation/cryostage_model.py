from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


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

    def __post_init__(self) -> None:
        if not math.isfinite(self.tau_s) or self.tau_s <= 0.0:
            raise ValueError("tau_s must be a finite positive value")
        if not math.isfinite(self.gain):
            raise ValueError("gain must be finite")
        if not math.isfinite(self.offset_C):
            raise ValueError("offset_C must be finite")


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
        alpha = math.exp(-dt_s / params.tau_s)
        T_plate_ss_C = params.gain * float(T_ref_samples_C[i - 1]) + params.offset_C
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
    row_type: str = "telemetry",
) -> CharacterizationRun:
    path = Path(path)
    rows: list[tuple[float, float, float]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if row_type and str(row.get("row_type", "")).lower() != row_type.lower():
                continue
            try:
                t_s = float(row[time_col])
                T_ref_C = float(row[T_ref_col])
                T_plate_C = float(row[T_plate_col])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((t_s, T_ref_C, T_plate_C))

    if not rows:
        raise ValueError(f"{path.name} has no valid rows for the requested columns")

    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    unique_mask = np.ones(arr.shape[0], dtype=bool)
    unique_mask[1:] = np.diff(arr[:, 0]) > 0.0
    arr = arr[unique_mask]
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


def default_characterization_run_paths(repo_root: str | Path | None = None) -> tuple[Path, ...]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_root)
    return tuple(
        sorted(
            (
                repo_root / "data" / "characterization_cryostage"
            ).glob("characterization_min*/cryostage_characterization_min*.csv")
        )
    )


def fit_default_cryostage_params(repo_root: str | Path | None = None) -> CryostageModelParams:
    runs = [load_characterization_run(path) for path in default_characterization_run_paths(repo_root)]
    return fit_first_order_model(runs)


def root_mean_square_error(T_true_C, T_pred_C) -> float:
    T_true_C = _as_float_array(T_true_C, name="T_true_C")
    T_pred_C = _as_float_array(T_pred_C, name="T_pred_C")
    if T_true_C.shape != T_pred_C.shape:
        raise ValueError("T_true_C and T_pred_C must have the same length")
    err = T_pred_C - T_true_C
    return float(np.sqrt(np.mean(err * err)))


DEFAULT_CRYOSTAGE_PARAMS = CryostageModelParams(
    tau_s=145.515679,
    gain=1.060517,
    offset_C=-0.005893,
)


__all__ = [
    "CharacterizationRun",
    "CryostageModelParams",
    "DEFAULT_CRYOSTAGE_PARAMS",
    "default_characterization_run_paths",
    "fit_default_cryostage_params",
    "fit_first_order_model",
    "load_characterization_run",
    "root_mean_square_error",
    "simulate_characterization_run",
    "simulate_plate_temperature",
]
