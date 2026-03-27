from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cryostage_model import CryostageModelParams
from open_loop_cascade import OpenLoopCascadeResult, run_open_loop_case
from trajectory_profiles import PiecewiseLinearTemperatureProfile


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _coerce_float_tuple(values, *, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of floats") from exc
    if len(out) == 0:
        raise ValueError(f"{name} must contain at least one entry")
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_strictly_increasing(values: tuple[float, ...], *, name: str) -> None:
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class OpenLoopProblemConfig:
    horizon_s: float
    cryostage_dt_s: float
    knot_times_s: tuple[float, ...]
    front_target_speed_m_per_s: float
    tracking_weight: float = 1.0
    smoothness_weight: float = 0.0
    terminal_weight: float = 0.0
    t_ignore_s: float = 0.0
    T_ref_bounds_C: tuple[float, float] = (-25.0, 25.0)
    require_monotone_nonincreasing: bool = True
    solver_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        horizon_s = float(self.horizon_s)
        cryostage_dt_s = float(self.cryostage_dt_s)
        front_target_speed_m_per_s = float(self.front_target_speed_m_per_s)
        tracking_weight = float(self.tracking_weight)
        smoothness_weight = float(self.smoothness_weight)
        terminal_weight = float(self.terminal_weight)
        t_ignore_s = float(self.t_ignore_s)
        knot_times_s = _coerce_float_tuple(self.knot_times_s, name="knot_times_s")
        T_ref_bounds_C = _coerce_float_tuple(self.T_ref_bounds_C, name="T_ref_bounds_C")
        solver_kwargs = dict(self.solver_kwargs)

        if not math.isfinite(horizon_s) or horizon_s <= 0.0:
            raise ValueError("horizon_s must be a finite positive value")
        if not math.isfinite(cryostage_dt_s) or cryostage_dt_s <= 0.0:
            raise ValueError("cryostage_dt_s must be a finite positive value")
        if knot_times_s[0] < 0.0:
            raise ValueError("knot_times_s must start at or after t=0")
        _validate_strictly_increasing(knot_times_s, name="knot_times_s")
        if knot_times_s[-1] > horizon_s + 1e-12:
            raise ValueError("knot_times_s must not extend beyond horizon_s")

        if len(T_ref_bounds_C) != 2:
            raise ValueError("T_ref_bounds_C must contain exactly two values")
        if T_ref_bounds_C[0] > T_ref_bounds_C[1]:
            raise ValueError("T_ref_bounds_C must satisfy min <= max")

        if not math.isfinite(front_target_speed_m_per_s) or front_target_speed_m_per_s < 0.0:
            raise ValueError("front_target_speed_m_per_s must be a finite non-negative value")
        for name, value in (
            ("tracking_weight", tracking_weight),
            ("smoothness_weight", smoothness_weight),
            ("terminal_weight", terminal_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if not math.isfinite(t_ignore_s) or t_ignore_s < 0.0:
            raise ValueError("t_ignore_s must be a finite non-negative value")

        if "t_after_fill_s" in solver_kwargs:
            t_after_fill_s = float(solver_kwargs["t_after_fill_s"])
            if abs(t_after_fill_s - horizon_s) > 1e-12:
                raise ValueError("solver_kwargs['t_after_fill_s'] must match horizon_s when provided")

        object.__setattr__(self, "horizon_s", horizon_s)
        object.__setattr__(self, "cryostage_dt_s", cryostage_dt_s)
        object.__setattr__(self, "front_target_speed_m_per_s", front_target_speed_m_per_s)
        object.__setattr__(self, "tracking_weight", tracking_weight)
        object.__setattr__(self, "smoothness_weight", smoothness_weight)
        object.__setattr__(self, "terminal_weight", terminal_weight)
        object.__setattr__(self, "t_ignore_s", t_ignore_s)
        object.__setattr__(self, "knot_times_s", knot_times_s)
        object.__setattr__(self, "T_ref_bounds_C", T_ref_bounds_C)
        object.__setattr__(self, "solver_kwargs", solver_kwargs)

    def cryostage_time_grid_s(self) -> np.ndarray:
        time_s = np.arange(0.0, self.horizon_s, self.cryostage_dt_s, dtype=np.float64)
        if time_s.size == 0 or time_s[0] > 0.0:
            time_s = np.insert(time_s, 0, 0.0)
        if time_s[-1] < self.horizon_s - 1e-12:
            time_s = np.append(time_s, self.horizon_s)
        return time_s

    def cascade_run_kwargs(self) -> dict[str, Any]:
        run_kwargs = dict(self.solver_kwargs)
        run_kwargs.setdefault("t_after_fill_s", self.horizon_s)
        return run_kwargs


@dataclass(frozen=True)
class FrontTrajectory:
    time_s: np.ndarray
    time_since_fill_s: np.ndarray
    z_front_m: np.ndarray

    def __post_init__(self) -> None:
        time_s = _as_float_array(self.time_s, name="time_s")
        time_since_fill_s = _as_float_array(self.time_since_fill_s, name="time_since_fill_s")
        z_front_m = _as_float_array(self.z_front_m, name="z_front_m")

        if time_s.shape != time_since_fill_s.shape or time_s.shape != z_front_m.shape:
            raise ValueError("time_s, time_since_fill_s, and z_front_m must have the same length")
        if time_s.size == 0:
            raise ValueError("front trajectory must contain at least one valid sample")
        if np.any(np.diff(time_s) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if np.any(np.diff(time_since_fill_s) <= 0.0):
            raise ValueError("time_since_fill_s must be strictly increasing")

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "time_since_fill_s", time_since_fill_s)
        object.__setattr__(self, "z_front_m", z_front_m)


@dataclass(frozen=True)
class OpenLoopObjectiveResult:
    objective_value: float
    tracking_mse: float
    smoothness_penalty: float
    terminal_penalty: float
    num_objective_samples: int
    cascade_result: OpenLoopCascadeResult
    front_trajectory: FrontTrajectory
    z_front_reference_m: np.ndarray

    def __float__(self) -> float:
        return float(self.objective_value)


def build_reference_profile_from_theta(
    theta,
    config: OpenLoopProblemConfig,
) -> PiecewiseLinearTemperatureProfile:
    theta_values = _coerce_float_tuple(theta, name="theta")
    if len(theta_values) != len(config.knot_times_s):
        raise ValueError(
            "theta must contain exactly one temperature per knot time "
            f"({len(config.knot_times_s)} expected, got {len(theta_values)})"
        )

    T_min_C, T_max_C = config.T_ref_bounds_C
    for i, temperature_C in enumerate(theta_values):
        if temperature_C < T_min_C or temperature_C > T_max_C:
            raise ValueError(
                f"theta[{i}]={temperature_C:.6g} is outside T_ref_bounds_C="
                f"({T_min_C:.6g}, {T_max_C:.6g})"
            )

    if config.require_monotone_nonincreasing:
        for i in range(1, len(theta_values)):
            if theta_values[i] > theta_values[i - 1] + 1e-12:
                raise ValueError("theta must be monotone non-increasing when requested")

    return PiecewiseLinearTemperatureProfile(
        knot_times_s=config.knot_times_s,
        knot_temperatures_C=theta_values,
    )


def load_front_csv(front_csv_path: str | Path) -> FrontTrajectory:
    front_csv_path = Path(front_csv_path)
    time_s: list[float] = []
    time_since_fill_s: list[float] = []
    z_front_m: list[float] = []

    with front_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fill_flag = float(row.get("fill_flag", "nan"))
                current_time_s = float(row["time_s"])
                current_time_since_fill_s = float(row["time_since_fill_s"])
                current_z_front_m = float(row["z_front_m"])
            except (KeyError, TypeError, ValueError):
                continue

            if np.isfinite(fill_flag) and fill_flag < 0.5:
                continue
            if not (
                np.isfinite(current_time_s)
                and np.isfinite(current_time_since_fill_s)
                and np.isfinite(current_z_front_m)
            ):
                continue

            time_s.append(current_time_s)
            time_since_fill_s.append(current_time_since_fill_s)
            z_front_m.append(current_z_front_m)

    if not time_s:
        raise ValueError(f"{front_csv_path} does not contain any valid post-fill front samples")

    return FrontTrajectory(
        time_s=np.asarray(time_s, dtype=np.float64),
        time_since_fill_s=np.asarray(time_since_fill_s, dtype=np.float64),
        z_front_m=np.asarray(z_front_m, dtype=np.float64),
    )


def build_front_reference(
    time_s,
    z_front_measured,
    config: OpenLoopProblemConfig,
) -> np.ndarray:
    time_s = _as_float_array(time_s, name="time_s")
    z_front_measured = _as_float_array(z_front_measured, name="z_front_measured")
    if time_s.shape != z_front_measured.shape:
        raise ValueError("time_s and z_front_measured must have the same length")
    if time_s.size == 0:
        raise ValueError("time_s and z_front_measured must contain at least one sample")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")

    valid_mask = time_s >= config.t_ignore_s
    if not np.any(valid_mask):
        raise ValueError("t_ignore_s excludes all available front samples")

    first_idx = int(np.flatnonzero(valid_mask)[0])
    t0_s = float(time_s[first_idx])
    z0_m = float(z_front_measured[first_idx])

    z_ref_m = np.full_like(time_s, np.nan, dtype=np.float64)
    z_ref_m[valid_mask] = z0_m + config.front_target_speed_m_per_s * (time_s[valid_mask] - t0_s)
    return z_ref_m


def _reference_smoothness_penalty(profile: PiecewiseLinearTemperatureProfile) -> float:
    knot_times_s = np.asarray(profile.knot_times_s, dtype=np.float64)
    knot_temperatures_C = np.asarray(profile.knot_temperatures_C, dtype=np.float64)
    if knot_times_s.size < 2:
        return 0.0
    slopes_C_per_s = np.diff(knot_temperatures_C) / np.diff(knot_times_s)
    return float(np.mean(slopes_C_per_s * slopes_C_per_s))


def evaluate_open_loop_objective(
    theta,
    config: OpenLoopProblemConfig,
    cryostage_params: CryostageModelParams,
    out_dir: str | Path,
    case_name: str,
    *,
    T_plate0_C: float | None = None,
) -> OpenLoopObjectiveResult:
    T_ref_profile_C = build_reference_profile_from_theta(theta, config)

    cascade_result = run_open_loop_case(
        time_s=config.cryostage_time_grid_s(),
        T_ref_profile_C=T_ref_profile_C,
        cryostage_params=cryostage_params,
        out_dir=out_dir,
        prefix=case_name,
        T_plate0_C=T_plate0_C,
        **config.cascade_run_kwargs(),
    )

    front_trajectory = load_front_csv(cascade_result.front_path)
    z_front_reference_m = build_front_reference(
        front_trajectory.time_since_fill_s,
        front_trajectory.z_front_m,
        config,
    )

    objective_mask = np.isfinite(z_front_reference_m)
    if not np.any(objective_mask):
        raise ValueError("objective reference does not contain any valid samples")

    tracking_error_m = front_trajectory.z_front_m[objective_mask] - z_front_reference_m[objective_mask]
    tracking_mse = float(np.mean(tracking_error_m * tracking_error_m))
    smoothness_penalty = _reference_smoothness_penalty(T_ref_profile_C)

    if config.terminal_weight > 0.0:
        terminal_error_m = float(tracking_error_m[-1])
        terminal_penalty = terminal_error_m * terminal_error_m
    else:
        terminal_penalty = 0.0

    objective_value = (
        config.tracking_weight * tracking_mse
        + config.smoothness_weight * smoothness_penalty
        + config.terminal_weight * terminal_penalty
    )

    return OpenLoopObjectiveResult(
        objective_value=float(objective_value),
        tracking_mse=tracking_mse,
        smoothness_penalty=smoothness_penalty,
        terminal_penalty=float(terminal_penalty),
        num_objective_samples=int(np.count_nonzero(objective_mask)),
        cascade_result=cascade_result,
        front_trajectory=front_trajectory,
        z_front_reference_m=z_front_reference_m,
    )


__all__ = [
    "FrontTrajectory",
    "OpenLoopObjectiveResult",
    "OpenLoopProblemConfig",
    "build_front_reference",
    "build_reference_profile_from_theta",
    "evaluate_open_loop_objective",
    "load_front_csv",
]
