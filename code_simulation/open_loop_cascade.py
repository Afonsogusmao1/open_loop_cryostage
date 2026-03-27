from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryostage_model import CryostageModelParams, simulate_plate_temperature
from solver import run_case
from trajectory_profiles import PiecewiseLinearTemperatureProfile


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_time_grid(time_s: np.ndarray) -> None:
    if time_s.size == 0:
        raise ValueError("time_s must contain at least one sample")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")


@dataclass(frozen=True)
class OpenLoopCascadeResult:
    cryostage_time_s: np.ndarray
    T_ref_C: np.ndarray
    T_plate_C: np.ndarray
    out_dir: Path
    xdmf_path: Path
    probes_path: Path
    front_path: Path
    front_curve_path: Path | None


def sampled_temperature_profile(time_s, temperature_C) -> PiecewiseLinearTemperatureProfile:
    time_s = _as_float_array(time_s, name="time_s")
    temperature_C = _as_float_array(temperature_C, name="temperature_C")
    if time_s.shape != temperature_C.shape:
        raise ValueError("time_s and temperature_C must have the same length")
    _validate_time_grid(time_s)
    return PiecewiseLinearTemperatureProfile(
        knot_times_s=tuple(float(ti) for ti in time_s),
        knot_temperatures_C=tuple(float(Ti) for Ti in temperature_C),
    )


def run_open_loop_case(
    *,
    time_s,
    T_ref_profile_C,
    cryostage_params: CryostageModelParams,
    out_dir: str | Path,
    prefix: str,
    T_plate0_C: float | None = None,
    **run_case_kwargs,
) -> OpenLoopCascadeResult:
    if "T_plate_C" in run_case_kwargs or "T_plate_profile_C" in run_case_kwargs:
        raise ValueError("run_case_kwargs must not override T_plate_C or T_plate_profile_C")

    time_s = _as_float_array(time_s, name="time_s")
    _validate_time_grid(time_s)

    T_ref_C = np.array([float(T_ref_profile_C(float(ti))) for ti in time_s], dtype=np.float64)

    if T_plate0_C is None:
        bcs = run_case_kwargs.get("bcs")
        if bcs is not None and hasattr(bcs, "T_room_C"):
            T_plate0_C = float(bcs.T_room_C)
        else:
            T_plate0_C = float(T_ref_C[0])

    T_plate_C = simulate_plate_temperature(
        time_s=time_s,
        T_ref_profile_C=T_ref_profile_C,
        params=cryostage_params,
        T_plate0_C=float(T_plate0_C),
    )
    T_plate_profile_C = sampled_temperature_profile(time_s, T_plate_C)

    out_dir = Path(out_dir)
    run_case(
        out_dir=out_dir,
        prefix=prefix,
        T_plate_C=float(T_plate_C[0]),
        T_plate_profile_C=T_plate_profile_C,
        **run_case_kwargs,
    )

    xdmf_path = out_dir / f"{prefix}.xdmf"
    probes_path = out_dir / f"{prefix}_probes.csv"
    front_path = out_dir / f"{prefix}_front.csv"
    front_curve_path = out_dir / f"{prefix}_front_curve.csv"
    if not front_curve_path.exists():
        front_curve_path = None

    return OpenLoopCascadeResult(
        cryostage_time_s=time_s,
        T_ref_C=T_ref_C,
        T_plate_C=T_plate_C,
        out_dir=out_dir,
        xdmf_path=xdmf_path,
        probes_path=probes_path,
        front_path=front_path,
        front_curve_path=front_curve_path,
    )


__all__ = [
    "OpenLoopCascadeResult",
    "run_open_loop_case",
    "sampled_temperature_profile",
]
