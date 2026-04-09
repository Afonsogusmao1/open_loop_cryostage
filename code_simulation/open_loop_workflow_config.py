from __future__ import annotations

"""Shared workflow defaults and config builders for the open-loop stack."""

from pathlib import Path

import numpy as np

from geometry import GeometryParams
from open_loop_problem import OpenLoopProblemConfig
from reachability_constraints import default_constraints_dir
from solver import FreezeStopOptions, PrefillOptions, ThermalBCs


DEFAULT_FORMULATION = "full_process_article"
DEFAULT_T_REF_BOUNDS_C = (-20.0, 0.0)
DEFAULT_REQUIRE_MONOTONE_NONINCREASING = True
DEFAULT_THETA0 = (-0.1, -5.5, -11.9, -17.0, -17.2)
DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR = default_constraints_dir(Path(__file__).resolve().parent)
DEFAULT_KNOT_TIME_SCHEDULE = "uniform"
SUPPORTED_KNOT_TIME_SCHEDULES = ("uniform", "early_dense", "mid_dense", "late_dense", "custom")
EARLY_LATE_DENSE_POWER = 1.5
MID_DENSE_POWER = 2.0


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


def parse_normalized_support_tau_arg(raw_support_tau: str | None, *, num_knots: int) -> tuple[float, ...] | None:
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


def parse_normalized_support_tau_by_n_arg(raw_support_map: str | None) -> dict[int, tuple[float, ...]]:
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


def default_theta0_for_config(config: OpenLoopProblemConfig) -> tuple[float, ...]:
    if len(config.knot_times_s) == len(DEFAULT_THETA0):
        return DEFAULT_THETA0
    base_tau = np.linspace(0.0, 1.0, len(DEFAULT_THETA0), dtype=np.float64)
    target_tau = np.asarray(config.knot_times_s, dtype=np.float64) / float(config.horizon_s)
    theta0 = np.interp(target_tau, base_tau, np.asarray(DEFAULT_THETA0, dtype=np.float64))
    theta0 = np.clip(theta0, config.T_ref_bounds_C[0], config.T_ref_bounds_C[1])
    return tuple(float(value) for value in theta0)


__all__ = [
    "DEFAULT_FORMULATION",
    "DEFAULT_KNOT_TIME_SCHEDULE",
    "DEFAULT_THETA0",
    "SUPPORTED_KNOT_TIME_SCHEDULES",
    "build_external_knot_times_s",
    "build_knot_time_normalized_support_tau",
    "build_problem_config",
    "default_theta0_for_config",
    "parse_normalized_support_tau_arg",
    "parse_normalized_support_tau_by_n_arg",
]
