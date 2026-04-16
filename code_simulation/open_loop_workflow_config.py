from __future__ import annotations

"""Shared workflow defaults and config builders for the open-loop stack."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from geometry import GeometryParams
from reachability_constraints import default_constraints_dir
from solver import FreezeStopOptions, PrefillOptions, ThermalBCs, normalize_front_definition_mode


ACTIVE_FORMULATION = "full_process_article"
ACTIVE_DEFAULT_T_REF_BOUNDS_C = (-20.0, 0.0)
ACTIVE_REQUIRE_MONOTONE_NONINCREASING = True
ACTIVE_DEFAULT_KNOT_TIME_SCHEDULE = "uniform"
ACTIVE_FULL_PROCESS_HORIZON_S = 2400.0
ACTIVE_FULL_PROCESS_CRYOSTAGE_DT_S = 4.0
ACTIVE_FULL_PROCESS_H_FILL_M = 15.0e-3
ACTIVE_FULL_PROCESS_FRONT_REFERENCE_MODE = "linear_full_process"
ACTIVE_FULL_PROCESS_FRONT_DEFINITION_MODE = "isotherm_Tf"
ACTIVE_FULL_PROCESS_SMOOTHNESS_WEIGHT = 0.02
ACTIVE_FULL_PROCESS_COMPLETION_WEIGHT = 0.0

SEED_TEMPLATE_SUPPORT_TAU = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED_TEMPLATE_TEMPERATURES_C = (-0.1, -5.5, -11.9, -17.0, -17.2)
LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT = len(SEED_TEMPLATE_SUPPORT_TAU)

# Backward-compatible aliases used by legacy scripts and older notebooks.
DEFAULT_FORMULATION = ACTIVE_FORMULATION
DEFAULT_T_REF_BOUNDS_C = ACTIVE_DEFAULT_T_REF_BOUNDS_C
DEFAULT_REQUIRE_MONOTONE_NONINCREASING = ACTIVE_REQUIRE_MONOTONE_NONINCREASING
DEFAULT_THETA0 = SEED_TEMPLATE_TEMPERATURES_C
DEFAULT_CHARACTERIZATION_CONSTRAINTS_DIR = default_constraints_dir(Path(__file__).resolve().parent)
DEFAULT_KNOT_TIME_SCHEDULE = ACTIVE_DEFAULT_KNOT_TIME_SCHEDULE
SUPPORTED_KNOT_TIME_SCHEDULES = ("uniform", "early_dense", "mid_dense", "late_dense", "custom")
EARLY_LATE_DENSE_POWER = 1.5
MID_DENSE_POWER = 2.0
_ALLOWED_REFERENCE_MODES = {"legacy_linear_speed", "linear_full_process", "saturating_full_process"}


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
    """Shared configuration object for one open-loop trajectory-design problem."""

    horizon_s: float
    cryostage_dt_s: float
    knot_times_s: tuple[float, ...]
    front_target_speed_m_per_s: float
    tracking_weight: float = 1.0
    smoothness_weight: float = 0.0
    terminal_weight: float = 0.0
    completion_weight: float = 0.0
    t_ignore_s: float = 0.0
    T_ref_bounds_C: tuple[float, float] = (-25.0, 25.0)
    require_monotone_nonincreasing: bool = True
    enforce_characterization_admissibility: bool = False
    characterization_constraints_dir: str | Path | None = None
    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    front_reference_mode: str = "legacy_linear_speed"
    front_reference_alpha: float = 4.0
    incomplete_penalty_value: float = 2.0

    def __post_init__(self) -> None:
        horizon_s = float(self.horizon_s)
        cryostage_dt_s = float(self.cryostage_dt_s)
        front_target_speed_m_per_s = float(self.front_target_speed_m_per_s)
        tracking_weight = float(self.tracking_weight)
        smoothness_weight = float(self.smoothness_weight)
        terminal_weight = float(self.terminal_weight)
        completion_weight = float(self.completion_weight)
        t_ignore_s = float(self.t_ignore_s)
        knot_times_s = _coerce_float_tuple(self.knot_times_s, name="knot_times_s")
        T_ref_bounds_C = _coerce_float_tuple(self.T_ref_bounds_C, name="T_ref_bounds_C")
        enforce_characterization_admissibility = bool(self.enforce_characterization_admissibility)
        characterization_constraints_dir = (
            None
            if self.characterization_constraints_dir is None
            else Path(self.characterization_constraints_dir)
        )
        solver_kwargs = dict(self.solver_kwargs)
        front_reference_mode = str(self.front_reference_mode)
        front_reference_alpha = float(self.front_reference_alpha)
        incomplete_penalty_value = float(self.incomplete_penalty_value)

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
            ("completion_weight", completion_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if not math.isfinite(t_ignore_s) or t_ignore_s < 0.0:
            raise ValueError("t_ignore_s must be a finite non-negative value")

        if front_reference_mode not in _ALLOWED_REFERENCE_MODES:
            raise ValueError(
                "front_reference_mode must be one of "
                f"{sorted(_ALLOWED_REFERENCE_MODES)!r}, got {front_reference_mode!r}"
            )
        if not math.isfinite(front_reference_alpha) or front_reference_alpha < 0.0:
            raise ValueError("front_reference_alpha must be a finite non-negative value")
        if not math.isfinite(incomplete_penalty_value) or incomplete_penalty_value <= 1.0:
            raise ValueError("incomplete_penalty_value must be a finite value greater than 1")

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
        object.__setattr__(self, "completion_weight", completion_weight)
        object.__setattr__(self, "t_ignore_s", t_ignore_s)
        object.__setattr__(self, "knot_times_s", knot_times_s)
        object.__setattr__(self, "T_ref_bounds_C", T_ref_bounds_C)
        object.__setattr__(self, "enforce_characterization_admissibility", enforce_characterization_admissibility)
        object.__setattr__(self, "characterization_constraints_dir", characterization_constraints_dir)
        object.__setattr__(self, "solver_kwargs", solver_kwargs)
        object.__setattr__(self, "front_reference_mode", front_reference_mode)
        object.__setattr__(self, "front_reference_alpha", front_reference_alpha)
        object.__setattr__(self, "incomplete_penalty_value", incomplete_penalty_value)

    @property
    def safety_cap_s(self) -> float:
        return float(self.horizon_s)

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
        raise ValueError("--custom-support-by-n must contain at least one parameter_count:support-list entry when provided")
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                "Each --custom-support-by-n entry must be written as 'parameter_count:t0,t1,...'"
            )
        n_text, support_text = entry.split(":", 1)
        num_knots = int(n_text.strip())
        support_parts = [part.strip() for part in support_text.split(",") if part.strip()]
        support_tau = _validated_normalized_support_tau(
            [float(part) for part in support_parts],
            num_knots=num_knots,
            name=f"custom support for parameter_count={num_knots}",
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


def _build_legacy_problem_config(*, front_definition_mode: str = ACTIVE_FULL_PROCESS_FRONT_DEFINITION_MODE) -> OpenLoopProblemConfig:
    front_definition_mode = normalize_front_definition_mode(front_definition_mode)
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
            "front_definition_mode": front_definition_mode,
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
    num_knots: int,
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
    front_definition_mode: str = ACTIVE_FULL_PROCESS_FRONT_DEFINITION_MODE,
) -> OpenLoopProblemConfig:
    front_definition_mode = normalize_front_definition_mode(front_definition_mode)
    knot_times_s = build_external_knot_times_s(
        horizon_s=ACTIVE_FULL_PROCESS_HORIZON_S,
        num_knots=num_knots,
        knot_time_schedule=knot_time_schedule,
        knot_time_custom_support_tau=knot_time_custom_support_tau,
    )
    return OpenLoopProblemConfig(
        horizon_s=ACTIVE_FULL_PROCESS_HORIZON_S,
        cryostage_dt_s=ACTIVE_FULL_PROCESS_CRYOSTAGE_DT_S,
        knot_times_s=knot_times_s,
        front_target_speed_m_per_s=ACTIVE_FULL_PROCESS_H_FILL_M / ACTIVE_FULL_PROCESS_HORIZON_S,
        tracking_weight=1.0,
        smoothness_weight=ACTIVE_FULL_PROCESS_SMOOTHNESS_WEIGHT,
        completion_weight=ACTIVE_FULL_PROCESS_COMPLETION_WEIGHT,
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
                H_fill=ACTIVE_FULL_PROCESS_H_FILL_M,
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
            "front_definition_mode": front_definition_mode,
            "probe_z_m": (3.0e-3, 6.2e-3, 11.0e-3),
            "probe_wall_inset_m": 1.0e-3,
            "Nz_front": 200,
            "enable_front_curve": False,
            "stop_when_wall_frozen": False,
            "use_tabulated_water_ice": True,
        },
        front_reference_mode=ACTIVE_FULL_PROCESS_FRONT_REFERENCE_MODE,
        front_reference_alpha=4.0,
        incomplete_penalty_value=2.0,
    )


def build_problem_config(
    *,
    formulation: str = DEFAULT_FORMULATION,
    num_knots: int | None = None,
    knot_time_schedule: str = DEFAULT_KNOT_TIME_SCHEDULE,
    knot_time_custom_support_tau: tuple[float, ...] | None = None,
    front_definition_mode: str | None = None,
) -> OpenLoopProblemConfig:
    resolved_front_definition_mode = normalize_front_definition_mode(
        ACTIVE_FULL_PROCESS_FRONT_DEFINITION_MODE if front_definition_mode is None else front_definition_mode
    )
    if formulation == "legacy_exploratory":
        if num_knots not in (None, LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT):
            raise ValueError("num_knots is only configurable for full_process_article")
        if knot_time_schedule != DEFAULT_KNOT_TIME_SCHEDULE or knot_time_custom_support_tau is not None:
            raise ValueError("knot-time schedules are only configurable for full_process_article")
        return _build_legacy_problem_config(front_definition_mode=resolved_front_definition_mode)
    if formulation == "full_process_article":
        if num_knots is None:
            raise ValueError("num_knots must be explicitly provided for full_process_article")
        resolved_num_knots = int(num_knots)
        return _build_full_process_problem_config(
            num_knots=resolved_num_knots,
            knot_time_schedule=knot_time_schedule,
            knot_time_custom_support_tau=knot_time_custom_support_tau,
            front_definition_mode=resolved_front_definition_mode,
    )
    raise ValueError(f"Unknown formulation={formulation!r}")


def default_theta0_for_config(config: OpenLoopProblemConfig) -> tuple[float, ...]:
    base_tau = np.asarray(SEED_TEMPLATE_SUPPORT_TAU, dtype=np.float64)
    seed_temperatures_C = np.asarray(SEED_TEMPLATE_TEMPERATURES_C, dtype=np.float64)
    target_tau = np.asarray(config.knot_times_s, dtype=np.float64) / float(config.horizon_s)
    theta0 = np.interp(target_tau, base_tau, seed_temperatures_C)
    theta0 = np.clip(theta0, config.T_ref_bounds_C[0], config.T_ref_bounds_C[1])
    return tuple(float(value) for value in theta0)


__all__ = [
    "ACTIVE_DEFAULT_KNOT_TIME_SCHEDULE",
    "ACTIVE_DEFAULT_T_REF_BOUNDS_C",
    "ACTIVE_FORMULATION",
    "ACTIVE_FULL_PROCESS_COMPLETION_WEIGHT",
    "ACTIVE_FULL_PROCESS_CRYOSTAGE_DT_S",
    "ACTIVE_FULL_PROCESS_FRONT_REFERENCE_MODE",
    "ACTIVE_FULL_PROCESS_FRONT_DEFINITION_MODE",
    "ACTIVE_FULL_PROCESS_H_FILL_M",
    "ACTIVE_FULL_PROCESS_HORIZON_S",
    "ACTIVE_FULL_PROCESS_SMOOTHNESS_WEIGHT",
    "ACTIVE_REQUIRE_MONOTONE_NONINCREASING",
    "OpenLoopProblemConfig",
    "LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT",
    "SEED_TEMPLATE_SUPPORT_TAU",
    "SEED_TEMPLATE_TEMPERATURES_C",
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
