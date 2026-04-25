from __future__ import annotations

"""TOML configuration loading for simulation profiles and BO settings."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import math
import tomllib

from code_simulation.core.arrays import coerce_float_tuple, validate_strictly_increasing
from code_simulation.core.paths import configs_dir
from code_simulation.simulation.ambient import FixedAmbientTemperature, InterpolatedAmbientTemperature


DEFAULT_SIMULATION_CONFIG_PATH = configs_dir("simulation_profiles.toml")
DEFAULT_BO_CONFIG_PATH = configs_dir("bo.toml")
DEFAULT_VELOCITY_CONTROL_CONFIG_PATH = configs_dir("velocity_control.toml")


@dataclass(frozen=True)
class SimulationProfileConfig:
    name: str
    cryostage_dt_s: float
    solver_dt_s: float
    Nr: int
    Nz: int
    Nz_front: int
    Nr_front_curve: int
    Nz_front_curve: int
    write_every_s: float
    write_field_output: bool
    write_probe_csv: bool
    show_progress: bool
    enable_front_curve: bool
    use_tabulated_water_ice: bool
    ambient_temperature_from_plate_C: object | None

    def __post_init__(self) -> None:
        for name in ("cryostage_dt_s", "solver_dt_s", "write_every_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{self.name}.{name} must be a finite positive value")
            object.__setattr__(self, name, value)
        for name in ("Nr", "Nz", "Nz_front", "Nr_front_curve", "Nz_front_curve"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{self.name}.{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SimulationProfilesFile:
    path: Path
    default_profile: str
    profiles: dict[str, SimulationProfileConfig]

    def get_profile(self, name: str | None = None) -> SimulationProfileConfig:
        resolved_name = self.default_profile if name is None else str(name)
        try:
            return self.profiles[resolved_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown simulation profile {resolved_name!r}; available profiles: {sorted(self.profiles)}"
            ) from exc


@dataclass(frozen=True)
class BOFileConfig:
    path: Path
    num_knots: int
    knot_time_schedule: str
    knot_time_custom_support_tau: tuple[float, ...] | None
    theta0_C: tuple[float, ...] | None
    T_ref_bounds_C: tuple[float, float]
    theta_bounds_C: tuple[tuple[float, float], ...] | None
    optimize_knot_times: bool
    random_seed: int
    init_points: int
    n_iter: int
    acquisition_kind: str
    acquisition_kappa: float
    acquisition_xi: float
    seed_with_theta0: bool
    parameterization_kind: str
    init_strategy: str
    init_local_sigma: float
    init_max_attempts_per_point: int
    local_refinement_points: int
    local_refinement_sigma: float
    infeasible_objective_penalty: float


@dataclass(frozen=True)
class VelocityTargetConfig:
    target_front_speed_mm_s: float
    control_z_min_mm: float
    control_z_max_mm: float

    def __post_init__(self) -> None:
        target_front_speed_mm_s = float(self.target_front_speed_mm_s)
        control_z_min_mm = float(self.control_z_min_mm)
        control_z_max_mm = float(self.control_z_max_mm)
        if not math.isfinite(target_front_speed_mm_s) or target_front_speed_mm_s <= 0.0:
            raise ValueError("velocity_target.target_front_speed_mm_s must be finite and positive")
        if not math.isfinite(control_z_min_mm) or not math.isfinite(control_z_max_mm):
            raise ValueError("velocity_target control z limits must be finite")
        if control_z_min_mm < 0.0 or control_z_max_mm <= control_z_min_mm:
            raise ValueError("velocity_target must satisfy 0 <= control_z_min_mm < control_z_max_mm")
        object.__setattr__(self, "target_front_speed_mm_s", target_front_speed_mm_s)
        object.__setattr__(self, "control_z_min_mm", control_z_min_mm)
        object.__setattr__(self, "control_z_max_mm", control_z_max_mm)


@dataclass(frozen=True)
class InitialConditionsConfig:
    initial_water_temperature_C: float
    initial_plate_temperature_C: float
    no_warm_hold: bool = True

    def __post_init__(self) -> None:
        initial_water_temperature_C = float(self.initial_water_temperature_C)
        initial_plate_temperature_C = float(self.initial_plate_temperature_C)
        if not math.isfinite(initial_water_temperature_C):
            raise ValueError("initial_conditions.initial_water_temperature_C must be finite")
        if not math.isfinite(initial_plate_temperature_C):
            raise ValueError("initial_conditions.initial_plate_temperature_C must be finite")
        if not bool(self.no_warm_hold):
            raise ValueError("initial_conditions.no_warm_hold must be true for the active velocity-control protocol")
        object.__setattr__(self, "initial_water_temperature_C", initial_water_temperature_C)
        object.__setattr__(self, "initial_plate_temperature_C", initial_plate_temperature_C)
        object.__setattr__(self, "no_warm_hold", True)


@dataclass(frozen=True)
class TemperatureUncertaintyConfig:
    characterization_temperature_margin_C: float = 0.5

    def __post_init__(self) -> None:
        characterization_temperature_margin_C = float(self.characterization_temperature_margin_C)
        if not math.isfinite(characterization_temperature_margin_C) or characterization_temperature_margin_C < 0.0:
            raise ValueError(
                "temperature_uncertainty.characterization_temperature_margin_C must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "characterization_temperature_margin_C",
            characterization_temperature_margin_C,
        )


@dataclass(frozen=True)
class ManualTrajectoryConfig:
    horizon_s: float
    num_knots: int
    knot_time_schedule: str
    knot_time_custom_support_tau: tuple[float, ...] | None
    theta_C: tuple[float, ...]
    T_ref_bounds_C: tuple[float, float]
    require_monotone_nonincreasing: bool

    def __post_init__(self) -> None:
        horizon_s = float(self.horizon_s)
        num_knots = int(self.num_knots)
        knot_time_schedule = str(self.knot_time_schedule).strip().lower()
        theta_C = coerce_float_tuple(self.theta_C, name="manual_trajectory.theta_C")
        T_ref_bounds_C = coerce_float_tuple(self.T_ref_bounds_C, name="manual_trajectory.T_ref_bounds_C")
        if not math.isfinite(horizon_s) or horizon_s <= 0.0:
            raise ValueError("manual_trajectory.horizon_s must be finite and positive")
        if num_knots < 2:
            raise ValueError("manual_trajectory.num_knots must be at least 2")
        if len(theta_C) != num_knots:
            raise ValueError("manual_trajectory.theta_C length must match num_knots")
        if len(T_ref_bounds_C) != 2 or T_ref_bounds_C[0] >= T_ref_bounds_C[1]:
            raise ValueError("manual_trajectory.T_ref_bounds_C must contain [lower, upper] with lower < upper")
        lower, upper = float(T_ref_bounds_C[0]), float(T_ref_bounds_C[1])
        for idx, value in enumerate(theta_C):
            if value < lower or value > upper:
                raise ValueError(f"manual_trajectory.theta_C[{idx}] is outside T_ref_bounds_C")
        if self.require_monotone_nonincreasing:
            for idx in range(1, len(theta_C)):
                if theta_C[idx] > theta_C[idx - 1] + 1.0e-12:
                    raise ValueError("manual_trajectory.theta_C must be monotone non-increasing")
        object.__setattr__(self, "horizon_s", horizon_s)
        object.__setattr__(self, "num_knots", num_knots)
        object.__setattr__(self, "knot_time_schedule", knot_time_schedule)
        object.__setattr__(self, "theta_C", theta_C)
        object.__setattr__(self, "T_ref_bounds_C", (lower, upper))
        object.__setattr__(self, "require_monotone_nonincreasing", bool(self.require_monotone_nonincreasing))


@dataclass(frozen=True)
class VelocityControlFileConfig:
    path: Path
    run_name: str
    simulation_profile: str
    velocity_target: VelocityTargetConfig
    initial_conditions: InitialConditionsConfig
    temperature_uncertainty: TemperatureUncertaintyConfig
    manual_trajectory: ManualTrajectoryConfig


def _read_toml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _ambient_model_from_table(raw_ambient: dict[str, Any]):
    mode = str(raw_ambient.get("mode", "fixed")).strip().lower()
    if mode == "fixed":
        return FixedAmbientTemperature(float(raw_ambient.get("ambient_temperature_C", raw_ambient.get("T_room_C", 5.75))))
    if mode == "interpolate_from_cryostage":
        cryostage_temperature_C = coerce_float_tuple(
            raw_ambient.get("cryostage_temperature_C"),
            name="ambient.cryostage_temperature_C",
        )
        ambient_temperature_C = coerce_float_tuple(
            raw_ambient.get("ambient_temperature_C"),
            name="ambient.ambient_temperature_C",
        )
        validate_strictly_increasing(cryostage_temperature_C, name="ambient.cryostage_temperature_C")
        return InterpolatedAmbientTemperature(
            cryostage_temperature_C=cryostage_temperature_C,
            ambient_temperature_C=ambient_temperature_C,
            extrapolation=str(raw_ambient.get("extrapolation", "clamp")),
        )
    raise ValueError("ambient.mode must be either 'fixed' or 'interpolate_from_cryostage'")


def _profile_from_table(name: str, raw_profile: dict[str, Any]) -> SimulationProfileConfig:
    raw_ambient = _table(raw_profile, "ambient")
    return SimulationProfileConfig(
        name=name,
        cryostage_dt_s=float(raw_profile["cryostage_dt_s"]),
        solver_dt_s=float(raw_profile["solver_dt_s"]),
        Nr=int(raw_profile["Nr"]),
        Nz=int(raw_profile["Nz"]),
        Nz_front=int(raw_profile["Nz_front"]),
        Nr_front_curve=int(raw_profile.get("Nr_front_curve", 25)),
        Nz_front_curve=int(raw_profile.get("Nz_front_curve", 400)),
        write_every_s=float(raw_profile.get("write_every_s", 1.0e9)),
        write_field_output=bool(raw_profile.get("write_field_output", False)),
        write_probe_csv=bool(raw_profile.get("write_probe_csv", False)),
        show_progress=bool(raw_profile.get("show_progress", False)),
        enable_front_curve=bool(raw_profile.get("enable_front_curve", False)),
        use_tabulated_water_ice=bool(raw_profile.get("use_tabulated_water_ice", True)),
        ambient_temperature_from_plate_C=_ambient_model_from_table(raw_ambient),
    )


def load_simulation_profiles(path: str | Path = DEFAULT_SIMULATION_CONFIG_PATH) -> SimulationProfilesFile:
    path = Path(path)
    data = _read_toml(path)
    default_profile = str(data.get("default_profile", "optimization"))
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("simulation profile config must contain a non-empty [profiles] table")
    profiles = {
        str(name): _profile_from_table(str(name), raw_profile)
        for name, raw_profile in raw_profiles.items()
    }
    if default_profile not in profiles:
        raise ValueError(f"default_profile={default_profile!r} is not present in [profiles]")
    return SimulationProfilesFile(path=path, default_profile=default_profile, profiles=profiles)


def load_simulation_profile(
    path: str | Path = DEFAULT_SIMULATION_CONFIG_PATH,
    *,
    profile_name: str | None = None,
) -> SimulationProfileConfig:
    return load_simulation_profiles(path).get_profile(profile_name)


def _optional_float_tuple(values, *, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    out = coerce_float_tuple(values, name=name, allow_empty=True)
    return None if len(out) == 0 else out


def _theta_bounds(values) -> tuple[tuple[float, float], ...] | None:
    if values is None or len(values) == 0:
        return None
    bounds: list[tuple[float, float]] = []
    for idx, raw_pair in enumerate(values):
        pair = coerce_float_tuple(raw_pair, name=f"theta_bounds_C[{idx}]")
        if len(pair) != 2:
            raise ValueError(f"theta_bounds_C[{idx}] must contain exactly two values")
        if pair[0] >= pair[1]:
            raise ValueError(f"theta_bounds_C[{idx}] must satisfy lower < upper")
        bounds.append((float(pair[0]), float(pair[1])))
    return tuple(bounds)


def load_bo_config(path: str | Path = DEFAULT_BO_CONFIG_PATH) -> BOFileConfig:
    path = Path(path)
    data = _read_toml(path)
    trajectory = _table(data, "trajectory")
    bo = _table(data, "bayesian_optimization")
    objective = _table(data, "objective")

    T_ref_bounds_C = coerce_float_tuple(trajectory.get("T_ref_bounds_C", (-20.0, 0.0)), name="T_ref_bounds_C")
    if len(T_ref_bounds_C) != 2 or T_ref_bounds_C[0] >= T_ref_bounds_C[1]:
        raise ValueError("T_ref_bounds_C must contain [lower, upper] with lower < upper")

    config = BOFileConfig(
        path=path,
        num_knots=int(trajectory.get("num_knots", 5)),
        knot_time_schedule=str(trajectory.get("knot_time_schedule", "uniform")),
        knot_time_custom_support_tau=_optional_float_tuple(
            trajectory.get("knot_time_custom_support_tau"),
            name="knot_time_custom_support_tau",
        ),
        theta0_C=_optional_float_tuple(trajectory.get("theta0_C"), name="theta0_C"),
        T_ref_bounds_C=(float(T_ref_bounds_C[0]), float(T_ref_bounds_C[1])),
        theta_bounds_C=_theta_bounds(trajectory.get("theta_bounds_C", [])),
        optimize_knot_times=bool(trajectory.get("optimize_knot_times", False)),
        random_seed=int(bo.get("random_seed", 17)),
        init_points=int(bo.get("init_points", 4)),
        n_iter=int(bo.get("n_iter", 12)),
        acquisition_kind=str(bo.get("acquisition_kind", "ucb")),
        acquisition_kappa=float(bo.get("acquisition_kappa", 2.576)),
        acquisition_xi=float(bo.get("acquisition_xi", 0.0)),
        seed_with_theta0=bool(bo.get("seed_with_theta0", True)),
        parameterization_kind=str(bo.get("parameterization_kind", "raw_theta")),
        init_strategy=str(bo.get("init_strategy", "uniform_random")),
        init_local_sigma=float(bo.get("init_local_sigma", 0.15)),
        init_max_attempts_per_point=int(bo.get("init_max_attempts_per_point", 40)),
        local_refinement_points=int(bo.get("local_refinement_points", 0)),
        local_refinement_sigma=float(bo.get("local_refinement_sigma", 0.08)),
        infeasible_objective_penalty=float(objective.get("infeasible_objective_penalty", 1.0e6)),
    )
    if config.num_knots < 2:
        raise ValueError("trajectory.num_knots must be at least 2")
    if config.theta0_C is not None and len(config.theta0_C) != config.num_knots:
        raise ValueError("trajectory.theta0_C length must match trajectory.num_knots")
    if config.optimize_knot_times:
        raise ValueError("trajectory.optimize_knot_times=true is reserved for a later workflow phase")
    if config.init_points < 0 or config.n_iter < 0:
        raise ValueError("BO init_points and n_iter must be non-negative")
    if config.acquisition_kind not in {"ucb", "ei", "poi"}:
        raise ValueError("BO acquisition_kind must be one of 'ucb', 'ei', or 'poi'")
    if config.parameterization_kind not in {"raw_theta", "monotone_unit_box"}:
        raise ValueError("BO parameterization_kind must be 'raw_theta' or 'monotone_unit_box'")
    if config.init_strategy not in {"uniform_random", "feasible_local", "feasible_local_deterministic"}:
        raise ValueError(
            "BO init_strategy must be 'uniform_random', 'feasible_local', or 'feasible_local_deterministic'"
        )
    if not math.isfinite(config.init_local_sigma) or config.init_local_sigma <= 0.0:
        raise ValueError("BO init_local_sigma must be finite and positive")
    if config.init_max_attempts_per_point <= 0:
        raise ValueError("BO init_max_attempts_per_point must be positive")
    if config.local_refinement_points < 0:
        raise ValueError("BO local_refinement_points must be non-negative")
    if not math.isfinite(config.local_refinement_sigma) or config.local_refinement_sigma <= 0.0:
        raise ValueError("BO local_refinement_sigma must be finite and positive")
    if not math.isfinite(config.infeasible_objective_penalty) or config.infeasible_objective_penalty <= 0.0:
        raise ValueError("objective.infeasible_objective_penalty must be finite and positive")
    return config


def load_velocity_control_config(
    path: str | Path = DEFAULT_VELOCITY_CONTROL_CONFIG_PATH,
) -> VelocityControlFileConfig:
    path = Path(path)
    data = _read_toml(path)
    run = _table(data, "run")
    velocity_target = _table(data, "velocity_target")
    initial_conditions = _table(data, "initial_conditions")
    temperature_uncertainty = _table(data, "temperature_uncertainty")
    manual_trajectory = _table(data, "manual_trajectory")

    custom_support = _optional_float_tuple(
        manual_trajectory.get("knot_time_custom_support_tau", []),
        name="manual_trajectory.knot_time_custom_support_tau",
    )
    config = VelocityControlFileConfig(
        path=path,
        run_name=str(run.get("run_name", "fine_confirm_v0p008_n3_uniform")),
        simulation_profile=str(run.get("simulation_profile", "optimization")),
        velocity_target=VelocityTargetConfig(
            target_front_speed_mm_s=float(velocity_target.get("target_front_speed_mm_s", 0.008)),
            control_z_min_mm=float(velocity_target.get("control_z_min_mm", 3.0)),
            control_z_max_mm=float(velocity_target.get("control_z_max_mm", 11.0)),
        ),
        initial_conditions=InitialConditionsConfig(
            initial_water_temperature_C=float(initial_conditions.get("initial_water_temperature_C", 12.5)),
            initial_plate_temperature_C=float(initial_conditions.get("initial_plate_temperature_C", 2.5)),
            no_warm_hold=bool(initial_conditions.get("no_warm_hold", True)),
        ),
        temperature_uncertainty=TemperatureUncertaintyConfig(
            characterization_temperature_margin_C=float(
                temperature_uncertainty.get("characterization_temperature_margin_C", 0.5)
            ),
        ),
        manual_trajectory=ManualTrajectoryConfig(
            horizon_s=float(manual_trajectory.get("horizon_s", 2400.0)),
            num_knots=int(manual_trajectory.get("num_knots", 3)),
            knot_time_schedule=str(manual_trajectory.get("knot_time_schedule", "uniform")),
            knot_time_custom_support_tau=custom_support,
            theta_C=coerce_float_tuple(manual_trajectory.get("theta_C", (0.0, -10.0, -20.0)), name="manual_trajectory.theta_C"),
            T_ref_bounds_C=coerce_float_tuple(
                manual_trajectory.get("T_ref_bounds_C", (-21.0, 0.0)),
                name="manual_trajectory.T_ref_bounds_C",
            ),
            require_monotone_nonincreasing=bool(manual_trajectory.get("require_monotone_nonincreasing", True)),
        ),
    )
    if not config.run_name.strip():
        raise ValueError("run.run_name must not be empty")
    if not config.simulation_profile.strip():
        raise ValueError("run.simulation_profile must not be empty")
    return config


def override_simulation_profile(
    profile: SimulationProfileConfig,
    *,
    cryostage_dt_s: float | None = None,
    solver_dt_s: float | None = None,
    Nr: int | None = None,
    Nz: int | None = None,
    Nz_front: int | None = None,
    ambient_mode: str | None = None,
) -> SimulationProfileConfig:
    updates: dict[str, object] = {}
    if cryostage_dt_s is not None:
        updates["cryostage_dt_s"] = float(cryostage_dt_s)
    if solver_dt_s is not None:
        updates["solver_dt_s"] = float(solver_dt_s)
    if Nr is not None:
        updates["Nr"] = int(Nr)
    if Nz is not None:
        updates["Nz"] = int(Nz)
    if Nz_front is not None:
        updates["Nz_front"] = int(Nz_front)
    if ambient_mode is not None:
        mode = str(ambient_mode).strip().lower()
        if mode == "fixed":
            updates["ambient_temperature_from_plate_C"] = None
        elif mode == "interpolate_from_cryostage":
            if not isinstance(profile.ambient_temperature_from_plate_C, InterpolatedAmbientTemperature):
                raise ValueError(
                    "--ambient-mode=interpolate_from_cryostage requires calibration points in the selected profile"
                )
        else:
            raise ValueError("--ambient-mode must be either 'fixed' or 'interpolate_from_cryostage'")
    return replace(profile, **updates) if updates else profile


__all__ = [
    "BOFileConfig",
    "DEFAULT_BO_CONFIG_PATH",
    "DEFAULT_SIMULATION_CONFIG_PATH",
    "DEFAULT_VELOCITY_CONTROL_CONFIG_PATH",
    "InitialConditionsConfig",
    "ManualTrajectoryConfig",
    "SimulationProfileConfig",
    "SimulationProfilesFile",
    "TemperatureUncertaintyConfig",
    "VelocityControlFileConfig",
    "VelocityTargetConfig",
    "load_bo_config",
    "load_simulation_profile",
    "load_simulation_profiles",
    "load_velocity_control_config",
    "override_simulation_profile",
]
