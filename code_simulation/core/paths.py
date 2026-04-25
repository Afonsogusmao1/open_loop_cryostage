from __future__ import annotations

from pathlib import Path


CODE_SIMULATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_SIMULATION_ROOT.parent
RESULTS_ROOT = CODE_SIMULATION_ROOT / "results"
ACTIVE_RESULTS_ROOT = RESULTS_ROOT / "active"
LEGACY_VELOCITY_CONTROL_RESULTS_ROOT = ACTIVE_RESULTS_ROOT / "bo_velocity_control"
CONFIGS_ROOT = CODE_SIMULATION_ROOT / "configs"
BAYESOPT_COMPAT_DIR = CODE_SIMULATION_ROOT / "_vendor_bayesopt_compat"


def code_simulation_root() -> Path:
    return CODE_SIMULATION_ROOT


def project_root() -> Path:
    return PROJECT_ROOT


def results_dir(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)


def active_results_dir(*parts: str) -> Path:
    return ACTIVE_RESULTS_ROOT.joinpath(*parts)


def active_knot_dir(num_knots: int, *parts: str) -> Path:
    return active_results_dir(f"n{int(num_knots)}", *parts)


def legacy_velocity_control_results_dir(*parts: str) -> Path:
    return LEGACY_VELOCITY_CONTROL_RESULTS_ROOT.joinpath(*parts)


def legacy_velocity_control_knot_dir(num_knots: int, *parts: str) -> Path:
    return legacy_velocity_control_results_dir(f"n{int(num_knots)}", *parts)


def _prefer_active_path(active_path: Path, legacy_path: Path) -> Path:
    if active_path.exists() or not legacy_path.exists():
        return active_path
    return legacy_path


def velocity_control_results_dir(*parts: str) -> Path:
    return active_results_dir(*parts)


def velocity_control_knot_dir(num_knots: int, *parts: str) -> Path:
    return _prefer_active_path(
        active_knot_dir(num_knots, *parts),
        legacy_velocity_control_knot_dir(num_knots, *parts),
    )


def velocity_control_coarse_dir(num_knots: int, *parts: str) -> Path:
    return _prefer_active_path(
        active_knot_dir(num_knots, "coarse", *parts),
        legacy_velocity_control_knot_dir(num_knots, "coarse", *parts),
    )


def velocity_control_fine_dir(num_knots: int, *parts: str) -> Path:
    return _prefer_active_path(
        active_knot_dir(num_knots, "fine", *parts),
        legacy_velocity_control_knot_dir(num_knots, "fine", *parts),
    )


def velocity_control_diagnostic_dir(num_knots: int, *parts: str) -> Path:
    return _prefer_active_path(
        active_knot_dir(num_knots, "diagnostico", *parts),
        legacy_velocity_control_knot_dir(num_knots, "diagnostico", *parts),
    )


def configs_dir(*parts: str) -> Path:
    return CONFIGS_ROOT.joinpath(*parts)


def bayesopt_compat_dir() -> Path:
    return BAYESOPT_COMPAT_DIR


__all__ = [
    "ACTIVE_RESULTS_ROOT",
    "BAYESOPT_COMPAT_DIR",
    "CODE_SIMULATION_ROOT",
    "CONFIGS_ROOT",
    "LEGACY_VELOCITY_CONTROL_RESULTS_ROOT",
    "PROJECT_ROOT",
    "RESULTS_ROOT",
    "active_knot_dir",
    "active_results_dir",
    "bayesopt_compat_dir",
    "code_simulation_root",
    "configs_dir",
    "legacy_velocity_control_knot_dir",
    "legacy_velocity_control_results_dir",
    "project_root",
    "results_dir",
    "velocity_control_coarse_dir",
    "velocity_control_diagnostic_dir",
    "velocity_control_fine_dir",
    "velocity_control_knot_dir",
    "velocity_control_results_dir",
]
