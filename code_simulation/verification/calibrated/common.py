from __future__ import annotations

"""Shared paths and discovery helpers for calibrated constant-plate runs."""

from dataclasses import dataclass
from pathlib import Path
import csv
import re

from code_simulation.core.paths import project_root, results_dir
from code_simulation.verification.run_pre_stabilized_calibrated_simulations import (
    DEFAULT_H_OUT_W_M2K,
    DEFAULT_T_FILL_C,
    _simulation_naming,
    ambient_temperature_for_plate_C,
)


CALIBRATED_SIMULATIONS_DIR = project_root() / "data" / "simulations_calibrated"
WITH_RHO_FIXED_DIR = CALIBRATED_SIMULATIONS_DIR / "with_rho_fixed"
WITH_RHO_TEMPERATURE_DEPENDENT_DIR = CALIBRATED_SIMULATIONS_DIR / "with_rho_temperature_dependent"
CALIBRATED_FIGURES_DIR = CALIBRATED_SIMULATIONS_DIR / "figures"
ACTIVE_VELOCITY_STUDY_DIR = results_dir("active", "velocity_study")
ARCHIVE_MANIFEST_PATH = WITH_RHO_FIXED_DIR / "archive_manifest.csv"

ACTIVE_TARGETS_C = (-5.0, -10.0, -15.0, -20.0, -21.0)
COMPARISON_TARGETS_C = (-10.0, -15.0, -20.0)
EXPERIMENTAL_SUBDIR_BY_TARGET = {
    -10.0: "min10",
    -15.0: "min15",
    -20.0: "min20",
}

TARGET_RE = re.compile(r"plate_(?P<sign>[mp])(?P<target>[0-9]+(?:p[0-9]+)?)")


@dataclass(frozen=True)
class CalibratedTargetPlan:
    target_C: float
    ambient_C: float
    output_dir: Path
    prefix: str


@dataclass(frozen=True)
class CalibratedSimulationOutput:
    target_C: float
    directory: Path
    probes_csv: Path
    front_csv: Path
    metadata_csv: Path | None
    prefix: str


@dataclass(frozen=True)
class ArchiveMove:
    source: Path
    destination: Path


@dataclass(frozen=True)
class ArchiveResult:
    status: str
    moves: tuple[ArchiveMove, ...]
    manifest_path: Path


def _value_tag(value: float, *, signed_zero_prefix: bool = False) -> str:
    value = float(value)
    if abs(value) < 5.0e-13:
        return "p0" if signed_zero_prefix else "0"
    prefix = "m" if value < 0.0 else ""
    text = f"{abs(value):.4f}".rstrip("0").rstrip(".")
    return prefix + text.replace(".", "p")


def target_filename_tag(target_C: float) -> str:
    target_C = float(target_C)
    if target_C < 0.0:
        return f"min{_value_tag(abs(target_C))}"
    return f"plus{_value_tag(target_C)}"


def build_target_plan(target_C: float, *, output_root: Path = WITH_RHO_TEMPERATURE_DEPENDENT_DIR) -> CalibratedTargetPlan:
    ambient_C = ambient_temperature_for_plate_C(target_C)
    naming = _simulation_naming(
        output_root=Path(output_root),
        T_plate_C=float(target_C),
        T_ambient_C=float(ambient_C),
        h_out_W_m2K=DEFAULT_H_OUT_W_M2K,
        T_fill_C=DEFAULT_T_FILL_C,
    )
    return CalibratedTargetPlan(
        target_C=float(target_C),
        ambient_C=float(ambient_C),
        output_dir=naming.output_dir,
        prefix=naming.prefix,
    )


def build_target_plans(
    targets_C: tuple[float, ...] = ACTIVE_TARGETS_C,
    *,
    output_root: Path = WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
) -> tuple[CalibratedTargetPlan, ...]:
    return tuple(build_target_plan(target_C, output_root=output_root) for target_C in targets_C)


def experimental_csv_paths(
    target_C: float,
    *,
    experiments_root: Path | None = None,
) -> tuple[Path, ...]:
    root = project_root() / "data" / "constant_plateT_water_ICT_readings" if experiments_root is None else Path(experiments_root)
    subdir = EXPERIMENTAL_SUBDIR_BY_TARGET.get(float(target_C))
    if subdir is None:
        return ()
    return tuple(sorted((root / subdir).glob("*.csv")))


def list_legacy_archive_candidates(
    base_dir: Path = CALIBRATED_SIMULATIONS_DIR,
) -> tuple[Path, ...]:
    base_dir = Path(base_dir)
    candidates: list[Path] = []
    for path in sorted(base_dir.iterdir(), key=lambda item: item.name):
        if path.name in {WITH_RHO_FIXED_DIR.name, WITH_RHO_TEMPERATURE_DEPENDENT_DIR.name}:
            continue
        if path.name == "figures" and path.is_dir():
            candidates.append(path)
        elif path.is_dir() and path.name.startswith("plate_"):
            candidates.append(path)
    return tuple(candidates)


def prepare_calibrated_results_layout(
    base_dir: Path = CALIBRATED_SIMULATIONS_DIR,
    *,
    dry_run: bool = False,
) -> ArchiveResult:
    base_dir = Path(base_dir)
    fixed_dir = base_dir / WITH_RHO_FIXED_DIR.name
    temp_dir = base_dir / WITH_RHO_TEMPERATURE_DEPENDENT_DIR.name
    manifest_path = fixed_dir / ARCHIVE_MANIFEST_PATH.name
    legacy_items = list_legacy_archive_candidates(base_dir)

    if manifest_path.exists():
        stray_legacy_dirs = tuple(item for item in legacy_items if item.name.startswith("plate_"))
        if stray_legacy_dirs:
            names = ", ".join(item.name for item in stray_legacy_dirs)
            raise RuntimeError(
                "Archive manifest already exists but legacy plate_* directories remain at the top level: "
                f"{names}"
            )
        return ArchiveResult(status="already_archived", moves=(), manifest_path=manifest_path)

    moves = tuple(ArchiveMove(source=item, destination=fixed_dir / item.name) for item in legacy_items)
    for move in moves:
        if move.destination.exists():
            raise FileExistsError(
                f"Cannot archive {move.source.name}: destination already exists at {move.destination}"
            )

    if dry_run:
        status = "would_archive" if moves else "would_initialize"
        return ArchiveResult(status=status, moves=moves, manifest_path=manifest_path)

    fixed_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for move in moves:
        move.source.rename(move.destination)

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_name", "archived_relpath"])
        for move in moves:
            writer.writerow([move.source.name, move.destination.relative_to(base_dir)])

    status = "archived" if moves else "initialized"
    return ArchiveResult(status=status, moves=moves, manifest_path=manifest_path)


def _target_from_directory_name(name: str) -> float | None:
    match = TARGET_RE.search(name)
    if match is None:
        return None
    magnitude = float(match.group("target").replace("p", "."))
    return -magnitude if match.group("sign") == "m" else magnitude


def _find_unique_file(directory: Path, pattern: str, *, required: bool) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        if required:
            raise FileNotFoundError(f"No files matched {pattern!r} in {directory}")
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one file matching {pattern!r} in {directory}, found {len(matches)}")
    return matches[0]


def discover_simulation_outputs(
    simulations_dir: Path = WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    *,
    expected_targets: tuple[float, ...] | None = None,
) -> dict[float, CalibratedSimulationOutput]:
    simulations_dir = Path(simulations_dir)
    outputs: dict[float, CalibratedSimulationOutput] = {}
    for directory in sorted(simulations_dir.glob("plate_*_probe_stabilized_*")):
        if not directory.is_dir():
            continue
        target_C = _target_from_directory_name(directory.name)
        if target_C is None:
            continue
        probes_csv = _find_unique_file(directory, "*_probes.csv", required=True)
        front_csv = _find_unique_file(directory, "*_front.csv", required=True)
        metadata_csv = _find_unique_file(directory, "*_metadata.csv", required=False)
        prefix = probes_csv.name[: -len("_probes.csv")]
        outputs[float(target_C)] = CalibratedSimulationOutput(
            target_C=float(target_C),
            directory=directory,
            probes_csv=probes_csv,
            front_csv=front_csv,
            metadata_csv=metadata_csv,
            prefix=prefix,
        )

    if expected_targets is not None:
        expected = {float(value) for value in expected_targets}
        missing = sorted(expected - set(outputs), reverse=True)
        if missing:
            raise RuntimeError(f"Missing simulation targets in {simulations_dir}: {missing}")
        outputs = {key: value for key, value in outputs.items() if key in expected}

    return dict(sorted(outputs.items(), key=lambda item: item[0], reverse=True))


__all__ = [
    "ACTIVE_TARGETS_C",
    "ACTIVE_VELOCITY_STUDY_DIR",
    "ARCHIVE_MANIFEST_PATH",
    "ArchiveMove",
    "ArchiveResult",
    "CALIBRATED_FIGURES_DIR",
    "CALIBRATED_SIMULATIONS_DIR",
    "COMPARISON_TARGETS_C",
    "CalibratedSimulationOutput",
    "CalibratedTargetPlan",
    "EXPERIMENTAL_SUBDIR_BY_TARGET",
    "WITH_RHO_FIXED_DIR",
    "WITH_RHO_TEMPERATURE_DEPENDENT_DIR",
    "build_target_plan",
    "build_target_plans",
    "discover_simulation_outputs",
    "experimental_csv_paths",
    "list_legacy_archive_candidates",
    "prepare_calibrated_results_layout",
    "target_filename_tag",
]
