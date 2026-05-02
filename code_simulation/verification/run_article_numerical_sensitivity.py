#!/usr/bin/env python3
"""Run article-level mesh, time-step, and front-sampling sensitivity cases."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from code_simulation.core.paths import project_root
from code_simulation.simulation.geometry import GeometryParams
from code_simulation.simulation.solver import (
    FreezeStopOptions,
    PhaseChangeParams,
    PrefillOptions,
    ThermalBCs,
    run_case,
)

AMBIENT_CALIBRATION_POINTS = {
    -20.0: 5.75,
    -15.0: 7.568,
    -10.0: 9.7078,
}

PROBE_Z_M = (3.0e-3, 6.2e-3, 11.0e-3)
PROBE_WALL_INSET_M = 1.0e-3

DEFAULT_T_FILL_C = 12.5
DEFAULT_H_W_M2K = 2.0
DEFAULT_T_AFTER_FILL_S = 3600.0
DEFAULT_WRITE_EVERY_S = 5.0


def value_tag(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    prefix = "m" if value < 0 else "p"
    text = f"{abs(value):.6g}".replace(".", "p")
    return prefix + text


def ambient_temperature_for_plate(T_plate_C: float) -> float:
    """Use measured ambient values for article validation setpoints."""
    if T_plate_C in AMBIENT_CALIBRATION_POINTS:
        return AMBIENT_CALIBRATION_POINTS[T_plate_C]
    raise ValueError(
        f"No ambient calibration value defined for T_plate_C={T_plate_C}. "
        f"Available: {sorted(AMBIENT_CALIBRATION_POINTS)}"
    )


def write_case_metadata(path: Path, row: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for key, value in row.items():
            writer.writerow([key, value])


def run_one_case(
    *,
    output_root: Path,
    case_group: str,
    case_name: str,
    T_plate_C: float,
    Nr: int,
    Nz: int,
    dt: float,
    Nz_front: int,
    Nr_front_curve: int,
    Nz_front_curve: int,
    overwrite: bool,
    write_field_output: bool,
    enable_front_curve: bool,
    quiet: bool,
) -> Path:
    Tamb_C = ambient_temperature_for_plate(T_plate_C)

    out_dir = (
        output_root
        / case_group
        / f"{case_name}_plate_{value_tag(T_plate_C)}_Nr{Nr}_Nz{Nz}_dt{value_tag(dt)}"
    )
    prefix = f"{case_group}_{case_name}_plate_{value_tag(T_plate_C)}"

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    geom = GeometryParams(
        R_in=7.5e-3,
        t_wall=2.0e-3,
        t_base=0.0,
        H_fill=15.0e-3,
        H_total=17.0e-3,
    )

    phase = PhaseChangeParams(Tf=0.0, L_latent=334000.0, dT_mushy=0.5)

    prefill = PrefillOptions(
        mode="probe_stabilized",
        probe_window_s=60.0,
        probe_tol_C=0.05,
        min_prefill_s=60.0,
        max_prefill_s=3600.0,
    )

    freeze_stop = FreezeStopOptions(
        mode="fillable_region",
        extra_subcooling_C=0.0,
    )

    bcs = ThermalBCs(
        T_room_C=Tamb_C,
        h_top=DEFAULT_H_W_M2K,
        h_side=DEFAULT_H_W_M2K,
    )

    metadata = {
        "case_group": case_group,
        "case_name": case_name,
        "T_plate_C": T_plate_C,
        "T_ambient_C": Tamb_C,
        "T_fill_C": DEFAULT_T_FILL_C,
        "h_top_W_m2K": DEFAULT_H_W_M2K,
        "h_side_W_m2K": DEFAULT_H_W_M2K,
        "Nr": Nr,
        "Nz": Nz,
        "dt_s": dt,
        "Nz_front": Nz_front,
        "Nr_front_curve": Nr_front_curve,
        "Nz_front_curve": Nz_front_curve,
        "probe_z_mm": "3.0,6.2,11.0",
        "probe_wall_inset_mm": "1.0",
        "prefill_mode": "probe_stabilized",
        "prefill_probe_window_s": "60.0",
        "prefill_probe_tol_C": "0.05",
        "prefill_min_s": "60.0",
        "prefill_max_s": "3600.0",
        "t_after_fill_s": DEFAULT_T_AFTER_FILL_S,
        "use_tabulated_water_ice": True,
        "front_definition_mode": "isotherm_Tf",
        "write_field_output": write_field_output,
        "enable_front_curve": enable_front_curve,
    }
    write_case_metadata(out_dir / f"{prefix}_metadata.csv", metadata)

    print(f"\nRunning {case_group}/{case_name}")
    print(f"  plate={T_plate_C} °C, Tamb={Tamb_C} °C")
    print(f"  Nr={Nr}, Nz={Nz}, dt={dt}")
    print(f"  out={out_dir}")

    run_case(
        out_dir=out_dir,
        prefix=prefix,
        geom=geom,
        Nr=int(Nr),
        Nz=int(Nz),
        dt=float(dt),
        pre_cool_s=0.0,
        t_after_fill_s=DEFAULT_T_AFTER_FILL_S,
        write_every=DEFAULT_WRITE_EVERY_S,
        write_field_output=bool(write_field_output),
        write_probe_csv=True,
        show_progress=not quiet,
        T_fill_C=DEFAULT_T_FILL_C,
        T_plate_C=float(T_plate_C),
        bcs=bcs,
        phase=phase,
        prefill=prefill,
        freeze_stop=freeze_stop,
        front_definition_mode="isotherm_Tf",
        probe_z_m=PROBE_Z_M,
        probe_wall_inset_m=PROBE_WALL_INSET_M,
        Nz_front=int(Nz_front),
        enable_front_curve=bool(enable_front_curve),
        Nr_front_curve=int(Nr_front_curve),
        Nz_front_curve=int(Nz_front_curve),
        front_curve_every_s=DEFAULT_WRITE_EVERY_S,
        stop_when_wall_frozen=False,
        use_tabulated_water_ice=True,
    )

    return out_dir


def build_cases(T_plate_C: float) -> list[dict]:
    cases = []

    # Mesh sensitivity: fixed dt and post-processing sampling.
    for name, Nr, Nz in [
        ("mesh_coarse", 90, 204),
        ("mesh_medium", 135, 306),
        ("mesh_selected", 180, 408),
        ("mesh_reference", 240, 544),
    ]:
        cases.append(
            dict(
                case_group="mesh_sensitivity",
                case_name=name,
                T_plate_C=T_plate_C,
                Nr=Nr,
                Nz=Nz,
                dt=0.25,
                Nz_front=800,
                Nr_front_curve=25,
                Nz_front_curve=400,
            )
        )

    # Time-step sensitivity: fixed selected mesh.
    for name, dt in [
        ("dt_coarse", 1.0),
        ("dt_medium", 0.5),
        ("dt_selected", 0.25),
        ("dt_reference", 0.125),
    ]:
        cases.append(
            dict(
                case_group="time_sensitivity",
                case_name=name,
                T_plate_C=T_plate_C,
                Nr=180,
                Nz=408,
                dt=dt,
                Nz_front=800,
                Nr_front_curve=25,
                Nz_front_curve=400,
            )
        )

    # Front-sampling sensitivity: same FEM solution parameters, only extraction grid changes.
    for name, Nz_front, Nr_front_curve, Nz_front_curve in [
        ("sampling_coarse", 400, 15, 300),
        ("sampling_selected", 800, 25, 400),
        ("sampling_fine", 1200, 35, 600),
    ]:
        cases.append(
            dict(
                case_group="front_sampling_sensitivity",
                case_name=name,
                T_plate_C=T_plate_C,
                Nr=180,
                Nz=408,
                dt=0.25,
                Nz_front=Nz_front,
                Nr_front_curve=Nr_front_curve,
                Nz_front_curve=Nz_front_curve,
            )
        )

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plate-c", type=float, default=-20.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "code_simulation" / "results" / "article_numerical_sensitivity",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-field-output", action="store_true")
    parser.add_argument("--enable-front-curve", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--only", choices=["mesh", "time", "sampling", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(args.plate_c)

    if args.only != "all":
        group_prefix = {
            "mesh": "mesh_sensitivity",
            "time": "time_sensitivity",
            "sampling": "front_sampling_sensitivity",
        }[args.only]
        cases = [case for case in cases if case["case_group"] == group_prefix]

    args.output_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        if args.dry_run:
            print(case)
            continue

        run_one_case(
            output_root=args.output_root,
            overwrite=args.overwrite,
            write_field_output=args.write_field_output,
            enable_front_curve=args.enable_front_curve,
            quiet=args.quiet,
            **case,
        )


if __name__ == "__main__":
    main()