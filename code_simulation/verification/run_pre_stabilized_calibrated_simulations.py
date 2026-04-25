#!/usr/bin/env python3
"""Run fine pre-stabilized constant-plate calibration simulations."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code_simulation.core.paths import project_root
from code_simulation.simulation.geometry import GeometryParams
from code_simulation.simulation.solver import (
    FreezeStopOptions,
    PhaseChangeParams,
    PrefillOptions,
    ThermalBCs,
    run_case,
)


AMBIENT_CALIBRATION_POINTS = (
    (-20.0, 5.75),
    (-15.0, 7.568),
    (-10.0, 9.7078),
)
DEFAULT_TARGETS_C = (0.0, -5.0, -21.0)
DEFAULT_T_FILL_C = 12.5
DEFAULT_H_OUT_W_M2K = 2.0
DEFAULT_T_AFTER_FILL_S = 3600.0
DEFAULT_WRITE_EVERY_S = 5.0
PROBE_Z_M = (3.0e-3, 6.2e-3, 11.0e-3)
PROBE_WALL_INSET_M = 1.0e-3


@dataclass(frozen=True)
class SimulationNaming:
    output_dir: Path
    prefix: str
    plate_tag: str
    ambient_tag: str


def _parse_targets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one target temperature")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target temperatures must be finite")
    return values


def _value_tag(value: float, *, signed_zero_prefix: bool = False) -> str:
    value = float(value)
    if abs(value) < 5.0e-13:
        return "p0" if signed_zero_prefix else "0"
    prefix = "m" if value < 0.0 else ""
    text = f"{abs(value):.4f}".rstrip("0").rstrip(".")
    return prefix + text.replace(".", "p")


def ambient_temperature_for_plate_C(T_plate_C: float) -> float:
    """Piecewise-linear interpolation/extrapolation from measured ambient points."""
    points = sorted(AMBIENT_CALIBRATION_POINTS, key=lambda item: item[0])
    x = np.asarray([point[0] for point in points], dtype=np.float64)
    y = np.asarray([point[1] for point in points], dtype=np.float64)
    target = float(T_plate_C)
    if target <= float(x[0]):
        x0, x1 = float(x[0]), float(x[1])
        y0, y1 = float(y[0]), float(y[1])
    elif target >= float(x[-1]):
        x0, x1 = float(x[-2]), float(x[-1])
        y0, y1 = float(y[-2]), float(y[-1])
    else:
        return float(np.interp(target, x, y))
    slope = (y1 - y0) / (x1 - x0)
    return float(y0 + slope * (target - x0))


def _simulation_naming(
    *,
    output_root: Path,
    T_plate_C: float,
    T_ambient_C: float,
    h_out_W_m2K: float,
    T_fill_C: float,
) -> SimulationNaming:
    plate_tag = _value_tag(T_plate_C, signed_zero_prefix=True)
    ambient_tag = _value_tag(T_ambient_C)
    h_tag = _value_tag(h_out_W_m2K)
    fill_tag = _value_tag(T_fill_C)
    suffix = (
        f"plate_{plate_tag}_probe_stabilized_"
        f"Tamb_{ambient_tag}_"
        f"h_{h_tag}_"
        f"Tfill_{fill_tag}_"
        "z_3p0_6p2_11p0mm_"
        "inset_1p0mm"
    )
    return SimulationNaming(
        output_dir=output_root / suffix,
        prefix=f"water_PLA_calib_{suffix}",
        plate_tag=plate_tag,
        ambient_tag=ambient_tag,
    )


def _write_metadata(
    path: Path,
    *,
    target_C: float,
    ambient_C: float,
    t_after_fill_s: float,
    write_field_output: bool,
    enable_front_curve: bool,
) -> None:
    rows = [
        ("T_plate_C", f"{float(target_C):.9f}"),
        ("T_ambient_C", f"{float(ambient_C):.9f}"),
        ("ambient_model", "piecewise_linear_extrapolated_from_plate_m10_m15_m20"),
        ("ambient_points", ";".join(f"{x:g}:{y:g}" for x, y in AMBIENT_CALIBRATION_POINTS)),
        ("T_fill_C", f"{DEFAULT_T_FILL_C:.9f}"),
        ("h_top_W_m2K", f"{DEFAULT_H_OUT_W_M2K:.9f}"),
        ("h_side_W_m2K", f"{DEFAULT_H_OUT_W_M2K:.9f}"),
        ("Nr", "180"),
        ("Nz", "408"),
        ("dt_s", "0.25"),
        ("Nz_front", "800"),
        ("probe_z_mm", "3.0,6.2,11.0"),
        ("probe_wall_inset_mm", "1.0"),
        ("prefill_mode", "probe_stabilized"),
        ("prefill_probe_window_s", "60.0"),
        ("prefill_probe_tol_C", "0.05"),
        ("prefill_min_s", "60.0"),
        ("prefill_max_s", "3600.0"),
        ("t_after_fill_s", f"{float(t_after_fill_s):.9f}"),
        ("dT_mushy_C", "0.5"),
        ("write_field_output", str(bool(write_field_output))),
        ("enable_front_curve", str(bool(enable_front_curve))),
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def run_target(
    *,
    target_C: float,
    output_root: Path,
    t_after_fill_s: float,
    overwrite: bool,
    write_field_output: bool,
    enable_front_curve: bool,
    show_progress: bool,
) -> Path:
    ambient_C = ambient_temperature_for_plate_C(target_C)
    naming = _simulation_naming(
        output_root=output_root,
        T_plate_C=target_C,
        T_ambient_C=ambient_C,
        h_out_W_m2K=DEFAULT_H_OUT_W_M2K,
        T_fill_C=DEFAULT_T_FILL_C,
    )
    if naming.output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{naming.output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(naming.output_dir)
    naming.output_dir.mkdir(parents=True, exist_ok=True)

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
    freeze_stop = FreezeStopOptions(mode="fillable_region", extra_subcooling_C=0.0)
    bcs = ThermalBCs(
        T_room_C=ambient_C,
        h_top=DEFAULT_H_OUT_W_M2K,
        h_side=DEFAULT_H_OUT_W_M2K,
    )

    print(
        f"\nRunning plate {target_C:.3f} C with extrapolated Tamb {ambient_C:.4f} C\n"
        f"  output: {naming.output_dir}"
    )
    _write_metadata(
        naming.output_dir / f"{naming.prefix}_metadata.csv",
        target_C=target_C,
        ambient_C=ambient_C,
        t_after_fill_s=t_after_fill_s,
        write_field_output=write_field_output,
        enable_front_curve=enable_front_curve,
    )
    run_case(
        out_dir=naming.output_dir,
        prefix=naming.prefix,
        geom=geom,
        Nr=180,
        Nz=408,
        dt=0.25,
        pre_cool_s=0.0,
        t_after_fill_s=float(t_after_fill_s),
        write_every=DEFAULT_WRITE_EVERY_S,
        write_field_output=bool(write_field_output),
        write_probe_csv=True,
        show_progress=bool(show_progress),
        T_fill_C=DEFAULT_T_FILL_C,
        T_plate_C=float(target_C),
        bcs=bcs,
        phase=phase,
        prefill=prefill,
        freeze_stop=freeze_stop,
        probe_z_m=PROBE_Z_M,
        probe_wall_inset_m=PROBE_WALL_INSET_M,
        Nz_front=800,
        enable_front_curve=bool(enable_front_curve),
        Nr_front_curve=25,
        Nz_front_curve=400,
        front_curve_every_s=DEFAULT_WRITE_EVERY_S,
        stop_when_wall_frozen=False,
        use_tabulated_water_ice=True,
        debug_material_probe_index=None,
    )
    return naming.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine constant-plate, pre-stabilized calibrated simulations."
    )
    parser.add_argument(
        "--targets-c",
        default=",".join(f"{value:g}" for value in DEFAULT_TARGETS_C),
        help="Comma-separated plate temperatures in C.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "data" / "simulations_calibrated",
        help="Root directory for calibrated simulation outputs.",
    )
    parser.add_argument("--t-after-fill-s", type=float, default=DEFAULT_T_AFTER_FILL_S)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-field-output", action="store_true", help="Write XDMF/H5 fields.")
    parser.add_argument("--enable-front-curve", action="store_true", help="Write curved-front CSV.")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver progress output.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved targets, ambient temperatures, and output paths without running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets_C = _parse_targets(args.targets_c)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for target_C in targets_C:
        ambient_C = ambient_temperature_for_plate_C(target_C)
        naming = _simulation_naming(
            output_root=output_root,
            T_plate_C=target_C,
            T_ambient_C=ambient_C,
            h_out_W_m2K=DEFAULT_H_OUT_W_M2K,
            T_fill_C=DEFAULT_T_FILL_C,
        )
        if args.dry_run:
            print(
                f"target={target_C:.6g} C  Tamb={ambient_C:.6g} C  "
                f"out={naming.output_dir}"
            )
            continue
        run_target(
            target_C=target_C,
            output_root=output_root,
            t_after_fill_s=float(args.t_after_fill_s),
            overwrite=bool(args.overwrite),
            write_field_output=bool(args.write_field_output),
            enable_front_curve=bool(args.enable_front_curve),
            show_progress=not bool(args.quiet),
        )


if __name__ == "__main__":
    main()
