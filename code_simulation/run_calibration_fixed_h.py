from __future__ import annotations

from pathlib import Path

from geometry import GeometryParams
from solver import (
    FreezeStopOptions,
    PhaseChangeParams,
    PrefillOptions,
    ThermalBCs,
    run_case,
)


FIXED_T_AMB_C = 5.75
FIXED_T_PLATE_C = -20.0
FIXED_T_FILL_C = 12.5
FIXED_H_OUT_W_M2K = 2.0

PROBE_Z_M = (3.0e-3, 6.2e-3, 11.0e-3)
PROBE_WALL_INSET_M = 1.0e-3


def _tag(value: float) -> str:
    """Convert a float into a filename-safe tag, e.g. 9.8594 -> 9p8594."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    out_root = root / "results_calibration" / (
        "plate_m20_probe_stabilized_"
        f"Tamb_{_tag(FIXED_T_AMB_C)}_"
        f"h_{_tag(FIXED_H_OUT_W_M2K)}_"
        f"Tfill_{_tag(FIXED_T_FILL_C)}_"
        "z_3p0_6p2_11p0mm_"
        "inset_1p0mm"
    )
    out_root.mkdir(parents=True, exist_ok=True)

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

    prefix = (
        "water_PLA_calib_plate_m20_probe_stabilized_"
        f"Tamb_{_tag(FIXED_T_AMB_C)}_"
        f"h_{_tag(FIXED_H_OUT_W_M2K)}_"
        f"Tfill_{_tag(FIXED_T_FILL_C)}_"
        "z_3p0_6p2_11p0mm_"
        "inset_1p0mm"
    )

    bcs = ThermalBCs(
        T_room_C=FIXED_T_AMB_C,
        h_top=FIXED_H_OUT_W_M2K,
        h_side=FIXED_H_OUT_W_M2K,
    )

    print(f"Using fixed ambient temperature Tamb = {FIXED_T_AMB_C:.4f} °C")
    print(f"Using fixed plate temperature Tplate = {FIXED_T_PLATE_C:.1f} °C")
    print(f"Using fixed fill temperature Tfill = {FIXED_T_FILL_C:.1f} °C")
    print(f"Using fixed convection coefficient h_out = {FIXED_H_OUT_W_M2K:.1f} W/m²K")
    print(
        "Using probe heights z = "
        f"{PROBE_Z_M[0]*1000:.1f}, {PROBE_Z_M[1]*1000:.1f}, {PROBE_Z_M[2]*1000:.1f} mm"
    )
    print(f"Using probe wall inset = {PROBE_WALL_INSET_M*1000:.1f} mm")
    print(f"Output folder: {out_root}")

    run_case(
        out_dir=out_root,
        prefix=prefix,
        geom=geom,
        Nr=180,
        Nz=408,
        dt=0.25,
        pre_cool_s=0.0,
        t_after_fill_s=3600.0,
        write_every=5.0,
        T_fill_C=FIXED_T_FILL_C,
        T_plate_C=FIXED_T_PLATE_C,
        bcs=bcs,
        phase=phase,
        prefill=prefill,
        freeze_stop=freeze_stop,
        probe_z_m=PROBE_Z_M,
        probe_wall_inset_m=PROBE_WALL_INSET_M,
        Nz_front=800,
        enable_front_curve=True,
        Nr_front_curve=25,
        Nz_front_curve=400,
        front_curve_every_s=5.0,
        stop_when_wall_frozen=False,
        use_tabulated_water_ice=True,
        debug_material_probe_index=2,
        debug_material_every_s=5.0,
    )


if __name__ == "__main__":
    main()
