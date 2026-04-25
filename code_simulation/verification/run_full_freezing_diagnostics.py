#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code_simulation.simulation.cryostage_model import CryostageModelParams, DEFAULT_CRYOSTAGE_PARAMS
from code_simulation.simulation.open_loop_cascade import run_open_loop_case
from code_simulation.core.config_files import (
    DEFAULT_SIMULATION_CONFIG_PATH,
    load_simulation_profiles,
    override_simulation_profile,
)
from code_simulation.optimization.open_loop_workflow_config import (
    LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT,
    apply_simulation_profile,
    build_problem_config,
)
from code_simulation.simulation.ambient import describe_ambient_temperature_model
from code_simulation.simulation.solver import FreezeStopOptions, PhaseChangeParams
from code_simulation.core.trajectory_profiles import ConstantTemperatureProfile, PiecewiseLinearTemperatureProfile
from code_simulation.core.paths import results_dir
from code_simulation.verification.front_summary import (
    first_freeze_complete_time as _first_freeze_complete_time,
    first_time_at_or_above as _first_time_at_or_above,
    load_front_columns as _load_front_columns,
    nan_to_str as _nan_to_str,
)


DEFAULT_OUT_ROOT_DIR = results_dir("full_freezing_diagnostics")
DEFAULT_RUN_NAME = "current_snapshot"
DEFAULT_EARLY_DROP_STUDY_SUMMARY = (
    results_dir("open_loop_study")
    / "sensitivity_h360_early_drop_k5_nm_iter16_fev36_ti0"
    / "study_summary.txt"
)


@dataclass(frozen=True)
class DiagnosticCase:
    case_name: str
    profile_label: str
    profile_source: str
    T_ref_profile_C: object


def _build_time_grid(horizon_s: float, dt_s: float) -> np.ndarray:
    time_s = np.arange(0.0, horizon_s, dt_s, dtype=np.float64)
    if time_s.size == 0 or time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
    if time_s[-1] < horizon_s - 1e-12:
        time_s = np.append(time_s, horizon_s)
    return time_s


def _parse_summary_txt(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if ": " not in line:
                continue
            key, value = line.rstrip("\n").split(": ", 1)
            summary[key] = value
    return summary


def _float_tuple_from_text(raw: str) -> tuple[float, ...]:
    values = ast.literal_eval(raw)
    return tuple(float(value) for value in values)


def _cryostage_params_from_summary(summary: dict[str, str]) -> CryostageModelParams:
    raw = summary.get("cryostage_params")
    if raw is None:
        return DEFAULT_CRYOSTAGE_PARAMS
    params = json.loads(raw)
    return CryostageModelParams(
        tau_s=float(params["tau_s"]),
        gain=float(params["gain"]),
        offset_C=float(params["offset_C"]),
    )


def _load_early_drop_case(summary_path: Path) -> tuple[DiagnosticCase, CryostageModelParams]:
    summary = _parse_summary_txt(summary_path)
    knot_times_s = _float_tuple_from_text(summary["knot_times_s"])
    best_theta = _float_tuple_from_text(summary["best_theta"])
    profile = PiecewiseLinearTemperatureProfile(
        knot_times_s=knot_times_s,
        knot_temperatures_C=best_theta,
    )
    case = DiagnosticCase(
        case_name="early_drop_h360_iter16_best_hold",
        profile_label="saved early-drop best profile held after 360 s",
        profile_source=str(summary_path.resolve()),
        T_ref_profile_C=profile,
    )
    return case, _cryostage_params_from_summary(summary)


def _delta(value: float, reference: float) -> float:
    return math.nan if not math.isfinite(value) else float(value - reference)


def _summarize_case(*, front_csv_path: Path, H_fill_m: float) -> dict[str, float]:
    cols = _load_front_columns(front_csv_path)
    centerline_top_arrival_s = _first_time_at_or_above(cols["time_since_fill_s"], cols["z_front_m"], H_fill_m - 1e-12)
    wall_top_arrival_s = _first_time_at_or_above(cols["time_since_fill_s"], cols["z_front_wall_m"], H_fill_m - 1e-12)
    full_freezing_s = _first_freeze_complete_time(cols["time_since_fill_s"], cols["freeze_complete_flag"])

    finite_tmax = cols["Tmax_fillable_C"][np.isfinite(cols["Tmax_fillable_C"])]
    last_Tmax_fillable_C = float(finite_tmax[-1]) if finite_tmax.size else math.nan

    return {
        "centerline_top_arrival_s": centerline_top_arrival_s,
        "wall_top_arrival_s": wall_top_arrival_s,
        "full_freezing_s": full_freezing_s,
        "centerline_to_full_gap_s": math.nan
        if not (math.isfinite(centerline_top_arrival_s) and math.isfinite(full_freezing_s))
        else float(full_freezing_s - centerline_top_arrival_s),
        "wall_to_full_gap_s": math.nan
        if not (math.isfinite(wall_top_arrival_s) and math.isfinite(full_freezing_s))
        else float(full_freezing_s - wall_top_arrival_s),
        "full_minus_180_s": _delta(full_freezing_s, 180.0),
        "full_minus_240_s": _delta(full_freezing_s, 240.0),
        "full_minus_360_s": _delta(full_freezing_s, 360.0),
        "last_Tmax_fillable_C": last_Tmax_fillable_C,
        "freeze_complete_reached": 1.0 if math.isfinite(full_freezing_s) else 0.0,
    }


def _write_summary_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "case_name",
        "profile_label",
        "profile_source",
        "front_csv_path",
        "probes_csv_path",
        "centerline_top_arrival_s",
        "wall_top_arrival_s",
        "full_freezing_s",
        "centerline_to_full_gap_s",
        "wall_to_full_gap_s",
        "full_minus_180_s",
        "full_minus_240_s",
        "full_minus_360_s",
        "last_Tmax_fillable_C",
        "freeze_complete_reached",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_txt(
    *,
    out_path: Path,
    rows: list[dict[str, str]],
    run_dir: Path,
    freeze_threshold_C: float,
    phase: PhaseChangeParams,
    freeze_stop: FreezeStopOptions,
    max_after_fill_s: float,
    dt_s: float,
    write_every_s: float,
) -> None:
    with out_path.open("w") as f:
        f.write("full_freezing_diagnostics\n\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(
            "freeze_complete_criterion: freeze_complete_flag = 1 when "
            "Tmax_fillable_C <= Tf - dT_mushy/2 - extra_subcooling_C\n"
        )
        f.write(
            f"freeze_threshold_C: {freeze_threshold_C:.6f} "
            f"(Tf={phase.Tf:.6f}, dT_mushy={phase.dT_mushy:.6f}, "
            f"extra_subcooling_C={freeze_stop.extra_subcooling_C:.6f})\n"
        )
        f.write(f"time_grid: 0..{max_after_fill_s:.1f} s with dt={dt_s:.3f} s\n")
        f.write(f"write_every_s: {write_every_s:.1f}\n\n")

        for row in rows:
            f.write(f"case_name: {row['case_name']}\n")
            f.write(f"profile_label: {row['profile_label']}\n")
            f.write(f"profile_source: {row['profile_source']}\n")
            f.write(f"centerline_top_arrival_s: {row['centerline_top_arrival_s']}\n")
            f.write(f"wall_top_arrival_s: {row['wall_top_arrival_s']}\n")
            f.write(f"full_freezing_s: {row['full_freezing_s']}\n")
            f.write(f"centerline_to_full_gap_s: {row['centerline_to_full_gap_s']}\n")
            f.write(f"wall_to_full_gap_s: {row['wall_to_full_gap_s']}\n")
            f.write(f"full_minus_180_s: {row['full_minus_180_s']}\n")
            f.write(f"full_minus_240_s: {row['full_minus_240_s']}\n")
            f.write(f"full_minus_360_s: {row['full_minus_360_s']}\n")
            f.write(f"last_Tmax_fillable_C: {row['last_Tmax_fillable_C']}\n")
            f.write(f"freeze_complete_reached: {row['freeze_complete_reached']}\n")
            f.write(f"front_csv_path: {row['front_csv_path']}\n")
            f.write(f"probes_csv_path: {row['probes_csv_path']}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight full-freezing diagnostics using the solver's existing freeze-complete event."
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Output folder name inside results/full_freezing_diagnostics.",
    )
    parser.add_argument(
        "--out-root-dir",
        default=str(DEFAULT_OUT_ROOT_DIR),
        help="Root folder where diagnostic outputs are written.",
    )
    parser.add_argument(
        "--max-after-fill-s",
        type=float,
        default=3600.0,
        help="Maximum post-fill simulation time. The solver still stops early at freeze completion.",
    )
    parser.add_argument(
        "--write-every-s",
        type=float,
        default=None,
        help="Field-write interval passed to the solver. Use a very large value for lean outputs.",
    )
    parser.add_argument(
        "--simulation-config",
        default=str(DEFAULT_SIMULATION_CONFIG_PATH),
        help="TOML file with named simulation profiles.",
    )
    parser.add_argument(
        "--simulation-profile",
        default=None,
        help="Simulation profile name from --simulation-config. Defaults to the TOML default_profile.",
    )
    parser.add_argument("--dry-run-config", action="store_true", help="Print effective settings and exit before running diagnostics.")
    parser.add_argument("--solver-dt-s", type=float, default=None, help="Override the selected simulation profile solver time step.")
    parser.add_argument("--cryostage-dt-s", type=float, default=None, help="Override the selected simulation profile cryostage sampling time step.")
    parser.add_argument("--nr", type=int, default=None, help="Override the selected simulation profile radial mesh nodes.")
    parser.add_argument("--nz", type=int, default=None, help="Override the selected simulation profile axial mesh nodes.")
    parser.add_argument("--nz-front", type=int, default=None, help="Override the selected simulation profile centerline front sampling nodes.")
    parser.add_argument(
        "--ambient-mode",
        default=None,
        choices=("fixed", "interpolate_from_cryostage"),
        help="Override the selected simulation profile ambient-temperature model.",
    )
    parser.add_argument(
        "--early-drop-study-summary",
        default=str(DEFAULT_EARLY_DROP_STUDY_SUMMARY),
        help="Path to the saved early-drop study_summary.txt used for the representative open-loop profile.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing run folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    simulation_profiles = load_simulation_profiles(args.simulation_config)
    simulation_profile_name = args.simulation_profile or simulation_profiles.default_profile
    simulation_profile = simulation_profiles.get_profile(simulation_profile_name)
    simulation_profile = override_simulation_profile(
        simulation_profile,
        cryostage_dt_s=args.cryostage_dt_s,
        solver_dt_s=args.solver_dt_s,
        Nr=args.nr,
        Nz=args.nz,
        Nz_front=args.nz_front,
        ambient_mode=args.ambient_mode,
    )
    config = build_problem_config(
        num_knots=LEGACY_COMPATIBILITY_TRAJECTORY_PARAMETER_COUNT,
    )
    config = apply_simulation_profile(config, simulation_profile)
    solver_kwargs = config.cascade_run_kwargs()
    solver_kwargs["t_after_fill_s"] = float(args.max_after_fill_s)
    if args.write_every_s is not None:
        solver_kwargs["write_every"] = float(args.write_every_s)
    solver_kwargs["enable_front_curve"] = False

    if args.dry_run_config:
        print(
            json.dumps(
                {
                    "run_name": str(args.run_name),
                    "simulation_config": str(Path(args.simulation_config)),
                    "simulation_profile": simulation_profile_name,
                    "max_after_fill_s": float(args.max_after_fill_s),
                    "cryostage_dt_s": float(config.cryostage_dt_s),
                    "solver": {
                        "Nr": int(solver_kwargs["Nr"]),
                        "Nz": int(solver_kwargs["Nz"]),
                        "dt": float(solver_kwargs["dt"]),
                        "Nz_front": int(solver_kwargs["Nz_front"]),
                        "write_every": float(solver_kwargs["write_every"]),
                        "write_field_output": bool(solver_kwargs.get("write_field_output", True)),
                        "write_probe_csv": bool(solver_kwargs.get("write_probe_csv", True)),
                        "enable_front_curve": bool(solver_kwargs.get("enable_front_curve", False)),
                        "ambient": describe_ambient_temperature_model(
                            solver_kwargs.get("ambient_temperature_from_plate_C")
                        ),
                    },
                },
                indent=2,
            )
        )
        return

    out_root_dir = Path(args.out_root_dir).resolve()
    run_dir = out_root_dir / args.run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{run_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    geom = solver_kwargs["geom"]
    phase = solver_kwargs.get("phase", PhaseChangeParams())
    freeze_stop = solver_kwargs.get("freeze_stop", FreezeStopOptions())
    freeze_threshold_C = phase.Tf - 0.5 * phase.dT_mushy - freeze_stop.extra_subcooling_C
    time_s = _build_time_grid(float(args.max_after_fill_s), config.cryostage_dt_s)

    early_drop_case, early_drop_params = _load_early_drop_case(Path(args.early_drop_study_summary).resolve())
    cases = [
        (
            DiagnosticCase(
                "const_ref_m10",
                "constant reference -10 C",
                "ConstantTemperatureProfile(-10.0)",
                ConstantTemperatureProfile(-10.0),
            ),
            DEFAULT_CRYOSTAGE_PARAMS,
        ),
        (
            DiagnosticCase(
                "const_ref_m15",
                "constant reference -15 C",
                "ConstantTemperatureProfile(-15.0)",
                ConstantTemperatureProfile(-15.0),
            ),
            DEFAULT_CRYOSTAGE_PARAMS,
        ),
        (
            DiagnosticCase(
                "const_ref_m20",
                "constant reference -20 C",
                "ConstantTemperatureProfile(-20.0)",
                ConstantTemperatureProfile(-20.0),
            ),
            DEFAULT_CRYOSTAGE_PARAMS,
        ),
        (early_drop_case, early_drop_params),
    ]

    summary_rows: list[dict[str, str]] = []
    for case, cryostage_params in cases:
        case_dir = run_dir / case.case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        result = run_open_loop_case(
            time_s=time_s,
            T_ref_profile_C=case.T_ref_profile_C,
            cryostage_params=cryostage_params,
            out_dir=case_dir,
            prefix=case.case_name,
            **solver_kwargs,
        )
        metrics = _summarize_case(front_csv_path=result.front_path, H_fill_m=geom.H_fill)

        row = {
            "case_name": case.case_name,
            "profile_label": case.profile_label,
            "profile_source": case.profile_source,
            "front_csv_path": str(result.front_path.resolve()),
            "probes_csv_path": str(result.probes_path.resolve()),
            "centerline_top_arrival_s": _nan_to_str(metrics["centerline_top_arrival_s"]),
            "wall_top_arrival_s": _nan_to_str(metrics["wall_top_arrival_s"]),
            "full_freezing_s": _nan_to_str(metrics["full_freezing_s"]),
            "centerline_to_full_gap_s": _nan_to_str(metrics["centerline_to_full_gap_s"]),
            "wall_to_full_gap_s": _nan_to_str(metrics["wall_to_full_gap_s"]),
            "full_minus_180_s": _nan_to_str(metrics["full_minus_180_s"]),
            "full_minus_240_s": _nan_to_str(metrics["full_minus_240_s"]),
            "full_minus_360_s": _nan_to_str(metrics["full_minus_360_s"]),
            "last_Tmax_fillable_C": _nan_to_str(metrics["last_Tmax_fillable_C"]),
            "freeze_complete_reached": str(int(metrics["freeze_complete_reached"])),
        }
        summary_rows.append(row)

        print(
            f"{case.case_name}: full_freezing_s={row['full_freezing_s']}  "
            f"centerline_gap_s={row['centerline_to_full_gap_s']}  "
            f"wall_gap_s={row['wall_to_full_gap_s']}"
        )

    csv_path = run_dir / "full_freezing_summary.csv"
    txt_path = run_dir / "full_freezing_summary.txt"
    _write_summary_csv(summary_rows, csv_path)
    _write_summary_txt(
        out_path=txt_path,
        rows=summary_rows,
        run_dir=run_dir,
        freeze_threshold_C=freeze_threshold_C,
        phase=phase,
        freeze_stop=freeze_stop,
        max_after_fill_s=float(args.max_after_fill_s),
        dt_s=float(config.cryostage_dt_s),
        write_every_s=float(args.write_every_s),
    )

    print(f"Summary CSV: {csv_path}")
    print(f"Summary TXT: {txt_path}")


if __name__ == "__main__":
    main()
