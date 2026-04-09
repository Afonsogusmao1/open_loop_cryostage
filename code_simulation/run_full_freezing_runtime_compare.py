#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_full_freezing_diagnostics import (
    DEFAULT_EARLY_DROP_STUDY_SUMMARY,
    _build_time_grid,
    _load_early_drop_case,
)
from open_loop_workflow_config import build_problem_config
from solver import PhaseChangeParams, FreezeStopOptions, run_case


DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / 'results' / 'full_freezing_runtime_compare'
DEFAULT_RUN_NAME = 'current_snapshot'
DEFAULT_CAP_S = 2400.0
DEFAULT_CONSTANT_PLATE_C = -10.0


@dataclass(frozen=True)
class RuntimeSetting:
    setting_name: str
    description: str
    dt_s: float
    write_every_s: float
    write_field_output: bool
    write_probe_csv: bool
    show_progress: bool


@dataclass(frozen=True)
class RuntimeCase:
    case_name: str
    case_label: str
    input_mode: str
    profile_source: str
    constant_plate_C: float | None = None
    T_ref_profile_C: object | None = None
    cryostage_params: object | None = None


def _load_front_columns(front_csv_path: Path) -> dict[str, np.ndarray]:
    columns = {
        'time_s': [],
        'time_since_fill_s': [],
        'z_front_m': [],
        'z_front_wall_m': [],
        'Tmax_fillable_C': [],
        'freeze_complete_flag': [],
    }
    with front_csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in columns:
                raw = row.get(key, '')
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = math.nan
                columns[key].append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def _first_time_at_or_above(time_since_fill_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    mask = np.isfinite(time_since_fill_s) & np.isfinite(values) & (values >= threshold)
    if not np.any(mask):
        return math.nan
    return float(time_since_fill_s[np.flatnonzero(mask)[0]])


def _first_freeze_complete_time(time_since_fill_s: np.ndarray, freeze_complete_flag: np.ndarray) -> float:
    mask = np.isfinite(time_since_fill_s) & np.isfinite(freeze_complete_flag) & (freeze_complete_flag >= 0.5)
    if not np.any(mask):
        return math.nan
    return float(time_since_fill_s[np.flatnonzero(mask)[0]])


def _nan_to_str(value: float) -> str:
    return 'nan' if not math.isfinite(value) else f'{value:.6f}'


def _summarize_case(*, front_csv_path: Path, H_fill_m: float) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    cols = _load_front_columns(front_csv_path)
    centerline_top_arrival_s = _first_time_at_or_above(cols['time_since_fill_s'], cols['z_front_m'], H_fill_m - 1e-12)
    wall_top_arrival_s = _first_time_at_or_above(cols['time_since_fill_s'], cols['z_front_wall_m'], H_fill_m - 1e-12)
    full_freezing_s = _first_freeze_complete_time(cols['time_since_fill_s'], cols['freeze_complete_flag'])
    finite_tmax = cols['Tmax_fillable_C'][np.isfinite(cols['Tmax_fillable_C'])]
    last_Tmax_fillable_C = float(finite_tmax[-1]) if finite_tmax.size else math.nan
    metrics = {
        'centerline_top_arrival_s': centerline_top_arrival_s,
        'wall_top_arrival_s': wall_top_arrival_s,
        'full_freezing_s': full_freezing_s,
        'centerline_to_full_gap_s': math.nan
        if not (math.isfinite(centerline_top_arrival_s) and math.isfinite(full_freezing_s))
        else float(full_freezing_s - centerline_top_arrival_s),
        'wall_to_full_gap_s': math.nan
        if not (math.isfinite(wall_top_arrival_s) and math.isfinite(full_freezing_s))
        else float(full_freezing_s - wall_top_arrival_s),
        'last_Tmax_fillable_C': last_Tmax_fillable_C,
        'freeze_complete_reached': 1.0 if math.isfinite(full_freezing_s) else 0.0,
    }
    return metrics, cols


def _front_history_error_metrics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, float]:
    ref_full_s = _first_freeze_complete_time(reference['time_since_fill_s'], reference['freeze_complete_flag'])
    cand_full_s = _first_freeze_complete_time(candidate['time_since_fill_s'], candidate['freeze_complete_flag'])

    compare_end_s = math.nan
    if math.isfinite(ref_full_s) and math.isfinite(cand_full_s):
        compare_end_s = float(min(ref_full_s, cand_full_s))
    else:
        ref_t = reference['time_since_fill_s'][np.isfinite(reference['time_since_fill_s'])]
        cand_t = candidate['time_since_fill_s'][np.isfinite(candidate['time_since_fill_s'])]
        if ref_t.size and cand_t.size:
            compare_end_s = float(min(ref_t[-1], cand_t[-1]))

    if not math.isfinite(compare_end_s):
        return {
            'compare_window_end_s': math.nan,
            'front_centerline_rmse_mm_vs_default': math.nan,
            'front_centerline_maxabs_mm_vs_default': math.nan,
            'front_wall_rmse_mm_vs_default': math.nan,
            'front_wall_maxabs_mm_vs_default': math.nan,
        }

    ref_mask = (
        np.isfinite(reference['time_since_fill_s'])
        & np.isfinite(reference['z_front_m'])
        & np.isfinite(reference['z_front_wall_m'])
        & (reference['time_since_fill_s'] >= 0.0)
        & (reference['time_since_fill_s'] <= compare_end_s + 1e-12)
    )
    cand_mask = (
        np.isfinite(candidate['time_since_fill_s'])
        & np.isfinite(candidate['z_front_m'])
        & np.isfinite(candidate['z_front_wall_m'])
        & (candidate['time_since_fill_s'] >= 0.0)
        & (candidate['time_since_fill_s'] <= compare_end_s + 1e-12)
    )

    ref_t = reference['time_since_fill_s'][ref_mask]
    ref_center = reference['z_front_m'][ref_mask]
    ref_wall = reference['z_front_wall_m'][ref_mask]
    cand_t = candidate['time_since_fill_s'][cand_mask]
    cand_center = candidate['z_front_m'][cand_mask]
    cand_wall = candidate['z_front_wall_m'][cand_mask]

    if ref_t.size < 2 or cand_t.size < 2:
        return {
            'compare_window_end_s': compare_end_s,
            'front_centerline_rmse_mm_vs_default': math.nan,
            'front_centerline_maxabs_mm_vs_default': math.nan,
            'front_wall_rmse_mm_vs_default': math.nan,
            'front_wall_maxabs_mm_vs_default': math.nan,
        }

    common_mask = (ref_t >= cand_t[0] - 1e-12) & (ref_t <= cand_t[-1] + 1e-12)
    ref_t = ref_t[common_mask]
    ref_center = ref_center[common_mask]
    ref_wall = ref_wall[common_mask]

    if ref_t.size < 2:
        return {
            'compare_window_end_s': compare_end_s,
            'front_centerline_rmse_mm_vs_default': math.nan,
            'front_centerline_maxabs_mm_vs_default': math.nan,
            'front_wall_rmse_mm_vs_default': math.nan,
            'front_wall_maxabs_mm_vs_default': math.nan,
        }

    cand_center_interp = np.interp(ref_t, cand_t, cand_center)
    cand_wall_interp = np.interp(ref_t, cand_t, cand_wall)
    center_err_mm = 1000.0 * (cand_center_interp - ref_center)
    wall_err_mm = 1000.0 * (cand_wall_interp - ref_wall)

    return {
        'compare_window_end_s': float(ref_t[-1]),
        'front_centerline_rmse_mm_vs_default': float(np.sqrt(np.mean(center_err_mm**2))),
        'front_centerline_maxabs_mm_vs_default': float(np.max(np.abs(center_err_mm))),
        'front_wall_rmse_mm_vs_default': float(np.sqrt(np.mean(wall_err_mm**2))),
        'front_wall_maxabs_mm_vs_default': float(np.max(np.abs(wall_err_mm))),
    }


def _build_settings() -> tuple[RuntimeSetting, ...]:
    return (
        RuntimeSetting(
            setting_name='default_dt2',
            description='current workflow defaults',
            dt_s=2.0,
            write_every_s=30.0,
            write_field_output=True,
            write_probe_csv=True,
            show_progress=True,
        ),
        RuntimeSetting(
            setting_name='lean_dt4',
            description='same mesh, lean outputs, dt=4 s',
            dt_s=4.0,
            write_every_s=1.0e9,
            write_field_output=False,
            write_probe_csv=False,
            show_progress=False,
        ),
        RuntimeSetting(
            setting_name='lean_dt6',
            description='same mesh, lean outputs, dt=6 s',
            write_every_s=1.0e9,
            dt_s=6.0,
            write_field_output=False,
            write_probe_csv=False,
            show_progress=False,
        ),
    )


def _build_cases(early_drop_summary_path: Path, constant_plate_C: float) -> tuple[RuntimeCase, ...]:
    early_drop_case, early_drop_params = _load_early_drop_case(early_drop_summary_path)
    return (
        RuntimeCase(
            case_name='const_plate_m10',
            case_label=f'constant plate {constant_plate_C:.0f} C',
            input_mode='constant_plate',
            profile_source=f'run_case(T_plate_C={constant_plate_C:.1f})',
            constant_plate_C=constant_plate_C,
        ),
        RuntimeCase(
            case_name=early_drop_case.case_name,
            case_label=early_drop_case.profile_label,
            input_mode='open_loop_reference',
            profile_source=early_drop_case.profile_source,
            T_ref_profile_C=early_drop_case.T_ref_profile_C,
            cryostage_params=early_drop_params,
        ),
    )


def _solver_kwargs_for_setting(base_solver_kwargs: dict, setting: RuntimeSetting, cap_s: float) -> dict:
    kwargs = dict(base_solver_kwargs)
    kwargs['Nr'] = int(base_solver_kwargs['Nr'])
    kwargs['Nz'] = int(base_solver_kwargs['Nz'])
    kwargs['dt'] = float(setting.dt_s)
    kwargs['t_after_fill_s'] = float(cap_s)
    kwargs['write_every'] = float(setting.write_every_s)
    kwargs['write_field_output'] = bool(setting.write_field_output)
    kwargs['write_probe_csv'] = bool(setting.write_probe_csv)
    kwargs['show_progress'] = bool(setting.show_progress)
    kwargs['enable_front_curve'] = False
    return kwargs


def _run_case_for_setting(*, case: RuntimeCase, case_dir: Path, prefix: str, solver_kwargs: dict, dt_s: float, cap_s: float):
    if case.input_mode == 'constant_plate':
        run_case(
            out_dir=case_dir,
            prefix=prefix,
            T_plate_C=float(case.constant_plate_C),
            **solver_kwargs,
        )
        return {
            'front_path': case_dir / f'{prefix}_front.csv',
            'probes_path': case_dir / f'{prefix}_probes.csv',
            'xdmf_path': case_dir / f'{prefix}.xdmf',
        }

    if case.input_mode == 'open_loop_reference':
        from open_loop_cascade import run_open_loop_case

        time_s = _build_time_grid(float(cap_s), float(dt_s))
        result = run_open_loop_case(
            time_s=time_s,
            T_ref_profile_C=case.T_ref_profile_C,
            cryostage_params=case.cryostage_params,
            out_dir=case_dir,
            prefix=prefix,
            **solver_kwargs,
        )
        return {
            'front_path': result.front_path,
            'probes_path': result.probes_path,
            'xdmf_path': result.xdmf_path,
        }

    raise ValueError(f'Unknown case.input_mode={case.input_mode!r}')


def _write_summary_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        'case_name',
        'case_label',
        'input_mode',
        'setting_name',
        'description',
        'dt_s',
        'Nr',
        'Nz',
        'write_every_s',
        'write_field_output',
        'write_probe_csv',
        'show_progress',
        'runtime_wall_s',
        'runtime_speedup_vs_default',
        'centerline_top_arrival_s',
        'wall_top_arrival_s',
        'full_freezing_s',
        'centerline_to_full_gap_s',
        'wall_to_full_gap_s',
        'full_freezing_delta_s_vs_default',
        'centerline_top_delta_s_vs_default',
        'wall_top_delta_s_vs_default',
        'compare_window_end_s',
        'front_centerline_rmse_mm_vs_default',
        'front_centerline_maxabs_mm_vs_default',
        'front_wall_rmse_mm_vs_default',
        'front_wall_maxabs_mm_vs_default',
        'freeze_complete_reached',
        'last_Tmax_fillable_C',
        'front_csv_path',
        'probes_csv_path',
        'xdmf_path',
        'profile_source',
    ]
    with out_path.open('w', newline='') as f:
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
    cap_s: float,
    base_solver_kwargs: dict,
) -> None:
    with out_path.open('w') as f:
        f.write('full_freezing_runtime_compare\n\n')
        f.write(f'run_dir: {run_dir}\n')
        f.write(
            'freeze_complete_criterion: freeze_complete_flag = 1 when '
            'Tmax_fillable_C <= Tf - dT_mushy/2 - extra_subcooling_C\n'
        )
        f.write(
            f'freeze_threshold_C: {freeze_threshold_C:.6f} '
            f'(Tf={phase.Tf:.6f}, dT_mushy={phase.dT_mushy:.6f}, '
            f'extra_subcooling_C={freeze_stop.extra_subcooling_C:.6f})\n'
        )
        f.write(
            'reference_mesh: '
            f"Nr={int(base_solver_kwargs['Nr'])}, Nz={int(base_solver_kwargs['Nz'])}, "
            f"use_tabulated_water_ice={bool(base_solver_kwargs['use_tabulated_water_ice'])}\n"
        )
        f.write(f'provisional_safety_cap_s: {cap_s:.1f}\n\n')

        for row in rows:
            f.write(f"case_name: {row['case_name']}\n")
            f.write(f"case_label: {row['case_label']}\n")
            f.write(f"setting_name: {row['setting_name']}\n")
            f.write(f"description: {row['description']}\n")
            f.write(f"runtime_wall_s: {row['runtime_wall_s']}\n")
            f.write(f"runtime_speedup_vs_default: {row['runtime_speedup_vs_default']}\n")
            f.write(f"centerline_top_arrival_s: {row['centerline_top_arrival_s']}\n")
            f.write(f"wall_top_arrival_s: {row['wall_top_arrival_s']}\n")
            f.write(f"full_freezing_s: {row['full_freezing_s']}\n")
            f.write(f"centerline_to_full_gap_s: {row['centerline_to_full_gap_s']}\n")
            f.write(f"wall_to_full_gap_s: {row['wall_to_full_gap_s']}\n")
            f.write(f"full_freezing_delta_s_vs_default: {row['full_freezing_delta_s_vs_default']}\n")
            f.write(f"centerline_top_delta_s_vs_default: {row['centerline_top_delta_s_vs_default']}\n")
            f.write(f"wall_top_delta_s_vs_default: {row['wall_top_delta_s_vs_default']}\n")
            f.write(f"compare_window_end_s: {row['compare_window_end_s']}\n")
            f.write(f"front_centerline_rmse_mm_vs_default: {row['front_centerline_rmse_mm_vs_default']}\n")
            f.write(f"front_centerline_maxabs_mm_vs_default: {row['front_centerline_maxabs_mm_vs_default']}\n")
            f.write(f"front_wall_rmse_mm_vs_default: {row['front_wall_rmse_mm_vs_default']}\n")
            f.write(f"front_wall_maxabs_mm_vs_default: {row['front_wall_maxabs_mm_vs_default']}\n")
            f.write(f"freeze_complete_reached: {row['freeze_complete_reached']}\n")
            f.write(f"last_Tmax_fillable_C: {row['last_Tmax_fillable_C']}\n")
            f.write(f"front_csv_path: {row['front_csv_path']}\n")
            f.write(f"probes_csv_path: {row['probes_csv_path']}\n")
            f.write(f"xdmf_path: {row['xdmf_path']}\n")
            f.write(f"profile_source: {row['profile_source']}\n\n")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare current default full-freezing settings against lean larger-dt candidates.'
    )
    parser.add_argument('--run-name', default=DEFAULT_RUN_NAME, help='Output folder name inside results/full_freezing_runtime_compare.')
    parser.add_argument('--out-root-dir', default=str(DEFAULT_OUT_ROOT_DIR), help='Root folder where outputs are written.')
    parser.add_argument('--cap-s', type=float, default=DEFAULT_CAP_S, help='Post-fill safety cap. Runs still stop early at freeze completion.')
    parser.add_argument('--constant-plate-c', type=float, default=DEFAULT_CONSTANT_PLATE_C, help='Constant plate temperature for the direct solver baseline case.')
    parser.add_argument('--early-drop-study-summary', default=str(DEFAULT_EARLY_DROP_STUDY_SUMMARY), help='Path to the saved early-drop study summary used for the representative open-loop profile.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing run folder.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root_dir = Path(args.out_root_dir).resolve()
    run_dir = out_root_dir / args.run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{run_dir} already exists. Re-run with --overwrite to replace it.')
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_problem_config()
    base_solver_kwargs = config.cascade_run_kwargs()
    geom = base_solver_kwargs['geom']
    phase = base_solver_kwargs.get('phase', PhaseChangeParams())
    freeze_stop = base_solver_kwargs.get('freeze_stop', FreezeStopOptions())
    freeze_threshold_C = phase.Tf - 0.5 * phase.dT_mushy - freeze_stop.extra_subcooling_C

    settings = _build_settings()
    cases = _build_cases(Path(args.early_drop_study_summary).resolve(), float(args.constant_plate_c))

    records: dict[tuple[str, str], dict] = {}
    for setting in settings:
        solver_kwargs = _solver_kwargs_for_setting(base_solver_kwargs, setting, float(args.cap_s))
        for case in cases:
            case_dir = run_dir / setting.setting_name / case.case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            prefix = case.case_name

            t0 = time.perf_counter()
            paths = _run_case_for_setting(
                case=case,
                case_dir=case_dir,
                prefix=prefix,
                solver_kwargs=solver_kwargs,
                dt_s=setting.dt_s,
                cap_s=float(args.cap_s),
            )
            runtime_wall_s = time.perf_counter() - t0

            metrics, front_cols = _summarize_case(front_csv_path=Path(paths['front_path']), H_fill_m=geom.H_fill)
            records[(case.case_name, setting.setting_name)] = {
                'case': case,
                'setting': setting,
                'solver_kwargs': solver_kwargs,
                'runtime_wall_s': runtime_wall_s,
                'metrics': metrics,
                'front_cols': front_cols,
                'paths': paths,
            }
            print(
                f"{setting.setting_name} | {case.case_name}: runtime={runtime_wall_s:.2f}s  "
                f"full_freezing_s={_nan_to_str(metrics['full_freezing_s'])}"
            )

    summary_rows: list[dict[str, str]] = []
    for case in cases:
        default_record = records[(case.case_name, 'default_dt2')]
        default_runtime_s = float(default_record['runtime_wall_s'])
        default_metrics = default_record['metrics']
        default_front_cols = default_record['front_cols']

        for setting in settings:
            record = records[(case.case_name, setting.setting_name)]
            metrics = record['metrics']
            front_error = _front_history_error_metrics(default_front_cols, record['front_cols'])
            runtime_speedup = math.nan
            if record['runtime_wall_s'] > 0.0:
                runtime_speedup = default_runtime_s / float(record['runtime_wall_s'])

            row = {
                'case_name': case.case_name,
                'case_label': case.case_label,
                'input_mode': case.input_mode,
                'setting_name': setting.setting_name,
                'description': setting.description,
                'dt_s': f'{setting.dt_s:.6f}',
                'Nr': str(int(record['solver_kwargs']['Nr'])),
                'Nz': str(int(record['solver_kwargs']['Nz'])),
                'write_every_s': f"{float(record['solver_kwargs']['write_every']):.6f}",
                'write_field_output': str(int(bool(record['solver_kwargs']['write_field_output']))),
                'write_probe_csv': str(int(bool(record['solver_kwargs']['write_probe_csv']))),
                'show_progress': str(int(bool(record['solver_kwargs']['show_progress']))),
                'runtime_wall_s': f"{float(record['runtime_wall_s']):.6f}",
                'runtime_speedup_vs_default': _nan_to_str(runtime_speedup),
                'centerline_top_arrival_s': _nan_to_str(metrics['centerline_top_arrival_s']),
                'wall_top_arrival_s': _nan_to_str(metrics['wall_top_arrival_s']),
                'full_freezing_s': _nan_to_str(metrics['full_freezing_s']),
                'centerline_to_full_gap_s': _nan_to_str(metrics['centerline_to_full_gap_s']),
                'wall_to_full_gap_s': _nan_to_str(metrics['wall_to_full_gap_s']),
                'full_freezing_delta_s_vs_default': _nan_to_str(metrics['full_freezing_s'] - default_metrics['full_freezing_s']),
                'centerline_top_delta_s_vs_default': _nan_to_str(metrics['centerline_top_arrival_s'] - default_metrics['centerline_top_arrival_s']),
                'wall_top_delta_s_vs_default': _nan_to_str(metrics['wall_top_arrival_s'] - default_metrics['wall_top_arrival_s']),
                'compare_window_end_s': _nan_to_str(front_error['compare_window_end_s']),
                'front_centerline_rmse_mm_vs_default': _nan_to_str(front_error['front_centerline_rmse_mm_vs_default']),
                'front_centerline_maxabs_mm_vs_default': _nan_to_str(front_error['front_centerline_maxabs_mm_vs_default']),
                'front_wall_rmse_mm_vs_default': _nan_to_str(front_error['front_wall_rmse_mm_vs_default']),
                'front_wall_maxabs_mm_vs_default': _nan_to_str(front_error['front_wall_maxabs_mm_vs_default']),
                'freeze_complete_reached': str(int(metrics['freeze_complete_reached'])),
                'last_Tmax_fillable_C': _nan_to_str(metrics['last_Tmax_fillable_C']),
                'front_csv_path': str(Path(record['paths']['front_path']).resolve()),
                'probes_csv_path': str(Path(record['paths']['probes_path']).resolve()) if Path(record['paths']['probes_path']).exists() else '',
                'xdmf_path': str(Path(record['paths']['xdmf_path']).resolve()) if Path(record['paths']['xdmf_path']).exists() else '',
                'profile_source': case.profile_source,
            }
            summary_rows.append(row)

    csv_path = run_dir / 'runtime_settings_summary.csv'
    txt_path = run_dir / 'runtime_settings_summary.txt'
    _write_summary_csv(summary_rows, csv_path)
    _write_summary_txt(
        out_path=txt_path,
        rows=summary_rows,
        run_dir=run_dir,
        freeze_threshold_C=freeze_threshold_C,
        phase=phase,
        freeze_stop=freeze_stop,
        cap_s=float(args.cap_s),
        base_solver_kwargs=base_solver_kwargs,
    )

    print(f'Summary CSV: {csv_path}')
    print(f'Summary TXT: {txt_path}')


if __name__ == '__main__':
    main()
