from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cryostage_model import CryostageModelParams
from open_loop_problem import (
    OpenLoopProblemConfig,
    build_reference_profile_from_theta,
    evaluate_open_loop_objective,
)


def _coerce_theta_tuple(values, *, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f'{name} must be a 1D array')
    if arr.size == 0:
        raise ValueError(f'{name} must contain at least one value')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must contain only finite values')
    return tuple(float(value) for value in arr)


@dataclass(frozen=True)
class OpenLoopOptimizationHistoryEntry:
    evaluation_index: int
    case_name: str
    theta: tuple[float, ...]
    objective_value: float
    is_valid: bool
    out_dir: Path
    error_message: str = ''


@dataclass(frozen=True)
class OpenLoopOptimizationResult:
    best_theta: tuple[float, ...]
    best_objective_value: float
    best_evaluation_index: int
    best_case_name: str
    success: bool
    status: int
    message: str
    method: str
    nfev: int
    nit: int | None
    run_dir: Path
    best_dir: Path
    history_csv_path: Path
    history: tuple[OpenLoopOptimizationHistoryEntry, ...]


_HISTORY_FIELDNAMES = [
    'evaluation_index',
    'case_name',
    'objective_value',
    'is_valid',
    'out_dir',
    'error_message',
    'theta_json',
]


def _initialize_history_csv(path: Path) -> None:
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDNAMES)
        writer.writeheader()


def _append_history_csv(path: Path, entry: OpenLoopOptimizationHistoryEntry) -> None:
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDNAMES)
        writer.writerow(
            {
                'evaluation_index': entry.evaluation_index,
                'case_name': entry.case_name,
                'objective_value': entry.objective_value,
                'is_valid': int(entry.is_valid),
                'out_dir': str(entry.out_dir),
                'error_message': entry.error_message,
                'theta_json': json.dumps(list(entry.theta)),
            }
        )


def optimize_open_loop_theta(
    theta0,
    config: OpenLoopProblemConfig,
    cryostage_params: CryostageModelParams,
    out_root_dir: str | Path,
    run_name: str,
    *,
    method: str = 'Nelder-Mead',
    options: dict[str, Any] | None = None,
    T_plate0_C: float | None = None,
) -> OpenLoopOptimizationResult:
    theta0 = _coerce_theta_tuple(theta0, name='theta0')
    build_reference_profile_from_theta(theta0, config)

    run_dir = Path(out_root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    evaluations_dir = run_dir / 'evaluations'
    evaluations_dir.mkdir()

    history_csv_path = run_dir / 'evaluation_history.csv'
    _initialize_history_csv(history_csv_path)

    history: list[OpenLoopOptimizationHistoryEntry] = []
    best_entry: OpenLoopOptimizationHistoryEntry | None = None
    best_objective_value = math.inf
    evaluation_index = 0

    def wrapped_objective(theta) -> float:
        nonlocal evaluation_index, best_entry, best_objective_value

        theta_tuple = _coerce_theta_tuple(theta, name='theta')
        evaluation_index += 1
        eval_name = f'eval_{evaluation_index:04d}'
        case_name = f'{run_name}_{eval_name}'
        out_dir = evaluations_dir / eval_name

        error_message = ''
        is_valid = False
        objective_value = math.inf

        try:
            objective_result = evaluate_open_loop_objective(
                theta_tuple,
                config,
                cryostage_params,
                out_dir,
                case_name,
                T_plate0_C=T_plate0_C,
            )
            objective_value = float(objective_result.objective_value)
            if math.isfinite(objective_value):
                is_valid = True
            else:
                error_message = f'Objective returned non-finite value: {objective_value}'
        except Exception as exc:
            error_message = f'{type(exc).__name__}: {exc}'

        entry = OpenLoopOptimizationHistoryEntry(
            evaluation_index=evaluation_index,
            case_name=case_name,
            theta=theta_tuple,
            objective_value=float(objective_value),
            is_valid=is_valid,
            out_dir=out_dir,
            error_message=error_message,
        )
        history.append(entry)
        _append_history_csv(history_csv_path, entry)

        if is_valid and objective_value < best_objective_value:
            best_objective_value = float(objective_value)
            best_entry = entry

        return float(objective_value)

    scipy_result = minimize(
        wrapped_objective,
        np.asarray(theta0, dtype=np.float64),
        method=method,
        options=dict(options or {}),
    )

    if best_entry is None:
        raise RuntimeError('Optimization completed without any valid objective evaluation')

    build_reference_profile_from_theta(best_entry.theta, config)
    if not math.isfinite(best_entry.objective_value):
        raise RuntimeError('Best optimization entry has a non-finite objective value')

    success = bool(scipy_result.success)
    status = int(scipy_result.status)
    message = str(scipy_result.message)
    optimizer_theta = _coerce_theta_tuple(getattr(scipy_result, 'x', theta0), name='optimizer_theta')
    try:
        build_reference_profile_from_theta(optimizer_theta, config)
    except Exception as exc:
        success = False
        message = (
            f'{message} | optimizer final theta invalid under active constraints: '
            f'{type(exc).__name__}: {exc}; returning best valid evaluation instead'
        )

    best_dir = run_dir / 'best'
    shutil.copytree(best_entry.out_dir, best_dir)

    return OpenLoopOptimizationResult(
        best_theta=best_entry.theta,
        best_objective_value=best_entry.objective_value,
        best_evaluation_index=best_entry.evaluation_index,
        best_case_name=best_entry.case_name,
        success=success,
        status=status,
        message=message,
        method=str(method),
        nfev=int(scipy_result.nfev),
        nit=(int(scipy_result.nit) if getattr(scipy_result, 'nit', None) is not None else None),
        run_dir=run_dir,
        best_dir=best_dir,
        history_csv_path=history_csv_path,
        history=tuple(history),
    )


__all__ = [
    'OpenLoopOptimizationHistoryEntry',
    'OpenLoopOptimizationResult',
    'optimize_open_loop_theta',
]
