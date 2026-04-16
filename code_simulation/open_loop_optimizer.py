from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cryostage_model import CryostageModelParams
from open_loop_bayesian_optimizer import (
    BayesianOptimizationBackendResult,
    BayesianOptimizationConfig,
    run_bayesian_optimization,
)
from open_loop_problem import (
    build_reference_profile_from_theta,
    evaluate_open_loop_objective,
)
from open_loop_workflow_config import OpenLoopProblemConfig


def _coerce_theta_tuple(values, *, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(value) for value in arr)


def _normalized_method(method: str) -> str:
    return str(method).strip().lower().replace("_", "-")


def _is_bayesian_method(method: str) -> bool:
    return _normalized_method(method) in {"bayesian-optimization", "bayesopt", "bo"}


def _optimizer_target_from_objective(objective_value: float, *, bayesian_backend: bool) -> float:
    objective_value = float(objective_value)
    return -objective_value if bayesian_backend else objective_value


@dataclass(frozen=True)
class OpenLoopOptimizationHistoryEntry:
    evaluation_index: int
    phase: str
    case_name: str
    raw_candidate: tuple[float, ...]
    theta: tuple[float, ...]
    objective_value: float
    optimizer_target_value: float
    is_valid: bool
    expensive_simulation_executed: bool
    incumbent_after_eval: bool
    best_objective_value_after_eval: float
    runtime_s: float
    out_dir: Path
    feasibility_status: str
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_index": int(self.evaluation_index),
            "phase": str(self.phase),
            "case_name": str(self.case_name),
            "raw_candidate": [float(value) for value in self.raw_candidate],
            "theta": [float(value) for value in self.theta],
            "objective_value": float(self.objective_value),
            "optimizer_target_value": float(self.optimizer_target_value),
            "is_valid": bool(self.is_valid),
            "expensive_simulation_executed": bool(self.expensive_simulation_executed),
            "incumbent_after_eval": bool(self.incumbent_after_eval),
            "best_objective_value_after_eval": float(self.best_objective_value_after_eval),
            "runtime_s": float(self.runtime_s),
            "out_dir": str(self.out_dir.resolve()),
            "feasibility_status": str(self.feasibility_status),
            "error_message": str(self.error_message),
        }


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
    "evaluation_index",
    "phase",
    "case_name",
    "objective_value",
    "optimizer_target_value",
    "is_valid",
    "expensive_simulation_executed",
    "incumbent_after_eval",
    "best_objective_value_after_eval",
    "runtime_s",
    "out_dir",
    "feasibility_status",
    "error_message",
    "raw_candidate_json",
    "theta_json",
]


def _initialize_history_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDNAMES)
        writer.writeheader()


def _append_history_csv(path: Path, entry: OpenLoopOptimizationHistoryEntry) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HISTORY_FIELDNAMES)
        writer.writerow(
            {
                "evaluation_index": entry.evaluation_index,
                "phase": entry.phase,
                "case_name": entry.case_name,
                "objective_value": entry.objective_value,
                "optimizer_target_value": entry.optimizer_target_value,
                "is_valid": int(entry.is_valid),
                "expensive_simulation_executed": int(entry.expensive_simulation_executed),
                "incumbent_after_eval": int(entry.incumbent_after_eval),
                "best_objective_value_after_eval": entry.best_objective_value_after_eval,
                "runtime_s": entry.runtime_s,
                "out_dir": str(entry.out_dir),
                "feasibility_status": entry.feasibility_status,
                "error_message": entry.error_message,
                "raw_candidate_json": json.dumps(list(entry.raw_candidate)),
                "theta_json": json.dumps(list(entry.theta)),
            }
        )


def _write_evaluation_metadata(path: Path, entry: OpenLoopOptimizationHistoryEntry) -> None:
    path.write_text(json.dumps(entry.to_dict(), indent=2), encoding="utf-8")


def optimize_open_loop_theta(
    theta0,
    config: OpenLoopProblemConfig,
    cryostage_params: CryostageModelParams,
    out_root_dir: str | Path,
    run_name: str,
    *,
    method: str = "Nelder-Mead",
    options: dict[str, Any] | None = None,
    bayesopt_config: BayesianOptimizationConfig | None = None,
    infeasible_objective_penalty: float = 1.0e6,
    T_plate0_C: float | None = None,
) -> OpenLoopOptimizationResult:
    """Orchestrate candidate evaluation, run folders, history, and incumbent tracking."""
    theta0 = _coerce_theta_tuple(theta0, name="theta0")
    build_reference_profile_from_theta(theta0, config)

    if not math.isfinite(float(infeasible_objective_penalty)):
        raise ValueError("infeasible_objective_penalty must be finite")

    method_key = _normalized_method(method)
    use_bayesian_backend = _is_bayesian_method(method)
    if use_bayesian_backend and bayesopt_config is None:
        bayesopt_config = BayesianOptimizationConfig()

    run_dir = Path(out_root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    evaluations_dir = run_dir / "evaluations"
    evaluations_dir.mkdir()

    history_csv_path = run_dir / "evaluation_history.csv"
    _initialize_history_csv(history_csv_path)

    history: list[OpenLoopOptimizationHistoryEntry] = []
    best_entry: OpenLoopOptimizationHistoryEntry | None = None
    best_objective_value = math.inf
    evaluation_index = 0

    def evaluate_candidate(theta, phase: str, raw_candidate=None) -> float:
        nonlocal evaluation_index, best_entry, best_objective_value

        theta_tuple = _coerce_theta_tuple(theta, name="theta")
        raw_candidate_tuple = theta_tuple if raw_candidate is None else _coerce_theta_tuple(raw_candidate, name="raw_candidate")
        evaluation_index += 1
        eval_name = f"eval_{evaluation_index:04d}"
        case_name = f"{run_name}_{eval_name}"
        out_dir = evaluations_dir / eval_name
        out_dir.mkdir()

        start_time = time.perf_counter()
        objective_value = float(infeasible_objective_penalty)
        optimizer_target_value = _optimizer_target_from_objective(
            objective_value,
            bayesian_backend=use_bayesian_backend,
        )
        is_valid = False
        expensive_simulation_executed = False
        feasibility_status = "infeasible"
        error_message = ""

        try:
            build_reference_profile_from_theta(theta_tuple, config)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
        else:
            expensive_simulation_executed = True
            try:
                objective_result = evaluate_open_loop_objective(
                    theta_tuple,
                    config,
                    cryostage_params,
                    out_dir,
                    case_name,
                    T_plate0_C=T_plate0_C,
                )
                raw_objective_value = float(objective_result.objective_value)
                if math.isfinite(raw_objective_value):
                    objective_value = raw_objective_value
                    is_valid = True
                    feasibility_status = "feasible"
                else:
                    feasibility_status = "evaluation_error"
                    error_message = f"Objective returned non-finite value: {raw_objective_value}"
            except Exception as exc:
                feasibility_status = "evaluation_error"
                error_message = f"{type(exc).__name__}: {exc}"

        optimizer_target_value = _optimizer_target_from_objective(
            objective_value,
            bayesian_backend=use_bayesian_backend,
        )
        incumbent_after_eval = bool(is_valid and objective_value < best_objective_value)
        best_objective_value_after_eval = objective_value if incumbent_after_eval else float(best_objective_value)
        runtime_s = float(time.perf_counter() - start_time)

        entry = OpenLoopOptimizationHistoryEntry(
            evaluation_index=evaluation_index,
            phase=str(phase),
            case_name=case_name,
            raw_candidate=raw_candidate_tuple,
            theta=theta_tuple,
            objective_value=float(objective_value),
            optimizer_target_value=float(optimizer_target_value),
            is_valid=bool(is_valid),
            expensive_simulation_executed=bool(expensive_simulation_executed),
            incumbent_after_eval=bool(incumbent_after_eval),
            best_objective_value_after_eval=float(best_objective_value_after_eval),
            runtime_s=float(runtime_s),
            out_dir=out_dir,
            feasibility_status=str(feasibility_status),
            error_message=str(error_message),
        )
        history.append(entry)
        _append_history_csv(history_csv_path, entry)
        _write_evaluation_metadata(out_dir / "evaluation_metadata.json", entry)

        if incumbent_after_eval:
            best_objective_value = float(objective_value)
            best_entry = entry

        return float(objective_value)

    if use_bayesian_backend:
        assert bayesopt_config is not None
        backend_result: BayesianOptimizationBackendResult = run_bayesian_optimization(
            theta0=theta0,
            default_bounds_C=config.T_ref_bounds_C,
            config=bayesopt_config,
            evaluate_candidate=evaluate_candidate,
        )
        success = bool(backend_result.success)
        status = int(backend_result.status)
        message = str(backend_result.message)
        resolved_method = str(backend_result.method)
        nfev = int(backend_result.nfev)
        nit = int(backend_result.nit)
    else:
        scipy_result = minimize(
            lambda theta: evaluate_candidate(theta, "legacy", raw_candidate=theta),
            np.asarray(theta0, dtype=np.float64),
            method=method,
            options=dict(options or {}),
        )

        success = bool(scipy_result.success)
        status = int(scipy_result.status)
        message = str(scipy_result.message)
        resolved_method = str(method)
        nfev = int(scipy_result.nfev)
        nit = int(scipy_result.nit) if getattr(scipy_result, "nit", None) is not None else None
        optimizer_theta = _coerce_theta_tuple(getattr(scipy_result, "x", theta0), name="optimizer_theta")
        try:
            build_reference_profile_from_theta(optimizer_theta, config)
        except Exception as exc:
            success = False
            message = (
                f"{message} | optimizer final theta invalid under active constraints: "
                f"{type(exc).__name__}: {exc}; returning best feasible evaluation instead"
            )

    if best_entry is None:
        raise RuntimeError(
            "Optimization completed without any feasible objective evaluation. "
            "Check evaluation_history.csv for the explicit inadmissibility or error reasons."
        )

    best_dir = run_dir / "best"
    shutil.copytree(best_entry.out_dir, best_dir)

    return OpenLoopOptimizationResult(
        best_theta=best_entry.theta,
        best_objective_value=best_entry.objective_value,
        best_evaluation_index=best_entry.evaluation_index,
        best_case_name=best_entry.case_name,
        success=bool(success),
        status=int(status),
        message=str(message),
        method=str(resolved_method),
        nfev=int(nfev),
        nit=nit,
        run_dir=run_dir,
        best_dir=best_dir,
        history_csv_path=history_csv_path,
        history=tuple(history),
    )


__all__ = [
    "OpenLoopOptimizationHistoryEntry",
    "OpenLoopOptimizationResult",
    "optimize_open_loop_theta",
]
