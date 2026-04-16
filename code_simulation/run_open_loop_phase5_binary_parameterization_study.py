#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from open_loop_bayesian_optimizer import bayes_opt_runtime_details
from open_loop_problem import front_definition_contract_summary, front_reference_contract_summary
from open_loop_workflow_config import ACTIVE_FORMULATION, DEFAULT_KNOT_TIME_SCHEDULE, build_problem_config, default_theta0_for_config
from run_open_loop_optimization import main as run_open_loop_optimization_main


PHASE5_SCIENTIFIC_QUESTION = (
    "Given two candidate T_ref(t) trajectory parameterizations, under the same active full-process "
    "workflow, fixed external time family, fixed objective, fixed operational observable, fixed target "
    "front trajectory, fixed admissibility stack, fixed reduced cryostage response path, fixed BO backend, "
    "budget, and seed policy, is either parameterization more defensible as the operational default for "
    "approximately linear freezing-front progression?"
)
PHASE5_FORMULATION = ACTIVE_FORMULATION
PHASE5_FIXED_KNOT_TIME_SCHEDULE = DEFAULT_KNOT_TIME_SCHEDULE
PHASE5_PARAMETERIZATION_ARMS = (
    {
        "name": "uniform_piecewise_linear_3",
        "num_knots": 3,
        "description": "Piecewise-linear T_ref(t) with 3 optimized absolute temperature parameters on fixed uniform support tau=(0, 0.5, 1).",
    },
    {
        "name": "uniform_piecewise_linear_4",
        "num_knots": 4,
        "description": "Piecewise-linear T_ref(t) with 4 optimized absolute temperature parameters on fixed uniform support tau=(0, 1/3, 2/3, 1).",
    },
)
PHASE5_DEFAULT_SEEDS = (17, 29, 41)
PHASE5_DEFAULT_INIT_POINTS = 4
PHASE5_DEFAULT_N_ITER = 8
PHASE5_DEFAULT_ACQ_KIND = "ucb"
PHASE5_DEFAULT_KAPPA = 2.0
PHASE5_DEFAULT_XI = 0.0
PHASE5_DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY = 1.0e6
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"
DEFAULT_STUDY_NAME = "phase5_binary_parameterization_uniform_n3_n4_seed17_29_41_init4_iter8"

BOUND_POLICY_TEMPLATE_SUPPORT_TAU = (0.0, 0.25, 0.5, 0.75, 1.0)
BOUND_POLICY_LOWER_TEMPERATURES_C = (-0.5, -9.0, -15.0, -18.0, -20.0)
BOUND_POLICY_UPPER_TEMPERATURES_C = (0.0, -3.0, -8.0, -12.0, -14.0)
POST_INITIAL_WARM_LIMIT_C = -5.0

MIN_MEDIAN_OBJECTIVE_IMPROVEMENT_FRACTION = 0.02
MAX_STABILITY_STD_RATIO = 1.25
MAX_FEASIBLE_FRACTION_DROP = 0.10

OUT_OF_SCOPE = (
    "No broad knot-count sweep.",
    "No external time-schedule comparison.",
    "No BO-budget or acquisition-function tuning campaign.",
    "No change to solver physics.",
    "No change to the operational front observable.",
    "No dT_mushy or front-definition sensitivity study.",
    "No experimental validation in this phase.",
)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 4.5),
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _bool_from_csv(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True", "TRUE"}


def _support_tau(config) -> tuple[float, ...]:
    return tuple(float(value) / float(config.horizon_s) for value in config.knot_times_s)


def _theta_bounds_for_support(support_tau: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
    base_tau = np.asarray(BOUND_POLICY_TEMPLATE_SUPPORT_TAU, dtype=np.float64)
    target_tau = np.asarray(support_tau, dtype=np.float64)
    lower = np.interp(target_tau, base_tau, np.asarray(BOUND_POLICY_LOWER_TEMPERATURES_C, dtype=np.float64))
    upper = np.interp(target_tau, base_tau, np.asarray(BOUND_POLICY_UPPER_TEMPERATURES_C, dtype=np.float64))
    if target_tau.size > 1:
        upper[1:] = np.minimum(upper[1:], POST_INITIAL_WARM_LIMIT_C)
    bounds = tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True))
    for idx, (lo, hi) in enumerate(bounds):
        if not lo < hi:
            raise ValueError(f"empty BO bound interval at theta[{idx}]: [{lo}, {hi}]")
    return bounds


def _seed_theta(config, theta_bounds_C: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    theta0 = np.asarray(default_theta0_for_config(config), dtype=np.float64)
    lower = np.asarray([pair[0] for pair in theta_bounds_C], dtype=np.float64)
    upper = np.asarray([pair[1] for pair in theta_bounds_C], dtype=np.float64)
    return tuple(float(value) for value in np.clip(theta0, lower, upper))


def _theta_arg(theta_C: tuple[float, ...]) -> str:
    return ",".join(f"{float(value):.6f}" for value in theta_C)


def _bounds_arg(theta_bounds_C: tuple[tuple[float, float], ...]) -> str:
    return ",".join(f"{lo:.6f}:{hi:.6f}" for lo, hi in theta_bounds_C)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_best_theta(path: Path, fallback_row: dict[str, str] | None) -> tuple[float, ...]:
    if path.exists():
        theta = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                theta.append(float(row["temperature_C"]))
        return tuple(theta)
    if fallback_row is None:
        return tuple()
    try:
        return tuple(float(value) for value in json.loads(fallback_row["theta_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return tuple()


def _summarize_run(
    *,
    arm: dict,
    seed: int,
    config,
    run_name: str,
    run_dir: Path,
    theta_bounds_C: tuple[tuple[float, float], ...],
    seed_theta_C: tuple[float, ...],
    status: str,
    message: str,
) -> dict:
    rows = _read_csv_rows(run_dir / "evaluation_history.csv")
    feasible_rows = [row for row in rows if _bool_from_csv(row.get("is_valid", "0"))]
    infeasible_rows = [row for row in rows if row.get("feasibility_status") == "infeasible"]
    error_rows = [row for row in rows if row.get("feasibility_status") == "evaluation_error"]
    seed_rows = [row for row in rows if row.get("phase") == "seed"]
    runtime_values = np.asarray([float(row["runtime_s"]) for row in rows], dtype=np.float64) if rows else np.asarray([])
    feasible_runtime_values = np.asarray([float(row["runtime_s"]) for row in feasible_rows], dtype=np.float64) if feasible_rows else np.asarray([])
    best_row = min(feasible_rows, key=lambda row: float(row["objective_value"])) if feasible_rows else None
    run_settings = _read_json(run_dir / "run_settings.json")
    artifacts = {str(key): str(value) for key, value in run_settings.get("artifacts", {}).items()}
    best_theta = _load_best_theta(run_dir / "best_theta_profile.csv", best_row)
    seed_objective = float(seed_rows[0]["objective_value"]) if seed_rows else math.nan
    best_objective = float(best_row["objective_value"]) if best_row is not None else math.nan
    resolved_status = status
    if status == "completed" and not feasible_rows:
        resolved_status = "no_feasible_evaluation"
    return {
        "parameterization": arm["name"],
        "parameter_count": int(arm["num_knots"]),
        "support_tau": _support_tau(config),
        "support_times_s": tuple(float(value) for value in config.knot_times_s),
        "seed": int(seed),
        "status": resolved_status,
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "theta_bounds_C": theta_bounds_C,
        "seed_theta_C": seed_theta_C,
        "seeded_baseline_feasible": bool(seed_rows and _bool_from_csv(seed_rows[0].get("is_valid", "0"))),
        "seed_objective_value": seed_objective,
        "best_objective_value": best_objective,
        "best_theta_C": best_theta,
        "best_evaluation_index": int(best_row["evaluation_index"]) if best_row is not None else -1,
        "total_evaluations": len(rows),
        "feasible_evaluations": len(feasible_rows),
        "infeasible_evaluations": len(infeasible_rows),
        "evaluation_errors": len(error_rows),
        "rejected_evaluations": len(infeasible_rows) + len(error_rows),
        "expensive_simulation_runs": sum(1 for row in rows if _bool_from_csv(row.get("expensive_simulation_executed", "0"))),
        "total_runtime_s": float(np.sum(runtime_values)) if runtime_values.size else 0.0,
        "median_feasible_runtime_s": float(np.median(feasible_runtime_values)) if feasible_runtime_values.size else math.nan,
        "best_evaluation_runtime_s": float(best_row["runtime_s"]) if best_row is not None else math.nan,
        "improved_beyond_seeded_baseline": bool(math.isfinite(seed_objective) and math.isfinite(best_objective) and best_objective < seed_objective - 1.0e-12),
        "dominated_by_rejections": (len(infeasible_rows) + len(error_rows)) > len(feasible_rows),
        "best_artifacts": artifacts,
        "key_message": message,
    }


def _run_one(arm: dict, seed: int, runs_root_dir: Path, bo: dict) -> dict:
    config = build_problem_config(
        formulation=PHASE5_FORMULATION,
        num_knots=int(arm["num_knots"]),
        knot_time_schedule=PHASE5_FIXED_KNOT_TIME_SCHEDULE,
    )
    support_tau = _support_tau(config)
    theta_bounds_C = _theta_bounds_for_support(support_tau)
    seed_theta_C = _seed_theta(config, theta_bounds_C)
    run_name = f"phase5_{arm['name']}_seed{seed}_init{bo['init_points']}_iter{bo['n_iter']}"
    arm_seed_root = runs_root_dir / arm["name"] / f"seed_{seed}"
    run_dir = arm_seed_root / run_name
    optimization_args = [
        "--run-name", run_name,
        "--out-root-dir", str(arm_seed_root),
        "--formulation", PHASE5_FORMULATION,
        "--num-knots", str(int(arm["num_knots"])),
        "--knot-time-schedule", PHASE5_FIXED_KNOT_TIME_SCHEDULE,
        f"--theta0={_theta_arg(seed_theta_C)}",
        "--method", "bayesian-optimization",
        "--seed", str(int(seed)),
        "--init-points", str(int(bo["init_points"])),
        "--n-iter", str(int(bo["n_iter"])),
        "--acq-kind", str(bo["acquisition_kind"]),
        "--kappa", str(float(bo["acquisition_kappa"])),
        "--xi", str(float(bo["acquisition_xi"])),
        f"--theta-bounds={_bounds_arg(theta_bounds_C)}",
        "--infeasible-objective-penalty", str(float(bo["infeasible_objective_penalty"])),
        "--smoke-test-note",
        "Member run of the Phase 5 binary trajectory-parameterization BO study; interpret through study-level outputs.",
    ]
    status = "completed"
    message = "completed"
    try:
        run_open_loop_optimization_main(optimization_args)
    except Exception as exc:
        status = "failed"
        message = f"{type(exc).__name__}: {exc}"
    return _summarize_run(
        arm=arm,
        seed=seed,
        config=config,
        run_name=run_name,
        run_dir=run_dir,
        theta_bounds_C=theta_bounds_C,
        seed_theta_C=seed_theta_C,
        status=status,
        message=message,
    )


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _aggregate(summaries: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    out = []
    for arm in PHASE5_PARAMETERIZATION_ARMS:
        rows = [row for row in summaries if row["parameterization"] == arm["name"]]
        best_values = _finite([row["best_objective_value"] for row in rows])
        total = sum(row["total_evaluations"] for row in rows)
        feasible = sum(row["feasible_evaluations"] for row in rows)
        rejected = sum(row["rejected_evaluations"] for row in rows)
        best_row = min([row for row in rows if math.isfinite(row["best_objective_value"])], key=lambda row: row["best_objective_value"]) if best_values.size else None
        out.append(
            {
                "parameterization": arm["name"],
                "parameter_count": int(arm["num_knots"]),
                "description": arm["description"],
                "requested_seed_count": len(seeds),
                "completed_seed_count": sum(1 for row in rows if row["status"] == "completed"),
                "finite_best_seed_count": int(best_values.size),
                "best_objective_min": float(np.min(best_values)) if best_values.size else math.nan,
                "best_objective_mean": float(np.mean(best_values)) if best_values.size else math.nan,
                "best_objective_median": float(np.median(best_values)) if best_values.size else math.nan,
                "best_objective_std": float(np.std(best_values, ddof=1)) if best_values.size >= 2 else math.nan,
                "best_objective_range": float(np.max(best_values) - np.min(best_values)) if best_values.size else math.nan,
                "best_seed": best_row["seed"] if best_row is not None else "",
                "best_run_dir": best_row["run_dir"] if best_row is not None else "",
                "best_theta_C": best_row["best_theta_C"] if best_row is not None else tuple(),
                "total_evaluations": total,
                "feasible_evaluations": feasible,
                "infeasible_evaluations": sum(row["infeasible_evaluations"] for row in rows),
                "evaluation_errors": sum(row["evaluation_errors"] for row in rows),
                "rejected_evaluations": rejected,
                "expensive_simulation_runs": sum(row["expensive_simulation_runs"] for row in rows),
                "feasible_fraction": feasible / total if total else math.nan,
                "rejected_fraction": rejected / total if total else math.nan,
            }
        )
    return out


def _relative_improvement(reference: float, candidate: float) -> float:
    if not (math.isfinite(reference) and math.isfinite(candidate)):
        return math.nan
    return float((reference - candidate) / max(abs(reference), 1.0e-12))


def _recommend(aggregate_rows: list[dict], seeds: tuple[int, ...]) -> dict:
    finite_rows = [row for row in aggregate_rows if math.isfinite(row["best_objective_median"])]
    if len(finite_rows) < 2:
        return {"decision": "no_carry_forward_default", "recommended_parameterization": None, "reason": "fewer than two finite aggregate results", "criteria": {}}
    candidate, comparator = sorted(finite_rows, key=lambda row: row["best_objective_median"])[:2]
    median_gain = _relative_improvement(comparator["best_objective_median"], candidate["best_objective_median"])
    mean_gain = _relative_improvement(comparator["best_objective_mean"], candidate["best_objective_mean"])
    best_gain = _relative_improvement(comparator["best_objective_min"], candidate["best_objective_min"])
    candidate_std = candidate["best_objective_std"]
    comparator_std = comparator["best_objective_std"]
    stability_ok = math.isfinite(candidate_std) and math.isfinite(comparator_std) and candidate_std <= max(MAX_STABILITY_STD_RATIO * comparator_std, 1.0e-12)
    feasibility_ok = candidate["feasible_fraction"] + MAX_FEASIBLE_FRACTION_DROP >= comparator["feasible_fraction"]
    criteria = {
        "candidate": candidate["parameterization"],
        "comparator": comparator["parameterization"],
        "median_objective_improvement_fraction": median_gain,
        "mean_objective_improvement_fraction": mean_gain,
        "best_objective_improvement_fraction": best_gain,
        "all_requested_seeds_have_finite_best_objectives": candidate["finite_best_seed_count"] == len(seeds) and comparator["finite_best_seed_count"] == len(seeds),
        "objective_advantage_ok": median_gain >= MIN_MEDIAN_OBJECTIVE_IMPROVEMENT_FRACTION and mean_gain > 0.0 and best_gain >= 0.0,
        "stability_ok": stability_ok,
        "feasibility_ok": feasibility_ok,
        "evaluation_errors_ok": candidate["evaluation_errors"] <= comparator["evaluation_errors"],
        "best_artifacts_ok": bool(candidate["best_run_dir"]),
        "min_median_improvement_required": MIN_MEDIAN_OBJECTIVE_IMPROVEMENT_FRACTION,
        "max_stability_std_ratio": MAX_STABILITY_STD_RATIO,
        "max_feasible_fraction_drop": MAX_FEASIBLE_FRACTION_DROP,
    }
    ok = all(
        bool(criteria[key])
        for key in (
            "all_requested_seeds_have_finite_best_objectives",
            "objective_advantage_ok",
            "stability_ok",
            "feasibility_ok",
            "evaluation_errors_ok",
            "best_artifacts_ok",
        )
    )
    if ok:
        return {
            "decision": "carry_forward_default",
            "recommended_parameterization": candidate["parameterization"],
            "reason": "one arm clears the objective, stability, feasibility, and executability checks",
            "criteria": criteria,
        }
    return {
        "decision": "no_carry_forward_default",
        "recommended_parameterization": None,
        "reason": "evidence is weak, ambiguous, seed-sensitive, or carries feasibility/executability burden under the conservative Phase 5 rule",
        "criteria": criteria,
    }


def _write_per_seed_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "parameterization", "parameter_count", "support_tau_json", "support_times_s_json", "seed", "status", "run_name", "run_dir",
        "theta_bounds_json", "seed_theta_json", "seeded_baseline_feasible", "seed_objective_value", "best_objective_value", "best_theta_json",
        "best_evaluation_index", "total_evaluations", "feasible_evaluations", "infeasible_evaluations", "evaluation_errors", "rejected_evaluations",
        "expensive_simulation_runs", "total_runtime_s", "median_feasible_runtime_s", "best_evaluation_runtime_s", "improved_beyond_seeded_baseline",
        "dominated_by_rejections", "best_artifacts_json", "key_message",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "parameterization": row["parameterization"],
                    "parameter_count": row["parameter_count"],
                    "support_tau_json": json.dumps(list(row["support_tau"])),
                    "support_times_s_json": json.dumps(list(row["support_times_s"])),
                    "seed": row["seed"],
                    "status": row["status"],
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "theta_bounds_json": json.dumps([list(pair) for pair in row["theta_bounds_C"]]),
                    "seed_theta_json": json.dumps(list(row["seed_theta_C"])),
                    "seeded_baseline_feasible": int(row["seeded_baseline_feasible"]),
                    "seed_objective_value": row["seed_objective_value"],
                    "best_objective_value": row["best_objective_value"],
                    "best_theta_json": json.dumps(list(row["best_theta_C"])),
                    "best_evaluation_index": row["best_evaluation_index"],
                    "total_evaluations": row["total_evaluations"],
                    "feasible_evaluations": row["feasible_evaluations"],
                    "infeasible_evaluations": row["infeasible_evaluations"],
                    "evaluation_errors": row["evaluation_errors"],
                    "rejected_evaluations": row["rejected_evaluations"],
                    "expensive_simulation_runs": row["expensive_simulation_runs"],
                    "total_runtime_s": row["total_runtime_s"],
                    "median_feasible_runtime_s": row["median_feasible_runtime_s"],
                    "best_evaluation_runtime_s": row["best_evaluation_runtime_s"],
                    "improved_beyond_seeded_baseline": int(row["improved_beyond_seeded_baseline"]),
                    "dominated_by_rejections": int(row["dominated_by_rejections"]),
                    "best_artifacts_json": json.dumps(row["best_artifacts"], sort_keys=True),
                    "key_message": row["key_message"],
                }
            )


def _write_aggregate_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "parameterization", "parameter_count", "description", "requested_seed_count", "completed_seed_count", "finite_best_seed_count",
        "best_objective_min", "best_objective_mean", "best_objective_median", "best_objective_std", "best_objective_range", "best_seed",
        "best_run_dir", "best_theta_json", "total_evaluations", "feasible_evaluations", "infeasible_evaluations", "evaluation_errors",
        "rejected_evaluations", "expensive_simulation_runs", "feasible_fraction", "rejected_fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["best_theta_json"] = json.dumps(list(row["best_theta_C"]))
            del out["best_theta_C"]
            writer.writerow({field: out.get(field, "") for field in fieldnames})


def _copy_if_exists(src_text: str | None, dest: Path) -> bool:
    if not src_text:
        return False
    src = Path(src_text)
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_best_artifacts(study_dir: Path, summaries: list[dict]) -> Path:
    registry = study_dir / "best_solution_registry.csv"
    with registry.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameterization", "seed", "best_objective_value", "best_run_dir", "copied_artifact_dir", "copied_files_json"])
        writer.writeheader()
        for arm in PHASE5_PARAMETERIZATION_ARMS:
            eligible = [row for row in summaries if row["parameterization"] == arm["name"] and math.isfinite(row["best_objective_value"])]
            if not eligible:
                writer.writerow({"parameterization": arm["name"], "seed": "", "best_objective_value": math.nan, "best_run_dir": "", "copied_artifact_dir": "", "copied_files_json": "[]"})
                continue
            best = min(eligible, key=lambda row: row["best_objective_value"])
            dest_dir = study_dir / "best_solutions" / arm["name"]
            copied = []
            for key, filename in {
                "best_theta_profile_csv": "best_theta_profile.csv",
                "best_summary_md": "best_solution_summary.md",
                "history_csv_path": "evaluation_history.csv",
                "objective_history_plot": "objective_history.png",
                "feasibility_plot": "feasible_vs_infeasible.png",
                "front_tracking_plot": "best_front_tracking.png",
                "plate_reference_trajectory_plot": "plate_reference_trajectory.png",
            }.items():
                if _copy_if_exists(best["best_artifacts"].get(key), dest_dir / filename):
                    copied.append(filename)
            best_dir_text = best["best_artifacts"].get("best_dir")
            best_dir = Path(best_dir_text) if best_dir_text else None
            if best_dir is not None and best_dir.exists():
                for src in sorted(best_dir.glob("*_front.csv")):
                    shutil.copy2(src, dest_dir / "best_front.csv")
                    copied.append("best_front.csv")
                    break
                for src in sorted(best_dir.glob("*_objective_summary.txt")):
                    shutil.copy2(src, dest_dir / "best_objective_summary.txt")
                    copied.append("best_objective_summary.txt")
                    break
            writer.writerow(
                {
                    "parameterization": best["parameterization"],
                    "seed": best["seed"],
                    "best_objective_value": best["best_objective_value"],
                    "best_run_dir": best["run_dir"],
                    "copied_artifact_dir": str(dest_dir.resolve()),
                    "copied_files_json": json.dumps(copied),
                }
            )
    return registry


def _plot_best_objective(path: Path, rows: list[dict], seeds: tuple[int, ...]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(seeds), dtype=np.float64)
    for arm in PHASE5_PARAMETERIZATION_ARMS:
        y = []
        for seed in seeds:
            match = [row for row in rows if row["parameterization"] == arm["name"] and row["seed"] == seed]
            y.append(match[0]["best_objective_value"] if match else math.nan)
        ax.plot(x, np.asarray(y, dtype=np.float64), marker="o", linewidth=2.0, label=arm["name"])
    ax.set_xticks(x, labels=[str(seed) for seed in seeds])
    ax.set_title("Phase 5 Best Objective by BO Seed")
    ax.set_xlabel("BO seed")
    ax.set_ylabel("Best objective value")
    finite = [row["best_objective_value"] for row in rows if math.isfinite(row["best_objective_value"]) and row["best_objective_value"] > 0.0]
    if len(finite) >= 2 and max(finite) / min(finite) >= 10.0:
        ax.set_yscale("log")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _plot_burden(path: Path, rows: list[dict]) -> None:
    labels = [row["parameterization"] for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    feasible = np.asarray([row["feasible_evaluations"] for row in rows], dtype=np.float64)
    infeasible = np.asarray([row["infeasible_evaluations"] for row in rows], dtype=np.float64)
    errors = np.asarray([row["evaluation_errors"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x, feasible, label="valid / feasible", color="#2A9D8F")
    ax.bar(x, infeasible, bottom=feasible, label="rejected / infeasible", color="#C44536")
    ax.bar(x, errors, bottom=feasible + infeasible, label="evaluation error", color="#6D597A")
    ax.set_xticks(x, labels=labels, rotation=10, ha="right")
    ax.set_title("Phase 5 Evaluation Burden")
    ax.set_ylabel("Evaluation count")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)


def _fmt(value: object, precision: int = 6) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{out:.{precision}e}" if math.isfinite(out) else "n/a"


def _tuple_text(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{float(value):.5g}" for value in values) + ")"


def _write_settings_md(path: Path, seeds: tuple[int, ...], bo: dict, bo_runtime: dict) -> None:
    first_config = build_problem_config(formulation=PHASE5_FORMULATION, num_knots=3, knot_time_schedule=PHASE5_FIXED_KNOT_TIME_SCHEDULE)
    front_contract = front_definition_contract_summary(first_config)
    reference_contract = front_reference_contract_summary(first_config)
    lines = ["# Phase 5 Binary BO Study Settings", "", "## Scientific question", PHASE5_SCIENTIFIC_QUESTION, "", "## Compared parameterizations"]
    for arm in PHASE5_PARAMETERIZATION_ARMS:
        config = build_problem_config(formulation=PHASE5_FORMULATION, num_knots=arm["num_knots"], knot_time_schedule=PHASE5_FIXED_KNOT_TIME_SCHEDULE)
        lines.extend(
            [
                f"- `{arm['name']}`: {arm['description']}",
                f"  support_tau=`{_support_tau(config)}`",
                f"  support_times_s=`{tuple(float(value) for value in config.knot_times_s)}`",
                f"  theta_bounds_C=`{_theta_bounds_for_support(_support_tau(config))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Fixed settings",
            "| Setting | Value |",
            "| --- | --- |",
            f"| Formulation | `{PHASE5_FORMULATION}` |",
            f"| External time family | `{PHASE5_FIXED_KNOT_TIME_SCHEDULE}` |",
            f"| BO backend | `bayesian-optimization {bo_runtime['package_version']}` |",
            f"| BO package path | `{bo_runtime['package_path']}` |",
            f"| Seeds | `{seeds}` |",
            f"| BO budget | `seed theta + {bo['init_points']} random + {bo['n_iter']} acquisition-guided evaluations per arm/seed` |",
            f"| Acquisition | `{bo['acquisition_kind']}`, kappa=`{bo['acquisition_kappa']}`, xi=`{bo['acquisition_xi']}` |",
            f"| Infeasible objective penalty | `{bo['infeasible_objective_penalty']}` |",
            f"| Front definition mode | `{front_contract['front_definition_mode']}` |",
            f"| Front threshold C | `{front_contract['front_threshold_C']:.9e}` |",
            f"| Front reference mode | `{reference_contract['front_reference_mode']}` |",
            f"| Target end s | `{reference_contract['target_end_s']:.6f}` |",
            f"| Implied target front speed m/s | `{reference_contract['implied_target_front_speed_m_per_s']:.9e}` |",
            f"| Objective weights | tracking=`{first_config.tracking_weight}`, smoothness=`{first_config.smoothness_weight}`, completion=`{first_config.completion_weight}`, terminal=`{first_config.terminal_weight}` |",
            "| Admissibility | active transient-plus-hold pre-screen before expensive simulation |",
            "| Reduced cryostage path | active reduced model / inner PID response path |",
            "| Solver physics | unchanged full-process solver settings from `build_problem_config` |",
            "",
            "## Decision rule",
            f"Carry forward only if one arm improves median objective by at least {100.0 * MIN_MEDIAN_OBJECTIVE_IMPROVEMENT_FRACTION:.1f}% while not materially worsening stability, feasibility, or executability.",
            "",
            "## Deliberately out of scope",
        ]
    )
    lines.extend(f"- {item}" for item in OUT_OF_SCOPE)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(path: Path, rows: list[dict], aggregates: list[dict], recommendation: dict, artifact_paths: dict[str, Path]) -> None:
    lines = [
        "# Phase 5 Binary BO Study Summary",
        "",
        "## Operational question",
        PHASE5_SCIENTIFIC_QUESTION,
        "",
        "## Per-seed results",
        "| Parameterization | Seed | Status | Best objective | Feasible | Rejected/infeasible | Eval errors | Best theta |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        theta = _tuple_text(row["best_theta_C"]) if row["best_theta_C"] else "n/a"
        lines.append(
            f"| `{row['parameterization']}` | {row['seed']} | `{row['status']}` | {_fmt(row['best_objective_value'])} | "
            f"{row['feasible_evaluations']} | {row['infeasible_evaluations']} | {row['evaluation_errors']} | `{theta}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate by parameterization",
            "| Parameterization | Finite seeds | Best | Median | Mean | Std | Feasible fraction | Rejected fraction | Best seed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| `{row['parameterization']}` | {row['finite_best_seed_count']} | {_fmt(row['best_objective_min'])} | "
            f"{_fmt(row['best_objective_median'])} | {_fmt(row['best_objective_mean'])} | {_fmt(row['best_objective_std'])} | "
            f"{row['feasible_fraction']:.3f} | {row['rejected_fraction']:.3f} | `{row['best_seed']}` |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            f"- Decision: `{recommendation['decision']}`",
            f"- Recommended parameterization: `{recommendation['recommended_parameterization']}`",
            f"- Reason: {recommendation['reason']}",
            f"- Decision criteria JSON: `{json.dumps(recommendation['criteria'], sort_keys=True)}`",
            "",
            "## Outputs",
        ]
    )
    for label, path_value in artifact_paths.items():
        lines.append(f"- {label}: `{path_value}`")
    lines.extend(["- Per-arm best trajectory artifacts: `best_solutions/<parameterization>/`", "", "## Deliberately left out"])
    lines.extend(f"- {item}" for item in OUT_OF_SCOPE)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the narrow Phase 5 binary BO trajectory-parameterization study.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--out-root-dir", default=str(DEFAULT_OUT_ROOT_DIR))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PHASE5_DEFAULT_SEEDS))
    parser.add_argument("--init-points", type=int, default=PHASE5_DEFAULT_INIT_POINTS)
    parser.add_argument("--n-iter", type=int, default=PHASE5_DEFAULT_N_ITER)
    parser.add_argument("--acq-kind", default=PHASE5_DEFAULT_ACQ_KIND, choices=("ucb", "ei", "poi"))
    parser.add_argument("--kappa", type=float, default=PHASE5_DEFAULT_KAPPA)
    parser.add_argument("--xi", type=float, default=PHASE5_DEFAULT_XI)
    parser.add_argument("--infeasible-objective-penalty", type=float, default=PHASE5_DEFAULT_INFEASIBLE_OBJECTIVE_PENALTY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _configure_matplotlib()
    if int(args.init_points) < 0 or int(args.n_iter) < 0:
        raise ValueError("--init-points and --n-iter must be non-negative")
    seeds = _parse_seeds(args.seeds)
    bo = {
        "method": "bayesian-optimization",
        "init_points": int(args.init_points),
        "n_iter": int(args.n_iter),
        "acquisition_kind": str(args.acq_kind),
        "acquisition_kappa": float(args.kappa),
        "acquisition_xi": float(args.xi),
        "seed_with_theta0": True,
        "infeasible_objective_penalty": float(args.infeasible_objective_penalty),
    }
    bo_runtime = bayes_opt_runtime_details()
    study_dir = Path(args.out_root_dir) / args.study_name
    runs_root_dir = study_dir / "runs"
    if study_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{study_dir} already exists. Re-run with --overwrite to replace it.")
        shutil.rmtree(study_dir)
    runs_root_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for arm in PHASE5_PARAMETERIZATION_ARMS:
        for seed in seeds:
            summaries.append(_run_one(arm, int(seed), runs_root_dir, bo))
    summaries.sort(key=lambda row: ([arm["name"] for arm in PHASE5_PARAMETERIZATION_ARMS].index(row["parameterization"]), row["seed"]))
    aggregates = _aggregate(summaries, seeds)
    recommendation = _recommend(aggregates, seeds)

    settings_md = study_dir / "study_settings.md"
    settings_json = study_dir / "study_settings.json"
    per_seed_csv = study_dir / "per_seed_summary.csv"
    aggregate_csv = study_dir / "aggregate_by_parameterization.csv"
    summary_md = study_dir / "study_summary.md"
    objective_plot = study_dir / "best_objective_by_seed.png"
    burden_plot = study_dir / "evaluation_burden_by_parameterization.png"
    _write_per_seed_csv(per_seed_csv, summaries)
    _write_aggregate_csv(aggregate_csv, aggregates)
    _write_settings_md(settings_md, seeds, bo, bo_runtime)
    _plot_best_objective(objective_plot, summaries, seeds)
    _plot_burden(burden_plot, aggregates)
    best_registry = _copy_best_artifacts(study_dir, summaries)
    artifact_paths = {
        "Study settings": settings_md,
        "Per-seed summary CSV": per_seed_csv,
        "Aggregate summary CSV": aggregate_csv,
        "Best-solution registry": best_registry,
        "Best objective by seed plot": objective_plot,
        "Evaluation burden plot": burden_plot,
    }
    _write_summary_md(summary_md, summaries, aggregates, recommendation, artifact_paths)
    settings_json.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "scientific_question": PHASE5_SCIENTIFIC_QUESTION,
                "formulation": PHASE5_FORMULATION,
                "fixed_external_time_family": PHASE5_FIXED_KNOT_TIME_SCHEDULE,
                "parameterization_arms": PHASE5_PARAMETERIZATION_ARMS,
                "seed_policy": {"seeds": list(seeds), "same_seed_set_for_every_arm": True},
                "budget_policy": {"seed_with_theta0": True, "init_points": int(args.init_points), "n_iter": int(args.n_iter), "total_requested_evaluations_per_arm_seed": 1 + int(args.init_points) + int(args.n_iter)},
                "bo_settings": bo,
                "bound_policy": {
                    "template_support_tau": BOUND_POLICY_TEMPLATE_SUPPORT_TAU,
                    "lower_temperatures_C": BOUND_POLICY_LOWER_TEMPERATURES_C,
                    "upper_temperatures_C": BOUND_POLICY_UPPER_TEMPERATURES_C,
                    "post_initial_warm_limit_C": POST_INITIAL_WARM_LIMIT_C,
                },
                "bo_runtime": bo_runtime,
                "decision_policy": {
                    "min_median_objective_improvement_fraction": MIN_MEDIAN_OBJECTIVE_IMPROVEMENT_FRACTION,
                    "max_stability_std_ratio": MAX_STABILITY_STD_RATIO,
                    "max_feasible_fraction_drop": MAX_FEASIBLE_FRACTION_DROP,
                },
                "recommendation": recommendation,
                "outputs": {key: str(value.resolve()) for key, value in artifact_paths.items()},
                "runs_root_dir": str(runs_root_dir.resolve()),
                "best_solutions_dir": str((study_dir / "best_solutions").resolve()),
                "out_of_scope": list(OUT_OF_SCOPE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Phase 5 binary BO study completed: {study_dir}")
    for row in aggregates:
        print(f"{row['parameterization']}: best={_fmt(row['best_objective_min'])}, median={_fmt(row['best_objective_median'])}, feasible={row['feasible_evaluations']}, rejected={row['rejected_evaluations']}")
    print(f"Recommendation: {recommendation['decision']} ({recommendation['reason']})")
    print(f"Study summary: {summary_md}")


if __name__ == "__main__":
    main()
