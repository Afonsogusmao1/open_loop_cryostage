#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryostage_model import DEFAULT_CRYOSTAGE_PARAMS, simulate_plate_temperature
from open_loop_optimizer import OpenLoopOptimizationResult, optimize_open_loop_theta
from open_loop_problem import build_front_reference, build_reference_profile_from_theta, load_front_csv
from run_open_loop_optimization import DEFAULT_THETA0, build_problem_config


DEFAULT_STUDY_NAME = "canonical_h240_k6_nm"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_study"


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


def _coerce_theta_tuple(values, *, name: str) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one entry")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(value) for value in arr)


def _parse_theta_arg(raw_theta: str | None) -> tuple[float, ...] | None:
    if raw_theta is None:
        return None
    parts = [part.strip() for part in raw_theta.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError("--theta0 must be a comma-separated list of floats")
    return _coerce_theta_tuple((float(part) for part in parts), name="theta0")


def _build_study_config(*, horizon_s: float, num_knots: int):
    base_config = build_problem_config()
    if num_knots < 2:
        raise ValueError("num_knots must be at least 2")
    knot_times_s = tuple(float(value) for value in np.linspace(0.0, horizon_s, int(num_knots)))
    return replace(base_config, horizon_s=float(horizon_s), knot_times_s=knot_times_s)


def _default_theta0_for_config(config) -> tuple[float, ...]:
    base_config = build_problem_config()
    base_tau = np.asarray(base_config.knot_times_s, dtype=np.float64) / float(base_config.horizon_s)
    study_tau = np.asarray(config.knot_times_s, dtype=np.float64) / float(config.horizon_s)
    theta0 = np.interp(study_tau, base_tau, np.asarray(DEFAULT_THETA0, dtype=np.float64))
    theta0 = np.clip(theta0, config.T_ref_bounds_C[0], config.T_ref_bounds_C[1])
    return _coerce_theta_tuple(theta0, name="default_theta0")


def _theta_satisfies_active_constraints(theta, config) -> bool:
    try:
        build_reference_profile_from_theta(theta, config)
    except Exception:
        return False
    return True


def _load_probe_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time_s: list[float] = []
    time_since_fill_s: list[float] = []
    probe_data: dict[str, list[float]] | None = None

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row")
        probe_columns = [name for name in reader.fieldnames if name.startswith("T_z") and name.endswith("_C")]
        if not probe_columns:
            raise ValueError(f"{path} does not contain any probe temperature columns")
        probe_data = {column: [] for column in probe_columns}

        for row in reader:
            try:
                current_time_s = float(row["time_s"])
                current_time_since_fill_s = float(row["time_since_fill_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path} contains an invalid probe row") from exc
            time_s.append(current_time_s)
            time_since_fill_s.append(current_time_since_fill_s)
            for column in probe_columns:
                probe_data[column].append(float(row[column]))

    assert probe_data is not None
    return (
        np.asarray(time_s, dtype=np.float64),
        np.asarray(time_since_fill_s, dtype=np.float64),
        {column: np.asarray(values, dtype=np.float64) for column, values in probe_data.items()},
    )


def _probe_label(column_name: str) -> str:
    label = column_name.removeprefix("T_z").removesuffix("_C").replace("p", ".")
    return f"z = {label.replace('mm', ' mm')}"


def _plot_series(
    *,
    x_s: np.ndarray,
    y_values,
    out_path: Path,
    title: str,
    ylabel: str,
    label: str,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(x_s, y_values, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_front_tracking(front_time_s: np.ndarray, z_front_m: np.ndarray, z_ref_m: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot(front_time_s, 1000.0 * z_front_m, linewidth=2.0, label=r"$z_{front}$")
    valid_mask = np.isfinite(z_ref_m)
    ax.plot(front_time_s[valid_mask], 1000.0 * z_ref_m[valid_mask], "--", linewidth=2.0, label=r"$z_{ref}$")
    ax.set_title("Front Tracking")
    ax.set_xlabel("Time Since Fill (s)")
    ax.set_ylabel("Front Position (mm)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_objective_history(result: OpenLoopOptimizationResult, out_path: Path) -> None:
    indices = np.asarray([entry.evaluation_index for entry in result.history], dtype=np.float64)
    objective_values = np.asarray(
        [
            entry.objective_value if entry.is_valid and math.isfinite(entry.objective_value) else np.nan
            for entry in result.history
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots()
    valid_mask = np.isfinite(objective_values)
    if np.any(valid_mask):
        ax.plot(
            indices[valid_mask],
            objective_values[valid_mask],
            marker="o",
            linewidth=1.5,
            markersize=4.0,
            label="valid evaluations",
        )
    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        invalid_y = np.full(np.count_nonzero(invalid_mask), result.best_objective_value)
        ax.scatter(indices[invalid_mask], invalid_y, marker="x", label="invalid")
    ax.scatter(
        [result.best_evaluation_index],
        [result.best_objective_value],
        color="tab:red",
        zorder=3,
        label="best",
    )
    ax.set_title("Optimization History")
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Objective Value")
    finite_values = objective_values[np.isfinite(objective_values)]
    if finite_values.size > 0 and np.all(finite_values > 0.0):
        spread = float(np.max(finite_values) / np.min(finite_values))
        if spread >= 10.0:
            ax.set_yscale("log")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)


def _plot_probe_temperatures(probes_path: Path, out_path: Path) -> Path | None:
    if not probes_path.exists():
        return None

    time_s, time_since_fill_s, probe_data = _load_probe_csv(probes_path)
    post_fill_mask = np.isfinite(time_since_fill_s)
    if np.any(post_fill_mask):
        x_values = time_since_fill_s[post_fill_mask]
        x_label = "Time Since Fill (s)"
        mask = post_fill_mask
    else:
        x_values = time_s
        x_label = "Time (s)"
        mask = slice(None)

    fig, ax = plt.subplots()
    for column_name, values in probe_data.items():
        ax.plot(x_values, values[mask], linewidth=1.75, label=_probe_label(column_name))
    ax.set_title("Best-Run Probe Temperatures")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Temperature (C)")
    ax.legend(loc="best")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _write_summary_files(summary_items: list[tuple[str, str]], *, txt_path: Path, csv_path: Path) -> None:
    with txt_path.open("w") as f:
        for key, value in summary_items:
            f.write(f"{key}: {value}\n")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(summary_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible open-loop study and post-process the best run.")
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME, help="Deterministic study folder name.")
    parser.add_argument(
        "--out-root-dir",
        default=str(DEFAULT_OUT_ROOT_DIR),
        help="Root folder where study folders are written.",
    )
    parser.add_argument("--horizon-s", type=float, default=240.0, help="Optimization horizon in seconds.")
    parser.add_argument("--num-knots", type=int, default=6, help="Number of T_ref knot temperatures.")
    parser.add_argument(
        "--theta0",
        default=None,
        help="Optional comma-separated initial theta override. Defaults to the canonical profile resampled to the knot grid.",
    )
    parser.add_argument("--method", default="Nelder-Mead", help="scipy.optimize.minimize method.")
    parser.add_argument("--maxiter", type=int, default=8, help="Maximum optimizer iterations.")
    parser.add_argument("--maxfev", type=int, default=18, help="Maximum objective evaluations.")
    parser.add_argument("--xatol", type=float, default=0.25, help="Absolute theta convergence tolerance.")
    parser.add_argument("--fatol", type=float, default=1.0e-9, help="Absolute objective convergence tolerance.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing study folder before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    config = _build_study_config(horizon_s=args.horizon_s, num_knots=args.num_knots)
    theta0 = _parse_theta_arg(args.theta0)
    if theta0 is None:
        theta0 = _default_theta0_for_config(config)
    if len(theta0) != len(config.knot_times_s):
        raise ValueError(
            "Initial theta length must match the number of knot times "
            f"({len(config.knot_times_s)} expected, got {len(theta0)})"
        )
    build_reference_profile_from_theta(theta0, config)

    out_root_dir = Path(args.out_root_dir)
    run_dir = out_root_dir / args.study_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{run_dir} already exists. Re-run with --overwrite to replace this deterministic study folder."
            )
        shutil.rmtree(run_dir)

    print(
        "Using canonical cryostage parameters "
        f"(tau_s={DEFAULT_CRYOSTAGE_PARAMS.tau_s:.6f}, "
        f"gain={DEFAULT_CRYOSTAGE_PARAMS.gain:.6f}, "
        f"offset_C={DEFAULT_CRYOSTAGE_PARAMS.offset_C:.6f})"
    )
    print(f"Study folder = {run_dir}")
    print(f"Active T_ref bounds = {config.T_ref_bounds_C}")
    print(f"Monotonicity required = {config.require_monotone_nonincreasing}")
    print(f"Knot times (s) = {config.knot_times_s}")
    print(f"Initial theta = {theta0}")

    result = optimize_open_loop_theta(
        theta0=theta0,
        config=config,
        cryostage_params=DEFAULT_CRYOSTAGE_PARAMS,
        out_root_dir=out_root_dir,
        run_name=args.study_name,
        method=args.method,
        options={
            "maxiter": args.maxiter,
            "maxfev": args.maxfev,
            "xatol": args.xatol,
            "fatol": args.fatol,
        },
    )

    analysis_dir = result.run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    best_front_path = result.best_dir / f"{result.best_case_name}_front.csv"
    best_probes_path = result.best_dir / f"{result.best_case_name}_probes.csv"
    best_xdmf_path = result.best_dir / f"{result.best_case_name}.xdmf"
    best_front = load_front_csv(best_front_path)
    z_front_reference_m = build_front_reference(best_front.time_since_fill_s, best_front.z_front_m, config)

    best_profile = build_reference_profile_from_theta(result.best_theta, config)
    cryostage_time_s = config.cryostage_time_grid_s()
    T_ref_C = np.asarray([float(best_profile(float(ti))) for ti in cryostage_time_s], dtype=np.float64)
    T_plate_C = simulate_plate_temperature(
        time_s=cryostage_time_s,
        T_ref_profile_C=best_profile,
        params=DEFAULT_CRYOSTAGE_PARAMS,
        T_plate0_C=float(config.solver_kwargs["bcs"].T_room_C),
    )

    T_ref_plot_path = analysis_dir / "T_ref_vs_time.png"
    T_plate_plot_path = analysis_dir / "T_plate_vs_time.png"
    front_plot_path = analysis_dir / "z_front_vs_z_ref.png"
    history_plot_path = analysis_dir / "objective_history.png"
    probes_plot_path = analysis_dir / "probe_temperatures.png"

    _plot_series(
        x_s=cryostage_time_s,
        y_values=T_ref_C,
        out_path=T_ref_plot_path,
        title="Best-Run Reference Temperature",
        ylabel="Temperature (C)",
        label=r"$T_{ref}$",
    )
    _plot_series(
        x_s=cryostage_time_s,
        y_values=T_plate_C,
        out_path=T_plate_plot_path,
        title="Best-Run Plate Temperature",
        ylabel="Temperature (C)",
        label=r"$T_{plate}$",
    )
    _plot_front_tracking(best_front.time_since_fill_s, best_front.z_front_m, z_front_reference_m, front_plot_path)
    _plot_objective_history(result, history_plot_path)
    probe_plot_written = _plot_probe_temperatures(best_probes_path, probes_plot_path)

    best_theta_is_valid = _theta_satisfies_active_constraints(result.best_theta, config)
    total_evaluations = len(result.history)
    valid_evaluations = sum(1 for entry in result.history if entry.is_valid)

    summary_items = [
        ("study_name", args.study_name),
        ("method", result.method),
        ("horizon_s", f"{config.horizon_s:.6f}"),
        ("cryostage_dt_s", f"{config.cryostage_dt_s:.6f}"),
        ("knot_times_s", json.dumps([float(value) for value in config.knot_times_s])),
        ("T_ref_bounds_C", json.dumps([float(value) for value in config.T_ref_bounds_C])),
        ("require_monotone_nonincreasing", json.dumps(bool(config.require_monotone_nonincreasing))),
        (
            "cryostage_params",
            json.dumps(
                {
                    "tau_s": float(DEFAULT_CRYOSTAGE_PARAMS.tau_s),
                    "gain": float(DEFAULT_CRYOSTAGE_PARAMS.gain),
                    "offset_C": float(DEFAULT_CRYOSTAGE_PARAMS.offset_C),
                }
            ),
        ),
        ("T_fill_C", f"{float(config.solver_kwargs['T_fill_C']):.6f}"),
        ("h_top", f"{float(config.solver_kwargs['bcs'].h_top):.6f}"),
        ("h_side", f"{float(config.solver_kwargs['bcs'].h_side):.6f}"),
        ("initial_theta", json.dumps([float(value) for value in theta0])),
        ("best_theta", json.dumps([float(value) for value in result.best_theta])),
        ("best_theta_satisfies_active_constraints", json.dumps(bool(best_theta_is_valid))),
        ("best_J", f"{result.best_objective_value:.9e}"),
        ("best_evaluation_index", str(result.best_evaluation_index)),
        ("total_evaluations", str(total_evaluations)),
        ("valid_evaluations", str(valid_evaluations)),
        ("scipy_nfev", str(result.nfev)),
        ("scipy_nit", "" if result.nit is None else str(result.nit)),
        ("optimizer_success", json.dumps(bool(result.success))),
        ("optimizer_status", str(result.status)),
        ("optimizer_message", result.message),
        ("run_dir", str(result.run_dir.resolve())),
        ("best_dir", str(result.best_dir.resolve())),
        ("history_csv_path", str(result.history_csv_path.resolve())),
        ("best_front_path", str(best_front_path.resolve())),
        ("best_probes_path", str(best_probes_path.resolve())),
        ("best_xdmf_path", str(best_xdmf_path.resolve())),
        ("analysis_dir", str(analysis_dir.resolve())),
        ("T_ref_plot_path", str(T_ref_plot_path.resolve())),
        ("T_plate_plot_path", str(T_plate_plot_path.resolve())),
        ("front_plot_path", str(front_plot_path.resolve())),
        ("history_plot_path", str(history_plot_path.resolve())),
        ("probe_plot_path", "" if probe_plot_written is None else str(probe_plot_written.resolve())),
    ]

    summary_txt_path = result.run_dir / "study_summary.txt"
    summary_csv_path = result.run_dir / "study_summary.csv"
    _write_summary_files(summary_items, txt_path=summary_txt_path, csv_path=summary_csv_path)

    print("Study summary")
    print(f"Best theta: {result.best_theta}")
    print(f"Best theta satisfies active constraints: {best_theta_is_valid}")
    print(f"Best J: {result.best_objective_value:.9e}")
    print(f"Evaluations: total={total_evaluations}, valid={valid_evaluations}, scipy_nfev={result.nfev}")
    print(
        f"Optimizer termination: success={result.success}, status={result.status}, "
        f"message={result.message}"
    )
    print(f"History CSV: {result.history_csv_path}")
    print(f"Best run folder: {result.best_dir}")
    print(f"Summary TXT: {summary_txt_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Analysis folder: {analysis_dir}")


if __name__ == "__main__":
    main()
