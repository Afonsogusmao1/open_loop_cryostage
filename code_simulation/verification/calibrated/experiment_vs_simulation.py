from __future__ import annotations

"""Generate calibrated experiment-vs-simulation figures for rho(T) runs."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import (
    ACTIVE_TARGETS_C,
    CALIBRATED_FIGURES_DIR,
    COMPARISON_TARGETS_C,
    WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    build_target_plan,
    discover_simulation_outputs,
    experimental_csv_paths,
    target_filename_tag,
)


PROBE_SPECS = (
    ("3.0 mm", ("T_z3p0mm_C", "T_z3C", "T_z1C", "T1"), ("T3", "T_3", "probe3"), "#118a8a"),
    ("6.2 mm", ("T_z6p2mm_C", "T_z7C", "T_z2C", "T2"), ("T7", "T_7", "probe7"), "#2563eb"),
    ("11.0 mm", ("T_z11p0mm_C", "T_z11p5mm_C", "T_z12C", "T_z3C", "T3"), ("T12", "T_12", "probe12"), "#d97706"),
)
DEFAULT_PRE_S = 20.0
DEFAULT_MAX_PLOT_S = 700.0
DEFAULT_Y_LIMS_C = (-25.0, 20.0)
LEGACY_MAX_PLOT_S_BY_TARGET_C = {
    -5.0: 1200.0,
    -10.0: 1200.0,
    -15.0: 900.0,
    -20.0: 550.0,
}
MANAGED_FIGURE_GLOBS = (
    "compare_sim_vs_experiment_*.png",
    "compare_sim_vs_experiment_panel_*.png",
)


@dataclass(frozen=True)
class ExperimentalRun:
    label: str
    t_rel_s: np.ndarray
    probe_data_C: dict[str, np.ndarray]
    median_dt_s: float


@dataclass(frozen=True)
class SimulationRun:
    t_rel_s: np.ndarray
    probe_data_C: dict[str, np.ndarray]


@dataclass(frozen=True)
class ExperimentStatistics:
    t_plot_s: np.ndarray
    mean_C: dict[str, np.ndarray]
    std_C: dict[str, np.ndarray]


def _parse_targets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one target temperature")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target temperatures must be finite")
    return values


def _value_tag(value: float) -> str:
    value = float(value)
    if abs(value) < 5.0e-13:
        return "0"
    text = f"{abs(value):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _legacy_plot_xmax(target_C: float) -> float:
    return float(LEGACY_MAX_PLOT_S_BY_TARGET_C.get(float(target_C), DEFAULT_MAX_PLOT_S))


def _managed_output_paths(output_dir: Path, *, target_C: float, fill_C: float) -> tuple[Path, Path]:
    target_tag = target_filename_tag(target_C)
    fill_tag = _value_tag(fill_C)
    return (
        output_dir / f"compare_sim_vs_experiment_{target_tag}_{fill_tag}_11p0mm.png",
        output_dir / f"compare_sim_vs_experiment_panel_{target_tag}_{fill_tag}_11p0mm.png",
    )


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    except csv.Error:
        return ","
    return str(dialect.delimiter)


def _as_float(raw: str | None) -> float:
    if raw is None:
        return math.nan
    text = str(raw).strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    delimiter = _detect_delimiter(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(_as_float(row.get(name)))
    return {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}


def _first_existing_column(available: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in available:
            return name
    raise KeyError(f"None of these columns were found: {candidates}")


def _smooth_series(values: np.ndarray, window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size < window:
        return values.copy()
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _detect_insertion_time_exp(
    t_s: np.ndarray,
    T3_C: np.ndarray,
    T7_C: np.ndarray,
    T12_C: np.ndarray,
    *,
    smooth_window: int = 9,
) -> float:
    t_s = np.asarray(t_s, dtype=np.float64)
    dt_s = np.diff(t_s)
    dt_s[dt_s == 0.0] = math.nan
    signals = (
        _smooth_series(T3_C, smooth_window),
        _smooth_series(T7_C, smooth_window),
        _smooth_series(T12_C, smooth_window),
    )
    derivatives = []
    for signal in signals:
        with np.errstate(invalid="ignore", divide="ignore"):
            derivatives.append(np.diff(signal) / dt_s)
    score = np.nanmax(np.vstack(derivatives), axis=0)
    if not np.any(np.isfinite(score)):
        raise RuntimeError("Could not detect experimental insertion time because all derivatives are non-finite.")
    idx = int(np.nanargmax(score))
    return 0.5 * float(t_s[idx] + t_s[idx + 1])


def _load_experimental_run(path: Path) -> ExperimentalRun:
    columns = _read_numeric_csv(path)
    t_col = _first_existing_column(columns, ("t_rec_s", "t_s", "time_s", "time"))
    t_s = columns[t_col].astype(np.float64, copy=True)
    finite_t = np.isfinite(t_s)
    if not np.any(finite_t):
        raise RuntimeError(f"{path} does not contain any finite time values")
    if float(np.nanmin(t_s[finite_t])) > 100.0:
        t_s = t_s - float(t_s[np.flatnonzero(finite_t)[0]])

    probe_data_C: dict[str, np.ndarray] = {}
    for label, _, exp_candidates, _ in PROBE_SPECS:
        probe_data_C[label] = columns[_first_existing_column(columns, exp_candidates)]

    t_insert_s = _detect_insertion_time_exp(
        t_s,
        probe_data_C["3.0 mm"],
        probe_data_C["6.2 mm"],
        probe_data_C["11.0 mm"],
    )
    t_rel_s = t_s - t_insert_s
    positive_dt = np.diff(t_s[np.isfinite(t_s)])
    positive_dt = positive_dt[positive_dt > 0.0]
    median_dt_s = float(np.median(positive_dt)) if positive_dt.size else math.nan
    return ExperimentalRun(
        label=path.stem,
        t_rel_s=t_rel_s,
        probe_data_C=probe_data_C,
        median_dt_s=median_dt_s,
    )


def _load_simulation_run(path: Path) -> SimulationRun:
    columns = _read_numeric_csv(path)
    time_abs_col = _first_existing_column(columns, ("time_s", "t_s", "time"))
    time_fill_col = _first_existing_column(columns, ("time_since_fill_s", "t_since_fill_s"))
    t_abs_s = columns[time_abs_col]
    t_since_fill_s = columns[time_fill_col]
    finite_since = np.isfinite(t_since_fill_s)
    if not np.any(finite_since):
        raise RuntimeError(f"{path} does not contain any finite time_since_fill_s values")
    i0 = int(np.flatnonzero(finite_since)[0])
    t_fill_s = float(t_abs_s[i0] - t_since_fill_s[i0])
    t_rel_s = np.where(np.isfinite(t_since_fill_s), t_since_fill_s, t_abs_s - t_fill_s)

    probe_data_C: dict[str, np.ndarray] = {}
    for label, sim_candidates, _, _ in PROBE_SPECS:
        probe_data_C[label] = columns[_first_existing_column(columns, sim_candidates)]
    return SimulationRun(t_rel_s=t_rel_s, probe_data_C=probe_data_C)


def _first_downward_crossing_time_s(time_s: np.ndarray, temperature_C: np.ndarray, threshold_C: float) -> float:
    finite = np.isfinite(time_s) & np.isfinite(temperature_C) & (time_s >= 0.0)
    t = np.asarray(time_s[finite], dtype=np.float64)
    y = np.asarray(temperature_C[finite], dtype=np.float64)
    if t.size < 2:
        return math.nan

    above_indices = np.flatnonzero(y > threshold_C)
    if above_indices.size == 0:
        return math.nan
    start = int(above_indices[0])
    if start >= t.size - 1:
        return math.nan

    y0 = y[start:-1]
    y1 = y[start + 1 :]
    transition_indices = np.flatnonzero((y0 > threshold_C) & (y1 <= threshold_C))
    if transition_indices.size == 0:
        exact_indices = np.flatnonzero(y[start:] == threshold_C)
        if exact_indices.size == 0:
            return math.nan
        return float(t[start + int(exact_indices[0])])

    idx = start + int(transition_indices[0])
    t0 = float(t[idx])
    t1 = float(t[idx + 1])
    temp0 = float(y[idx])
    temp1 = float(y[idx + 1])
    if temp1 == temp0:
        return t1
    fraction = (threshold_C - temp0) / (temp1 - temp0)
    return float(t0 + fraction * (t1 - t0))


def _panel_plot_xmax(simulation_run: SimulationRun, *, pre_s: float) -> float:
    crossings_s = [
        _first_downward_crossing_time_s(simulation_run.t_rel_s, simulation_run.probe_data_C[label], 0.0)
        for label, _, _, _ in PROBE_SPECS
    ]
    finite_crossings_s = [value for value in crossings_s if math.isfinite(value)]
    if not finite_crossings_s:
        finite_times_s = simulation_run.t_rel_s[np.isfinite(simulation_run.t_rel_s) & (simulation_run.t_rel_s >= 0.0)]
        if finite_times_s.size == 0:
            return float(DEFAULT_MAX_PLOT_S)
        return float(finite_times_s[-1] + float(pre_s))
    return float(max(finite_crossings_s) + float(pre_s))


def _read_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        if {"parameter", "value"} <= set(reader.fieldnames):
            return {str(row["parameter"]): str(row["value"]) for row in reader}
    return {}


def _simulation_style_values(target_C: float, metadata: dict[str, str]) -> tuple[float, float, float]:
    plan = build_target_plan(target_C)
    ambient_C = float(metadata.get("T_ambient_C", plan.ambient_C))
    fill_C = float(metadata.get("T_fill_C", 12.5))
    h_out_W_m2K = float(metadata.get("h_top_W_m2K", 2.0))
    return ambient_C, fill_C, h_out_W_m2K


def _build_statistics(
    experimental_runs: list[ExperimentalRun],
    *,
    pre_s: float,
    max_plot_s: float,
) -> ExperimentStatistics:
    if not experimental_runs:
        raise ValueError("expected at least one experimental run")

    dt_candidates = np.asarray(
        [run.median_dt_s for run in experimental_runs if math.isfinite(run.median_dt_s) and run.median_dt_s > 0.0],
        dtype=np.float64,
    )
    if dt_candidates.size == 0:
        raise RuntimeError("Could not determine a common experimental time step")
    dt_s = float(np.median(dt_candidates))

    start_rel_s = max(
        -float(pre_s),
        max(float(np.nanmin(run.t_rel_s[np.isfinite(run.t_rel_s)])) for run in experimental_runs),
    )
    end_rel_s = min(
        float(max_plot_s) - float(pre_s),
        min(float(np.nanmax(run.t_rel_s[np.isfinite(run.t_rel_s)])) for run in experimental_runs),
    )
    if not math.isfinite(start_rel_s) or not math.isfinite(end_rel_s) or end_rel_s <= start_rel_s:
        raise RuntimeError("Experimental runs do not share a valid common support interval")

    grid_rel_s = np.arange(start_rel_s, end_rel_s + 0.5 * dt_s, dt_s, dtype=np.float64)
    mean_C: dict[str, np.ndarray] = {}
    std_C: dict[str, np.ndarray] = {}
    ddof = 1 if len(experimental_runs) > 1 else 0
    for label, _, _, _ in PROBE_SPECS:
        stacked = []
        for run in experimental_runs:
            finite = np.isfinite(run.t_rel_s) & np.isfinite(run.probe_data_C[label])
            t_rel_s = run.t_rel_s[finite]
            y_C = run.probe_data_C[label][finite]
            order = np.argsort(t_rel_s)
            stacked.append(np.interp(grid_rel_s, t_rel_s[order], y_C[order]))
        stack_arr = np.vstack(stacked)
        mean_C[label] = np.mean(stack_arr, axis=0)
        std_C[label] = np.std(stack_arr, axis=0, ddof=ddof)

    return ExperimentStatistics(
        t_plot_s=grid_rel_s + float(pre_s),
        mean_C=mean_C,
        std_C=std_C,
    )


def _trim_series(
    t_rel_s: np.ndarray,
    values: np.ndarray,
    *,
    pre_s: float,
    max_plot_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(t_rel_s) & np.isfinite(values)
    t_plot_s = t_rel_s[finite] + float(pre_s)
    y = values[finite]
    mask = (t_plot_s >= 0.0) & (t_plot_s <= float(max_plot_s) + 1.0e-12)
    return t_plot_s[mask], y[mask]


def _apply_axes_style(ax, *, y_lims_C: tuple[float, float], max_plot_s: float) -> None:
    ax.set_xlim(0.0, float(max_plot_s))
    ax.set_ylim(*y_lims_C)
    ax.grid(True, alpha=0.22)
    ax.axhline(0.0, color="0.7", lw=0.9, ls=":")


def _subtitle(target_C: float, ambient_C: float, fill_C: float, h_out_W_m2K: float) -> str:
    return (
        f"T_plate = {target_C:.1f} C, Tamb = {ambient_C:.2f} C, "
        f"Tfill = {fill_C:.1f} C, h = {h_out_W_m2K:.1f} W/m2K, inset = 1 mm"
    )


def _plot_legacy_comparison(
    *,
    target_C: float,
    ambient_C: float,
    fill_C: float,
    h_out_W_m2K: float,
    experimental_runs: list[ExperimentalRun],
    simulation_run: SimulationRun,
    out_path: Path,
    pre_s: float,
    max_plot_s: float,
    y_lims_C: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    cmap = plt.get_cmap("tab10")
    probe_alphas = (1.0, 0.65, 0.35)

    for exp_index, exp_run in enumerate(experimental_runs):
        color = cmap(exp_index % 10)
        for probe_index, (label, _, _, _) in enumerate(PROBE_SPECS):
            t_plot_s, values_C = _trim_series(
                exp_run.t_rel_s,
                exp_run.probe_data_C[label],
                pre_s=pre_s,
                max_plot_s=max_plot_s,
            )
            ax.plot(t_plot_s, values_C, color=color, lw=1.8, alpha=probe_alphas[probe_index])

    for probe_index, (label, _, _, _) in enumerate(PROBE_SPECS):
        t_plot_s, values_C = _trim_series(
            simulation_run.t_rel_s,
            simulation_run.probe_data_C[label],
            pre_s=pre_s,
            max_plot_s=max_plot_s,
        )
        ax.plot(t_plot_s, values_C, color="black", lw=2.2, ls="--", alpha=probe_alphas[probe_index])

    _apply_axes_style(ax, y_lims_C=y_lims_C, max_plot_s=max_plot_s)
    ax.set_xlabel("Time (s, insertion at 20 s)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Experimental runs vs calibrated simulation")
    fig.text(0.5, 0.945, _subtitle(target_C, ambient_C, fill_C, h_out_W_m2K), ha="center", fontsize=10, color="0.35")

    exp_handles = [
        Line2D([0], [0], color=cmap(exp_index % 10), lw=2.5, label=exp_run.label)
        for exp_index, exp_run in enumerate(experimental_runs)
    ]
    leg1 = ax.legend(handles=exp_handles, title="Experimental runs", loc="upper right", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)

    probe_handles = [
        Line2D([0], [0], color="0.25", lw=3.0, alpha=probe_alphas[0], label="3.0 mm"),
        Line2D([0], [0], color="0.25", lw=3.0, alpha=probe_alphas[1], label="6.2 mm"),
        Line2D([0], [0], color="0.25", lw=3.0, alpha=probe_alphas[2], label="11.0 mm"),
        Line2D([0], [0], color="black", lw=2.4, ls="--", label="Simulation"),
    ]
    ax.legend(handles=probe_handles, title="Probes / simulation", loc="lower left", fontsize=9, title_fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_statistical_panel(
    *,
    target_C: float,
    ambient_C: float,
    fill_C: float,
    h_out_W_m2K: float,
    experimental_runs: list[ExperimentalRun],
    experimental_stats: ExperimentStatistics,
    simulation_run: SimulationRun,
    out_path: Path,
    pre_s: float,
    max_plot_s: float,
    y_lims_C: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 11.0), sharex=True)
    fig.suptitle("Experimental water-fill runs vs calibrated simulation", y=0.985, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.956, _subtitle(target_C, ambient_C, fill_C, h_out_W_m2K), ha="center", fontsize=10, color="0.35")
    fig.text(
        0.5,
        0.937,
        "Thin gray traces = individual runs; colored line/band = experimental mean +/- 1 sigma; dashed black = simulation",
        ha="center",
        fontsize=9,
        color="0.45",
    )

    legend_handles = [
        Line2D([0], [0], color="0.7", lw=1.2, label="Individual runs"),
        Line2D([0], [0], color=PROBE_SPECS[0][3], lw=2.5, label="Experimental mean"),
        Patch(facecolor=PROBE_SPECS[0][3], edgecolor="none", alpha=0.18, label="Experimental +/-1 sigma"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="Simulation"),
    ]

    for axis_index, (ax, (label, _, _, color)) in enumerate(zip(axes, PROBE_SPECS, strict=True)):
        for exp_run in experimental_runs:
            t_plot_s, values_C = _trim_series(
                exp_run.t_rel_s,
                exp_run.probe_data_C[label],
                pre_s=pre_s,
                max_plot_s=max_plot_s,
            )
            ax.plot(t_plot_s, values_C, color="0.78", lw=0.9, alpha=0.95)

        ax.fill_between(
            experimental_stats.t_plot_s,
            experimental_stats.mean_C[label] - experimental_stats.std_C[label],
            experimental_stats.mean_C[label] + experimental_stats.std_C[label],
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(experimental_stats.t_plot_s, experimental_stats.mean_C[label], color=color, lw=2.4)

        sim_t_plot_s, sim_values_C = _trim_series(
            simulation_run.t_rel_s,
            simulation_run.probe_data_C[label],
            pre_s=pre_s,
            max_plot_s=max_plot_s,
        )
        ax.plot(sim_t_plot_s, sim_values_C, color="black", lw=2.0, ls="--", alpha=0.9)

        _apply_axes_style(ax, y_lims_C=y_lims_C, max_plot_s=max_plot_s)
        ax.set_ylabel("Temperature (C)")
        ax.text(
            0.012,
            0.86,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25"},
        )
        if axis_index == 0:
            ax.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=8.8, frameon=False)

    axes[-1].set_xlabel("Time (s, insertion at 20 s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def generate_calibrated_experiment_vs_simulation_figures(
    *,
    simulations_dir: Path = WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    output_dir: Path = CALIBRATED_FIGURES_DIR,
    experiments_root: Path | None = None,
    targets_C: tuple[float, ...] = ACTIVE_TARGETS_C,
    pre_s: float = DEFAULT_PRE_S,
    max_plot_s: float = DEFAULT_MAX_PLOT_S,
    y_lims_C: tuple[float, float] = DEFAULT_Y_LIMS_C,
) -> tuple[Path, ...]:
    simulations_dir = Path(simulations_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in MANAGED_FIGURE_GLOBS:
        for path in output_dir.glob(pattern):
            path.unlink()

    outputs = discover_simulation_outputs(simulations_dir)
    written_paths: list[Path] = []

    for target_C in targets_C:
        simulation_output = outputs.get(float(target_C))
        if simulation_output is None:
            if float(target_C) in COMPARISON_TARGETS_C:
                raise RuntimeError(f"Missing simulation output for comparison target {target_C:.1f} C in {simulations_dir}")
            print(f"[figures] missing simulation output for {target_C:.1f} C; skipping")
            continue

        exp_paths = experimental_csv_paths(target_C, experiments_root=experiments_root)
        if not exp_paths:
            print(f"[figures] no experimental data for {target_C:.1f} C; skipping comparison plots")
            continue

        metadata = _read_metadata(simulation_output.metadata_csv)
        ambient_C, fill_C, h_out_W_m2K = _simulation_style_values(target_C, metadata)
        legacy_path, panel_path = _managed_output_paths(output_dir, target_C=target_C, fill_C=fill_C)
        experimental_runs = [_load_experimental_run(path) for path in exp_paths]
        simulation_run = _load_simulation_run(simulation_output.probes_csv)
        legacy_max_plot_s = _legacy_plot_xmax(target_C)
        panel_max_plot_s = _panel_plot_xmax(simulation_run, pre_s=pre_s)
        experimental_stats = _build_statistics(experimental_runs, pre_s=pre_s, max_plot_s=panel_max_plot_s)

        _plot_legacy_comparison(
            target_C=target_C,
            ambient_C=ambient_C,
            fill_C=fill_C,
            h_out_W_m2K=h_out_W_m2K,
            experimental_runs=experimental_runs,
            simulation_run=simulation_run,
            out_path=legacy_path,
            pre_s=pre_s,
            max_plot_s=legacy_max_plot_s,
            y_lims_C=y_lims_C,
        )
        _plot_statistical_panel(
            target_C=target_C,
            ambient_C=ambient_C,
            fill_C=fill_C,
            h_out_W_m2K=h_out_W_m2K,
            experimental_runs=experimental_runs,
            experimental_stats=experimental_stats,
            simulation_run=simulation_run,
            out_path=panel_path,
            pre_s=pre_s,
            max_plot_s=panel_max_plot_s,
            y_lims_C=y_lims_C,
        )
        print(f"[figures] wrote {legacy_path}")
        print(f"[figures] wrote {panel_path}")
        written_paths.extend((legacy_path, panel_path))

    return tuple(written_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibrated experiment-vs-simulation comparison figures for the rho(T) suite."
    )
    parser.add_argument("--simulations-dir", type=Path, default=WITH_RHO_TEMPERATURE_DEPENDENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=CALIBRATED_FIGURES_DIR)
    parser.add_argument("--experiments-root", type=Path, default=None)
    parser.add_argument(
        "--targets-c",
        default=",".join(f"{value:g}" for value in ACTIVE_TARGETS_C),
        help="Comma-separated simulation targets to inspect. Targets without experimental data are skipped.",
    )
    parser.add_argument("--pre-s", type=float, default=DEFAULT_PRE_S)
    parser.add_argument("--max-plot-s", type=float, default=DEFAULT_MAX_PLOT_S)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_calibrated_experiment_vs_simulation_figures(
        simulations_dir=Path(args.simulations_dir),
        output_dir=Path(args.output_dir),
        experiments_root=None if args.experiments_root is None else Path(args.experiments_root),
        targets_C=_parse_targets(args.targets_c),
        pre_s=float(args.pre_s),
        max_plot_s=float(args.max_plot_s),
    )


if __name__ == "__main__":
    main()
