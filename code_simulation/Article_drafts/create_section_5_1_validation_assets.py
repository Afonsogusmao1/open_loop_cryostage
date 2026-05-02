from __future__ import annotations

import csv
import html
import math
import struct
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from code_simulation.verification.calibrated.common import (
    WITH_RHO_TEMPERATURE_DEPENDENT_DIR,
    discover_simulation_outputs,
    experimental_csv_paths,
    target_filename_tag,
)
from code_simulation.verification.calibrated.experiment_vs_simulation import (
    DEFAULT_PRE_S,
    DEFAULT_Y_LIMS_C,
    PROBE_SPECS,
    _build_statistics,
    _first_downward_crossing_time_s,
    _load_experimental_run,
    _load_simulation_run,
    _panel_plot_xmax,
    _plot_statistical_panel,
    _read_metadata,
    _simulation_style_values,
    _trim_series,
)


ARTICLE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ARTICLE_DIR / "figures" / "section_5_1_validation"
DOCX_PATH = ARTICLE_DIR / "section_5_1_validation_experiments.docx"
CROSSING_CSV_PATH = FIGURE_DIR / "section_5_1_zero_crossing_times.csv"
ERROR_CSV_PATH = FIGURE_DIR / "section_5_1_temperature_history_errors.csv"
TARGETS_C = (-10.0, -15.0, -20.0)
THREE_CONDITION_PANEL_Y_LIMS_C = DEFAULT_Y_LIMS_C
EMU_PER_INCH = 914400


@dataclass(frozen=True)
class TargetData:
    target_C: float
    experimental_paths: tuple[Path, ...]
    simulation_probes_path: Path
    simulation_metadata_path: Path | None
    experimental_runs: list
    simulation_run: object
    ambient_C: float
    fill_C: float
    h_out_W_m2K: float


@dataclass(frozen=True)
class CrossingRow:
    target_C: float
    probe_label: str
    probe_color: str
    exp_crossings_s: tuple[float, ...]
    exp_mean_s: float
    exp_sd_s: float
    sim_s: float
    sim_minus_exp_s: float
    sim_minus_exp_pct: float


@dataclass(frozen=True)
class ErrorRow:
    target_C: float
    probe_label: str
    rmse_C: float
    mae_C: float
    bias_C: float
    max_abs_C: float
    comparison_end_s: float


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ARTICLE_DIR.parent.parent.resolve()))
    except ValueError:
        return str(path)


def _target_label(target_C: float) -> str:
    return f"{target_C:.0f} deg C"


def _figure_stem_for_target(target_C: float) -> str:
    return f"fig_5_1_temperature_history_{target_filename_tag(target_C)}"


def _three_condition_panel_stem() -> str:
    return "fig_5_1_temperature_history_three_condition_panel"


def _format_float(value: float, digits: int = 1) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _load_target_data() -> tuple[TargetData, ...]:
    simulation_outputs = discover_simulation_outputs(WITH_RHO_TEMPERATURE_DEPENDENT_DIR)
    loaded: list[TargetData] = []
    for target_C in TARGETS_C:
        simulation_output = simulation_outputs.get(float(target_C))
        if simulation_output is None:
            raise RuntimeError(f"Missing rho(T) simulation output for {target_C:g} C")

        experimental_paths = experimental_csv_paths(target_C)
        if len(experimental_paths) != 5:
            raise RuntimeError(
                f"Expected 5 experimental CSV files for {target_C:g} C, found {len(experimental_paths)}"
            )

        metadata = _read_metadata(simulation_output.metadata_csv)
        ambient_C, fill_C, h_out_W_m2K = _simulation_style_values(target_C, metadata)
        loaded.append(
            TargetData(
                target_C=float(target_C),
                experimental_paths=experimental_paths,
                simulation_probes_path=simulation_output.probes_csv,
                simulation_metadata_path=simulation_output.metadata_csv,
                experimental_runs=[_load_experimental_run(path) for path in experimental_paths],
                simulation_run=_load_simulation_run(simulation_output.probes_csv),
                ambient_C=ambient_C,
                fill_C=fill_C,
                h_out_W_m2K=h_out_W_m2K,
            )
        )
    return tuple(loaded)


def _crossing_rows(targets: tuple[TargetData, ...]) -> tuple[CrossingRow, ...]:
    rows: list[CrossingRow] = []
    for target in targets:
        for label, _, _, color in PROBE_SPECS:
            exp_crossings = tuple(
                _first_downward_crossing_time_s(run.t_rel_s, run.probe_data_C[label], 0.0)
                for run in target.experimental_runs
            )
            exp_values = np.asarray(exp_crossings, dtype=np.float64)
            exp_mean = float(np.nanmean(exp_values))
            exp_sd = float(np.nanstd(exp_values, ddof=1))
            sim_s = _first_downward_crossing_time_s(
                target.simulation_run.t_rel_s,
                target.simulation_run.probe_data_C[label],
                0.0,
            )
            delta_s = float(sim_s - exp_mean)
            rows.append(
                CrossingRow(
                    target_C=target.target_C,
                    probe_label=label,
                    probe_color=color,
                    exp_crossings_s=exp_crossings,
                    exp_mean_s=exp_mean,
                    exp_sd_s=exp_sd,
                    sim_s=float(sim_s),
                    sim_minus_exp_s=delta_s,
                    sim_minus_exp_pct=float(100.0 * delta_s / exp_mean),
                )
            )
    return tuple(rows)


def _error_rows(targets: tuple[TargetData, ...]) -> tuple[ErrorRow, ...]:
    rows: list[ErrorRow] = []
    for target in targets:
        panel_max_plot_s = _panel_plot_xmax(target.simulation_run, pre_s=DEFAULT_PRE_S)
        experimental_stats = _build_statistics(
            target.experimental_runs,
            pre_s=DEFAULT_PRE_S,
            max_plot_s=panel_max_plot_s,
        )
        relative_time_s = experimental_stats.t_plot_s - DEFAULT_PRE_S
        valid_grid = np.isfinite(relative_time_s) & (relative_time_s >= 0.0)
        if not np.any(valid_grid):
            raise RuntimeError(f"No valid post-fill comparison interval for {target.target_C:g} C")
        for label, _, _, _ in PROBE_SPECS:
            sim_time = np.asarray(target.simulation_run.t_rel_s, dtype=np.float64)
            sim_temp = np.asarray(target.simulation_run.probe_data_C[label], dtype=np.float64)
            valid_sim = np.isfinite(sim_time) & np.isfinite(sim_temp)
            order = np.argsort(sim_time[valid_sim])
            sim_interp = np.interp(
                relative_time_s[valid_grid],
                sim_time[valid_sim][order],
                sim_temp[valid_sim][order],
            )
            exp_mean = experimental_stats.mean_C[label][valid_grid]
            error = sim_interp - exp_mean
            rows.append(
                ErrorRow(
                    target_C=target.target_C,
                    probe_label=label,
                    rmse_C=float(np.sqrt(np.mean(error**2))),
                    mae_C=float(np.mean(np.abs(error))),
                    bias_C=float(np.mean(error)),
                    max_abs_C=float(np.max(np.abs(error))),
                    comparison_end_s=float(relative_time_s[valid_grid][-1]),
                )
            )
    return tuple(rows)


def _write_crossing_csv(rows: tuple[CrossingRow, ...]) -> None:
    with CROSSING_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target_C",
            "probe_label",
            "probe_color",
            "exp_mean_s",
            "exp_sd_s",
            "sim_s",
            "sim_minus_exp_s",
            "sim_minus_exp_pct",
            "exp_run_1_s",
            "exp_run_2_s",
            "exp_run_3_s",
            "exp_run_4_s",
            "exp_run_5_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "target_C": f"{row.target_C:.6f}",
                "probe_label": row.probe_label,
                "probe_color": row.probe_color,
                "exp_mean_s": f"{row.exp_mean_s:.6f}",
                "exp_sd_s": f"{row.exp_sd_s:.6f}",
                "sim_s": f"{row.sim_s:.6f}",
                "sim_minus_exp_s": f"{row.sim_minus_exp_s:.6f}",
                "sim_minus_exp_pct": f"{row.sim_minus_exp_pct:.6f}",
            }
            for index, value in enumerate(row.exp_crossings_s, start=1):
                payload[f"exp_run_{index}_s"] = f"{value:.6f}"
            writer.writerow(payload)


def _write_error_csv(rows: tuple[ErrorRow, ...]) -> None:
    with ERROR_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "target_C",
            "probe_label",
            "rmse_C",
            "mae_C",
            "bias_C",
            "max_abs_C",
            "comparison_end_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "target_C": f"{row.target_C:.6f}",
                    "probe_label": row.probe_label,
                    "rmse_C": f"{row.rmse_C:.6f}",
                    "mae_C": f"{row.mae_C:.6f}",
                    "bias_C": f"{row.bias_C:.6f}",
                    "max_abs_C": f"{row.max_abs_C:.6f}",
                    "comparison_end_s": f"{row.comparison_end_s:.6f}",
                }
            )


def _plot_temperature_history_panels(targets: tuple[TargetData, ...]) -> tuple[Path, ...]:
    written: list[Path] = []
    for target in targets:
        stem = _figure_stem_for_target(target.target_C)
        panel_max_plot_s = _panel_plot_xmax(target.simulation_run, pre_s=DEFAULT_PRE_S)
        experimental_stats = _build_statistics(
            target.experimental_runs,
            pre_s=DEFAULT_PRE_S,
            max_plot_s=panel_max_plot_s,
        )
        for suffix in ("png", "pdf"):
            out_path = FIGURE_DIR / f"{stem}.{suffix}"
            _plot_statistical_panel(
                target_C=target.target_C,
                ambient_C=target.ambient_C,
                fill_C=target.fill_C,
                h_out_W_m2K=target.h_out_W_m2K,
                experimental_runs=target.experimental_runs,
                experimental_stats=experimental_stats,
                simulation_run=target.simulation_run,
                out_path=out_path,
                pre_s=DEFAULT_PRE_S,
                max_plot_s=panel_max_plot_s,
                y_lims_C=DEFAULT_Y_LIMS_C,
            )
            written.append(out_path)
    return tuple(written)


def _plot_temperature_history_three_condition_panel(targets: tuple[TargetData, ...]) -> tuple[Path, Path]:
    stats_by_target = {}
    xmax_by_target = {}
    for target in targets:
        panel_max_plot_s = _panel_plot_xmax(target.simulation_run, pre_s=DEFAULT_PRE_S)
        xmax_by_target[target.target_C] = panel_max_plot_s
        stats_by_target[target.target_C] = _build_statistics(
            target.experimental_runs,
            pre_s=DEFAULT_PRE_S,
            max_plot_s=panel_max_plot_s,
        )

    with plt.rc_context(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.titlesize": 11,
        }
    ):
        fig, axes = plt.subplots(3, 1, figsize=(11.8, 7.6), sharey=True)
        for ax, target in zip(axes, targets, strict=True):
            max_plot_s = float(xmax_by_target[target.target_C])
            stats = stats_by_target[target.target_C]

            for label, _, _, _color in PROBE_SPECS:
                for exp_run in target.experimental_runs:
                    t_plot_s, values_C = _trim_series(
                        exp_run.t_rel_s,
                        exp_run.probe_data_C[label],
                        pre_s=DEFAULT_PRE_S,
                        max_plot_s=max_plot_s,
                    )
                    ax.plot(t_plot_s, values_C, color="0.78", lw=0.45, alpha=0.22, zorder=1)

            for label, _, _, color in PROBE_SPECS:
                ax.fill_between(
                    stats.t_plot_s,
                    stats.mean_C[label] - stats.std_C[label],
                    stats.mean_C[label] + stats.std_C[label],
                    color=color,
                    alpha=0.08,
                    linewidth=0.0,
                    zorder=2,
                )
                ax.plot(
                    stats.t_plot_s,
                    stats.mean_C[label],
                    color=color,
                    lw=1.25,
                    label=f"{label} exp. mean",
                    zorder=3,
                )
                sim_t_plot_s, sim_values_C = _trim_series(
                    target.simulation_run.t_rel_s,
                    target.simulation_run.probe_data_C[label],
                    pre_s=DEFAULT_PRE_S,
                    max_plot_s=max_plot_s,
                )
                ax.plot(
                    sim_t_plot_s,
                    sim_values_C,
                    color=color,
                    lw=1.05,
                    ls="--",
                    label=f"{label} simulation",
                    zorder=4,
                )

            ax.set_xlim(0.0, max_plot_s)
            ax.set_ylim(*THREE_CONDITION_PANEL_Y_LIMS_C)
            ax.grid(True, alpha=0.22)
            ax.axhline(0.0, color="0.65", lw=0.7, ls=":")
            ax.set_title(f"Plate {_target_label(target.target_C)}")
            ax.set_axisbelow(True)
            ax.text(
                0.012,
                0.83,
                f"Tamb = {target.ambient_C:.2f} deg C",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="0.35",
                bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "round,pad=0.25"},
            )

        for ax in axes:
            ax.set_ylabel("Temperature (deg C)")
        axes[-1].set_xlabel("Time (s, insertion at 20 s)")
        probe_handles = [
            Line2D([0], [0], color=color, lw=1.4, label=label)
            for label, _, _, color in PROBE_SPECS
        ]
        style_handles = [
            Line2D([0], [0], color="0.35", lw=1.25, label="experimental mean"),
            Patch(facecolor="0.65", edgecolor="none", alpha=0.12, label="+/- 1 SD"),
            Line2D([0], [0], color="0.35", lw=1.05, ls="--", label="simulation"),
            Line2D([0], [0], color="0.78", lw=0.45, label="individual runs"),
        ]
        fig.legend(
            handles=[*probe_handles, *style_handles],
            loc="upper center",
            ncol=7,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.suptitle("Experimental mean vs rho(T) calibrated simulation across constant-plate tests", y=0.995)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.875, hspace=0.34)

        stem = _three_condition_panel_stem()
        png_path = FIGURE_DIR / f"{stem}.png"
        pdf_path = FIGURE_DIR / f"{stem}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return png_path, pdf_path


def _plot_zero_crossing_summary(rows: tuple[CrossingRow, ...]) -> tuple[Path, Path]:
    rows_by_target = {
        target_C: [row for row in rows if float(row.target_C) == float(target_C)]
        for target_C in TARGETS_C
    }
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
        offsets = np.linspace(-0.10, 0.10, 5)
        x_positions = np.arange(len(PROBE_SPECS), dtype=float)

        for ax, target_C in zip(axes, TARGETS_C, strict=True):
            target_rows = rows_by_target[target_C]
            means = np.asarray([row.exp_mean_s for row in target_rows], dtype=np.float64)
            sds = np.asarray([row.exp_sd_s for row in target_rows], dtype=np.float64)
            sims = np.asarray([row.sim_s for row in target_rows], dtype=np.float64)
            colors = [row.probe_color for row in target_rows]

            ax.bar(
                x_positions,
                means,
                yerr=sds,
                capsize=4,
                width=0.62,
                color=colors,
                edgecolor="0.20",
                linewidth=0.7,
                alpha=0.95,
                zorder=2,
            )
            for probe_index, row in enumerate(target_rows):
                ax.scatter(
                    x_positions[probe_index] + offsets,
                    row.exp_crossings_s,
                    s=18,
                    color="black",
                    alpha=0.85,
                    zorder=3,
                )
            ax.scatter(
                x_positions,
                sims,
                marker="D",
                s=42,
                color="black",
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

            max_y = float(np.nanmax(np.concatenate([means + sds, sims]))) * 1.18
            ax.set_ylim(0.0, max_y)
            ax.set_title(f"Plate {_target_label(target_C)}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels([label for label, _, _, _ in PROBE_SPECS])
            ax.set_xlabel("Probe height")
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            if ax is axes[0]:
                ax.set_ylabel("0 deg C crossing time after fill (s)")

        handles = [
            Patch(facecolor="0.65", edgecolor="0.20", label="Experimental mean +/- SD"),
            Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="Individual runs"),
            Line2D([0], [0], marker="D", color="black", lw=0, markersize=6, label="Simulation"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
        fig.suptitle("0 deg C crossing times: experiments vs rho(T) calibrated simulation", y=0.995)
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.77, wspace=0.28)

        png_path = FIGURE_DIR / "fig_5_1_zero_crossing_times.png"
        pdf_path = FIGURE_DIR / "fig_5_1_zero_crossing_times.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return png_path, pdf_path


def _caption_for_temperature_panel(target: TargetData) -> str:
    return (
        f"Experimental and simulated temperature histories for the constant-plate "
        f"{_target_label(target.target_C)} validation experiment. Thin gray lines are the five "
        "experimental water-fill runs, colored lines and shaded bands show the experimental mean "
        "+/- 1 SD at each probe height, and the dashed black line is the calibrated rho(T) "
        f"simulation using Tfill = {target.fill_C:.1f} deg C and h = {target.h_out_W_m2K:.1f} "
        "W m^-2 K^-1."
    )


def _caption_for_three_condition_panel() -> str:
    return (
        "Experimental and simulated temperature histories for the three constant-plate "
        "validation experiments. Each panel corresponds to one plate setpoint and contains "
        "the three thermocouple heights. Thin gray traces show individual experimental runs, "
        "solid colored curves and shaded bands show the experimental mean +/- 1 SD, and "
        "dashed colored curves show the corresponding rho(T) calibrated simulation."
    )


def _caption_for_crossing() -> str:
    return (
        "Comparison of experimentally inferred and simulated 0 deg C crossing times for "
        "the constant-temperature validation experiments. Bars show the mean of five "
        "experimental runs, error bars show 1 SD, black circles show individual runs, "
        "and black diamonds show the corresponding rho(T) simulation."
    )


def _crossing_markdown_table(rows: tuple[CrossingRow, ...]) -> str:
    lines = [
        "| Plate setpoint | Probe | Experimental mean +/- SD (s) | Simulation (s) | Simulation - experiment (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_target_label(row.target_C)} | {row.probe_label} | "
            f"{row.exp_mean_s:.1f} +/- {row.exp_sd_s:.1f} | {row.sim_s:.1f} | "
            f"{row.sim_minus_exp_s:+.1f} |"
        )
    return "\n".join(lines)


def _write_temperature_readme(target: TargetData) -> Path:
    stem = _figure_stem_for_target(target.target_C)
    readme_path = FIGURE_DIR / f"README_{stem}.md"
    experimental_sources = "\n".join(f"- `{_repo_rel(path)}`" for path in target.experimental_paths)
    text = f"""# {stem}

## Files

- `{stem}.png`
- `{stem}.pdf`

## Data Sources

Experimental CSV files:

{experimental_sources}

Simulation CSV file:

- `{_repo_rel(target.simulation_probes_path)}`

Metadata:

- `{_repo_rel(target.simulation_metadata_path) if target.simulation_metadata_path else 'metadata not found'}`

## Method

- Experimental runs were aligned to the fill/insertion event using the same derivative-based onset detection as the calibrated comparison plotting workflow.
- The five aligned experimental runs were interpolated onto a common time base.
- Colored lines show the experimental mean and shaded bands show +/- 1 SD.
- The simulation trace comes from `data/simulations_calibrated/with_rho_temperature_dependent`.
- Time is plotted with the fill/insertion event displayed at 20 s to preserve the pre-fill context.

## Caption

{_caption_for_temperature_panel(target)}

## Interpretation Notes

- This figure is a validation check for the constant-temperature baseline, not an optimized trajectory result.
- The model captures the dominant cooling transient and the ordering of the three probe responses.
- Remaining late-stage mismatch, especially at upper probe heights, should be discussed as a limitation of the conduction-dominated model and not as a failure of the open-loop optimization framework.
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _write_three_condition_panel_readme(targets: tuple[TargetData, ...]) -> Path:
    stem = _three_condition_panel_stem()
    readme_path = FIGURE_DIR / f"README_{stem}.md"
    experimental_sources = []
    simulation_sources = []
    metadata_sources = []
    for target in targets:
        experimental_sources.extend(f"- `{_repo_rel(path)}`" for path in target.experimental_paths)
        simulation_sources.append(f"- `{_repo_rel(target.simulation_probes_path)}`")
        if target.simulation_metadata_path is not None:
            metadata_sources.append(f"- `{_repo_rel(target.simulation_metadata_path)}`")
    text = f"""# {stem}

## Files

- `{stem}.png`
- `{stem}.pdf`

## Data Sources

Experimental CSV files:

{chr(10).join(experimental_sources)}

Simulation CSV files:

{chr(10).join(simulation_sources)}

Metadata:

{chr(10).join(metadata_sources)}

## Method

- This is the main Section 5.1 comparison panel requested for the article.
- The figure contains three vertically stacked subplots, one for each constant plate temperature: -10, -15, and -20 deg C.
- Within each subplot, the three thermocouple heights are plotted together: 3.0, 6.2, and 11.0 mm.
- Experimental runs were aligned to the fill/insertion event and interpolated onto a common time base.
- Solid colored curves show experimental means and translucent bands show +/- 1 SD.
- Dashed curves in the same colors show the corresponding `with_rho_temperature_dependent` calibrated simulation.
- Thin gray traces show individual experimental runs for context.
- Line widths are intentionally thin to match the compact comparison-plot style used in the working draft.

## Caption

{_caption_for_three_condition_panel()}

## Interpretation Notes

- This panel is intended to replace the separate per-temperature/per-thermocouple panels in the main text.
- It makes the temperature-setpoint effect visible in one figure while preserving the three thermocouple comparisons inside each condition.
- The largest systematic mismatch remains in the later/upper-probe response, consistent with the limitations of a conduction-dominated model.
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _write_crossing_readme(targets: tuple[TargetData, ...], rows: tuple[CrossingRow, ...]) -> Path:
    readme_path = FIGURE_DIR / "README_fig_5_1_zero_crossing_times.md"
    experimental_sources = []
    simulation_sources = []
    for target in targets:
        experimental_sources.extend(f"- `{_repo_rel(path)}`" for path in target.experimental_paths)
        simulation_sources.append(f"- `{_repo_rel(target.simulation_probes_path)}`")
    text = f"""# fig_5_1_zero_crossing_times

## Files

- `fig_5_1_zero_crossing_times.png`
- `fig_5_1_zero_crossing_times.pdf`
- `section_5_1_zero_crossing_times.csv`

## Data Sources

Experimental CSV files:

{chr(10).join(experimental_sources)}

Simulation CSV files:

{chr(10).join(simulation_sources)}

## Method

- For each experimental run and simulated probe trace, the reported time is the first downward crossing of 0 deg C after the fill/insertion event.
- Crossing times were linearly interpolated between neighboring samples.
- Bars show mean experimental crossing time for n = 5 runs, error bars show sample SD, black circles show the five individual runs, and black diamonds show the simulation.
- Probe colors match the temperature-history panels: 3.0 mm teal, 6.2 mm blue, and 11.0 mm orange.

## Values

{_crossing_markdown_table(rows)}

## Caption

{_caption_for_crossing()}

## Interpretation Notes

- Agreement is strongest close to the cold plate, where the imposed thermal boundary dominates.
- The simulated crossing tends to occur later than the experimental mean at the upper probes, particularly at 11.0 mm.
- This pattern is consistent with a purely conductive model that does not explicitly resolve liquid-phase natural convection, mixing during and after filling, or local disturbances around thermocouple junctions.
"""
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def _write_figure_readmes(targets: tuple[TargetData, ...], rows: tuple[CrossingRow, ...]) -> tuple[Path, ...]:
    paths = [_write_temperature_readme(target) for target in targets]
    paths.append(_write_three_condition_panel_readme(targets))
    paths.append(_write_crossing_readme(targets, rows))
    return tuple(paths)


def _paragraph_xml(style: str, text: str, *, bold: bool = False, italic: bool = False) -> str:
    escaped = html.escape(text)
    rpr_parts = []
    if bold:
        rpr_parts.append("<w:b/>")
    if italic:
        rpr_parts.append("<w:i/>")
    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>" if rpr_parts else ""
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f"<w:r>{rpr}<w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"
    )


def _table_xml(rows: list[list[str]]) -> str:
    table_rows = []
    for row_index, row in enumerate(rows):
        cells = []
        for cell in row:
            shading = '<w:shd w:fill="EDEDED"/>' if row_index == 0 else ""
            cell_text = html.escape(cell)
            cells.append(
                "<w:tc>"
                f"<w:tcPr>{shading}<w:tcMar>"
                '<w:top w:w="80" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
                '<w:bottom w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
                "</w:tcMar></w:tcPr>"
                "<w:p><w:pPr><w:pStyle w:val=\"Normal\"/></w:pPr>"
                f"<w:r><w:t xml:space=\"preserve\">{cell_text}</w:t></w:r>"
                "</w:p></w:tc>"
            )
        table_rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        + "".join(table_rows)
        + "</w:tbl>"
    )


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG file")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _image_xml(rel_id: str, name: str, doc_pr_id: int, image_path: Path, *, width_in: float = 6.45) -> str:
    width_px, height_px = _png_size(image_path)
    height_in = width_in * float(height_px) / float(width_px)
    cx = int(width_in * EMU_PER_INCH)
    cy = int(height_in * EMU_PER_INCH)
    escaped_name = html.escape(name)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{doc_pr_id}" name="{escaped_name}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{doc_pr_id}" name="{escaped_name}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def _docx_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="150" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="260"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="260" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="220" w:after="100"/><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="80" w:after="200"/></w:pPr>
    <w:rPr><w:i/><w:sz w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="table" w:styleId="TableGrid">
    <w:name w:val="Table Grid"/>
    <w:qFormat/>
    <w:tblPr>
      <w:tblBorders>
        <w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>
        <w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>
      </w:tblBorders>
    </w:tblPr>
  </w:style>
</w:styles>
"""


def _docx_core_xml() -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Section 5.1 validation experiments under constant-temperature protocols</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def _docx_content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def _docx_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Word</Application>
</Properties>
"""


def _docx_root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def _crossing_doc_table(rows: tuple[CrossingRow, ...]) -> list[list[str]]:
    table = [["Plate", "Probe", "Experimental mean +/- SD (s)", "Simulation (s)", "Sim - exp (s)"]]
    for row in rows:
        table.append(
            [
                _target_label(row.target_C),
                row.probe_label,
                f"{row.exp_mean_s:.1f} +/- {row.exp_sd_s:.1f}",
                f"{row.sim_s:.1f}",
                f"{row.sim_minus_exp_s:+.1f}",
            ]
        )
    return table


def _error_doc_table(rows: tuple[ErrorRow, ...]) -> list[list[str]]:
    table = [["Plate", "Probe", "RMSE (deg C)", "MAE (deg C)", "Bias (deg C)"]]
    for row in rows:
        table.append(
            [
                _target_label(row.target_C),
                row.probe_label,
                f"{row.rmse_C:.2f}",
                f"{row.mae_C:.2f}",
                f"{row.bias_C:+.2f}",
            ]
        )
    return table


def _pooled_error_text(rows: tuple[ErrorRow, ...]) -> str:
    mae_values = []
    rmse_values = []
    for row in rows:
        mae_values.append(row.mae_C)
        rmse_values.append(row.rmse_C)
    return (
        f"Across the nine probe/condition comparisons, the mean MAE was "
        f"{float(np.mean(mae_values)):.2f} deg C and the mean RMSE was "
        f"{float(np.mean(rmse_values)):.2f} deg C when each simulation was compared "
        "against the experimental mean over the plotted post-fill interval."
    )


def _docx_document_xml(
    targets: tuple[TargetData, ...],
    crossing_rows: tuple[CrossingRow, ...],
    error_rows: tuple[ErrorRow, ...],
    image_entries: list[tuple[str, str, Path]],
) -> str:
    body_parts: list[str] = []
    body_parts.append(_paragraph_xml("Title", "5.1 Validation experiments under constant-temperature protocols"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            "This subsection evaluates whether the calibrated freezing model reproduces the "
            "thermal response measured during baseline water-fill experiments at constant "
            "cold-plate temperatures. These experiments provide the validation step before "
            "the same model is used for open-loop trajectory design.",
        )
    )

    body_parts.append(_paragraph_xml("Heading1", "Analysis workflow"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            "The validation data comprised five independent water-fill experiments at each "
            "constant plate setpoint: -10, -15, and -20 deg C. The experimental traces were "
            "aligned to the fill/insertion event, interpolated onto a common time base, and "
            "summarized as the mean +/- 1 SD at each embedded thermocouple. The simulated "
            "traces were taken from the `with_rho_temperature_dependent` calibrated suite, "
            "using a fill temperature of 12.5 deg C, an effective external heat-transfer "
            "coefficient of 2 W m^-2 K^-1, and probe positions of 3.0, 6.2, and 11.0 mm "
            "with a 1 mm wall inset. The 0 deg C crossing time was defined as the first "
            "downward crossing after filling and was obtained by linear interpolation between "
            "neighboring samples.",
        )
    )
    body_parts.append(
        _table_xml(
            [
                ["Plate setpoint", "Experimental replicates", "Simulation source", "Tamb used in model"],
                *[
                    [
                        _target_label(target.target_C),
                        str(len(target.experimental_paths)),
                        "with_rho_temperature_dependent",
                        f"{target.ambient_C:.3f} deg C",
                    ]
                    for target in targets
                ],
            ]
        )
    )

    body_parts.append(_paragraph_xml("Heading1", "Temperature-history agreement"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            "The simulated thermal histories reproduce the dominant cooling behavior of the "
            "experiments across all three plate temperatures. In each case, the model captures "
            "the rapid cooling near the cold plate, the ordering of the probe responses with "
            "height, and the gradual late-stage approach toward subzero temperatures. "
            + _pooled_error_text(error_rows),
        )
    )
    body_parts.append(_table_xml(_error_doc_table(error_rows)))

    body_parts.append(_paragraph_xml("Heading1", "0 deg C crossing times"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            "The agreement is strongest at the lowest probe, where the imposed plate boundary "
            "dominates the response. The simulated 3.0 mm crossing differs from the experimental "
            "mean by -1.1 s at -10 deg C, +2.2 s at -15 deg C, and +0.1 s at -20 deg C. "
            "At the 6.2 mm and 11.0 mm probes, the model generally predicts later crossings, "
            "with the largest offsets at the upper probe during the late freezing stage.",
        )
    )
    body_parts.append(_table_xml(_crossing_doc_table(crossing_rows)))

    body_parts.append(_paragraph_xml("Heading1", "Interpretation"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            "The residual mismatch has a physically consistent pattern. As distance from the "
            "plate increases, the measured response is more sensitive to processes that are not "
            "explicitly resolved in the current model, including liquid-phase natural convection, "
            "mixing immediately after filling, possible disturbances around the thermocouple "
            "junctions, and uncertainty in the effective local air boundary condition. The model "
            "should therefore be interpreted as a conduction-dominated predictive framework for "
            "trajectory design, not as a complete thermo-fluid description of the liquid phase. "
            "Within that intended use, the agreement in the near-plate response, the correct "
            "probe ordering, and the bounded upper-probe timing errors support using the model "
            "as the computational plant for the open-loop design study.",
        )
    )

    body_parts.append(_paragraph_xml("Heading1", "Figures"))
    for index, (rel_id, caption, image_path) in enumerate(image_entries, start=1):
        body_parts.append(_image_xml(rel_id, image_path.name, index, image_path))
        body_parts.append(_paragraph_xml("Caption", f"Figure 5.1-{index}. {caption}"))

    body_parts.append(_paragraph_xml("Heading1", "Generated files"))
    body_parts.append(
        _paragraph_xml(
            "Normal",
            f"Figures, READMEs, and CSV summaries are stored in `{_repo_rel(FIGURE_DIR)}`. "
            "The crossing-time summary is stored in `section_5_1_zero_crossing_times.csv`, "
            "and the temperature-history error summary is stored in "
            "`section_5_1_temperature_history_errors.csv`.",
        )
    )

    body = "\n".join(body_parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="900" w:bottom="1440" w:left="900"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def _write_docx(
    targets: tuple[TargetData, ...],
    crossing_rows: tuple[CrossingRow, ...],
    error_rows: tuple[ErrorRow, ...],
    image_paths: tuple[Path, ...],
) -> Path:
    image_entries = [(f"rId{index + 1}", "", image_path) for index, image_path in enumerate(image_paths, start=1)]
    caption_by_name = {
        f"{_three_condition_panel_stem()}.png": _caption_for_three_condition_panel(),
        "fig_5_1_zero_crossing_times.png": _caption_for_crossing(),
    }
    for target in targets:
        caption_by_name[f"{_figure_stem_for_target(target.target_C)}.png"] = _caption_for_temperature_panel(target)
    image_entries = [(rel_id, caption_by_name[image_path.name], image_path) for rel_id, _, image_path in image_entries]

    relationships = [
        f'<Relationship Id="{rel_id}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/image{index}.png"/>'
        for index, (rel_id, _, _) in enumerate(image_entries, start=1)
    ]
    document_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relationships)
        + "</Relationships>\n"
    )
    document_xml = _docx_document_xml(targets, crossing_rows, error_rows, image_entries)

    out_path = DOCX_PATH
    try:
        archive_context = zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED)
    except PermissionError:
        out_path = DOCX_PATH.with_name(f"{DOCX_PATH.stem}_vertical.docx")
        archive_context = zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED)

    with archive_context as archive:
        archive.writestr("[Content_Types].xml", _docx_content_types_xml())
        archive.writestr("_rels/.rels", _docx_root_rels_xml())
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", _docx_styles_xml())
        archive.writestr("docProps/core.xml", _docx_core_xml())
        archive.writestr("docProps/app.xml", _docx_app_xml())
        for index, (_, _, image_path) in enumerate(image_entries, start=1):
            archive.write(image_path, f"word/media/image{index}.png")
    return out_path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    targets = _load_target_data()
    crossing_rows = _crossing_rows(targets)
    error_rows = _error_rows(targets)

    _write_crossing_csv(crossing_rows)
    _write_error_csv(error_rows)
    temperature_paths = _plot_temperature_history_panels(targets)
    three_condition_png, three_condition_pdf = _plot_temperature_history_three_condition_panel(targets)
    crossing_png, crossing_pdf = _plot_zero_crossing_summary(crossing_rows)
    readme_paths = _write_figure_readmes(targets, crossing_rows)

    docx_path = _write_docx(
        targets,
        crossing_rows,
        error_rows,
        (three_condition_png, crossing_png),
    )

    print(f"Wrote {docx_path}")
    print(f"Wrote {CROSSING_CSV_PATH}")
    print(f"Wrote {ERROR_CSV_PATH}")
    for path in (*temperature_paths, three_condition_png, three_condition_pdf, crossing_png, crossing_pdf, *readme_paths):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
