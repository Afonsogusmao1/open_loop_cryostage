from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from code_simulation.simulation.cryostage_model import (
    DEFAULT_CRYOSTAGE_PARAMS,
    default_characterization_run_paths,
    load_characterization_run,
    simulate_characterization_run,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
FIG_DIR = BASE_DIR / "figures"
PNG_PATH = FIG_DIR / "Figure_SX_cryostage_dry_cooling_response.png"
PDF_PATH = FIG_DIR / "Figure_SX_cryostage_dry_cooling_response.pdf"
CAPTION_PATH = BASE_DIR / "Figure_SX_cryostage_dry_cooling_response_caption.md"

TARGET_RE = re.compile(r"characterization_min(?P<target>\d+)", re.IGNORECASE)

COLORS = {
    -5.0: "#0072B2",
    -10.0: "#009E73",
    -15.0: "#D55E00",
    -20.0: "#CC79A7",
}


def target_from_path(path: Path) -> float:
    match = TARGET_RE.search(path.parent.name)
    if not match:
        raise ValueError(f"Cannot infer target from {path}")
    return -float(match.group("target"))


def telemetry_sampling_summary(paths: list[Path]) -> tuple[int, float, float]:
    total_rows = 0
    intervals: list[float] = []
    for path in paths:
        run = load_characterization_run(path, active_power_threshold=1.0)
        times = list(run.time_s)
        total_rows += len(times)
        intervals.extend(
            float(t1 - t0)
            for t0, t1 in zip(times[:-1], times[1:], strict=False)
            if t1 > t0
        )
    dt = float(np.median(np.asarray(intervals, dtype=np.float64)))
    return total_rows, dt, 1.0 / dt


def load_grouped_runs():
    grouped: dict[float, list[tuple[Path, object, np.ndarray]]] = defaultdict(list)
    paths = list(default_characterization_run_paths(PROJECT_ROOT))
    for path in paths:
        target = target_from_path(path)
        run = load_characterization_run(path, active_power_threshold=1.0)
        predicted = simulate_characterization_run(run, DEFAULT_CRYOSTAGE_PARAMS)
        grouped[target].append((path, run, predicted))
    return paths, dict(sorted(grouped.items()))


def mean_profile_for_target(runs: list[tuple[Path, object, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_common_t = min(float(run.time_s[-1]) for _, run, _ in runs)
    grid_s = np.arange(0.0, np.floor(max_common_t) + 1.0, 1.0)
    measured = []
    modelled = []
    for _, run, predicted in runs:
        measured.append(np.interp(grid_s, run.time_s, run.T_plate_C))
        modelled.append(np.interp(grid_s, run.time_s, predicted))
    return grid_s, np.mean(np.vstack(measured), axis=0), np.mean(np.vstack(modelled), axis=0)


def make_figure() -> None:
    paths, grouped = load_grouped_runs()
    total_rows, median_dt_s, sampling_hz = telemetry_sampling_summary(paths)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    ax_raw, ax_fit = axes

    for target, runs in grouped.items():
        color = COLORS[target]
        for index, (_, run, _) in enumerate(runs):
            label = f"{target:.0f} °C, n=3" if index == 0 else None
            ax_raw.plot(
                run.time_s / 60.0,
                run.T_plate_C,
                color=color,
                alpha=0.42,
                linewidth=1.0,
                label=label,
            )
        ax_raw.axhline(target, color=color, linewidth=0.7, alpha=0.35, linestyle=":")

        grid_s, measured_mean, modelled_mean = mean_profile_for_target(runs)
        ax_fit.plot(
            grid_s / 60.0,
            measured_mean,
            color=color,
            linewidth=1.8,
            solid_capstyle="round",
        )
        ax_fit.plot(
            grid_s / 60.0,
            modelled_mean,
            color=color,
            linewidth=1.8,
            linestyle=(0, (4, 2)),
            solid_capstyle="round",
        )

    ax_raw.set_title("A  Active dry-cooling telemetry")
    ax_raw.set_xlabel("Time [min]")
    ax_raw.set_ylabel("Plate temperature [°C]")
    ax_raw.set_xlim(left=0)
    ax_raw.set_ylim(-23, 11)
    ax_raw.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.8)
    ax_raw.legend(frameon=False, loc="upper right", handlelength=2.4)

    ax_fit.set_title("B  Setpoint-dependent first-order fit")
    ax_fit.set_xlabel("Time [min]")
    ax_fit.set_ylabel("Plate temperature [°C]")
    ax_fit.set_xlim(left=0)
    ax_fit.set_ylim(-23, 11)
    ax_fit.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.8)

    color_handles = [
        Line2D([0], [0], color=COLORS[target], lw=2.0, label=f"{target:.0f} °C")
        for target in sorted(grouped)
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", lw=1.8, label="Measured mean"),
        Line2D([0], [0], color="#333333", lw=1.8, linestyle=(0, (4, 2)), label="Model mean"),
    ]
    first_legend = ax_fit.legend(handles=color_handles, frameon=False, loc="upper right", title="Setpoint")
    ax_fit.add_artist(first_legend)
    ax_fit.legend(handles=style_handles, frameon=False, loc="lower left")

    tau_text = ", ".join(f"{value:.1f}" for value in DEFAULT_CRYOSTAGE_PARAMS.response_tau_s)
    steady_text = ", ".join(f"{value:.1f}" for value in DEFAULT_CRYOSTAGE_PARAMS.steady_plate_C)
    param_text = (
        r"$\tau(T_{ref})$ [s]" + "\n"
        + tau_text
        + "\n"
        + r"$T_\infty(T_{ref})$ [$^\circ$C]"
        + "\n"
        + steady_text
    )
    ax_fit.text(
        0.98,
        0.05,
        param_text,
        transform=ax_fit.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=450, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    caption = f"""# Figure SX. Cryostage dry-cooling response used for the first-order response model

Suggested caption:

**Figure SX. Dry-cooling characterization of the cryostage plate-temperature response.** (A) Calibrated plate-temperature traces recorded during active dry cooling to fixed reference setpoints of -5, -10, -15, and -20 °C, with three independent runs per setpoint. Each trace is aligned to the first telemetry row with controller power above 1.0. Dotted horizontal lines indicate the corresponding reference setpoints. (B) Mean measured plate-temperature response for each setpoint compared with the calibrated setpoint-dependent first-order response model used in the open-loop trajectory-design workflow. The fitted response-time lookup was \\(\\tau(T_{{ref}})=[{tau_text}]\\) s at \\(T_{{ref}}=[-20,-15,-10,-5]\\,^\\circ\\mathrm{{C}}\\), and the fitted steady plate-temperature lookup was \\(T_\\infty(T_{{ref}})=[{steady_text}]\\,^\\circ\\mathrm{{C}}\\). Telemetry was recorded with a median sampling interval of {median_dt_s:.3f} s, corresponding to approximately {sampling_hz:.1f} Hz.

Suggested in-text mention:

The dry-cooling traces and corresponding first-order model responses used for this fit are shown in Figure SX.

Data included:

- Characterization files: {len(paths)}
- Active telemetry samples retained for plotting: {total_rows}
- Median sampling interval: {median_dt_s:.3f} s
- Effective sampling rate: {sampling_hz:.1f} Hz
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")


def main() -> None:
    make_figure()
    print(PNG_PATH)
    print(PDF_PATH)
    print(CAPTION_PATH)


if __name__ == "__main__":
    main()
