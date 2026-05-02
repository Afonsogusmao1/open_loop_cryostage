from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D

from code_simulation.simulation.cryostage_model import DEFAULT_CRYOSTAGE_PARAMS


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "characterization_cryostage"
ORIGINAL_SCRIPT = DATA_DIR / "plot_characterization_assays.py"
RESULTS_GLOB = "characterization_min*/cryostage_characterization_min*.csv"

OUT_DIR = BASE_DIR / "figures"
PNG_PATH = OUT_DIR / "Figure_SX_characterization_T_cal_vs_time_cooling_start_model_overlay.png"
PDF_PATH = OUT_DIR / "Figure_SX_characterization_T_cal_vs_time_cooling_start_model_overlay.pdf"
CAPTION_PATH = BASE_DIR / "Figure_SX_characterization_T_cal_vs_time_cooling_start_model_overlay_caption.md"


def load_original_plot_module() -> Any:
    spec = importlib.util.spec_from_file_location("plot_characterization_assays", ORIGINAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting module from {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_curve_by_target(
    *,
    original_module: Any,
    runs: list[dict[str, object]],
    time_col: str,
    value_col: str,
    row_type: str,
    cooling_window_points: int,
    cooling_drop_c: float,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, float]]] = {}

    for run_index in runs:
        path = Path(run_index["path"])
        target_c = float(run_index["target_c"])

        df = original_module.load_run(
            path,
            time_col=time_col,
            value_col=value_col,
            row_type=row_type,
        )
        start_time_abs = original_module.detect_cooling_start_time(
            df=df,
            time_col=time_col,
            value_col=value_col,
            smooth_window=cooling_window_points,
            drop_threshold_c=cooling_drop_c,
        )
        first_time_abs = float(df[time_col].iloc[0])
        start_time_rel = float(start_time_abs - first_time_abs)
        end_time_rel = float(df[time_col].iloc[-1] - first_time_abs - start_time_rel)
        if end_time_rel <= 0.0:
            continue

        T0_C = float(np.interp(start_time_abs, df[time_col].to_numpy(), df[value_col].to_numpy()))
        grouped.setdefault(target_c, []).append((T0_C, end_time_rel))

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for target_c, initial_and_end in grouped.items():
        T0_C = float(np.median([item[0] for item in initial_and_end]))
        common_end_s = min(float(item[1]) for item in initial_and_end)
        grid_s = np.arange(0.0, np.floor(common_end_s) + 1.0, 1.0)
        steady_C = DEFAULT_CRYOSTAGE_PARAMS.steady_plate_for_reference_C(target_c)
        tau_s = DEFAULT_CRYOSTAGE_PARAMS.response_tau_for_reference_C(target_c)
        model_C = steady_C + (T0_C - steady_C) * np.exp(-grid_s / tau_s)
        curves[target_c] = (grid_s, model_C)
    return curves


def make_figure() -> None:
    original = load_original_plot_module()
    paths = sorted(DATA_DIR.glob(RESULTS_GLOB))
    if not paths:
        raise RuntimeError(f"No characterization files found under {DATA_DIR}")

    runs = original.build_run_index(paths)
    time_col = "panel_t_s"
    value_col = "T_cal"
    row_type = "telemetry"
    cooling_window_points = 31
    cooling_drop_c = 0.05

    targets = sorted({float(run["target_c"]) for run in runs}, reverse=True)
    cmap = plt.get_cmap("tab10")
    target_colors = {target: cmap(index % 10) for index, target in enumerate(targets)}
    model_color = "#003f8c"
    model_path_effects = [pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()]
    model_curves = model_curve_by_target(
        original_module=original,
        runs=runs,
        time_col=time_col,
        value_col=value_col,
        row_type=row_type,
        cooling_window_points=cooling_window_points,
        cooling_drop_c=cooling_drop_c,
    )

    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.0),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots()

    for run in runs:
        path = Path(run["path"])
        target_c = float(run["target_c"])
        replicate = str(run["replicate"])
        linestyle = original.REPLICATE_LINESTYLES.get(replicate, "-.")
        color = target_colors[target_c]
        df = original.load_run(
            path,
            time_col=time_col,
            value_col=value_col,
            row_type=row_type,
        )
        start_time = original.detect_cooling_start_time(
            df=df,
            time_col=time_col,
            value_col=value_col,
            smooth_window=cooling_window_points,
            drop_threshold_c=cooling_drop_c,
        )
        df["t_rel_s"] = df[time_col] - start_time
        df = df[df["t_rel_s"] >= 0].copy()
        if df.empty:
            continue
        ax.plot(
            df["t_rel_s"],
            df[value_col],
            color=color,
            linestyle=linestyle,
            linewidth=1.25,
            zorder=2,
        )

    for target_c in targets:
        if target_c not in model_curves:
            continue
        time_s, model_C = model_curves[target_c]
        ax.plot(
            time_s,
            model_C,
            color=model_color,
            linestyle=(0, (6, 4)),
            linewidth=2.4,
            alpha=0.95,
            zorder=4,
            path_effects=model_path_effects,
        )

    ax.set_xlabel("Time since cooling start (s)")
    ax.set_ylabel("T_cal (degC)")
    ax.set_xlim(left=0)

    target_handles = [
        Line2D([0], [0], color=target_colors[target], linewidth=2.5, label=f"{target:.0f} degC")
        for target in targets
    ]
    replicate_handles = [
        Line2D([0], [0], color="black", linewidth=2.5, linestyle=linestyle, label=f"Run {replicate}")
        for replicate, linestyle in original.REPLICATE_LINESTYLES.items()
    ]
    model_handle = Line2D(
        [0],
        [0],
        color=model_color,
        linewidth=2.4,
        linestyle=(0, (6, 4)),
        label="First-order model",
    )

    legend_targets = ax.legend(
        handles=target_handles,
        title="Target",
        loc="upper right",
    )
    ax.add_artist(legend_targets)
    legend_model = ax.legend(
        handles=[model_handle],
        loc="lower right",
        bbox_to_anchor=(0.875, 0.02),
        borderaxespad=0.0,
    )
    ax.add_artist(legend_model)
    ax.legend(
        handles=replicate_handles,
        title="Replicate",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        borderaxespad=0.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    CAPTION_PATH.write_text(
        "\n".join(
            [
                "# Figure SX. Cryostage dry-cooling response with first-order model overlay",
                "",
                "Suggested caption:",
                "",
                "**Figure SX. Dry-cooling characterization of the cryostage plate-temperature response.** "
                "Calibrated plate-temperature traces recorded during dry cooling to fixed reference "
                "setpoints of -5, -10, -15, and -20 degC, with three independent runs per setpoint. "
                "The dark-blue dashed curves show the corresponding first-order plate-temperature response "
                "model used in the open-loop trajectory-design workflow, evaluated from the median "
                "measured plate temperature at the detected cooling start for each setpoint.",
                "",
                "Suggested in-text mention:",
                "",
                "The dry-cooling traces and corresponding first-order model responses used for this fit are shown in Figure SX.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    make_figure()
    print(PNG_PATH)
    print(PDF_PATH)
    print(CAPTION_PATH)


if __name__ == "__main__":
    main()
