#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


FILE_RE = re.compile(
    r"cryostage_characterization_min(?P<target>\d+)_(?P<replicate>[A-Z]+)\.csv$",
    re.IGNORECASE,
)
REPLICATE_LINESTYLES = {
    "I": "-",
    "II": "--",
    "III": ":",
}


def default_results_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "cryostage_characterization" / "results"


def default_output_path(column: str) -> Path:
    safe_column = re.sub(r"[^A-Za-z0-9_]+", "_", column.strip()) or "temperature"
    return Path(__file__).resolve().parent / "figures" / f"characterization_{safe_column}_vs_time.png"


def parse_filename(path: Path) -> Tuple[float, str]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(
            f"File name does not match expected pattern 'cryostage_characterization_minX_<run>.csv': {path.name}"
        )
    target_c = -float(match.group("target"))
    replicate = match.group("replicate").upper()
    return target_c, replicate


def load_run(path: Path, time_col: str, value_col: str, row_type: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", sep=None, engine="python", on_bad_lines="skip")

    required = [time_col, value_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    if row_type:
        if "row_type" not in df.columns:
            raise ValueError(f"{path.name} does not contain a 'row_type' column")
        df = df[df["row_type"].astype(str).str.lower() == row_type.lower()].copy()

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[time_col, value_col]).sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="first")
    if df.empty:
        raise ValueError(f"{path.name} has no valid rows after filtering")

    t0 = float(df[time_col].iloc[0])
    df = df[[time_col, value_col]].copy()
    df["t_rel_s"] = df[time_col] - t0
    return df


def detect_cooling_start_time(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    smooth_window: int,
    drop_threshold_c: float,
) -> float:
    if smooth_window < 1:
        smooth_window = 1
    if smooth_window % 2 == 0:
        smooth_window += 1

    smooth_values = (
        df[value_col]
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    running_max = pd.Series(smooth_values).cummax().to_numpy()
    drop_from_max = running_max - smooth_values
    onset_candidates = (drop_from_max >= drop_threshold_c).nonzero()[0]
    if len(onset_candidates) == 0:
        return float(df[time_col].iloc[0])
    return float(df[time_col].iloc[int(onset_candidates[0])])


def build_run_index(paths: List[Path]) -> List[Dict[str, object]]:
    runs: List[Dict[str, object]] = []
    for path in paths:
        target_c, replicate = parse_filename(path)
        runs.append(
            {
                "path": path,
                "target_c": target_c,
                "replicate": replicate,
            }
        )
    return sorted(runs, key=lambda item: (item["target_c"], item["replicate"]), reverse=True)


def plot_runs(
    runs: List[Dict[str, object]],
    time_col: str,
    value_col: str,
    row_type: str,
    align_cooling_start: bool,
    cooling_window_points: int,
    cooling_drop_c: float,
    title: str,
    out_path: Path,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
) -> None:
    targets = sorted({float(run["target_c"]) for run in runs}, reverse=True)
    cmap = plt.get_cmap("tab10")
    target_colors = {target: cmap(index % 10) for index, target in enumerate(targets)}

    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.0),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots()

    for run in runs:
        path = Path(run["path"])
        target_c = float(run["target_c"])
        replicate = str(run["replicate"])
        linestyle = REPLICATE_LINESTYLES.get(replicate, "-.")
        color = target_colors[target_c]
        df = load_run(path, time_col=time_col, value_col=value_col, row_type=row_type)
        if align_cooling_start:
            start_time = detect_cooling_start_time(
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
            linewidth=2.0,
        )

    ax.set_title(title)
    ax.set_xlabel("Time since cooling start (s)" if align_cooling_start else "Time (s)")
    ax.set_ylabel(f"{value_col} (degC)")

    if xmax is not None:
        ax.set_xlim(0, xmax)
    else:
        ax.set_xlim(left=0)

    if ymin is not None or ymax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(
            ymin if ymin is not None else current_ymin,
            ymax if ymax is not None else current_ymax,
        )

    target_handles = [
        Line2D([0], [0], color=target_colors[target], linewidth=2.5, label=f"{target:.0f} degC")
        for target in targets
    ]
    replicate_handles = [
        Line2D([0], [0], color="black", linewidth=2.5, linestyle=linestyle, label=f"Run {replicate}")
        for replicate, linestyle in REPLICATE_LINESTYLES.items()
    ]

    legend_targets = ax.legend(
        handles=target_handles,
        title="Target",
        loc="upper right",
    )
    ax.add_artist(legend_targets)
    ax.legend(
        handles=replicate_handles,
        title="Replicate",
        loc="lower right",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay cryostage characterization assays on a single T(t) graph."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir(),
        help="Folder containing cryostage_characterization_min*.csv files.",
    )
    parser.add_argument(
        "--glob",
        default="cryostage_characterization_min*.csv",
        help="Glob pattern used inside --results-dir.",
    )
    parser.add_argument(
        "--column",
        default="T_cal",
        help="Temperature column to plot. Example: T_cal, T3, T7, T12, Tamb.",
    )
    parser.add_argument(
        "--time-col",
        default="panel_t_s",
        help="Time column used for the x axis.",
    )
    parser.add_argument(
        "--row-type",
        default="telemetry",
        help="Row type to keep before plotting. Use an empty string to keep all rows.",
    )
    parser.add_argument(
        "--align-cooling-start",
        action="store_true",
        help="Shift each curve so t=0 is the first sustained temperature decrease.",
    )
    parser.add_argument(
        "--cooling-window-points",
        type=int,
        default=31,
        help="Rolling window size used to smooth the temperature before onset detection.",
    )
    parser.add_argument(
        "--cooling-drop-c",
        type=float,
        default=0.05,
        help="Temperature drop below the running maximum required to mark cooling start.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to scripts/figures/characterization_<column>_vs_time.png",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--xmax", type=float, default=None, help="Optional x axis upper limit in seconds.")
    parser.add_argument("--ymin", type=float, default=None, help="Optional y axis lower limit in degC.")
    parser.add_argument("--ymax", type=float, default=None, help="Optional y axis upper limit in degC.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    paths = sorted(results_dir.glob(args.glob))
    if not paths:
        raise SystemExit(f"No files matched '{args.glob}' inside {results_dir}")

    runs = build_run_index(paths)
    out_path = args.out or default_output_path(args.column)
    title = args.title or f"Cryostage characterization: {args.column}(t)"

    plot_runs(
        runs=runs,
        time_col=args.time_col,
        value_col=args.column,
        row_type=args.row_type,
        align_cooling_start=args.align_cooling_start,
        cooling_window_points=args.cooling_window_points,
        cooling_drop_c=args.cooling_drop_c,
        title=title,
        out_path=out_path,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
    )

    print(f"[OK] Loaded {len(runs)} assays from {results_dir}")
    print(f"[OK] Saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
