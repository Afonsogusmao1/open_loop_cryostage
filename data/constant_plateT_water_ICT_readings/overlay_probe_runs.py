#!/usr/bin/env python3
"""
Overlay cryostage probe curves from multiple CSV runs on a single graph.

Adds optional event-based alignment so each run can be shifted to the first
sharp T3 rise, which is useful when logging starts before the actual run.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np
import pandas as pd

PROBES = ["T3", "T7", "T12"]
AMBIENT = "Tamb"
TIME_COL = "t_rec_s"
ALPHAS = {"T3": 1.00, "T7": 0.70, "T12": 0.40}
LINEWIDTHS = {"T3": 2.6, "T7": 2.6, "T12": 2.6}


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, comment="#", sep=None, engine="python", on_bad_lines="skip")
    except Exception as e1:
        try:
            df = pd.read_csv(path, comment="#", sep=",", engine="python", on_bad_lines="skip")
        except Exception as e2:
            raise ValueError(f"Could not read CSV {path}. Auto-detect error: {e1}. Comma-read error: {e2}")
    required = [TIME_COL, *PROBES, AMBIENT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    # Drop obviously bad negative recorder times. One uploaded file contains a
    # single stray row with t_rec_s < 0 that would otherwise corrupt summaries.
    df = df[df[TIME_COL] >= 0].copy()
    df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="first")
    return df


def extract_run_label(path: str) -> str:
    name = Path(path).stem
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else name


def first_downward_zero_crossing(time_s: np.ndarray, temp_c: np.ndarray) -> float:
    if len(time_s) < 2:
        return math.nan
    for i in range(len(temp_c) - 1):
        t0, t1 = time_s[i], time_s[i + 1]
        y0, y1 = temp_c[i], temp_c[i + 1]
        if np.isnan(y0) or np.isnan(y1) or np.isnan(t0) or np.isnan(t1):
            continue
        if y0 > 0 and y1 <= 0:
            if y1 == y0:
                return float(t1)
            frac = (0.0 - y0) / (y1 - y0)
            return float(t0 + frac * (t1 - t0))
    return math.nan


def detect_onset_time(df: pd.DataFrame, probe: str = "T3", slope_threshold_c_per_s: float = 20.0) -> float:
    """
    Detect the first sharp rise in the chosen probe.
    This is a good proxy for the real experimental start when recording begins early.
    """
    t = df[TIME_COL].to_numpy(dtype=float)
    y = df[probe].to_numpy(dtype=float)
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    dy = np.diff(y)
    valid = dt > 0
    slope = np.full_like(dy, np.nan, dtype=float)
    slope[valid] = dy[valid] / dt[valid]
    idx = np.where(slope > slope_threshold_c_per_s)[0]
    if len(idx) == 0:
        return 0.0
    return float(t[idx[0]])


def summarize_runs(paths: List[str], align_onset: bool = False, pre_onset_seconds: float = 5.0) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    """
    CSV summary times are always referenced to the estimated fill/insertion moment.

    This is intentionally independent from the plotting shift used to show a few
    seconds before filling on the graph. Therefore:
    - plot time may be t_raw - onset + pre_onset_seconds
    - CSV crossing time is always t_raw - onset
    """
    per_run = []
    for path in paths:
        df = load_csv(path)
        onset_s = detect_onset_time(df, probe="T3")
        row = {
            "file": Path(path).name,
            "run_label": extract_run_label(path),
            "onset_T3_raw_s": onset_s,
            "start_T3_C": float(df["T3"].iloc[0]),
            "start_T7_C": float(df["T7"].iloc[0]),
            "start_T12_C": float(df["T12"].iloc[0]),
            "start_Tamb_C": float(df["Tamb"].iloc[0]),
        }
        t_after_fill = df[TIME_COL].to_numpy(dtype=float) - onset_s
        for probe in PROBES:
            row[f"crossing_{probe}_s"] = first_downward_zero_crossing(
                t_after_fill,
                df[probe].to_numpy(dtype=float),
            )
        per_run.append(row)

    metrics = {
        "onset_T3_raw_s": [r["onset_T3_raw_s"] for r in per_run],
        "crossing_T3_s": [r["crossing_T3_s"] for r in per_run],
        "crossing_T7_s": [r["crossing_T7_s"] for r in per_run],
        "crossing_T12_s": [r["crossing_T12_s"] for r in per_run],
        "start_T3_C": [r["start_T3_C"] for r in per_run],
        "start_T7_C": [r["start_T7_C"] for r in per_run],
        "start_T12_C": [r["start_T12_C"] for r in per_run],
        "start_Tamb_C": [r["start_Tamb_C"] for r in per_run],
    }
    overall = {}
    for key, values in metrics.items():
        arr = np.array(values, dtype=float)
        overall[key] = {
            "mean": float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else math.nan,
            "std": float(np.nanstd(arr, ddof=1)) if np.sum(~np.isnan(arr)) > 1 else math.nan,
            "n": int(np.sum(~np.isnan(arr))),
        }
    return per_run, overall


def write_per_run_csv(rows: List[Dict], out_path: str) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_overall_csv(overall: Dict[str, Dict[str, float]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "n"])
        for metric, stats in overall.items():
            writer.writerow([metric, stats["mean"], stats["std"], stats["n"]])


def add_custom_probe_legend(ax, paths: List[str], cmap) -> None:
    rows = []
    for i, path in enumerate(paths):
        color = cmap(i % 10)
        label = extract_run_label(path)
        row = HPacker(
            children=[
                TextArea(f"{label}   ", textprops=dict(color="black", fontsize=10.5)),
                TextArea("T3", textprops=dict(color=to_rgba(color, ALPHAS["T3"]), fontsize=10.5, fontweight="bold")),
                TextArea("   ", textprops=dict(fontsize=10.5)),
                TextArea("T7", textprops=dict(color=to_rgba(color, ALPHAS["T7"]), fontsize=10.5, fontweight="bold")),
                TextArea("   ", textprops=dict(fontsize=10.5)),
                TextArea("T12", textprops=dict(color=to_rgba(color, ALPHAS["T12"]), fontsize=10.5, fontweight="bold")),
            ],
            align="baseline",
            pad=0,
            sep=0,
        )
        rows.append(row)

    legend_box = VPacker(children=rows, align="left", pad=0, sep=4)
    anchored = AnchoredOffsetbox(
        loc="upper right",
        child=legend_box,
        pad=0.2,
        frameon=True,
        bbox_to_anchor=(0.995, 0.995),
        bbox_transform=ax.transAxes,
        borderpad=0.5,
    )
    ax.add_artist(anchored)


def plot_overlay(paths: List[str], out_png: str, xmax: Optional[float], ymin: Optional[float], ymax: Optional[float], title: Optional[str], align_onset: bool, pre_onset_seconds: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap("tab10")

    for i, path in enumerate(paths):
        color = cmap(i % 10)
        df = load_csv(path)
        t0 = detect_onset_time(df, probe="T3") if align_onset else 0.0
        t_shift = pre_onset_seconds if align_onset else 0.0
        t = df[TIME_COL].to_numpy(dtype=float) - t0 + t_shift

        for probe in PROBES:
            ax.plot(
                t,
                df[probe].to_numpy(dtype=float),
                color=color,
                alpha=ALPHAS[probe],
                linewidth=LINEWIDTHS[probe],
                solid_capstyle="round",
            )

    ax.axhline(0.0, color="gray", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Time (s)" if not align_onset else f"Time (s, insertion at {pre_onset_seconds:g} s)")
    ax.set_ylabel("Temperature (°C)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    if ymin is not None or ymax is not None:
        current = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else current[0], ymax if ymax is not None else current[1])

    add_custom_probe_legend(ax, paths, cmap)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help="Glob pattern for CSV files")
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--out-prefix", type=str, default="overlay_alpha")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--align-onset", action="store_true", help="Shift each run so the first sharp T3 rise is placed at the chosen pre-onset time")
    p.add_argument("--pre-onset-seconds", type=float, default=5.0, help="Seconds to keep before insertion when --align-onset is used")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = outdir / f"{args.out_prefix}.png"
    out_per_run = outdir / f"{args.out_prefix}_per_run_summary.csv"
    out_overall = outdir / f"{args.out_prefix}_overall_summary.csv"

    per_run, overall = summarize_runs(paths, align_onset=args.align_onset, pre_onset_seconds=args.pre_onset_seconds)
    plot_overlay(paths, str(out_png), args.xmax, args.ymin, args.ymax, args.title, align_onset=args.align_onset, pre_onset_seconds=args.pre_onset_seconds)
    write_per_run_csv(per_run, str(out_per_run))
    write_overall_csv(overall, str(out_overall))

    print(f"Saved: {out_png}")
    print(f"Saved: {out_per_run}")
    print(f"Saved: {out_overall}")


if __name__ == "__main__":
    main()
