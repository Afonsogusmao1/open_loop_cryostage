#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str)
    ap.add_argument("--time-col", default="t_rec_s")
    ap.add_argument("--out", default="fig_all_temps.png")
    ap.add_argument("--also-svg", action="store_true")
    ap.add_argument("--plot-power", action="store_true")
    ap.add_argument("--power-col", default="power")
    ap.add_argument("--plot-setpoint", action="store_true")
    ap.add_argument("--setpoint-col", default="set")
    ap.add_argument("--cols", nargs="*", default=["T_cal", "T3", "T7", "T12", "Tamb"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns:
        raise SystemExit(f"Missing time column '{args.time_col}'. Columns: {list(df.columns)}")

    t = df[args.time_col].to_numpy(float)
    t = t - t[0]  # force 0 start even if you pass t_s by accident

    plt.rcParams.update({
        "figure.figsize": (9.0, 4.5),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")

    # temperatures
    for c in args.cols:
        if c in df.columns:
            y = df[c].to_numpy(float)
            ax.plot(t, y, label=c)
        else:
            print(f"[WARN] Column not found, skipped: {c}")

    # setpoint
    if args.plot_setpoint and (args.setpoint_col in df.columns):
        sp = df[args.setpoint_col].to_numpy(float)
        ax.plot(t, sp, linestyle="--", linewidth=1.2, label="setpoint")

    # power on secondary axis
    if args.plot_power and (args.power_col in df.columns):
        ax2 = ax.twinx()
        p = df[args.power_col].to_numpy(float)
        ax2.plot(t, p, linewidth=1.0, alpha=0.6, label="power (%)")
        ax2.set_ylabel("Power (%)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax.legend(loc="best")

    out = Path(args.out)
    fig.savefig(out)
    if args.also_svg:
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()