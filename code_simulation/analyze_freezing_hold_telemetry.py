#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLATE_COL = "T_cal"
SETPOINT_COL = "set"
TIME_COL = "t_rec_s"
THERMOCOUPLE_COLS = ("T3", "T7", "T12")


@dataclass(frozen=True)
class HoldAnalysisConfig:
    band_C: float = 0.5
    hold_entry_dwell_s: float = 60.0
    smoothing_window_s: float = 10.0
    rolling_std_window_s: float = 60.0
    drift_window_s: float = 300.0
    support_duration_start_s: float = 300.0
    support_duration_grid_s: float = 60.0
    support_min_runs: int = 2
    support_min_pass_fraction: float = 0.6
    support_min_fraction_in_band: float = 0.9
    support_max_mean_abs_error_C: float = 0.25
    support_max_abs_mean_error_C: float = 0.25
    support_max_abs_drift_C: float = 0.3


@dataclass(frozen=True)
class HoldRun:
    path: Path
    target_C: float
    time_s: np.ndarray
    plate_raw_C: np.ndarray
    plate_smooth_C: np.ndarray
    setpoint_C: np.ndarray
    hold_start_s: float
    hold_start_index: int
    qualified_hold: bool
    sample_dt_s: float

    @property
    def hold_time_s(self) -> np.ndarray:
        return self.time_s[self.hold_start_index :] - self.time_s[self.hold_start_index]

    @property
    def hold_plate_raw_C(self) -> np.ndarray:
        return self.plate_raw_C[self.hold_start_index :]

    @property
    def hold_plate_smooth_C(self) -> np.ndarray:
        return self.plate_smooth_C[self.hold_start_index :]

    @property
    def hold_duration_s(self) -> float:
        if self.hold_time_s.size == 0:
            return 0.0
        return float(self.hold_time_s[-1])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_input_dir() -> Path:
    return repo_root() / "data" / "constant_plateT_water_ICT_readings"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "characterization_constraints" / "stage2_hold_telemetry"


def rolling_window_samples(window_s: float, dt_s: float) -> int:
    return max(1, int(round(float(window_s) / max(float(dt_s), 1e-9))))


def _longest_true_interval_s(time_s: np.ndarray, mask: np.ndarray) -> float:
    if time_s.size == 0 or mask.size == 0:
        return 0.0
    longest_s = 0.0
    start_idx: int | None = None
    for idx, is_true in enumerate(mask):
        if is_true and start_idx is None:
            start_idx = idx
        at_end = idx == mask.size - 1
        if start_idx is not None and ((not is_true) or at_end):
            end_idx = idx if not is_true else idx + 1
            if end_idx - start_idx >= 1:
                longest_s = max(longest_s, float(time_s[end_idx - 1] - time_s[start_idx]))
            start_idx = None
    return float(longest_s)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    return float(np.quantile(finite, q))


def _safe_stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "min": math.nan,
            "median": math.nan,
            "max": math.nan,
            "p10": math.nan,
            "p90": math.nan,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def load_hold_run(path: Path, config: HoldAnalysisConfig) -> HoldRun:
    df = pd.read_csv(path)
    required = [TIME_COL, SETPOINT_COL, PLATE_COL, *THERMOCOUPLE_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[TIME_COL, SETPOINT_COL, PLATE_COL]).copy()
    df = df[df[TIME_COL] >= 0.0].sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="first")
    if df.empty:
        raise ValueError(f"{path} does not contain usable plate telemetry")

    time_s = df[TIME_COL].to_numpy(dtype=np.float64)
    plate_raw_C = df[PLATE_COL].to_numpy(dtype=np.float64)
    setpoint_C = df[SETPOINT_COL].to_numpy(dtype=np.float64)
    dominant_set = pd.Series(setpoint_C).round(6).mode(dropna=True)
    if dominant_set.empty:
        raise ValueError(f"{path} does not contain a valid setpoint")
    target_C = float(dominant_set.iloc[0])
    if time_s.size < 2:
        raise ValueError(f"{path} does not contain enough samples for hold analysis")

    sample_dt_s = float(np.median(np.diff(time_s)))
    smooth_samples = rolling_window_samples(config.smoothing_window_s, sample_dt_s)
    plate_smooth_C = (
        pd.Series(plate_raw_C)
        .rolling(smooth_samples, center=True, min_periods=max(3, smooth_samples // 2))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=np.float64)
    )

    in_band_smooth = np.abs(plate_smooth_C - target_C) <= float(config.band_C)
    hold_start_index = 0
    hold_start_s = float(time_s[0])
    qualified_hold = False
    dwell_j = 0
    for idx in range(time_s.size):
        while dwell_j < time_s.size and time_s[dwell_j] - time_s[idx] < float(config.hold_entry_dwell_s):
            dwell_j += 1
        if dwell_j >= time_s.size:
            break
        if bool(np.all(in_band_smooth[idx:dwell_j])):
            hold_start_index = idx
            hold_start_s = float(time_s[idx])
            qualified_hold = True
            break

    return HoldRun(
        path=path,
        target_C=target_C,
        time_s=time_s,
        plate_raw_C=plate_raw_C,
        plate_smooth_C=plate_smooth_C,
        setpoint_C=setpoint_C,
        hold_start_s=hold_start_s,
        hold_start_index=int(hold_start_index),
        qualified_hold=bool(qualified_hold),
        sample_dt_s=sample_dt_s,
    )


def compute_hold_metrics(
    run: HoldRun,
    *,
    duration_limit_s: float | None,
    config: HoldAnalysisConfig,
) -> dict[str, float]:
    hold_time_s = run.hold_time_s
    if hold_time_s.size == 0:
        raise ValueError(f"{run.path} does not contain a hold interval")

    if duration_limit_s is None:
        use_mask = np.ones_like(hold_time_s, dtype=bool)
    else:
        use_mask = hold_time_s <= float(duration_limit_s) + 1e-12
    if np.count_nonzero(use_mask) < 2:
        raise ValueError(f"{run.path} does not contain enough hold samples for duration_limit_s={duration_limit_s}")

    t_s = hold_time_s[use_mask]
    plate_raw_C = run.hold_plate_raw_C[use_mask]
    plate_smooth_C = run.hold_plate_smooth_C[use_mask]
    error_raw_C = plate_raw_C - run.target_C
    error_smooth_C = plate_smooth_C - run.target_C
    in_band_raw = np.abs(error_raw_C) <= float(config.band_C)
    in_band_smooth = np.abs(error_smooth_C) <= float(config.band_C)

    rolling_std_samples = rolling_window_samples(config.rolling_std_window_s, run.sample_dt_s)
    rolling_std_C = (
        pd.Series(error_raw_C)
        .rolling(rolling_std_samples, min_periods=max(5, rolling_std_samples // 2))
        .std()
        .to_numpy(dtype=np.float64)
    )

    drift_samples = rolling_window_samples(config.drift_window_s, run.sample_dt_s)
    rolling_mean_error_C = (
        pd.Series(error_smooth_C)
        .rolling(drift_samples, min_periods=drift_samples)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    abs_rolling_mean_error_C = np.abs(rolling_mean_error_C)

    return {
        "duration_s": float(t_s[-1]),
        "mean_plate_temperature_C": float(np.mean(plate_raw_C)),
        "mean_signed_error_C": float(np.mean(error_raw_C)),
        "mean_abs_deviation_C": float(np.mean(np.abs(error_raw_C))),
        "fraction_in_band_raw": float(np.mean(in_band_raw)),
        "fraction_in_band_smooth": float(np.mean(in_band_smooth)),
        "longest_continuous_in_band_raw_s": _longest_true_interval_s(t_s, in_band_raw),
        "longest_continuous_in_band_smooth_s": _longest_true_interval_s(t_s, in_band_smooth),
        "rolling_std_60s_p95_C": _safe_quantile(rolling_std_C, 0.95),
        "rolling_std_60s_max_C": _safe_quantile(rolling_std_C, 1.0),
        "rolling_mean_error_300s_abs_p95_C": _safe_quantile(abs_rolling_mean_error_C, 0.95),
        "rolling_mean_error_300s_abs_max_C": _safe_quantile(abs_rolling_mean_error_C, 1.0),
    }


def run_passes_support(metrics: dict[str, float], config: HoldAnalysisConfig) -> bool:
    return bool(
        abs(float(metrics["mean_signed_error_C"])) <= float(config.support_max_abs_mean_error_C)
        and float(metrics["mean_abs_deviation_C"]) <= float(config.support_max_mean_abs_error_C)
        and float(metrics["fraction_in_band_smooth"]) >= float(config.support_min_fraction_in_band)
        and float(metrics["rolling_mean_error_300s_abs_max_C"]) <= float(config.support_max_abs_drift_C)
    )


def summarize_runs(runs: list[HoldRun], config: HoldAnalysisConfig) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], pd.DataFrame]:
    per_run_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    target_summaries: dict[float, dict[str, Any]] = {}

    for run in runs:
        full_metrics = compute_hold_metrics(run, duration_limit_s=None, config=config)
        full_metrics["run_passes_full_duration_support_rule"] = bool(run_passes_support(full_metrics, config))
        full_metrics["source_file"] = str(run.path.resolve())
        full_metrics["source_filename"] = run.path.name
        full_metrics["observed_target_C"] = float(run.target_C)
        full_metrics["qualified_hold"] = bool(run.qualified_hold)
        full_metrics["hold_start_s"] = float(run.hold_start_s)
        full_metrics["hold_entry_dwell_s"] = float(config.hold_entry_dwell_s)
        per_run_rows.append(full_metrics)

    per_run_df = pd.DataFrame(per_run_rows).sort_values(["observed_target_C", "source_filename"]).reset_index(drop=True)

    for target_C, group_df in per_run_df.groupby("observed_target_C", sort=True):
        target_runs = [run for run in runs if abs(run.target_C - float(target_C)) <= 1e-12]
        qualified_runs = [run for run in target_runs if run.qualified_hold]
        qualified_durations = [run.hold_duration_s for run in qualified_runs]
        full_group = group_df.copy().reset_index(drop=True)
        max_grid_duration_s = (
            math.floor(max(qualified_durations) / float(config.support_duration_grid_s)) * float(config.support_duration_grid_s)
            if qualified_durations
            else 0.0
        )
        supported_duration_s = 0.0

        if qualified_runs and max_grid_duration_s >= float(config.support_duration_start_s):
            duration_grid_s = np.arange(
                float(config.support_duration_start_s),
                max_grid_duration_s + 0.5 * float(config.support_duration_grid_s),
                float(config.support_duration_grid_s),
                dtype=np.float64,
            )
            for duration_s in duration_grid_s:
                available_runs = [run for run in qualified_runs if run.hold_duration_s >= duration_s - 1e-12]
                window_metrics: list[dict[str, Any]] = []
                for run in available_runs:
                    metrics = compute_hold_metrics(run, duration_limit_s=float(duration_s), config=config)
                    metrics["source_filename"] = run.path.name
                    metrics["passes_support"] = bool(run_passes_support(metrics, config))
                    window_metrics.append(metrics)

                available_count = len(window_metrics)
                pass_count = int(sum(bool(row["passes_support"]) for row in window_metrics))
                pass_fraction = float(pass_count / available_count) if available_count else 0.0
                is_supported = bool(
                    available_count >= int(config.support_min_runs)
                    and pass_fraction >= float(config.support_min_pass_fraction)
                )
                if is_supported:
                    supported_duration_s = float(duration_s)

                support_rows.append(
                    {
                        "observed_target_C": float(target_C),
                        "hold_duration_s": float(duration_s),
                        "available_run_count": int(available_count),
                        "pass_run_count": int(pass_count),
                        "pass_fraction": float(pass_fraction),
                        "is_supported": bool(is_supported),
                        "median_fraction_in_band_smooth": _safe_stat(
                            [float(row["fraction_in_band_smooth"]) for row in window_metrics]
                        )["median"],
                        "median_mean_abs_deviation_C": _safe_stat(
                            [float(row["mean_abs_deviation_C"]) for row in window_metrics]
                        )["median"],
                        "median_abs_mean_signed_error_C": _safe_stat(
                            [abs(float(row["mean_signed_error_C"])) for row in window_metrics]
                        )["median"],
                        "median_abs_rolling_mean_error_300s_max_C": _safe_stat(
                            [float(row["rolling_mean_error_300s_abs_max_C"]) for row in window_metrics]
                        )["median"],
                    }
                )

        target_summaries[float(target_C)] = {
            "observed_target_C": float(target_C),
            "source_files": [str(run.path.resolve()) for run in target_runs],
            "qualified_hold_run_count": int(sum(bool(run.qualified_hold) for run in target_runs)),
            "total_run_count": int(len(target_runs)),
            "conservative_supported_hold_duration_s": float(supported_duration_s),
            "qualified_hold_duration_s_stats": _safe_stat(qualified_durations),
            "full_run_metrics": {
                "mean_signed_error_C": _safe_stat(full_group["mean_signed_error_C"].tolist()),
                "mean_abs_deviation_C": _safe_stat(full_group["mean_abs_deviation_C"].tolist()),
                "fraction_in_band_raw": _safe_stat(full_group["fraction_in_band_raw"].tolist()),
                "fraction_in_band_smooth": _safe_stat(full_group["fraction_in_band_smooth"].tolist()),
                "longest_continuous_in_band_smooth_s": _safe_stat(
                    full_group["longest_continuous_in_band_smooth_s"].tolist()
                ),
                "rolling_std_60s_p95_C": _safe_stat(full_group["rolling_std_60s_p95_C"].tolist()),
                "rolling_mean_error_300s_abs_max_C": _safe_stat(
                    full_group["rolling_mean_error_300s_abs_max_C"].tolist()
                ),
            },
            "support_rule_thresholds": {
                "support_min_runs": int(config.support_min_runs),
                "support_min_pass_fraction": float(config.support_min_pass_fraction),
                "support_min_fraction_in_band_smooth": float(config.support_min_fraction_in_band),
                "support_max_mean_abs_error_C": float(config.support_max_mean_abs_error_C),
                "support_max_abs_mean_error_C": float(config.support_max_abs_mean_error_C),
                "support_max_abs_drift_C": float(config.support_max_abs_drift_C),
            },
            "full_duration_failed_runs": [
                str(row["source_filename"])
                for _, row in full_group.iterrows()
                if not bool(row["run_passes_full_duration_support_rule"])
            ],
        }

    support_df = pd.DataFrame(support_rows).sort_values(["observed_target_C", "hold_duration_s"]).reset_index(drop=True)
    return per_run_df, target_summaries, support_df


def plot_hold_overlays(runs: list[HoldRun], out_path: Path, config: HoldAnalysisConfig) -> None:
    targets = sorted({float(run.target_C) for run in runs})
    fig, axes = plt.subplots(len(targets), 1, figsize=(10.5, 3.4 * len(targets)), constrained_layout=True)
    if len(targets) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for ax, target_C in zip(axes, targets):
        target_runs = [run for run in runs if abs(run.target_C - target_C) <= 1e-12]
        for idx, run in enumerate(target_runs):
            color = cmap(idx % 10)
            ax.plot(
                run.hold_time_s,
                run.hold_plate_smooth_C,
                color=color,
                linewidth=1.8,
                label=run.path.stem,
            )
        ax.axhline(target_C, color="black", linewidth=1.2, linestyle="--")
        ax.axhline(target_C + float(config.band_C), color="gray", linewidth=0.9, linestyle=":")
        ax.axhline(target_C - float(config.band_C), color="gray", linewidth=0.9, linestyle=":")
        ax.set_title(f"Observed hold telemetry at {target_C:.0f} C")
        ax.set_xlabel("Time since qualified hold start [s]")
        ax.set_ylabel("Plate temperature T_cal [C]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Experimental plate holds from freezing runs", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hold_support(per_run_df: pd.DataFrame, support_df: pd.DataFrame, out_path: Path) -> None:
    targets = sorted(float(value) for value in per_run_df["observed_target_C"].unique())
    x_positions = np.arange(len(targets), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    supported = []
    for target_C in targets:
        rows = support_df[support_df["observed_target_C"] == target_C]
        supported.append(float(rows.loc[rows["is_supported"], "hold_duration_s"].max()) if not rows.empty else 0.0)
    axes[0].bar(x_positions, supported, color="#4477AA")
    axes[0].set_xticks(x_positions, [f"{target:.0f} C" for target in targets])
    axes[0].set_ylabel("Supported hold duration [s]")
    axes[0].set_title("Conservative empirical hold support")
    axes[0].grid(True, axis="y", alpha=0.25)

    for idx, target_C in enumerate(targets):
        group = per_run_df[per_run_df["observed_target_C"] == target_C]
        axes[1].scatter(
            np.full(len(group), x_positions[idx], dtype=np.float64),
            group["fraction_in_band_smooth"].to_numpy(dtype=np.float64),
            color="#228833",
            s=45,
        )
    axes[1].set_xticks(x_positions, [f"{target:.0f} C" for target in targets])
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Fraction in ±0.5 C band")
    axes[1].set_title("Full-run hold quality across repeats")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown_summary(
    out_path: Path,
    *,
    input_paths: list[Path],
    target_summaries: dict[float, dict[str, Any]],
    config: HoldAnalysisConfig,
    figures_dir: Path,
) -> None:
    lines = [
        "# Long-Duration Plate Hold Characterization",
        "",
        "## Source data",
        "- Evidence type: hardware-side cryostage plate telemetry from freezing experiments, using `T_cal` as the plate temperature measurement.",
        "- The same experimental CSV files also contain `set` (commanded target) plus sample thermocouples `T3`, `T7`, and `T12`.",
        "- Sample thermocouples are retained as context, but they are not the primary evidence for hardware admissibility.",
        "- Files analyzed:",
    ]
    for path in input_paths:
        lines.append(f"  - `{path}`")
    lines.extend(
        [
            "",
            "## Hold-period definition",
            (
                f"- For each freezing-run CSV, the hold period begins at the first timestamp where a "
                f"{config.smoothing_window_s:.0f} s median-smoothed `T_cal` trace stays inside `setpoint ± {config.band_C:.1f} C` "
                f"for a continuous `{config.hold_entry_dwell_s:.0f} s` dwell."
            ),
            "- Hold-quality metrics are then computed from that qualified hold start to the end of the recording.",
            (
                f"- Long-duration support is evaluated on hold durations from `{config.support_duration_start_s:.0f} s` upward in "
                f"`{config.support_duration_grid_s:.0f} s` increments."
            ),
            "",
            "## Support rule",
            (
                f"- A target-duration pair is marked supported when at least `{config.support_min_runs}` runs remain available at that duration "
                f"and at least `{100.0 * config.support_min_pass_fraction:.0f}%` of those runs satisfy all of:"
            ),
            f"  - smoothed fraction in band >= {config.support_min_fraction_in_band:.2f}",
            f"  - mean absolute plate error <= {config.support_max_mean_abs_error_C:.2f} C",
            f"  - absolute mean signed plate error <= {config.support_max_abs_mean_error_C:.2f} C",
            f"  - maximum absolute {config.drift_window_s:.0f} s rolling-mean plate error <= {config.support_max_abs_drift_C:.2f} C",
            "",
            "## Per-target summary",
        ]
    )

    for target_C in sorted(target_summaries):
        summary = target_summaries[target_C]
        lines.append(
            f"- `{target_C:.0f} C`: supported hold duration = {summary['conservative_supported_hold_duration_s']:.0f} s; "
            f"qualified runs = {summary['qualified_hold_run_count']} / {summary['total_run_count']}; "
            f"full-run smooth in-band median = {summary['full_run_metrics']['fraction_in_band_smooth']['median']:.3f}; "
            f"worst full-run drift = {summary['full_run_metrics']['rolling_mean_error_300s_abs_max_C']['max']:.3f} C."
        )
        if summary["full_duration_failed_runs"]:
            lines.append("  - Full-duration failures: " + ", ".join(f"`{name}`" for name in summary["full_duration_failed_runs"]))

    lines.extend(
        [
            "",
            "## Figures",
            f"- Hold overlays: `{figures_dir / 'plate_hold_overlays_by_target.png'}`",
            f"- Hold support summary: `{figures_dir / 'hold_support_summary.png'}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract long-duration plate hold support from freezing-run cryostage telemetry."
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir()),
        help="Directory containing the freezing-run CSV files grouped by setpoint.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_output_dir()),
        help="Directory where hold characterization artifacts are written.",
    )
    args = parser.parse_args()

    config = HoldAnalysisConfig()
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("min*/*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No freezing-run CSV files found under {input_dir}")

    runs = [load_hold_run(path, config) for path in input_paths]
    per_run_df, target_summaries, support_df = summarize_runs(runs, config)

    per_run_csv_path = out_dir / "per_run_hold_metrics.csv"
    support_csv_path = out_dir / "hold_duration_support_grid.csv"
    summary_json_path = out_dir / "hold_summary.json"
    summary_md_path = out_dir / "hold_summary.md"

    per_run_df.to_csv(per_run_csv_path, index=False)
    support_df.to_csv(support_csv_path, index=False)

    summary_payload = {
        "source_files": [str(path.resolve()) for path in input_paths],
        "evidence_role": {
            "primary_hardware_hold_evidence": "hardware-side cryostage plate telemetry from T_cal in freezing-run CSV files",
            "context_only": "sample thermocouple columns T3/T7/T12 inside the same experimental CSV files",
        },
        "config": {
            "band_C": float(config.band_C),
            "hold_entry_dwell_s": float(config.hold_entry_dwell_s),
            "smoothing_window_s": float(config.smoothing_window_s),
            "rolling_std_window_s": float(config.rolling_std_window_s),
            "drift_window_s": float(config.drift_window_s),
            "support_duration_start_s": float(config.support_duration_start_s),
            "support_duration_grid_s": float(config.support_duration_grid_s),
            "support_min_runs": int(config.support_min_runs),
            "support_min_pass_fraction": float(config.support_min_pass_fraction),
            "support_min_fraction_in_band": float(config.support_min_fraction_in_band),
            "support_max_mean_abs_error_C": float(config.support_max_mean_abs_error_C),
            "support_max_abs_mean_error_C": float(config.support_max_abs_mean_error_C),
            "support_max_abs_drift_C": float(config.support_max_abs_drift_C),
        },
        "support_duration_s_by_observed_target_C": {
            f"{target_C:.1f}": float(summary["conservative_supported_hold_duration_s"])
            for target_C, summary in sorted(target_summaries.items())
        },
        "targets": [target_summaries[target_C] for target_C in sorted(target_summaries)],
        "artifacts": {
            "per_run_hold_metrics_csv": str(per_run_csv_path.resolve()),
            "hold_duration_support_grid_csv": str(support_csv_path.resolve()),
            "hold_summary_markdown": str(summary_md_path.resolve()),
            "plate_hold_overlays_png": str((figures_dir / "plate_hold_overlays_by_target.png").resolve()),
            "hold_support_summary_png": str((figures_dir / "hold_support_summary.png").resolve()),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    plot_hold_overlays(runs, figures_dir / "plate_hold_overlays_by_target.png", config)
    plot_hold_support(per_run_df, support_df, figures_dir / "hold_support_summary.png")
    write_markdown_summary(
        summary_md_path,
        input_paths=input_paths,
        target_summaries=target_summaries,
        config=config,
        figures_dir=figures_dir,
    )

    print(f"Wrote hold characterization to {out_dir}")
    print(f"Per-run CSV: {per_run_csv_path}")
    print(f"Support grid CSV: {support_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
