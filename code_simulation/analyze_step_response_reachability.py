#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

REQUIRED_COLUMNS = ("row_type", "panel_t_s", "set", "T_cal")
DEFAULT_RATE_WINDOWS_S = (10.0, 30.0)
DEFAULT_REACHABILITY_WINDOWS_S = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0)
DEFAULT_PROGRESS_GRID_S = np.arange(0.0, 301.0, 5.0)


@dataclass(frozen=True)
class AnalysisConfig:
    tolerance_c: float = 0.5
    smooth_window_s: float = 3.0
    initial_window_s: float = 5.0
    steady_window_s: float = 60.0
    onset_threshold_c: float = 0.2
    onset_slope_window_s: float = 10.0
    onset_slope_threshold_c_per_min: float = 0.5
    rate_windows_s: tuple[float, ...] = DEFAULT_RATE_WINDOWS_S
    reachability_windows_s: tuple[float, ...] = DEFAULT_REACHABILITY_WINDOWS_S
    progress_grid_s: np.ndarray = field(default_factory=lambda: DEFAULT_PROGRESS_GRID_S.copy())


@dataclass
class RunAnalysis:
    run_id: str
    path: Path
    target_c: float
    replicate: str
    direction: str
    direction_sign: float
    metrics: dict[str, object]
    df: pd.DataFrame


def default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_root() -> Path:
    return default_repo_root() / "data" / "characterization_cryostage"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "characterization_constraints" / "stage1_reachability"


def parse_filename(path: Path) -> tuple[float, str]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(
            f"File name does not match expected pattern 'cryostage_characterization_minX_<run>.csv': {path.name}"
        )
    target_c = -float(match.group("target"))
    replicate = match.group("replicate").upper()
    return target_c, replicate


def odd_window_points(dt_s: float, window_s: float, *, min_points: int = 3) -> int:
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        return min_points
    points = max(int(round(window_s / dt_s)), min_points)
    if points % 2 == 0:
        points += 1
    return points


def centered_slope_c_per_min(time_s: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    slopes = np.full_like(values, np.nan, dtype=np.float64)
    if time_s.size < 3 or window_s <= 0.0:
        return slopes

    half_window_s = 0.5 * float(window_s)
    valid = (time_s - half_window_s >= time_s[0]) & (time_s + half_window_s <= time_s[-1])
    if not np.any(valid):
        return slopes

    left = np.interp(time_s[valid] - half_window_s, time_s, values)
    right = np.interp(time_s[valid] + half_window_s, time_s, values)
    slopes[valid] = (right - left) / float(window_s) * 60.0
    return slopes


def forward_progress_c(time_s: np.ndarray, values: np.ndarray, direction_sign: float, window_s: float) -> np.ndarray:
    progress = np.full_like(values, np.nan, dtype=np.float64)
    if time_s.size < 2 or window_s <= 0.0:
        return progress

    valid = time_s + float(window_s) <= time_s[-1]
    if not np.any(valid):
        return progress

    future_values = np.interp(time_s[valid] + float(window_s), time_s, values)
    progress[valid] = direction_sign * (future_values - values[valid])
    return progress


def conservative_high(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(arr))


def conservative_low(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def format_float(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def load_run_dataframe(path: Path, config: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = pd.read_csv(path, comment="#", sep=None, engine="python", on_bad_lines="skip")
    missing = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    telemetry = raw_df[raw_df["row_type"].astype(str).str.lower() == "telemetry"].copy()
    telemetry_rows = int(len(telemetry))

    telemetry["panel_t_s"] = pd.to_numeric(telemetry["panel_t_s"], errors="coerce")
    telemetry["set"] = pd.to_numeric(telemetry["set"], errors="coerce")
    telemetry["T_cal"] = pd.to_numeric(telemetry["T_cal"], errors="coerce")

    clean = telemetry.dropna(subset=["panel_t_s", "set", "T_cal"]).sort_values("panel_t_s").copy()
    dropped_non_numeric_rows = telemetry_rows - int(len(clean))
    duplicate_time_rows = int(clean.duplicated(subset=["panel_t_s"]).sum())
    clean = clean.drop_duplicates(subset=["panel_t_s"], keep="first").copy()

    if clean.empty:
        raise ValueError(f"{path.name} has no valid telemetry rows after cleaning")

    clean["t_file_s"] = clean["panel_t_s"] - float(clean["panel_t_s"].iloc[0])

    dt = clean["t_file_s"].diff().dropna()
    median_dt_s = float(dt.median()) if not dt.empty else float("nan")
    smooth_points = odd_window_points(median_dt_s, config.smooth_window_s)
    clean["T_smooth_C"] = (
        clean["T_cal"]
        .rolling(window=smooth_points, center=True, min_periods=1)
        .median()
    )

    metadata = {
        "telemetry_rows": telemetry_rows,
        "dropped_non_numeric_rows": dropped_non_numeric_rows,
        "duplicate_time_rows_dropped": duplicate_time_rows,
        "median_dt_s": median_dt_s,
        "smooth_points": smooth_points,
    }
    return clean.reset_index(drop=True), metadata


def detect_reference_step_time_s(time_s: np.ndarray, set_c: np.ndarray, target_c: float) -> float:
    is_target = np.isclose(set_c, target_c, atol=0.1)
    if not np.any(is_target):
        return float("nan")

    start_idx = np.flatnonzero(is_target)[0]
    if start_idx == 0:
        if np.any(~is_target):
            return float(time_s[0])
        return float("nan")
    if np.any(~is_target[:start_idx]):
        return float(time_s[start_idx])
    return float("nan")


def detect_temperature_onset_time_s(
    time_s: np.ndarray,
    temp_smooth_c: np.ndarray,
    initial_temp_c: float,
    direction_sign: float,
    signed_slope_c_per_min: np.ndarray,
    reference_step_time_s: float,
    config: AnalysisConfig,
) -> tuple[float, str]:
    progress_c = direction_sign * (temp_smooth_c - initial_temp_c)

    search_mask = np.ones_like(time_s, dtype=bool)
    if math.isfinite(reference_step_time_s):
        search_mask &= time_s >= reference_step_time_s

    onset_candidates = np.flatnonzero(
        search_mask
        & (progress_c >= config.onset_threshold_c)
        & (signed_slope_c_per_min >= config.onset_slope_threshold_c_per_min)
    )
    if onset_candidates.size:
        return float(time_s[int(onset_candidates[0])]), "temperature_departure"

    if math.isfinite(reference_step_time_s):
        return float(reference_step_time_s), "reference_step_only"
    return 0.0, "file_start_fallback"


def compute_initial_temperature_c(df: pd.DataFrame, reference_step_time_s: float, initial_window_s: float) -> float:
    if math.isfinite(reference_step_time_s) and reference_step_time_s > 0.0:
        mask = df["t_file_s"] <= reference_step_time_s
    else:
        mask = df["t_file_s"] <= min(initial_window_s, float(df["t_file_s"].iloc[-1]))
    if not np.any(mask):
        mask = df.index <= min(len(df) - 1, 10)
    return float(df.loc[mask, "T_cal"].median())


def compute_settling_times_s(time_s: np.ndarray, temp_smooth_c: np.ndarray, target_c: float, tolerance_c: float, onset_time_s: float) -> tuple[float, float]:
    search_mask = time_s >= onset_time_s
    error_c = np.abs(temp_smooth_c - target_c)
    in_band = error_c <= tolerance_c

    first_enter_time_s = float("nan")
    candidates = np.flatnonzero(search_mask & in_band)
    if candidates.size:
        first_enter_time_s = float(time_s[int(candidates[0])])

    settling_time_s = float("nan")
    search_indices = np.flatnonzero(search_mask)
    if search_indices.size:
        within_after_onset = in_band[search_indices]
        settled_suffix = np.logical_and.accumulate(within_after_onset[::-1])[::-1]
        settling_candidates = search_indices[within_after_onset & settled_suffix]
        if settling_candidates.size:
            settling_time_s = float(time_s[int(settling_candidates[0])])

    return first_enter_time_s, settling_time_s


def compute_reverse_excursion_c(values: np.ndarray, direction_sign: float) -> float:
    if values.size == 0:
        return float("nan")
    if direction_sign < 0.0:
        running_extreme = np.minimum.accumulate(values)
        return float(np.max(values - running_extreme))
    running_extreme = np.maximum.accumulate(values)
    return float(np.max(running_extreme - values))


def analyze_run(path: Path, config: AnalysisConfig) -> RunAnalysis:
    target_from_name_c, replicate = parse_filename(path)
    df, load_meta = load_run_dataframe(path, config)

    set_values = df["set"].to_numpy(dtype=np.float64)
    time_s = df["t_file_s"].to_numpy(dtype=np.float64)
    temp_raw_c = df["T_cal"].to_numpy(dtype=np.float64)
    temp_smooth_c = df["T_smooth_C"].to_numpy(dtype=np.float64)

    dominant_set_series = pd.Series(set_values).round(3).mode(dropna=True)
    target_c = float(dominant_set_series.iloc[0]) if not dominant_set_series.empty else target_from_name_c
    reference_step_time_s = detect_reference_step_time_s(time_s, set_values, target_c)
    initial_temp_c = compute_initial_temperature_c(df, reference_step_time_s, config.initial_window_s)

    if abs(target_c - initial_temp_c) < 0.05:
        direction_sign = 0.0
        direction = "flat"
    elif target_c < initial_temp_c:
        direction_sign = -1.0
        direction = "cooling"
    else:
        direction_sign = 1.0
        direction = "warming"

    slope_onset_c_per_min = centered_slope_c_per_min(time_s, temp_smooth_c, config.onset_slope_window_s)
    signed_slope_onset_c_per_min = direction_sign * slope_onset_c_per_min
    onset_time_s, onset_method = detect_temperature_onset_time_s(
        time_s=time_s,
        temp_smooth_c=temp_smooth_c,
        initial_temp_c=initial_temp_c,
        direction_sign=direction_sign,
        signed_slope_c_per_min=signed_slope_onset_c_per_min,
        reference_step_time_s=reference_step_time_s,
        config=config,
    )

    alignment_temp_c = float(np.interp(onset_time_s, time_s, temp_smooth_c))

    steady_start_s = max(0.0, float(time_s[-1]) - config.steady_window_s)
    steady_mask = time_s >= steady_start_s
    final_steady_temp_c = float(np.median(temp_raw_c[steady_mask]))

    first_enter_band_time_s, settling_time_s = compute_settling_times_s(
        time_s=time_s,
        temp_smooth_c=temp_smooth_c,
        target_c=target_c,
        tolerance_c=config.tolerance_c,
        onset_time_s=onset_time_s,
    )

    error_c = temp_smooth_c - target_c
    if direction_sign < 0.0:
        overshoot_beyond_target_c = float(np.max(np.maximum(target_c - temp_smooth_c, 0.0)))
    elif direction_sign > 0.0:
        overshoot_beyond_target_c = float(np.max(np.maximum(temp_smooth_c - target_c, 0.0)))
    else:
        overshoot_beyond_target_c = 0.0

    slope_columns: dict[str, np.ndarray] = {}
    for window_s in config.rate_windows_s:
        column = f"slope_{int(window_s)}s_C_per_min"
        slope_columns[column] = centered_slope_c_per_min(time_s, temp_smooth_c, window_s)
        df[column] = slope_columns[column]

    onset_mask = time_s >= onset_time_s
    signed_10s = direction_sign * slope_columns["slope_10s_C_per_min"][onset_mask]
    signed_30s = direction_sign * slope_columns["slope_30s_C_per_min"][onset_mask]

    clean_signed_10s = signed_10s[np.isfinite(signed_10s)]
    clean_signed_30s = signed_30s[np.isfinite(signed_30s)]
    clean_slope_10s = slope_columns["slope_10s_C_per_min"][onset_mask]
    clean_slope_10s = clean_slope_10s[np.isfinite(clean_slope_10s)]

    reverse_excursion_c = compute_reverse_excursion_c(temp_smooth_c[onset_mask], direction_sign)

    set_counts = pd.Series(set_values).round(3).value_counts(dropna=True)
    dominant_set_fraction = float(set_counts.iloc[0] / len(set_values))
    setpoint_change_count = int(np.count_nonzero(np.abs(np.diff(set_values)) > 0.1))

    anomaly_flags: list[str] = []
    if load_meta["dropped_non_numeric_rows"]:
        anomaly_flags.append(f"dropped_non_numeric_rows={load_meta['dropped_non_numeric_rows']}")
    if load_meta["duplicate_time_rows_dropped"]:
        anomaly_flags.append(f"duplicate_times={load_meta['duplicate_time_rows_dropped']}")
    if setpoint_change_count:
        anomaly_flags.append(f"set_changes={setpoint_change_count}")
    if abs(target_c - target_from_name_c) > 0.25:
        anomaly_flags.append("filename_target_mismatch")
    if not math.isfinite(first_enter_band_time_s):
        anomaly_flags.append("never_entered_band")
    if not math.isfinite(settling_time_s):
        anomaly_flags.append("never_settled")
    if direction == "flat":
        anomaly_flags.append("near_zero_step")
    if load_meta["median_dt_s"] > 0.15:
        anomaly_flags.append("slower_sampling")

    metrics = {
        "run_id": path.stem,
        "file_path": str(path.resolve()),
        "nominal_setpoint_C": target_c,
        "filename_target_C": target_from_name_c,
        "replicate": replicate,
        "telemetry_rows": load_meta["telemetry_rows"],
        "dropped_non_numeric_rows": load_meta["dropped_non_numeric_rows"],
        "duplicate_time_rows_dropped": load_meta["duplicate_time_rows_dropped"],
        "median_dt_s": load_meta["median_dt_s"],
        "setpoint_change_count": setpoint_change_count,
        "dominant_set_fraction": dominant_set_fraction,
        "initial_plate_temp_C": initial_temp_c,
        "final_steady_plate_temp_C": final_steady_temp_c,
        "step_amplitude_C": target_c - initial_temp_c,
        "observed_plate_change_C": final_steady_temp_c - initial_temp_c,
        "response_direction": direction,
        "reference_step_time_s": reference_step_time_s,
        "alignment_time_s": onset_time_s,
        "alignment_method": onset_method,
        "observed_dead_time_s": (
            onset_time_s - reference_step_time_s
            if math.isfinite(reference_step_time_s)
            else float("nan")
        ),
        "time_to_first_enter_band_s": (
            first_enter_band_time_s - onset_time_s
            if math.isfinite(first_enter_band_time_s)
            else float("nan")
        ),
        "settling_time_s": (
            settling_time_s - onset_time_s
            if math.isfinite(settling_time_s)
            else float("nan")
        ),
        "peak_signed_error_low_C": float(np.min(error_c)),
        "peak_signed_error_high_C": float(np.max(error_c)),
        "overshoot_beyond_target_C": overshoot_beyond_target_c,
        "final_bias_C": final_steady_temp_c - target_c,
        "peak_negative_rate_10s_C_per_min": float(np.min(clean_slope_10s)) if clean_slope_10s.size else float("nan"),
        "peak_positive_rate_10s_C_per_min": float(np.max(clean_slope_10s)) if clean_slope_10s.size else float("nan"),
        "peak_toward_target_rate_10s_C_per_min": float(np.max(clean_signed_10s)) if clean_signed_10s.size else float("nan"),
        "p90_toward_target_rate_10s_C_per_min": float(np.quantile(clean_signed_10s, 0.90)) if clean_signed_10s.size else float("nan"),
        "peak_toward_target_rate_30s_C_per_min": float(np.max(clean_signed_30s)) if clean_signed_30s.size else float("nan"),
        "p90_toward_target_rate_30s_C_per_min": float(np.quantile(clean_signed_30s, 0.90)) if clean_signed_30s.size else float("nan"),
        "peak_reverse_rate_10s_C_per_min": float(np.max(np.maximum(-clean_signed_10s, 0.0))) if clean_signed_10s.size else float("nan"),
        "max_reverse_excursion_C": reverse_excursion_c,
        "anomaly_flags": ";".join(anomaly_flags),
    }

    df["aligned_t_s"] = df["t_file_s"] - onset_time_s
    df["progress_from_alignment_C"] = direction_sign * (alignment_temp_c - df["T_smooth_C"])
    df["error_to_target_C"] = df["T_smooth_C"] - target_c
    df["directional_slope_10s_C_per_min"] = direction_sign * df["slope_10s_C_per_min"]
    df["directional_slope_30s_C_per_min"] = direction_sign * df["slope_30s_C_per_min"]
    df["run_id"] = path.stem
    df["nominal_setpoint_C"] = target_c

    return RunAnalysis(
        run_id=path.stem,
        path=path,
        target_c=target_c,
        replicate=replicate,
        direction=direction,
        direction_sign=direction_sign,
        metrics=metrics,
        df=df,
    )


def compute_aggregated_metrics(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setpoint_c, group in per_run_df.groupby("nominal_setpoint_C", sort=True):
        rows.append(
            {
                "group": f"setpoint_{setpoint_c:.0f}C",
                "nominal_setpoint_C": setpoint_c,
                "n_runs": int(len(group)),
                "initial_plate_temp_median_C": float(group["initial_plate_temp_C"].median()),
                "step_amplitude_median_C": float(group["step_amplitude_C"].median()),
                "conservative_first_enter_band_s": conservative_high(group["time_to_first_enter_band_s"]),
                "conservative_settling_time_s": conservative_high(group["settling_time_s"]),
                "conservative_overshoot_C": conservative_high(group["overshoot_beyond_target_C"]),
                "conservative_peak_toward_rate_10s_C_per_min": conservative_low(group["peak_toward_target_rate_10s_C_per_min"]),
                "conservative_p90_toward_rate_30s_C_per_min": conservative_low(group["p90_toward_target_rate_30s_C_per_min"]),
                "conservative_reverse_excursion_C": conservative_high(group["max_reverse_excursion_C"]),
                "final_bias_median_C": float(group["final_bias_C"].median()),
            }
        )

    rows.append(
        {
            "group": "overall",
            "nominal_setpoint_C": float("nan"),
            "n_runs": int(len(per_run_df)),
            "initial_plate_temp_median_C": float(per_run_df["initial_plate_temp_C"].median()),
            "step_amplitude_median_C": float(per_run_df["step_amplitude_C"].median()),
            "conservative_first_enter_band_s": conservative_high(per_run_df["time_to_first_enter_band_s"]),
            "conservative_settling_time_s": conservative_high(per_run_df["settling_time_s"]),
            "conservative_overshoot_C": conservative_high(per_run_df["overshoot_beyond_target_C"]),
            "conservative_peak_toward_rate_10s_C_per_min": conservative_low(per_run_df["peak_toward_target_rate_10s_C_per_min"]),
            "conservative_p90_toward_rate_30s_C_per_min": conservative_low(per_run_df["p90_toward_target_rate_30s_C_per_min"]),
            "conservative_reverse_excursion_C": conservative_high(per_run_df["max_reverse_excursion_C"]),
            "final_bias_median_C": float(per_run_df["final_bias_C"].median()),
        }
    )
    return pd.DataFrame(rows)


def compute_window_reachability(runs: list[RunAnalysis], config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_s in config.reachability_windows_s:
        per_run_rows: list[dict[str, object]] = []
        for run in runs:
            progress = forward_progress_c(
                run.df["t_file_s"].to_numpy(dtype=np.float64),
                run.df["T_smooth_C"].to_numpy(dtype=np.float64),
                run.direction_sign,
                window_s,
            )
            valid_mask = (run.df["t_file_s"].to_numpy(dtype=np.float64) >= float(run.metrics["alignment_time_s"])) & np.isfinite(progress)
            observed = progress[valid_mask]
            max_progress_c = float(np.max(observed)) if observed.size else float("nan")
            per_run_rows.append(
                {
                    "window_s": window_s,
                    "group": f"setpoint_{run.target_c:.0f}C",
                    "nominal_setpoint_C": run.target_c,
                    "run_id": run.run_id,
                    "max_progress_C": max_progress_c,
                }
            )
            rows.append(per_run_rows[-1])

        per_run_df = pd.DataFrame(per_run_rows)
        rows.append(
            {
                "window_s": window_s,
                "group": "overall_conservative",
                "nominal_setpoint_C": float("nan"),
                "run_id": "",
                "max_progress_C": conservative_low(per_run_df["max_progress_C"]),
            }
        )
        for setpoint_c, group_df in per_run_df.groupby("nominal_setpoint_C", sort=True):
            rows.append(
                {
                    "window_s": window_s,
                    "group": f"setpoint_{setpoint_c:.0f}C_conservative",
                    "nominal_setpoint_C": setpoint_c,
                    "run_id": "",
                    "max_progress_C": conservative_low(group_df["max_progress_C"]),
                }
            )
    return pd.DataFrame(rows)


def compute_aligned_progress_envelope(runs: list[RunAnalysis], config: AnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group_runs in [("overall", runs)] + [
        (f"setpoint_{target_c:.0f}C", [run for run in runs if run.target_c == target_c])
        for target_c in sorted({run.target_c for run in runs})
    ]:
        for elapsed_s in config.progress_grid_s:
            values: list[float] = []
            for run in group_runs:
                aligned_t = run.df["aligned_t_s"].to_numpy(dtype=np.float64)
                progress = run.df["progress_from_alignment_C"].to_numpy(dtype=np.float64)
                if elapsed_s < aligned_t[0] or elapsed_s > aligned_t[-1]:
                    continue
                values.append(float(np.interp(elapsed_s, aligned_t, progress)))
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            rows.append(
                {
                    "group": group_name,
                    "elapsed_s": float(elapsed_s),
                    "n_runs": int(arr.size),
                    "progress_min_C": float(np.min(arr)) if arr.size else float("nan"),
                    "progress_p25_C": float(np.quantile(arr, 0.25)) if arr.size else float("nan"),
                    "progress_median_C": float(np.median(arr)) if arr.size else float("nan"),
                    "progress_max_C": float(np.max(arr)) if arr.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.5),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
        }
    )


def target_colors(runs: list[RunAnalysis]) -> dict[float, object]:
    targets = sorted({run.target_c for run in runs}, reverse=True)
    cmap = plt.get_cmap("viridis")
    return {target: cmap(idx / max(len(targets) - 1, 1)) for idx, target in enumerate(targets)}


def plot_aligned_temperature_overlays(runs: list[RunAnalysis], config: AnalysisConfig, out_path: Path) -> None:
    setup_plot_style()
    colors = target_colors(runs)
    targets = sorted({run.target_c for run in runs}, reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True, sharey=False)
    for ax, target_c in zip(axes.ravel(), targets):
        ax.axhspan(target_c - config.tolerance_c, target_c + config.tolerance_c, color="#dfeec8", alpha=0.65)
        ax.axhline(target_c, color="#3f3f3f", linewidth=1.2)
        for run in [item for item in runs if item.target_c == target_c]:
            linestyle = REPLICATE_LINESTYLES.get(run.replicate, "-.")
            aligned = run.df[run.df["aligned_t_s"] >= 0.0]
            ax.plot(
                aligned["aligned_t_s"],
                aligned["T_smooth_C"],
                color=colors[target_c],
                linestyle=linestyle,
                linewidth=2.0,
                label=run.replicate,
            )
            first_enter = run.metrics["time_to_first_enter_band_s"]
            settling = run.metrics["settling_time_s"]
            if math.isfinite(first_enter):
                ax.scatter(first_enter, target_c, color=colors[target_c], marker="o", s=20)
            if math.isfinite(settling):
                ax.scatter(settling, target_c, color=colors[target_c], marker="s", s=20)
        ax.set_title(f"Target {target_c:.0f} C")
        ax.set_xlim(0.0, 320.0)
        ax.set_xlabel("Aligned time since response onset (s)")
        ax.set_ylabel("Plate temperature T_cal (C)")
    fig.suptitle("Cryostage step responses aligned by measured onset", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_band_overlays(runs: list[RunAnalysis], config: AnalysisConfig, out_path: Path) -> None:
    setup_plot_style()
    colors = target_colors(runs)
    targets = sorted({run.target_c for run in runs}, reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True, sharey=True)
    for ax, target_c in zip(axes.ravel(), targets):
        ax.axhspan(-config.tolerance_c, config.tolerance_c, color="#fee8c8", alpha=0.7)
        ax.axhline(0.0, color="#3f3f3f", linewidth=1.2)
        for run in [item for item in runs if item.target_c == target_c]:
            linestyle = REPLICATE_LINESTYLES.get(run.replicate, "-.")
            aligned = run.df[run.df["aligned_t_s"] >= 0.0]
            ax.plot(
                aligned["aligned_t_s"],
                aligned["error_to_target_C"],
                color=colors[target_c],
                linestyle=linestyle,
                linewidth=2.0,
            )
        ax.set_title(f"Target {target_c:.0f} C")
        ax.set_xlim(0.0, 320.0)
        ax.set_xlabel("Aligned time since response onset (s)")
        ax.set_ylabel("T_cal - setpoint (C)")
    fig.suptitle("Band-settling view using the operational ±0.5 C tolerance", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_rate_profiles(runs: list[RunAnalysis], out_path: Path) -> None:
    setup_plot_style()
    colors = target_colors(runs)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)
    for ax, column, title in [
        (axes[0], "slope_10s_C_per_min", "Centered 10 s plate-temperature slope"),
        (axes[1], "slope_30s_C_per_min", "Centered 30 s plate-temperature slope"),
    ]:
        ax.axhline(0.0, color="#3f3f3f", linewidth=1.0)
        for run in runs:
            aligned = run.df[run.df["aligned_t_s"] >= 0.0]
            ax.plot(
                aligned["aligned_t_s"],
                aligned[column],
                color=colors[run.target_c],
                linestyle=REPLICATE_LINESTYLES.get(run.replicate, "-."),
                linewidth=1.8,
                alpha=0.95,
            )
        ax.set_title(title)
        ax.set_ylabel("dT/dt (C/min)")
        ax.set_xlim(0.0, 320.0)
    axes[1].set_xlabel("Aligned time since response onset (s)")
    handles = [
        plt.Line2D([0], [0], color=colors[target], linewidth=2.5, label=f"{target:.0f} C")
        for target in sorted(colors, reverse=True)
    ]
    axes[0].legend(handles=handles, title="Target")
    fig.suptitle("Observed cooling and rebound rates from smoothed T_cal", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_reachability_summary(
    aggregated_df: pd.DataFrame,
    window_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    out_path: Path,
) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))

    overall_window = window_df[window_df["group"] == "overall_conservative"].sort_values("window_s")
    axes[0].plot(overall_window["window_s"], overall_window["max_progress_C"], color="#1b9e77", linewidth=2.5)
    axes[0].set_title("Conservative finite-window cooling envelope")
    axes[0].set_xlabel("Window duration (s)")
    axes[0].set_ylabel("Max conservative cooling over window (C)")

    overall_progress = progress_df[progress_df["group"] == "overall"].sort_values("elapsed_s")
    axes[1].fill_between(
        overall_progress["elapsed_s"],
        overall_progress["progress_min_C"],
        overall_progress["progress_max_C"],
        color="#ccece6",
        alpha=0.7,
    )
    axes[1].plot(overall_progress["elapsed_s"], overall_progress["progress_median_C"], color="#0c7c59", linewidth=2.5)
    axes[1].plot(overall_progress["elapsed_s"], overall_progress["progress_min_C"], color="#045a3d", linewidth=1.5, linestyle="--")
    axes[1].set_title("Aligned progress envelope across all runs")
    axes[1].set_xlabel("Aligned time since response onset (s)")
    axes[1].set_ylabel("Cooling progress from onset (C)")
    axes[1].set_xlim(0.0, 300.0)

    overall_row = aggregated_df[aggregated_df["group"] == "overall"].iloc[0]
    text = "\n".join(
        [
            f"Conservative first band entry: {format_float(float(overall_row['conservative_first_enter_band_s']), 1)} s",
            f"Conservative settling time: {format_float(float(overall_row['conservative_settling_time_s']), 1)} s",
            f"Conservative peak 10 s rate: {format_float(float(overall_row['conservative_peak_toward_rate_10s_C_per_min']), 2)} C/min",
            f"Conservative p90 30 s rate: {format_float(float(overall_row['conservative_p90_toward_rate_30s_C_per_min']), 2)} C/min",
        ]
    )
    axes[1].text(
        0.98,
        0.02,
        text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Conservative empirical reachability summary", fontsize=14, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def build_summary_payload(
    input_paths: list[Path],
    per_run_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    window_df: pd.DataFrame,
    config: AnalysisConfig,
) -> dict[str, object]:
    overall = aggregated_df[aggregated_df["group"] == "overall"].iloc[0]
    overall_window = (
        window_df[window_df["group"] == "overall_conservative"]
        .sort_values("window_s")
        .reset_index(drop=True)
    )

    by_setpoint: dict[str, object] = {}
    for _, row in aggregated_df[aggregated_df["group"] != "overall"].iterrows():
        by_setpoint[f"{float(row['nominal_setpoint_C']):.0f}C"] = {
            "n_runs": int(row["n_runs"]),
            "conservative_first_enter_band_s": float(row["conservative_first_enter_band_s"]),
            "conservative_settling_time_s": float(row["conservative_settling_time_s"]),
            "conservative_overshoot_C": float(row["conservative_overshoot_C"]),
            "conservative_peak_toward_rate_10s_C_per_min": float(row["conservative_peak_toward_rate_10s_C_per_min"]),
            "conservative_p90_toward_rate_30s_C_per_min": float(row["conservative_p90_toward_rate_30s_C_per_min"]),
            "conservative_reverse_excursion_C": float(row["conservative_reverse_excursion_C"]),
        }

    return {
        "source_files": [str(path.resolve()) for path in input_paths],
        "tolerance_band_C": config.tolerance_c,
        "alignment_convention": {
            "primary": "temperature onset based on the first sustained measured departure toward the target on smoothed T_cal",
            "details": (
                "If a setpoint change is visible inside the file, dead time is referenced to that timestamp. "
                "Otherwise the run is aligned at the first sample where 10 s directional slope exceeds "
                f"{config.onset_slope_threshold_c_per_min:.2f} C/min and smoothed T_cal has moved by at least "
                f"{config.onset_threshold_c:.2f} C from the initial baseline."
            ),
        },
        "conservative_envelope_definition": {
            "time_metrics": "Maximum observed value across runs for first band entry, settling time, overshoot, and reverse excursion.",
            "rate_metrics": "Minimum observed value across runs for toward-target rate summaries, so later trajectory limits do not assume best-case performance.",
            "window_reachability": (
                "For each run and window W, compute the largest observed forward cooling drop over any W-second interval "
                "after alignment using smoothed T_cal. The overall conservative envelope is the minimum of those per-run maxima."
            ),
        },
        "overall_limits": {
            "conservative_first_enter_band_s": float(overall["conservative_first_enter_band_s"]),
            "conservative_settling_time_s": float(overall["conservative_settling_time_s"]),
            "conservative_overshoot_C": float(overall["conservative_overshoot_C"]),
            "conservative_peak_toward_rate_10s_C_per_min": float(overall["conservative_peak_toward_rate_10s_C_per_min"]),
            "conservative_p90_toward_rate_30s_C_per_min": float(overall["conservative_p90_toward_rate_30s_C_per_min"]),
            "conservative_reverse_excursion_C": float(overall["conservative_reverse_excursion_C"]),
            "window_reachability_C": {
                f"{float(row['window_s']):.0f}s": float(row["max_progress_C"])
                for _, row in overall_window.iterrows()
            },
        },
        "by_setpoint": by_setpoint,
        "limitations": [
            "The available characterization runs are all cooling responses from roughly 9 C toward colder setpoints.",
            "Observed positive dT/dt values reflect rebound or measurement noise, not dedicated warming-step characterization.",
            "Only one file shows an explicit pre-step setpoint transition inside the recording, so dead time is sparsely observable.",
        ],
        "per_run_anomalies": {
            row["run_id"]: row["anomaly_flags"]
            for _, row in per_run_df.iterrows()
            if isinstance(row["anomaly_flags"], str) and row["anomaly_flags"]
        },
    }


def write_markdown_summary(
    out_path: Path,
    input_paths: list[Path],
    per_run_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    window_df: pd.DataFrame,
    config: AnalysisConfig,
) -> None:
    overall = aggregated_df[aggregated_df["group"] == "overall"].iloc[0]
    overall_window = window_df[window_df["group"] == "overall_conservative"].sort_values("window_s")

    lines: list[str] = []
    lines.append("# Cryostage Reachability Constraints")
    lines.append("")
    lines.append("## Files Found")
    lines.append("")
    for path in input_paths:
        lines.append(f"- `{path.as_posix()}`")
    lines.append("")
    lines.append("## Inferred CSV Schema And Conventions")
    lines.append("")
    lines.append("- All 12 files share the same header and are parsed with comment-prefixed metadata skipped.")
    lines.append("- The analysis keeps `row_type == telemetry` rows and uses `panel_t_s`, `set`, and `T_cal` as the core columns.")
    lines.append("- `set` is effectively constant at the nominal target in 11 of 12 files; `cryostage_characterization_min10_III.csv` includes a short pre-step `set = 0 C` segment before the target is applied.")
    lines.append("- The firmware characterization-state columns are present but inactive in these logs, so the run segmentation must come from the recorded `set` column and the measured plate response itself.")
    lines.append(f"- The operational settling band is defined as `setpoint ± {config.tolerance_c:.1f} C`.")
    lines.append("")
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("- Initial plate temperature is the median `T_cal` over the first 5 s of each run, or the pre-step portion when a reference change is visible.")
    lines.append("- Final steady temperature is the median `T_cal` over the last 60 s of each run.")
    lines.append("- Response alignment uses the first measured departure toward the target on smoothed `T_cal`; when a `set` transition is visible, dead time is measured relative to that event.")
    lines.append("- First-entry time is the first aligned time at which smoothed `T_cal` enters the `±0.5 C` band.")
    lines.append("- Settling time is the first aligned time after which smoothed `T_cal` stays inside the band for the remainder of the recorded run.")
    lines.append("- Rolling rates use centered 10 s and 30 s finite-difference slopes computed on the smoothed `T_cal` trace.")
    lines.append("- Finite-window reachability uses the largest observed cooling drop over forward windows of 10 s to 300 s; the conservative envelope takes the minimum of those per-run maxima.")
    lines.append("")
    lines.append("## Conservative Envelope Definition")
    lines.append("")
    lines.append("- Conservative time-to-band and settling values are the worst observed values across the available runs.")
    lines.append("- Conservative toward-target rates are the smallest observed run-level rate summaries, so later trajectory design does not assume best-case performance.")
    lines.append("- Conservative finite-window cooling reachability is the minimum across runs of the best observed forward cooling change over each window length.")
    lines.append("")
    lines.append("## Main Practical Conclusions")
    lines.append("")
    lines.append(
        f"- Across all runs, conservative first entry into the `±{config.tolerance_c:.1f} C` band is {format_float(float(overall['conservative_first_enter_band_s']), 1)} s after measured onset."
    )
    lines.append(
        f"- Across all runs, conservative settling time is {format_float(float(overall['conservative_settling_time_s']), 1)} s after measured onset."
    )
    lines.append(
        f"- Conservative toward-target cooling rates are {format_float(float(overall['conservative_peak_toward_rate_10s_C_per_min']), 2)} C/min on 10 s windows and {format_float(float(overall['conservative_p90_toward_rate_30s_C_per_min']), 2)} C/min for the 30 s p90 summary."
    )
    lines.append(
        f"- Conservative non-monotone rebound after onset stays within {format_float(float(overall['conservative_reverse_excursion_C']), 2)} C, so the responses are practically close to monotone once the plate starts moving."
    )
    reachability_pairs = []
    for _, row in overall_window.iterrows():
        window_s = float(row["window_s"])
        if window_s in {30.0, 60.0, 120.0, 180.0}:
            reachability_pairs.append(f"{int(window_s)} s -> {format_float(float(row['max_progress_C']), 2)} C")
    if reachability_pairs:
        lines.append("- Conservative finite-window cooling change: " + ", ".join(reachability_pairs) + ".")
    lines.append("- Dedicated warming-step characterization is still missing, so any future warming constraints should not be inferred from this dataset alone.")
    lines.append("")
    lines.append("## Per-Run Notes")
    lines.append("")
    for _, row in per_run_df.iterrows():
        flags = row["anomaly_flags"] if isinstance(row["anomaly_flags"], str) and row["anomaly_flags"] else "none"
        lines.append(
            f"- `{row['run_id']}`: first-band {format_float(float(row['time_to_first_enter_band_s']), 1)} s, "
            f"settling {format_float(float(row['settling_time_s']), 1)} s, "
            f"peak 10 s toward-target rate {format_float(float(row['peak_toward_target_rate_10s_C_per_min']), 2)} C/min, "
            f"flags `{flags}`."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cryostage step-response metrics and conservative reachability constraints from characterization CSVs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Folder containing characterization_min*/cryostage_characterization_min*.csv.",
    )
    parser.add_argument(
        "--glob",
        default="characterization_min*/cryostage_characterization_min*.csv",
        help="Glob pattern used inside --data-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Folder where CSV, JSON, markdown, and figure outputs will be written.",
    )
    parser.add_argument("--tolerance-c", type=float, default=0.5, help="Operational tolerance band around the target.")
    parser.add_argument("--smooth-window-s", type=float, default=3.0, help="Rolling median smoothing window for T_cal.")
    parser.add_argument("--steady-window-s", type=float, default=60.0, help="Terminal window used for steady-temperature estimates.")
    parser.add_argument("--initial-window-s", type=float, default=5.0, help="Initial window used for the starting-temperature estimate.")
    parser.add_argument("--onset-threshold-c", type=float, default=0.2, help="Minimum smoothed temperature departure used to mark response onset.")
    parser.add_argument(
        "--onset-slope-window-s",
        type=float,
        default=10.0,
        help="Centered slope window used for onset detection.",
    )
    parser.add_argument(
        "--onset-slope-threshold-c-per-min",
        type=float,
        default=0.5,
        help="Minimum directional slope used to mark response onset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        tolerance_c=float(args.tolerance_c),
        smooth_window_s=float(args.smooth_window_s),
        initial_window_s=float(args.initial_window_s),
        steady_window_s=float(args.steady_window_s),
        onset_threshold_c=float(args.onset_threshold_c),
        onset_slope_window_s=float(args.onset_slope_window_s),
        onset_slope_threshold_c_per_min=float(args.onset_slope_threshold_c_per_min),
    )

    input_paths = sorted(args.data_root.glob(args.glob))
    if not input_paths:
        raise SystemExit(f"No input files matched {args.glob!r} under {args.data_root}")

    runs = [analyze_run(path, config) for path in input_paths]
    per_run_df = pd.DataFrame([run.metrics for run in runs]).sort_values(["nominal_setpoint_C", "replicate"]).reset_index(drop=True)
    aggregated_df = compute_aggregated_metrics(per_run_df)
    window_df = compute_window_reachability(runs, config)
    progress_df = compute_aligned_progress_envelope(runs, config)
    summary_payload = build_summary_payload(input_paths, per_run_df, aggregated_df, window_df, config)

    out_dir = args.output_dir
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = out_dir / "per_run_metrics.csv"
    aggregated_path = out_dir / "aggregated_metrics_by_setpoint.csv"
    window_path = out_dir / "window_reachability_envelope.csv"
    progress_path = out_dir / "aligned_progress_envelope.csv"
    summary_json_path = out_dir / "reachability_summary.json"
    summary_md_path = out_dir / "characterization_constraints_summary.md"

    per_run_df.to_csv(per_run_path, index=False)
    aggregated_df.to_csv(aggregated_path, index=False)
    window_df.to_csv(window_path, index=False)
    progress_df.to_csv(progress_path, index=False)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_markdown_summary(summary_md_path, input_paths, per_run_df, aggregated_df, window_df, config)

    plot_aligned_temperature_overlays(runs, config, figures_dir / "aligned_temperature_overlays_by_setpoint.png")
    plot_error_band_overlays(runs, config, figures_dir / "aligned_error_band_by_setpoint.png")
    plot_rate_profiles(runs, figures_dir / "rolling_rate_profiles.png")
    plot_reachability_summary(aggregated_df, window_df, progress_df, figures_dir / "reachability_envelope_summary.png")

    print(f"Per-run metrics: {per_run_path}")
    print(f"Aggregated metrics: {aggregated_path}")
    print(f"Window reachability envelope: {window_path}")
    print(f"Aligned progress envelope: {progress_path}")
    print(f"Reachability JSON: {summary_json_path}")
    print(f"Technical note: {summary_md_path}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
