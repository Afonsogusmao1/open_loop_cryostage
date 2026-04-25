from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlateTrackingSeries:
    time_s: np.ndarray
    T_ref_C: np.ndarray
    T_plate_C: np.ndarray
    plate_error_C: np.ndarray
    abs_plate_error_C: np.ndarray
    in_evaluation_window: np.ndarray


@dataclass(frozen=True)
class PlateTrackingSummary:
    tolerance_C: float
    evaluation_window_start_s: float
    evaluation_window_end_s: float
    num_window_samples: int
    max_abs_plate_error_C: float
    rmse_plate_error_C: float
    mean_plate_error_C: float
    mean_abs_plate_error_C: float
    fraction_within_tolerance: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tolerance_C": float(self.tolerance_C),
            "evaluation_window_start_s": float(self.evaluation_window_start_s),
            "evaluation_window_end_s": float(self.evaluation_window_end_s),
            "num_window_samples": int(self.num_window_samples),
            "max_abs_plate_error_C": float(self.max_abs_plate_error_C),
            "rmse_plate_error_C": float(self.rmse_plate_error_C),
            "mean_plate_error_C": float(self.mean_plate_error_C),
            "mean_abs_plate_error_C": float(self.mean_abs_plate_error_C),
            "fraction_within_tolerance": float(self.fraction_within_tolerance),
        }


def _as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def summarize_plate_tracking(
    *,
    time_s,
    T_ref_C,
    T_plate_C,
    tolerance_C: float,
    evaluation_window_start_s: float,
    evaluation_window_end_s: float | None,
) -> tuple[PlateTrackingSummary, PlateTrackingSeries]:
    time = _as_float_array(time_s, name="time_s")
    T_ref = _as_float_array(T_ref_C, name="T_ref_C")
    T_plate = _as_float_array(T_plate_C, name="T_plate_C")
    if time.shape != T_ref.shape or time.shape != T_plate.shape:
        raise ValueError("time_s, T_ref_C, and T_plate_C must have the same shape")

    tolerance_C = float(tolerance_C)
    if not math.isfinite(tolerance_C) or tolerance_C < 0.0:
        raise ValueError("tolerance_C must be finite and non-negative")

    finite = np.isfinite(time) & np.isfinite(T_ref) & np.isfinite(T_plate)
    evaluation_mask = finite.copy()
    start_s = float(evaluation_window_start_s)
    end_s = math.nan if evaluation_window_end_s is None else float(evaluation_window_end_s)
    if math.isfinite(start_s):
        evaluation_mask &= time >= start_s
    if math.isfinite(end_s):
        evaluation_mask &= time <= end_s

    plate_error_C = T_plate - T_ref
    abs_plate_error_C = np.abs(plate_error_C)

    if np.any(evaluation_mask):
        err = plate_error_C[evaluation_mask]
        abs_err = abs_plate_error_C[evaluation_mask]
        max_abs_plate_error_C = float(np.max(abs_err))
        rmse_plate_error_C = float(np.sqrt(np.mean(err * err)))
        mean_plate_error_C = float(np.mean(err))
        mean_abs_plate_error_C = float(np.mean(abs_err))
        fraction_within_tolerance = float(np.mean(abs_err <= tolerance_C))
    else:
        max_abs_plate_error_C = math.nan
        rmse_plate_error_C = math.nan
        mean_plate_error_C = math.nan
        mean_abs_plate_error_C = math.nan
        fraction_within_tolerance = math.nan

    summary = PlateTrackingSummary(
        tolerance_C=tolerance_C,
        evaluation_window_start_s=start_s,
        evaluation_window_end_s=end_s,
        num_window_samples=int(np.count_nonzero(evaluation_mask)),
        max_abs_plate_error_C=max_abs_plate_error_C,
        rmse_plate_error_C=rmse_plate_error_C,
        mean_plate_error_C=mean_plate_error_C,
        mean_abs_plate_error_C=mean_abs_plate_error_C,
        fraction_within_tolerance=fraction_within_tolerance,
    )
    series = PlateTrackingSeries(
        time_s=time,
        T_ref_C=T_ref,
        T_plate_C=T_plate,
        plate_error_C=plate_error_C,
        abs_plate_error_C=abs_plate_error_C,
        in_evaluation_window=evaluation_mask,
    )
    return summary, series


def plate_tracking_success(summary: PlateTrackingSummary) -> bool:
    return bool(
        math.isfinite(summary.mean_abs_plate_error_C)
        and summary.mean_abs_plate_error_C <= float(summary.tolerance_C) + 1.0e-12
    )


def write_plate_tracking_summary_csv(path: str | Path, summary: PlateTrackingSummary) -> None:
    row = summary.to_dict()
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_plate_tracking_timeseries_csv(path: str | Path, series: PlateTrackingSeries) -> None:
    fieldnames = (
        "time_s",
        "T_ref_C",
        "T_plate_C",
        "plate_error_C",
        "abs_plate_error_C",
        "in_evaluation_window",
    )
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(series.time_s.size):
            writer.writerow(
                {
                    "time_s": float(series.time_s[idx]),
                    "T_ref_C": float(series.T_ref_C[idx]),
                    "T_plate_C": float(series.T_plate_C[idx]),
                    "plate_error_C": float(series.plate_error_C[idx]),
                    "abs_plate_error_C": float(series.abs_plate_error_C[idx]),
                    "in_evaluation_window": int(bool(series.in_evaluation_window[idx])),
                }
            )


__all__ = [
    "PlateTrackingSeries",
    "PlateTrackingSummary",
    "plate_tracking_success",
    "summarize_plate_tracking",
    "write_plate_tracking_summary_csv",
    "write_plate_tracking_timeseries_csv",
]
