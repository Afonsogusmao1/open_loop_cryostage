from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


def load_front_columns(front_csv_path: Path) -> dict[str, np.ndarray]:
    columns = {
        "time_s": [],
        "time_since_fill_s": [],
        "z_front_m": [],
        "z_front_wall_m": [],
        "Tmax_fillable_C": [],
        "freeze_complete_flag": [],
    }
    with front_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in columns:
                raw = row.get(key, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = math.nan
                columns[key].append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def first_time_at_or_above(time_since_fill_s: np.ndarray, values: np.ndarray, threshold: float) -> float:
    mask = np.isfinite(time_since_fill_s) & np.isfinite(values) & (values >= threshold)
    if not np.any(mask):
        return math.nan
    return float(time_since_fill_s[np.flatnonzero(mask)[0]])


def first_freeze_complete_time(time_since_fill_s: np.ndarray, freeze_complete_flag: np.ndarray) -> float:
    mask = np.isfinite(time_since_fill_s) & np.isfinite(freeze_complete_flag) & (freeze_complete_flag >= 0.5)
    if not np.any(mask):
        return math.nan
    return float(time_since_fill_s[np.flatnonzero(mask)[0]])


def nan_to_str(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.6f}"


__all__ = ["first_freeze_complete_time", "first_time_at_or_above", "load_front_columns", "nan_to_str"]

