from __future__ import annotations

import csv
import json
from pathlib import Path


def theta_arg(theta_C: tuple[float, ...]) -> str:
    return ",".join(f"{float(value):.6f}" for value in theta_C)


def bounds_arg(theta_bounds_C: tuple[tuple[float, float], ...]) -> str:
    return ",".join(f"{float(lower_C):.6f}:{float(upper_C):.6f}" for lower_C, upper_C in theta_bounds_C)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_empty_best_theta_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["knot_index", "time_s", "temperature_C"])


__all__ = ["bounds_arg", "read_csv_rows", "read_json", "theta_arg", "write_empty_best_theta_csv"]

