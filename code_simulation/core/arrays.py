from __future__ import annotations

import math

import numpy as np


def as_float_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def coerce_float_tuple(
    values,
    *,
    name: str,
    allow_empty: bool = False,
    require_finite: bool = True,
) -> tuple[float, ...]:
    try:
        out = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of floats") from exc
    if not allow_empty and len(out) == 0:
        raise ValueError(f"{name} must contain at least one entry")
    if require_finite and not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def validate_strictly_increasing(values: tuple[float, ...] | np.ndarray, *, name: str) -> None:
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            raise ValueError(f"{name} must be strictly increasing")


__all__ = ["as_float_array", "coerce_float_tuple", "validate_strictly_increasing"]
