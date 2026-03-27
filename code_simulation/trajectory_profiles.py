from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


def _coerce_float_tuple(values, *, name: str) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of floats") from exc


def _validate_strictly_increasing(values: tuple[float, ...], *, name: str) -> None:
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class ConstantTemperatureProfile:
    temperature_C: float

    def __call__(self, t_s: float) -> float:
        return float(self.temperature_C)


@dataclass(frozen=True)
class StepTemperatureProfile:
    breakpoints_s: tuple[float, ...]
    values_C: tuple[float, ...]

    def __post_init__(self) -> None:
        breakpoints_s = _coerce_float_tuple(self.breakpoints_s, name="breakpoints_s")
        values_C = _coerce_float_tuple(self.values_C, name="values_C")
        object.__setattr__(self, "breakpoints_s", breakpoints_s)
        object.__setattr__(self, "values_C", values_C)

        if len(values_C) != len(breakpoints_s) + 1:
            raise ValueError("values_C must have exactly len(breakpoints_s) + 1 entries")
        _validate_strictly_increasing(breakpoints_s, name="breakpoints_s")

    def __call__(self, t_s: float) -> float:
        idx = bisect_right(self.breakpoints_s, float(t_s))
        return float(self.values_C[idx])


@dataclass(frozen=True)
class PiecewiseLinearTemperatureProfile:
    knot_times_s: tuple[float, ...]
    knot_temperatures_C: tuple[float, ...]

    def __post_init__(self) -> None:
        knot_times_s = _coerce_float_tuple(self.knot_times_s, name="knot_times_s")
        knot_temperatures_C = _coerce_float_tuple(
            self.knot_temperatures_C,
            name="knot_temperatures_C",
        )
        object.__setattr__(self, "knot_times_s", knot_times_s)
        object.__setattr__(self, "knot_temperatures_C", knot_temperatures_C)

        if len(knot_times_s) == 0:
            raise ValueError("knot_times_s must contain at least one knot")
        if len(knot_times_s) != len(knot_temperatures_C):
            raise ValueError("knot_times_s and knot_temperatures_C must have the same length")
        _validate_strictly_increasing(knot_times_s, name="knot_times_s")

    def __call__(self, t_s: float) -> float:
        t_s = float(t_s)
        if len(self.knot_times_s) == 1 or t_s <= self.knot_times_s[0]:
            return float(self.knot_temperatures_C[0])
        if t_s >= self.knot_times_s[-1]:
            return float(self.knot_temperatures_C[-1])

        idx = bisect_right(self.knot_times_s, t_s)
        t0 = self.knot_times_s[idx - 1]
        t1 = self.knot_times_s[idx]
        T0 = self.knot_temperatures_C[idx - 1]
        T1 = self.knot_temperatures_C[idx]
        alpha = (t_s - t0) / (t1 - t0)
        return float(T0 + alpha * (T1 - T0))


__all__ = [
    "ConstantTemperatureProfile",
    "StepTemperatureProfile",
    "PiecewiseLinearTemperatureProfile",
]
