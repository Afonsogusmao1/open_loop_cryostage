from __future__ import annotations

"""Ambient-temperature models used by the freezing solver."""

from dataclasses import dataclass
import math

import numpy as np

from code_simulation.core.arrays import coerce_float_tuple, validate_strictly_increasing


@dataclass(frozen=True)
class FixedAmbientTemperature:
    """Return a constant ambient temperature, independent of plate temperature."""

    temperature_C: float

    def __post_init__(self) -> None:
        temperature_C = float(self.temperature_C)
        if not math.isfinite(temperature_C):
            raise ValueError("temperature_C must be finite")
        object.__setattr__(self, "temperature_C", temperature_C)

    def __call__(self, plate_temperature_C: float) -> float:
        return float(self.temperature_C)


@dataclass(frozen=True)
class InterpolatedAmbientTemperature:
    """Linearly interpolate ambient temperature from modeled plate/cryostage temperature."""

    cryostage_temperature_C: tuple[float, ...]
    ambient_temperature_C: tuple[float, ...]
    extrapolation: str = "clamp"

    def __post_init__(self) -> None:
        cryostage_temperature_C = coerce_float_tuple(
            self.cryostage_temperature_C,
            name="cryostage_temperature_C",
        )
        ambient_temperature_C = coerce_float_tuple(
            self.ambient_temperature_C,
            name="ambient_temperature_C",
        )
        extrapolation = str(self.extrapolation).strip().lower()

        if len(cryostage_temperature_C) != len(ambient_temperature_C):
            raise ValueError("cryostage_temperature_C and ambient_temperature_C must have the same length")
        if len(cryostage_temperature_C) < 2:
            raise ValueError("at least two calibration points are required")
        validate_strictly_increasing(cryostage_temperature_C, name="cryostage_temperature_C")
        if extrapolation not in {"clamp", "error"}:
            raise ValueError("extrapolation must be either 'clamp' or 'error'")

        object.__setattr__(self, "cryostage_temperature_C", cryostage_temperature_C)
        object.__setattr__(self, "ambient_temperature_C", ambient_temperature_C)
        object.__setattr__(self, "extrapolation", extrapolation)

    def __call__(self, plate_temperature_C: float) -> float:
        plate_temperature_C = float(plate_temperature_C)
        if not math.isfinite(plate_temperature_C):
            raise ValueError("plate_temperature_C must be finite")

        x = np.asarray(self.cryostage_temperature_C, dtype=np.float64)
        y = np.asarray(self.ambient_temperature_C, dtype=np.float64)
        if self.extrapolation == "error" and (
            plate_temperature_C < float(x[0]) or plate_temperature_C > float(x[-1])
        ):
            raise ValueError(
                "plate_temperature_C is outside the calibrated ambient-temperature range "
                f"[{float(x[0]):.6g}, {float(x[-1]):.6g}] C"
            )
        return float(np.interp(plate_temperature_C, x, y))


def describe_ambient_temperature_model(model) -> dict[str, object]:
    if model is None:
        return {"mode": "fixed", "source": "ThermalBCs.T_room_C"}
    if isinstance(model, FixedAmbientTemperature):
        return {
            "mode": "fixed",
            "ambient_temperature_C": float(model.temperature_C),
        }
    if isinstance(model, InterpolatedAmbientTemperature):
        return {
            "mode": "interpolate_from_cryostage",
            "cryostage_temperature_C": [float(value) for value in model.cryostage_temperature_C],
            "ambient_temperature_C": [float(value) for value in model.ambient_temperature_C],
            "extrapolation": str(model.extrapolation),
        }
    return {"mode": type(model).__name__}


__all__ = [
    "FixedAmbientTemperature",
    "InterpolatedAmbientTemperature",
    "describe_ambient_temperature_model",
]
