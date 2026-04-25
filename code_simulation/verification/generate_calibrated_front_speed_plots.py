from __future__ import annotations

"""Compatibility wrapper for the calibrated front-speed plotting CLI."""

from code_simulation.verification.calibrated.front_speed_plots import (
    generate_calibrated_front_speed_plots,
    main,
    parse_args,
)


__all__ = [
    "generate_calibrated_front_speed_plots",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
