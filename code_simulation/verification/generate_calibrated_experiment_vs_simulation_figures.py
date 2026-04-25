from __future__ import annotations

"""Compatibility wrapper for the calibrated comparison plotting CLI."""

from code_simulation.verification.calibrated.experiment_vs_simulation import (
    generate_calibrated_experiment_vs_simulation_figures,
    main,
    parse_args,
)


__all__ = [
    "generate_calibrated_experiment_vs_simulation_figures",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
