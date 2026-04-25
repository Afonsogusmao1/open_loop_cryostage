#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility wrapper for the calibrated rho(T) suite CLI."""

from code_simulation.verification.calibrated.run_suite import main, parse_args


__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
