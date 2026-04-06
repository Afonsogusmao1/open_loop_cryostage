#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from run_open_loop_optimization import main as run_open_loop_optimization_main


DEFAULT_THETA_BOUNDS = "-0.5:0.0,-9.0:-3.0,-15.0:-8.0,-18.0:-12.0,-20.0:-14.0"
DEFAULT_OUT_ROOT_DIR = Path(__file__).resolve().parent / "results" / "open_loop_bayesian_optimization"


def main() -> None:
    run_open_loop_optimization_main(
        [
            "--run-name",
            "bo_smoke_full_process_k5_seed17_init2_iter2",
            "--out-root-dir",
            str(DEFAULT_OUT_ROOT_DIR),
            "--formulation",
            "full_process_article",
            "--num-knots",
            "5",
            "--method",
            "bayesian-optimization",
            "--seed",
            "17",
            "--init-points",
            "2",
            "--n-iter",
            "2",
            "--acq-kind",
            "ucb",
            "--kappa",
            "2.0",
            "--xi",
            "0.0",
            f"--theta-bounds={DEFAULT_THETA_BOUNDS}",
            "--infeasible-objective-penalty",
            "1000000",
            "--overwrite",
            "--smoke-test-note",
            "Short fixed-knot Bayesian-optimization smoke test with the canonical 5-knot full-process formulation.",
        ]
    )


if __name__ == "__main__":
    main()
