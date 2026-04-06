#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reachability_constraints import (
    ReachabilityConstraints,
    check_piecewise_linear_trajectory_admissibility,
    conservative_first_entry_time_s,
    conservative_hold_support_s,
    conservative_settling_time_s,
    default_constraints_dir,
    load_reachability_constraints,
    max_transient_window_s,
)


@dataclass(frozen=True)
class ExampleTrajectory:
    name: str
    description: str
    knot_times_s: tuple[float, ...]
    knot_temperatures_C: tuple[float, ...]


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "characterization_constraints" / "admissibility_diagnostics"


def build_example_trajectories() -> list[ExampleTrajectory]:
    return [
        ExampleTrajectory(
            name="feasible_two_regime_profile",
            description="Reaches -10 C with time to settle, holds there within observed support, then reaches -15 C and holds again within empirical hold support.",
            knot_times_s=(0.0, 600.0, 1800.0, 2100.0, 2400.0),
            knot_temperatures_C=(-0.1, -10.0, -10.0, -15.0, -15.0),
        ),
        ExampleTrajectory(
            name="feasible_intermediate_minus12_hold",
            description="Uses a conservative hold-support basis at -15 C to justify an intermediate -12 C hold of limited duration.",
            knot_times_s=(0.0, 700.0, 1300.0, 1900.0),
            knot_temperatures_C=(-0.1, -12.0, -12.0, -18.0),
        ),
        ExampleTrajectory(
            name="infeasible_fast_30s_drop",
            description="Front-loaded cooling request that exceeds the empirical finite-window transient envelope.",
            knot_times_s=(0.0, 30.0, 600.0, 1200.0, 2400.0),
            knot_temperatures_C=(-0.1, -6.0, -10.0, -15.0, -20.0),
        ),
        ExampleTrajectory(
            name="infeasible_early_minus15_arrival",
            description="Attempts to begin a -15 C hold before the conservative settling time measured in the cryostage characterization.",
            knot_times_s=(0.0, 360.0, 720.0, 2400.0),
            knot_temperatures_C=(-0.1, -15.0, -20.0, -20.0),
        ),
        ExampleTrajectory(
            name="infeasible_long_minus20_hold",
            description="Reaches -20 C in time but requests a hold much longer than the empirical freezing-run plate support.",
            knot_times_s=(0.0, 600.0, 1500.0, 2700.0),
            knot_temperatures_C=(-0.1, -10.0, -20.0, -20.0),
        ),
        ExampleTrajectory(
            name="infeasible_warming_segment",
            description="Contains an unsupported warming segment between adjacent knots.",
            knot_times_s=(0.0, 600.0, 1200.0, 1800.0, 2400.0),
            knot_temperatures_C=(-0.1, -6.0, -4.0, -12.0, -16.0),
        ),
        ExampleTrajectory(
            name="infeasible_below_supported_target",
            description="Requests a final knot colder than the characterized and observed hardware-support range.",
            knot_times_s=(0.0, 600.0, 1200.0, 1800.0, 2400.0),
            knot_temperatures_C=(-0.1, -5.0, -10.0, -15.0, -21.0),
        ),
    ]


def evaluate_examples(
    examples: list[ExampleTrajectory],
    constraints: ReachabilityConstraints,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for example in examples:
        report = check_piecewise_linear_trajectory_admissibility(
            example.knot_times_s,
            example.knot_temperatures_C,
            constraints=constraints,
            require_monotone_nonincreasing=True,
        )
        results.append(
            {
                "name": example.name,
                "description": example.description,
                "knot_times_s": [float(value) for value in example.knot_times_s],
                "knot_temperatures_C": [float(value) for value in example.knot_temperatures_C],
                "is_admissible": bool(report.is_admissible),
                "failure_summary": report.failure_summary(),
                "report": report.to_dict(),
            }
        )
    return results


def write_json_report(
    *,
    out_path: Path,
    constraints: ReachabilityConstraints,
    example_results: list[dict[str, Any]],
) -> None:
    transient_targets = [float(value) for value in constraints.characterized_targets_C]
    hold_targets = [float(value) for value in constraints.observed_hold_targets_C]
    payload = {
        "constraints_dir": str(constraints.constraints_dir.resolve()),
        "constraint_files": {
            "transient_summary_json": str(constraints.summary_json_path.resolve()),
            "transient_aggregated_metrics_csv": str(constraints.aggregated_csv_path.resolve()),
            "transient_window_envelope_csv": str(constraints.window_csv_path.resolve()),
            "hold_summary_json": str(constraints.hold_summary_json_path.resolve()),
            "hold_support_grid_csv": str(constraints.hold_support_csv_path.resolve()),
        },
        "admissibility_rules": {
            "monotone_nonincreasing_hard_enforced": True,
            "warming_supported": bool(constraints.warming_supported),
            "finite_window_transient_rule": (
                "For segment durations up to the largest characterized transient window, each segment must stay within the conservative empirical finite-window cooling envelope. "
                "This rule is not extrapolated beyond the characterized window horizon."
            ),
            "arrival_rule": (
                "If a segment ends at a characterized cold target, cumulative arrival time must be no earlier than the conservative first-entry time into the ±0.5 C band. "
                "Interpolation between characterized targets uses the worse of the two bracketing targets."
            ),
            "settling_rule": (
                "If a hold begins after reaching a cold target, cumulative arrival time must also be no earlier than the conservative settling time into the ±0.5 C band."
            ),
            "hold_rule": (
                "After a cold target is reached and settled, the remaining time spent within the target band must not exceed the empirical long-duration plate-hold support extracted from freezing-run T_cal telemetry."
            ),
            "hold_support_basis_rule": constraints.hold_support_basis_rule,
            "outside_support_rule": (
                "Targets colder than the characterized and observed support remain rejected rather than being treated as validated behavior."
            ),
            "transient_characterized_window_limit_s": float(max_transient_window_s(constraints)),
        },
        "transient_summary_by_target_C": {
            f"{target:.1f}": {
                "conservative_first_entry_time_s": float(conservative_first_entry_time_s(target, constraints)),
                "conservative_settling_time_s": float(conservative_settling_time_s(target, constraints)),
            }
            for target in transient_targets
        },
        "hold_support_by_target_C": {
            f"{target:.1f}": {
                "supported_hold_duration_s": float(constraints.hold_supported_duration_s_by_target_C[target]),
            }
            for target in hold_targets
        },
        "examples": example_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_summary(
    *,
    out_path: Path,
    constraints: ReachabilityConstraints,
    example_results: list[dict[str, Any]],
    figure_path: Path,
) -> None:
    lines = [
        "# Revised Admissibility Diagnostics",
        "",
        "## Loaded constraint files",
        f"- Transient summary JSON: `{constraints.summary_json_path}`",
        f"- Transient aggregated metrics CSV: `{constraints.aggregated_csv_path}`",
        f"- Finite-window envelope CSV: `{constraints.window_csv_path}`",
        f"- Hold summary JSON: `{constraints.hold_summary_json_path}`",
        f"- Hold support grid CSV: `{constraints.hold_support_csv_path}`",
        "",
        "## Enforced rules",
        "- Monotone non-increasing cooling remains hard-enforced in the active workflow.",
        "- Warming remains unsupported beyond the ±0.5 C practical tolerance band.",
        f"- The finite-window transient cooling envelope is enforced only up to `{max_transient_window_s(constraints):.0f} s`; the old post-window average-rate extrapolation is no longer used as the main long-duration rule.",
        "- Reaching a cold target still requires cumulative arrival no earlier than the conservative first-entry time into the ±0.5 C band from the step-response characterization.",
        "- Beginning a hold at that target additionally requires cumulative arrival no earlier than the conservative settling time from the step-response characterization.",
        f"- Long-duration hold feasibility is then capped by empirical freezing-run plate telemetry, using the conservative basis rule: {constraints.hold_support_basis_rule}",
        "",
        "## Example trajectories",
    ]
    for result in example_results:
        status = "PASS" if result["is_admissible"] else "FAIL"
        lines.append(f"- `{result['name']}`: {status}. {result['failure_summary']}")
    lines.extend(
        [
            "",
            "## Plot",
            f"- Two-regime constraint overview: `{figure_path}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_two_regime_constraints(
    *,
    out_path: Path,
    constraints: ReachabilityConstraints,
    example_results: list[dict[str, Any]],
) -> None:
    transient_targets = np.asarray(constraints.characterized_targets_C, dtype=np.float64)
    first_entry = np.asarray([conservative_first_entry_time_s(float(t), constraints) for t in transient_targets], dtype=np.float64)
    settling = np.asarray([conservative_settling_time_s(float(t), constraints) for t in transient_targets], dtype=np.float64)

    hold_targets = np.asarray(constraints.characterized_targets_C, dtype=np.float64)
    hold_support = np.asarray([conservative_hold_support_s(float(t), constraints)[0] for t in hold_targets], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    left, right = axes

    left.plot(first_entry, transient_targets, color="#4477AA", linewidth=2.2, marker="o", label="First-entry time")
    left.plot(settling, transient_targets, color="#CC6677", linewidth=2.2, marker="s", label="Settling time")

    right.plot(hold_support, hold_targets, color="#228833", linewidth=2.2, marker="o", label="Supported hold duration")

    cmap = plt.get_cmap("tab10")
    for idx, example in enumerate(example_results):
        color = cmap(idx % 10)
        marker = "o" if example["is_admissible"] else "X"
        segment_results = example["report"]["segment_results"]
        arrival_times = []
        arrival_targets = []
        hold_times = []
        hold_targets_used = []
        for segment in segment_results:
            if segment["characterized_target_check_applied"]:
                arrival_times.append(float(segment["cumulative_time_to_end_s"]))
                arrival_targets.append(float(segment["T_end_C"]))
            if segment["empirical_hold_check_applied"]:
                hold_times.append(float(segment["requested_hold_duration_s"]))
                hold_targets_used.append(float(segment["T_end_C"]))

        if arrival_times:
            left.scatter(arrival_times, arrival_targets, color=color, edgecolor="black", linewidth=0.5, marker=marker, s=70)
        if hold_times:
            right.scatter(hold_times, hold_targets_used, color=color, edgecolor="black", linewidth=0.5, marker=marker, s=70)

    left.set_title("Transient reachability constraints")
    left.set_xlabel("Cumulative arrival time [s]")
    left.set_ylabel("Target temperature [C]")
    left.grid(True, alpha=0.25)
    left.legend(fontsize=8, loc="best")

    right.set_title("Long-duration hold constraints")
    right.set_xlabel("Requested hold duration after arrival [s]")
    right.set_ylabel("Target temperature [C]")
    right.grid(True, alpha=0.25)
    right.legend(fontsize=8, loc="best")

    fig.suptitle("Two-regime admissibility: transient reachability plus empirical hold support", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revised two-regime trajectory admissibility diagnostics.")
    parser.add_argument(
        "--constraints-dir",
        default=str(default_constraints_dir(Path(__file__).resolve().parent)),
        help="Directory containing the Stage 1 transient characterization artifacts. The matching Stage 2 hold artifacts are resolved as a sibling directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(default_output_dir()),
        help="Directory where revised admissibility diagnostic artifacts are written.",
    )
    args = parser.parse_args()

    constraints = load_reachability_constraints(args.constraints_dir)
    out_dir = Path(args.out_dir).resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    examples = build_example_trajectories()
    example_results = evaluate_examples(examples, constraints)

    json_path = out_dir / "revised_admissibility_summary.json"
    markdown_path = out_dir / "revised_admissibility_summary.md"
    example_json_path = out_dir / "example_trajectory_admissibility_report.json"
    figure_path = figures_dir / "two_regime_constraints_overview.png"

    write_json_report(out_path=json_path, constraints=constraints, example_results=example_results)
    example_json_path.write_text(json.dumps(example_results, indent=2), encoding="utf-8")
    plot_two_regime_constraints(out_path=figure_path, constraints=constraints, example_results=example_results)
    write_markdown_summary(
        out_path=markdown_path,
        constraints=constraints,
        example_results=example_results,
        figure_path=figure_path,
    )

    print(f"Wrote diagnostics to {out_dir}")
    print(f"Revised summary JSON: {json_path}")
    print(f"Example report JSON: {example_json_path}")
    print(f"Markdown summary: {markdown_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
