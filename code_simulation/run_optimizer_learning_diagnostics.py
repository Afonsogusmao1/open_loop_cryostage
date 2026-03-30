#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryostage_model import DEFAULT_CRYOSTAGE_PARAMS, simulate_plate_temperature
from open_loop_problem import build_reference_profile_from_theta, load_front_csv
from run_open_loop_optimization import build_problem_config


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8.8, 4.8),
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


@dataclass(frozen=True)
class HistoryEntry:
    evaluation_index: int
    case_name: str
    objective_value: float
    is_valid: bool
    out_dir: Path
    error_message: str
    theta: tuple[float, ...]

    @property
    def front_path(self) -> Path:
        return self.out_dir / f"{self.case_name}_front.csv"

    @property
    def probes_path(self) -> Path:
        return self.out_dir / f"{self.case_name}_probes.csv"


@dataclass(frozen=True)
class StudyData:
    study_dir: Path
    study_name: str
    summary: dict[str, str]
    config: object
    history: tuple[HistoryEntry, ...]
    valid_history: tuple[HistoryEntry, ...]
    best_entry: HistoryEntry


def _read_summary_txt(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if ": " not in line:
                continue
            key, value = line.rstrip("\n").split(": ", 1)
            summary[key] = value
    return summary


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean text, got {raw!r}")


def _build_config_from_summary(summary: dict[str, str], *, t_ignore_s: float):
    base_config = build_problem_config(formulation="legacy_exploratory")
    knot_times_s = tuple(float(value) for value in ast.literal_eval(summary["knot_times_s"]))
    horizon_s = float(summary["horizon_s"])
    bounds = tuple(float(value) for value in ast.literal_eval(summary["T_ref_bounds_C"]))
    monotone = _parse_bool(summary["require_monotone_nonincreasing"])
    return replace(
        base_config,
        horizon_s=horizon_s,
        knot_times_s=knot_times_s,
        T_ref_bounds_C=bounds,
        require_monotone_nonincreasing=monotone,
        t_ignore_s=float(t_ignore_s),
    )


def _load_history(path: Path) -> tuple[HistoryEntry, ...]:
    entries: list[HistoryEntry] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            entries.append(
                HistoryEntry(
                    evaluation_index=int(row["evaluation_index"]),
                    case_name=row["case_name"],
                    objective_value=float(row["objective_value"]),
                    is_valid=(row["is_valid"] == "1"),
                    out_dir=Path(row["out_dir"]),
                    error_message=row["error_message"],
                    theta=tuple(float(value) for value in ast.literal_eval(row["theta_json"])),
                )
            )
    return tuple(entries)


def _load_study(study_dir: Path, *, t_ignore_s: float) -> StudyData:
    summary = _read_summary_txt(study_dir / "study_summary.txt")
    history = _load_history(study_dir / "evaluation_history.csv")
    valid_history = tuple(entry for entry in history if entry.is_valid and math.isfinite(entry.objective_value))
    if not valid_history:
        raise RuntimeError(f"{study_dir} does not contain any valid evaluations")
    best_entry = min(valid_history, key=lambda entry: entry.objective_value)
    return StudyData(
        study_dir=study_dir,
        study_name=summary["study_name"],
        summary=summary,
        config=_build_config_from_summary(summary, t_ignore_s=t_ignore_s),
        history=history,
        valid_history=valid_history,
        best_entry=best_entry,
    )


def _select_representative_entries(study: StudyData) -> list[tuple[str, HistoryEntry]]:
    valid_entries = list(study.valid_history)
    label_map: dict[int, list[str]] = {}

    candidate_labels = [
        ("initial", valid_entries[0]),
        ("mid_1", valid_entries[len(valid_entries) // 3]),
        ("mid_2", valid_entries[(2 * len(valid_entries)) // 3]),
        ("late", valid_entries[-1]),
        ("best", study.best_entry),
    ]
    for label, entry in candidate_labels:
        label_map.setdefault(entry.evaluation_index, []).append(label)

    ordered_entries: list[tuple[str, HistoryEntry]] = []
    seen: set[int] = set()
    for label, entry in candidate_labels:
        if entry.evaluation_index in seen:
            continue
        seen.add(entry.evaluation_index)
        combined_label = "/".join(label_map[entry.evaluation_index])
        ordered_entries.append((combined_label, entry))
    return ordered_entries


def _objective_plot_y_for_invalid(valid_values: np.ndarray) -> float:
    if valid_values.size == 0:
        return 1.0
    ymax = float(np.max(valid_values))
    return ymax * 1.05 if ymax > 0.0 else 1.0


def _plot_objective_history(study: StudyData, selected_entries: list[tuple[str, HistoryEntry]], out_path: Path) -> None:
    eval_indices = np.asarray([entry.evaluation_index for entry in study.history], dtype=np.float64)
    objective_values = np.asarray(
        [entry.objective_value if entry.is_valid and math.isfinite(entry.objective_value) else np.nan for entry in study.history],
        dtype=np.float64,
    )
    valid_mask = np.isfinite(objective_values)
    valid_values = objective_values[valid_mask]
    best_so_far = np.minimum.accumulate(
        np.asarray(
            [
                entry.objective_value if entry.is_valid and math.isfinite(entry.objective_value) else math.inf
                for entry in study.history
            ],
            dtype=np.float64,
        )
    )

    fig, ax = plt.subplots()
    if np.any(valid_mask):
        ax.plot(
            eval_indices[valid_mask],
            objective_values[valid_mask],
            marker="o",
            linewidth=1.5,
            markersize=4.0,
            label="valid objective",
        )

    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        invalid_y = np.full(np.count_nonzero(invalid_mask), _objective_plot_y_for_invalid(valid_values))
        ax.scatter(eval_indices[invalid_mask], invalid_y, marker="x", label="invalid")

    finite_best = np.where(np.isfinite(best_so_far), best_so_far, np.nan)
    ax.step(eval_indices, finite_best, where="post", linestyle="--", linewidth=1.5, label="best so far")

    for label, entry in selected_entries:
        ax.scatter(
            [entry.evaluation_index],
            [entry.objective_value],
            s=55.0,
            zorder=3,
            label=f"{label} (eval {entry.evaluation_index:04d})",
        )

    ax.set_title(f"{study.study_name}: Objective vs Evaluation")
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Objective Value")
    if valid_values.size > 0 and np.all(valid_values > 0.0):
        spread = float(np.max(valid_values) / np.min(valid_values))
        if spread >= 10.0:
            ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(out_path)
    plt.close(fig)


def _reconstruct_temperature_curves(entry: HistoryEntry, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = config.cryostage_time_grid_s()
    profile = build_reference_profile_from_theta(entry.theta, config)
    T_ref_C = np.asarray([float(profile(float(ti))) for ti in time_s], dtype=np.float64)
    T_plate_C = simulate_plate_temperature(
        time_s=time_s,
        T_ref_profile_C=profile,
        params=DEFAULT_CRYOSTAGE_PARAMS,
        T_plate0_C=float(config.solver_kwargs["bcs"].T_room_C),
    )
    return time_s, T_ref_C, T_plate_C


def _plot_temperature_family(
    *,
    study: StudyData,
    selected_entries: list[tuple[str, HistoryEntry]],
    out_path: Path,
    which: str,
) -> None:
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(selected_entries)))
    for color, (label, entry) in zip(colors, selected_entries):
        time_s, T_ref_C, T_plate_C = _reconstruct_temperature_curves(entry, study.config)
        values = T_ref_C if which == "T_ref" else T_plate_C
        ax.plot(
            time_s,
            values,
            color=color,
            linewidth=2.0,
            label=f"eval {entry.evaluation_index:04d} {label}  J={entry.objective_value:.3e}",
        )

    ax.set_title(f"{study.study_name}: {which} Evolution")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend(loc="best", fontsize=8.2)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_front_family(study: StudyData, selected_entries: list[tuple[str, HistoryEntry]], out_path: Path) -> None:
    fig, ax = plt.subplots()
    colors = plt.cm.plasma(np.linspace(0.12, 0.9, len(selected_entries)))
    for color, (label, entry) in zip(colors, selected_entries):
        front = load_front_csv(entry.front_path)
        ax.plot(
            front.time_since_fill_s,
            1000.0 * front.z_front_m,
            color=color,
            linewidth=2.0,
            label=f"eval {entry.evaluation_index:04d} {label}  J={entry.objective_value:.3e}",
        )

    ax.set_title(f"{study.study_name}: Front Evolution")
    ax.set_xlabel("Time Since Fill (s)")
    ax.set_ylabel("Front Position (mm)")
    ax.legend(loc="best", fontsize=8.2)
    fig.savefig(out_path)
    plt.close(fig)


def _write_selected_csv(study: StudyData, selected_entries: list[tuple[str, HistoryEntry]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "evaluation_index",
                "objective_value",
                "theta_json",
                "front_path",
                "probes_path",
            ]
        )
        for label, entry in selected_entries:
            writer.writerow(
                [
                    label,
                    entry.evaluation_index,
                    f"{entry.objective_value:.12e}",
                    repr(list(entry.theta)),
                    str(entry.front_path),
                    str(entry.probes_path),
                ]
            )


def _write_study_summary(study: StudyData, selected_entries: list[tuple[str, HistoryEntry]], out_path: Path) -> None:
    initial_entry = study.valid_history[0]
    best_entry = study.best_entry
    initial_J = float(initial_entry.objective_value)
    best_J = float(best_entry.objective_value)
    improvement_pct = 100.0 * (initial_J - best_J) / initial_J if initial_J > 0.0 else 0.0

    with out_path.open("w") as f:
        f.write(f"study_name: {study.study_name}\n")
        f.write(f"total_evaluations: {len(study.history)}\n")
        f.write(f"valid_evaluations: {len(study.valid_history)}\n")
        f.write(f"invalid_evaluations: {len(study.history) - len(study.valid_history)}\n")
        f.write(f"initial_valid_eval: {initial_entry.evaluation_index}\n")
        f.write(f"initial_valid_J: {initial_J:.12e}\n")
        f.write(f"best_eval: {best_entry.evaluation_index}\n")
        f.write(f"best_J: {best_J:.12e}\n")
        f.write(f"improvement_percent_from_initial_valid: {improvement_pct:.6f}\n")
        f.write("selected_evaluations:\n")
        for label, entry in selected_entries:
            f.write(
                f"  - {label}: eval={entry.evaluation_index:04d}, "
                f"J={entry.objective_value:.12e}, theta={list(entry.theta)}\n"
            )


def _plot_comparison_best_so_far(studies: list[StudyData], out_path: Path) -> None:
    fig, ax = plt.subplots()
    for study in studies:
        eval_indices = np.asarray([entry.evaluation_index for entry in study.history], dtype=np.float64)
        best_so_far = np.minimum.accumulate(
            np.asarray(
                [
                    entry.objective_value if entry.is_valid and math.isfinite(entry.objective_value) else math.inf
                    for entry in study.history
                ],
                dtype=np.float64,
            )
        )
        finite_best = np.where(np.isfinite(best_so_far), best_so_far, np.nan)
        ax.step(eval_indices, finite_best, where="post", linewidth=2.0, label=study.study_name)

    ax.set_title("360 s Studies: Best-So-Far Objective vs Evaluation")
    ax.set_xlabel("Evaluation Index")
    ax.set_ylabel("Best Objective So Far")
    ax.legend(loc="best", fontsize=8.5)
    fig.savefig(out_path)
    plt.close(fig)


def _write_comparison_summary(studies: list[StudyData], out_path: Path) -> None:
    with out_path.open("w") as f:
        for study in studies:
            initial_entry = study.valid_history[0]
            best_entry = study.best_entry
            improvement_pct = 100.0 * (initial_entry.objective_value - best_entry.objective_value) / initial_entry.objective_value
            f.write(f"study_name: {study.study_name}\n")
            f.write(f"  initial_valid_eval: {initial_entry.evaluation_index}\n")
            f.write(f"  initial_valid_J: {initial_entry.objective_value:.12e}\n")
            f.write(f"  best_eval: {best_entry.evaluation_index}\n")
            f.write(f"  best_J: {best_entry.objective_value:.12e}\n")
            f.write(f"  improvement_percent_from_initial_valid: {improvement_pct:.6f}\n")
            f.write(f"  best_theta: {list(best_entry.theta)}\n")
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process saved open-loop studies into optimizer-learning diagnostics.")
    parser.add_argument(
        "--study-dir",
        action="append",
        required=True,
        help="Path to a saved open-loop study directory. Repeat this flag for multiple studies.",
    )
    parser.add_argument(
        "--t-ignore-s",
        type=float,
        default=0.0,
        help="Objective ignored-initial-period used when reconstructing study configs.",
    )
    parser.add_argument(
        "--comparison-dir-name",
        default="optimizer_learning_h360_k5",
        help="Folder name for cross-study comparison artifacts under results/open_loop_study.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()

    study_dirs = [Path(path).resolve() for path in args.study_dir]
    studies = [_load_study(study_dir, t_ignore_s=args.t_ignore_s) for study_dir in study_dirs]

    for study in studies:
        analysis_dir = study.study_dir / "analysis" / "optimizer_learning"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        selected_entries = _select_representative_entries(study)

        _plot_objective_history(study, selected_entries, analysis_dir / "objective_vs_evaluation.png")
        _plot_temperature_family(
            study=study,
            selected_entries=selected_entries,
            out_path=analysis_dir / "T_ref_evolution.png",
            which="T_ref",
        )
        _plot_temperature_family(
            study=study,
            selected_entries=selected_entries,
            out_path=analysis_dir / "T_plate_evolution.png",
            which="T_plate",
        )
        _plot_front_family(study, selected_entries, analysis_dir / "z_front_evolution.png")
        _write_selected_csv(study, selected_entries, analysis_dir / "selected_evaluations.csv")
        _write_study_summary(study, selected_entries, analysis_dir / "learning_summary.txt")

        print(f"{study.study_name}: wrote {analysis_dir}")

    comparison_dir = studies[0].study_dir.parent / args.comparison_dir_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _plot_comparison_best_so_far(studies, comparison_dir / "comparison_best_so_far_vs_evaluation.png")
    _write_comparison_summary(studies, comparison_dir / "comparison_summary.txt")
    print(f"comparison: wrote {comparison_dir}")


if __name__ == "__main__":
    main()
