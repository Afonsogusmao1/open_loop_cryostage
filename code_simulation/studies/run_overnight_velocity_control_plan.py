#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import tomllib

from code_simulation.core.paths import (
    project_root,
    velocity_control_diagnostic_dir,
    velocity_control_results_dir,
    results_dir,
)


DEFAULT_OUTPUT_ROOT = results_dir("archive", "overnight_velocity_control")
DEFAULT_THREE_KNOT_STUDY_ROOT = velocity_control_diagnostic_dir(3)
DEFAULT_BO_RESULTS_ROOT = velocity_control_results_dir()
DEFAULT_FINE_RESULTS_ROOT = velocity_control_results_dir()
DEFAULT_COARSE_WORKERS = 2
DEFAULT_FINE_WORKERS = 1
TARGETS_PHASE4_MM_S = (0.006, 0.010)
SEEDS_PHASE4 = (17, 29)
THETA0_N4_C = (0.0, -7.0, -14.0, -21.0)
TARGET_TOL_MM_S = 5.0e-7


def _coarse_run_dir(root: Path, *, num_knots: int, run_name: str) -> Path:
    return Path(root) / f"n{int(num_knots)}" / "coarse" / run_name


def _fine_run_dir(root: Path, *, num_knots: int, run_name: str) -> Path:
    return Path(root) / f"n{int(num_knots)}" / "fine" / run_name


@dataclass(frozen=True)
class QueueJob:
    phase: str
    job_kind: str
    run_name: str
    cwd: Path
    argv: tuple[str, ...]
    simulation_profile: str | None
    target_front_speed_mm_s: float | None
    num_knots: int | None
    knot_time_schedule: str | None
    seed: int | None
    output_dir: Path | None
    source_path: Path | None = None
    notes: str = ""


@dataclass(frozen=True)
class ExecutionRecord:
    phase: str
    job_kind: str
    run_name: str
    status: str
    returncode: int | None
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    target_front_speed_mm_s: float | None
    num_knots: int | None
    knot_time_schedule: str | None
    seed: int | None
    simulation_profile: str | None
    reused_existing_result: int
    log_path: str
    command: str
    notes: str


@dataclass(frozen=True)
class CoarseSelection:
    target_front_speed_mm_s: float
    run_name: str
    seed: int | None
    schedule: str
    simulation_profile: str
    objective_value: float
    achieved_direct_speed_mm_s: float
    direct_speed_relative_error_pct: float
    rmse_plate_error_C: float
    theta_C: tuple[float, ...]
    run_dir: Path


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_speed_tag(value_mm_s: float) -> str:
    return f"{float(value_mm_s):0.3f}".replace(".", "p")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _extract_flag(argv: tuple[str, ...], flag: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _normalize_python_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    if not argv:
        return argv
    if argv[0] == "python":
        return (sys.executable, *argv[1:])
    return argv


def _bo_run_complete(run_dir: Path) -> bool:
    required = (
        run_dir / "effective_config.toml",
        run_dir / "bo_history.csv",
        run_dir / "best_tracking_summary.csv",
        run_dir / "best_theta_profile.csv",
        run_dir / "plate_tracking_summary.csv",
    )
    return all(path.exists() for path in required)


def _fine_run_complete(run_dir: Path) -> bool:
    required = (
        run_dir / "effective_config.toml",
        run_dir / "velocity_tracking_summary.csv",
        run_dir / "thermocouple_interval_speeds.csv",
        run_dir / "plate_tracking_summary.csv",
    )
    return all(path.exists() for path in required)


def _parse_shell_python_commands(script_path: Path) -> list[tuple[Path, tuple[str, ...]]]:
    commands: list[tuple[Path, tuple[str, ...]]] = []
    cwd = project_root()
    lines = script_path.read_text().splitlines()
    idx = 0
    while idx < len(lines):
        raw_line = lines[idx].rstrip()
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        if stripped.startswith("cd "):
            cwd = Path(stripped[3:].strip())
            idx += 1
            continue
        if stripped.startswith("python -m "):
            parts: list[str] = []
            while idx < len(lines):
                current = lines[idx].rstrip()
                current_stripped = current.strip()
                if current_stripped.endswith("\\"):
                    parts.append(current_stripped[:-1].strip())
                    idx += 1
                    continue
                parts.append(current_stripped)
                idx += 1
                break
            argv = tuple(shlex.split(" ".join(parts)))
            commands.append((cwd, _normalize_python_argv(argv)))
            continue
        idx += 1
    return commands


def _build_phase1_jobs(
    *,
    uniform_commands_path: Path,
    bo_results_root: Path,
) -> list[QueueJob]:
    jobs: list[QueueJob] = []
    for cwd, argv in _parse_shell_python_commands(uniform_commands_path):
        run_name = _extract_flag(argv, "--run-name")
        if not run_name:
            continue
        output_dir = _coarse_run_dir(
            bo_results_root,
            num_knots=_parse_optional_int(_extract_flag(argv, "--num-knots")) or 0,
            run_name=run_name,
        )
        jobs.append(
            QueueJob(
                phase="phase1_close_3k_uniform_seed41",
                job_kind="coarse_bo",
                run_name=run_name,
                cwd=cwd,
                argv=argv,
                simulation_profile=_extract_flag(argv, "--simulation-profile"),
                target_front_speed_mm_s=_parse_optional_float(_extract_flag(argv, "--target-front-speed-mm-s")),
                num_knots=_parse_optional_int(_extract_flag(argv, "--num-knots")),
                knot_time_schedule=_extract_flag(argv, "--knot-time-schedule"),
                seed=_parse_optional_int(_extract_flag(argv, "--seed")),
                output_dir=output_dir,
                source_path=uniform_commands_path,
                notes="missing_3k_uniform_seed41",
            )
        )
    return jobs


def _build_phase2_job() -> QueueJob:
    argv = (
        sys.executable,
        "-m",
        "code_simulation.studies.run_bo_3knot_target_and_spacing_study",
        "--overwrite",
    )
    return QueueJob(
        phase="phase2_refresh_3k_study",
        job_kind="study_refresh",
        run_name="refresh_bo_3knot_target_and_spacing_study",
        cwd=project_root(),
        argv=argv,
        simulation_profile=None,
        target_front_speed_mm_s=None,
        num_knots=None,
        knot_time_schedule=None,
        seed=None,
        output_dir=None,
        source_path=None,
        notes="refresh_3knot_decision_summary",
    )


def _build_phase3_jobs(*, bo_results_root: Path) -> list[QueueJob]:
    jobs: list[QueueJob] = []
    for target_mm_s in TARGETS_PHASE4_MM_S:
        target_tag = _format_speed_tag(target_mm_s)
        for seed in SEEDS_PHASE4:
            run_name = f"bo_v{target_tag}_n4_uniform_seed{seed}"
            argv = (
                sys.executable,
                "-m",
                "code_simulation.optimization.run_velocity_control_bo",
                "--target-front-speed-mm-s",
                f"{target_mm_s:.3f}",
                "--num-knots",
                "4",
                "--knot-time-schedule",
                "uniform",
                f"--theta0-c={','.join(f'{value:g}' for value in THETA0_N4_C)}",
                "--seed",
                str(seed),
                "--simulation-profile",
                "optimization",
                "--run-name",
                run_name,
                "--overwrite",
            )
            jobs.append(
                QueueJob(
                    phase="phase3_n4_coarse_pilot",
                    job_kind="coarse_bo",
                    run_name=run_name,
                    cwd=project_root(),
                    argv=argv,
                    simulation_profile="optimization",
                    target_front_speed_mm_s=target_mm_s,
                    num_knots=4,
                    knot_time_schedule="uniform",
                    seed=seed,
                    output_dir=_coarse_run_dir(bo_results_root, num_knots=4, run_name=run_name),
                    source_path=None,
                    notes="pilot_n4_uniform",
                )
            )
    return jobs


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _job_is_complete(job: QueueJob) -> bool:
    if job.job_kind == "coarse_bo" and job.output_dir is not None:
        return _bo_run_complete(job.output_dir)
    if job.job_kind == "fine_confirmation" and job.output_dir is not None:
        return _fine_run_complete(job.output_dir)
    return False


def _queue_manifest_rows(jobs: Iterable[QueueJob]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in jobs:
        existing_result = int(_job_is_complete(job))
        rows.append(
            {
                "phase": job.phase,
                "job_kind": job.job_kind,
                "run_name": job.run_name,
                "target_front_speed_mm_s": _format_optional(job.target_front_speed_mm_s),
                "num_knots": _format_optional(job.num_knots),
                "knot_time_schedule": job.knot_time_schedule or "",
                "seed": _format_optional(job.seed),
                "simulation_profile": job.simulation_profile or "",
                "existing_result_detected": existing_result,
                "output_dir": str(job.output_dir) if job.output_dir is not None else "",
                "source_path": str(job.source_path) if job.source_path is not None else "",
                "command": shlex.join(job.argv),
                "notes": job.notes,
            }
        )
    return rows


def _format_optional(value: object | None) -> str:
    if value is None:
        return ""
    return str(value)


def _run_job(
    *,
    job: QueueJob,
    logs_dir: Path,
    dry_run: bool,
) -> ExecutionRecord:
    start = datetime.now(timezone.utc)
    log_path = logs_dir / f"{job.phase}__{job.run_name}.log"
    command_text = shlex.join(job.argv)
    if dry_run:
        print(f"[dry-run] {job.phase}: {command_text}")
        end = datetime.now(timezone.utc)
        return ExecutionRecord(
            phase=job.phase,
            job_kind=job.job_kind,
            run_name=job.run_name,
            status="planned",
            returncode=None,
            start_time_utc=start.replace(microsecond=0).isoformat(),
            end_time_utc=end.replace(microsecond=0).isoformat(),
            duration_s=0.0,
            target_front_speed_mm_s=job.target_front_speed_mm_s,
            num_knots=job.num_knots,
            knot_time_schedule=job.knot_time_schedule,
            seed=job.seed,
            simulation_profile=job.simulation_profile,
            reused_existing_result=0,
            log_path=str(log_path),
            command=command_text,
            notes=job.notes,
        )
    if _job_is_complete(job):
        end = datetime.now(timezone.utc)
        return ExecutionRecord(
            phase=job.phase,
            job_kind=job.job_kind,
            run_name=job.run_name,
            status="completed",
            returncode=0,
            start_time_utc=start.replace(microsecond=0).isoformat(),
            end_time_utc=end.replace(microsecond=0).isoformat(),
            duration_s=0.0,
            target_front_speed_mm_s=job.target_front_speed_mm_s,
            num_knots=job.num_knots,
            knot_time_schedule=job.knot_time_schedule,
            seed=job.seed,
            simulation_profile=job.simulation_profile,
            reused_existing_result=1,
            log_path=str(log_path),
            command=command_text,
            notes=f"{job.notes}; existing_result_reused",
        )
    logs_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        process = subprocess.run(
            job.argv,
            cwd=str(job.cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    end = datetime.now(timezone.utc)
    completed = _job_is_complete(job) if process.returncode == 0 else False
    status = "completed" if completed else "failed"
    notes = job.notes
    if process.returncode != 0:
        notes = f"{job.notes}; subprocess_returncode={process.returncode}"
    elif not completed:
        notes = f"{job.notes}; missing_expected_outputs"
    return ExecutionRecord(
        phase=job.phase,
        job_kind=job.job_kind,
        run_name=job.run_name,
        status=status,
        returncode=int(process.returncode),
        start_time_utc=start.replace(microsecond=0).isoformat(),
        end_time_utc=end.replace(microsecond=0).isoformat(),
        duration_s=max((end - start).total_seconds(), 0.0),
        target_front_speed_mm_s=job.target_front_speed_mm_s,
        num_knots=job.num_knots,
        knot_time_schedule=job.knot_time_schedule,
        seed=job.seed,
        simulation_profile=job.simulation_profile,
        reused_existing_result=0,
        log_path=str(log_path),
        command=command_text,
        notes=notes,
    )


def _run_parallel_jobs(
    *,
    jobs: list[QueueJob],
    max_workers: int,
    logs_dir: Path,
    dry_run: bool,
) -> list[ExecutionRecord]:
    if not jobs:
        return []
    if dry_run:
        return [_run_job(job=job, logs_dir=logs_dir, dry_run=True) for job in jobs]
    records: list[ExecutionRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_job, job=job, logs_dir=logs_dir, dry_run=False): job for job in jobs}
        for future in as_completed(future_map):
            records.append(future.result())
    records.sort(key=lambda record: (record.phase, record.run_name))
    return records


def _run_serial_jobs(
    *,
    jobs: list[QueueJob],
    logs_dir: Path,
    dry_run: bool,
) -> list[ExecutionRecord]:
    return [_run_job(job=job, logs_dir=logs_dir, dry_run=dry_run) for job in jobs]


def _read_single_row(path: Path) -> dict[str, str]:
    rows = _read_csv_rows(path)
    return rows[0] if rows else {}


def _best_objective_from_history(path: Path) -> float:
    feasible_values: list[float] = []
    for row in _read_csv_rows(path):
        is_valid = str(row.get("is_valid", "")).strip() in {"1", "true", "True", "TRUE"}
        if not is_valid:
            continue
        try:
            feasible_values.append(float(row["objective_value"]))
        except (TypeError, ValueError, KeyError):
            continue
    return min(feasible_values) if feasible_values else math.nan


def _read_theta_profile(path: Path) -> tuple[float, ...]:
    theta_C: list[float] = []
    for row in _read_csv_rows(path):
        try:
            theta_C.append(float(row["temperature_C"]))
        except (TypeError, ValueError, KeyError):
            continue
    return tuple(theta_C)


def _load_coarse_selection(run_dir: Path) -> CoarseSelection | None:
    effective_path = run_dir / "effective_config.toml"
    tracking_path = run_dir / "best_tracking_summary.csv"
    plate_path = run_dir / "plate_tracking_summary.csv"
    theta_path = run_dir / "best_theta_profile.csv"
    history_path = run_dir / "bo_history.csv"
    if not all(path.exists() for path in (effective_path, tracking_path, plate_path, theta_path, history_path)):
        return None
    config = _read_toml(effective_path)
    tracking_row = _read_single_row(tracking_path)
    plate_row = _read_single_row(plate_path)
    target_mm_s = float(tracking_row.get("target_front_speed_mm_s", "nan"))
    achieved_mm_s = float(tracking_row.get("actual_interval_speed_mm_s", "nan"))
    if not (math.isfinite(target_mm_s) and math.isfinite(achieved_mm_s)):
        return None
    objective_value = _best_objective_from_history(history_path)
    if not math.isfinite(objective_value):
        return None
    theta_C = _read_theta_profile(theta_path)
    run_cfg = dict(config.get("run", {}))
    bo_cfg = dict(config.get("bayesian_optimization", {}))
    trajectory_cfg = dict(config.get("trajectory", {}))
    relative_error_pct = 100.0 * abs(achieved_mm_s - target_mm_s) / max(abs(target_mm_s), 1.0e-12)
    seed_raw = bo_cfg.get("random_seed", None)
    seed = None if seed_raw in (None, "") else int(seed_raw)
    return CoarseSelection(
        target_front_speed_mm_s=target_mm_s,
        run_name=str(run_cfg.get("run_name", run_dir.name)),
        seed=seed,
        schedule=str(trajectory_cfg.get("knot_time_schedule", "uniform")),
        simulation_profile=str(run_cfg.get("simulation_profile", "")),
        objective_value=objective_value,
        achieved_direct_speed_mm_s=achieved_mm_s,
        direct_speed_relative_error_pct=relative_error_pct,
        rmse_plate_error_C=float(plate_row.get("rmse_plate_error_C", "nan")),
        theta_C=theta_C,
        run_dir=run_dir,
    )


def _select_best_n4_candidates(bo_results_root: Path) -> dict[float, CoarseSelection]:
    selections: dict[float, CoarseSelection] = {}
    for target_mm_s in TARGETS_PHASE4_MM_S:
        candidates: list[CoarseSelection] = []
        target_tag = _format_speed_tag(target_mm_s)
        for seed in SEEDS_PHASE4:
            run_dir = _coarse_run_dir(
                bo_results_root,
                num_knots=4,
                run_name=f"bo_v{target_tag}_n4_uniform_seed{seed}",
            )
            selection = _load_coarse_selection(run_dir)
            if selection is not None:
                candidates.append(selection)
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                item.objective_value,
                item.direct_speed_relative_error_pct,
                item.rmse_plate_error_C,
                item.run_name,
            )
        )
        selections[target_mm_s] = candidates[0]
    return selections


def _build_phase4_jobs(
    *,
    selections: dict[float, CoarseSelection],
    fine_results_root: Path,
) -> tuple[list[QueueJob], list[dict[str, object]]]:
    jobs: list[QueueJob] = []
    rows: list[dict[str, object]] = []
    for target_mm_s in TARGETS_PHASE4_MM_S:
        selection = selections.get(target_mm_s)
        target_tag = _format_speed_tag(target_mm_s)
        fine_run_name = f"fine_confirm_v{target_tag}_n4_uniform_best"
        if selection is None:
            rows.append(
                {
                    "target_front_speed_mm_s": f"{target_mm_s:.3f}",
                    "selected_coarse_run_name": "",
                    "selected_seed": "",
                    "objective_value": "",
                    "achieved_direct_speed_mm_s": "",
                    "direct_speed_relative_error_pct": "",
                    "rmse_plate_error_C": "",
                    "theta_C": "",
                    "recommended_fine_run_name": fine_run_name,
                    "fine_status": "skipped_no_feasible_candidate",
                    "notes": "no_feasible_n4_coarse_candidate",
                }
            )
            continue
        theta_text = ",".join(f"{value:.15g}" for value in selection.theta_C)
        argv = (
            sys.executable,
            "-m",
            "code_simulation.verification.run_velocity_control_evaluation",
            "--simulation-profile",
            "full_process_article",
            "--target-front-speed-mm-s",
            f"{target_mm_s:.3f}",
            f"--theta-c={theta_text}",
            "--num-knots",
            "4",
            "--knot-time-schedule",
            "uniform",
            "--run-name",
            fine_run_name,
            "--overwrite",
        )
        jobs.append(
            QueueJob(
                phase="phase4_n4_fine_confirmation",
                job_kind="fine_confirmation",
                run_name=fine_run_name,
                cwd=project_root(),
                argv=argv,
                simulation_profile="full_process_article",
                target_front_speed_mm_s=target_mm_s,
                num_knots=4,
                knot_time_schedule="uniform",
                seed=selection.seed,
                output_dir=_fine_run_dir(fine_results_root, num_knots=4, run_name=fine_run_name),
                source_path=None,
                notes=f"selected_from_{selection.run_name}",
            )
        )
        rows.append(
            {
                "target_front_speed_mm_s": f"{target_mm_s:.3f}",
                "selected_coarse_run_name": selection.run_name,
                "selected_seed": _format_optional(selection.seed),
                "objective_value": f"{selection.objective_value:.16g}",
                "achieved_direct_speed_mm_s": f"{selection.achieved_direct_speed_mm_s:.16g}",
                "direct_speed_relative_error_pct": f"{selection.direct_speed_relative_error_pct:.16g}",
                "rmse_plate_error_C": f"{selection.rmse_plate_error_C:.16g}",
                "theta_C": ",".join(f"{value:.15g}" for value in selection.theta_C),
                "recommended_fine_run_name": fine_run_name,
                "fine_status": "planned",
                "notes": "",
            }
        )
    return jobs, rows


def _best_three_knot_reference(schedule_summary_path: Path, *, target_mm_s: float) -> dict[str, str] | None:
    rows = _read_csv_rows(schedule_summary_path)
    candidates: list[dict[str, str]] = []
    for row in rows:
        try:
            candidate_target = float(row.get("target_front_speed_mm_s", "nan"))
        except (TypeError, ValueError):
            continue
        if abs(candidate_target - target_mm_s) > TARGET_TOL_MM_S:
            continue
        if str(row.get("status", "")).strip() != "completed":
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            -int(str(row.get("overall_success", "0")).strip() or "0"),
            abs(float(row.get("direct_speed_relative_error_pct", "nan"))),
            float(row.get("objective_value", "nan")),
            float(row.get("rmse_plate_error_C", "nan")),
            str(row.get("selected_run_name", "")),
        )
    )
    return candidates[0]


def _load_fine_metrics(run_dir: Path) -> dict[str, float | str] | None:
    if not _fine_run_complete(run_dir):
        return None
    tracking_row = _read_single_row(run_dir / "velocity_tracking_summary.csv")
    plate_row = _read_single_row(run_dir / "plate_tracking_summary.csv")
    if not tracking_row or not plate_row:
        return None
    try:
        target_mm_s = float(tracking_row["target_front_speed_mm_s"])
        achieved_mm_s = float(tracking_row["actual_interval_speed_mm_s"])
    except (TypeError, ValueError, KeyError):
        return None
    rel_err_pct = 100.0 * abs(achieved_mm_s - target_mm_s) / max(abs(target_mm_s), 1.0e-12)
    return {
        "target_front_speed_mm_s": target_mm_s,
        "achieved_direct_speed_mm_s": achieved_mm_s,
        "direct_speed_relative_error_pct": rel_err_pct,
        "tracking_rmse_mm": float(tracking_row.get("tracking_rmse_mm", "nan")),
        "rmse_plate_error_C": float(plate_row.get("rmse_plate_error_C", "nan")),
        "mean_abs_plate_error_C": float(plate_row.get("mean_abs_plate_error_C", "nan")),
        "fraction_within_tolerance": float(plate_row.get("fraction_within_tolerance", "nan")),
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_overnight_summary(
    *,
    path: Path,
    dry_run: bool,
    phase1_jobs: list[QueueJob],
    refreshed_decision_summary_path: Path | None,
    selected_rows: list[dict[str, object]],
    schedule_summary_path: Path | None,
    fine_results_root: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Overnight Velocity Control Summary")
    lines.append("")
    lines.append(f"Generated at `{_timestamp_utc()}`.")
    lines.append("")
    if dry_run:
        lines.append("This was a dry-run. No BO or fine job was executed.")
        lines.append("")
    lines.append("## 1. 3-Knot Uniform Seed41 Gap")
    total_phase1 = len(phase1_jobs)
    completed_phase1 = sum(1 for job in phase1_jobs if _job_is_complete(job))
    if total_phase1 == 0:
        lines.append("- No phase-1 jobs were discovered from the 3-knot study command file.")
    elif completed_phase1 >= total_phase1:
        lines.append(f"- The explicit `3-knot uniform, seed 41` gap is closed: `{completed_phase1}/{total_phase1}` expected jobs are complete.")
    else:
        lines.append(
            f"- The explicit `3-knot uniform, seed 41` gap is still open: "
            f"`{completed_phase1}/{total_phase1}` expected jobs are complete."
        )
    lines.append("")
    lines.append("## 2. Refreshed 3-Knot Decision Summary")
    if refreshed_decision_summary_path is None or not refreshed_decision_summary_path.exists():
        lines.append("- Decision summary refresh is unavailable.")
    else:
        lines.append("| target (mm/s) | coarse completed | coarse successes | fine confirmations | fine successes | recovered by 3 knots | BO-search-limited | 3-knot-limited | plant-limited |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in _read_csv_rows(refreshed_decision_summary_path):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("target_front_speed_mm_s", "")),
                        str(row.get("num_completed_coarse_runs", "")),
                        str(row.get("num_coarse_overall_successes", "")),
                        str(row.get("num_fine_confirmations", "")),
                        str(row.get("num_fine_successes", "")),
                        str(row.get("recovered_by_3knots", "")),
                        str(row.get("likely_bo_search_limited", "")),
                        str(row.get("likely_3knot_parameterization_limited", "")),
                        str(row.get("likely_plant_inner_response_limited", "")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## 3. Did 4 Knots Improve The Low Target (0.006)?")
    lines.extend(
        _compare_n4_against_n3(
            target_mm_s=0.006,
            selected_rows=selected_rows,
            schedule_summary_path=schedule_summary_path,
            fine_results_root=fine_results_root,
        )
    )
    lines.append("")
    lines.append("## 4. Did 4 Knots Improve The High Target (0.010)?")
    lines.extend(
        _compare_n4_against_n3(
            target_mm_s=0.010,
            selected_rows=selected_rows,
            schedule_summary_path=schedule_summary_path,
            fine_results_root=fine_results_root,
        )
    )
    lines.append("")
    lines.append("## 5. 4-Knot Fine Confirmations")
    for target_mm_s in TARGETS_PHASE4_MM_S:
        target_tag = _format_speed_tag(target_mm_s)
        run_name = f"fine_confirm_v{target_tag}_n4_uniform_best"
        fine_metrics = _load_fine_metrics(_fine_run_dir(fine_results_root, num_knots=4, run_name=run_name))
        if fine_metrics is None:
            selected_row = next(
                (row for row in selected_rows if abs(float(row["target_front_speed_mm_s"]) - target_mm_s) <= TARGET_TOL_MM_S),
                None,
            )
            fine_status = selected_row["fine_status"] if selected_row is not None else "not_planned"
            lines.append(f"- `{run_name}`: `{fine_status}`.")
        else:
            lines.append(
                f"- `{run_name}` completed: direct speed = `{fine_metrics['achieved_direct_speed_mm_s']:.6f} mm/s`, "
                f"relative error = `{fine_metrics['direct_speed_relative_error_pct']:.3f}%`, "
                f"tracking RMSE = `{fine_metrics['tracking_rmse_mm']:.3f} mm`, "
                f"plate RMSE = `{fine_metrics['rmse_plate_error_C']:.3f} C`."
            )
    lines.append("")
    lines.append("## Morning Decision Rule")
    lines.append("- If 4 knots materially improve either unresolved extreme, expand the 4-knot study tomorrow.")
    lines.append("- If 4 knots do not improve the unresolved extremes, only then discuss 5 knots or variable knot times.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _compare_n4_against_n3(
    *,
    target_mm_s: float,
    selected_rows: list[dict[str, object]],
    schedule_summary_path: Path | None,
    fine_results_root: Path,
) -> list[str]:
    lines: list[str] = []
    selected_row = next(
        (row for row in selected_rows if abs(float(row["target_front_speed_mm_s"]) - target_mm_s) <= TARGET_TOL_MM_S),
        None,
    )
    if schedule_summary_path is None or not schedule_summary_path.exists():
        lines.append("- No 3-knot schedule summary is available for comparison.")
        return lines
    three_ref = _best_three_knot_reference(schedule_summary_path, target_mm_s=target_mm_s)
    if three_ref is None:
        lines.append("- No completed 3-knot reference run was found.")
        return lines
    lines.append(
        f"- Best 3-knot reference: `{three_ref.get('selected_run_name', '')}` "
        f"({three_ref.get('schedule', '')}, seed {three_ref.get('seed', '')}), "
        f"direct speed = `{float(three_ref.get('achieved_direct_speed_mm_s', 'nan')):.6f} mm/s`, "
        f"relative error = `{float(three_ref.get('direct_speed_relative_error_pct', 'nan')):.3f}%`, "
        f"plate RMSE = `{float(three_ref.get('rmse_plate_error_C', 'nan')):.3f} C`."
    )
    if selected_row is None or not str(selected_row.get("selected_coarse_run_name", "")).strip():
        lines.append("- No feasible 4-knot coarse candidate was selected.")
        return lines
    lines.append(
        f"- Selected 4-knot coarse candidate: `{selected_row['selected_coarse_run_name']}` "
        f"(seed {selected_row['selected_seed']}), direct speed = "
        f"`{float(selected_row['achieved_direct_speed_mm_s']):.6f} mm/s`, relative error = "
        f"`{float(selected_row['direct_speed_relative_error_pct']):.3f}%`, plate RMSE = "
        f"`{float(selected_row['rmse_plate_error_C']):.3f} C`."
    )
    target_tag = _format_speed_tag(target_mm_s)
    fine_run_name = f"fine_confirm_v{target_tag}_n4_uniform_best"
    fine_metrics = _load_fine_metrics(_fine_run_dir(fine_results_root, num_knots=4, run_name=fine_run_name))
    if fine_metrics is None:
        lines.append(f"- 4-knot fine confirmation `{fine_run_name}` is `{selected_row['fine_status']}`.")
        improved = float(selected_row["direct_speed_relative_error_pct"]) < float(
            three_ref.get("direct_speed_relative_error_pct", "inf")
        )
        lines.append(
            "- Provisional coarse comparison: "
            + ("4 knots improved the direct-speed error over the best 3-knot reference." if improved else "4 knots did not yet improve the direct-speed error over the best 3-knot reference.")
        )
        return lines
    improved = float(fine_metrics["direct_speed_relative_error_pct"]) < float(three_ref.get("direct_speed_relative_error_pct", "inf"))
    lines.append(
        f"- 4-knot fine confirmation `{fine_run_name}` completed with direct speed = "
        f"`{fine_metrics['achieved_direct_speed_mm_s']:.6f} mm/s`, relative error = "
        f"`{fine_metrics['direct_speed_relative_error_pct']:.3f}%`, plate RMSE = "
        f"`{fine_metrics['rmse_plate_error_C']:.3f} C`."
    )
    lines.append(
        "- Fine comparison verdict: "
        + ("4 knots improved on the best 3-knot reference." if improved else "4 knots did not improve on the best 3-knot reference.")
    )
    return lines


def _write_execution_log(path: Path, records: list[ExecutionRecord]) -> None:
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "phase": record.phase,
                "job_kind": record.job_kind,
                "run_name": record.run_name,
                "status": record.status,
                "returncode": "" if record.returncode is None else record.returncode,
                "start_time_utc": record.start_time_utc,
                "end_time_utc": record.end_time_utc,
                "duration_s": f"{record.duration_s:.6f}",
                "target_front_speed_mm_s": _format_optional(record.target_front_speed_mm_s),
                "num_knots": _format_optional(record.num_knots),
                "knot_time_schedule": record.knot_time_schedule or "",
                "seed": _format_optional(record.seed),
                "simulation_profile": record.simulation_profile or "",
                "reused_existing_result": record.reused_existing_result,
                "log_path": record.log_path,
                "command": record.command,
                "notes": record.notes,
            }
        )
    _write_csv(
        path,
        [
            "phase",
            "job_kind",
            "run_name",
            "status",
            "returncode",
            "start_time_utc",
            "end_time_utc",
            "duration_s",
            "target_front_speed_mm_s",
            "num_knots",
            "knot_time_schedule",
            "seed",
            "simulation_profile",
            "reused_existing_result",
            "log_path",
            "command",
            "notes",
        ],
        rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the overnight velocity-control orchestration plan.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"bundle output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--three-knot-study-root",
        type=Path,
        default=DEFAULT_THREE_KNOT_STUDY_ROOT,
        help="existing 3-knot study root",
    )
    parser.add_argument(
        "--bo-results-root",
        type=Path,
        default=DEFAULT_BO_RESULTS_ROOT,
        help="root directory for BO run outputs",
    )
    parser.add_argument(
        "--fine-results-root",
        type=Path,
        default=DEFAULT_FINE_RESULTS_ROOT,
        help="root directory for fine confirmation outputs",
    )
    parser.add_argument(
        "--coarse-workers",
        type=int,
        default=DEFAULT_COARSE_WORKERS,
        help="number of coarse jobs to run in parallel",
    )
    parser.add_argument(
        "--fine-workers",
        type=int,
        default=DEFAULT_FINE_WORKERS,
        help="number of fine jobs to run in parallel (kept at 1 by policy)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the overnight queue without launching jobs")
    parser.add_argument("--overwrite", action="store_true", help="clear the overnight bundle before writing new files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.coarse_workers) < 1:
        raise ValueError("--coarse-workers must be at least 1")
    if int(args.fine_workers) != 1:
        raise ValueError("--fine-workers must be 1 for this overnight policy")

    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"

    uniform_commands_path = Path(args.three_knot_study_root) / "target_vs_achived" / "run_commands.sh"
    schedule_summary_path = Path(args.three_knot_study_root) / "schedule_comparison" / "schedule_summary.csv"
    decision_summary_src = Path(args.three_knot_study_root) / "decision_summary.csv"

    phase1_jobs = _build_phase1_jobs(
        uniform_commands_path=uniform_commands_path,
        bo_results_root=Path(args.bo_results_root),
    )
    phase2_job = _build_phase2_job()
    phase3_jobs = _build_phase3_jobs(bo_results_root=Path(args.bo_results_root))

    queue_rows = _queue_manifest_rows([*phase1_jobs, phase2_job, *phase3_jobs])
    _write_csv(
        output_root / "queue_manifest.csv",
        [
            "phase",
            "job_kind",
            "run_name",
            "target_front_speed_mm_s",
            "num_knots",
            "knot_time_schedule",
            "seed",
            "simulation_profile",
            "existing_result_detected",
            "output_dir",
            "source_path",
            "command",
            "notes",
        ],
        queue_rows,
    )

    execution_records: list[ExecutionRecord] = []
    execution_records.extend(
        _run_parallel_jobs(
            jobs=phase1_jobs,
            max_workers=int(args.coarse_workers),
            logs_dir=logs_dir,
            dry_run=bool(args.dry_run),
        )
    )
    execution_records.extend(
        _run_serial_jobs(
            jobs=[phase2_job],
            logs_dir=logs_dir,
            dry_run=bool(args.dry_run),
        )
    )
    refreshed_decision_summary_dst = output_root / "refreshed_3knot_decision_summary.csv"
    refreshed_decision_summary_path: Path | None = decision_summary_src if decision_summary_src.exists() else None
    _copy_if_exists(decision_summary_src, refreshed_decision_summary_dst)

    execution_records.extend(
        _run_parallel_jobs(
            jobs=phase3_jobs,
            max_workers=int(args.coarse_workers),
            logs_dir=logs_dir,
            dry_run=bool(args.dry_run),
        )
    )

    selections = _select_best_n4_candidates(Path(args.bo_results_root))
    phase4_jobs, selected_rows = _build_phase4_jobs(
        selections=selections,
        fine_results_root=Path(args.fine_results_root),
    )
    phase4_queue_rows = _queue_manifest_rows(phase4_jobs)
    if phase4_queue_rows:
        _write_csv(
            output_root / "queue_manifest.csv",
            [
                "phase",
                "job_kind",
                "run_name",
                "target_front_speed_mm_s",
                "num_knots",
                "knot_time_schedule",
                "seed",
                "simulation_profile",
                "existing_result_detected",
                "output_dir",
                "source_path",
                "command",
                "notes",
            ],
            queue_rows + phase4_queue_rows,
        )
    if args.dry_run:
        for row in selected_rows:
            if row["fine_status"] == "planned":
                row["fine_status"] = "planned_dry_run"
    execution_records.extend(
        _run_serial_jobs(
            jobs=phase4_jobs,
            logs_dir=logs_dir,
            dry_run=bool(args.dry_run),
        )
    )
    if not args.dry_run:
        fine_status_by_run_name = {record.run_name: record.status for record in execution_records if record.job_kind == "fine_confirmation"}
        for row in selected_rows:
            fine_run_name = str(row["recommended_fine_run_name"])
            if fine_run_name in fine_status_by_run_name:
                row["fine_status"] = fine_status_by_run_name[fine_run_name]
    _write_csv(
        output_root / "selected_n4_coarse_for_fine.csv",
        [
            "target_front_speed_mm_s",
            "selected_coarse_run_name",
            "selected_seed",
            "objective_value",
            "achieved_direct_speed_mm_s",
            "direct_speed_relative_error_pct",
            "rmse_plate_error_C",
            "theta_C",
            "recommended_fine_run_name",
            "fine_status",
            "notes",
        ],
        selected_rows,
    )

    _write_execution_log(output_root / "execution_log.csv", execution_records)
    _write_overnight_summary(
        path=output_root / "overnight_summary.md",
        dry_run=bool(args.dry_run),
        phase1_jobs=phase1_jobs,
        refreshed_decision_summary_path=refreshed_decision_summary_dst if refreshed_decision_summary_dst.exists() else None,
        selected_rows=selected_rows,
        schedule_summary_path=schedule_summary_path if schedule_summary_path.exists() else None,
        fine_results_root=Path(args.fine_results_root),
    )
    print(f"Overnight bundle: {output_root}")
    print(f"Queue manifest: {output_root / 'queue_manifest.csv'}")
    print(f"Execution log: {output_root / 'execution_log.csv'}")
    print(f"Selected n4 coarse rows: {output_root / 'selected_n4_coarse_for_fine.csv'}")
    print(f"Summary: {output_root / 'overnight_summary.md'}")


if __name__ == "__main__":
    main()
