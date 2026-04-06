from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReachabilityConstraints:
    constraints_dir: Path
    summary_json_path: Path
    aggregated_csv_path: Path
    window_csv_path: Path
    hold_summary_json_path: Path
    hold_support_csv_path: Path
    tolerance_band_C: float
    characterized_targets_C: tuple[float, ...]
    overall_window_s: np.ndarray
    overall_window_drop_C: np.ndarray
    setpoint_window_drop_C: dict[float, np.ndarray]
    first_entry_time_s_by_target_C: dict[float, float]
    settling_time_s_by_target_C: dict[float, float]
    reverse_excursion_C_by_target_C: dict[float, float]
    peak_toward_rate_10s_C_per_min_by_target_C: dict[float, float]
    supported_cooling_target_range_C: tuple[float, float]
    warming_supported: bool
    observed_hold_targets_C: tuple[float, ...]
    hold_supported_duration_s_by_target_C: dict[float, float]
    hold_support_basis_rule: str


@dataclass(frozen=True)
class SegmentAdmissibilityResult:
    segment_index: int
    t_start_s: float
    t_end_s: float
    duration_s: float
    T_start_C: float
    T_end_C: float
    requested_cooling_drop_C: float
    requested_warming_rise_C: float
    average_cooling_rate_C_per_s: float
    finite_window_cooling_check_applied: bool
    max_admissible_total_drop_C: float
    limiting_window_s: float
    hold_like_after_segment: bool
    requested_hold_duration_s: float
    characterized_target_check_applied: bool
    conservative_first_entry_time_s: float
    conservative_settling_time_s: float
    cumulative_time_to_end_s: float
    empirical_hold_check_applied: bool
    empirical_hold_basis_target_C: float
    conservative_supported_hold_duration_s: float
    supported_target_range_check_passed: bool
    local_cooling_check_passed: bool
    arrival_band_check_passed: bool
    settling_check_passed: bool
    hold_check_passed: bool
    unsupported_warming: bool
    is_admissible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": int(self.segment_index),
            "t_start_s": float(self.t_start_s),
            "t_end_s": float(self.t_end_s),
            "duration_s": float(self.duration_s),
            "T_start_C": float(self.T_start_C),
            "T_end_C": float(self.T_end_C),
            "requested_cooling_drop_C": float(self.requested_cooling_drop_C),
            "requested_warming_rise_C": float(self.requested_warming_rise_C),
            "average_cooling_rate_C_per_s": float(self.average_cooling_rate_C_per_s),
            "finite_window_cooling_check_applied": bool(self.finite_window_cooling_check_applied),
            "max_admissible_total_drop_C": float(self.max_admissible_total_drop_C),
            "limiting_window_s": float(self.limiting_window_s),
            "hold_like_after_segment": bool(self.hold_like_after_segment),
            "requested_hold_duration_s": float(self.requested_hold_duration_s),
            "characterized_target_check_applied": bool(self.characterized_target_check_applied),
            "conservative_first_entry_time_s": float(self.conservative_first_entry_time_s),
            "conservative_settling_time_s": float(self.conservative_settling_time_s),
            "cumulative_time_to_end_s": float(self.cumulative_time_to_end_s),
            "empirical_hold_check_applied": bool(self.empirical_hold_check_applied),
            "empirical_hold_basis_target_C": float(self.empirical_hold_basis_target_C),
            "conservative_supported_hold_duration_s": float(self.conservative_supported_hold_duration_s),
            "supported_target_range_check_passed": bool(self.supported_target_range_check_passed),
            "local_cooling_check_passed": bool(self.local_cooling_check_passed),
            "arrival_band_check_passed": bool(self.arrival_band_check_passed),
            "settling_check_passed": bool(self.settling_check_passed),
            "hold_check_passed": bool(self.hold_check_passed),
            "unsupported_warming": bool(self.unsupported_warming),
            "is_admissible": bool(self.is_admissible),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class TrajectoryAdmissibilityReport:
    constraints_dir: Path
    knot_times_s: tuple[float, ...]
    knot_temperatures_C: tuple[float, ...]
    require_monotone_nonincreasing: bool
    monotone_cooling_hard_enforced: bool
    warming_supported: bool
    is_admissible: bool
    segment_results: tuple[SegmentAdmissibilityResult, ...]
    reasons: tuple[str, ...]

    def failure_summary(self) -> str:
        if self.is_admissible:
            return "trajectory is admissible under the active transient-plus-hold constraints"
        if self.reasons:
            return "; ".join(self.reasons)
        failing_segments = [segment for segment in self.segment_results if not segment.is_admissible]
        if not failing_segments:
            return "trajectory is inadmissible under the active transient-plus-hold constraints"
        return "; ".join(
            f"segment {segment.segment_index}: {', '.join(segment.reasons)}"
            for segment in failing_segments
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints_dir": str(self.constraints_dir.resolve()),
            "knot_times_s": [float(value) for value in self.knot_times_s],
            "knot_temperatures_C": [float(value) for value in self.knot_temperatures_C],
            "require_monotone_nonincreasing": bool(self.require_monotone_nonincreasing),
            "monotone_cooling_hard_enforced": bool(self.monotone_cooling_hard_enforced),
            "warming_supported": bool(self.warming_supported),
            "is_admissible": bool(self.is_admissible),
            "reasons": list(self.reasons),
            "segment_results": [segment.to_dict() for segment in self.segment_results],
        }


class TrajectoryAdmissibilityError(ValueError):
    def __init__(self, report: TrajectoryAdmissibilityReport):
        self.report = report
        super().__init__(report.failure_summary())


def _as_float_tuple(values, *, name: str) -> tuple[float, ...]:
    try:
        out = tuple(float(value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of floats") from exc
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain only finite values")
    return out


def default_constraints_dir(repo_root: str | Path | None = None) -> Path:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent
    repo_root = Path(repo_root)
    base = repo_root if repo_root.name == "code_simulation" else repo_root / "code_simulation"
    return base / "results" / "characterization_constraints" / "stage1_reachability"


def default_hold_constraints_dir(repo_root: str | Path | None = None) -> Path:
    stage1_dir = default_constraints_dir(repo_root)
    return stage1_dir.parent / "stage2_hold_telemetry"


def _coerce_constraints_dir(path: str | Path | None) -> Path:
    if path is None:
        return default_constraints_dir()

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()

    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    search_roots: list[Path] = []
    if candidate.parts and candidate.parts[0] == "code_simulation":
        search_roots.append(project_root)
    search_roots.extend((Path.cwd(), module_dir, project_root))

    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    fallback_root = project_root if candidate.parts and candidate.parts[0] == "code_simulation" else Path.cwd()
    return (fallback_root / candidate).resolve()


def _resolve_constraints_artifact_dirs(path: str | Path | None) -> tuple[Path, Path]:
    resolved = _coerce_constraints_dir(path)
    if resolved.name == "stage1_reachability":
        return resolved, resolved.parent / "stage2_hold_telemetry"
    if resolved.name == "stage2_hold_telemetry":
        return resolved.parent / "stage1_reachability", resolved
    if resolved.name == "characterization_constraints":
        return resolved / "stage1_reachability", resolved / "stage2_hold_telemetry"
    return resolved, resolved.parent / "stage2_hold_telemetry"


@lru_cache(maxsize=8)
def _load_constraints_cached(stage1_dir_text: str, hold_dir_text: str) -> ReachabilityConstraints:
    constraints_dir = Path(stage1_dir_text)
    hold_dir = Path(hold_dir_text)
    summary_json_path = constraints_dir / "reachability_summary.json"
    aggregated_csv_path = constraints_dir / "aggregated_metrics_by_setpoint.csv"
    window_csv_path = constraints_dir / "window_reachability_envelope.csv"
    hold_summary_json_path = hold_dir / "hold_summary.json"
    hold_support_csv_path = hold_dir / "hold_duration_support_grid.csv"

    for required_path in (
        summary_json_path,
        aggregated_csv_path,
        window_csv_path,
        hold_summary_json_path,
        hold_support_csv_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing admissibility constraint artifact: {required_path}")

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    hold_summary = json.loads(hold_summary_json_path.read_text(encoding="utf-8"))
    tolerance_band_C = float(summary["tolerance_band_C"])
    warming_supported = False

    first_entry_time_s_by_target_C: dict[float, float] = {}
    settling_time_s_by_target_C: dict[float, float] = {}
    reverse_excursion_C_by_target_C: dict[float, float] = {}
    peak_toward_rate_10s_C_per_min_by_target_C: dict[float, float] = {}
    characterized_targets: list[float] = []

    with aggregated_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["group"] == "overall":
                continue
            target_c = float(row["nominal_setpoint_C"])
            characterized_targets.append(target_c)
            first_entry_time_s_by_target_C[target_c] = float(row["conservative_first_enter_band_s"])
            settling_time_s_by_target_C[target_c] = float(row["conservative_settling_time_s"])
            reverse_excursion_C_by_target_C[target_c] = float(row["conservative_reverse_excursion_C"])
            peak_toward_rate_10s_C_per_min_by_target_C[target_c] = float(row["conservative_peak_toward_rate_10s_C_per_min"])

    characterized_targets = sorted(set(characterized_targets))
    if not characterized_targets:
        raise ValueError(f"{aggregated_csv_path} does not contain any characterized targets")

    overall_windows_s: list[float] = []
    overall_window_drop_C: list[float] = []
    setpoint_window_rows: dict[float, list[tuple[float, float]]] = {target: [] for target in characterized_targets}

    with window_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = str(row["group"])
            window_s = float(row["window_s"])
            max_progress_C = float(row["max_progress_C"])
            if group == "overall_conservative":
                overall_windows_s.append(window_s)
                overall_window_drop_C.append(max_progress_C)
                continue
            if not group.endswith("_conservative") or not group.startswith("setpoint_"):
                continue
            nominal_text = row.get("nominal_setpoint_C", "")
            if nominal_text == "":
                continue
            target_c = float(nominal_text)
            if target_c in setpoint_window_rows:
                setpoint_window_rows[target_c].append((window_s, max_progress_C))

    if not overall_windows_s:
        raise ValueError(f"{window_csv_path} does not contain an overall conservative envelope")

    sort_idx = np.argsort(np.asarray(overall_windows_s, dtype=np.float64))
    overall_window_s = np.asarray(overall_windows_s, dtype=np.float64)[sort_idx]
    overall_window_drop = np.asarray(overall_window_drop_C, dtype=np.float64)[sort_idx]

    setpoint_window_drop_C: dict[float, np.ndarray] = {}
    for target_c, rows in setpoint_window_rows.items():
        rows = sorted(rows, key=lambda item: item[0])
        if not rows:
            continue
        setpoint_window_drop_C[target_c] = np.asarray([value for _, value in rows], dtype=np.float64)

    hold_supported_duration_s_by_target_C: dict[float, float] = {}
    support_duration_map = hold_summary.get("support_duration_s_by_observed_target_C", {})
    for target_text, duration_s in support_duration_map.items():
        hold_supported_duration_s_by_target_C[float(target_text)] = float(duration_s)
    observed_hold_targets = tuple(sorted(hold_supported_duration_s_by_target_C))
    if not observed_hold_targets:
        raise ValueError(f"{hold_summary_json_path} does not contain any observed hold targets")

    return ReachabilityConstraints(
        constraints_dir=constraints_dir,
        summary_json_path=summary_json_path,
        aggregated_csv_path=aggregated_csv_path,
        window_csv_path=window_csv_path,
        hold_summary_json_path=hold_summary_json_path,
        hold_support_csv_path=hold_support_csv_path,
        tolerance_band_C=tolerance_band_C,
        characterized_targets_C=tuple(characterized_targets),
        overall_window_s=overall_window_s,
        overall_window_drop_C=overall_window_drop,
        setpoint_window_drop_C=setpoint_window_drop_C,
        first_entry_time_s_by_target_C=first_entry_time_s_by_target_C,
        settling_time_s_by_target_C=settling_time_s_by_target_C,
        reverse_excursion_C_by_target_C=reverse_excursion_C_by_target_C,
        peak_toward_rate_10s_C_per_min_by_target_C=peak_toward_rate_10s_C_per_min_by_target_C,
        supported_cooling_target_range_C=(float(min(characterized_targets)), float(max(characterized_targets))),
        warming_supported=warming_supported,
        observed_hold_targets_C=observed_hold_targets,
        hold_supported_duration_s_by_target_C=hold_supported_duration_s_by_target_C,
        hold_support_basis_rule=(
            "For a requested hold target between observed freezing-run setpoints, use the nearest observed target that is equally cold or colder. "
            "This assigns hold support from a hardware condition that is at least as demanding as the requested target."
        ),
    )


def load_reachability_constraints(constraints_dir: str | Path | None = None) -> ReachabilityConstraints:
    stage1_dir, hold_dir = _resolve_constraints_artifact_dirs(constraints_dir)
    return _load_constraints_cached(str(stage1_dir), str(hold_dir))


def max_transient_window_s(constraints: ReachabilityConstraints) -> float:
    return float(constraints.overall_window_s[-1])


def interpolate_overall_conservative_cooling_drop_C(
    duration_s: float,
    constraints: ReachabilityConstraints,
) -> tuple[float, float]:
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be a finite non-negative value")
    if duration_s == 0.0:
        return 0.0, 0.0

    x = np.concatenate(([0.0], constraints.overall_window_s))
    y = np.concatenate(([0.0], constraints.overall_window_drop_C))
    max_window_s = max_transient_window_s(constraints)
    if duration_s <= max_window_s:
        return float(np.interp(duration_s, x, y)), duration_s
    return math.nan, max_window_s


def _conservative_bracketing_value(
    target_temperature_C: float,
    mapping_by_target_C: dict[float, float],
    constraints: ReachabilityConstraints,
) -> float:
    target_temperature_C = float(target_temperature_C)
    min_target_C, max_target_C = constraints.supported_cooling_target_range_C
    if target_temperature_C < min_target_C - 1e-12 or target_temperature_C > max_target_C + 1e-12:
        return math.nan

    targets = np.asarray(constraints.characterized_targets_C, dtype=np.float64)
    values = np.asarray([mapping_by_target_C[float(value)] for value in constraints.characterized_targets_C], dtype=np.float64)
    exact_idx = np.flatnonzero(np.isclose(targets, target_temperature_C, atol=1e-12))
    if exact_idx.size:
        return float(values[int(exact_idx[0])])

    insert_idx = int(np.searchsorted(targets, target_temperature_C, side="left"))
    lower_idx = max(insert_idx - 1, 0)
    upper_idx = min(insert_idx, targets.size - 1)
    return float(max(values[lower_idx], values[upper_idx]))


def conservative_first_entry_time_s(target_temperature_C: float, constraints: ReachabilityConstraints) -> float:
    return _conservative_bracketing_value(target_temperature_C, constraints.first_entry_time_s_by_target_C, constraints)


def conservative_settling_time_s(target_temperature_C: float, constraints: ReachabilityConstraints) -> float:
    return _conservative_bracketing_value(target_temperature_C, constraints.settling_time_s_by_target_C, constraints)


def conservative_hold_support_s(
    target_temperature_C: float,
    constraints: ReachabilityConstraints,
) -> tuple[float, float]:
    target_temperature_C = float(target_temperature_C)
    min_hold_target_C = float(min(constraints.observed_hold_targets_C))
    _, max_transient_target_C = constraints.supported_cooling_target_range_C
    if target_temperature_C < min_hold_target_C - 1e-12 or target_temperature_C > max_transient_target_C + 1e-12:
        return math.nan, math.nan

    hold_targets = np.asarray(constraints.observed_hold_targets_C, dtype=np.float64)
    basis_idx = int(np.searchsorted(hold_targets, target_temperature_C, side="right") - 1)
    if basis_idx < 0:
        return math.nan, math.nan
    basis_target_C = float(hold_targets[basis_idx])
    return float(constraints.hold_supported_duration_s_by_target_C[basis_target_C]), basis_target_C


def _hold_duration_after_segment_end_s(
    knot_times_s: tuple[float, ...],
    knot_temperatures_C: tuple[float, ...],
    *,
    segment_end_index: int,
    tolerance_band_C: float,
) -> float:
    target_C = float(knot_temperatures_C[segment_end_index])
    band_floor_C = target_C - float(tolerance_band_C)
    arrival_time_s = float(knot_times_s[segment_end_index])
    current_time_s = arrival_time_s
    current_temp_C = target_C

    for next_idx in range(segment_end_index + 1, len(knot_times_s)):
        next_time_s = float(knot_times_s[next_idx])
        next_temp_C = float(knot_temperatures_C[next_idx])
        if next_temp_C >= band_floor_C - 1e-12:
            current_time_s = next_time_s
            current_temp_C = next_temp_C
            continue

        if abs(next_temp_C - current_temp_C) <= 1e-12:
            leave_time_s = current_time_s
        else:
            frac = (band_floor_C - current_temp_C) / (next_temp_C - current_temp_C)
            frac = min(max(frac, 0.0), 1.0)
            leave_time_s = current_time_s + frac * (next_time_s - current_time_s)
        return max(float(leave_time_s - arrival_time_s), 0.0)

    return max(float(current_time_s - arrival_time_s), 0.0)


def check_segment_admissibility(
    *,
    segment_index: int,
    t_start_s: float,
    t_end_s: float,
    T_start_C: float,
    T_end_C: float,
    cumulative_time_to_end_s: float,
    hold_like_after_segment: bool,
    requested_hold_duration_s: float,
    constraints: ReachabilityConstraints,
    require_monotone_nonincreasing: bool,
) -> SegmentAdmissibilityResult:
    duration_s = float(t_end_s) - float(t_start_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("segment durations must be finite and strictly positive")

    tolerance_band_C = float(constraints.tolerance_band_C)
    requested_cooling_drop_C = max(float(T_start_C) - float(T_end_C), 0.0)
    requested_warming_rise_C = max(float(T_end_C) - float(T_start_C), 0.0)
    average_cooling_rate_C_per_s = requested_cooling_drop_C / duration_s

    reasons: list[str] = []
    unsupported_warming = requested_warming_rise_C > tolerance_band_C
    if unsupported_warming:
        reasons.append(
            f"unsupported warming request of {requested_warming_rise_C:.2f} C over {duration_s:.1f} s"
        )
    elif require_monotone_nonincreasing and float(T_end_C) > float(T_start_C) + 1e-12:
        reasons.append("segment violates monotone non-increasing trajectory requirement")

    min_supported_target_C, warmest_characterized_target_C = constraints.supported_cooling_target_range_C
    supported_target_range_check_passed = float(T_end_C) >= min_supported_target_C - 1e-12
    if not supported_target_range_check_passed:
        reasons.append(
            f"segment endpoint {float(T_end_C):.2f} C is colder than characterized support {min_supported_target_C:.2f} C"
        )

    max_window_s = max_transient_window_s(constraints)
    finite_window_cooling_check_applied = duration_s <= max_window_s + 1e-12
    max_admissible_total_drop_C = math.nan
    limiting_window_s = max_window_s
    local_cooling_check_passed = True
    if finite_window_cooling_check_applied:
        max_admissible_total_drop_C, limiting_window_s = interpolate_overall_conservative_cooling_drop_C(duration_s, constraints)
        local_cooling_check_passed = requested_cooling_drop_C <= max_admissible_total_drop_C + 1e-12
        if not local_cooling_check_passed:
            reasons.append(
                "segment cooling demand exceeds the conservative finite-window transient envelope: "
                f"requested {requested_cooling_drop_C:.2f} C over {duration_s:.1f} s, "
                f"allowed {max_admissible_total_drop_C:.2f} C"
            )

    characterized_target_check_applied = False
    conservative_first_entry = math.nan
    conservative_settling = math.nan
    arrival_band_check_passed = True
    settling_check_passed = True
    if float(T_end_C) <= warmest_characterized_target_C + 1e-12 and float(T_end_C) >= min_supported_target_C - 1e-12:
        characterized_target_check_applied = True
        conservative_first_entry = conservative_first_entry_time_s(float(T_end_C), constraints)
        conservative_settling = conservative_settling_time_s(float(T_end_C), constraints)

        if math.isfinite(conservative_first_entry) and cumulative_time_to_end_s < conservative_first_entry - 1e-12:
            arrival_band_check_passed = False
            reasons.append(
                "segment ends at a characterized cold target before the conservative first-entry time into the ±0.5 C band: "
                f"arrival {cumulative_time_to_end_s:.1f} s, required {conservative_first_entry:.1f} s"
            )

        if hold_like_after_segment and math.isfinite(conservative_settling) and cumulative_time_to_end_s < conservative_settling - 1e-12:
            settling_check_passed = False
            reasons.append(
                "requested hold begins before the conservative settling time at the target: "
                f"arrival {cumulative_time_to_end_s:.1f} s, required {conservative_settling:.1f} s"
            )

    empirical_hold_check_applied = requested_hold_duration_s > 1e-12
    empirical_hold_basis_target_C = math.nan
    conservative_supported_hold_duration_s = math.nan
    hold_check_passed = True
    if empirical_hold_check_applied:
        conservative_supported_hold_duration_s, empirical_hold_basis_target_C = conservative_hold_support_s(float(T_end_C), constraints)
        if not math.isfinite(conservative_supported_hold_duration_s):
            hold_check_passed = False
            reasons.append(
                f"no empirical long-duration hold support is available for target {float(T_end_C):.2f} C"
            )
        elif requested_hold_duration_s > conservative_supported_hold_duration_s + 1e-12:
            hold_check_passed = False
            reasons.append(
                "requested hold duration exceeds empirical plate-hold support: "
                f"requested {requested_hold_duration_s:.1f} s at {float(T_end_C):.2f} C, "
                f"supported {conservative_supported_hold_duration_s:.1f} s using basis target {empirical_hold_basis_target_C:.1f} C"
            )

    is_admissible = not reasons
    return SegmentAdmissibilityResult(
        segment_index=int(segment_index),
        t_start_s=float(t_start_s),
        t_end_s=float(t_end_s),
        duration_s=float(duration_s),
        T_start_C=float(T_start_C),
        T_end_C=float(T_end_C),
        requested_cooling_drop_C=float(requested_cooling_drop_C),
        requested_warming_rise_C=float(requested_warming_rise_C),
        average_cooling_rate_C_per_s=float(average_cooling_rate_C_per_s),
        finite_window_cooling_check_applied=bool(finite_window_cooling_check_applied),
        max_admissible_total_drop_C=float(max_admissible_total_drop_C) if math.isfinite(max_admissible_total_drop_C) else math.nan,
        limiting_window_s=float(limiting_window_s),
        hold_like_after_segment=bool(hold_like_after_segment),
        requested_hold_duration_s=float(requested_hold_duration_s),
        characterized_target_check_applied=bool(characterized_target_check_applied),
        conservative_first_entry_time_s=float(conservative_first_entry) if math.isfinite(conservative_first_entry) else math.nan,
        conservative_settling_time_s=float(conservative_settling) if math.isfinite(conservative_settling) else math.nan,
        cumulative_time_to_end_s=float(cumulative_time_to_end_s),
        empirical_hold_check_applied=bool(empirical_hold_check_applied),
        empirical_hold_basis_target_C=float(empirical_hold_basis_target_C) if math.isfinite(empirical_hold_basis_target_C) else math.nan,
        conservative_supported_hold_duration_s=float(conservative_supported_hold_duration_s) if math.isfinite(conservative_supported_hold_duration_s) else math.nan,
        supported_target_range_check_passed=bool(supported_target_range_check_passed),
        local_cooling_check_passed=bool(local_cooling_check_passed),
        arrival_band_check_passed=bool(arrival_band_check_passed),
        settling_check_passed=bool(settling_check_passed),
        hold_check_passed=bool(hold_check_passed),
        unsupported_warming=bool(unsupported_warming),
        is_admissible=bool(is_admissible),
        reasons=tuple(reasons),
    )


def check_piecewise_linear_trajectory_admissibility(
    knot_times_s,
    knot_temperatures_C,
    *,
    constraints: ReachabilityConstraints | None = None,
    constraints_dir: str | Path | None = None,
    require_monotone_nonincreasing: bool = True,
) -> TrajectoryAdmissibilityReport:
    knot_times = _as_float_tuple(knot_times_s, name="knot_times_s")
    knot_temperatures = _as_float_tuple(knot_temperatures_C, name="knot_temperatures_C")
    if len(knot_times) != len(knot_temperatures):
        raise ValueError("knot_times_s and knot_temperatures_C must have the same length")
    for idx in range(1, len(knot_times)):
        if knot_times[idx] <= knot_times[idx - 1]:
            raise ValueError("knot_times_s must be strictly increasing")

    if constraints is None:
        constraints = load_reachability_constraints(constraints_dir)

    segment_results: list[SegmentAdmissibilityResult] = []
    reasons: list[str] = []
    start_time_s = float(knot_times[0])

    for segment_index in range(1, len(knot_times)):
        requested_hold_duration_s = _hold_duration_after_segment_end_s(
            knot_times,
            knot_temperatures,
            segment_end_index=segment_index,
            tolerance_band_C=constraints.tolerance_band_C,
        )
        hold_like_after_segment = requested_hold_duration_s > 1e-12
        segment = check_segment_admissibility(
            segment_index=segment_index,
            t_start_s=knot_times[segment_index - 1],
            t_end_s=knot_times[segment_index],
            T_start_C=knot_temperatures[segment_index - 1],
            T_end_C=knot_temperatures[segment_index],
            cumulative_time_to_end_s=float(knot_times[segment_index] - start_time_s),
            hold_like_after_segment=hold_like_after_segment,
            requested_hold_duration_s=requested_hold_duration_s,
            constraints=constraints,
            require_monotone_nonincreasing=require_monotone_nonincreasing,
        )
        segment_results.append(segment)
        if not segment.is_admissible:
            reasons.append(f"segment {segment.segment_index}: {', '.join(segment.reasons)}")

    is_admissible = not reasons
    return TrajectoryAdmissibilityReport(
        constraints_dir=constraints.constraints_dir,
        knot_times_s=knot_times,
        knot_temperatures_C=knot_temperatures,
        require_monotone_nonincreasing=bool(require_monotone_nonincreasing),
        monotone_cooling_hard_enforced=bool(require_monotone_nonincreasing),
        warming_supported=bool(constraints.warming_supported),
        is_admissible=bool(is_admissible),
        segment_results=tuple(segment_results),
        reasons=tuple(reasons),
    )


__all__ = [
    "ReachabilityConstraints",
    "SegmentAdmissibilityResult",
    "TrajectoryAdmissibilityError",
    "TrajectoryAdmissibilityReport",
    "check_piecewise_linear_trajectory_admissibility",
    "check_segment_admissibility",
    "conservative_first_entry_time_s",
    "conservative_hold_support_s",
    "conservative_settling_time_s",
    "default_constraints_dir",
    "default_hold_constraints_dir",
    "interpolate_overall_conservative_cooling_drop_C",
    "load_reachability_constraints",
    "max_transient_window_s",
]
