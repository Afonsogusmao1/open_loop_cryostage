#!/usr/bin/env python3
"""Analyze article numerical sensitivity simulations."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from code_simulation.core.paths import project_root


ROOT = project_root() / "code_simulation" / "results" / "article_numerical_sensitivity"
OUT = ROOT / "numerical_sensitivity_summary.csv"

GROUP_REFERENCES = {
    "mesh_sensitivity": "mesh_reference",
    "time_sensitivity": "dt_reference",
    "front_sampling_sensitivity": "sampling_fine",
}

PASSAGE_HEIGHTS_MM = (3.0, 6.2, 11.0)


def read_metadata(case_dir: Path) -> dict[str, str]:
    files = list(case_dir.glob("*_metadata.csv"))
    if not files:
        raise FileNotFoundError(f"No metadata file found in {case_dir}")
    out = {}
    with files[0].open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["parameter"]] = row["value"]
    return out


def read_csv_numeric(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for name in cols:
                value = row[name]
                try:
                    cols[name].append(float(value))
                except (TypeError, ValueError):
                    cols[name].append(np.nan)
    return {key: np.asarray(vals, dtype=float) for key, vals in cols.items()}


def find_one(case_dir: Path, suffix: str) -> Path | None:
    files = list(case_dir.glob(f"*{suffix}"))
    return files[0] if files else None


def common_valid_time(case: dict[str, np.ndarray], ref: dict[str, np.ndarray], time_col: str) -> np.ndarray:
    t = case[time_col]
    tr = ref[time_col]
    mask = np.isfinite(t)
    mask &= t >= max(np.nanmin(tr), np.nanmin(t))
    mask &= t <= min(np.nanmax(tr), np.nanmax(t))
    return t[mask]


def interp_ref(ref: dict[str, np.ndarray], col: str, t_query: np.ndarray, time_col: str) -> np.ndarray:
    tr = ref[time_col]
    yr = ref[col]
    mask = np.isfinite(tr) & np.isfinite(yr)
    if np.count_nonzero(mask) < 2:
        return np.full_like(t_query, np.nan, dtype=float)
    return np.interp(t_query, tr[mask], yr[mask])


def history_error(
    case: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    cols: list[str],
    *,
    time_col: str = "time_since_fill_s",
) -> tuple[float, float]:
    t = common_valid_time(case, ref, time_col)
    if t.size == 0:
        return math.nan, math.nan

    errs = []
    case_time = case[time_col]
    valid_t = np.isfinite(case_time)
    for col in cols:
        if col not in case or col not in ref:
            continue
        yc = case[col]
        mask = valid_t & np.isfinite(yc) & (case_time >= t[0]) & (case_time <= t[-1])
        if np.count_nonzero(mask) < 2:
            continue
        yref = interp_ref(ref, col, case_time[mask], time_col)
        errs.append(yc[mask] - yref)

    if not errs:
        return math.nan, math.nan

    e = np.concatenate(errs)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return math.nan, math.nan
    return float(np.max(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def freeze_completion_time(front: dict[str, np.ndarray]) -> float:
    if "freeze_complete_flag" not in front:
        return math.nan
    flag = front["freeze_complete_flag"]
    t = front["time_since_fill_s"]
    idx = np.where((flag >= 0.5) & np.isfinite(t))[0]
    if len(idx) == 0:
        return math.nan
    return float(t[idx[0]])


def passage_time(front: dict[str, np.ndarray], height_mm: float, col: str = "z_front_mm") -> float:
    if col not in front:
        return math.nan
    t = front["time_since_fill_s"]
    z = front[col]
    mask = np.isfinite(t) & np.isfinite(z)
    t = t[mask]
    z = z[mask]
    if t.size < 2:
        return math.nan
    idx = np.where(z >= height_mm)[0]
    if len(idx) == 0:
        return math.nan
    j = int(idx[0])
    if j == 0:
        return float(t[0])
    z0, z1 = z[j - 1], z[j]
    t0, t1 = t[j - 1], t[j]
    if abs(z1 - z0) < 1e-12:
        return float(t1)
    return float(t0 + (height_mm - z0) * (t1 - t0) / (z1 - z0))


def read_front_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open() as f:
        first = f.readline().strip()
        if not first.startswith("# radii_mm:"):
            raise ValueError(f"Unexpected front-curve header in {path}")
        radii = np.asarray([float(x) for x in first.split(":", 1)[1].split(",")], dtype=float)
        reader = csv.DictReader(f)
        times = []
        values = []
        z_cols = [name for name in reader.fieldnames or [] if name.startswith("z_front_r")]
        for row in reader:
            try:
                t = float(row["time_since_fill_s"])
            except (KeyError, ValueError):
                continue
            vals = []
            for col in z_cols:
                try:
                    vals.append(float(row[col]))
                except ValueError:
                    vals.append(np.nan)
            times.append(t)
            values.append(vals)
    return np.asarray(times, dtype=float), radii, np.asarray(values, dtype=float)


def curve_error(case_path: Path | None, ref_path: Path | None) -> tuple[float, float]:
    if case_path is None or ref_path is None:
        return math.nan, math.nan

    tc, rc, zc = read_front_curve(case_path)
    tr, rr, zr = read_front_curve(ref_path)

    if tc.size == 0 or tr.size == 0:
        return math.nan, math.nan

    t0 = max(np.nanmin(tc), np.nanmin(tr))
    t1 = min(np.nanmax(tc), np.nanmax(tr))
    mask_t = np.isfinite(tc) & (tc >= t0) & (tc <= t1)

    errs = []
    for ti, zcase_row in zip(tc[mask_t], zc[mask_t]):
        # Interpolate reference in time at each reference radius.
        zref_time = []
        for j in range(len(rr)):
            col = zr[:, j]
            valid = np.isfinite(tr) & np.isfinite(col)
            if np.count_nonzero(valid) < 2:
                zref_time.append(np.nan)
            else:
                zref_time.append(np.interp(ti, tr[valid], col[valid]))
        zref_time = np.asarray(zref_time, dtype=float)

        valid_r = np.isfinite(rr) & np.isfinite(zref_time)
        if np.count_nonzero(valid_r) < 2:
            continue

        zref_at_case_r = np.interp(rc, rr[valid_r], zref_time[valid_r])
        diff = zcase_row - zref_at_case_r
        diff = diff[np.isfinite(diff)]
        if diff.size:
            errs.append(diff)

    if not errs:
        return math.nan, math.nan

    e = np.concatenate(errs)
    return float(np.max(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def load_case(case_dir: Path) -> dict:
    metadata = read_metadata(case_dir)
    probes_path = find_one(case_dir, "_probes.csv")
    front_path = find_one(case_dir, "_front.csv")
    curve_path = find_one(case_dir, "_front_curve.csv")

    if probes_path is None or front_path is None:
        raise FileNotFoundError(f"Missing probes/front CSV in {case_dir}")

    return {
        "dir": case_dir,
        "metadata": metadata,
        "probes": read_csv_numeric(probes_path),
        "front": read_csv_numeric(front_path),
        "curve_path": curve_path,
    }


def main() -> None:
    rows = []

    for group, ref_case_name in GROUP_REFERENCES.items():
        group_dir = ROOT / group
        case_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
        cases = [load_case(p) for p in case_dirs]

        ref = None
        for case in cases:
            if case["metadata"].get("case_name") == ref_case_name:
                ref = case
                break
        if ref is None:
            raise RuntimeError(f"No reference case {ref_case_name} found in {group_dir}")

        ref_freeze = freeze_completion_time(ref["front"])
        ref_passages = {
            h: passage_time(ref["front"], h, "z_front_mm")
            for h in PASSAGE_HEIGHTS_MM
        }

        for case in cases:
            md = case["metadata"]

            probe_cols = [
                col for col in case["probes"].keys()
                if col.startswith("T_z")
            ]
            eT_max, eT_rms = history_error(case["probes"], ref["probes"], probe_cols)

            ez_max, ez_rms = history_error(case["front"], ref["front"], ["z_front_mm"])
            ezw_max, ezw_rms = history_error(case["front"], ref["front"], ["z_front_wall_mm"])

            freeze_t = freeze_completion_time(case["front"])
            freeze_diff_s = freeze_t - ref_freeze if np.isfinite(freeze_t) and np.isfinite(ref_freeze) else math.nan
            freeze_diff_pct = 100.0 * freeze_diff_s / ref_freeze if np.isfinite(ref_freeze) and ref_freeze != 0 else math.nan

            ecurve_max, ecurve_rms = curve_error(case["curve_path"], ref["curve_path"])

            row = {
                "group": group,
                "case_name": md.get("case_name", ""),
                "reference_case": ref_case_name,
                "Nr": md.get("Nr", ""),
                "Nz": md.get("Nz", ""),
                "dt_s": md.get("dt_s", ""),
                "Nz_front": md.get("Nz_front", ""),
                "Nr_front_curve": md.get("Nr_front_curve", ""),
                "Nz_front_curve": md.get("Nz_front_curve", ""),
                "probe_T_err_max_C": eT_max,
                "probe_T_err_rms_C": eT_rms,
                "z_front_err_max_mm": ez_max,
                "z_front_err_rms_mm": ez_rms,
                "z_front_wall_err_max_mm": ezw_max,
                "z_front_wall_err_rms_mm": ezw_rms,
                "curve_front_err_max_mm": ecurve_max,
                "curve_front_err_rms_mm": ecurve_rms,
                "freeze_time_s": freeze_t,
                "freeze_time_diff_s": freeze_diff_s,
                "freeze_time_diff_pct": freeze_diff_pct,
            }

            for h in PASSAGE_HEIGHTS_MM:
                pt = passage_time(case["front"], h, "z_front_mm")
                ref_pt = ref_passages[h]
                row[f"passage_{h:g}mm_s"] = pt
                row[f"passage_{h:g}mm_diff_s"] = pt - ref_pt if np.isfinite(pt) and np.isfinite(ref_pt) else math.nan

            rows.append(row)

    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT}")
    print()
    for row in rows:
        print(
            f"{row['group']:28s} {row['case_name']:18s} "
            f"eT_rms={row['probe_T_err_rms_C']:.4g} C  "
            f"ez_rms={row['z_front_err_rms_mm']:.4g} mm  "
            f"freeze_diff={row['freeze_time_diff_s']:.4g} s"
        )


if __name__ == "__main__":
    main()
