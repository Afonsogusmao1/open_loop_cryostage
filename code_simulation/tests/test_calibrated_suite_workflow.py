from __future__ import annotations

import csv
from pathlib import Path

from code_simulation.verification.calibrated.common import (
    prepare_calibrated_results_layout,
)
from code_simulation.verification.calibrated.experiment_vs_simulation import (
    generate_calibrated_experiment_vs_simulation_figures,
)
from code_simulation.verification.calibrated.front_speed_plots import (
    generate_calibrated_front_speed_plots,
)
from code_simulation.verification.run_pre_stabilized_calibrated_simulations import (
    DEFAULT_T_FILL_C,
    ambient_temperature_for_plate_C,
)


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata(path: Path, *, target_C: float, ambient_C: float) -> None:
    rows = [
        {"parameter": "T_plate_C", "value": f"{target_C:.6f}"},
        {"parameter": "T_ambient_C", "value": f"{ambient_C:.6f}"},
        {"parameter": "T_fill_C", "value": f"{DEFAULT_T_FILL_C:.6f}"},
        {"parameter": "h_top_W_m2K", "value": "2.000000"},
    ]
    _write_rows(path, ["parameter", "value"], rows)


def _make_simulation_case(root: Path, *, target_C: float) -> Path:
    ambient_C = ambient_temperature_for_plate_C(target_C)
    prefix = (
        f"water_PLA_calib_plate_m{abs(int(target_C))}"
        "_probe_stabilized_Tamb_test_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm"
    )
    case_dir = root / (
        f"plate_m{abs(int(target_C))}"
        "_probe_stabilized_Tamb_test_h_2_Tfill_12p5_z_3p0_6p2_11p0mm_inset_1p0mm"
    )
    probe_rows = []
    front_rows = []
    for step in range(0, 81):
        time_s = 0.25 * step
        time_since_fill_s = "" if step < 8 else f"{0.25 * (step - 8):.6f}"
        probe_rows.append(
            {
                "time_s": f"{time_s:.6f}",
                "time_since_fill_s": time_since_fill_s,
                "T_z3p0mm_C": f"{8.0 - 0.18 * max(step - 8, 0):.6f}",
                "T_z6p2mm_C": f"{8.5 - 0.12 * max(step - 8, 0):.6f}",
                "T_z11p0mm_C": f"{9.0 - 0.08 * max(step - 8, 0):.6f}",
            }
        )
        front_rows.append(
            {
                "time_s": f"{time_s:.6f}",
                "time_since_fill_s": time_since_fill_s,
                "fill_flag": "0" if step < 8 else "1",
                "z_front_m": "" if step < 8 else f"{0.0002 * (step - 8):.9f}",
                "z_front_mm": "" if step < 8 else f"{0.2 * (step - 8):.6f}",
                "z_front_rel_mm": "" if step < 8 else f"{0.2 * (step - 8):.6f}",
                "v_front_mm_per_s": "" if step < 9 else "0.800000",
                "z_front_wall_m": "" if step < 8 else f"{0.00018 * (step - 8):.9f}",
                "z_front_wall_mm": "" if step < 8 else f"{0.18 * (step - 8):.6f}",
                "v_front_wall_mm_per_s": "" if step < 9 else "0.720000",
                "Tmax_fillable_C": "" if step < 8 else f"{5.0 - 0.1 * (step - 8):.6f}",
                "freeze_complete_flag": "0",
            }
        )

    _write_rows(
        case_dir / f"{prefix}_probes.csv",
        ["time_s", "time_since_fill_s", "T_z3p0mm_C", "T_z6p2mm_C", "T_z11p0mm_C"],
        probe_rows,
    )
    _write_rows(
        case_dir / f"{prefix}_front.csv",
        [
            "time_s",
            "time_since_fill_s",
            "fill_flag",
            "z_front_m",
            "z_front_mm",
            "z_front_rel_mm",
            "v_front_mm_per_s",
            "z_front_wall_m",
            "z_front_wall_mm",
            "v_front_wall_mm_per_s",
            "Tmax_fillable_C",
            "freeze_complete_flag",
        ],
        front_rows,
    )
    _write_metadata(case_dir / f"{prefix}_metadata.csv", target_C=target_C, ambient_C=ambient_C)
    return case_dir


def _make_experimental_group(root: Path, *, label: str) -> None:
    for run_idx in range(5):
        rows = []
        for step in range(0, 1201):
            t_s = 0.1 * step
            if t_s < 20.0:
                T3 = -4.0 + 0.02 * run_idx
                T7 = -1.0 + 0.01 * run_idx
                T12 = 3.5 + 0.02 * run_idx
            else:
                rel = t_s - 20.0
                T3 = 12.0 - 0.07 * rel + 0.08 * run_idx
                T7 = 10.0 - 0.05 * rel + 0.06 * run_idx
                T12 = 8.0 - 0.035 * rel + 0.05 * run_idx
            rows.append(
                {
                    "t_rec_s": f"{t_s:.6f}",
                    "T3": f"{T3:.6f}",
                    "T7": f"{T7:.6f}",
                    "T12": f"{T12:.6f}",
                    "Tamb": "9.700000",
                }
            )
        _write_rows(
            root / label / f"cryostage_log_{label}_{run_idx + 1}.csv",
            ["t_rec_s", "T3", "T7", "T12", "Tamb"],
            rows,
        )


def test_prepare_calibrated_results_layout_is_idempotent(tmp_path: Path) -> None:
    base_dir = tmp_path / "simulations_calibrated"
    for name in ("plate_m5_probe_stabilized_demo", "plate_p0_probe_stabilized_demo", "figures"):
        (base_dir / name).mkdir(parents=True)

    dry_run = prepare_calibrated_results_layout(base_dir, dry_run=True)
    assert dry_run.status == "would_archive"
    assert {move.source.name for move in dry_run.moves} == {
        "plate_m5_probe_stabilized_demo",
        "plate_p0_probe_stabilized_demo",
        "figures",
    }

    archived = prepare_calibrated_results_layout(base_dir, dry_run=False)
    assert archived.status == "archived"
    assert (base_dir / "with_rho_fixed" / "plate_m5_probe_stabilized_demo").is_dir()
    assert (base_dir / "with_rho_fixed" / "figures").is_dir()
    assert archived.manifest_path.exists()

    rerun = prepare_calibrated_results_layout(base_dir, dry_run=False)
    assert rerun.status == "already_archived"


def test_generate_calibrated_experiment_vs_simulation_figures(tmp_path: Path) -> None:
    simulations_dir = tmp_path / "simulations"
    experiments_root = tmp_path / "experiments"
    figures_dir = tmp_path / "figures"
    _make_simulation_case(simulations_dir, target_C=-10.0)
    _make_experimental_group(experiments_root, label="min10")

    written = generate_calibrated_experiment_vs_simulation_figures(
        simulations_dir=simulations_dir,
        output_dir=figures_dir,
        experiments_root=experiments_root,
        targets_C=(-10.0,),
    )

    names = {path.name for path in written}
    assert "compare_sim_vs_experiment_min10_12p5_11p0mm.png" in names
    assert "compare_sim_vs_experiment_panel_min10_12p5_11p0mm.png" in names
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_calibrated_front_speed_plots(tmp_path: Path) -> None:
    simulations_dir = tmp_path / "simulations"
    output_dir = tmp_path / "velocity_study"
    targets = (-5.0, -10.0)
    for target_C in targets:
        _make_simulation_case(simulations_dir, target_C=target_C)

    output_dir.mkdir(parents=True)
    (output_dir / "discussion.md").write_text("stale", encoding="utf-8")
    (output_dir / "front_velocity_vs_time.png").write_text("stale", encoding="utf-8")

    written = generate_calibrated_front_speed_plots(
        simulations_dir=simulations_dir,
        output_dir=output_dir,
        expected_targets=targets,
    )

    expected_names = {
        "front_velocity_raw_vs_time.png",
        "front_velocity_no_moving_average_vs_time.png",
        "front_velocity_vs_time.png",
        "front_position_vs_time.png",
        "tc_segment_speeds_by_temperature.png",
        "speed_summary.csv",
        "README.md",
    }
    assert expected_names == {path.name for path in written}
    assert not (output_dir / "discussion.md").exists()
    for name in expected_names:
        assert (output_dir / name).exists()
        assert (output_dir / name).stat().st_size > 0
    assert "generate_calibrated_front_speed_plots" in (output_dir / "README.md").read_text(encoding="utf-8")
