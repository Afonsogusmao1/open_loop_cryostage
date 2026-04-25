from __future__ import annotations

import csv
import sys
from pathlib import Path

from code_simulation.studies import n8_uniform_from_n3_impl as n8_study


def test_n8_uniform_from_n3_dry_run_reports_locked_matrix(monkeypatch, capsys, tmp_path: Path) -> None:
    bo_runs_root = tmp_path / "bo_runs"
    bo_runs_root.mkdir()
    output_root = tmp_path / "diagnostico"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_n8_uniform_from_n3",
            "--dry-run",
            "--bo-runs-root",
            str(bo_runs_root),
            "--output-root",
            str(output_root),
        ],
    )

    n8_study.main()
    captured = capsys.readouterr().out

    assert "Expected coarse matrix size: 50" in captured
    assert "Interpolated n8 theta0:" in captured
    assert "--num-knots 8" in captured
    assert "--knot-time-schedule uniform" in captured
    assert "--dry-run-config" in captured


def test_n8_uniform_from_n3_writes_diagnostic_bundle(monkeypatch, tmp_path: Path) -> None:
    bo_runs_root = tmp_path / "bo_runs"
    bo_runs_root.mkdir()
    output_root = tmp_path / "diagnostico"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_n8_uniform_from_n3",
            "--bo-runs-root",
            str(bo_runs_root),
            "--output-root",
            str(output_root),
            "--overwrite",
        ],
    )

    n8_study.main()

    expected_files = {
        "README.md",
        "scope_note.md",
        "final_study_summary.md",
        "study_summary.csv",
        "decision_summary.csv",
        "fine_confirmation_candidates.csv",
        "run_commands.sh",
        "fine_confirmation_commands.sh",
        "representative_dry_run_command.sh",
    }
    assert expected_files == {path.name for path in output_root.iterdir()}

    with (output_root / "study_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50

    decision_summary = (output_root / "decision_summary.csv").read_text(encoding="utf-8")
    fine_candidates = (output_root / "fine_confirmation_candidates.csv").read_text(encoding="utf-8")
    run_commands = (output_root / "run_commands.sh").read_text(encoding="utf-8")

    assert "study_conclusion" in decision_summary
    assert "recommended_confirmation_run_name" in fine_candidates
    assert "code_simulation.optimization.run_velocity_control_bo" in run_commands
