from __future__ import annotations

from code_simulation.core import paths as paths_mod


def _patch_results_roots(monkeypatch, tmp_path):
    results_root = tmp_path / "results"
    active_root = results_root / "active"
    legacy_root = active_root / "bo_velocity_control"
    monkeypatch.setattr(paths_mod, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(paths_mod, "ACTIVE_RESULTS_ROOT", active_root)
    monkeypatch.setattr(paths_mod, "LEGACY_VELOCITY_CONTROL_RESULTS_ROOT", legacy_root)
    return results_root, active_root, legacy_root


def test_velocity_control_results_dir_uses_active_root(monkeypatch, tmp_path) -> None:
    _results_root, active_root, _legacy_root = _patch_results_roots(monkeypatch, tmp_path)

    assert paths_mod.velocity_control_results_dir() == active_root
    assert paths_mod.active_knot_dir(8, "coarse") == active_root / "n8" / "coarse"


def test_velocity_control_dirs_fall_back_to_legacy_layout(monkeypatch, tmp_path) -> None:
    _results_root, active_root, legacy_root = _patch_results_roots(monkeypatch, tmp_path)
    legacy_coarse = legacy_root / "n3" / "coarse"
    legacy_fine = legacy_root / "n3" / "fine"
    legacy_coarse.mkdir(parents=True)
    legacy_fine.mkdir(parents=True)

    assert not (active_root / "n3").exists()
    assert paths_mod.velocity_control_knot_dir(3) == legacy_root / "n3"
    assert paths_mod.velocity_control_coarse_dir(3) == legacy_coarse
    assert paths_mod.velocity_control_fine_dir(3) == legacy_fine


def test_velocity_control_dirs_prefer_active_layout_for_new_runs(monkeypatch, tmp_path) -> None:
    _results_root, active_root, legacy_root = _patch_results_roots(monkeypatch, tmp_path)
    active_diagnostic = active_root / "n8" / "diagnostico"
    legacy_diagnostic = legacy_root / "n8" / "diagnostico"
    active_diagnostic.mkdir(parents=True)
    legacy_diagnostic.mkdir(parents=True)

    assert paths_mod.velocity_control_knot_dir(8) == active_root / "n8"
    assert paths_mod.velocity_control_diagnostic_dir(8) == active_diagnostic
    assert paths_mod.velocity_control_coarse_dir(8) == active_root / "n8" / "coarse"

