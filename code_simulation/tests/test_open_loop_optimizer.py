from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from code_simulation.optimization.open_loop_bayesian_optimizer import BayesianOptimizationBackendResult, BayesianOptimizationConfig
from code_simulation.optimization import open_loop_optimizer as optimizer_mod
from code_simulation.optimization.open_loop_workflow_config import build_external_knot_times_s
from code_simulation.optimization.reachability_constraints import check_piecewise_linear_trajectory_admissibility


@dataclass(frozen=True)
class _PreparedCandidate:
    theta: tuple[float, ...]


def test_bayesopt_does_not_abort_on_inadmissible_theta0(monkeypatch, tmp_path: Path) -> None:
    theta0 = (-4.0, -8.0, -12.0)
    valid_theta = (-3.5, -7.0, -11.0)
    config = SimpleNamespace(T_ref_bounds_C=(-21.0, 0.0))
    cryostage_params = object()

    def fake_prepare(theta, _config):
        theta = tuple(float(value) for value in theta)
        if theta == theta0:
            raise ValueError("inadmissible theta0")
        return _PreparedCandidate(theta=theta)

    def fake_objective(theta, _config, _cryostage_params, _out_dir, _case_name, **_kwargs):
        theta = tuple(float(value) for value in theta)
        return SimpleNamespace(objective_value=float(sum(abs(value) for value in theta)))

    def fake_run_bayesian_optimization(*, theta0, default_bounds_C, config, evaluate_candidate, precheck_candidate):
        assert tuple(theta0) == theta0_values
        assert default_bounds_C == (-21.0, 0.0)
        assert isinstance(config, BayesianOptimizationConfig)

        seed_objective = evaluate_candidate(theta0_values, "seed", raw_candidate=theta0_values)
        assert seed_objective == 1.0e6

        precheck_candidate(valid_theta)
        evaluate_candidate(valid_theta, "random", raw_candidate=valid_theta)

        return BayesianOptimizationBackendResult(
            method="bayesopt",
            success=True,
            status=0,
            message="ok",
            nfev=2,
            nit=1,
            random_seed=17,
            init_points=1,
            n_iter=0,
            acquisition_kind="ei",
            acquisition_kappa=2.576,
            acquisition_xi=0.01,
            theta_bounds_C=(( -21.0, 0.0),) * 3,
            parameterization_kind="monotone_unit_box",
            search_bounds=((0.0, 1.0),) * 3,
            init_strategy="feasible_local",
            init_local_sigma=0.15,
            init_max_attempts_per_point=40,
            init_candidate_attempts=1,
            init_precheck_rejections=0,
            init_accepted_candidates=1,
            init_global_fallback_accepts=0,
            local_refinement_points=0,
            local_refinement_sigma=0.08,
            local_refinement_candidate_attempts=0,
            local_refinement_precheck_rejections=0,
            local_refinement_accepted_candidates=0,
            local_refinement_improved_candidates=0,
            package_version="test",
            package_path=tmp_path,
        )

    theta0_values = theta0
    monkeypatch.setattr(optimizer_mod, "prepare_open_loop_candidate", fake_prepare)
    monkeypatch.setattr(optimizer_mod, "run_bayesian_optimization", fake_run_bayesian_optimization)

    result = optimizer_mod.optimize_open_loop_theta(
        theta0=theta0,
        config=config,
        cryostage_params=cryostage_params,
        out_root_dir=tmp_path,
        run_name="bo_theta0_inadmissible",
        method="bo",
        bayesopt_config=BayesianOptimizationConfig(
            random_seed=17,
            init_points=1,
            n_iter=0,
            acquisition_kind="ei",
            acquisition_xi=0.01,
            theta_bounds_C=(( -21.0, 0.0),) * 3,
            seed_with_theta0=True,
            parameterization_kind="monotone_unit_box",
            init_strategy="feasible_local",
            init_local_sigma=0.15,
            init_max_attempts_per_point=40,
            local_refinement_points=0,
            local_refinement_sigma=0.08,
        ),
        objective_evaluator=fake_objective,
    )

    assert result.best_theta == valid_theta
    assert result.best_evaluation_index == 2
    assert len(result.history) == 2
    assert result.history[0].phase == "seed"
    assert result.history[0].is_valid is False
    assert result.history[0].feasibility_status == "infeasible"
    assert "inadmissible theta0" in result.history[0].error_message
    assert result.history[1].phase == "random"
    assert result.history[1].is_valid is True
    assert result.best_dir.is_dir()


def test_n8_uniform_n3_anchor_pass_through_profile_is_admissible() -> None:
    knot_times_s = build_external_knot_times_s(
        horizon_s=2400.0,
        num_knots=8,
        knot_time_schedule="uniform",
        knot_time_custom_support_tau=None,
    )
    theta_C = (
        -4.339973144034226,
        -6.439515477137384,
        -8.53905781024054,
        -10.638600143343697,
        -12.595795033464528,
        -14.410642480603028,
        -16.225489927741527,
        -18.04033737488003,
    )

    report = check_piecewise_linear_trajectory_admissibility(
        knot_times_s,
        theta_C,
        require_monotone_nonincreasing=True,
        allowed_cold_extrapolation_C=1.0,
        characterization_temperature_margin_C=0.5,
    )

    assert report.is_admissible is True


def test_explicit_early_minus15_hold_remains_inadmissible() -> None:
    report = check_piecewise_linear_trajectory_admissibility(
        (0.0, 360.0, 720.0, 2400.0),
        (-0.1, -15.0, -15.0, -20.0),
        require_monotone_nonincreasing=True,
        characterization_temperature_margin_C=0.5,
    )

    assert report.is_admissible is False
    assert "conservative settling time" in report.failure_summary()
