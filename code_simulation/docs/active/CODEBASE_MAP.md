# Codebase Map

## Core Flow

The active open-loop workflow follows this chain:

`theta -> T_ref(t) -> cryostage model -> T_plate(t) -> freezing solver -> z_front(t) -> objective`

The optimization is open-loop: the optimizer chooses `T_ref` knot temperatures `theta_C`, not direct front-control actions.

The current velocity-control BO objective follows the same simulation chain, then tracks `z_front(t)` against a constant-speed reference over `2.5-11.5 mm`.

## Main Modules

- `optimization/open_loop_workflow_config.py`: shared defaults, trajectory time-parameterization builders, and `OpenLoopProblemConfig` factories for active workflows
- `optimization/open_loop_problem.py`: converts `theta` into a reference profile, runs admissibility checks, loads front trajectories, and evaluates the scalar objective
- `simulation/open_loop_cascade.py`: couples the reduced cryostage model with the full freezing solver and returns the generated artifacts for one case
- `optimization/open_loop_optimizer.py`: coordinates evaluation history, run directories, incumbent tracking, and optimizer backend integration
- `optimization/open_loop_bayesian_optimizer.py`: thin BO backend wrapper around the vendored `bayes_opt` compatibility layer
- `optimization/velocity_objective.py`: constant front-speed objective and thermocouple-equivalent diagnostics
- `optimization/run_velocity_control_bo.py`: active velocity-control BO entry point
- `optimization/velocity_bo_reporting.py`: BO summaries, figures, and Markdown report generation
- `studies/run_n8_uniform_from_n3.py`: active `n8` uniform-from-`n3` study summarizer, queue builder, and fine-candidate selector
- `studies/run_bo_3knot_target_and_spacing_study.py`: active `n3/diagnostico` target sweep and spacing-comparison summarizer
- `studies/run_n3_bo_robustness_full_range.py`: tuned `_bov2_` `n3` robustness-study runner across the full target-speed range
- `studies/run_bo_4knot_uniform_full_range_study.py`: archived 4-knot uniform sweep summarizer and fine-candidate selector
- `core/config_files.py`: TOML loader for editable simulation profiles and BO settings
- `verification/run_velocity_control_evaluation.py`: manual confirmation of one cryostage trajectory against a target front speed
- `optimization/reachability_constraints.py`: transient and hold admissibility logic used to reject infeasible trajectories before expensive simulation
- `simulation/cryostage_model.py`: reduced inner-stage model used to transform `T_ref(t)` into modeled plate temperature
- `simulation/ambient.py`: fixed and interpolated ambient-temperature models used by the solver
- `simulation/solver.py`: full freezing simulation
- `simulation/front_tracking.py`: front extraction and post-processing utilities
- `core/trajectory_profiles.py`: reusable temperature profile classes
- `core/paths.py`: canonical project, result, and vendored BO compatibility paths

## Entry Points

- `python -m code_simulation.optimization.run_velocity_control_bo`: active velocity-control BO workflow
- `python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config`: inspect the effective velocity-control BO settings without running solver cases
- `python -m code_simulation.studies.run_n8_uniform_from_n3 --dry-run`: inspect the active `n8` uniform-from-`n3` campaign and its locked 50-run coarse matrix
- `python -m code_simulation.verification.run_velocity_control_evaluation --dry-run-config`: inspect the manual velocity-control evaluation without running a solver case
- `python -m code_simulation.verification.run_velocity_control_evaluation --overwrite`: run one manual trajectory against the configured target speed into `results/active/n{N}/fine/`
- `python -m code_simulation.studies.run_bo_3knot_target_and_spacing_study --dry-run`: inspect the `n3/diagnostico` study scan without writing outputs
- `python -m code_simulation.studies.run_n3_bo_robustness_full_range --dry-run`: inspect the tuned `_bov2_` `n3` robustness study scan without writing outputs
- `python -m code_simulation.studies.run_bo_4knot_uniform_full_range_study --dry-run`: inspect the archived 4-knot study scan without writing outputs
- `python -m code_simulation.studies.run_open_loop_phase5_binary_parameterization_study`: active Phase 5 runner
- `python -m code_simulation.studies.run_open_loop_schedule_sensitivity_study`: controlled external schedule sensitivity support
- `python -m code_simulation.verification.run_reachability_diagnostics`: admissibility diagnostics
- `python -m code_simulation.verification.run_front_speed_reachability_study`: cryostage-constrained front-speed reachability study

## Supporting And Legacy Scripts

- `python -m code_simulation.studies.run_open_loop_fixed_n_bo_study`: older BO comparison driver kept for context
- `python -m code_simulation.studies.run_open_loop_study`: legacy exploratory workflow
- `python -m code_simulation.verification.run_full_freezing_diagnostics` and `python -m code_simulation.verification.run_full_freezing_runtime_compare`: lower-level full-solver inspection scripts

## Practical Navigation

If you want to understand one active optimization run end-to-end, read files in this order:

1. `code_simulation.optimization.run_velocity_control_bo`
2. `code_simulation/configs/SETTINGS.md`
3. `optimization/velocity_objective.py`
4. `verification/run_velocity_control_evaluation.py` for manual confirmation
5. `optimization/open_loop_workflow_config.py`
6. `optimization/open_loop_optimizer.py`
7. `optimization/open_loop_problem.py`
8. `simulation/open_loop_cascade.py`
9. `simulation/cryostage_model.py`
10. `simulation/ambient.py`
11. `simulation/solver.py`
12. `simulation/front_tracking.py`
