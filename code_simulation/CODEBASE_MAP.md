# Codebase Map

## Core Flow

The active open-loop workflow follows this chain:

`theta -> T_ref(t) -> cryostage model -> T_plate(t) -> freezing solver -> z_front(t) -> objective`

The optimization is open-loop: the optimizer chooses trajectory parameters `theta`, not direct front-control actions.

## Main Modules

- `open_loop_workflow_config.py`: shared defaults, trajectory time-parameterization builders, and `OpenLoopProblemConfig` factories for active workflows
- `open_loop_problem.py`: converts `theta` into a reference profile, runs admissibility checks, loads front trajectories, and evaluates the scalar objective
- `open_loop_cascade.py`: couples the reduced cryostage model with the full freezing solver and returns the generated artifacts for one case
- `open_loop_optimizer.py`: coordinates evaluation history, run directories, incumbent tracking, and optimizer backend integration
- `open_loop_bayesian_optimizer.py`: thin BO backend wrapper around the vendored `bayes_opt` compatibility layer
- `reachability_constraints.py`: transient and hold admissibility logic used to reject infeasible trajectories before expensive simulation
- `cryostage_model.py`: reduced inner-stage model used to transform `T_ref(t)` into modeled plate temperature
- `solver.py`: full freezing simulation
- `front_tracking.py`: front extraction and post-processing utilities

## Entry Points

- `run_open_loop_optimization.py`: main single-run CLI for the active BO workflow
- `run_open_loop_phase5_binary_parameterization_study.py`: active Phase 5 narrow two-arm BO comparison runner for candidate trajectory parameterizations
- `run_open_loop_schedule_sensitivity_study.py`: controlled external time-schedule sensitivity support runner
- `run_reachability_diagnostics.py`: diagnostics for admissibility support data and screening behavior

## Supporting And Legacy Scripts

- `run_open_loop_fixed_n_bo_study.py`: older BO comparison driver kept for context
- `run_open_loop_study.py`: legacy exploratory workflow
- `run_optimizer_learning_diagnostics.py`: diagnostics for older exploratory studies
- `run_full_freezing_diagnostics.py` and `run_full_freezing_runtime_compare.py`: lower-level full-solver inspection scripts

## Practical Navigation

If you want to understand one active optimization run end-to-end, read files in this order:

1. `run_open_loop_optimization.py`
2. `open_loop_workflow_config.py`
3. `open_loop_optimizer.py`
4. `open_loop_problem.py`
5. `open_loop_cascade.py`
6. `cryostage_model.py`
7. `solver.py`
8. `front_tracking.py`
