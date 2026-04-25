# Open_loop

`Open_loop` is a scientific software repository for model-based open-loop design of a plate-temperature reference trajectory for the inner PID-tracked cryostage, with the aim of obtaining approximately linear freezing-front progression during freezing.

## Current workflow status

- The active open-loop workflow is now velocity-control BO.
- The optimization variables are `T_ref` knot temperatures only.
- The current workflow uses externally fixed knot times for the reference trajectory.
- The active campaign is currently `n8` uniform BO anchored on the defended `n3` uniform result, while the workflow architecture remains generic across knot counts.
- The current methodological priority is to:
  1. align code defaults with the intended active workflow,
  2. make the freezing-front definition explicit,
  3. test observable robustness and sensitivity,
  4. run velocity-control BO for a user-specified front speed.
- The repository should not claim that the final trajectory parameterization is scientifically settled.

The control separation must stay explicit:

`theta -> T_ref(t) -> inner PID / cryostage model response -> T_plate(t) -> freezing solver -> z_front(t) -> objective`

The outer BO layer designs the reference trajectory. The inner PID tracking layer is already part of the plant-side cryostage behavior and must not be blurred into the outer optimization problem.

## Active admissibility policy

- Characterization-derived transient admissibility is enforced for admissible cooling segments.
- Long-duration hold admissibility remains available for diagnostics, but the active velocity-control protocol starts without a warm hold.
- Characterization support is interpreted with an explicit `+-0.5 C` comparison margin from `code_simulation/configs/velocity_control.toml`; this margin does not widen the hard physical BO bounds `[-21.0, 0.0] C`.
- Candidates are screened before expensive simulation; inadmissible candidates are rejected early and assigned the configured deterministic penalty objective.
- Warming is not currently supported and must not be silently assumed.

## Repository map

- `code_simulation`: packaged implementation, diagnostics, status notes, and result bundles for the open-loop optimization stack.
- `code_simulation/configs`: editable TOML settings for simulation profiles, velocity-control targets, BO budget/trajectory settings, and ambient-temperature interpolation.
- `code_simulation/core`: shared arrays, paths, plotting, CSV, and trajectory-profile helpers.
- `code_simulation/simulation`: cryostage model, geometry/material definitions, front tracking, freezing solver, and cascade execution.
- `code_simulation/optimization`: active problem configuration, admissibility constraints, objective evaluation, optimizer orchestration, BO backend, and single-run BO implementation.
- `code_simulation/verification`: calibration, reachability, hold-telemetry, and full-freezing diagnostic workflows.
- `code_simulation/studies`: Phase 5, schedule-sensitivity, fixed-N, and legacy study runners.
- `code_simulation/reporting`: final workflow export/reporting utilities.
- `code_simulation/results`: existing and future result artifacts; this directory remains outside the code package split.
- `data`: characterization data, validation inputs, calibrated comparisons, and related experimental reference material.

## Active Commands

- `python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config`: print the effective velocity-control BO configuration without launching simulations.
- `python -m code_simulation.optimization.run_velocity_control_bo --simulation-profile optimization`: run the coarse velocity-control BO workflow.
- `python -m code_simulation.verification.run_velocity_control_evaluation --dry-run-config`: inspect one manual velocity-control evaluation from `configs/velocity_control.toml`.
- `python -m code_simulation.verification.run_velocity_control_evaluation --overwrite`: run one manual cryostage trajectory against the configured target front speed and write it under `results/active/n{N}/fine/`.
- `python -m code_simulation.studies.run_n8_uniform_from_n3 --dry-run`: inspect the active `n8` uniform-from-`n3` campaign, including the locked 50-run coarse matrix and representative BO command.
- `python -m code_simulation.studies.run_bo_3knot_target_and_spacing_study --dry-run`: inspect the active `n3/diagnostico` target/spacing study scan without writing outputs.
- `python -m code_simulation.studies.run_bo_4knot_uniform_full_range_study --dry-run`: inspect the archived 4-knot uniform full-range study scan without writing outputs.
- `python -m code_simulation.studies.run_open_loop_phase5_binary_parameterization_study`: current Phase 5 runner for the narrow two-arm BO comparison between candidate trajectory parameterizations.
- `python -m code_simulation.studies.run_open_loop_schedule_sensitivity_study`: controlled support runner for external time-schedule sensitivity, not the default Phase 5 path.
- `python -m code_simulation.verification.run_reachability_diagnostics`: diagnostics for the active transient-plus-hold admissibility layer.
- `python -m code_simulation.verification.run_full_freezing_diagnostics --dry-run-config`: inspect full-freezing diagnostic simulation settings without launching solver cases.
- `python -m code_simulation.verification.run_front_speed_reachability_study --dry-run-config`: inspect the cryostage-constrained front-speed reachability study without launching simulations.

## Historical and contextual study entry points

- `python -m code_simulation.studies.run_open_loop_fixed_n_bo_study`: earlier BO comparison runner retained as historical comparison material.
- `python -m code_simulation.studies.run_open_loop_study`: older exploratory or legacy study workflow kept for historical context, not as the current default optimization path.

## Where to look next

- `code_simulation/configs/SETTINGS.md`
- `code_simulation/docs/active/CURRENT_WORKFLOW.md`
- `code_simulation/docs/active/CODEBASE_MAP.md`
- `AGENTS.md`
