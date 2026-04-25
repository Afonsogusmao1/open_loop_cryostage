# AGENTS.md

## Scope

This repository is scientific software for offline open-loop cryostage trajectory design. The active implementation lives in `code_simulation`, while `data` contains experimental inputs, characterization material, and validation references that should be treated as source data rather than casual edit targets.

## Active workflow facts

- The active open-loop workflow is BO-based.
- The workflow objective is model-based open-loop design of a plate-temperature reference trajectory for the inner PID, to obtain approximately linear freezing-front progression.
- The outer design variables are trajectory parameters `theta`; the existing inner PID tracking layer is not replaced or re-optimized.
- The intended cascade remains:
  `theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> J(theta)`.
- The current active workflow uses externally fixed time parameterizations for the reference trajectory.
- The active production campaign is `n8` uniform BO anchored on the defended `n3` uniform bundle.
- The main unresolved methodological question is whether the current trajectory parameterization is scientifically adequate and robust.
- Warming is not currently supported. Do not silently assume admissible warming segments.

## Active admissibility policy

- Active admissibility is applied before the expensive freezing simulation is launched.
- The admissibility stack combines characterization-derived transient admissibility, long-duration hold admissibility derived from freezing-run plate telemetry, and early rejection of infeasible candidates.
- In the active BO workflow, infeasible candidates receive the configured deterministic penalty objective and are logged without running the expensive cascade.

## Current workflow labels to preserve

- Active workflow: BO-based open-loop trajectory design
- Control architecture: outer BO over trajectory parameters, inner cryostage response represented through the reduced model
- Scientific status: final trajectory parameterization not yet closed
- Hard constraint: no unsupported warming

## Authoritative entry points

- `python -m code_simulation.optimization.run_velocity_control_bo`
- `python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config`
- `python -m code_simulation.verification.run_velocity_control_evaluation --dry-run-config`
- `python -m code_simulation.studies.run_n8_uniform_from_n3 --dry-run`
- `python -m code_simulation.studies.run_open_loop_phase5_binary_parameterization_study`
- `python -m code_simulation.studies.run_open_loop_schedule_sensitivity_study`
- `python -m code_simulation.verification.run_reachability_diagnostics`
- `code_simulation/docs/active/CURRENT_WORKFLOW.md`
- `code_simulation/docs/active/CODEBASE_MAP.md`

The implementation lives under:

- `code_simulation/optimization`
- `code_simulation/simulation`
- `code_simulation/verification`
- `code_simulation/studies`
- `code_simulation/reporting`
- `code_simulation/core`
- `code_simulation/configs`

## Active versus legacy

- Treat the BO-based `full_process_article` path as active.
- Treat `run_open_loop_phase5_binary_parameterization_study.py` as the active Phase 5 comparison runner: a narrow two-arm parameterization study under the fixed active workflow.
- Treat `run_open_loop_schedule_sensitivity_study.py` as controlled support for external time-schedule sensitivity, not as the Phase 5 default comparison path.
- Treat `run_open_loop_fixed_n_bo_study.py` as earlier study support, not as the current default comparison workflow.
- Treat `run_open_loop_study.py` as legacy exploratory support for older study campaigns, not as the current authoritative optimization workflow.
- Historical study folders under `code_simulation/results/open_loop_study/` and older non-BO optimization folders under `code_simulation/results/open_loop_optimization/` remain useful for context, but they are not the current baseline.

## Current methodological caveats

- The main unresolved methodological question is whether the externally fixed time parameterization is too crude beyond the current defended `n8 <- n3` campaign.
- The main unresolved scientific question is not a specific knot-count choice, but whether the chosen trajectory representation and observable definition are robust enough for a defensible BO study.
- Do not change mesh, time-step, BO budget, knot count, velocity-control targets, warm-hold setup, or ambient-temperature behavior by editing scientific modules first. Prefer `code_simulation/configs/simulation_profiles.toml`, `code_simulation/configs/velocity_control.toml`, and `code_simulation/configs/bo.toml`, then validate with `--dry-run-config`.
