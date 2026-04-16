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
- The repository should not anchor the active scientific discussion on any preselected knot count.
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

- `code_simulation/run_open_loop_optimization.py`
- `code_simulation/run_open_loop_phase5_binary_parameterization_study.py`
- `code_simulation/run_open_loop_schedule_sensitivity_study.py`
- `code_simulation/run_reachability_diagnostics.py`
- `code_simulation/CURRENT_WORKFLOW.md`
- `code_simulation/CODEBASE_MAP.md`

## Active versus legacy

- Treat the BO-based `full_process_article` path as active.
- Treat `run_open_loop_phase5_binary_parameterization_study.py` as the active Phase 5 comparison runner: a narrow two-arm parameterization study under the fixed active workflow.
- Treat `run_open_loop_schedule_sensitivity_study.py` as controlled support for external time-schedule sensitivity, not as the Phase 5 default comparison path.
- Treat `run_open_loop_fixed_n_bo_study.py` as earlier study support, not as the current default comparison workflow.
- Treat `run_open_loop_study.py` and `run_optimizer_learning_diagnostics.py` as legacy or exploratory support for older study campaigns, not as the current authoritative optimization workflow.
- Historical study folders under `code_simulation/results/open_loop_study/` and older non-BO optimization folders under `code_simulation/results/open_loop_optimization/` remain useful for context, but they are not the current baseline.

## Current methodological caveats

- The main unresolved methodological question is whether the externally fixed time parameterization is too crude.
- The main unresolved scientific question is not a specific knot-count choice, but whether the chosen trajectory representation and observable definition are robust enough for a defensible BO study.
