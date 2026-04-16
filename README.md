# Open_loop

`Open_loop` is a scientific software repository for model-based open-loop design of a plate-temperature reference trajectory for the inner PID-tracked cryostage, with the aim of obtaining approximately linear freezing-front progression during freezing.

## Current workflow status

- The active open-loop workflow is BO-based.
- The optimization variables are trajectory parameters only.
- The current workflow uses externally fixed time parameterizations for the reference trajectory.
- The repository should not anchor the active scientific discussion on any specific knot count.
- The current methodological priority is to:
  1. align code defaults with the intended active workflow,
  2. make the freezing-front definition explicit,
  3. test observable robustness and sensitivity,
  4. only then resume a narrow BO study.
- The repository should not claim that the final trajectory parameterization is scientifically settled.

The control separation must stay explicit:

`theta -> T_ref(t) -> inner PID / cryostage model response -> T_plate(t) -> freezing solver -> z_front(t) -> objective`

The outer BO layer designs the reference trajectory. The inner PID tracking layer is already part of the plant-side cryostage behavior and must not be blurred into the outer optimization problem.

## Active admissibility policy

- Characterization-derived transient admissibility is enforced for admissible cooling segments.
- Long-duration hold admissibility is enforced using support derived from freezing-run plate telemetry.
- Candidates are screened before expensive simulation; inadmissible candidates are rejected early and assigned the configured deterministic penalty objective.
- Warming is not currently supported and must not be silently assumed.

## Repository map

- `code_simulation`: active implementation, workflow runners, diagnostics, status notes, and result bundles for the open-loop optimization stack.
- `data`: characterization data, validation inputs, calibrated comparisons, and related experimental reference material.

## Active scripts

- `code_simulation/run_open_loop_optimization.py`: current single-run BO entry point for the active formulation.
- `code_simulation/run_open_loop_phase5_binary_parameterization_study.py`: current Phase 5 runner for the narrow two-arm BO comparison between candidate trajectory parameterizations.
- `code_simulation/run_open_loop_schedule_sensitivity_study.py`: controlled support runner for external time-schedule sensitivity, not the default Phase 5 path.
- `code_simulation/run_reachability_diagnostics.py`: diagnostics for the active transient-plus-hold admissibility layer.

## Historical and contextual study entry points

- `code_simulation/run_open_loop_fixed_n_bo_study.py`: earlier BO comparison runner retained as historical comparison material.
- `code_simulation/run_open_loop_study.py` and `code_simulation/run_optimizer_learning_diagnostics.py`: older exploratory or legacy study workflows and should not be treated as the current default optimization path.

## Where to look next

- `OPEN_LOOP_SCIENTIFIC_SPEC.MD`
- `OPEN_LOOP_IMPLEMENTATION_PHASES_AND_PROMPTS.md`
- `code_simulation/CURRENT_WORKFLOW.md`
- `code_simulation/CODEBASE_MAP.md`
- `AGENTS.md`

## Authoritative documents

- `OPEN_LOOP_SCIENTIFIC_SPEC.MD` — scientific objective, modelling assumptions, open questions, and what must not be claimed.
- `OPEN_LOOP_IMPLEMENTATION_PHASES_AND_PROMPTS.md` — phased implementation plan and prompt structure.
- `AGENTS.md` — repository operating rules for agent-based work.
