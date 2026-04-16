# Current Workflow Status

## Active optimization path

The active open-loop workflow is the BO-based `full_process_article` path.

- Main single-run entry point: `run_open_loop_optimization.py`
- Current Phase 5 comparison-study entry point: `run_open_loop_phase5_binary_parameterization_study.py`
- External schedule-sensitivity support entry point: `run_open_loop_schedule_sensitivity_study.py`
- Admissibility diagnostics entry point: `run_reachability_diagnostics.py`

The workflow objective remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, to obtain approximately linear freezing-front progression.

The outer design variables are trajectory parameters `theta`. The current active workflow uses externally fixed time parameterizations for the reference trajectory.

The control architecture remains:

`theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> J(theta)`

The inner PID tracking layer is distinct from the outer BO-based reference-trajectory design layer and should remain documented that way.

## Active workflow interpretation

The active workflow should be understood as a trajectory-design workflow, not as a workflow centered on any preselected knot count.

The present scientific focus is:

- aligning the code with the intended active workflow,
- making the front definition explicit,
- testing observable robustness,
- running the Phase 5 narrow BO study on exactly two candidate trajectory parameterizations.

## Active admissibility behavior

- Admissibility includes characterization-derived transient admissibility.
- Admissibility includes long-duration hold admissibility derived from freezing-run plate telemetry.
- Admissibility is checked before expensive simulation through the active reference-profile validation path.
- Inadmissible candidates are rejected early, logged as infeasible, and assigned the configured deterministic penalty objective without launching the expensive cascade.
- Warming is not currently supported and must not be silently assumed.

## Active versus historical study support

- Active: BO-based full-process workflow in `run_open_loop_optimization.py`
- Active Phase 5 binary parameterization comparison: `run_open_loop_phase5_binary_parameterization_study.py`
- Controlled external schedule-sensitivity support: `run_open_loop_schedule_sensitivity_study.py`
- Historical BO support: `run_open_loop_fixed_n_bo_study.py`
- Legacy or exploratory context: `run_open_loop_study.py`, `run_optimizer_learning_diagnostics.py`, and older `results/open_loop_study/` material

## Current unresolved methodological questions

- Whether the externally fixed time parameterization is still too crude
- Whether the operational front definition is robust enough for optimization
- Whether the chosen observable is the right one for a defensible BO study
- Whether the final trajectory parameterization is scientifically adequate

## Current practical rule

Do not treat the current workflow as anchored to a specific knot-count discussion.

Treat the current workflow as a staged program:

1. align defaults,
2. fix front definition,
3. test observable robustness,
4. only then run a narrow BO study.
