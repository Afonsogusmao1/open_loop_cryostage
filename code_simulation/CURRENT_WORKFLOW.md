# Current Workflow Status

## Active optimization path
The active open-loop workflow is the BO-based `full_process_article` path.

- Main single-run entry point: [run_open_loop_optimization.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_optimization.py)
- Current comparison-study entry point: [run_open_loop_schedule_sensitivity_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_schedule_sensitivity_study.py)
- Admissibility diagnostics entry point: [run_reachability_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_reachability_diagnostics.py)
- Historical locked-workflow export entry point: [export_final_locked_n3_workflow.py](/home/fenics/shared/Open_loop/code_simulation/export_final_locked_n3_workflow.py)

The workflow objective remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, to obtain approximately linear freezing-front progression.

The outer design variables are the knot temperatures `theta`. Knot times come from `OpenLoopProblemConfig.knot_times_s` and are externally fixed for the chosen knot count and external schedule; they are not optimized by the active workflow.

The control architecture remains:

`theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> J(theta)`

The inner PID tracking layer is distinct from the outer BO-based reference-trajectory design layer and should remain documented that way.

## Decision/status memo
The current workflow status is defined by two completed study campaigns.

### 1. External knot-time schedule sensitivity
- Study folder: `code_simulation/results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`
- Compared schedules: `uniform`, `early_dense`, `late_dense`
- Compared knot counts: `N=3`, `N=4`
- Result:
  - `uniform` preferred `N=4`
  - `early_dense` preferred `N=3`
  - `late_dense` preferred `N=4`
- Consequence: the `N=3` versus `N=4` ranking is schedule-dependent.
- Workflow consequence: `uniform` is now the default external knot-time schedule for ongoing comparisons and workflow development.

### 2. Uniform multiseed confirmation
- Study folders:
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed29/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed41/`
- Compared knot counts: `N=3`, `N=4`
- Shared settings: `uniform` schedule, `init_points=6`, `n_iter=16`, seeds `17`, `29`, and `41`
- Result:
  - `N=3` is more stable across seeds
  - `N=4` shows upside but higher optimizer sensitivity
  - `N=4` does not yet demonstrate robust superiority under `uniform`
- Consequence: model order remains unresolved under the default `uniform` schedule.

## Current workflow labels
Use the following labels consistently.

- Historical reference workflow:
  - `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`
  - This remains reproducible historical reference material from the earlier locked-`N=3` decision.
- Current default external knot-time schedule:
  - `uniform`
- Pragmatic working default for ongoing engineering use:
  - `uniform + N=3`
- Current unresolved challenger for ongoing workflow development:
  - `uniform + N=4`
- Scientific conclusion status:
  - knot count is not scientifically settled under the current workflow

No current repository document should claim that `N=3` is the authoritative baseline in a strong scientific sense, and no current repository document should claim that `N=4` has already replaced it.

## Active admissibility behavior
- Admissibility includes characterization-derived transient admissibility.
- Admissibility includes long-duration hold admissibility derived from freezing-run plate telemetry.
- Admissibility is checked before expensive simulation through `build_reference_profile_from_theta(...)`.
- Inadmissible candidates are rejected early, logged as infeasible, and assigned the configured deterministic penalty objective without launching the expensive cascade.
- Warming is not currently supported and must not be silently assumed.

## Active versus historical study support
- Active: BO-based fixed-`N` full-process workflow in `run_open_loop_optimization.py`.
- Active comparison support: `run_open_loop_schedule_sensitivity_study.py`.
- Historical fixed-`N` BO campaign support: `run_open_loop_fixed_n_bo_study.py`.
- Legacy or exploratory context: `run_open_loop_study.py`, `run_optimizer_learning_diagnostics.py`, and older `results/open_loop_study/` material.

## Current unresolved methodological questions
- The main unresolved methodological question is whether the externally fixed knot-time schedule is still too crude, even with `uniform` adopted as the default comparison schedule.
- The current unresolved model-order question is whether `N=4` can show robust superiority over `N=3` under `uniform` with stronger optimizer evidence.
