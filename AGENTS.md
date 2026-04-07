# AGENTS.md

## Scope
This repository is scientific software for offline open-loop cryostage trajectory design. The active implementation lives in [code_simulation](/home/fenics/shared/Open_loop/code_simulation), while [data](/home/fenics/shared/Open_loop/data) contains experimental inputs, characterization material, and validation references that should be treated as source data rather than casual edit targets.

## Active workflow facts
- The active open-loop workflow is BO-based.
- The workflow objective remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, to obtain approximately linear freezing-front progression.
- The outer design variables are knot temperatures `theta`; the existing inner PID tracking layer is not replaced or re-optimized.
- The intended cascade remains `theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> J(theta)`.
- Knot times are externally fixed by the chosen formulation, knot count, and external schedule. The active workflow does not optimize knot times.
- The current default external knot-time schedule for ongoing comparisons is `uniform`.
- The locked `N=3` bundle is now a reproducible historical reference, not a current authoritative baseline in a strong scientific sense.
- For ongoing engineering workflow use, describe `uniform + N=3` as the pragmatic working default.
- Describe `uniform + N=4` as the current unresolved challenger or exploratory candidate.
- Do not document or imply that knot count is scientifically settled.
- Warming is not currently supported. Do not silently assume admissible warming segments.

## Active admissibility policy
- Active admissibility is applied before the expensive freezing simulation is launched.
- The admissibility stack combines characterization-derived transient admissibility, long-duration hold admissibility derived from freezing-run plate telemetry, and early rejection of infeasible candidates.
- In the active BO workflow, infeasible candidates receive the configured deterministic penalty objective and are logged without running the expensive cascade.

## Status-defining studies
- External schedule sensitivity: `code_simulation/results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`
  - Result: the `N=3` versus `N=4` ranking is schedule-dependent.
  - Consequence: `uniform` is now the default external knot-time schedule for ongoing comparisons.
- Uniform multiseed confirmation: `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`, `..._seed29/`, and `..._seed41/`
  - Result: `N=3` is more stable across seeds; `N=4` has upside but higher optimizer sensitivity and no robust superiority.
  - Consequence: model order remains unresolved under the default `uniform` schedule.

## Current workflow labels to preserve
- Historical reference workflow: `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`
- Current default external schedule: `uniform`
- Pragmatic working default: `uniform + N=3`
- Unresolved challenger: `uniform + N=4`
- Scientific status: knot count unresolved under the current workflow

## Authoritative entry points
- Single fixed-`N` BO run: [code_simulation/run_open_loop_optimization.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_optimization.py)
- Current schedule/model-order comparison study: [code_simulation/run_open_loop_schedule_sensitivity_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_schedule_sensitivity_study.py)
- Admissibility diagnostics: [code_simulation/run_reachability_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_reachability_diagnostics.py)
- Historical locked workflow export: [code_simulation/export_final_locked_n3_workflow.py](/home/fenics/shared/Open_loop/code_simulation/export_final_locked_n3_workflow.py)
- Current workflow note: [code_simulation/CURRENT_WORKFLOW.md](/home/fenics/shared/Open_loop/code_simulation/CURRENT_WORKFLOW.md)
- Results index: [RESULTS_INDEX.md](/home/fenics/shared/Open_loop/RESULTS_INDEX.md)
- Decision/status memo: [DECISION_STATUS_MEMO.md](/home/fenics/shared/Open_loop/DECISION_STATUS_MEMO.md)

## Active versus legacy
- Treat the BO-based `full_process_article` path as active.
- Treat [code_simulation/run_open_loop_fixed_n_bo_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_fixed_n_bo_study.py) as earlier fixed-`N` study support, not as the current default comparison workflow.
- Treat [code_simulation/run_open_loop_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_study.py) and [code_simulation/run_optimizer_learning_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_optimizer_learning_diagnostics.py) as legacy or exploratory support for older study campaigns, not as the current authoritative optimization workflow.
- Historical study folders under `code_simulation/results/open_loop_study/` and older non-BO optimization folders under `code_simulation/results/open_loop_optimization/` remain useful for context, but they are not the current baseline.

## Current methodological caveats
- The main unresolved methodological question is whether the externally fixed knot-time schedule is too crude.
- The current unresolved model-order question is whether `N=4` can show robust superiority over `N=3` under the default `uniform` schedule with stronger optimization evidence.
