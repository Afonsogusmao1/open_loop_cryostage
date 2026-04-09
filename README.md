# Open_loop

`Open_loop` is a scientific software repository for model-based open-loop design of a plate-temperature reference trajectory for the inner PID-tracked cryostage, with the aim of obtaining approximately linear freezing-front progression during freezing.

## Current workflow status
- The active open-loop workflow is BO-based.
- The optimization variables are knot temperatures only.
- Knot times are externally fixed by the selected formulation, knot count, and external schedule. The active workflow does not optimize knot times.
- The current default external knot-time schedule for ongoing comparisons and workflow development is `uniform`.
- The previously locked `N=3` workflow exported under `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/` remains a reproducible historical reference, not a current authoritative baseline in a strong scientific sense.
- The completed external schedule-sensitivity study showed that the `N=3` versus `N=4` ranking is schedule-dependent.
- The completed uniform multiseed confirmation study showed that `N=3` is more stable across seeds, while `N=4` shows upside but higher optimizer sensitivity and no robust superiority.
- The pragmatic working default for ongoing engineering workflow use is `uniform + N=3`.
- The current unresolved challenger for ongoing workflow development is `uniform + N=4`.
- The repository should not claim that knot count is scientifically settled under the current workflow.

The control separation must stay explicit:

`theta (temperature knots) -> T_ref(t) -> inner PID / cryostage model response -> T_plate(t) -> freezing solver -> z_front(t) -> objective`

The outer BO layer designs the reference trajectory. The inner PID tracking layer is already part of the plant-side cryostage behavior and must not be blurred into the outer optimization problem.

## Active admissibility policy
- Characterization-derived transient admissibility is enforced for piecewise-linear cooling segments.
- Long-duration hold admissibility is enforced using support derived from freezing-run plate telemetry.
- Candidates are screened before expensive simulation; inadmissible candidates are rejected early and assigned the configured deterministic penalty objective.
- Warming is not currently supported and must not be silently assumed.

## Repository map
- [code_simulation](/home/fenics/shared/Open_loop/code_simulation): active implementation, workflow runners, diagnostics, status notes, and result bundles for the open-loop optimization stack.
- [data](/home/fenics/shared/Open_loop/data): characterization data, validation inputs, calibrated comparisons, and related experimental reference material.

## Active scripts
- [code_simulation/run_open_loop_optimization.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_optimization.py): current single-run BO entry point for the `full_process_article` formulation.
- [code_simulation/run_open_loop_schedule_sensitivity_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_schedule_sensitivity_study.py): current study runner for external knot-time schedule comparisons and uniform-schedule `N=3` versus `N=4` workflow checks.
- [code_simulation/run_reachability_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_reachability_diagnostics.py): diagnostics for the active transient-plus-hold admissibility layer.
- [code_simulation/export_final_locked_n3_workflow.py](/home/fenics/shared/Open_loop/code_simulation/export_final_locked_n3_workflow.py): exports the historical locked `N=3` workflow bundle.

## Historical and contextual study entry points
- [code_simulation/run_open_loop_fixed_n_bo_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_fixed_n_bo_study.py): earlier fixed-`N` BO comparison runner retained as historical comparison material.
- [code_simulation/run_open_loop_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_study.py) and [code_simulation/run_optimizer_learning_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_optimizer_learning_diagnostics.py) reflect older exploratory or legacy study workflows and should not be treated as the current default optimization path.

## Where to look next
- Current workflow note: [code_simulation/CURRENT_WORKFLOW.md](/home/fenics/shared/Open_loop/code_simulation/CURRENT_WORKFLOW.md)
- Code map: [code_simulation/CODEBASE_MAP.md](/home/fenics/shared/Open_loop/code_simulation/CODEBASE_MAP.md)
- Results index: [RESULTS_INDEX.md](/home/fenics/shared/Open_loop/RESULTS_INDEX.md)
- Decision/status memo: [DECISION_STATUS_MEMO.md](/home/fenics/shared/Open_loop/DECISION_STATUS_MEMO.md)
- Agent guidance: [AGENTS.md](/home/fenics/shared/Open_loop/AGENTS.md)
