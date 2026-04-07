# code_simulation/AGENTS.md

## Folder role
This folder contains the active implementation of the open-loop cryostage optimization workflow, including the optimizer runners, admissibility checks, cascade coupling, solver integration, and authoritative result bundles.

## Active workflow
- The active optimization workflow is BO-based.
- The workflow objective remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, to obtain approximately linear freezing-front progression.
- The optimized variables are reference-temperature knot values, not knot times.
- Knot times are externally fixed by `OpenLoopProblemConfig.knot_times_s` for the selected formulation, knot count, and external schedule.
- The current default external knot-time schedule for ongoing comparisons is `uniform`.
- The locked `N=3` workflow exported from the earlier BO confirmation study is now historical reference material rather than a current authoritative baseline in a strong sense.
- For ongoing engineering workflow use, `uniform + N=3` is the pragmatic working default.
- `uniform + N=4` remains an unresolved challenger or exploratory candidate.
- Do not describe knot count as scientifically settled.

## Keep the control architecture explicit
- Outer layer: BO over `theta`.
- Inner layer: existing PID-tracked cryostage response represented through the reduced cryostage model.
- Solver input: modeled plate trajectory, not raw `T_ref`.

Do not document or imply that the optimizer directly controls the front in closed loop.

## Current status-defining studies
- Schedule sensitivity study: `results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`
  - `N=3` versus `N=4` ranking changed with the external knot-time schedule.
  - `uniform` is therefore the default schedule for ongoing comparisons.
- Uniform multiseed confirmation study: `results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`, `..._seed29/`, and `..._seed41/`
  - `N=3` is more stable across seeds.
  - `N=4` shows upside but higher optimizer sensitivity.
  - `N=4` does not yet demonstrate robust superiority.
  - Model order remains unresolved under `uniform`.

## Active entry points
- [run_open_loop_optimization.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_optimization.py): main BO runner for a single fixed knot count.
- [run_open_loop_schedule_sensitivity_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_schedule_sensitivity_study.py): current comparison runner for external schedule studies and uniform-schedule model-order checks.
- [run_reachability_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_reachability_diagnostics.py): active transient-plus-hold admissibility diagnostics.
- [export_final_locked_n3_workflow.py](/home/fenics/shared/Open_loop/code_simulation/export_final_locked_n3_workflow.py): exports the historical locked workflow bundle.

## Historical or contextual scripts
- [run_open_loop_fixed_n_bo_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_fixed_n_bo_study.py): earlier fixed-`N` BO comparison runner retained for historical context.
- [run_open_loop_study.py](/home/fenics/shared/Open_loop/code_simulation/run_open_loop_study.py) is a legacy exploratory study runner built around the `legacy_exploratory` formulation.
- [run_optimizer_learning_diagnostics.py](/home/fenics/shared/Open_loop/code_simulation/run_optimizer_learning_diagnostics.py) analyzes those older study outputs and should not be described as the current optimization path.

## Admissibility facts to preserve in docs
- Admissibility includes characterization-derived transient admissibility.
- Admissibility includes long-duration hold admissibility derived from freezing-run plate telemetry.
- Admissibility is used for early rejection before expensive simulation.
- Warming is not currently supported.

## Current caveats
- The main unresolved methodological question is whether the externally fixed knot-time schedule is too crude.
- The current workflow should distinguish carefully between the historical locked `N=3` reference, the pragmatic `uniform + N=3` working default, and the unresolved `uniform + N=4` challenger.
