# Results Index

## Current workflow status
- Current default external knot-time schedule: `uniform`
- Pragmatic working default for ongoing engineering workflow use: `uniform + N=3`
- Current unresolved challenger for ongoing workflow development: `uniform + N=4`
- Historical locked workflow reference: `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`

The locked `N=3` workflow remains reproducible historical reference material. It should no longer be described as the current authoritative baseline in a strong scientific sense, and it should not be used to claim that knot count is scientifically settled.

## Historical locked workflow reference bundle
- Bundle: `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`
- Locked source run at the time of export: `code_simulation/results/open_loop_bayesian_optimization/fixed_n_bo_confirmation_k3_k4_seed29_init4_iter8/runs/bo_compare_full_process_k3_seed29_init4_iter8/`
- Locked knot count at the time of export: `N = 3`
- Locked uniform knot times at the time of export: `(0.0, 1200.0, 2400.0) s`
- Locked knot temperatures at the time of export: `(-0.5, -15.0, -20.0) C`

## Main status-defining studies

### 1. External knot-time schedule sensitivity
- Folder: `code_simulation/results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`
- Purpose: compare `N=3` and `N=4` under the same BO settings while changing only the externally fixed knot-time schedule.
- Compared schedules: `uniform`, `early_dense`, `late_dense`
- Key outcomes:
  - `uniform`: `N=4` best objective `2.533405138e-01`, `N=3` best objective `2.538915394e-01`
  - `early_dense`: `N=3` best objective `2.588256647e-01`, `N=4` best objective `3.035365257e-01`
  - `late_dense`: `N=4` best objective `2.566860378e-01`, `N=3` best objective `2.711955936e-01`
- Decision consequence:
  - the `N=3` versus `N=4` ranking is schedule-dependent
  - `uniform` became the default external knot-time schedule for ongoing comparisons

### 2. Uniform multiseed confirmation
- Folders:
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed29/`
  - `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed41/`
- Purpose: compare `N=3` and `N=4` under the chosen default schedule `uniform` with multiple BO seeds.
- Shared settings: `init_points=6`, `n_iter=16`, seeds `17`, `29`, and `41`
- Key outcomes by seed:
  - seed `17`: `N=4` best objective `2.473153266e-01`, `N=3` best objective `2.545899717e-01`
  - seed `29`: `N=3` best objective `2.538915394e-01`, `N=4` best objective `2.729151682e-01`
  - seed `41`: `N=4` best objective `2.464709220e-01`, `N=3` best objective `2.538915394e-01`
- Aggregated outcomes:
  - `N=3` mean best objective: `2.541243501e-01`
  - `N=3` median best objective: `2.538915394e-01`
  - `N=4` mean best objective: `2.555671389e-01`
  - `N=4` median best objective: `2.473153266e-01`
- Decision consequence:
  - `N=3` is more stable across seeds
  - `N=4` shows upside but higher optimizer sensitivity
  - `N=4` does not yet demonstrate robust superiority
  - `uniform + N=3` is the pragmatic working default, while `uniform + N=4` remains an unresolved challenger

## Admissibility basis for the active workflow
- Transient admissibility artifacts: `code_simulation/results/characterization_constraints/stage1_reachability/`
- Long-duration hold support artifacts: `code_simulation/results/characterization_constraints/stage2_hold_telemetry/`
- Combined diagnostics: `code_simulation/results/characterization_constraints/admissibility_diagnostics/`

The active admissibility layer combines characterization-derived transient admissibility, long-duration hold admissibility from freezing-run plate telemetry, and early rejection before expensive simulation. Warming is not currently supported.

## Interpretation caveat
These results should be read as sensitivity analysis of externally fixed workflow design choices. The repository should not claim that knot count is scientifically settled, and the current default schedule `uniform` should not be over-interpreted as a globally optimal time discretization.
