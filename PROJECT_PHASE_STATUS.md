# Open_loop Project Phase Status

## Current Phase

The project is in a late integration and decision-narrowing phase, not an early build phase.

Core infrastructure already exists:

- a reduced cryostage model
- a FEniCSx freezing solver with front tracking
- an explicit open-loop cascade from knot temperatures to `T_ref`, `T_plate`, and freezing response
- an admissibility gate derived from characterization and freezing-hold evidence
- a BO workflow for optimizing knot temperatures under fixed knot-time schedules
- study runners and evidence packages for comparing schedule families, knot counts, and optimizer settings

The current question is therefore mostly scientific and practical, not architectural: what is already settled enough to use, and what still needs evidence before claiming a stronger conclusion.

## What Has Already Been Built

- Current active workflow code under `code_simulation/` for BO-based full-process optimization.
- Current admissibility implementation that combines transient reachability with empirical long-duration hold support.
- Study tooling for:
  - fixed-`N` BO comparisons
  - schedule-sensitivity BO comparisons
  - earlier Nelder-Mead sensitivity and optimizer-learning studies
  - full-freezing diagnostics
  - runtime-vs-accuracy diagnostics
- Root-level decision documents that already synthesize much of the current status.
- Supporting experimental data, calibrated simulation cases, and evidence-figure exports.

## What The Completed Studies Actually Establish

### Admissibility and physical support

- Stage-1 cryostage characterization establishes a conservative finite-window cooling envelope and settling behavior for downward target steps.
- Stage-2 freezing telemetry establishes that long holds at `-10 C`, `-15 C`, and `-20 C` have empirical support for bounded durations.
- The current admissibility rule is therefore evidence-backed for conservative downward cooling plus supported holds.
- Warming is not supported by the present characterization basis. That is a real limitation, not just a missing convenience feature.

### Cryostage-model adequacy

- The reduced cryostage model validation bundle indicates the model is serviceable enough to remain in the optimization loop.
- The validation evidence supports continued practical use of the reduced model.
- The validation bundle does not, by itself, settle the best knot count or schedule family.

### Historical optimization branch

- The earlier `open_loop_study` branch establishes that open-loop optimization is tractable and that early-drop style schedules were more promising than the older delayed-drop or vanilla forms within that earlier objective/workflow regime.
- The optimizer-learning and budget-sensitivity studies establish diminishing returns after moderate Nelder-Mead budget increases in that older regime.
- Those studies are historically useful, but they are not the current scientific decision basis for the BO-era workflow.

### Current BO-era evidence

- The BO schedule-sensitivity study establishes that the apparent winner between `N=3` and `N=4` changes with schedule family.
- The multiseed uniform confirmation runs establish that even under the same `uniform` schedule, winner identity is not stable across seeds.
- The current evidence therefore supports a conservative conclusion:
  - `N=4` has upside in some cases
  - `N=3` is often more stable and is the documented practical default
  - the repository does not prove that either `N=3` or `N=4` is definitively optimal

## What Is Settled

- The project should be interpreted through the BO-based full-process workflow, not through the older short-horizon Nelder-Mead studies.
- The optimization variables are knot temperatures, while knot times are currently externally fixed.
- The explicit cascade should remain explicit:
  - `theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> objective`
- Admissibility should remain active and should continue to combine transient and hold-support evidence.
- Warming should not be treated as supported.
- `uniform` is a valid pragmatic schedule family for the current workflow and is the current comparison baseline.

## What Is Only A Pragmatic Working Default

- `uniform + N=3` is the current pragmatic working default.
- That default is documented consistently in the current memos, but it is not scientifically settled.
- The codebase is not perfectly synchronized with that documented default:
  - `run_open_loop_optimization.py` still defaults to `num_knots = 5`
  - some auxiliary diagnostics still point to a legacy early-drop profile rather than the BO default
- Because of those mismatches, "current default" should be interpreted as a documented operational choice, not as a universally encoded repo-wide setting.

## What Remains Scientifically Unresolved

- Whether `N=3` or `N=4` is actually preferable under the current BO workflow.
- Whether the current externally fixed knot-time schedules are too crude, such that knot-count comparisons are partly confounded by schedule design.
- Whether broader schedule families beyond the currently emphasized `uniform`, `early_dense`, and `late_dense` choices matter materially.
- Whether additional evidence would change the balance between stability and upside enough to justify moving away from `uniform + N=3`.
- Whether future scientific objectives should continue to optimize only knot temperatures, or should revisit the knot-time parameterization itself.

## Whether Optimization Is In A Good Stopping State For Now

Practically: yes, probably.

- If the near-term goal is to have a conservative, documented, and reproducible working default for downstream validation or scientific use, the repository appears to be in a reasonable stopping state with `uniform + N=3`.
- That stopping point is especially defensible because the current evidence does not robustly prove `N=4` superiority.

Scientifically: not fully.

- The repository does not provide a clean final scientific closure on knot count.
- The remaining uncertainty is not just noise around a settled answer; it is structured uncertainty tied to schedule family and seed dependence.

The best interpretation is therefore:

- optimization is in a good temporary stopping state for practical work
- optimization is not in a final stopping state for a stronger scientific claim about the best knot count

## Genuinely Necessary Next Steps

### Practical next steps

- Align the operational default across docs, code entry points, and diagnostic scripts.
- Refresh auxiliary diagnostics using the actual current BO default rather than the legacy early-drop profile.
- Keep the current admissibility policy explicit and documented as a hard boundary.

### Scientific next steps

- If more optimization work is justified, run a narrowly scoped adjudication study rather than a broad new exploration.
- The most defensible next unresolved question is either:
  - `uniform, N=3` versus `uniform, N=4` under a clearly fixed BO budget and seed policy
  - whether fixed knot-time schedules should remain fixed before spending more effort on knot-count claims
- If warming matters scientifically or operationally, collect warming characterization first. The current repository does not support that behavior.

## What Should Not Be Done Next

- Do not claim that `N=3` is definitively optimal.
- Do not claim that `N=4` is definitively optimal.
- Do not use the locked `N=3` workflow package as proof that the knot-count question is settled.
- Do not treat the older `open_loop_study` early-drop results as the current workflow baseline.
- Do not treat auxiliary runtime or full-freezing diagnostics that use the legacy early-drop profile as direct evidence about the current BO default.
- Do not launch another wide knot-count sweep before deciding whether the fixed knot-time parameterization is itself the limiting simplification.
- Do not extrapolate admissibility beyond the characterized downward-cooling and supported-hold regime.

## Bottom Line

The repository is not in a "build the pipeline" phase anymore. It is in a "choose how much uncertainty you are willing to carry" phase.

The conservative reading is:

- the pipeline is built
- the current BO workflow is usable
- `uniform + N=3` is a reasonable practical default
- the stronger scientific claim about the best knot count is still unresolved

That means the next move should be either:

- stop optimization for now and use the current practical default consistently
- or run one tightly targeted study aimed only at the unresolved decision, not another broad exploratory campaign
