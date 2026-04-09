# Next Steps Roadmap

## 1. Where the project is now

The project is no longer in a pipeline-building phase. The core BO-based full-process workflow exists, is evidence-backed enough to run, and already has a pragmatic working default.

The repository evidence points to this current state:

- the active workflow is the BO full-process workflow, not the older Nelder-Mead study branch
- the control cascade is explicit and should stay explicit
- admissibility is now evidence-backed by both cryostage transient characterization and long-duration hold telemetry
- `uniform + N=3` is the current practical working default
- `N=3` versus `N=4` is still scientifically unresolved

So the project is now in a decision phase: either freeze a practical default and move on, or ask one new narrow scientific question and collect only the evidence needed for that question.

## 2. What the project already has

The repository already has the pieces needed for serious downstream work:

- a reduced cryostage model and a validated cryostage-response layer
- a FEniCSx freezing solver with front tracking
- an explicit open-loop cascade from knot temperatures to `T_ref`, `T_plate`, and front evolution
- a current objective and evaluation path under the BO full-process workflow
- admissibility logic that rejects unsupported targets, unsupported warming, and infeasible transitions
- current BO studies for schedule sensitivity and multiseed `N=3` versus `N=4` confirmation
- historical result branches that preserve how the workflow evolved
- decision memos and summary CSVs that already synthesize much of the current status

In practical terms, the project does not need another general build-out. It needs a choice about what uncertainty is still worth reducing.

## 3. What still blocks scientific closure

Scientific closure is still blocked by a small number of real issues, not by missing infrastructure.

- The `N=3` versus `N=4` question is unresolved. The winner changes across schedule families, and even under `uniform` the winner changes across seeds.
- The current knot-count question may be partly confounded by fixed knot-time schedules. If the timing parameterization is too restrictive, more knot-count searching may answer the wrong question.
- The auxiliary full-freezing and runtime diagnostics are not centered on the current BO default. They still rely on a legacy early-drop profile source.
- The repo's documented working default and the runnable CLI defaults are not fully aligned. The docs say `uniform + N=3`, but the main runner still defaults to 5 knots.
- Warming remains outside the evidence base. That means part of the admissibility space is still scientifically unsupported.

## 4. What can be paused

Several lines of work can be paused without losing the current project center of gravity.

- More work on the older `open_loop_study` Nelder-Mead branch.
- More effort on the older `open_loop_optimization` branch as if it were the current baseline.
- More optimizer-budget tuning in the old short-horizon regime.
- More packaging or polishing of the locked historical `N=3` workflow as though it resolves the current knot-count question.
- More broad knot-count sweeps without first deciding whether the real unresolved question is knot count or schedule-time parameterization.
- More evidence-figure polishing for historical bundles that are no longer decision-critical.

These branches are still useful for history and provenance, but they are no longer the main decision surface.

## 5. Immediate next steps

These are the next steps that have the highest decision value right now.

### A. Freeze one canonical current default

Pick one explicit "carry-forward" default and make it the single operational reference.

That default should be:

- BO full-process workflow
- explicit cascade preserved
- admissibility on
- `uniform` schedule family
- `N=3` as the pragmatic default
- no warming moves

This is not the same as claiming scientific optimality. It is choosing one working baseline so the project stops drifting between historical and current configurations.

### B. Align docs, scripts, and saved diagnostics around that default

Before any new science run, remove ambiguity about what "current default" means.

Do this by:

- making one named current-default bundle the reference point in project communication
- updating the runnable default configuration so it does not silently point to `N=5`
- regenerating the auxiliary full-freezing and runtime snapshots using the actual BO default rather than the legacy early-drop profile

This step matters because otherwise the repo will keep mixing current scientific conclusions with older operational artifacts.

### C. Decide whether the project wants practical closure or scientific closure

Make this decision explicitly.

- If the goal is practical closure: stop optimization now, standardize the working default, and move to downstream validation, reporting, or use.
- If the goal is scientific closure: authorize one tightly scoped study aimed only at the unresolved question.

Without this decision, more optimization work will likely produce more folders rather than more clarity.

## 6. Medium-term next steps

The medium-term roadmap depends on which question is worth paying for.

### Path 1: Practical closure

If the project is already good enough operationally:

- keep `uniform + N=3` as the working default
- run downstream checks, analyses, and reporting from that baseline
- treat the knot-count question as explicitly unresolved but not decision-critical
- preserve the current evidence memos as the justification for stopping

### Path 2: Scientific closure on knot count

If the project wants a stronger scientific statement about `N=3` versus `N=4`, the next study should be narrow and precommitted.

Recommended design:

- compare only `uniform, N=3` versus `uniform, N=4`
- hold BO budget policy fixed in advance
- hold seed policy fixed in advance
- predefine the stopping rule and interpretation rule before running
- judge both objective performance and stability, not just best single-run objective

The purpose of this study should be to answer one narrow question, not to reopen broad exploration.

### Path 3: Reframe the real question

If the team believes the fixed knot-time schedules are the real bottleneck, then the next scientific move should not be another wide `N` sweep.

Instead, the next move should be to test whether timing parameterization, not knot count, is the dominant unresolved simplification.

### Path 4: Expand the admissibility envelope

Only do this if there is a concrete scientific or operational need.

If warming or more complex transitions matter, collect the missing characterization first. Do not optimize into unsupported regions and then treat that as evidence.

## 7. Recommended default working configuration to carry forward

This is the most defensible working configuration to carry forward now.

- Workflow: BO full-process workflow
- Problem framing: keep the explicit `theta -> T_ref -> cryostage model / inner PID response -> T_plate -> freezing solver -> z_front -> objective` cascade
- Admissibility: on, using the current transient-plus-hold evidence base
- Schedule family: `uniform`
- Knot count: `N=3`
- Interpretation: pragmatic working default only, not a proof of optimality
- Motion restrictions: no warming moves unless new characterization supports them
- Operational stance: use the current lean full-process code configuration as the engineering baseline, but do not elevate every code default to a separate scientific conclusion

In short: carry forward the current BO workflow, keep the admissibility guardrails, use `uniform + N=3`, and stop calling that choice "settled science."

## 8. Risks of continuing optimization without a new question

Continuing optimization without a fresh, explicit question has real costs.

- It can create more apparent evidence without reducing the actual decision uncertainty.
- It can blur the difference between historical artifacts and current workflow evidence.
- It can overinterpret stochastic wins from one seed or one schedule family.
- It can spend effort refining knot count while the real limitation may be the fixed timing parameterization.
- It can encourage accidental drift away from the current admissibility-supported regime.
- It can make the repository harder to read by producing more result branches that do not change the recommendation.

The practical risk is not just wasted compute. It is decision dilution.

## 9. Short supervisor update in Portuguese

O projeto ja tem a infraestrutura principal pronta: modelo reduzido do cryostage, solver de congelamento com front tracking, filtro de admissibilidade e workflow de otimizacao por Bayesian Optimization. Hoje, a melhor leitura do repositorio e que `uniform + N=3` deve ser mantido como configuracao de trabalho pragmatica, mas a questao cientifica `N=3` versus `N=4` ainda nao esta fechada. Se o objetivo for pratico, eu recomendo parar a otimizacao agora, padronizar a configuracao atual e seguir para validacao e consolidacao. Se o objetivo for cientifico, o proximo passo deve ser um unico estudo curto e predefinido para decidir a questao remanescente, e nao uma nova rodada ampla de exploracao.
