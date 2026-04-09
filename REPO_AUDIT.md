# Open_loop Repository Audit

## Scope

This audit targets `/home/fenics/shared/Open_loop`, which is the nested `Open_loop` repository inside the current workspace root.

The repository is too large to read every artifact line by line, so this audit prioritized:

- current root decision documents and workflow memos
- active workflow code under `code_simulation/`
- major result bundles under `code_simulation/results/`
- supporting experimental and calibration assets under `data/`
- evidence-package figures and summary CSVs used in current status documents

## Top-Level Structure

- `README.md`: current high-level project description and current result interpretation.
- `README.txt`: older narrative document from an earlier transition point; useful historical context, but not the main current authority.
- `AGENTS.md`: repository-specific operating notes.
- `RESULTS_INDEX.md`: curated index of the main result bundles and how they should be interpreted.
- `DECISION_STATUS_MEMO.md`: current decision memo on workflow status and what is settled versus unsettled.
- `OPTIMIZATION_EVIDENCE_MEMO.md`: current knot-count evidence memo.
- `optimization_evidence_summary.csv`: compact summary table for the current knot-count evidence memo.
- `optimization_evidence_figures/`: exported figures used to communicate the current BO evidence package.
- `code_simulation/`: main source-code area for the freezing model, cryostage model, optimization workflow, admissibility rules, and study runners.
- `data/`: experimental data, characterization data, calibrated simulation cases, and related plotting or device-control code.

## Important Subfolders Inspected

- `code_simulation/`: active source code and workflow scripts.
- `code_simulation/results/characterization_constraints/`: admissibility evidence from cryostage transients and long-duration freezing holds.
- `code_simulation/results/cryostage_model_validation/`: reduced cryostage model validation artifacts.
- `code_simulation/results/open_loop_bayesian_optimization/`: current BO-era comparison studies and confirmation studies.
- `code_simulation/results/open_loop_final_workflow/`: exported locked historical reference workflow.
- `code_simulation/results/open_loop_optimization/`: older full-process optimization outputs.
- `code_simulation/results/open_loop_study/`: earlier Nelder-Mead study era, including 180 s / 360 s / 600 s workflows and optimizer-learning studies.
- `code_simulation/results/full_freezing_diagnostics/`: auxiliary full-freezing checks.
- `code_simulation/results/full_freezing_runtime_compare/`: auxiliary runtime-vs-accuracy checks.
- `data/characterization_cryostage/`: raw cryostage characterization data, analysis code, figures, and embedded/desktop characterization tooling.
- `data/constant_plateT_water_ICT_readings/`: raw freezing-run telemetry used for long hold support evidence.
- `data/simulations_calibrated/`: calibrated simulation cases and experiment-vs-simulation comparison assets.

## Folder Roles

- Source code lives mainly in `code_simulation/` and also in `data/characterization_cryostage/cryostage_characterization/` for characterization acquisition/control tooling.
- Workflow scripts live mainly in `code_simulation/run_open_loop_optimization.py`, `code_simulation/run_open_loop_schedule_sensitivity_study.py`, `code_simulation/run_open_loop_fixed_n_bo_study.py`, `code_simulation/run_full_freezing_diagnostics.py`, `code_simulation/run_full_freezing_runtime_compare.py`, and `code_simulation/export_final_locked_n3_workflow.py`.
- Study runners live mainly in `code_simulation/run_open_loop_study.py`, `code_simulation/run_optimizer_learning_diagnostics.py`, `code_simulation/run_reachability_diagnostics.py`, `code_simulation/analyze_step_response_reachability.py`, and `code_simulation/analyze_freezing_hold_telemetry.py`.
- Optimization outputs live mainly in `code_simulation/results/open_loop_bayesian_optimization/`, `code_simulation/results/open_loop_optimization/`, `code_simulation/results/open_loop_study/`, and `code_simulation/results/open_loop_final_workflow/`.
- Decision and status memos live mainly in `RESULTS_INDEX.md`, `DECISION_STATUS_MEMO.md`, `OPTIMIZATION_EVIDENCE_MEMO.md`, and `code_simulation/CURRENT_WORKFLOW.md`.
- Summary CSVs include `optimization_evidence_summary.csv`, many `study_summary.csv` files under `code_simulation/results/`, `full_freezing_summary.csv`, `runtime_settings_summary.csv`, and older `evaluation_history.csv` files.
- Plots and evidence packages live mainly in `optimization_evidence_figures/`, `data/characterization_cryostage/figures/`, `data/simulations_calibrated/figures/`, and figure/front-tracking outputs inside result bundles.

## Key Documents Inspected

- `README.md`: current high-level overview of the BO-era workflow and status.
- `AGENTS.md`: repo-local operating instructions.
- `code_simulation/AGENTS.md`: code-area specific working notes.
- `code_simulation/CURRENT_WORKFLOW.md`: strongest single document describing the active workflow, current defaults, and unresolved decisions.
- `RESULTS_INDEX.md`: maps result bundles to their intended interpretation.
- `DECISION_STATUS_MEMO.md`: current project decision state.
- `OPTIMIZATION_EVIDENCE_MEMO.md`: current evidence memo on `N=3` versus `N=4`.
- `optimization_evidence_summary.csv`: compact current evidence table used by the memo.
- `README.txt`: historical context document from an earlier phase.

## Key Code Files Inspected

- `code_simulation/run_open_loop_optimization.py`: main active orchestration entry point. Builds the full-process configuration, chooses BO as the default method, defines the externally fixed knot-time schedules, and wires the admissibility-aware optimization workflow together.
- `code_simulation/open_loop_problem.py`: current objective definition and simulation-evaluation path. This is where the reference profile is built from knot temperatures, the cascade is executed, the PDE solver is called, and the final objective components are assembled.
- `code_simulation/open_loop_optimizer.py`: optimization-loop wrapper, feasibility handling, early rejection, penalties, and evaluation logging.
- `code_simulation/open_loop_bayesian_optimizer.py`: BO backend configuration, seed handling, and best-feasible-point selection.
- `code_simulation/open_loop_cascade.py`: explicit `theta -> T_ref -> cryostage model -> T_plate -> freezing solver` cascade logic.
- `code_simulation/reachability_constraints.py`: current admissibility policy. Combines transient reachability and long-duration empirical hold support, rejects unsupported warming, and enforces conservative reachable target logic.
- `code_simulation/trajectory_profiles.py`: reference temperature profile primitives.
- `code_simulation/cryostage_model.py`: reduced cryostage model used in the cascade.
- `code_simulation/solver.py`: FEniCSx freezing solver and front-tracking machinery.
- `code_simulation/run_open_loop_schedule_sensitivity_study.py`: current BO study runner for comparing fixed schedule families and `N`.
- `code_simulation/run_open_loop_fixed_n_bo_study.py`: older BO fixed-`N` comparison runner.
- `code_simulation/run_open_loop_study.py`: earlier Nelder-Mead study runner from the older workflow branch.
- `code_simulation/run_optimizer_learning_diagnostics.py`: older optimizer-budget and optimizer-learning comparison tooling.
- `code_simulation/run_reachability_diagnostics.py`: admissibility diagnostic workflow.
- `code_simulation/analyze_step_response_reachability.py`: stage-1 transient reachability analysis.
- `code_simulation/analyze_freezing_hold_telemetry.py`: stage-2 long-hold support analysis.
- `code_simulation/run_full_freezing_diagnostics.py`: auxiliary full-freezing comparison script.
- `code_simulation/run_full_freezing_runtime_compare.py`: auxiliary runtime-vs-accuracy comparison script.
- `code_simulation/export_final_locked_n3_workflow.py`: exporter for the historical locked `N=3` workflow package.

## Key Result Folders Inspected

- `code_simulation/results/characterization_constraints/stage1_reachability/`: conservative transient cooling envelope derived from cryostage characterization runs.
- `code_simulation/results/characterization_constraints/stage2_hold_telemetry/`: empirical support for long hold durations at `-10 C`, `-15 C`, and `-20 C`.
- `code_simulation/results/characterization_constraints/admissibility_diagnostics/`: older and revised admissibility summaries. The revised summary matches the current two-regime transient-plus-hold logic.
- `code_simulation/results/cryostage_model_validation/validation_20260329/`: reduced cryostage model validation against characterization runs.
- `code_simulation/results/open_loop_bayesian_optimization/schedule_sensitivity_first_pass_n3_n4/`: main current comparison study showing schedule-family dependence and no robust `N=3`/`N=4` winner.
- `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed17/`: uniform-schedule confirmation run, seed 17.
- `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed29/`: uniform-schedule confirmation run, seed 29.
- `code_simulation/results/open_loop_bayesian_optimization/uniform_confirmation_n3_n4_multiseed_seed41/`: uniform-schedule confirmation run, seed 41.
- `code_simulation/results/open_loop_bayesian_optimization/fixed_n_bo_comparison_full_process_seed17_init4_iter8/`: earlier BO comparison across several `N` values; historically important but superseded by later confirmation and schedule-sensitivity evidence.
- `code_simulation/results/open_loop_bayesian_optimization/fixed_n_bo_confirmation_k3_k4_seed29_init4_iter8/`: historical BO confirmation run that produced the locked `N=3` reference workflow.
- `code_simulation/results/open_loop_final_workflow/locked_n3_seed29_init4_iter8/`: reproducible historical reference package, explicitly not proof that `N=3` is scientifically settled.
- `code_simulation/results/open_loop_study/`: older Nelder-Mead study branch for short-horizon and early-drop versus delayed-drop exploration.
- `code_simulation/results/open_loop_optimization/`: older full-process optimization outputs predating the current BO-centered status.
- `code_simulation/results/full_freezing_diagnostics/current_snapshot/`: auxiliary full-freezing checks, but based on a legacy early-drop profile rather than the current BO default.
- `code_simulation/results/full_freezing_runtime_compare/current_snapshot/`: auxiliary runtime study, also based on a legacy early-drop profile and an earlier default snapshot.

## Practical Interpretation Of The Repository State

- The repository contains far more than CSV outputs. It includes current workflow code, historical workflow code, study scripts, decision memos, evidence summaries, calibrated simulation cases, raw characterization datasets, and even embedded/desktop tooling for cryostage characterization.
- The current project center of gravity is the BO-based full-process workflow under `code_simulation/`, together with the current decision documents at the repository root.
- The current scientific decision point is not "how to make the pipeline run" but "how much more optimization evidence is worth collecting before declaring the present workflow good enough."
- Historical bundles remain important because the repository preserves older optimization eras rather than deleting them. They should be treated as context and evidence of workflow development, not automatically as the current baseline.

## High-Value Current-State Findings

- The strongest current workflow authority is the combination of `code_simulation/CURRENT_WORKFLOW.md`, `DECISION_STATUS_MEMO.md`, `OPTIMIZATION_EVIDENCE_MEMO.md`, and `optimization_evidence_summary.csv`.
- Those current documents consistently describe the active formulation as the full-process BO workflow, the control cascade as explicit, admissibility as based on transient plus long-duration hold evidence, the pragmatic working default as `uniform + N=3`, and the scientific status as unresolved between `N=3` and `N=4`.
- The code largely matches that story, but not perfectly. The active main runner defaults to BO and the full-process formulation, the schedule families are implemented, and admissibility plus early rejection are implemented, but the CLI default knot count in `run_open_loop_optimization.py` still points to `5` and some auxiliary diagnostics still benchmark a legacy early-drop profile rather than the current BO default.

## Audit Boundary

- I inspected the repository recursively with emphasis on active code, summary artifacts, and current decision documents.
- I did not re-run simulations or regenerate studies.
- I did not inspect every per-evaluation artifact inside every result bundle; instead, I prioritized the summary documents, study CSVs, and code that define how those bundles are produced and interpreted.
