# AGENTS.md — Open_loop project

## Scope
This repository maintains an **offline open-loop trajectory design workflow** for controlling freezing-front evolution in a cryostage system.
The modular workflow already exists, and future work should start from the **current implementation state**, not from the original implementation roadmap.
The exploratory **180 s / 240 s / 360 s** studies should be treated as completed scoping work.
The work must stay aligned with the current article-oriented objective: move the existing cascade toward **full freezing / total solidification** while preserving physically plausible cryostage trajectories and article-useful front-behaviour interpretation.

## Core control assumptions — do not violate
1. This is **open-loop trajectory control**, not real-time closed-loop front control.
2. The optimization objective is based on **front position tracking**, not direct raw front-velocity control.
3. The manipulated variable is the **cryostage reference temperature trajectory** `T_ref(t)`.
4. The cryostage already has an **inner PID**. Do not replace it.
5. The freezing simulation must receive the **actual plate or base trajectory** `T_plate(t)`, not `T_ref(t)` directly.
6. The intended cascade is:
   `theta -> T_ref(t) -> cryostage model -> T_plate(t) -> solver -> z_front(t) -> J(theta)`
7. Do not bypass the cryostage model.
8. Do not optimize against noisy instantaneous front velocity.

## Repository context
The workspace has two major areas:
- `code_simulation/`
  - active implementation area for the open-loop workflow
  - this is where future code edits should normally happen
  - the current modular pipeline and study runners already live here
- `data/`
  - experimental and reference material around the project
  - includes cryostage characterization results, firmware and material related to characterization, experimental freezing readings, calibrated simulation outputs, and comparison material against real cryostage and freezing results

Treat `code_simulation/` as the main implementation area.
Treat `data/` as reference input, validation material, and historical output. Do not casually rewrite it.

## Current defaults to assume unless a task says otherwise
- active code lives under `Open_loop/code_simulation`
- `T_ref_bounds_C = (-20.0, 0.0)`
- `require_monotone_nonincreasing = True`
- `T_fill_C = 12.5`
- `h_top = 2.0`
- `h_side = 2.0`

## Development philosophy
- Make the **minimum clean modifications necessary**.
- Preserve the existing architecture and naming style whenever possible.
- Keep backward compatibility with the current fixed-temperature workflow.
- Prefer **small modular additions** over large rewrites.
- Use **small prompts**, **small diffs**, and **separate chats** for separate questions whenever possible.
- Work phase by phase. Do not try to re-architect the whole workflow in one pass unless explicitly asked.
- If the codebase already partially supports a requested feature, extend it rather than re-implementing it.
- Do **not** refactor unless a real blocking issue is confirmed.

## Mandatory workflow for code tasks
For any nontrivial implementation task:
1. Inspect the relevant existing files first.
2. Confirm the current implementation state and identify only **real blockers**.
3. Summarize the minimum file or function changes required.
4. Implement only the requested phase.
5. Run or define at least one small smoke test when code changes are made.
6. Compare outputs critically instead of assuming a run is correct because it completed.
7. Summarize the diff clearly.

## Current implementation state
The modular open-loop workflow is already implemented.
The active pipeline is:
`theta -> T_ref(t) -> cryostage model -> T_plate(t) -> solver -> z_front(t) -> J(theta)`

There is already a study runner producing:
- run summaries
- evaluation history
- best-run artifacts
- standard plots for `T_ref(t)`, `T_plate(t)`, front tracking, and optimization history

Treat the exploratory **180 s / 240 s / 360 s** phase as complete.
The next article-facing phase is **not** another short-window exploratory study; it is a **single full-process open-loop optimization** that runs from fill through solver-detected freeze completion.

## Current technical priorities
1. Inspect the current implementation before proposing changes.
2. Identify only real blockers in the existing workflow.
3. Use the solver's existing **freeze-complete** logic for the next full-freezing phase instead of inventing a separate completion criterion.
4. Formulate the next control problem as **one full-process open-loop optimization**, not as more exploratory short-horizon studies.
5. Keep the objective based on **front-position behaviour** and reference-tracking-style interpretation, not direct raw front-velocity control.
6. Validate the cryostage reduced model only to the extent needed to support the full-freezing phase and article interpretation.
7. Do new architecture work only if a real blocking issue is confirmed.

## Guidance for current work

### Implementation checks first
- Start with the current entry points and active modules:
  - `run_open_loop_optimization.py`
  - `run_open_loop_study.py`
  - `run_optimizer_learning_diagnostics.py`
  - `open_loop_problem.py`
  - `open_loop_cascade.py`
  - `cryostage_model.py`
- Assume the pipeline exists unless inspection shows a real gap.
- Prefer inspecting current behaviour over proposing new abstractions.

### Cryostage reduced-model validation
- Treat reduced-model validation as an active priority, not as future architecture work.
- Compare measured and simulated `T_plate(t)` carefully.
- Check whether the simplest current model is adequate before proposing a more complex one.
- Use data-path and folder-name checks before concluding that the model itself is wrong.

### Completed exploratory phase
- Treat the exploratory **180 s / 240 s / 360 s** studies as completed background work.
- Use the saved study outputs and diagnostics for interpretation, not as a prompt to build another short-window study campaign.
- If an old exploratory result is revisited, do it only to extract a comparison or initialize the next full-process formulation.

### Article-oriented interpretation
- Focus on comparisons that help the article:
  - short-window exploratory results versus full-process behaviour
  - cryostage-model adequacy for the full-freezing phase
  - front-position behaviour up to total solidification
  - robustness of the final full-process optimum
- Prefer concise tables and plots with critical interpretation over large volumes of output.

## Known checks before the next phase
- Check the existing freeze-stop path before adding new logic. In `solver.py`, `FreezeStopOptions(mode="fillable_region")` already writes `freeze_complete_flag` and stops when the water-filled region is fully frozen.
- Check how `T_plate0_C` is initialized in the cascade. In `open_loop_cascade.run_open_loop_case()`, if `T_plate0_C` is left as `None`, it falls back to `bcs.T_room_C` when available, otherwise to `T_ref_C[0]`. Keep that initialization consistent when comparing legacy exploratory runs and future full-freezing runs.
- Check whether a requested change genuinely needs new orchestration. `run_open_loop_optimization.py` and `run_open_loop_study.py` are working baselines; extend them only if the full-process formulation truly requires it.

## File and naming preferences
- Reuse existing naming patterns where possible.
- Add new files only if clearly useful.
- Prefer the current implementation locations over creating parallel structures.
- Active files already include:
  - `cryostage_model.py`
  - `trajectory_profiles.py`
  - `open_loop_cascade.py`
  - `open_loop_problem.py`
  - `open_loop_optimizer.py`
  - `run_open_loop_optimization.py`
  - `run_open_loop_study.py`
  - `run_optimizer_learning_diagnostics.py`
- If the repository already has a better-fitting location or name, prefer that over creating parallel structures.

## Validation expectations
Every important edit should preserve the following:
1. The existing open-loop pipeline still runs end to end.
2. The existing fixed-temperature solver still works.
3. Existing calibration scripts such as `run_calibration_fixed_h.py` remain usable.
4. Study runs still produce summaries, history, best-run artifacts, and standard plots.
5. Front extraction and post-processing continue to work with the current path.

## Testing guidance
Always define or run at least one fast smoke test.
Good smoke tests include:
- a very short optimization or study run with tight iteration limits
- a comparison between two small runs that differ in one controlled setting
- a tiny cryostage model simulation against one characterization CSV
- a quick validation that the expected summary, history, and plot artifacts are produced

## Prohibited behaviour
- Do not redesign the entire project.
- Do not treat the workflow as if it still needs to be built from scratch.
- Do not introduce real-time control.
- Do not feed `T_ref` directly into the freezing solver.
- Do not replace the current front-tracking logic unless explicitly asked.
- Do not replace the solver formulation unless explicitly asked.
- Do not add unnecessary abstraction layers.
- Do not refactor unless a real blocking issue is confirmed.
- Do not optimize against direct front-velocity noise.
- Do not reopen the project as another round of exploratory short-window studies unless explicitly asked.

## Preferred output style when responding
When asked to implement code:
- be concrete
- show touched files
- explain only what is necessary
- summarize the diff clearly
Avoid long essays.
