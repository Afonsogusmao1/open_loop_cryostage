# AGENTS.md — Open_loop project

## Scope
This repository develops an **offline open-loop trajectory design framework** for controlling freezing-front evolution in a cryostage system.
The work must stay aligned with the article-oriented objective: generate physically plausible cryostage temperature trajectories and quantify their effect on freezing-front evolution.

## Core control assumptions — do not violate
1. This is **open-loop trajectory control**, not real-time closed-loop front control.
2. The optimization objective is based on **front position tracking**, not direct raw front-velocity control.
3. The manipulated variable is the **cryostage reference temperature trajectory** `T_ref(t)`.
4. The cryostage already has an **inner PID**. Do not replace it.
5. The freezing simulation must receive the **actual plate/base trajectory** `T_plate(t)`, not `T_ref(t)` directly.
6. The intended cascade is:
   `theta -> T_ref(t) -> cryostage model -> T_plate(t) -> 2D freezing model -> z_front(t) -> J(theta)`
7. Do not bypass the cryostage model.
8. Do not optimize against noisy instantaneous front velocity.

## Repository context
The workspace is organized as:
- `data/characterization_cryostage`
  - repeated cryostage runs at setpoints such as `-5`, `-10`, `-15`, `-20 °C`
  - used to identify a reduced closed-loop cryostage model mapping `T_ref(t) -> T_plate(t)`
- `data/constant_plateT_water_ICT_readings`
  - experimental water freezing runs with thermocouple data
  - used for calibration and validation of the 2D thermal model
- `data/simulations_calibrated`
  - baseline calibrated 2D simulation outputs
  - includes probe CSVs, front tracking CSVs, curved front CSVs, and ParaView-compatible files
- `code_simulation`
  - active development area for the simulation code
  - typical key files: `solver.py`, `geometry.py`, `materials.py`, `front_tracking.py`

Treat `data/` as reference input and baseline outputs.
Treat `code_simulation/` as the main implementation area.

## Development philosophy
- Make the **minimum clean modifications necessary**.
- Preserve the existing architecture and naming style whenever possible.
- Keep backward compatibility with the current fixed-temperature workflow.
- Prefer **small modular additions** over large rewrites.
- Work phase-by-phase; do not try to implement the whole framework in one pass unless explicitly asked.
- If the codebase already partially supports a requested feature, extend it rather than re-implementing it.

## Mandatory workflow for code tasks
For any nontrivial implementation task:
1. Inspect the relevant existing files first.
2. Summarize the minimum file/function changes required.
3. Implement only the requested phase.
4. Run or define at least one small smoke test.
5. Summarize the diff clearly.

## Current technical priorities
The expected implementation order is:
1. Review/support time-varying plate temperature in the 2D freezing solver.
2. Implement trajectory parameterization classes for `T_ref(t)`.
3. Identify and simulate a low-order cryostage closed-loop model from characterization CSVs.
4. Build the front-position-based objective function.
5. Define a clear problem-setup layer for simulation, optimization, and trajectory settings.
6. Implement the optimization driver.
7. Add output plots/CSVs only after the optimization path is working.

## Guidance for each phase

### Phase 1 — Time-varying plate temperature in the 2D solver
- Preserve the existing fixed `T_plate_C` path.
- Add backward-compatible support for a time-varying `T_plate(t)` profile.
- Prefer extending `run_case(...)` rather than replacing it.
- Use a small helper such as `evaluate_plate_temperature(t, profile)` or a small profile dataclass.
- Do not rewrite the PDE setup from scratch.

### Phase 2 — Trajectory parameterization
- Prefer **classes** over loose functions when reasonable.
- Start with a monotone piecewise-linear cooling trajectory.
- Also support simple testing trajectories such as:
  - constant step,
  - ramp,
  - parabola.
- Include temperature bounds.
- Allow optional slope/ramp-rate bounds.
- Keep the design robust and physically plausible.

### Phase 3 — Cryostage model from characterization CSVs
- This should be a **separate module** from the freezing solver.
- Start with the simplest acceptable low-order model:
  - first-order,
  - optionally first-order plus dead time if clearly needed.
- Prefer a global model over gain scheduling unless the data clearly demand otherwise.
- Deliver:
  - loader/parser,
  - fitted parameter object,
  - forward simulator,
  - measured-vs-fitted diagnostic plots.
- Also provide a small script or notebook to visualize the response to test trajectories such as step, ramp, parabola, or piecewise linear.

### Phase 4 — Objective function
- Use **front position tracking** against a reference `z_ref(t)`.
- Do not optimize raw instantaneous velocity directly.
- The default main term should be front-position tracking error.
- Penalties must have configurable weights.
- Include the option to ignore the first `x` seconds of the trajectory in the cost.
- Optional penalties may include:
  - excessive `dT_ref/dt`,
  - bound violations,
  - front non-monotonicity or other pathological behaviour.

### Phase 5 — Problem setup and optimization driver
Before optimizing, define the full problem setup explicitly:
- freezing simulation settings,
- cryostage simulation settings,
- target freeze time or target front progression,
- cost weights,
- initial time to discard,
- chosen trajectory family,
- optimizer settings.

Only after that, implement the driver.
Start with a **simple derivative-free optimizer**, preferably `scipy.optimize.minimize` with **Nelder-Mead** unless there is a strong reason to use something else.

### Phase 6 — Outputs
Outputs are secondary to correctness.
Once the optimization loop works, add:
- plots of `T_ref(t)`, `T_plate(t)`, `z_front(t)`, `z_ref(t)`, and optionally `v_front(t)`;
- CSV exports for optimized trajectories and front outputs;
- a short summary of fitted cryostage parameters, optimized decision variables, and final cost.

## File and naming preferences
- Reuse existing naming patterns where possible.
- Add new files only if clearly useful.
- Reasonable additions may include:
  - `cryostage_identification.py`
  - `cryostage_model.py`
  - `plate_temperature_profile.py`
  - `trajectory_profiles.py`
  - `open_loop_objective.py`
  - `open_loop_problem.py`
  - `optimize_open_loop.py`
- If the repository already has a better-fitting location or name, prefer that over creating parallel structures.

## Validation expectations
Every important edit should preserve the following:
1. The existing fixed-temperature solver still works.
2. Existing calibration scripts such as `run_calibration_fixed_h.py` remain usable.
3. Front extraction continues to work with the new path.
4. Time-varying plate temperatures can be injected without breaking the current baseline path.

## Testing guidance
Always define or run at least one fast smoke test.
Good smoke tests include:
- a very short simulation with scalar constant plate temperature,
- the same short simulation with an equivalent constant profile object,
- a simple step or ramp plate profile to verify boundary updating,
- a tiny cryostage model simulation against one characterization CSV.

## Prohibited behaviour
- Do not redesign the entire project.
- Do not introduce real-time control.
- Do not feed `T_ref` directly into the freezing solver.
- Do not replace the current front-tracking logic unless explicitly asked.
- Do not replace the solver formulation unless explicitly asked.
- Do not add unnecessary abstraction layers.
- Do not optimize against direct front-velocity noise.

## Preferred output style when responding
When asked to implement code:
- be concrete,
- show touched files,
- explain only what is necessary,
- summarize the diff clearly.
Avoid long essays.

