# Current Workflow Status

## Active optimization path

The active open-loop workflow is velocity-control BO for a user-specified constant freezing-front speed.

- Main single-run entry point: `python -m code_simulation.optimization.run_velocity_control_bo`
- Effective-config inspection: `python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config`
- Active `n8` campaign summarizer and queue builder: `python -m code_simulation.studies.run_n8_uniform_from_n3 --dry-run`
- Current Phase 5 comparison-study entry point: `python -m code_simulation.studies.run_open_loop_phase5_binary_parameterization_study`
- External schedule-sensitivity support entry point: `python -m code_simulation.studies.run_open_loop_schedule_sensitivity_study`
- Admissibility diagnostics entry point: `python -m code_simulation.verification.run_reachability_diagnostics`
- Full-freezing diagnostics can also read the simulation profiles: `python -m code_simulation.verification.run_full_freezing_diagnostics --dry-run-config`
- Front-speed reachability planning/runner: `python -m code_simulation.verification.run_front_speed_reachability_study --dry-run-config`
- Manual velocity-control confirmation: `python -m code_simulation.verification.run_velocity_control_evaluation --dry-run-config`
- N3 target/spacing diagnostico summarizer: `python -m code_simulation.studies.run_bo_3knot_target_and_spacing_study --dry-run`
- Archived 4-knot uniform full-range study summarizer: `python -m code_simulation.studies.run_bo_4knot_uniform_full_range_study --dry-run`

The workflow objective remains model-based open-loop design of a plate-temperature reference trajectory for the inner PID, now targeting a configurable constant front speed over `2.5-11.5 mm`.

The currently locked production campaign is:

- `n8` temperature knots
- `uniform` knot schedule
- seeds `17, 29, 41, 53, 67`
- prior and per-knot bounds derived by monotone interpolation of the defended `n3` uniform anchor at `0.008 mm/s`
- new outputs written under `results/active/n8/...`

The outer design variables are trajectory temperatures `theta_C`. The current active workflow uses externally fixed knot times for the reference trajectory.

Editable run settings live in `code_simulation/configs`:

- `simulation_profiles.toml` contains the coarse `optimization` profile and the fine `full_process_article` profile.
- `velocity_control.toml` contains the target front speed, the `2.5-11.5 mm` direct-objective interval, initial water/base temperatures, and the no-hold protocol.
- `velocity_control.toml` also contains the active characterization temperature margin, currently `+-0.5 C`, used only when comparing requested temperatures against characterization-supported targets and support bands.
- `bo.toml` contains the active knot count, fixed knot-time schedule, `theta0_C`, BO budget, seed, acquisition settings, and objective penalty.
- The ambient air temperature is currently interpolated from the modeled plate/cryostage temperature in both profiles, using the measured `(-20 C, 5.75 C)`, `(-15 C, 7.568 C)`, and `(-10 C, 9.7078 C)` calibration points with clamp outside that range.

The practical distinction is:

- `optimization`: coarse mesh `Nr=36`, `Nz=72`, `dt=4.0 s`, `Nz_front=200`, lean outputs.
- `full_process_article`: fine mesh `Nr=180`, `Nz=408`, `dt=0.25 s`, `Nz_front=800`, fuller outputs.

The control architecture remains:

`theta -> T_ref(t) -> cryostage model / inner PID response -> T_plate(t) -> freezing solver -> z_front(t) -> J(theta)`

The inner PID tracking layer is distinct from the outer BO-based reference-trajectory design layer and should remain documented that way.

## Current velocity-control step

The manual velocity-control runner evaluates a user-defined `T_ref(t)` trajectory against the same constant target front-speed objective used by the BO runner. The active protocol starts with water in the mold at about `12.5 C`, the plate/base at `2.5 C`, and no warm hold before cooling. The direct tracking interval is `2.5-11.5 mm`, which adds a `0.5 mm` margin around the thermocouple span `3.0-11.0 mm`.

The manual runner is intentionally not an optimizer. It is a validation layer for the target-speed reference, no-hold protocol, outputs, and diagnostics used to confirm BO results.

## Active workflow interpretation

The architecture remains a trajectory-design workflow rather than a permanently fixed knot-count ideology, but the current execution campaign is intentionally locked to `n8` uniform-from-`n3` for comparability and throughput.

The present scientific focus is:

- keeping the reorganized codebase aligned with the active workflow,
- running the locked `n8` coarse matrix over the admissible target range,
- generating `n8` diagnostic bundles and selecting fine confirmations,
- using the temperature-dependent tabulated water/ice baseline consistently in both coarse and fine stages.

## Active admissibility behavior

- Admissibility includes characterization-derived transient admissibility.
- Admissibility includes long-duration hold admissibility derived from freezing-run plate telemetry.
- Characterization matching uses the explicit temperature-comparison margin from `velocity_control.toml`, without widening the hard physical bounds `[-21.0, 0.0] C`.
- Admissibility is checked before expensive simulation through the active reference-profile validation path.
- Inadmissible candidates are rejected early, logged as infeasible, and assigned the configured deterministic penalty objective without launching the expensive cascade.
- Warming is not currently supported and must not be silently assumed.

## Active versus historical study support

- Active: velocity-control BO in `code_simulation.optimization.run_velocity_control_bo`
- Active campaign bundle builder: `code_simulation.studies.run_n8_uniform_from_n3`
- Active Phase 5 binary parameterization comparison: `code_simulation.studies.run_open_loop_phase5_binary_parameterization_study`
- Controlled external schedule-sensitivity support: `code_simulation.studies.run_open_loop_schedule_sensitivity_study`
- Historical BO support: `code_simulation.studies.run_open_loop_fixed_n_bo_study`
- Legacy or exploratory context: `code_simulation.studies.run_open_loop_study` and older `results/open_loop_study/` material

## Current unresolved methodological questions

- Whether the externally fixed time parameterization is still too crude
- Whether the operational front definition is robust enough for optimization
- Whether the chosen observable is the right one for a defensible BO study
- Whether the final trajectory parameterization is scientifically adequate

## Current practical rule

Treat the current workflow as a staged program:

1. maintain the active defaults and organized layout,
2. run the locked `n8` uniform campaign,
3. summarize robustness across the 5-seed target matrix,
4. promote selected candidates to `results/active/n8/fine/`.
