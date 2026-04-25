# Open-Loop Settings

This folder contains editable settings for the active freezing and Bayesian-optimization workflows. The goal is to change run definitions without editing Python source files.

## Simulation Profiles

`simulation_profiles.toml` contains named simulation profiles. Use `optimization` for BO objective evaluations and `full_process_article` for fine final/article-style simulations.

- `cryostage_dt_s`: Sampling interval for the reduced cryostage response model and the generated `T_plate(t)` profile.
- `solver_dt_s`: Time step used by the freezing PDE solver.
- `Nr`, `Nz`: Radial and axial mesh resolution for the axisymmetric solver.
- `Nz_front`: Number of centerline samples used to locate the freezing front. The fine profile uses `800`; the BO profile uses `200` to reduce runtime.
- `Nr_front_curve`, `Nz_front_curve`: Curved-front sampling resolution when curved-front output is enabled.
- `write_every_s`: Field-output cadence. A very large value effectively disables repeated field writes.
- `write_field_output`: Whether to write XDMF/H5 field output.
- `write_probe_csv`: Whether to write probe temperature CSV output.
- `show_progress`: Whether the solver prints progress messages.
- `enable_front_curve`: Whether the solver writes curved-front diagnostics.
- `use_tabulated_water_ice`: Whether to use tabulated water/ice thermal properties, including ice density in the transient `rho*cp` storage term.

## Ambient Model

Each profile has an `[ambient]` table.

- `mode = "interpolate_from_cryostage"` makes ambient air temperature depend on the modeled plate/cryostage temperature.
- `cryostage_temperature_C` stores the measured cryostage/plate calibration temperatures.
- `ambient_temperature_C` stores the corresponding measured ambient air temperatures.
- `extrapolation = "clamp"` means values colder than `-20 C` use the `-20 C` ambient value, and values warmer than `-10 C` use the `-10 C` ambient value.

The current calibration points are:

| Cryostage/plate temperature (C) | Ambient temperature (C) |
| --- | --- |
| -20.0 | 5.75 |
| -15.0 | 7.568 |
| -10.0 | 9.7078 |

`ThermalBCs.T_room_C` remains the initial/fallback ambient value. When the interpolation model is enabled, the convective ambient constant is updated during the transient solve from the modeled plate temperature.

## Velocity-Control Settings

`velocity_control.toml` defines the physical velocity-control target shared by the manual evaluator and the velocity-control BO runner.

- `run_name`: Output folder name under `code_simulation/results/active/n<num_knots>/fine/`.
- `simulation_profile`: Simulation profile loaded from `simulation_profiles.toml`.
- `target_front_speed_mm_s`: Desired constant freezing-front speed in `mm/s`.
- `control_z_min_mm`, `control_z_max_mm`: Axial interval used for the direct front-tracking objective. The active configuration uses `2.5` to `11.5 mm`, which adds a `0.5 mm` margin around the thermocouple span `3.0-11.0 mm`.
- `initial_water_temperature_C`: Water temperature immediately after filling the mold.
- `initial_plate_temperature_C`: Modeled plate/base temperature at the instant cooling starts.
- `no_warm_hold`: Must be `true` in the active protocol. The simulation starts from water in the mold and begins cooling immediately.
- `characterization_temperature_margin_C`: Symmetric margin used only when comparing requested cryostage temperatures against the characterization-supported targets and support bands. The active value is `0.5 C`. It does not widen the hard physical BO bounds.
- `horizon_s`: Duration of the cooling trajectory after filling.
- `num_knots`: Number of manual `T_ref` knot temperatures.
- `knot_time_schedule`: Fixed distribution of knot times across the cooling horizon.
- `knot_time_custom_support_tau`: Normalized knot support for `custom` schedules.
- `theta_C`: Manual `T_ref` knot temperatures being evaluated.
- `T_ref_bounds_C`: Physical bounds applied to the manual cooling trajectory.
- `require_monotone_nonincreasing`: Whether the manual trajectory must cool monotonically.

## BO Settings

`bo.toml` controls the active BO run.

- `num_knots`: Number of optimized temperature values in `theta`.
- `knot_time_schedule`: Fixed distribution of knot times across the horizon. Supported values are `uniform`, `early_dense`, `mid_dense`, `late_dense`, and `custom`.
- `knot_time_custom_support_tau`: Normalized support times for `custom`; leave empty for non-custom schedules.
- `theta0_C`: Initial BO trajectory evaluated before random/acquisition suggestions when `seed_with_theta0` is enabled.
- The active defaults in `bo.toml` are locked to the `n8` uniform campaign anchored on the defended `n3` uniform result.
- `T_ref_bounds_C`: Global admissible range for requested reference temperatures.
- `theta_bounds_C`: Optional per-knot BO bounds. Leave empty to use `T_ref_bounds_C` for every knot.
- `optimize_knot_times`: Reserved for a later phase. It must remain `false`; the current BO optimizes temperatures only.
- Direct characterization support is interpreted with the `velocity_control.toml` margin. With the active `+-0.5 C` margin and characterized targets `-20, -15, -10, -5 C`, the direct-support band is `-20.5 to -4.5 C`. Values outside that band are still allowed if they remain inside the hard physical BO bounds, but they are treated as outside direct characterization support.
- `random_seed`: Random seed used by the BO backend.
- `init_points`: Number of random BO suggestions.
- `n_iter`: Number of acquisition-guided BO suggestions.
- `acquisition_kind`: Acquisition function, one of `ucb`, `ei`, or `poi`.
- `acquisition_kappa`: UCB exploration parameter.
- `acquisition_xi`: EI/POI improvement margin.
- `seed_with_theta0`: Whether to evaluate the canonical seed trajectory before BO suggestions.
- `parameterization_kind`: BO search-space parameterization. `monotone_unit_box` keeps the BO in a normalized monotone-feasible space and reconstructs the physical `theta`.
- `init_strategy`: Initial BO candidate generation strategy. Supported values are `uniform_random`, `feasible_local`, and `feasible_local_deterministic`.
- `init_local_sigma`: Typical perturbation radius for local BO initialization in normalized BO space.
- `init_max_attempts_per_point`: Maximum number of admissibility-prechecked attempts per requested init point.
- `local_refinement_points`: Optional number of deterministic local refinement evaluations around the incumbent after the main BO loop. Set `0` to disable.
- `local_refinement_sigma`: Typical perturbation radius for the post-BO local refinement in normalized BO space.
- `infeasible_objective_penalty`: Deterministic objective returned when admissibility rejects a candidate before simulation.

## Typical Commands

Inspect the effective velocity-control BO settings without launching simulations:

```bash
python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config
```

Inspect the freezing-diagnostics settings without launching solver cases:

```bash
python -m code_simulation.verification.run_full_freezing_diagnostics --simulation-profile full_process_article --dry-run-config
```

Inspect the front-speed reachability candidates and profiles without launching solver cases:

```bash
python -m code_simulation.verification.run_front_speed_reachability_study --dry-run-config
```

Inspect the manual velocity-control evaluation without launching a solver case:

```bash
python -m code_simulation.verification.run_velocity_control_evaluation --dry-run-config
```

Run one manual velocity-control evaluation:

```bash
python -m code_simulation.verification.run_velocity_control_evaluation --overwrite
```

Inspect the velocity-control BO configuration without launching solver cases:

```bash
python -m code_simulation.optimization.run_velocity_control_bo --dry-run-config
```

Inspect the locked `n8` uniform-from-`n3` campaign bundle without launching simulations:

```bash
python -m code_simulation.studies.run_n8_uniform_from_n3 --dry-run
```

Run the active velocity-control BO using the coarse optimization profile and current `bo.toml` defaults:

```bash
python -m code_simulation.optimization.run_velocity_control_bo \
  --target-front-speed-mm-s 0.008 \
  --num-knots 8 \
  --simulation-profile optimization \
  --run-name bo_v0p008_n8_uniform_n3anchor_seed17 \
  --overwrite
```
