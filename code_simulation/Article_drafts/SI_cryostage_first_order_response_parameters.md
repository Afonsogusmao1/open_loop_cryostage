# Supplementary note: first-order cryostage response parameters

The plate-temperature response parameters used in the open-loop trajectory evaluations were obtained from the cryostage characterization telemetry, rather than adjusted during the trajectory optimization. The calibration uses the files matching `data/characterization_cryostage/characterization_min*/cryostage_characterization_min*.csv`. In the implementation, comment-prefixed metadata lines are skipped, only rows with `row_type == telemetry` are retained, and the columns `panel_t_s`, `set`, `T_cal`, and `power` are interpreted as sample time, requested reference temperature, calibrated plate temperature, and controller output, respectively. Each run is trimmed to the active-cooling interval beginning at the first telemetry row with `power > 1.0`; the samples are then sorted by time, duplicate time stamps are removed, and the time axis is shifted so that the first active sample occurs at `t = 0`.

The fitted response is the same first-order recurrence used during the freezing simulations, but the response time and steady plate temperature are functions of the requested reference temperature:

```text
T_plate,i = alpha_i T_plate,i-1 + (1 - alpha_i) T_plate,ss(T_ref,i-1),
alpha_i = exp(-Delta t_i / tau(T_ref,i-1)).
```

For each characterized setpoint, the active step-response trajectory is fitted as `T_plate(t) = h(t) T_plate(0) + (1 - h(t)) T_plate,ss`, with `h(t) = exp(-t/tau)`. The code evaluates `tau` on a logarithmically spaced grid from 10 to 250 s and estimates `T_plate,ss` by least squares over the full active trajectory of the three replicate runs at that setpoint. The active cooling intervals lasted 454.3-795.5 s after trimming, corresponding to 6.0-8.0 fitted time constants for their respective setpoints; therefore `T_plate,ss` is treated as a fitted asymptotic steady-state term rather than a temperature read after an arbitrary 100-300 s hold. During freezing simulations, `tau(T_ref)` and `T_plate,ss(T_ref)` are obtained by linear interpolation of the characterized lookup values, with linear extrapolation only just outside the characterized interval.

This calibration should be interpreted as a low-order empirical mapping from requested reference temperature to measured plate temperature under the characterization conditions. It does not use freezing-front information and does not define a feedback law for freezing-front control. In the open-loop simulations, it is used only to generate the time-dependent plate-temperature input that is applied as the lower boundary condition in the freezing solver.

## Table SX. Calibrated first-order response parameters

| Quantity | Symbol | Code field | Value | Unit | Estimation |
| --- | --- | --- | --- | --- | --- |
| Representative plate-temperature response time constant | median tau | tau_s | 98.088190 | s | Median of the setpoint-dependent response-time lookup. |
| Linear summary gain for the steady-state lookup | g | gain | 1.007606 | dimensionless | Least-squares linear summary of the fitted T_plate,ss(T_ref) lookup. |
| Linear summary offset for the steady-state lookup | b | offset_C | 0.386078 | deg C | Least-squares linear summary of the fitted T_plate,ss(T_ref) lookup. |
| Characterized reference temperatures | T_ref | reference_temperatures_C | -20, -15, -10, -5 | deg C | Nominal setpoints of the active dry-cooling characterization assays. |
| Setpoint-dependent response time constants | tau(T_ref) | response_tau_s | 75.296, 98.220, 97.957, 99.014 | s | Independent active step-response trajectory fit at each characterized setpoint. |
| Setpoint-dependent steady plate temperatures | T_plate,ss(T_ref) | steady_plate_C | -19.769, -14.762, -9.612, -4.693 | deg C | Independent active step-response trajectory fit at each characterized setpoint. |

## Fit and data summary

| Item | Value |
| --- | --- |
| Characterization files used | 12 |
| Nominal reference targets | -20, -15, -10, -5 deg C |
| Replicates per nominal target | -20 C: 3, -15 C: 3, -10 C: 3, -5 C: 3 |
| Active cooling threshold | First telemetry row with power > 1.0 |
| Active telemetry samples retained | 64320 |
| Adjacent active sample pairs | 64308 |
| Median telemetry sampling interval | 0.111 s |
| Active cooling duration after trimming | 454.3-795.5 s across runs; by target: -20 C: 454.3-486.8 s, -15 C: 651.9-681.4 s, -10 C: 663.6-689.6 s, -5 C: 674.4-795.5 s |
| Active duration relative to fitted response time | 6.0-8.0 fitted time constants |
| Time-constant search interval | 10-250 s for each characterized setpoint |
| Fitting criterion | Minimum pooled active step-response trajectory RMSE at each setpoint |
| Pooled recursive trajectory RMSE on active telemetry | 0.378905 deg C |
| Mean recursive RMSE by nominal target on active telemetry | -20 C: 0.351 C, -15 C: 0.368 C, -10 C: 0.375 C, -5 C: 0.404 C |

## Characterization files included by the default loader

- `data/characterization_cryostage/characterization_min10/cryostage_characterization_min10_I.csv`
- `data/characterization_cryostage/characterization_min10/cryostage_characterization_min10_II.csv`
- `data/characterization_cryostage/characterization_min10/cryostage_characterization_min10_III.csv`
- `data/characterization_cryostage/characterization_min15/cryostage_characterization_min15_I.csv`
- `data/characterization_cryostage/characterization_min15/cryostage_characterization_min15_II.csv`
- `data/characterization_cryostage/characterization_min15/cryostage_characterization_min15_III.csv`
- `data/characterization_cryostage/characterization_min20/cryostage_characterization_min20_I.csv`
- `data/characterization_cryostage/characterization_min20/cryostage_characterization_min20_II.csv`
- `data/characterization_cryostage/characterization_min20/cryostage_characterization_min20_III.csv`
- `data/characterization_cryostage/characterization_min5/cryostage_characterization_min5_I.csv`
- `data/characterization_cryostage/characterization_min5/cryostage_characterization_min5_II.csv`
- `data/characterization_cryostage/characterization_min5/cryostage_characterization_min5_III.csv`
