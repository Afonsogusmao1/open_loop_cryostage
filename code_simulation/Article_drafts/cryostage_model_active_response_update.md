# Cryostage model update: setpoint-dependent active response

## Why the model was changed

The original cryostage model used a first-order recurrence calibrated from all telemetry rows in the characterization CSVs:

```text
T_plate,i = alpha_i T_plate,i-1 + (1 - alpha_i)(g T_ref,i-1 + b),
alpha_i = exp(-Delta t_i / tau).
```

That calibration minimized one-step recurrence error. It also included initial telemetry where the requested setpoint was already present in the CSV, but the controller output had not yet started acting. This made the fitted model look acceptable locally while producing poor full-trajectory predictions.

A first correction kept a single global time constant but refitted the complete active step response. That reduced the active-trajectory RMSE, but the residuals still showed a systematic setpoint dependence: the -20 C response was faster than the warmer setpoints.

The implemented model therefore keeps the same first-order structure, but makes both the response time and steady plate temperature functions of the requested reference temperature:

```text
dT_plate/dt = (T_inf(T_ref) - T_plate) / tau(T_ref)
```

For each characterized setpoint, the fit uses only the active-cooling interval beginning at the first telemetry row with `power > 1.0`, and fits the complete active step response:

```text
T_plate(t) = h(t) T_plate(0) + (1 - h(t)) T_inf,
h(t) = exp(-t / tau).
```

During freezing simulations, `tau(T_ref)` and `T_inf(T_ref)` are obtained by linear interpolation of the fitted lookup values, with linear extrapolation just outside the characterized range.

## Model selected

The implemented default parameters are:

| `T_ref` (deg C) | `tau(T_ref)` (s) | `T_inf(T_ref)` (deg C) |
| ---: | ---: | ---: |
| -20 | 75.296198 | -19.769483 |
| -15 | 98.219855 | -14.761845 |
| -10 | 97.956524 | -9.612014 |
| -5 | 99.014102 | -4.692656 |

The legacy scalar fields in `DEFAULT_CRYOSTAGE_PARAMS` are only summaries of the lookup: `tau_s = 98.088190 s`, `gain = 1.007606`, and `offset_C = 0.386079 deg C`. The simulation uses the lookup fields.

## Diagnostic comparison

| Model / data alignment | RMSE | MAE | Max abs. error |
| --- | ---: | ---: | ---: |
| Old model, raw full CSVs | 1.455 C | 0.907 C | 11.031 C |
| Old model, active-only CSVs | 1.732 C | 1.181 C | 6.986 C |
| Single-tau active first-order model | 0.585 C | 0.419 C | 2.595 C |
| Setpoint-dependent active first-order model | 0.379 C | 0.225 C | 1.740 C |
| Setpoint-dependent model, leave-one-replicate-out | 0.381 C | 0.226 C | 1.760 C |

Mean active recursive RMSE by setpoint with the implemented model:

| Setpoint | Mean RMSE |
| --- | ---: |
| -5 C | 0.403 C |
| -10 C | 0.375 C |
| -15 C | 0.368 C |
| -20 C | 0.351 C |

Second-order and extra-delay candidates were also checked against the same active telemetry. They did not improve the single-tau fit: the two-pole candidate gave approximately 0.674 C RMSE, and adding an explicit delay worsened it further. The selected model is therefore still low order and article-explainable, but no longer forces all setpoints to share the same response time.

## Consequence for BO results

All existing BO and fine-confirmation results generated before this change used older cryostage dynamics. They should be treated as diagnostic only. Any article-quality n8 BO/fine campaign should be regenerated with the setpoint-dependent active-response cryostage model.
