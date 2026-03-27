# Cryostage Characterization

This project extends the existing `Cryostage_Firmware` baseline into a dedicated characterization workflow while keeping the same PlatformIO firmware structure and the same PC-side serial-control philosophy.

## Structure

- `include/`
  - `control.hpp`: existing PID-based temperature controller core
  - `serial_cmd.hpp`: serial command parser extended with `char ...` commands
  - `characterization.hpp`: new staircase state machine
- `src/`
  - `main.cpp`: sensor sampling, control scheduling, telemetry streaming
  - `control.cpp`: PID/safety logic reused from the baseline with clearer fault reporting
  - `serial_cmd.cpp`: manual commands plus automatic characterization commands
  - `characterization.cpp`: automatic step/stabilization/dwell/timeout logic
- `scripts/cryostage_panel.py`: desktop panel for manual mode, automatic mode, live plots, and CSV logging

## Build And Upload

1. Open the `cryostage_characterization` folder in PlatformIO.
2. Build:

```bash
pio run
```

3. Upload:

```bash
pio run --target upload
```

4. Open a serial monitor if needed:

```bash
pio device monitor -b 115200
```

## Run The Desktop Panel

From the `cryostage_characterization` project folder:

```bash
python scripts/cryostage_panel.py --port COM4
```

If you do not pass `--port`, choose the port from the panel and click `Connect`.

## Manual Mode

Manual mode keeps the original workflow:

1. Connect from the panel.
2. Arm the Peltiers.
3. Enable PID.
4. Enter a manual setpoint and apply it.
5. Start or stop CSV recording from the panel when needed.

Equivalent firmware commands:

- `arm 1`
- `arm 0`
- `pid on`
- `pid off`
- `set -10`
- `status`
- `stream on`

## Automatic Characterization Mode

Default staircase:

`+5 -> 0 -> -5 -> -10 -> -15 -> -18 degC`

Recommended panel flow:

1. Connect.
2. Review the sequence and stabilization settings.
3. Click `Apply Settings`.
4. Optionally start CSV recording.
5. Click `Start Characterization`.

Available firmware commands:

- `char start`
- `char pause`
- `char resume`
- `char stop`
- `char abort`
- `char status`
- `char config`
- `char seq 5,0,-5,-10,-15,-18`
- `char tol 0.5`
- `char dwell 300`
- `char postdwell 300`
- `char maxhold 900`
- `char continue_timeout 1`
- `char slope 0.05`
- `char slope_window 60`

## Sequence Logic

The automatic state machine uses:

- `IDLE`
- `STARTING`
- `MOVING_TO_SETPOINT`
- `STABILIZING`
- `DWELLING`
- `STEP_COMPLETE`
- `PAUSED`
- `FINISHED`
- `ABORTED`
- `FAULT`

Behavior summary:

- `STARTING`: arms the hardware if needed, enables PID, and loads the first target
- `MOVING_TO_SETPOINT`: waits until the filtered plate temperature enters the tolerance band
- `STABILIZING`: requires the filtered plate temperature to stay in band for the configured stabilization dwell
- `DWELLING`: after stabilization, holds at that target for the configured post-stabilization dwell
- `STEP_COMPLETE`: advances to the next setpoint
- `PAUSED`: keeps PID active at the current target and freezes step progression timers

Default timing:

- tolerance band: `+-0.5 degC`
- stabilization dwell: `300 s`
- post-stabilization dwell: `300 s`
- maximum hold per step: `900 s`
- slope criterion: `|dT/dt| < 0.05 degC/min` over `60 s`

The stabilization decision uses a filtered plate temperature rather than the raw RTD reading to reduce false triggers from measurement noise.

## Telemetry And CSV Logging

Firmware telemetry keeps the original `>` key/value line format and appends characterization fields such as:

- `char_en`
- `char_state`
- `char_step`
- `char_total`
- `char_target`
- `char_time_s`
- `stable`
- `timeout`
- `char_filtered`
- `char_slope`
- `fault`
- `fault_msg`

The desktop panel writes the CSV. Each file contains:

- commented metadata header lines starting with `#`
- a normal CSV table with telemetry rows
- event rows with `row_type=event`

Logged event markers include:

- characterization started
- characterization paused
- characterization resumed
- step changed
- stabilization achieved
- timeout
- characterization finished
- characterization aborted
- fault occurred

## Safety Behavior

On `char stop`, `char abort`, or a characterization fault:

- PID is disabled
- output power is forced to zero
- Peltiers are disarmed

During `PAUSED`:

- PID remains enabled
- the active step target is held
- the sequence timer is frozen
- no step advancement occurs until `resume`

Unsafe manual commands are blocked while the automatic characterization state machine is active or paused.

## RTD Fault Handling

- MAX31865 RTD fault bits are ignored by default as shutdown triggers in this project.
- The firmware still reads the RTD temperature and keeps the PID running on the raw RTD value.
- The filtered temperature is reserved for telemetry and characterization stabilization logic.
- Output shutdown now depends on implausible temperature readings, not on transient RTD fault bits caused by PWM/EMI.

## Assumptions

- The existing RTD reading remains the controlled plate temperature variable.
- The default PID gains in `include/control.hpp` were bench-validated on 2026-03-23 with the practical constraints of `<=89%` power and no excursions below `-20 degC`.
- `char dwell` is used for the stabilization dwell, while `char postdwell` controls the extra hold after stabilization.
- The same hardware pinout and sensor stack from the baseline project are still valid.

## TODO / Follow-Up

- Run a real hardware validation sweep to tune the stabilization filter, slope threshold, and timeout settings for publication-grade repeatability.
- Confirm whether the final `FINISHED` behavior should leave the last setpoint active instead of performing a safe shutdown.
- If needed, add a dedicated firmware command to export configuration/version information in a more structured machine-readable block.
