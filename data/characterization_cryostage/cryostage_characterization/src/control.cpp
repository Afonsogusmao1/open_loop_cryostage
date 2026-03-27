#include "control.hpp"

#include <math.h>
#include <string.h>

PeltierTempController::PeltierTempController(AllArmedFn allArmed, ArmAllFn armAll, ApplyFn apply, const Config &cfg)
    : allArmed_(allArmed),
      armAll_(armAll),
      apply_(apply),
      cfg_(cfg),
      pid_(&pid_input_, &pid_output_, &pid_setpoint_, Kp_, Ki_, Kd_, DIRECT) // direction set in begin()
{
}

void PeltierTempController::begin()
{
    pid_.SetOutputLimits(cfg_.power_min, cfg_.power_max);
    pid_.SetSampleTime((int)cfg_.control_ms);
    pid_.SetMode(MANUAL);
    pid_.SetControllerDirection(cfg_.cooling_reverse ? REVERSE : DIRECT);

    resetSensorFilter_();
    pid_enabled_ = false;
    pid_active_ = false;
    pid_output_ = 0.0;

    power_cmd_ = 0.0f;
    power_target_ = 0.0f;
    power_out_ = 0.0f;
    strncpy(fault_reason_, "none", sizeof(fault_reason_) - 1);
    fault_reason_[sizeof(fault_reason_) - 1] = '\0';

    applyPowerOut_();
}

void PeltierTempController::applyPowerOut_()
{
    apply_(power_out_);
}

void PeltierTempController::resetSensorFilter_()
{
    t_raw_ = NAN;
    t_cal_ = NAN;
    sensor_history_[0] = NAN;
    sensor_history_[1] = NAN;
    sensor_history_[2] = NAN;
    sensor_history_count_ = 0;
    sensor_history_head_ = 0;
    sensor_filter_valid_ = false;
}

float PeltierTempController::filterSensorReading_(float t_cal_raw)
{
    float filtered = t_cal_raw;

    if (cfg_.sensor_filter_median3)
    {
        sensor_history_[sensor_history_head_] = t_cal_raw;
        sensor_history_head_ = (sensor_history_head_ + 1) % 3;
        if (sensor_history_count_ < 3)
            sensor_history_count_++;

        if (sensor_history_count_ == 2)
        {
            filtered = 0.5f * (sensor_history_[0] + sensor_history_[1]);
        }
        else if (sensor_history_count_ == 3)
        {
            const float a = sensor_history_[0];
            const float b = sensor_history_[1];
            const float c = sensor_history_[2];

            if ((a <= b && b <= c) || (c <= b && b <= a))
                filtered = b;
            else if ((b <= a && a <= c) || (c <= a && a <= b))
                filtered = a;
            else
                filtered = c;
        }
    }

    const float alpha = constrain(cfg_.sensor_filter_alpha, 0.001f, 1.0f);
    if (!sensor_filter_valid_ || alpha >= 0.999f)
    {
        t_cal_ = filtered;
        sensor_filter_valid_ = true;
        return t_cal_;
    }

    t_cal_ = t_cal_ + alpha * (filtered - t_cal_);
    return t_cal_;
}

float PeltierTempController::rampTowards_(float current, float target, float dt_s) const
{
    if (cfg_.ramp_rate_pct_per_s <= 0.0f)
        return target;

    const float max_delta = cfg_.ramp_rate_pct_per_s * dt_s;
    const float delta = target - current;

    if (fabsf(delta) <= max_delta)
        return target;
    return current + (delta > 0.0f ? max_delta : -max_delta);
}

void PeltierTempController::pidSetEnabled_(bool on)
{
    pid_enabled_ = on;

    if (!pid_enabled_)
    {
        if (pid_active_)
            pid_.SetMode(MANUAL);
        pid_active_ = false;
        pid_output_ = 0.0;
    }
}

void PeltierTempController::inhibitOutput_(const char *reason)
{
    // Cut output to zero and pause PID actuation.
    // Keep the driver armed state unchanged so control can resume automatically.
    power_cmd_ = 0.0f;
    power_target_ = 0.0f;
    power_out_ = 0.0f;
    applyPowerOut_();

    if (pid_active_)
        pid_.SetMode(MANUAL);
    pid_active_ = false;

    resetSensorFilter_();
    sensor_inhibit_ = true;
    sensor_ok_ = false;
    sensor_good_streak_ = 0;

    strncpy(fault_reason_, reason, sizeof(fault_reason_) - 1);
    fault_reason_[sizeof(fault_reason_) - 1] = '\0';

    Serial.printf("[ERR] Output inhibited: %s\r\n", reason);
}

void PeltierTempController::onSensorFault(uint8_t fault_code)
{
    last_rtd_fault_ = fault_code;
    if (rtd_fault_streak_ < 255)
        rtd_fault_streak_++;

    // By default this project ignores RTD fault bits as shutdown triggers
    // because they are frequently caused by PWM/EMI. Protection is instead
    // enforced by plausibility checks on the measured temperature.
    if (cfg_.rtd_fault_trip_count == 0)
    {
        if (cfg_.print_transient_rtd_faults)
        {
            Serial.printf("[WARN] RTD fault 0x%02X ignored (streak=%u)\r\n",
                          fault_code,
                          rtd_fault_streak_);
        }
        return;
    }

    if (rtd_fault_streak_ < cfg_.rtd_fault_trip_count)
    {
        if (cfg_.print_transient_rtd_faults)
        {
            Serial.printf("[WARN] RTD fault 0x%02X (%u/%u)\r\n",
                          fault_code,
                          rtd_fault_streak_,
                          cfg_.rtd_fault_trip_count);
        }
        // Keep controlling and wait to see if the fault persists.
        return;
    }

    // Persistent RTD fault -> inhibit output until recovery.
    if (!sensor_inhibit_)
        inhibitOutput_("RTD_fault");
}

void PeltierTempController::onSensorReading(float t_cal)
{
    t_raw_ = t_cal;
    rtd_fault_streak_ = 0;

    const bool sane = isfinite(t_raw_) &&
                      (t_raw_ >= cfg_.temp_min_plausible) &&
                      (t_raw_ <= cfg_.temp_max_plausible);

    if (!sane)
    {
        sensor_ok_ = false;
        sensor_good_streak_ = 0;
        resetSensorFilter_();

        if (!sensor_inhibit_)
            inhibitOutput_("temp_out_of_range");
        return;
    }

    filterSensorReading_(t_raw_);
    sensor_ok_ = true;
    if (sensor_good_streak_ < 255)
        sensor_good_streak_++;

    if (sensor_inhibit_ && sensor_good_streak_ >= cfg_.sensor_recover_samples)
    {
        sensor_inhibit_ = false;
        strncpy(fault_reason_, "none", sizeof(fault_reason_) - 1);
        fault_reason_[sizeof(fault_reason_) - 1] = '\0';
        Serial.println("[OK] Sensor recovered; resuming control");
    }
}

void PeltierTempController::updatePidActivation_()
{
    const bool should_run = pid_enabled_ && allArmed_() && sensor_ok_ && !sensor_inhibit_;

    if (should_run && !pid_active_)
    {
        // Bumpless transfer: start output where we are.
        pid_output_ = (double)power_out_;

        pid_.SetOutputLimits(cfg_.power_min, cfg_.power_max);
        pid_.SetSampleTime((int)cfg_.control_ms);
        pid_.SetTunings(Kp_, Ki_, Kd_);
        pid_.SetMode(AUTOMATIC);

        pid_active_ = true;
    }
    else if (!should_run && pid_active_)
    {
        pid_.SetMode(MANUAL);
        pid_active_ = false;
        pid_output_ = 0.0;
    }
}

void PeltierTempController::update(float dt_s)
{
    // If disarmed -> output must be 0.
    if (!allArmed_())
    {
        power_target_ = 0.0f;
        power_out_ = 0.0f;
        applyPowerOut_();
        updatePidActivation_();
        return;
    }

    // Sensor bad/inhibited -> output must be 0.
    if (!sensor_ok_ || sensor_inhibit_)
    {
        power_target_ = 0.0f;
        power_out_ = 0.0f;
        applyPowerOut_();
        updatePidActivation_();
        return;
    }

    updatePidActivation_();

    if (pid_active_)
    {
        // Keep control on the raw RTD.  The filtered signal is for
        // telemetry/plotting only.
        pid_input_ = (double)t_raw_;
        pid_.Compute();
        power_target_ = (float)pid_output_;
    }
    else
    {
        power_target_ = power_cmd_;
    }

    power_target_ = constrain(power_target_, cfg_.power_min, cfg_.power_max);
    power_out_ = rampTowards_(power_out_, power_target_, dt_s);
    power_out_ = constrain(power_out_, cfg_.power_min, cfg_.power_max);

    applyPowerOut_();
}

bool PeltierTempController::cmdArm()
{
    if (sensor_inhibit_)
    {
        Serial.println("[ERR] Cannot arm: sensor fault/inhibit active");
        return false;
    }

    // Start from 0 output when arming.
    power_out_ = 0.0f;
    power_target_ = 0.0f;
    applyPowerOut_();

    return armAll_(true);
}

void PeltierTempController::cmdDisarm()
{
    pidSetEnabled_(false);

    power_out_ = 0.0f;
    power_target_ = 0.0f;
    applyPowerOut_();

    armAll_(false);
}

void PeltierTempController::cmdOff()
{
    pidSetEnabled_(false);

    power_cmd_ = 0.0f;
    power_target_ = 0.0f;
    power_out_ = 0.0f;
    applyPowerOut_();
}

void PeltierTempController::cmdSetManualPower(float pct)
{
    power_cmd_ = constrain(pct, cfg_.power_min, cfg_.power_max);
}

void PeltierTempController::cmdPidEnable(bool on)
{
    pidSetEnabled_(on);
}

void PeltierTempController::cmdSetpoint(float degC)
{
    const float sp = constrain(degC, cfg_.temp_min_plausible, cfg_.temp_max_plausible);
    pid_setpoint_ = (double)sp;
}

void PeltierTempController::cmdSetKp(double v)
{
    Kp_ = v;
    pid_.SetTunings(Kp_, Ki_, Kd_);
}

void PeltierTempController::cmdSetKi(double v)
{
    Ki_ = v;
    pid_.SetTunings(Kp_, Ki_, Kd_);
}

void PeltierTempController::cmdSetKd(double v)
{
    Kd_ = v;
    pid_.SetTunings(Kp_, Ki_, Kd_);
}

void PeltierTempController::printHelp() const
{
    Serial.println();
    Serial.println("Commands:");
    Serial.println("  arm                -> enable all Peltier drivers (EN HIGH)");
    Serial.println("  disarm             -> disable all Peltier drivers (EN LOW), output = 0");
    Serial.println("  arm 0|1            -> 1=arm, 0=disarm");
    Serial.println("  power <0-100>      -> set manual power (PID must be OFF)");
    Serial.println("  off                -> set power=0 and disable PID (keeps armed state)");
    Serial.println("  set <degC>         -> set PID target temperature (alias: setpoint <degC>)");
    Serial.println("  pid 0|1            -> disable/enable closed-loop control (alias: pid off|on)");
    Serial.println("  kp <v> / ki <v> / kd <v> -> update PID gains");
    Serial.println("  status             -> print current state, characterization, and thermocouples");
    Serial.println("  stream 0|1         -> stop/start streaming telemetry lines");
    Serial.println("  char help          -> show characterization commands");
    Serial.println("  help               -> show this menu");
    Serial.println();
}

void PeltierTempController::printStatus(bool stream_on) const
{
    Serial.printf(
        "[STATUS] armed=%d | mode=%s | setpoint=%.2f | T_raw=%.4f | T_cal=%.4f | pid_out=%.2f | power_cmd=%.1f | power_out=%.1f | stream=%d | kp=%.4f | ki=%.4f | kd=%.4f | sensor_ok=%d | inhibit=%d | fault_code=0x%02X | fault_msg=%s\r\n",
        armed() ? 1 : 0,
        modeName(),
        (float)pid_setpoint_,
        t_raw_,
        (float)t_cal_,
        (float)pid_output_,
        power_cmd_,
        powerOut(),
        stream_on ? 1 : 0,
        (float)Kp_,
        (float)Ki_,
        (float)Kd_,
        sensor_ok_ ? 1 : 0,
        sensor_inhibit_ ? 1 : 0,
        last_rtd_fault_,
        fault_reason_);
}
