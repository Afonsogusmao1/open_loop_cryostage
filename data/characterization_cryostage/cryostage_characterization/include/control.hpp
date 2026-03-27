#pragma once

#include <Arduino.h>
#include <PID_v1.h>

class PeltierTempController
{
public:
    using AllArmedFn = bool (*)();
    using ArmAllFn = bool (*)(bool on);
    using ApplyFn = void (*)(float duty_pct);

    struct Config
    {
        uint32_t control_ms = 100;

        float power_min = 0.0f;
        float power_max = 100.0f;

        // 0 disables ramp limiting
        float ramp_rate_pct_per_s = 120.0f;

        float temp_min_plausible = -120.0f;
        float temp_max_plausible = 200.0f;

        // MAX31865 fault bits are often noisy under PWM/EMI. In this project
        // we prefer to trust the measured temperature value and only trip on
        // implausible readings. Set >0 only if you explicitly want RTD-fault
        // bits to hard-stop the output again.
        uint8_t rtd_fault_trip_count = 0;

        // Print transient fault warnings while counting toward trip.
        bool print_transient_rtd_faults = false;

        uint8_t sensor_recover_samples = 3;

        // Light filtering for telemetry / characterization only.
        // The PID itself runs on the raw RTD value.
        bool sensor_filter_median3 = true;
        float sensor_filter_alpha = 0.12f;

        // Cooling-only: more PWM => colder -> REVERSE is typical.
        bool cooling_reverse = true;
    };

    PeltierTempController(AllArmedFn allArmed, ArmAllFn armAll, ApplyFn apply, const Config &cfg);

    void begin();

    // --- Sensor / safety ---
    void onSensorFault(uint8_t fault_code);
    void onSensorReading(float t_cal);

    // --- Control tick (call at fixed rate) ---
    void update(float dt_s);

    // --- Commands ---
    bool cmdArm(); // returns false if inhibited
    void cmdDisarm();
    void cmdOff(); // output=0 + PID off; keeps armed state

    void cmdSetManualPower(float pct); // only meaningful if PID is OFF
    void cmdPidEnable(bool on);

    void cmdSetpoint(float degC);
    void cmdSetKp(double v);
    void cmdSetKi(double v);
    void cmdSetKd(double v);

    // --- Printing helpers ---
    void printHelp() const;
    void printStatus(bool stream_on) const;

    // --- Status getters ---
    bool pidEnabled() const { return pid_enabled_; }
    bool pidActive() const { return pid_active_; }
    bool armed() const { return allArmed_(); }
    bool sensorOk() const { return sensor_ok_; }
    bool sensorInhibit() const { return sensor_inhibit_; }
    bool faultActive() const { return sensor_inhibit_; }
    uint8_t lastFault() const { return last_rtd_fault_; }
    const char *faultReason() const { return fault_reason_; }

    float tRaw() const { return t_raw_; }
    float tCal() const { return t_cal_; }

    float powerCmd() const { return power_cmd_; }
    float powerTarget() const { return power_target_; }
    float powerOut() const { return allArmed_() ? power_out_ : 0.0f; }

    double setpoint() const { return pid_setpoint_; }
    double pidOutput() const { return pid_output_; }

    double kp() const { return Kp_; }
    double ki() const { return Ki_; }
    double kd() const { return Kd_; }

    const char *modeName() const { return pid_enabled_ ? "pid" : "open-loop"; }

private:
    AllArmedFn allArmed_;
    ArmAllFn armAll_;
    ApplyFn apply_;
    Config cfg_;

    // sensor state
    float t_raw_ = NAN;
    float t_cal_ = NAN;
    bool sensor_ok_ = false;
    uint8_t last_rtd_fault_ = 0;
    uint8_t rtd_fault_streak_ = 0;
    bool sensor_inhibit_ = false;
    uint8_t sensor_good_streak_ = 0;
    char fault_reason_[32] = "none";
    float sensor_history_[3] = {NAN, NAN, NAN};
    uint8_t sensor_history_count_ = 0;
    uint8_t sensor_history_head_ = 0;
    bool sensor_filter_valid_ = false;

    // power state
    float power_cmd_ = 0.0f;
    float power_target_ = 0.0f;
    float power_out_ = 0.0f;

    // PID state
    bool pid_enabled_ = false;
    bool pid_active_ = false;

    double pid_input_ = 0.0;
    double pid_output_ = 0.0;
    double pid_setpoint_ = 0.0;

    // Bench-validated on 2026-03-23 with <=89% power and no excursions below -20 degC.
    double Kp_ = 2.563308;
    double Ki_ = 0.038976;
    double Kd_ = 0.403400;

    PID pid_;

    // internals
    void applyPowerOut_();
    void pidSetEnabled_(bool on);
    void updatePidActivation_();
    void inhibitOutput_(const char *reason);
    float rampTowards_(float current, float target, float dt_s) const;
    void resetSensorFilter_();
    float filterSensorReading_(float t_cal_raw);
};
