#include <Arduino.h>
#include <SPI.h>

#include "characterization.hpp"
#include "control.hpp"
#include "peltier_bts.hpp"
#include "pins.hpp"
#include "rtd_calibration.hpp"
#include "rtd_max31865.hpp"
#include "serial_cmd.hpp"
#include "tc_manager.hpp"

static constexpr char FW_VERSION[] = "cryostage_characterization_v3_rtdfault_ignored";

// ===== RTD =====
static RtdMax31865 rtd(PIN_CS_RTD, RNOMINAL, RREF);
static constexpr auto WIRE_MODE = RtdMax31865::WireMode::W3;
static constexpr auto MAINS_FILTER = RtdMax31865::MainsFilter::Hz50;

// ===== Scheduling =====
static constexpr uint32_t STREAM_MS = 100;
static constexpr uint32_t SENSOR_MS = 100;
static constexpr uint32_t CONTROL_MS = 100;

static bool stream_on = true;

// ===== Thermocouples =====
static TcManager tc(PIN_CS_TC3, PIN_CS_TC7, PIN_CS_TC12, PIN_CS_TAMB);

// ===== Peltiers =====
static PeltierBTS p1(25, 5, 0);
static PeltierBTS p2(26, 17, 1);
static PeltierBTS p3(32, 21, 2);
static PeltierBTS p4(33, 22, 3);

// --- Hardware wrappers for controller ---
static bool hwAllArmed()
{
    return p1.isArmed() && p2.isArmed() && p3.isArmed() && p4.isArmed();
}

static void hwApplyAll(float duty_pct)
{
    p1.setDutyPercent(duty_pct);
    p2.setDutyPercent(duty_pct);
    p3.setDutyPercent(duty_pct);
    p4.setDutyPercent(duty_pct);
}

static bool hwArmAll(bool on)
{
    p1.arm(on);
    p2.arm(on);
    p3.arm(on);
    p4.arm(on);
    return true;
}

// ===== Controller configuration =====
static PeltierTempController::Config CTRL_CFG = []()
{
    PeltierTempController::Config c;
    c.control_ms = CONTROL_MS;
    c.power_min = 0.0f;
    c.power_max = 89.0f;
    c.ramp_rate_pct_per_s = 120.0f;
    c.temp_min_plausible = -120.0f;
    c.temp_max_plausible = 200.0f;
    // Ignore MAX31865 fault bits by default. In practice they are often PWM
    // artefacts; we trust the actual temperature reading and only trip on
    // implausible values.
    c.rtd_fault_trip_count = 0;
    c.print_transient_rtd_faults = false;
    c.sensor_recover_samples = 3;
    c.sensor_filter_median3 = true;
    c.sensor_filter_alpha = 0.12f;
    c.cooling_reverse = true;
    return c;
}();

static CharacterizationController::Config CHAR_CFG = []()
{
    CharacterizationController::Config c;
    c.tolerance_c = 0.5f;
    c.stabilization_dwell_s = 300;
    c.dwell_after_stable_s = 300;
    c.max_hold_s = 900;
    c.continue_on_timeout = true;
    c.slope_limit_c_per_min = 0.05f;
    c.slope_window_s = 60;
    c.ema_alpha = 0.10f;
    return c;
}();

static PeltierTempController ctrl(hwAllArmed, hwArmAll, hwApplyAll, CTRL_CFG);
static CharacterizationController char_ctrl(ctrl, CHAR_CFG);
static SerialCmdParser parser(ctrl, tc, char_ctrl, stream_on);

static void emitTelemetry()
{
    const uint32_t ts = millis();

    const float setpoint = (float)ctrl.setpoint();
    const bool pid_on = ctrl.pidEnabled();
    const bool armed = hwAllArmed();
    const float t_raw = ctrl.tRaw();
    const float t_cal = ctrl.tCal();
    const float power = ctrl.powerOut();

    const auto &c3 = tc.ch(0);
    const auto &c7 = tc.ch(1);
    const auto &c12 = tc.ch(2);
    const auto &camb = tc.ch(3);

    const bool fault_flag = ctrl.faultActive() || char_ctrl.faultActive();
    char fault_msg[48];
    if (char_ctrl.faultActive())
    {
        snprintf(fault_msg, sizeof(fault_msg), "%s", char_ctrl.faultMessage());
    }
    else if (ctrl.faultActive())
    {
        if (ctrl.lastFault() != 0)
            snprintf(fault_msg, sizeof(fault_msg), "%s_0x%02X", ctrl.faultReason(), ctrl.lastFault());
        else
            snprintf(fault_msg, sizeof(fault_msg), "%s", ctrl.faultReason());
    }
    else
    {
        snprintf(fault_msg, sizeof(fault_msg), "none");
    }

    auto emitTc = [&](const char *key, float v, bool ok)
    {
        Serial.print(',');
        Serial.print(key);
        Serial.print(':');
        if (ok)
            Serial.print(v, 4);
        else
            Serial.print("nan");
    };

    Serial.print('>');
    Serial.print("ts:");
    Serial.print(ts);
    Serial.print(",set:");
    Serial.print(setpoint, 2);
    Serial.print(",pid:");
    Serial.print(pid_on ? 1 : 0);
    Serial.print(",armed:");
    Serial.print(armed ? 1 : 0);
    Serial.print(",T_cal:");
    Serial.print(t_cal, 4);
    Serial.print(",T_raw:");
    Serial.print(t_raw, 4);
    Serial.print(",power:");
    Serial.print(power, 1);

    emitTc("T3", c3.cal_C, c3.valid);
    emitTc("T7", c7.cal_C, c7.valid);
    emitTc("T12", c12.cal_C, c12.valid);
    emitTc("Tamb", camb.cal_C, camb.valid);

    emitTc("T3_raw", c3.raw_C, c3.valid);
    emitTc("T7_raw", c7.raw_C, c7.valid);
    emitTc("T12_raw", c12.raw_C, c12.valid);
    emitTc("Tamb_raw", camb.raw_C, camb.valid);

    Serial.print(",f3:");
    if (c3.fault < 16)
        Serial.print('0');
    Serial.print(c3.fault, HEX);
    Serial.print(",f7:");
    if (c7.fault < 16)
        Serial.print('0');
    Serial.print(c7.fault, HEX);
    Serial.print(",f12:");
    if (c12.fault < 16)
        Serial.print('0');
    Serial.print(c12.fault, HEX);
    Serial.print(",famb:");
    if (camb.fault < 16)
        Serial.print('0');
    Serial.print(camb.fault, HEX);

    Serial.print(",char_en:");
    Serial.print(char_ctrl.enabled() ? 1 : 0);
    Serial.print(",char_state:");
    Serial.print(char_ctrl.stateName());
    Serial.print(",char_step:");
    Serial.print(char_ctrl.stepNumber());
    Serial.print(",char_total:");
    Serial.print(char_ctrl.totalSteps());
    Serial.print(",char_target:");
    Serial.print(char_ctrl.currentTarget(), 2);
    Serial.print(",char_time_s:");
    Serial.print(char_ctrl.stepElapsedMs(ts) / 1000.0f, 1);
    Serial.print(",stable:");
    Serial.print(char_ctrl.stabilized() ? 1 : 0);
    Serial.print(",timeout:");
    Serial.print(char_ctrl.timeoutFlag() ? 1 : 0);
    Serial.print(",char_in_band:");
    Serial.print(char_ctrl.inBand() ? 1 : 0);
    Serial.print(",char_filtered:");
    if (char_ctrl.filteredTempValid())
        Serial.print(char_ctrl.filteredTemp(), 4);
    else
        Serial.print("nan");
    Serial.print(",char_slope:");
    Serial.print(char_ctrl.slopeCPerMin(), 5);
    Serial.print(",fault:");
    Serial.print(fault_flag ? 1 : 0);
    Serial.print(",fault_msg:");
    Serial.print(fault_msg);
    Serial.print(",fw:");
    Serial.print(FW_VERSION);

    Serial.println();
}

void setup()
{
    Serial.begin(115200);
    delay(200);

    SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI);
    rtd.begin(WIRE_MODE, MAINS_FILTER);
    tc.begin();

    p1.begin();
    p2.begin();
    p3.begin();
    p4.begin();

    hwApplyAll(0.0f);
    hwArmAll(false);

    ctrl.begin();
    char_ctrl.begin();

    ctrl.printHelp();
    ctrl.printStatus(stream_on);
    char_ctrl.printConfig();
    char_ctrl.printStatus(millis());
}

void loop()
{
    const uint32_t now = millis();

    // --- Sensor sampling ---
    static uint32_t last_sensor = 0;
    if (now - last_sensor >= SENSOR_MS)
    {
        last_sensor = now;

        // Always try to read the RTD temperature, even if the MAX31865 fault
        // bits are set. Under PWM these bits are often EMI artefacts, while the
        // temperature reading itself can still be perfectly usable for control.
        const uint8_t f = rtd.readFault();
        if (f)
        {
            rtd.clearFault();
            delay(2);

            const uint8_t f2 = rtd.readFault();
            if (f2)
            {
                ctrl.onSensorFault(f2);
                rtd.clearFault();
            }
            else
            {
                ctrl.onSensorFault(f);
            }
        }

        const float t_meas = rtd.readTempC();
        const float t_cal = calibrateTempC(t_meas);
        ctrl.onSensorReading(t_cal);

        tc.sampleAll();
    }

    // --- Control tick ---
    static uint32_t last_ctrl = 0;
    if (now - last_ctrl >= CONTROL_MS)
    {
        const float dt_s = (now - last_ctrl) / 1000.0f;
        last_ctrl = now;
        char_ctrl.update(now, dt_s);
        ctrl.update(dt_s);
    }

    // --- Streaming ---
    static uint32_t last_stream = 0;
    if (stream_on && (now - last_stream >= STREAM_MS))
    {
        last_stream = now;
        emitTelemetry();
    }

    // --- Serial commands ---
    parser.poll();
}
