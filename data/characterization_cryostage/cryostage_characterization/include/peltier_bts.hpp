#pragma once
#include <Arduino.h>

class PeltierBTS
{
public:
    PeltierBTS(int rpwm_pin, int en_pin, int ledc_channel,
               int pwm_freq_hz = 20000, int pwm_res_bits = 10);

    void begin();
    void arm(bool on);
    void setDutyPercent(float pct);

    // getters for status/UI
    bool isArmed() const { return armed_; }
    float getDutyPercent() const { return duty_pct_; }

private:
    int rpwm_pin_;
    int en_pin_;
    int ch_;
    int freq_;
    int res_bits_;
    float duty_pct_;
    bool armed_;

    int pctToDuty(float pct) const;
};
