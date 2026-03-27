#include "peltier_bts.hpp"

PeltierBTS::PeltierBTS(int rpwm_pin, int en_pin, int ledc_channel,
                       int pwm_freq_hz, int pwm_res_bits)
    : rpwm_pin_(rpwm_pin),
      en_pin_(en_pin),
      ch_(ledc_channel),
      freq_(pwm_freq_hz),
      res_bits_(pwm_res_bits),
      duty_pct_(0.0f),
      armed_(false) {}

int PeltierBTS::pctToDuty(float pct) const
{
    pct = constrain(pct, 0.0f, 100.0f);
    int maxDuty = (1 << res_bits_) - 1;
    return (int)lround((pct / 100.0f) * maxDuty);
}

void PeltierBTS::begin()
{
    pinMode(en_pin_, OUTPUT);
    digitalWrite(en_pin_, LOW);
    armed_ = false;

    ledcSetup(ch_, freq_, res_bits_);
    ledcAttachPin(rpwm_pin_, ch_);
    ledcWrite(ch_, 0);
    duty_pct_ = 0.0f;
}

void PeltierBTS::arm(bool on)
{
    if (!on)
    {
        ledcWrite(ch_, 0);
        duty_pct_ = 0.0f;
        digitalWrite(en_pin_, LOW);
        armed_ = false;
        return;
    }
    digitalWrite(en_pin_, HIGH);
    armed_ = true;
    ledcWrite(ch_, pctToDuty(duty_pct_));
}

void PeltierBTS::setDutyPercent(float pct)
{
    duty_pct_ = constrain(pct, 0.0f, 100.0f);
    if (armed_)
        ledcWrite(ch_, pctToDuty(duty_pct_));
    else
        ledcWrite(ch_, 0);
}
