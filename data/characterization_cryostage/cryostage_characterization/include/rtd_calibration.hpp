#pragma once

// T_cal = a * T_meas + b
static constexpr float CAL_A = 0.90096545f;
static constexpr float CAL_B = 0.23752064f;

inline float calibrateTempC(float T_meas)
{
    return CAL_A * T_meas + CAL_B;
}
