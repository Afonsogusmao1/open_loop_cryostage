// include/tc_coeffs.hpp
#pragma once

struct TcCoeff
{
    float a;
    float b;
};

// Replace these with your calibrated coefficients:
static constexpr TcCoeff TC3 = {0.93444306f, 1.07136786f};  // z=3mm
static constexpr TcCoeff TC7 = {0.93842745f, 1.01716721f};  // z=7mm
static constexpr TcCoeff TC12 = {0.95224917f, 0.68714428f}; // z=12mm
static constexpr TcCoeff TCAMB = {1.000000f, 0.000000f};    // ambient (Tamb)