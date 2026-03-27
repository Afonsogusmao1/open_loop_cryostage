#pragma once

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_MAX31856.h>

#include "tc_coeffs.hpp"

class TcManager
{
public:
  static constexpr int N = 4;

  struct ChannelState
  {
    const char *label = nullptr; // "T3", "T7", "T12"
    int z_mm = 0;
    int cs_pin = -1;
    bool present = false;

    // latest sample
    uint8_t fault = 0;
    bool valid = false;
    float raw_C = NAN;
    float cal_C = NAN;

    // applied coefficients
    float a = 1.0f;
    float b = 0.0f;
  };

  TcManager(int cs_T3, int cs_T7, int cs_T12, int cs_Tamb);

  void begin();
  void sampleAll();
  const ChannelState &ch(int idx) const { return ch_[idx]; }
  void printStatus();

private:
  Adafruit_MAX31856 dev_[N];
  ChannelState ch_[N];

  void initOne_(int idx);
  void sampleOne_(int idx);
  bool isValid_(uint8_t fault, float raw) const;
};
