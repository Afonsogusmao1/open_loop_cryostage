#pragma once
#include <Arduino.h>
#include <Adafruit_MAX31865.h>

class RtdMax31865
{
public:
    enum class WireMode
    {
        W2,
        W3,
        W4
    };

    enum class MainsFilter
    {
        Hz50,
        Hz60
    };

    RtdMax31865(int cs_pin, float rnominal, float rref);

    void begin(WireMode mode, MainsFilter filter = MainsFilter::Hz60);
    float readTempC(); // temperatura "crua" em °C (da lib)
    uint8_t readFault();
    void clearFault(); // faults

private:
    Adafruit_MAX31865 max_;
    float rnominal_;
    float rref_;
};
