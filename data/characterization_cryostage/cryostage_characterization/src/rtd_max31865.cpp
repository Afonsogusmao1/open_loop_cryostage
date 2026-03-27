#include "rtd_max31865.hpp"

RtdMax31865::RtdMax31865(int cs_pin, float rnominal, float rref)
    : max_(cs_pin), rnominal_(rnominal), rref_(rref) {}

void RtdMax31865::begin(WireMode mode, MainsFilter filter)
{
    switch (mode)
    {
    case WireMode::W2:
        max_.begin(MAX31865_2WIRE);
        break;
    case WireMode::W3:
        max_.begin(MAX31865_3WIRE);
        break;
    case WireMode::W4:
        max_.begin(MAX31865_4WIRE);
        break;
    }

    max_.enable50Hz(filter == MainsFilter::Hz50);
}

float RtdMax31865::readTempC()
{
    return max_.temperature(rnominal_, rref_);
}

uint8_t RtdMax31865::readFault()
{
    return max_.readFault();
}

void RtdMax31865::clearFault()
{
    max_.clearFault();
}
