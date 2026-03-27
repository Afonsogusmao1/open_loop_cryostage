#include "tc_manager.hpp"

static void printFaultFlags_(uint8_t f)
{
  if (!f)
  {
    Serial.print("none");
    return;
  }
  bool first = true;
  auto add = [&](const char *s)
  {
    if (!first)
      Serial.print("|");
    Serial.print(s);
    first = false;
  };

  if (f & MAX31856_FAULT_CJRANGE)
    add("CJ_RANGE");
  if (f & MAX31856_FAULT_TCRANGE)
    add("TC_RANGE");
  if (f & MAX31856_FAULT_CJHIGH)
    add("CJ_HIGH");
  if (f & MAX31856_FAULT_CJLOW)
    add("CJ_LOW");
  if (f & MAX31856_FAULT_TCHIGH)
    add("TC_HIGH");
  if (f & MAX31856_FAULT_TCLOW)
    add("TC_LOW");
  if (f & MAX31856_FAULT_OVUV)
    add("OVUV");
  if (f & MAX31856_FAULT_OPEN)
    add("OPEN");
}

TcManager::TcManager(int cs_T3, int cs_T7, int cs_T12, int cs_Tamb)
    : dev_{
          Adafruit_MAX31856(cs_T3, &SPI),
          Adafruit_MAX31856(cs_T7, &SPI),
          Adafruit_MAX31856(cs_T12, &SPI),
          Adafruit_MAX31856(cs_Tamb, &SPI)}
{
  ch_[0].label = "T3";
  ch_[0].z_mm = 3;
  ch_[0].cs_pin = cs_T3;
  ch_[0].a = TC3.a;
  ch_[0].b = TC3.b;
  ch_[1].label = "T7";
  ch_[1].z_mm = 7;
  ch_[1].cs_pin = cs_T7;
  ch_[1].a = TC7.a;
  ch_[1].b = TC7.b;
  ch_[2].label = "T12";
  ch_[2].z_mm = 12;
  ch_[2].cs_pin = cs_T12;
  ch_[2].a = TC12.a;
  ch_[2].b = TC12.b;
  ch_[3].label = "TAMB";
  ch_[3].z_mm = 0;
  ch_[3].cs_pin = cs_Tamb;
  ch_[3].a = TCAMB.a;
  ch_[3].b = TCAMB.b;
}

void TcManager::begin()
{
  for (int i = 0; i < N; ++i)
    initOne_(i);
}

void TcManager::initOne_(int idx)
{
  auto &c = ch_[idx];
  c.present = dev_[idx].begin();
  if (!c.present)
  {
    Serial.printf("[ERR] %s: MAX31856 begin() failed (CS=%d)\n", c.label, c.cs_pin);
    return;
  }

  dev_[idx].setThermocoupleType(MAX31856_TCTYPE_T);
  dev_[idx].setNoiseFilter(MAX31856_NOISE_FILTER_50HZ);
  dev_[idx].setConversionMode(MAX31856_CONTINUOUS);

  // generous thresholds to avoid false positives; faults still read via readFault()
  dev_[idx].setColdJunctionFaultThreshholds(-40, 85);
  dev_[idx].setTempFaultThreshholds(-200.0f, 400.0f);
}

bool TcManager::isValid_(uint8_t fault, float raw) const
{
  if (fault != 0)
    return false;
  if (!isfinite(raw))
    return false;
  return true;
}

void TcManager::sampleOne_(int idx)
{
  auto &c = ch_[idx];
  if (!c.present)
  {
    c.valid = false;
    c.raw_C = NAN;
    c.cal_C = NAN;
    c.fault = 0xFF;
    return;
  }

  const float raw = dev_[idx].readThermocoupleTemperature();
  const uint8_t fault = dev_[idx].readFault();

  c.fault = fault;
  if (!isValid_(fault, raw))
  {
    c.valid = false;
    c.raw_C = NAN;
    c.cal_C = NAN;
    return;
  }

  c.valid = true;
  c.raw_C = raw;
  c.cal_C = c.a * raw + c.b;
}

void TcManager::sampleAll()
{
  for (int i = 0; i < 4; ++i)
    sampleOne_(i);
}

void TcManager::printStatus()
{
  Serial.println("---- thermocouples ----");
  for (int i = 0; i < 4; ++i)
  {
    const auto &c = ch_[i];
    Serial.printf("%s (z=%dmm, CS=%d): present=%d valid=%d raw=",
                  c.label, c.z_mm, c.cs_pin, c.present ? 1 : 0, c.valid ? 1 : 0);
    if (c.valid)
      Serial.printf("%.4f", c.raw_C);
    else
      Serial.print("nan");

    Serial.print(" cal=");
    if (c.valid)
      Serial.printf("%.4f", c.cal_C);
    else
      Serial.print("nan");

    Serial.print(" fault=0x");
    if (c.fault < 16)
      Serial.print("0");
    Serial.print(c.fault, HEX);
    Serial.print(" [");
    printFaultFlags_(c.fault);
    Serial.println("]");

    Serial.printf("  a=%.8f b=%.8f\n", c.a, c.b);
  }
  Serial.println("----------------------");
}
