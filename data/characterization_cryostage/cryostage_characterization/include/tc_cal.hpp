#pragma once

#include <Arduino.h>

// Two-point linear calibration:
//   T_cal = a * T_raw + b

struct TcCalPoint {
  bool valid = false;
  float Tref_C = NAN;
  float raw_C = NAN;
};

struct TcLinearCal {
  float a = 1.0f;
  float b = 0.0f;
  TcCalPoint p_low;
  TcCalPoint p_high;

  void clearAll() {
    a = 1.0f;
    b = 0.0f;
    p_low = TcCalPoint{};
    p_high = TcCalPoint{};
  }

  float apply(float raw_C) const { return a * raw_C + b; }

  bool canFit() const {
    if (!p_low.valid || !p_high.valid) return false;
    if (!isfinite(p_low.raw_C) || !isfinite(p_high.raw_C)) return false;
    if (!isfinite(p_low.Tref_C) || !isfinite(p_high.Tref_C)) return false;
    return fabsf(p_high.raw_C - p_low.raw_C) > 1e-6f;
  }

  bool fit() {
    if (!canFit()) return false;
    const float dT = p_high.Tref_C - p_low.Tref_C;
    const float dR = p_high.raw_C - p_low.raw_C;
    a = dT / dR;
    b = p_low.Tref_C - a * p_low.raw_C;
    return isfinite(a) && isfinite(b);
  }

  // Store a calibration point in low/high slots.
  // Keeps them sorted by Tref.
  void storePoint(float Tref_C, float rawAvg_C) {
    TcCalPoint p;
    p.valid = true;
    p.Tref_C = Tref_C;
    p.raw_C = rawAvg_C;

    // Overwrite if Tref is close.
    if (p_low.valid && fabsf(Tref_C - p_low.Tref_C) < 0.5f) {
      p_low = p;
      sortByTref_();
      return;
    }
    if (p_high.valid && fabsf(Tref_C - p_high.Tref_C) < 0.5f) {
      p_high = p;
      sortByTref_();
      return;
    }

    if (!p_low.valid && !p_high.valid) {
      p_low = p;
      return;
    }
    if (!p_low.valid) {
      p_low = p;
      sortByTref_();
      return;
    }
    if (!p_high.valid) {
      p_high = p;
      sortByTref_();
      return;
    }

    // Replace the closest in Tref.
    const float dL = fabsf(Tref_C - p_low.Tref_C);
    const float dH = fabsf(Tref_C - p_high.Tref_C);
    if (dL <= dH) p_low = p;
    else p_high = p;

    sortByTref_();
  }

private:
  void sortByTref_() {
    if (p_low.valid && p_high.valid && p_low.Tref_C > p_high.Tref_C) {
      TcCalPoint tmp = p_low;
      p_low = p_high;
      p_high = tmp;
    }
  }
};
