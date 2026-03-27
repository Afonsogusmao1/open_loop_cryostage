# materials.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SolidProps:
    rho: float  # kg/m3
    cp: float   # J/kg/K
    k: float    # W/m/K


# ---- Baseline constants (keep baseline behavior unchanged) ----
PLA_DEFAULT = SolidProps(rho=1250.0, cp=1800.0, k=0.13)
AIR_DEFAULT = SolidProps(rho=1.2, cp=1005.0, k=0.026)
WATER_CONST_DEFAULT = SolidProps(rho=1000.0, cp=4180.0, k=0.6)


# ---- Tabulated water/ice properties (°C) ----
# Ice (descending to 0)
T_TABLE_ICE = np.array(
    [-100.0, -90.0, -80.0, -70.0, -60.0,
     -50.0, -40.0, -35.0, -30.0, -25.0,
     -20.0, -15.0, -10.0,  -5.0,   0.0],
    dtype=np.float64
)

K_TABLE_ICE = np.array(
    [3.48, 3.34, 3.19, 3.05, 2.90,
     2.76, 2.63, 2.57, 2.50, 2.45,
     2.39, 2.34, 2.30, 2.25, 2.22],
    dtype=np.float64
)

CP_TABLE_ICE = 1000.0 * np.array(  # kJ/kg/K -> J/kg/K
    [1.389, 1.463, 1.536, 1.609, 1.681,
     1.751, 1.818, 1.851, 1.882, 1.913,
     1.943, 1.972, 2.000, 2.027, 2.050],
    dtype=np.float64
)

# Liquid water (ascending)
T_TABLE_LIQ = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0], dtype=np.float64)

CP_TABLE_LIQ = np.array(
    [4217.0, 4204.0, 4193.0, 4185.0, 4182.0, 4181.0, 4179.0, 4179.0, 4184.0],
    dtype=np.float64
)

K_TABLE_LIQ = np.array(
    [0.566, 0.571, 0.580, 0.588, 0.598, 0.606, 0.614, 0.631, 0.651],
    dtype=np.float64
)


def interp_clipped(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    np.interp with clipping to xp bounds (no extrapolation beyond ends).
    xp must be ascending.
    """
    x = np.asarray(x, dtype=np.float64)
    x_clip = np.clip(x, float(xp[0]), float(xp[-1]))
    return np.interp(x_clip, xp, fp)


def water_ice_k_cp_from_tables(T_C: np.ndarray, Tf: float) -> tuple[np.ndarray, np.ndarray]:
    """
    k(T), cp(T) by table lookup with phase selection:
      - ice tables when T < Tf
      - liquid tables when T >= Tf

    Temperatures are in °C.
    """
    T_C = np.asarray(T_C, dtype=np.float64)

    is_ice = T_C < Tf
    is_liq = ~is_ice

    k = np.empty_like(T_C, dtype=np.float64)
    cp = np.empty_like(T_C, dtype=np.float64)

    if np.any(is_ice):
        k[is_ice] = interp_clipped(T_C[is_ice], T_TABLE_ICE, K_TABLE_ICE)
        cp[is_ice] = interp_clipped(T_C[is_ice], T_TABLE_ICE, CP_TABLE_ICE)

    if np.any(is_liq):
        k[is_liq] = interp_clipped(T_C[is_liq], T_TABLE_LIQ, K_TABLE_LIQ)
        cp[is_liq] = interp_clipped(T_C[is_liq], T_TABLE_LIQ, CP_TABLE_LIQ)

    return k, cp


def apparent_cp_bump(T_C: np.ndarray, Tf: float, L_latent: float, dT_mushy: float) -> np.ndarray:
    """
    Apparent heat capacity bump for latent heat:
      cp_eff = cp + L/dT_mushy  within [Tf - dT_mushy/2, Tf + dT_mushy/2]
    """
    T_C = np.asarray(T_C, dtype=np.float64)
    bump = np.zeros_like(T_C, dtype=np.float64)

    half = 0.5 * dT_mushy
    in_band = (T_C >= (Tf - half)) & (T_C <= (Tf + half))
    bump[in_band] = (L_latent / dT_mushy)
    return bump
