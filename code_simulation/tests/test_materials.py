from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from code_simulation.simulation.materials import (
    RHO_TABLE_ICE,
    T_TABLE_ICE,
    WATER_CONST_DEFAULT,
    water_ice_k_cp_from_tables,
    water_ice_k_cp_rho_from_tables,
)


def test_water_ice_k_cp_rho_from_tables_matches_ice_density_nodes() -> None:
    _k, _cp, rho = water_ice_k_cp_rho_from_tables(T_TABLE_ICE[:-1], Tf=0.0)
    assert_allclose(rho, RHO_TABLE_ICE[:-1])


def test_water_ice_k_cp_rho_from_tables_interpolates_and_clips_ice_density() -> None:
    T_C = np.array([-200.0, -95.0, -37.5], dtype=np.float64)
    _k, _cp, rho = water_ice_k_cp_rho_from_tables(T_C, Tf=0.0)

    expected_rho = np.array(
        [
            925.7,
            0.5 * (925.7 + 924.9),
            0.5 * (920.8 + 920.4),
        ],
        dtype=np.float64,
    )
    assert_allclose(rho, expected_rho)


def test_water_ice_k_cp_rho_from_tables_keeps_liquid_density_constant() -> None:
    T_C = np.array([0.0, 5.0, 20.0, 80.0], dtype=np.float64)
    _k, _cp, rho = water_ice_k_cp_rho_from_tables(T_C, Tf=0.0)
    assert_allclose(rho, WATER_CONST_DEFAULT.rho)


def test_water_ice_k_cp_from_tables_remains_backward_compatible() -> None:
    T_C = np.array([-20.0, -2.5, 0.0, 12.5, 70.0], dtype=np.float64)
    k_old, cp_old = water_ice_k_cp_from_tables(T_C, Tf=0.0)
    k_new, cp_new, _rho = water_ice_k_cp_rho_from_tables(T_C, Tf=0.0)

    assert_allclose(k_old, k_new)
    assert_allclose(cp_old, cp_new)
