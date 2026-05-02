from __future__ import annotations

import numpy as np

from code_simulation.simulation.cryostage_model import (
    CharacterizationRun,
    CryostageModelParams,
    fit_first_order_step_response_model,
    fit_first_order_temperature_lookup_model,
    load_characterization_run,
    simulate_plate_temperature,
)


def test_load_characterization_run_trims_to_active_power(tmp_path):
    path = tmp_path / "cryostage_characterization_min10_I.csv"
    path.write_text(
        "\n".join(
            [
                "# metadata: ignored",
                "row_type,panel_t_s,set,T_cal,power",
                "telemetry,10.0,-10.0,9.0,0.0",
                "telemetry,11.0,-10.0,9.0,0.0",
                "telemetry,12.0,-10.0,8.8,2.0",
                "telemetry,13.0,-10.0,8.3,3.0",
            ]
        ),
        encoding="utf-8",
    )

    run = load_characterization_run(path, active_power_threshold=1.0)

    np.testing.assert_allclose(run.time_s, [0.0, 1.0])
    np.testing.assert_allclose(run.T_ref_C, [-10.0, -10.0])
    np.testing.assert_allclose(run.T_plate_C, [8.8, 8.3])


def test_step_response_fit_recovers_synthetic_first_order_params():
    tau_s = 42.0
    gain = 1.08
    offset_C = 0.65
    time_s = np.linspace(0.0, 240.0, 121)
    runs = []
    for target_C in (-5.0, -10.0, -15.0, -20.0):
        T0_C = 9.0 + 0.02 * abs(target_C)
        steady_C = gain * target_C + offset_C
        h = np.exp(-time_s / tau_s)
        T_plate_C = h * T0_C + (1.0 - h) * steady_C
        runs.append(
            CharacterizationRun(
                name=f"synthetic_{target_C:g}",
                time_s=time_s,
                T_ref_C=np.full_like(time_s, target_C),
                T_plate_C=T_plate_C,
            )
        )

    fitted = fit_first_order_step_response_model(
        runs,
        tau_bounds_s=(20.0, 80.0),
        num_tau=240,
        num_refinement_tau=300,
    )

    assert abs(fitted.tau_s - tau_s) < 0.1
    assert abs(fitted.gain - gain) < 1e-3
    assert abs(fitted.offset_C - offset_C) < 1e-3


def test_temperature_lookup_fit_recovers_setpoint_dependent_response():
    time_s = np.linspace(0.0, 220.0, 111)
    expected = {
        -20.0: (70.0, -19.8),
        -10.0: (95.0, -9.6),
        -5.0: (105.0, -4.7),
    }
    runs = []
    for target_C, (tau_s, steady_C) in expected.items():
        T0_C = 8.5
        h = np.exp(-time_s / tau_s)
        runs.append(
            CharacterizationRun(
                name=f"synthetic_{target_C:g}",
                time_s=time_s,
                T_ref_C=np.full_like(time_s, target_C),
                T_plate_C=h * T0_C + (1.0 - h) * steady_C,
            )
        )

    fitted = fit_first_order_temperature_lookup_model(
        runs,
        tau_bounds_s=(40.0, 140.0),
        num_tau=500,
    )

    assert fitted.reference_temperatures_C == (-20.0, -10.0, -5.0)
    for reference_C, expected_values in expected.items():
        tau_s, steady_C = expected_values
        assert abs(fitted.response_tau_for_reference_C(reference_C) - tau_s) < 0.2
        assert abs(fitted.steady_plate_for_reference_C(reference_C) - steady_C) < 0.02


def test_temperature_lookup_simulation_uses_interpolated_dynamics():
    params = CryostageModelParams(
        tau_s=10.0,
        gain=1.0,
        offset_C=0.0,
        reference_temperatures_C=(-20.0, -10.0),
        response_tau_s=(20.0, 40.0),
        steady_plate_C=(-19.0, -9.0),
    )
    time_s = np.asarray([0.0, 2.0])
    result = simulate_plate_temperature(time_s, lambda _t: -15.0, params, T_plate0_C=1.0)

    tau_s = 30.0
    steady_C = -14.0
    alpha = np.exp(-2.0 / tau_s)
    expected = alpha * 1.0 + (1.0 - alpha) * steady_C
    np.testing.assert_allclose(result, [1.0, expected])
