from __future__ import annotations

import math

from code_simulation.optimization.velocity_objective import (
    ConstantVelocityObjectiveConfig,
    FrontTrajectory,
    VelocityTrackingSummary,
    direct_speed_error_penalty,
    segment_speed_error_penalty,
    segment_speed_rows,
)


def _tracking_summary(*, achieved_speed_mm_s: float) -> VelocityTrackingSummary:
    return VelocityTrackingSummary(
        target_front_speed_mm_s=0.013,
        control_z_min_mm=2.5,
        control_z_max_mm=11.5,
        t_at_control_z_min_s=0.0,
        t_at_control_z_max_s=1.0,
        expected_t_at_control_z_max_s=1.0,
        actual_interval_speed_mm_s=float(achieved_speed_mm_s),
        regression_speed_mm_s=float(achieved_speed_mm_s),
        tracking_mse=0.0,
        tracking_rmse_mm=0.0,
        tracking_mean_error_mm=0.0,
        tracking_max_abs_error_mm=0.0,
        completion_penalty=0.0,
        reached_control_z_min=True,
        reached_control_z_max=True,
        num_tracking_samples=10,
    )


def test_direct_speed_penalty_has_deadband_tolerance() -> None:
    config = ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=0.013,
        control_z_min_mm=2.5,
        control_z_max_mm=11.5,
        direct_speed_weight=10.0,
        direct_speed_tolerance_pct=2.0,
    )

    penalty, signed_error, error_pct = direct_speed_error_penalty(
        _tracking_summary(achieved_speed_mm_s=0.01313),
        config,
        incomplete_penalty_value=2.0,
    )

    assert penalty == 0.0
    assert math.isclose(signed_error, 0.01)
    assert math.isclose(error_pct, 1.0)


def test_direct_speed_penalty_only_counts_error_outside_tolerance() -> None:
    config = ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=0.013,
        control_z_min_mm=2.5,
        control_z_max_mm=11.5,
        direct_speed_weight=10.0,
        direct_speed_tolerance_pct=2.0,
    )

    penalty, signed_error, error_pct = direct_speed_error_penalty(
        _tracking_summary(achieved_speed_mm_s=0.01456),
        config,
        incomplete_penalty_value=2.0,
    )

    assert math.isclose(signed_error, 0.12)
    assert math.isclose(error_pct, 12.0)
    assert math.isclose(penalty, 0.10 * 0.10)


def test_direct_speed_penalty_marks_missing_speed_as_incomplete() -> None:
    config = ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=0.013,
        control_z_min_mm=2.5,
        control_z_max_mm=11.5,
        direct_speed_weight=10.0,
        direct_speed_tolerance_pct=2.0,
    )

    penalty, signed_error, error_pct = direct_speed_error_penalty(
        _tracking_summary(achieved_speed_mm_s=math.nan),
        config,
        incomplete_penalty_value=2.0,
    )

    assert penalty == 2.0
    assert math.isnan(signed_error)
    assert math.isnan(error_pct)


def test_segment_speed_penalty_catches_nonconstant_velocity_with_correct_global_speed() -> None:
    config = ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=1.0,
        control_z_min_mm=0.0,
        control_z_max_mm=3.0,
        segment_speed_weight=10.0,
        segment_speed_tolerance_pct=0.0,
        segment_speed_num_segments=3,
    )
    front = FrontTrajectory(
        time_s=[0.0, 0.5, 2.0, 3.0],
        time_since_fill_s=[0.0, 0.5, 2.0, 3.0],
        z_front_m=[0.0, 0.001, 0.002, 0.003],
    )

    summary = segment_speed_error_penalty(front, config, incomplete_penalty_value=2.0)
    rows = segment_speed_rows(front, config)

    assert [row["speed_mm_s"] for row in rows] == [2.0, 2.0 / 3.0, 1.0]
    assert summary.segment_speed_num_valid_segments == 3
    assert math.isclose(summary.segment_speed_spread_mm_s, 2.0 - 2.0 / 3.0)
    assert summary.segment_speed_penalty > 0.0


def test_segment_speed_penalty_is_zero_for_constant_segment_speed() -> None:
    config = ConstantVelocityObjectiveConfig(
        target_front_speed_mm_s=1.0,
        control_z_min_mm=0.0,
        control_z_max_mm=3.0,
        segment_speed_weight=10.0,
        segment_speed_tolerance_pct=0.0,
        segment_speed_num_segments=3,
    )
    front = FrontTrajectory(
        time_s=[0.0, 1.0, 2.0, 3.0],
        time_since_fill_s=[0.0, 1.0, 2.0, 3.0],
        z_front_m=[0.0, 0.001, 0.002, 0.003],
    )

    summary = segment_speed_error_penalty(front, config, incomplete_penalty_value=2.0)

    assert summary.segment_speed_num_valid_segments == 3
    assert summary.segment_speed_penalty == 0.0
    assert summary.segment_speed_rmse_pct == 0.0


def test_segment_speed_weight_requires_at_least_two_segments() -> None:
    try:
        ConstantVelocityObjectiveConfig(
            target_front_speed_mm_s=0.013,
            control_z_min_mm=2.5,
            control_z_max_mm=11.5,
            segment_speed_weight=10.0,
            segment_speed_num_segments=1,
        )
    except ValueError as exc:
        assert "segment_speed_num_segments" in str(exc)
    else:
        raise AssertionError("expected ValueError")
