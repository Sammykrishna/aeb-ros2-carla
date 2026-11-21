import numpy as np
import pytest

from aeb_bringup.carla_scenario_node import bumper_gap_m, target_speed_mps

BRAKING = dict(type="braking", target_kmh=50, target_decel_mps2=6.0, brake_after_s=1.0)


def test_braking_target_holds_speed_until_brake_time():
    assert target_speed_mps(BRAKING, 0.5) == pytest.approx(50 / 3.6)


def test_braking_target_decelerates_linearly():
    assert target_speed_mps(BRAKING, 2.0) == pytest.approx(50 / 3.6 - 6.0)


def test_braking_target_stops_at_zero():
    assert target_speed_mps(BRAKING, 10.0) == 0.0


def test_moving_target_ignores_time():
    assert target_speed_mps(dict(type="moving", target_kmh=20), 7.0) == pytest.approx(20 / 3.6)


def test_bumper_gap_subtracts_both_half_lengths():
    gap_m, lat_m = bumper_gap_m(np.array([10.0, 0.0]), np.array([1.0, 0.0]), 2.4, 1.9)
    assert gap_m == pytest.approx(5.7) and lat_m == pytest.approx(0.0)


def test_bumper_gap_reports_lateral_offset_on_rotated_heading():
    gap_m, lat_m = bumper_gap_m(np.array([-2.0, 10.0]), np.array([0.0, 1.0]), 0.0, 0.0)
    assert gap_m == pytest.approx(10.0) and lat_m == pytest.approx(2.0)
