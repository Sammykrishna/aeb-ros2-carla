import numpy as np
import pytest

from aeb_controller.tracking import RangeTracker

DT_S = 0.05


def feed(tracker: RangeTracker, ranges_m: np.ndarray) -> float:
    for i, r in enumerate(ranges_m):
        tracker.update(i * DT_S, float(r))
    return (len(ranges_m) - 1) * DT_S


def test_stationary_target_closing_speed_is_quiet_despite_noise():
    rng = np.random.default_rng(0)
    t_s = np.arange(0, 1.5, DT_S)
    tracker = RangeTracker()
    now_s = feed(tracker, 60.0 - 13.9 * t_s + rng.normal(0, 0.1, len(t_s)))
    range_m, closing_mps, _ = tracker.estimate(now_s)
    assert closing_mps == pytest.approx(13.9, abs=0.6)
    assert range_m == pytest.approx(60.0 - 13.9 * now_s, abs=0.3)


def test_braking_target_shows_positive_closing_acceleration():
    rng = np.random.default_rng(1)
    t_s = np.arange(0, 3.0, DT_S)
    ranges_m = 40.0 - 0.5 * 2.0 * np.clip(t_s - 1.0, 0.0, None) ** 2 + rng.normal(0, 0.05, len(t_s))
    tracker = RangeTracker()
    now_s = feed(tracker, ranges_m)
    _, closing_mps, accel_mps2 = tracker.estimate(now_s)
    assert closing_mps == pytest.approx(4.0, abs=1.0)
    assert 0.8 < accel_mps2 < 3.5


def test_extrapolates_to_now_to_cover_pipeline_latency():
    tracker = RangeTracker()
    now_s = feed(tracker, 50.0 - 10.0 * np.arange(0, 1.0, DT_S))
    range_m, _, _ = tracker.estimate(now_s + 0.1)
    assert range_m == pytest.approx(50.0 - 10.0 * (now_s + 0.1), abs=0.05)


def test_needs_enough_history():
    tracker = RangeTracker()
    now_s = feed(tracker, np.array([30.0, 29.5, 29.0]))
    assert tracker.estimate(now_s) is None


def test_stale_track_returns_none_and_gap_resets_history():
    tracker = RangeTracker()
    now_s = feed(tracker, 50.0 - 10.0 * np.arange(0, 1.0, DT_S))
    assert tracker.estimate(now_s + 0.5) is None
    tracker.update(now_s + 2.0, 20.0)
    assert len(tracker.t_s) == 1


def test_range_jump_resets_track():
    tracker = RangeTracker()
    now_s = feed(tracker, 50.0 - 10.0 * np.arange(0, 1.0, DT_S))
    tracker.update(now_s + DT_S, 12.0)
    assert len(tracker.t_s) == 1
