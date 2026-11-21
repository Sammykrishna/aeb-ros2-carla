from dataclasses import replace

import numpy as np
import pytest

from aeb_controller.state_machine import (AebParams, AebState, AebStateMachine, compute_ttc, lead_range_m,
                                          required_decel_mps2)

P = AebParams()
V50 = 50 / 3.6


def settle(sm: AebStateMachine, ego_mps: float, range_m: float | None, closing_mps: float, t0_s: float = 0.0,
           secs: float = 0.4, target_decel_mps2: float = 0.0, dt_s: float = 0.02):
    out = None
    for i in range(int(secs / dt_s) + 1):
        out = sm.step(t0_s + i * dt_s, ego_mps, range_m, closing_mps, target_decel_mps2)
    return out


def test_ttc_inf_when_not_closing():
    assert compute_ttc(30.0, 0.0) == float("inf")
    assert compute_ttc(30.0, -5.0) == float("inf")


def test_ttc_value():
    assert compute_ttc(20.0, 10.0) == 2.0


def test_lead_range_picks_nearest_in_corridor():
    pts_xy_m = np.array([[30.0, 0.5], [12.0, 3.0], [18.0, -1.0], [-5.0, 0.0]])
    assert lead_range_m(pts_xy_m) == 18.0


def test_lead_range_none_when_empty():
    assert lead_range_m(np.empty((0, 2))) is None


def test_no_decel_needed_when_not_closing():
    assert required_decel_mps2(5.0, 0.0, V50, 0.0, P) == 0.0


def test_required_decel_matches_stopping_distance_plus_margins():
    eff_range_m = 30.0 - V50 * P.latency_s - P.margin_m
    assert required_decel_mps2(30.0, V50, V50, 0.0, P) == pytest.approx(V50 ** 2 / (2 * eff_range_m))


def test_braking_target_raises_requirement_only_when_it_matters():
    plain = required_decel_mps2(15.0, 10.0, V50, 0.0, P)
    braking = required_decel_mps2(15.0, 10.0, V50, 3.0, P)
    assert braking > plain
    assert required_decel_mps2(15.0, 10.0, V50, 0.2, P) == plain


def test_idle_when_far_and_escalation_ladder_for_stationary_target():
    sm = AebStateMachine()
    assert settle(sm, V50, 90.0, V50)[0] == AebState.IDLE
    assert settle(sm, V50, 55.0, V50, t0_s=0.5)[0] == AebState.WARNING
    assert settle(sm, V50, 34.0, V50, t0_s=1.0)[0] == AebState.PARTIAL_BRAKE
    assert settle(sm, V50, 28.0, V50, t0_s=1.5)[0] == AebState.FULL_BRAKE


def test_full_brake_commanded_early_enough_to_stop_from_50kmh():
    sm = AebStateMachine()
    gap_m, t_s, state = 90.0, 0.0, AebState.IDLE
    while state != AebState.FULL_BRAKE:
        gap_m, t_s = gap_m - V50 * 0.04, t_s + 0.04
        state = sm.step(t_s, V50, gap_m, V50)[0]
    assert gap_m > V50 ** 2 / (2 * P.max_decel_mps2) + 2.0


def test_single_sample_above_threshold_does_not_start_braking():
    sm = AebStateMachine()
    assert settle(sm, V50, 90.0, V50, secs=0.2)[0] == AebState.IDLE
    assert sm.step(0.30, V50, 34.0, V50)[0] == AebState.WARNING
    assert settle(sm, V50, 34.0, V50, t0_s=0.32)[0] == AebState.PARTIAL_BRAKE


def test_demand_beyond_the_cars_limit_bypasses_the_debounce():
    assert AebStateMachine().step(0.0, V50, 12.0, V50)[0] == AebState.FULL_BRAKE


def test_partial_brake_level_follows_demand_through_the_measured_pedal_map():
    sm = AebStateMachine()
    st, brake, _, a_req_mps2 = settle(sm, V50, 34.0, V50)
    assert st == AebState.PARTIAL_BRAKE
    assert brake == pytest.approx((P.partial_decel_margin * a_req_mps2 - P.brake_offset_mps2) / P.brake_gain_mps2)
    assert P.min_partial_brake < brake < 1.0


def test_partial_brake_decays_slowly_as_demand_falls():
    sm = AebStateMachine()
    brake_a = settle(sm, V50, 34.0, V50)[1]
    brake_b = settle(sm, V50, 80.0, V50, t0_s=0.5, secs=0.5)[1]
    assert brake_b == pytest.approx(brake_a - P.brake_decay_per_s * 0.6, abs=0.03)


def test_partial_brake_never_drops_when_decay_is_off():
    sm = AebStateMachine(replace(P, brake_decay_per_s=0.0))
    brake_a = settle(sm, V50, 34.0, V50)[1]
    assert settle(sm, V50, 80.0, V50, t0_s=0.5, secs=0.5)[1] == brake_a


def test_partial_brake_decay_stops_at_the_minimum_level():
    sm = AebStateMachine(replace(P, release_s=100.0))
    settle(sm, V50, 34.0, V50)
    assert settle(sm, V50, 80.0, V50, t0_s=0.5, secs=3.0)[1] == P.min_partial_brake


def test_brake_latches_until_stopped_then_holds_then_releases():
    sm = AebStateMachine()
    sm.step(0.0, V50, 12.0, V50)
    st, brake, _, _ = sm.step(0.1, 10.0, None, 0.0)
    assert st == AebState.FULL_BRAKE and brake == 1.0
    assert sm.step(0.2, 0.0, 5.0, 0.0)[0] == AebState.HOLD
    assert sm.step(1.0, 0.0, None, 0.0)[0] == AebState.HOLD
    assert sm.step(2.5, 0.0, None, 0.0)[0] == AebState.IDLE


def test_short_spike_in_target_decel_is_ignored():
    plain = required_decel_mps2(30.0, 8.3, V50, 0.0, P)
    spike = settle(AebStateMachine(), V50, 30.0, 8.3, secs=0.3, target_decel_mps2=3.0)[3]
    assert spike == pytest.approx(plain)


def test_sustained_target_decel_is_used():
    plain = required_decel_mps2(30.0, 8.3, V50, 0.0, P)
    sustained = settle(AebStateMachine(), V50, 30.0, 8.3, secs=0.7, target_decel_mps2=3.0)[3]
    assert sustained > plain


def test_strong_target_decel_is_used_immediately():
    plain = required_decel_mps2(30.0, 8.3, V50, 0.0, P)
    assert AebStateMachine().step(0.0, V50, 30.0, 8.3, 6.0)[3] > plain


def test_target_decel_held_after_braking_starts():
    sm = AebStateMachine()
    assert settle(sm, V50, 18.0, 3.0, secs=0.3, target_decel_mps2=6.0)[0] in (AebState.PARTIAL_BRAKE, AebState.FULL_BRAKE)
    settle(sm, V50 - 1, 17.0, 2.0, t0_s=0.5, secs=0.2, target_decel_mps2=0.0)
    assert sm.target_decel_mps2 == 6.0


def test_warning_does_not_flicker_on_a_single_low_sample():
    sm = AebStateMachine()
    assert sm.step(0.00, V50, 55.0, V50)[0] == AebState.WARNING
    assert sm.step(0.05, V50, 90.0, V50)[0] == AebState.WARNING
    assert sm.step(0.60, V50, 90.0, V50)[0] == AebState.IDLE


def test_brake_releases_after_the_threat_has_been_gone_for_a_while():
    sm = AebStateMachine()
    sm.step(0.0, V50, 12.0, V50)
    assert settle(sm, 8.0, None, 0.0, t0_s=0.1, secs=0.8)[0] in (AebState.PARTIAL_BRAKE, AebState.FULL_BRAKE)
    st, brake, _, _ = settle(sm, 8.0, None, 0.0, t0_s=0.9, secs=0.4)
    assert st == AebState.IDLE and brake == 0.0


def test_brake_does_not_release_while_the_threat_is_still_there():
    sm = AebStateMachine()
    sm.step(0.0, V50, 12.0, V50)
    assert settle(sm, 8.0, 30.0, 8.0, t0_s=0.1, secs=2.0)[0] in (AebState.PARTIAL_BRAKE, AebState.FULL_BRAKE)


def test_can_brake_again_after_a_release():
    sm = AebStateMachine()
    sm.step(0.0, V50, 12.0, V50)
    settle(sm, 8.0, None, 0.0, t0_s=0.1, secs=1.5)
    assert sm.state == AebState.IDLE
    assert sm.step(2.0, V50, 12.0, V50)[0] == AebState.FULL_BRAKE


def test_full_brake_needs_the_demand_sustained_unless_it_exceeds_the_cars_limit():
    sm = AebStateMachine()
    assert settle(sm, V50, 28.0, V50, secs=0.2)[0] == AebState.PARTIAL_BRAKE
    assert settle(sm, V50, 28.0, V50, t0_s=0.22, secs=0.2)[0] == AebState.FULL_BRAKE


def test_a_brief_overshoot_past_the_full_threshold_does_not_latch_full_brake():
    sm = AebStateMachine()
    settle(sm, V50, 34.0, V50, secs=0.4)
    assert sm.step(0.42, V50, 29.0, V50)[0] == AebState.PARTIAL_BRAKE
    assert settle(sm, V50, 34.0, V50, t0_s=0.44, secs=0.4)[0] == AebState.PARTIAL_BRAKE


def test_full_brake_steps_down_to_partial_when_the_demand_falls():
    sm = AebStateMachine()
    sm.step(0.0, V50, 12.0, V50)
    assert sm.state == AebState.FULL_BRAKE
    assert settle(sm, 8.0, 30.0, 8.0, t0_s=0.1, secs=0.2)[0] == AebState.FULL_BRAKE
    st, brake, _, _ = settle(sm, 8.0, 30.0, 8.0, t0_s=0.4, secs=0.5)
    assert st == AebState.PARTIAL_BRAKE and brake < 1.0
