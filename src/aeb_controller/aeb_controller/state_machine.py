from dataclasses import dataclass
from enum import Enum

import numpy as np


class AebState(str, Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    PARTIAL_BRAKE = "PARTIAL_BRAKE"
    FULL_BRAKE = "FULL_BRAKE"
    HOLD = "HOLD"


@dataclass
class AebParams:
    max_decel_mps2: float = 4.5
    warn_ratio: float = 0.3
    partial_ratio: float = 0.65
    full_ratio: float = 0.9
    min_partial_brake: float = 0.2
    full_brake: float = 1.0
    brake_gain_mps2: float = 3.83
    brake_offset_mps2: float = 0.67
    partial_decel_margin: float = 1.1
    brake_decay_per_s: float = 0.6
    latency_s: float = 0.2
    margin_m: float = 3.0
    min_target_decel_mps2: float = 2.0
    strong_target_decel_mps2: float = 4.0
    target_decel_confirm_s: float = 0.5
    confirm_s: float = 0.1
    full_confirm_s: float = 0.3
    full_release_s: float = 0.3
    urgent_ratio: float = 1.0
    release_ratio: float = 0.25
    release_s: float = 1.0
    min_closing_speed_mps: float = 0.5
    min_dwell_s: float = 0.4
    stopped_speed_mps: float = 0.1
    hold_time_s: float = 2.0


ESCALATION = [AebState.IDLE, AebState.WARNING, AebState.PARTIAL_BRAKE, AebState.FULL_BRAKE]
FIRST_BRAKING = ESCALATION.index(AebState.PARTIAL_BRAKE)
FULL_BRAKE_IDX = ESCALATION.index(AebState.FULL_BRAKE)
BRAKING = (AebState.PARTIAL_BRAKE, AebState.FULL_BRAKE)


def compute_ttc(range_m: float | None, closing_mps: float, min_closing_mps: float = 0.5) -> float:
    if range_m is None or closing_mps < min_closing_mps:
        return float("inf")
    return max(range_m, 0.0) / closing_mps


def lead_range_m(pts_xy_m: np.ndarray, half_width_m: float = 1.5) -> float | None:
    ahead_x_m = pts_xy_m[(pts_xy_m[:, 0] > 0.0) & (np.abs(pts_xy_m[:, 1]) < half_width_m), 0]
    return float(ahead_x_m.min()) if ahead_x_m.size else None


def required_decel_mps2(range_m: float, closing_mps: float, ego_speed_mps: float,
                        target_decel_mps2: float, p: AebParams) -> float:
    if closing_mps < p.min_closing_speed_mps:
        return 0.0
    eff_range_m = max(range_m - closing_mps * p.latency_s - p.margin_m, 0.1)
    rel_decel_mps2 = closing_mps ** 2 / (2.0 * eff_range_m)
    if target_decel_mps2 < p.min_target_decel_mps2:
        return rel_decel_mps2
    # a braking target keeps rolling until it stops, which buys the ego that extra distance
    target_stop_m = max(ego_speed_mps - closing_mps, 0.0) ** 2 / (2.0 * target_decel_mps2)
    return max(rel_decel_mps2, ego_speed_mps ** 2 / (2.0 * (eff_range_m + target_stop_m)))


class AebStateMachine:
    def __init__(self, params: AebParams | None = None):
        self.p = params or AebParams()
        self.reset()

    def reset(self):
        self.state = AebState.IDLE
        self.since_s = 0.0
        self.hold_until_s = 0.0
        self.brake_cmd = 0.0
        self.target_decel_mps2 = 0.0
        self.decel_since_s: float | None = None
        self.low_since_s: float | None = None
        self.full_low_since_s: float | None = None
        self.wanted_since_s: dict[int, float] = {}
        self.prev_step_s = 0.0

    def _set(self, state: AebState, now_s: float):
        if state != self.state:
            self.state, self.since_s = state, now_s

    def _confirmed_target_decel(self, now_s: float, est_mps2: float) -> float:
        p = self.p
        if est_mps2 < p.min_target_decel_mps2:
            self.decel_since_s = None
            return 0.0
        self.decel_since_s = now_s if self.decel_since_s is None else self.decel_since_s
        # the estimate is noisy enough that a short spike looks like a braking target
        strong = est_mps2 >= p.strong_target_decel_mps2
        return est_mps2 if strong or now_s - self.decel_since_s >= p.target_decel_confirm_s else 0.0

    def _wanted(self, ratio: float) -> AebState:
        p = self.p
        if ratio >= p.full_ratio:
            return AebState.FULL_BRAKE
        if ratio >= p.partial_ratio:
            return AebState.PARTIAL_BRAKE
        return AebState.WARNING if ratio >= p.warn_ratio else AebState.IDLE

    def _debounced(self, now_s: float, wanted: AebState, ratio: float) -> AebState:
        p = self.p
        idx = ESCALATION.index(wanted)
        for level in range(FIRST_BRAKING, len(ESCALATION)):
            if idx >= level:
                self.wanted_since_s.setdefault(level, now_s)
            else:
                self.wanted_since_s.pop(level, None)
        if ratio >= p.urgent_ratio:
            return AebState.FULL_BRAKE
        if idx < FIRST_BRAKING:
            return wanted
        needed_s = {FULL_BRAKE_IDX: p.full_confirm_s}
        held = [level for level, since_s in self.wanted_since_s.items()
                if now_s - since_s >= needed_s.get(level, p.confirm_s) - 1e-9]
        return ESCALATION[max(held)] if held else AebState.WARNING

    def _step_down_from_full(self, now_s: float, ratio: float):
        p = self.p
        if self.state != AebState.FULL_BRAKE or ratio >= p.partial_ratio:
            self.full_low_since_s = None
            return
        self.full_low_since_s = now_s if self.full_low_since_s is None else self.full_low_since_s
        if now_s - self.full_low_since_s >= p.full_release_s:
            self._set(AebState.PARTIAL_BRAKE, now_s)
            self.full_low_since_s = None
            self.wanted_since_s.pop(FULL_BRAKE_IDX, None)

    def _release_if_threat_gone(self, now_s: float, ratio: float):
        p = self.p
        if ratio >= p.release_ratio:
            self.low_since_s = None
            return
        self.low_since_s = now_s if self.low_since_s is None else self.low_since_s
        if now_s - self.low_since_s >= p.release_s:
            self._set(AebState.IDLE, now_s)
            self.target_decel_mps2, self.low_since_s = 0.0, None
            self.wanted_since_s.clear()

    def step(self, now_s: float, ego_speed_mps: float, range_m: float | None, closing_mps: float,
             target_decel_mps2: float = 0.0) -> tuple[AebState, float, float, float]:
        p = self.p
        ttc_s = compute_ttc(range_m, closing_mps, p.min_closing_speed_mps)
        dt_s, self.prev_step_s = now_s - self.prev_step_s, now_s

        if self.state == AebState.HOLD:
            if now_s >= self.hold_until_s:
                self._set(AebState.IDLE, now_s)
            return self.state, p.full_brake, ttc_s, 0.0

        braking = self.state in BRAKING
        if braking and ego_speed_mps <= p.stopped_speed_mps:
            self._set(AebState.HOLD, now_s)
            self.hold_until_s = now_s + p.hold_time_s
            return self.state, p.full_brake, ttc_s, 0.0

        confirmed_mps2 = self._confirmed_target_decel(now_s, target_decel_mps2)
        # our own braking shrinks the measured closing acceleration, so keep the confirmed value from before it started
        self.target_decel_mps2 = max(self.target_decel_mps2, confirmed_mps2) if braking else confirmed_mps2
        a_req_mps2 = 0.0 if range_m is None else required_decel_mps2(
            range_m, closing_mps, ego_speed_mps, self.target_decel_mps2, p)
        ratio = a_req_mps2 / p.max_decel_mps2

        if braking:
            self._release_if_threat_gone(now_s, ratio)
            self._step_down_from_full(now_s, ratio)
            braking = self.state in BRAKING

        target = self._debounced(now_s, self._wanted(ratio), ratio)
        escalating = ESCALATION.index(target) > ESCALATION.index(self.state)
        dwelling = now_s - self.since_s < p.min_dwell_s
        # once braking, only escalate until released or stopped; below that, dwell to stop threshold flicker
        if escalating or (not braking and not dwelling):
            self._set(target, now_s)

        if self.state in BRAKING:
            demand = (p.partial_decel_margin * a_req_mps2 - p.brake_offset_mps2) / p.brake_gain_mps2
            if self.state == AebState.FULL_BRAKE:
                self.brake_cmd = p.full_brake
            else:
                # follow the demand down slowly: a latched peak from a noisy estimate keeps braking harder than needed
                decayed = self.brake_cmd - p.brake_decay_per_s * max(dt_s, 0.0)
                self.brake_cmd = max(float(np.clip(demand, p.min_partial_brake, p.full_brake)), decayed)
        else:
            self.brake_cmd = 0.0
        return self.state, self.brake_cmd, ttc_s, a_req_mps2
