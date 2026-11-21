from collections import deque

import numpy as np


class RangeTracker:
    def __init__(self, window_s: float = 1.0, stale_s: float = 0.3, min_samples: int = 5, jump_m: float = 5.0):
        self.window_s, self.stale_s, self.min_samples, self.jump_m = window_s, stale_s, min_samples, jump_m
        self.t_s: deque[float] = deque()
        self.range_m: deque[float] = deque()

    def reset(self):
        self.t_s.clear()
        self.range_m.clear()

    def update(self, t_s: float, range_m: float):
        if self.t_s and (t_s - self.t_s[-1] > self.stale_s or abs(range_m - self.range_m[-1]) > self.jump_m):
            self.reset()
        if self.t_s and t_s <= self.t_s[-1]:
            return
        self.t_s.append(t_s)
        self.range_m.append(range_m)
        while t_s - self.t_s[0] > self.window_s:
            self.t_s.popleft()
            self.range_m.popleft()

    def estimate(self, now_s: float) -> tuple[float, float, float] | None:
        if not self.t_s or now_s - self.t_s[-1] > self.stale_s:
            return None
        t_s = np.array(self.t_s) - self.t_s[-1]
        span_s = -t_s[0]
        if len(t_s) < self.min_samples or span_s < 0.2:
            return None
        # a quadratic fit also gives the closing acceleration, but needs a longer window to stay quiet
        deg = 2 if len(t_s) >= 10 and span_s >= 0.6 else 1
        coef = np.polyfit(t_s, np.array(self.range_m), deg)
        dt_s = now_s - self.t_s[-1]
        range_accel_mps2 = 2.0 * coef[0] if deg == 2 else 0.0
        # extrapolate to now: the measurement is older than the tick by the pipeline latency
        return float(np.polyval(coef, dt_s)), float(-np.polyval(np.polyder(coef), dt_s)), float(-range_accel_mps2)
