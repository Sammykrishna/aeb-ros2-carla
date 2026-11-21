import itertools
import sys

import numpy as np
from collections import deque
from dataclasses import replace

from aeb_controller.state_machine import AebParams, AebStateMachine

DT_S = 0.02
SEEN_GAP_M = 75.0
TRACK_WARMUP_S = 0.3
MEAS_DELAY_S = 0.05
TARGET_DECEL_LAG_S = 0.5
# tracker estimate noise measured by replaying it on the recorded fused ranges (std, correlation time)
CLOSING_NOISE = (0.6, 0.3)
TARGET_DECEL_NOISE = (1.3, 0.6)
SEEDS = range(5)
# ego_kmh, target_kmh, gap_m, target_decel_mps2, target_brake_after_s
SCENARIOS = {"CCRs_20": (20, 0, 80, 0.0, 0.0), "CCRs_30": (30, 0, 80, 0.0, 0.0), "CCRs_40": (40, 0, 80, 0.0, 0.0),
             "CCRs_50": (50, 0, 80, 0.0, 0.0), "CCRm_50_20": (50, 20, 80, 0.0, 0.0),
             "CCRb_50_d2": (50, 50, 40, 2.0, 1.0), "CCRb_50_d6": (50, 50, 12, 6.0, 1.0)}
# fitted on the recorded bags: decel = 0.67 + 3.83 * brake (4.5 m/s2 at full brake), ~0.15 s to respond
NOMINAL = dict(gain_mps2=3.83, offset_mps2=0.67, cmd_delay_s=0.15, lag_s=0.1, range_bias_m=0.0)


def ar1(prev: float, noise: tuple[float, float], rng: np.random.Generator) -> float:
    std, tau_s = noise
    keep = np.exp(-DT_S / tau_s)
    return prev * keep + std * np.sqrt(1.0 - keep ** 2) * rng.standard_normal()


def simulate(params: AebParams, scenario: tuple, plant: dict, seed: int = 0) -> tuple[float, float, float | None]:
    rng = np.random.default_rng(seed)
    closing_err, decel_err = 0.0, 0.0
    ego_kmh, tgt_kmh, gap0_m, tgt_decel_mps2, tgt_brake_s = scenario
    ego_mps, tgt0_mps = ego_kmh / 3.6, tgt_kmh / 3.6
    sm = AebStateMachine(params)
    gap_m, decel_mps2, t_s, seen_s, a_t_est = gap0_m, 0.0, 0.0, None, 0.0
    min_gap_m, brake_gap_m = gap0_m, None
    brake_q = deque([0.0] * round(plant["cmd_delay_s"] / DT_S))
    hist = deque(maxlen=round(MEAS_DELAY_S / DT_S) + 1)
    while t_s < 25.0:
        braking_tgt = tgt_decel_mps2 > 0.0 and t_s >= tgt_brake_s
        tgt_mps = max(0.0, tgt0_mps - tgt_decel_mps2 * (t_s - tgt_brake_s)) if braking_tgt else tgt0_mps
        hist.append((gap_m, ego_mps - tgt_mps, tgt_decel_mps2 if braking_tgt and tgt_mps > 0 else 0.0))
        seen_s = t_s if seen_s is None and gap_m <= SEEN_GAP_M else seen_s
        tracked = seen_s is not None and t_s - seen_s >= TRACK_WARMUP_S
        meas_gap_m, closing_mps, true_decel = hist[0]
        a_t_est += (true_decel - a_t_est) * DT_S / TARGET_DECEL_LAG_S
        closing_err, decel_err = ar1(closing_err, CLOSING_NOISE, rng), ar1(decel_err, TARGET_DECEL_NOISE, rng)
        _, brake, _, _ = sm.step(t_s, ego_mps, meas_gap_m + plant["range_bias_m"] if tracked else None,
                                 closing_mps + closing_err if tracked else 0.0, max(a_t_est + decel_err, 0.0) if tracked else 0.0)
        if brake > 0 and brake_gap_m is None:
            brake_gap_m = gap_m
        brake_q.append(brake)
        b = brake_q.popleft()
        decel_cmd = plant["offset_mps2"] + plant["gain_mps2"] * b if b > 0 else 0.0
        decel_mps2 += (decel_cmd - decel_mps2) * DT_S / plant["lag_s"]
        ego_mps = max(0.0, ego_mps - decel_mps2 * DT_S)
        gap_m -= (ego_mps - tgt_mps) * DT_S
        min_gap_m = min(min_gap_m, gap_m)
        t_s += DT_S
        if gap_m <= 0.0:
            return 0.0, (ego_mps - tgt_mps) * 3.6, brake_gap_m
        if ego_mps <= 0.05:
            break
    return min_gap_m, 0.0, brake_gap_m


def median_run(params: AebParams, scenario: tuple, plant: dict) -> tuple[float, float, float | None]:
    runs = [simulate(params, scenario, plant, seed) for seed in SEEDS]
    return (float(np.median([r[0] for r in runs])), float(np.median([r[1] for r in runs])),
            float(np.median([r[2] for r in runs if r[2] is not None])))


def validate():
    # the controller that produced the last CARLA run: brake level was a_req / 4.5, margin 0.5 m
    old = AebParams(min_partial_brake=0.3, brake_gain_mps2=4.5, brake_offset_mps2=0.0, partial_decel_margin=1.0,
                    margin_m=0.5, min_target_decel_mps2=0.5)
    carla = {"CCRs_20": 5.64, "CCRs_30": 11.34, "CCRs_40": 12.10, "CCRs_50": 11.78, "CCRm_50_20": 19.72,
             "CCRb_50_d2": 14.27, "CCRb_50_d6": "collision at 25 km/h"}
    print(f"{'scenario':<12}{'sim min gap / impact':>24}{'first brake (sim)':>19}   CARLA min gap")
    for name, scenario in SCENARIOS.items():
        gap_m, impact_kmh, brake_gap_m = median_run(old, scenario, NOMINAL)
        res = f"collision {impact_kmh:.0f} km/h" if impact_kmh else f"{gap_m:.2f} m"
        print(f"{name:<12}{res:>24}{brake_gap_m:>16.1f} m   {carla[name]}")


def sweep():
    perturbations = [dict(NOMINAL, gain_mps2=g, offset_mps2=o, range_bias_m=b, cmd_delay_s=d)
                     for g in (3.3, 4.3) for o in (0.3, 1.0) for b in (0.0, 0.4) for d in (0.15, 0.25)]
    print(f"{'candidate':<50}{'collisions/96':>14}{'CCRs_50 brake':>14}{'CCRs_50 gap':>12}{'CCRs_30 gap':>12}{'CCRm gap':>10}"
          f"{'d2 gap':>8}{'worst min gap':>14}{'d6 impact':>10}")
    for partial, full, margin, decay in itertools.product((0.5, 0.55, 0.65), (0.8, 0.9), (2.0, 3.0), (0.3, 0.6)):
        params = replace(AebParams(), partial_ratio=partial, full_ratio=full, margin_m=margin, brake_decay_per_s=decay)
        nominal = {n: median_run(params, s, NOMINAL) for n, s in SCENARIOS.items()}
        worst = [simulate(params, s, p) for n, s in SCENARIOS.items() if n != "CCRb_50_d6" for p in perturbations]
        safe_gaps = [r[0] for r in worst if r[1] == 0]
        print(f"{f'partial {partial} full {full} margin {margin} decay {decay}':<50}{sum(r[1] > 0 for r in worst):>14}"
              f"{nominal['CCRs_50'][2]:>12.1f} m{nominal['CCRs_50'][0]:>10.1f} m{nominal['CCRs_30'][0]:>10.1f} m"
              f"{nominal['CCRm_50_20'][0]:>8.1f} m{nominal['CCRb_50_d2'][0]:>6.1f} m{(min(safe_gaps) if safe_gaps else 0.0):>12.1f} m"
              f"{nominal['CCRb_50_d6'][1]:>8.0f}")


if __name__ == "__main__":
    validate() if sys.argv[1:] == ["validate"] else sweep()
