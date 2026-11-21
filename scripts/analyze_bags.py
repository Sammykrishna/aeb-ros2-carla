import glob
import sys

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

NAMES = ["CCRs_20", "CCRs_30", "CCRs_40", "CCRs_50", "CCRm_50_20", "CCRb_50_d2", "CCRb_50_d6",
         "FP_none", "FP_adjacent", "FP_follow"]
STATE_ORDER = ["IDLE", "WARNING", "PARTIAL_BRAKE", "FULL_BRAKE", "HOLD"]
TOPICS = {"/aeb/objects_3d", "/aeb/state", "/aeb/brake_cmd", "/ground_truth/gap", "/ground_truth/lateral_offset",
          "/ego/odom", "/scenario/result"}
FUSED_BINS_M = [(15, 25), (25, 35), (35, 45), (45, 60)]


def analyze(bag_dir: str) -> dict:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    started, gap_m, speed_mps, result = False, None, 0.0, ""
    speeds_mps, lat_m, states, fused, errs_m, max_brake = [], [], [], [], [], 0.0
    first_obj_m = first_brake = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in TOPICS:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/ego/odom":
            started, speed_mps = True, msg.twist.twist.linear.x
            speeds_mps.append(speed_mps)
        if not started:
            continue
        if topic == "/ground_truth/gap":
            gap_m = msg.data
        elif topic == "/ground_truth/lateral_offset":
            lat_m.append(abs(msg.data))
        elif topic == "/aeb/objects_3d" and gap_m is not None:
            fused.append((gap_m, bool(msg.detections)))
            if msg.detections:
                errs_m.append(msg.detections[0].bbox.center.position.x - gap_m)
                first_obj_m = first_obj_m or gap_m
        elif topic == "/aeb/state" and (not states or states[-1] != msg.data):
            states.append(msg.data)
        elif topic == "/aeb/brake_cmd" and msg.data > 0:
            max_brake = max(max_brake, msg.data)
            first_brake = first_brake or (gap_m, speed_mps * 3.6)
        elif topic == "/scenario/result":
            result = msg.data
    _, verdict, min_gap = result.split(":")
    fused_rates = {}
    for lo, hi in FUSED_BINS_M:
        in_bin = [hit for g, hit in fused if lo <= g < hi]
        fused_rates[(lo, hi)] = float(np.mean(in_bin)) if in_bin else float("nan")
    return dict(verdict=verdict, min_gap_m=float(min_gap.split("=")[1]),
                impact_kmh=speeds_mps[-2] * 3.6 if verdict == "COLLISION" else 0.0,
                first_obj_m=first_obj_m, first_brake=first_brake, fused_rates=fused_rates,
                flickers=sum(a == "WARNING" and b == "IDLE" for a, b in zip(states, states[1:])),
                max_brake=max_brake, worst_state=max(states, key=STATE_ORDER.index) if states else "-",
                max_lat_m=max(lat_m), err_m=float(np.mean(errs_m)) if errs_m else float("nan"))


def main():
    runs = {d: {n: analyze(sorted(glob.glob(f"{d}/{n}_2*"))[-1]) for n in NAMES if glob.glob(f"{d}/{n}_2*")} for d in sys.argv[1:]}
    for bag_dir, res in runs.items():
        print(f"\n{bag_dir}")
        print(f"{'scenario':<11}{'result':<20}{'min gap':>8}{'1st object':>11}{'first brake':>17}"
              + "".join(f"{f'fused {lo}-{hi}m':>16}" for lo, hi in FUSED_BINS_M) + f"{'flickers':>9}{'range err':>10}{'|lat|':>7}{'max brake':>10}{'worst state':>15}")
        for n, r in res.items():
            verdict = r["verdict"] + (f" ({r['impact_kmh']:.0f} km/h)" if r["impact_kmh"] else "")
            brake = f"{r['first_brake'][0]:.1f} m @ {r['first_brake'][1]:.0f}" if r["first_brake"] else "none"
            first_obj = f"{r['first_obj_m']:.1f} m" if r["first_obj_m"] else "none"
            print(f"{n:<11}{verdict:<20}{r['min_gap_m']:>8.2f}{first_obj:>11}{brake:>17}"
                  + "".join(f"{r['fused_rates'][b]:>16.0%}" for b in FUSED_BINS_M)
                  + f"{r['flickers']:>9}{r['err_m']:>+10.2f}{r['max_lat_m']:>7.2f}{r['max_brake']:>10.2f}{r['worst_state']:>15}")


if __name__ == "__main__":
    main()
