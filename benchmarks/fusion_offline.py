import csv
import glob

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from aeb_fusion.geometry import nearest_cluster_xyz, project_to_image

BAGS = ["CCRs_20", "CCRs_30", "CCRs_40", "CCRs_50", "CCRm_50_20", "CCRb_50_d2"]
CAM_IN_LIDAR_M = np.array([1.5, 0.0, -0.2])
BUMPER_M = 2.4
IN_LANE_PX = 120
GAP_BINS_M = [(5, 15), (15, 25), (25, 35), (35, 45), (45, 60)]
WRONG_ERR_M = 2.0


def median_xyz_in_box(pts_lidar, u_px, v_px, in_front, box_px, shrink, min_points):
    cx_px, cy_px, w_px, h_px = box_px
    in_box = in_front & (np.abs(u_px - cx_px) < w_px * shrink / 2) & (np.abs(v_px - cy_px) < h_px * shrink / 2)
    return np.median(pts_lidar[in_box], axis=0) if in_box.sum() >= min_points else None


def nearest_z_filtered_xyz(pts_lidar, u_px, v_px, in_front, box_px, min_points, z_min_m):
    return nearest_cluster_xyz(pts_lidar[pts_lidar[:, 2] > z_min_m], u_px[pts_lidar[:, 2] > z_min_m],
                               v_px[pts_lidar[:, 2] > z_min_m], in_front[pts_lidar[:, 2] > z_min_m],
                               box_px, 1.0, min_points)


VARIANTS = {
    "A current (shrink .5, min 3, median)": lambda *a: median_xyz_in_box(*a, 0.5, 3),
    "B shrink .8, min 3, median": lambda *a: median_xyz_in_box(*a, 0.8, 3),
    "C shrink .8, min 2, median": lambda *a: median_xyz_in_box(*a, 0.8, 2),
    "D full box, min 2, nearest cluster": lambda *a: nearest_cluster_xyz(*a, 1.0, 2),
    "E shrink .8, min 2, nearest cluster": lambda *a: nearest_cluster_xyz(*a, 0.8, 2),
    "F full box, min 2, nearest, z>-1.5": lambda *a: nearest_z_filtered_xyz(*a, 2, -1.5),
    "G full box, min 1, nearest, z>-1.5": lambda *a: nearest_z_filtered_xyz(*a, 1, -1.5),
}


def load_frames(bag_dir: str) -> list[dict]:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    frames: dict[tuple[int, int], dict] = {}
    K, latest = None, None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ("/ego/camera/camera_info", "/ego/lidar/points", "/ground_truth/gap", "/aeb/detections_2d"):
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/ego/camera/camera_info":
            K = np.array(msg.k).reshape(3, 3)
        elif topic == "/ego/lidar/points":
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            frames[key] = dict(pts=np.frombuffer(msg.data, np.float32).reshape(-1, 3).copy(), boxes=[], gap_m=None, K=K)
            latest = key
        elif topic == "/ground_truth/gap" and latest in frames:
            frames[latest]["gap_m"] = msg.data
        elif topic == "/aeb/detections_2d":
            frame = frames.get((msg.header.stamp.sec, msg.header.stamp.nanosec))
            if frame is not None:
                frame["boxes"] = [(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y)
                                  for d in msg.detections if abs(d.bbox.center.position.x - K[0, 2]) < IN_LANE_PX]
    return [f for f in frames.values() if f["gap_m"] is not None and f["boxes"]]


def main():
    frames = []
    for name in BAGS:
        clip = load_frames(sorted(glob.glob(f"../bags/{name}_2*"))[-1])
        print(f"{name}: {len(clip)} frames with an in-lane YOLO box", flush=True)
        frames += clip
    gaps_m = np.array([f["gap_m"] for f in frames])

    rows = []
    for variant, fuse in VARIANTS.items():
        err_m = np.full(len(frames), np.nan)
        for i, f in enumerate(frames):
            u_px, v_px, in_front = project_to_image(f["pts"], CAM_IN_LIDAR_M, f["K"])
            xyz_m = fuse(f["pts"], u_px, v_px, in_front, f["boxes"][0])
            if xyz_m is not None:
                err_m[i] = xyz_m[0] - BUMPER_M - f["gap_m"]
        for lo, hi in GAP_BINS_M + [(5, 60)]:
            in_bin = (gaps_m >= lo) & (gaps_m < hi)
            ok = in_bin & ~np.isnan(err_m)
            rows.append({"variant": variant, "gap_bin_m": f"{lo}-{hi}", "frames": int(in_bin.sum()),
                         "fused_rate": round(float(ok.sum() / max(in_bin.sum(), 1)), 2),
                         "wrong_rate": round(float((np.abs(err_m[ok]) > WRONG_ERR_M).sum() / max(in_bin.sum(), 1)), 2),
                         "err_mean_m": round(float(np.mean(err_m[ok])), 2) if ok.any() else "",
                         "err_mae_m": round(float(np.mean(np.abs(err_m[ok]))), 2) if ok.any() else ""})
    with open("fusion_offline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'variant':<40}" + "".join(f"{f'{lo}-{hi} m':>13}" for lo, hi in GAP_BINS_M) + f"{'all 5-60 m':>13}")
    print(f"{'(fused rate / wrong rate)':<40}")
    for variant in VARIANTS:
        cells = [r for r in rows if r["variant"] == variant]
        print(f"{variant:<40}" + "".join(f"{f'{r['fused_rate']:.0%}/{r['wrong_rate']:.0%}':>13}" for r in cells))
    print(f"\n{'variant':<40}" + "".join(f"{f'{lo}-{hi} m':>13}" for lo, hi in GAP_BINS_M) + f"{'all 5-60 m':>13}")
    print(f"{'(mean error / MAE, m)':<40}")
    for variant in VARIANTS:
        cells = [r for r in rows if r["variant"] == variant]
        print(f"{variant:<40}" + "".join(f"{(f'{r['err_mean_m']:+.2f}/{r['err_mae_m']:.2f}' if r['err_mae_m'] != '' else '-'):>13}" for r in cells))
    n_in_box = [f for f in frames if f["gap_m"] > 25]
    print(f"\n{len(frames)} frames total, {len(n_in_box)} beyond 25 m")


if __name__ == "__main__":
    main()
