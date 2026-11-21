import csv
import glob
import sys
import time

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from ultralytics import YOLO

BAGS = ["CCRs_20", "CCRs_30", "CCRs_40"]
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"]
IMGSZ = [640, 960]
ROAD_USERS = [0, 1, 2, 3, 5, 7]
CENTER_HALF_PX = 120
GAP_BINS_M = [(5, 15), (15, 25), (25, 35), (35, 45), (45, 60)]


def load_frames(bag_dir: str) -> list[tuple[np.ndarray, float]]:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    frames, pending = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/ego/camera/image":
            msg = deserialize_message(data, get_message(types[topic]))
            pending.append(np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy())
        elif topic == "/ground_truth/gap":
            gap_m = deserialize_message(data, get_message(types[topic])).data
            frames += [(img, gap_m) for img in pending]
            pending = []
    return frames


def lead_conf(model: YOLO, img_bgr: np.ndarray, imgsz: int) -> float:
    boxes = model.predict(img_bgr, conf=0.25, imgsz=imgsz, device="cuda:0",
                          classes=ROAD_USERS, verbose=False)[0].boxes
    center_x_px = boxes.xywh[:, 0].cpu().numpy()
    in_lane = np.abs(center_x_px - img_bgr.shape[1] / 2) < CENTER_HALF_PX
    return float(boxes.conf.cpu().numpy()[in_lane].max()) if in_lane.any() else 0.0


def first_range_m(gaps_m: np.ndarray, detected: np.ndarray, run: int = 3) -> float:
    for i in range(len(detected) - run + 1):
        if detected[i:i + run].all():
            return float(gaps_m[i])
    return float("nan")


def main():
    clips = {name: load_frames(sorted(glob.glob(f"../bags/{name}_2*"))[-1]) for name in BAGS}
    n_frames = sum(len(f) for f in clips.values())
    print(f"{n_frames} frames from {list(clips)}", flush=True)
    warm_img = next(iter(clips.values()))[0][0]
    fields = ["model", "imgsz", "ms_mean", "ms_p95", "vram_mb", "first_m_conf0.4", "first_m_conf0.25"] + \
             [f"recall_{lo}-{hi}m" for lo, hi in GAP_BINS_M]
    with open("detector_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for weights in MODELS:
            model = YOLO(weights)
            for imgsz in IMGSZ:
                for _ in range(5):
                    lead_conf(model, warm_img, imgsz)
                torch.cuda.reset_peak_memory_stats()
                confs, gaps, times_ms, first04, first025 = [], [], [], [], []
                for frames in clips.values():
                    clip_conf = []
                    for img, gap_m in frames:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        clip_conf.append(lead_conf(model, img, imgsz))
                        times_ms.append((time.perf_counter() - t0) * 1e3)
                    clip_conf, clip_gap = np.array(clip_conf), np.array([g for _, g in frames])
                    first04.append(first_range_m(clip_gap, clip_conf >= 0.4))
                    first025.append(first_range_m(clip_gap, clip_conf >= 0.25))
                    confs.append(clip_conf)
                    gaps.append(clip_gap)
                conf, gap = np.concatenate(confs), np.concatenate(gaps)
                row = {"model": weights, "imgsz": imgsz, "ms_mean": round(np.mean(times_ms), 1),
                       "ms_p95": round(np.percentile(times_ms, 95), 1),
                       "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                       "first_m_conf0.4": round(np.nanmean(first04), 1),
                       "first_m_conf0.25": round(np.nanmean(first025), 1)}
                for lo, hi in GAP_BINS_M:
                    in_bin = (gap >= lo) & (gap < hi)
                    row[f"recall_{lo}-{hi}m"] = round(float((conf[in_bin] >= 0.4).mean()), 2) if in_bin.any() else ""
                writer.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    sys.exit(main())
