import glob
import sys

import cv2
import numpy as np
from PIL import Image
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

STATE_BGR = {"IDLE": (80, 200, 0), "WARNING": (0, 200, 255), "PARTIAL_BRAKE": (0, 130, 255),
             "FULL_BRAKE": (60, 60, 240), "HOLD": (230, 140, 40)}
OUT_SIZE = (640, 360)


def load_frames(bag_dir: str) -> list[dict]:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    latest = dict(state="IDLE", brake=0.0, gap_m=float("nan"), speed_mps=0.0)
    frames: dict[tuple[int, int], dict] = {}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in ("/ego/camera/image", "/aeb/detections_2d", "/aeb/objects_3d", "/aeb/state", "/aeb/brake_cmd",
                         "/ground_truth/gap", "/ego/odom"):
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/aeb/state":
            latest["state"] = msg.data
        elif topic == "/aeb/brake_cmd":
            latest["brake"] = msg.data
        elif topic == "/ground_truth/gap":
            latest["gap_m"] = msg.data
        elif topic == "/ego/odom":
            latest["speed_mps"] = msg.twist.twist.linear.x
        elif topic == "/ego/camera/image":
            img_bgr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()
            frames[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = dict(img=img_bgr, boxes=[], seen_m=None, **latest)
        else:
            frame = frames.get((msg.header.stamp.sec, msg.header.stamp.nanosec))
            if frame is None:
                continue
            if topic == "/aeb/detections_2d":
                frame["boxes"] = [(d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y,
                                   d.results[0].hypothesis.score) for d in msg.detections]
            else:
                ahead_m = [d.bbox.center.position.x for d in msg.detections if d.bbox.center.position.x > 0]
                frame["seen_m"] = min(ahead_m) if ahead_m else None
    return list(frames.values())


def draw(frame: dict, title: str) -> np.ndarray:
    img = frame["img"].copy()
    for cx, cy, w, h, score in frame["boxes"]:
        p1, p2 = (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2))
        cv2.rectangle(img, p1, p2, (80, 200, 0), 2)
        cv2.putText(img, f"car {score:.2f}", (p1[0], max(p1[1] - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 0), 2)
    if frame["seen_m"] is not None and frame["boxes"]:
        cx, cy, w, h, _ = frame["boxes"][0]
        cv2.putText(img, f"AEB sees {frame['seen_m']:.0f} m", (int(cx - w / 2), int(cy + h / 2) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    shade = img.copy()
    cv2.rectangle(shade, (0, 0), (img.shape[1], 92), (25, 25, 25), -1)
    img = cv2.addWeighted(shade, 0.65, img, 0.35, 0)
    chip = STATE_BGR.get(frame["state"], (200, 200, 200))
    cv2.rectangle(img, (0, 0), (img.shape[1], 10), chip, -1)
    cv2.rectangle(img, (14, 50), (38, 74), chip, -1)
    cv2.putText(img, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"AEB {frame['state']}   brake {frame['brake']:.2f}", (46, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"{frame['speed_mps'] * 3.6:.0f} km/h", (640, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"true gap {frame['gap_m']:.1f} m", (640, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def make_gif(bag_dir: str, out_path: str, title: str, first_gap_m: float, every: int = 2):
    frames = [f for f in load_frames(bag_dir) if f["gap_m"] <= first_gap_m]
    frames = frames[::every]
    pil = [Image.fromarray(cv2.cvtColor(draw(f, title), cv2.COLOR_BGR2RGB)).resize(OUT_SIZE, Image.LANCZOS)
           .quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    durations = [50 * every] * len(pil)
    durations[-1] = 1500
    pil[0].save(out_path, save_all=True, append_images=pil[1:], duration=durations, loop=0, optimize=True)
    print(f"{out_path}: {len(pil)} frames")


if __name__ == "__main__":
    latest = lambda name: sorted(glob.glob(f"bags/{name}_2*"))[-1]
    make_gif(latest("CCRs_50"), "docs/media/pov_ccrs50.gif", "Stationary car ahead, ego at 50 km/h", 60.0)
    make_gif(latest("CCRb_50_d6"), "docs/media/pov_ccrb_d6.gif", "Lead car brakes hard, 12 m gap", 99.0)
    sys.exit(0)
