import csv
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, "scripts")
from analyze_bags import analyze

OUT = "docs/media"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b8b7b1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def last_bag(root: str, name: str) -> str:
    return sorted(glob.glob(f"{root}/{name}_2*"))[-1]


def fusion_by_distance():
    rows = [r for r in csv.DictReader(open("benchmarks/fusion_offline.csv")) if r["gap_bin_m"] != "5-60"]
    bins = sorted({r["gap_bin_m"] for r in rows}, key=lambda b: int(b.split("-")[0]))
    rate = lambda prefix: [float(next(r["fused_rate"] for r in rows if r["variant"].startswith(prefix) and r["gap_bin_m"] == b)) * 100 for b in bins]
    before, after = rate("A "), rate("D ")
    x, w = np.arange(len(bins)), 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w / 2 - 0.01, before, w, color=ORANGE, label="Before: shrunk box, median of at least 3 points")
    ax.bar(x + w / 2 + 0.01, after, w, color=BLUE, label="After: full box, nearest cluster of at least 2 points")
    for i in (3, 4):
        ax.text(x[i] - w / 2, before[i] + 2, f"{before[i]:.0f}%", ha="center", color=INK, fontsize=10)
        ax.text(x[i] + w / 2, after[i] + 2, f"{after[i]:.0f}%", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [f"{b.replace('-', ' to ')} m" for b in bins])
    ax.set_ylim(0, 138)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Frames where fusion produced a range")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_xlabel("True distance to the target")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    ax.set_title("Fusion kept losing the target beyond 25 m, and the fix recovered it", loc="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fusion_by_distance.png", dpi=160)


def detector_tradeoff():
    rows = list(csv.DictReader(open("benchmarks/detector_benchmark.csv")))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for r in rows:
        name = f"{r['model'].replace('.pt', '')} @ {r['imgsz']}"
        x, y = float(r["ms_mean"]), float(r["first_m_conf0.4"])
        if name == "yolov8n @ 640":
            ax.scatter(x, y, s=90, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate("YOLOv8n at 640 px\nwhere I started", (x, y), (x + 1.4, y - 5), color=INK, fontsize=10)
        elif name == "yolo11s @ 960":
            ax.scatter(x, y, s=90, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate("YOLO11s at 960 px\nwhat I chose", (x, y), (x + 0.8, y + 1.5), color=INK, fontsize=10)
        else:
            ax.scatter(x, y, s=60, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(name.replace("yolo", "").replace(" @ ", " @ "), (x, y), (x + 0.35, y - 2.2), color=MUTED, fontsize=8.5)
    ax.scatter([], [], s=60, color=GREY, label="Other models tested")
    ax.set_xlabel("Inference time per frame (ms, RTX 4050 Laptop)")
    ax.set_ylabel("Distance where the car is first detected (m)")
    ax.set_xlim(3, 23)
    ax.set_ylim(28, 82)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_title("A bigger model and a bigger input paid off for a few extra milliseconds", loc="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/detector_tradeoff.png", dpi=160)


def read_run(bag_dir: str) -> dict:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id="mcap"), ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    names = {"/ego/odom": "speed", "/ground_truth/gap": "gap", "/aeb/brake_cmd": "brake"}
    t0, series = None, {name: [] for name in names.values()}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in names:
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == "/ego/odom":
            t0 = t0 or t
        if t0 is None:
            continue
        value = msg.twist.twist.linear.x * 3.6 if topic == "/ego/odom" else msg.data
        series[names[topic]].append(((t - t0) * 1e-9, value))
    return {name: np.array(values) for name, values in series.items()}


def timelines():
    runs = {"CCRs_50": ("Stationary target, ego at 50 km/h", (2.0, 6.4)),
            "CCRb_50_d6": ("Lead car brakes hard at 6 m/s2, 12 m gap", (0.0, 3.6))}
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.4))
    for col, (name, (title, (lo, hi))) in enumerate(runs.items()):
        s = read_run(last_bag("bags", name))
        brake = s["brake"]
        start = brake[brake[:, 1] > 0][0, 0]
        panels = [("Ego speed (km/h)", s["speed"], (0, 56)), ("Gap to target (m)", s["gap"], (0, 85 if name == "CCRs_50" else 14)),
                  ("Brake command", brake, (0, 1.1))]
        for row, (label, data, ylim) in enumerate(panels):
            ax = axes[row, col]
            keep = (data[:, 0] >= lo) & (data[:, 0] <= hi)
            ax.plot(data[keep, 0], data[keep, 1], color=BLUE, linewidth=2)
            ax.axvline(start, color=ORANGE, linewidth=1.5, linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(*ylim)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=11.5, color=INK)
                ax.text(start + 0.04, ylim[1] * 0.9, "AEB starts braking", color=INK, fontsize=9.5)
            if row == 2:
                ax.set_xlabel("Time since the run started (s)")
        if name == "CCRb_50_d6":
            axes[0, col].annotate("impact at 25 km/h", (3.45, 25), (2.15, 6), color=INK, fontsize=9.5,
                                  arrowprops=dict(arrowstyle="-", color=MUTED))
            axes[1, col].axvline(1.0, color=GREY, linewidth=1.5, linestyle=":")
            axes[1, col].text(0.96, 12.3, "lead car starts braking", color=INK, fontsize=9.5, ha="right")
        else:
            axes[2, col].text(6.35, 0.55, "brake held\nafter the stop", color=INK, fontsize=9.5, ha="right")
            axes[0, col].annotate("stopped 11.7 m short", (5.9, 0), (4.55, 14), color=INK, fontsize=9.5,
                                  arrowprops=dict(arrowstyle="-", color=MUTED))
    fig.suptitle("Two runs through the same controller", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/timelines.png", dpi=160)


def stop_gaps():
    names = ["CCRs_20", "CCRs_30", "CCRs_40", "CCRs_50", "CCRm_50_20", "CCRb_50_d2"]
    labels = ["Stationary 20 km/h", "Stationary 30 km/h", "Stationary 40 km/h", "Stationary 50 km/h", "Slow lead, 50 vs 20 km/h", "Braking lead, 2 m/s2"]
    before = [analyze(last_bag("bags_v7", n))["min_gap_m"] for n in names]
    after = [analyze(last_bag("bags", n))["min_gap_m"] for n in names]
    y, h = np.arange(len(names))[::-1], 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(y + h / 2 + 0.01, before, h, color=ORANGE, label="Before the controller fixes")
    ax.barh(y - h / 2 - 0.01, after, h, color=BLUE, label="After")
    for yy, b, a in zip(y, before, after):
        ax.text(b + 0.3, yy + h / 2 + 0.01, f"{b:.1f} m", va="center", fontsize=9.5, color=INK_2)
        ax.text(a + 0.3, yy - h / 2 - 0.01, f"{a:.1f} m", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 25)
    ax.set_xlabel("Distance left to the target when the car stopped")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title("Moving and braking leads stopped much closer after the fixes", loc="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stop_gaps.png", dpi=160)


if __name__ == "__main__":
    for make in (fusion_by_distance, detector_tradeoff, timelines, stop_gaps):
        make()
        print("wrote", make.__name__)
