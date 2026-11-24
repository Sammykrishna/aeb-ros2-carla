import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

from .regression_stats import check_expectations, markdown_report, summarize

SCENARIOS = ["CCRs_20", "CCRs_30", "CCRs_40", "CCRs_50", "CCRm_50_20", "CCRb_50_d2", "CCRb_50_d6",
             "FP_none", "FP_adjacent", "FP_follow"]
LAUNCH_TIMEOUT_S = 150


def carla_up(host: str = "localhost", port: int = 2000) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def preload_town(town: str = "Town04"):
    import carla  # not needed by the stats tests, so only imported when a server is there
    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)
    if not client.get_world().get_map().name.endswith(town):  # switching maps with yolo on the gpu has crashed the 6 GB card
        client.load_world(town)


def load_expectations() -> dict:
    with open(get_package_share_directory("aeb_bringup") + "/config/regression.yaml") as f:
        return yaml.safe_load(f)


def kill_group(pid: int, sig: int):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:  # nothing from the launch is left, which is the normal case
        pass


def run_one(name: str, seed: int, out_dir: Path | str) -> dict:
    path = Path(out_dir) / name / f"{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ros2", "launch", "aeb_bringup", "aeb.launch.py", f"scenario:={name}", f"seed:={seed}",
           f"summary_path:={path}", "record:=false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=LAUNCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        kill_group(proc.pid, signal.SIGINT)
        time.sleep(5)
        kill_group(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    kill_group(proc.pid, signal.SIGKILL)  # nodes outlive a launch that was killed
    if path.exists():
        return json.loads(path.read_text())
    return dict(scenario=name, seed=seed, result="ERROR", detail=out[-600:])


def run_all(scenarios: list[str], runs: int, out_dir: Path | str, seed0: int = 0, retries: int = 1) -> list[dict]:
    results = []
    for name in scenarios:
        for seed in range(seed0, seed0 + runs):
            if not carla_up():
                sys.exit(f"CARLA stopped answering before {name} seed {seed}; rerun the same command to resume")
            for _ in range(retries + 1):
                result = run_one(name, seed, out_dir)
                if result["result"] != "ERROR":
                    break
            print(f"{name:<12} seed {seed:>3}: {result['result']:<11} min gap {result.get('min_gap_m', float('nan')):7.2f} m", flush=True)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--out", default="regression_runs")
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--report", default="docs/regression_report.md")
    args = parser.parse_args()
    if not carla_up():
        sys.exit("no CARLA server on localhost:2000")
    preload_town()

    runs = run_all(args.scenarios, args.runs, args.out, args.seed0)
    summary = summarize(runs)
    violations = check_expectations(summary, load_expectations())
    report = markdown_report(summary, violations)
    print("\n" + report)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report + "\n")
    Path(args.out, "summary.json").write_text(json.dumps(dict(summary=summary, violations=violations, runs=runs), indent=1))
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
