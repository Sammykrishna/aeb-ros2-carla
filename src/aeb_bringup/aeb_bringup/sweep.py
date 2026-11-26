import argparse
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .regression import carla_up, kill_group, preload_town, run_all
from .regression_stats import sweep_report

RMWS = {
    "fastrtps": dict(env={"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}),
    "fastrtps_udp": dict(env={"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp", "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4"}),
    "cyclonedds": dict(env={"RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp"}),
    "zenoh": dict(env={"RMW_IMPLEMENTATION": "rmw_zenoh_cpp"}, router=True),
}
QOS = {"be1": ("best_effort", 1), "be5": ("best_effort", 5), "rel1": ("reliable", 1), "rel10": ("reliable", 10)}
SCENARIOS = ["CCRs_50", "CCRb_50_d2", "CCRb_50_d6"]


def burn_cpu():
    while True:
        sum(i * i for i in range(10_000))


def start_load(workers: int) -> list[multiprocessing.Process]:
    procs = [multiprocessing.Process(target=burn_cpu, daemon=True) for _ in range(workers)]
    for proc in procs:
        proc.start()
    return procs


def start_router(env: dict) -> subprocess.Popen:
    router = subprocess.Popen(["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"], env={**os.environ, **env},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(3.0)
    return router


def sweep(rmws: list[str], qos_names: list[str], scenarios: list[str], runs: int, out_dir: str, seed0: int,
          load_workers: int = 0) -> dict[str, list[dict]]:
    results = {}
    for rmw in rmws:
        spec = RMWS[rmw]
        router = start_router(spec["env"]) if spec.get("router") else None
        try:
            for qos_name in qos_names:
                reliability, depth = QOS[qos_name]
                config = f"{rmw}_{qos_name}" + (f"_load{load_workers}" if load_workers else "")
                print(f"== {config}", flush=True)
                load = start_load(load_workers)
                try:
                    results[config] = run_all(scenarios, runs, Path(out_dir) / config, seed0,
                                              launch_args=[f"qos_reliability:={reliability}", f"qos_depth:={depth}"], env=spec["env"])
                finally:
                    for proc in load:
                        proc.kill()
        finally:
            if router:
                kill_group(router.pid, signal.SIGKILL)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmw", nargs="+", default=list(RMWS), choices=list(RMWS))
    parser.add_argument("--qos", nargs="+", default=list(QOS), choices=list(QOS))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--load", type=int, default=0, help="busy-loop worker processes to run alongside the pipeline")
    parser.add_argument("--out", default="sweep_runs")
    parser.add_argument("--report", default="docs/sweep_report.md")
    args = parser.parse_args()
    if not carla_up():
        sys.exit("no CARLA server on localhost:2000")
    preload_town()

    results = sweep(args.rmw, args.qos, args.scenarios, args.runs, args.out, args.seed0, args.load)
    report = sweep_report(results)
    print("\n" + report)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report + "\n")
    Path(args.out, "summary.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
