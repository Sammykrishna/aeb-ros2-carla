import math

import numpy as np

PASSING = ("PASS",)


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def latency_summary(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return dict(objects_received=0, latency_p50_ms=None, latency_p95_ms=None, latency_max_ms=None)
    p50, p95 = np.percentile(latencies_ms, [50, 95])
    return dict(objects_received=len(latencies_ms), latency_p50_ms=float(p50),
                latency_p95_ms=float(p95), latency_max_ms=float(max(latencies_ms)))


def _spread(values: list[float]) -> dict | None:
    return dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values))) if values else None


def summarize(runs: list[dict]) -> dict[str, dict]:
    summary = {}
    for name in sorted({r["scenario"] for r in runs}):
        mine = [r for r in runs if r["scenario"] == name]
        results = [r["result"] for r in mine]
        passes = sum(res in PASSING for res in results)
        low, high = wilson_interval(passes, len(mine))
        # a false-positive run drives past its target, so its gap goes negative and means nothing
        finite_gaps = [r["min_gap_m"] for r in mine if "min_gap_m" in r and r["min_gap_m"] < 900 and not r.get("false_positive")]
        summary[name] = dict(
            runs=len(mine), passes=passes, pass_rate=passes / len(mine), ci_low=low, ci_high=high,
            counts={res: results.count(res) for res in sorted(set(results))},
            min_gap_m=_spread(finite_gaps),
            first_brake_gap_m=_spread([r["first_brake_gap_m"] for r in mine if r.get("first_brake_gap_m") is not None]),
            impact_kmh=_spread([r["impact_kmh"] for r in mine if r["result"] == "COLLISION"]),
            false_brakes=results.count("FALSE_BRAKE"))
    return summary


def check_expectations(summary: dict[str, dict], expectations: dict[str, dict]) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for name, s in summary.items():
        exp = expectations.get(name, {})
        found = []
        if exp.get("expect") == "mitigate":
            bad = [res for res in s["counts"] if res not in ("PASS", "COLLISION")]
            if bad:
                found.append(f"unexpected results {bad}")
            worst = s["impact_kmh"]["max"] if s["impact_kmh"] else 0.0
            if worst > exp["max_impact_kmh"]:
                found.append(f"impact {worst:.0f} km/h above the {exp['max_impact_kmh']} km/h limit")
        elif s["pass_rate"] < exp.get("min_pass_rate", 0.9):
            found.append(f"pass rate {s['pass_rate']:.0%} below {exp.get('min_pass_rate', 0.9):.0%} ({s['counts']})")
        if found:
            violations[name] = found
    return violations


def markdown_report(summary: dict[str, dict], violations: dict[str, list[str]]) -> str:
    lines = ["| Scenario | Runs | Pass | Pass rate | 95% interval | Min gap, mean (range) | First brake, mean | Worst impact |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        gap = s["min_gap_m"]
        gap_txt = f"{gap['mean']:.1f} m ({gap['min']:.1f} to {gap['max']:.1f})" if gap else "none"
        brake = f"{s['first_brake_gap_m']['mean']:.1f} m" if s["first_brake_gap_m"] else "no brake"
        impact = f"{s['impact_kmh']['max']:.0f} km/h" if s["impact_kmh"] else "none"
        lines.append(f"| {name} | {s['runs']} | {s['passes']} | {s['pass_rate']:.0%} | "
                     f"{s['ci_low']:.0%} to {s['ci_high']:.0%} | {gap_txt} | {brake} | {impact} |")
    if violations:
        lines += ["", "Expectations not met:"] + [f"- {name}: {'; '.join(v)}" for name, v in violations.items()]
    return "\n".join(lines)


def latency_rollup(runs: list[dict]) -> dict:
    timed = [r for r in runs if r.get("latency_p50_ms") is not None]
    sent = sum(r.get("frames_sent", 0) for r in runs)
    return dict(
        p50_ms=float(np.median([r["latency_p50_ms"] for r in timed])) if timed else None,
        p95_ms=float(np.median([r["latency_p95_ms"] for r in timed])) if timed else None,
        worst_ms=float(max(r["latency_max_ms"] for r in timed)) if timed else None,
        delivered=sum(r.get("objects_received", 0) for r in runs) / sent if sent else 0.0)


def sweep_report(runs_by_config: dict[str, list[dict]]) -> str:
    lines = ["| Config | Scenario | Runs | Pass | 95% interval | Min gap, mean | First brake, mean | Latency p50 / p95 / worst | Delivered |",
             "|---|---|---|---|---|---|---|---|---|"]
    for config, runs in runs_by_config.items():
        for name, s in summarize(runs).items():
            mine = [r for r in runs if r["scenario"] == name]
            lat = latency_rollup(mine)
            gap = f"{s['min_gap_m']['mean']:.1f} m" if s["min_gap_m"] else "none"
            brake = f"{s['first_brake_gap_m']['mean']:.1f} m" if s["first_brake_gap_m"] else "no brake"
            lat_txt = "n/a" if lat["p50_ms"] is None else f"{lat['p50_ms']:.0f} / {lat['p95_ms']:.0f} / {lat['worst_ms']:.0f} ms"
            lines.append(f"| {config} | {name} | {s['runs']} | {s['passes']} | {s['ci_low']:.0%} to {s['ci_high']:.0%} | "
                         f"{gap} | {brake} | {lat_txt} | {lat['delivered']:.0%} |")
    return "\n".join(lines)
