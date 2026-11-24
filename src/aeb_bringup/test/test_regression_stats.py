import numpy as np
import pytest

from aeb_bringup.regression_stats import check_expectations, markdown_report, summarize, wilson_interval
from aeb_bringup.scenario_cfg import TARGET_MODELS, jitter_cfg

CCR = dict(type="stationary", ego_kmh=50, target_kmh=0, gap_m=80)


def run(scenario: str, result: str, **extra) -> dict:
    return dict(scenario=scenario, seed=0, result=result, min_gap_m=5.0, **extra)


def test_wilson_interval_all_passes_has_a_lower_bound_well_below_one():
    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.722, abs=0.01) and high == pytest.approx(1.0)


def test_wilson_interval_shrinks_with_more_runs():
    assert wilson_interval(30, 30)[0] > wilson_interval(10, 10)[0]


def test_wilson_interval_handles_no_runs_and_no_passes():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 10)[0] == 0.0


def test_summarize_counts_and_pass_rate():
    runs = [run("CCRs_50", "PASS")] * 8 + [run("CCRs_50", "COLLISION", impact_kmh=20.0)] * 2
    s = summarize(runs)["CCRs_50"]
    assert s["runs"] == 10 and s["passes"] == 8 and s["pass_rate"] == 0.8
    assert s["counts"] == {"COLLISION": 2, "PASS": 8} and s["impact_kmh"]["max"] == 20.0


def test_summarize_ignores_the_placeholder_gap_of_runs_without_a_target():
    runs = [dict(scenario="FP_none", seed=0, result="PASS", min_gap_m=999.0)]
    assert summarize(runs)["FP_none"]["min_gap_m"] is None


def test_pass_rate_expectation_is_checked():
    runs = [run("CCRs_50", "PASS")] * 8 + [run("CCRs_50", "COLLISION", impact_kmh=20.0)] * 2
    assert check_expectations(summarize(runs), {"CCRs_50": {"min_pass_rate": 0.9}}).keys() == {"CCRs_50"}
    assert not check_expectations(summarize(runs), {"CCRs_50": {"min_pass_rate": 0.8}})


def test_any_false_brake_fails_a_false_positive_scenario():
    runs = [run("FP_none", "PASS")] * 9 + [run("FP_none", "FALSE_BRAKE")]
    assert "FP_none" in check_expectations(summarize(runs), {"FP_none": {"min_pass_rate": 1.0}})


def test_mitigation_expectation_allows_collisions_below_the_impact_limit():
    exp = {"CCRb_50_d6": {"expect": "mitigate", "max_impact_kmh": 35}}
    ok = [run("CCRb_50_d6", "COLLISION", impact_kmh=25.0)] * 5
    too_hard = ok + [run("CCRb_50_d6", "COLLISION", impact_kmh=41.0)]
    assert not check_expectations(summarize(ok), exp)
    assert "CCRb_50_d6" in check_expectations(summarize(too_hard), exp)


def test_markdown_report_lists_violations():
    runs = [run("CCRs_50", "COLLISION", impact_kmh=20.0)] * 3
    summary = summarize(runs)
    text = markdown_report(summary, check_expectations(summary, {}))
    assert "| CCRs_50 | 3 | 0 | 0% |" in text and "Expectations not met" in text


def test_jitter_is_reproducible_and_bounded():
    a, b = jitter_cfg(CCR, np.random.default_rng(3)), jitter_cfg(CCR, np.random.default_rng(3))
    assert a == b and a != jitter_cfg(CCR, np.random.default_rng(4))
    assert 49.0 <= a["ego_kmh"] <= 51.0 and 76.0 <= a["gap_m"] <= 84.0 and abs(a["lateral_offset_m"]) <= 0.2
    assert a["target_model"] in TARGET_MODELS


def test_jitter_keeps_a_stationary_target_stationary_and_a_missing_target_missing():
    assert jitter_cfg(CCR, np.random.default_rng(0))["target_kmh"] == 0
    out = jitter_cfg(dict(type="none", no_target=True, ego_kmh=50, gap_m=0), np.random.default_rng(0))
    assert "target_model" not in out and out["gap_m"] == 0


def test_jitter_moves_the_braking_onset_of_a_braking_lead():
    cfg = dict(type="braking", ego_kmh=50, target_kmh=50, gap_m=12, brake_after_s=1.0, target_decel_mps2=6.0)
    assert abs(jitter_cfg(cfg, np.random.default_rng(1))["brake_after_s"] - 1.0) <= 0.2


def test_summarize_skips_the_gap_of_false_positive_runs_that_drive_past_the_target():
    runs = [dict(scenario="FP_adjacent", seed=0, result="PASS", min_gap_m=-57.0, false_positive=True)]
    assert summarize(runs)["FP_adjacent"]["min_gap_m"] is None
