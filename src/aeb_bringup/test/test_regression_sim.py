import os

import pytest

from aeb_bringup.regression import SCENARIOS, carla_up, load_expectations, preload_town, run_one
from aeb_bringup.regression_stats import check_expectations, summarize

pytestmark = pytest.mark.skipif(os.environ.get("AEB_REGRESSION") != "1" or not carla_up(),
                                reason="needs AEB_REGRESSION=1 and a CARLA server on localhost:2000")


@pytest.fixture(scope="module", autouse=True)
def town():
    preload_town()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_meets_its_expectation_over_varied_runs(scenario, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp(scenario)
    runs = [run_one(scenario, seed, out_dir) for seed in range(int(os.environ.get("AEB_REGRESSION_RUNS", "3")))]
    assert not check_expectations(summarize(runs), load_expectations())
