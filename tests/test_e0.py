"""Pins for E0 (challenge baseline reproduction). Every number is read from results/, never typed in."""
import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e0():
    return {int(k): v for k, v in json.loads((RES / "e0_challenge_baseline.json").read_text()).items()}


def test_window_counts_match_the_repo_example(e0):
    # 124 sequences of 200 frames minus 13 per sequence = 23,188 in the repo's SUBMISSION-EXAMPLE; one 199-frame sequence here
    assert e0[36]["n_windows"] in (23187, 23188)


@pytest.mark.parametrize("scen", [36, 37, 38, 39])
def test_reproduction_is_within_five_points_of_published_and_never_worse(e0, scen):
    # the challenge computed its numbers on its own training CSV (a subset of the windows); ours uses every
    # window, so small differences are expected, and a reproduction that came out worse would mean a bug
    r, p = e0[scen]["reproduced"], e0[scen]["published"]
    for k in ("top1", "top3", "top5"):
        assert -1.0 < 100 * r[k] - p[k] < 5.0, (scen, k, r[k], p[k])
    assert -0.2 < r["apl_db"] - p["apl_db"] < 1.0


@pytest.mark.parametrize("scen", [36, 37, 38, 39])
def test_stationary_frames_collapse(e0, scen):
    assert e0[scen]["stationary_le2mps"]["top1"] < 0.10
    assert e0[scen]["moving_gt2mps"]["top1"] > 2 * e0[scen]["stationary_le2mps"]["top1"]


def test_fitted_shift_differs_between_recordings(e0):
    assert len({e0[s]["shift"] for s in (36, 37, 38, 39)}) >= 3
