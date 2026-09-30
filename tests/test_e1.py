"""Pins for E1 (offset origin). Numbers are read from results/e1d_final.json and results/e1c_lag_per_seq.json."""
import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e1():
    return json.loads((RES / "e1d_final.json").read_text())


@pytest.fixture(scope="module")
def lag():
    return {int(k): v for k, v in json.loads((RES / "e1c_lag_per_seq.json").read_text()).items()}


def test_codebook_slope_is_one_constant_near_the_nominal(e1):
    # 64 beams over +-45 deg is 40.7 beams/rad if uniform in angle; the fitted value must sit near it
    assert 35 < e1["K_joint"] < 48


def test_box_is_rigid_back_is_180_from_front(e1):
    for k, v in e1["scenarios"].items():
        assert abs(v["back_minus_front_deg"]) < 2.0, (k, v["back_minus_front_deg"])


def test_front_boresight_differs_between_recordings_by_a_few_degrees(e1):
    b = [v["boresight_deg"][0] for v in e1["scenarios"].values()]
    assert 2.0 < max(b) - min(b) < 10.0


def test_scenario_37_has_a_lag_of_about_one_second_and_the_others_do_not(e1, lag):
    assert 8 <= e1["lag_frames"]["37"] <= 14
    for k in ("36", "38", "39"):
        assert -2 <= e1["lag_frames"][k] <= 2
    assert 8 <= lag[37]["median_best_k"] <= 14
    for k in (36, 38, 39):
        assert -2 <= lag[k]["median_best_k"] <= 2


def test_source_transfer_fails_and_one_rotation_recovers_most_of_it(e1):
    for key, v in e1["transfer"].items():
        assert v["rotation_refit"]["top1"] >= v["source"]["top1"] - 0.02, key
        assert v["rotation_refit"]["apl_db"] >= v["source"]["apl_db"] - 0.05, key
    worst_src = min(v["source"]["top1"] for v in e1["transfer"].values())
    assert worst_src < 0.05
    recovered = [v["rotation_refit"]["top1"] / max(v["own"]["top1"], 1e-9) for v in e1["transfer"].values()]
    assert sum(r > 0.85 for r in recovered) >= 9, recovered


def test_rotations_needed_are_within_six_degrees(e1):
    assert all(abs(v["rotation_deg"]) < 6.0 for v in e1["transfer"].values())
