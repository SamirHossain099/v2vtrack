"""Pins for E2 (tracker) and E4 (known-truth arm). Numbers are read from results/, never typed in."""
import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e2():
    return json.loads((RES / "e2_tracker_src36_h0.json").read_text())["runs"]


@pytest.fixture(scope="module")
def e2_nolag():
    return json.loads((RES / "e2_tracker_src36_h0_nolag.json").read_text())["runs"]


@pytest.fixture(scope="module")
def e4():
    return json.loads((RES / "e4_known_truth.json").read_text())


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_static_source_map_fails_off_domain(e2, scen):
    assert e2[f"{scen}|static|S0|L0"]["top1"] < 0.15


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_tracker_with_one_sweep_per_two_seconds_recovers(e2, scen):
    st, tr = e2[f"{scen}|static|S0|L0"], e2[f"{scen}|kalman|S20|L0"]
    assert tr["top1"] > 0.30 and tr["top1"] > 2.5 * st["top1"]
    assert tr["apl_db"] > st["apl_db"] + 0.5


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_tracker_beats_the_offset_baselines_at_matched_feedback(e2, scen):
    for S, L in ((20, 0), (100, 0), (20, 2)):
        k = e2[f"{scen}|kalman|S{S}|L{L}"]["top1"]
        assert k >= e2[f"{scen}|index-offset-ma|S{S}|L{L}"]["top1"] - 0.005
        assert k >= e2[f"{scen}|robust-window|S{S}|L{L}"]["top1"] - 0.005


@pytest.mark.parametrize("scen", [36, 37, 38, 39])
def test_heading_hold_lifts_stationary_frames(e2, scen):
    assert e2[f"{scen}|kalman|S20|L0"]["stationary"]["top1"] > 1.5 * e2[f"{scen}|static|S0|L0"]["stationary"]["top1"]


def test_beam_hold_alone_loses_the_beam_without_full_sweeps(e2):
    for scen in (36, 37, 38, 39):
        assert e2[f"{scen}|beam-hold|S0|L2"]["apl_db"] < -10


def test_lag_correction_matters_only_for_37(e2, e2_nolag):
    k = "kalman|S20|L0"
    assert e2[f"37|{k}"]["top1"] - e2_nolag[f"37|{k}"]["top1"] > 0.05
    for scen in (36, 38, 39):
        assert abs(e2[f"{scen}|{k}"]["top1"] - e2_nolag[f"{scen}|{k}"]["top1"]) < 0.04


def test_known_truth_rotation_is_recovered_below_the_beam_width(e4):
    bw = 90 / 64
    assert e4["sign_check_K_positive"] and 35 < e4["K_fit"] < 48
    for key, v in e4["runs"].items():
        assert v["median_abs_err_deg"] < bw, (key, v)
        assert v["p90_abs_err_deg"] < 2 * bw, (key, v)
