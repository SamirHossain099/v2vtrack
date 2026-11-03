"""Pins for the Communications Letters version. Every claim is read from results/; the claims the letter makes, checked against results/ only (public)."""
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


@pytest.fixture(scope="module")
def e2():
    return json.loads((RES / "e2_tracker_letter_nolabelK_nolag.json").read_text())



def top1(e2, s, key):
    return e2["runs"][f"{s}|{key}"]["top1"]



def test_evaluation_uses_no_target_information(e2):
    # rule 27: slope from the codebook specification, no timing correction, source-only parameters
    assert abs(e2["K"] - 63 / (np.pi / 2)) < 1e-2
    assert e2["lag"] is False and e2["source"] == 36



def test_source_only_slope_matches_the_specification_closely():
    ks = json.loads((RES / "k_source_only.json").read_text())
    assert abs(ks["K_source36"] - ks["K_angle"]) < 2.5



@pytest.mark.parametrize("scen", [37, 38, 39])
def test_tracker_beats_static_map_and_index_offset_at_four_measurements(e2, scen):
    k = top1(e2, scen, "kalman|S0|L2")
    assert k > 3 * top1(e2, scen, "static|S0|L0")
    assert k > top1(e2, scen, "index-offset-ma|S0|L2")
    assert k > top1(e2, scen, "robust-window|S0|L2")
    assert k > top1(e2, scen, "beam-hold|S0|L2")



@pytest.mark.parametrize("scen", [37, 38, 39])
def test_tracker_leads_at_sparse_full_sweeps(e2, scen):
    k = top1(e2, scen, "kalman|S100|L0")
    for other in ("index-offset-ma|S100|L0", "robust-window|S100|L0", "beam-hold|S100|L0"):
        assert k > top1(e2, scen, other), other



def test_the_conceded_regime_is_real(e2):
    # with occasional full sweeps beam hold matches the tracker in top-1 but loses more power; with frequent sweeps
    # it is better outright on the urban streams. The letter says both.
    for scen in (37, 38, 39):
        bh, k = e2["runs"][f"{scen}|beam-hold|S500|L2"], e2["runs"][f"{scen}|kalman|S500|L2"]
        assert bh["top1"] >= k["top1"] - 0.01
        assert bh["apl_db"] < k["apl_db"] - 1.0
    for scen in (38, 39):
        assert top1(e2, scen, "beam-hold|S20|L2") > top1(e2, scen, "kalman|S20|L2")



def test_learned_network_collapses_and_finetuning_stays_below_tracker(e2):
    e3 = json.loads((RES / "e3_learned_letter_nolag.json").read_text())
    assert e3["lag"] is False
    agg = defaultdict(list)
    for k, v in e3["runs"].items():
        scen, mode, S, lr, seed = k.split("|")
        agg[(int(scen), mode, S)].append(v["top1"])
    for scen in (37, 38, 39):
        assert np.mean(agg[(scen, "learned-source", "S0")]) < 0.12
        ft = max(np.mean(v) for (s_, m, S), v in agg.items() if s_ == scen and m == "learned-supft" and S == "S20")
        assert ft < top1(e2, scen, "kalman|S20|L0")



def test_ablation_supports_each_stated_design_choice():
    e7 = json.loads((RES / "e7_ablation.json").read_text())["runs"]
    full, edge = e7["36|full|L2"], e7["36|edge maxima used|L2"]
    # on the source, edge maxima drive the state far off and cost accuracy and power: the interior rule is chosen there
    assert abs(edge["delta_final_deg"]) > 20 and edge["top1"] < full["top1"] and edge["apl_db"] < full["apl_db"] - 1
    for s in (36, 37, 38, 39):
        for setting in ("L2", "S100"):
            # the agreement gate is inert: the letter says it never changed a prediction
            assert e7[f"{s}|no agreement gate|{setting}"]["top1"] == e7[f"{s}|full|{setting}"]["top1"]
            assert e7[f"{s}|no agreement gate|{setting}"]["apl_db"] == e7[f"{s}|full|{setting}"]["apl_db"]
        # the standstill hold lifts accuracy on stationary rows everywhere
        assert e7[f"{s}|full|L2"]["stationary"]["top1"] > e7[f"{s}|no standstill hold|L2"]["stationary"]["top1"]



def test_standstill_hold_tradeoff_is_stated_correctly():
    e7 = json.loads((RES / "e7_ablation.json").read_text())["runs"]
    for s in (37, 38, 39):
        assert e7[f"{s}|no standstill hold|L2"]["apl_db"] > e7[f"{s}|full|L2"]["apl_db"]  # slightly lower loss without the hold
