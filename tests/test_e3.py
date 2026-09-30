"""Pins for E3 (learned baselines). Numbers read from results/e3_learned_src36.json and e3_learned_lrwide.json."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

RES = Path(__file__).resolve().parents[1] / "results"


def _agg(name):
    runs = json.loads((RES / name).read_text())["runs"]
    agg = defaultdict(list)
    for k, v in runs.items():
        scen, mode, S, lr, seed = k.split("|")
        agg[(int(scen), mode, S, lr)].append(v["top1"])
    return {k: float(np.mean(v)) for k, v in agg.items()}, {k: len(v) for k, v in agg.items()}


@pytest.fixture(scope="module")
def e3():
    return _agg("e3_learned_src36.json")


@pytest.fixture(scope="module")
def e2():
    return json.loads((RES / "e2_tracker_src36_h0.json").read_text())["runs"]


def test_three_seeds_everywhere(e3):
    assert all(n == 3 for n in e3[1].values())


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_source_network_collapses_off_domain_below_the_static_map(e3, e2, scen):
    assert e3[0][(scen, "learned-source", "S0", "lr0")] < 0.10
    assert e3[0][(scen, "learned-source", "S0", "lr0")] <= e2[f"{scen}|static|S0|L0"]["top1"] + 0.02


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_label_free_adaptation_does_not_help_the_network(e3, scen):
    src = e3[0][(scen, "learned-source", "S0", "lr0")]
    assert e3[0][(scen, "learned-norm", "S0", "lr0")] <= src + 0.01
    assert max(v for k, v in e3[0].items() if k[0] == scen and k[1] == "learned-tent") <= src + 0.01


@pytest.mark.parametrize("scen", [37, 38, 39])
def test_tracker_beats_tuned_supervised_finetuning_at_matched_budget(e3, e2, scen):
    for S in ("S20", "S100"):
        best = max(v for k, v in e3[0].items() if k[0] == scen and k[1] == "learned-supft" and k[2] == S)
        assert e2[f"{scen}|kalman|{S}|L0"]["top1"] > best + 0.12, (scen, S, best)


def test_finetuning_learning_rate_optimum_is_interior():
    wide, _ = _agg("e3_learned_lrwide.json")
    for scen in (38, 39):
        assert wide[(scen, "learned-supft", "S20", "lr0.003")] >= wide[(scen, "learned-supft", "S20", "lr0.01")]
        assert wide[(scen, "learned-supft", "S20", "lr0.003")] >= wide[(scen, "learned-supft", "S20", "lr0.03")]
