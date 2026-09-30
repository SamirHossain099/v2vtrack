"""Pins for E5 (continual stream). Numbers read from results/e5_continual_src36.json."""
import json
from pathlib import Path

import pytest

RES = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def e5():
    return json.loads((RES / "e5_continual_src36.json").read_text())


def test_order_is_the_documented_one(e5):
    assert e5["order"] == [36, 37, 38, 39, 36]


def test_tracker_matches_standalone_streams_and_loses_nothing_on_return(e5):
    e2 = json.loads((RES / "e2_tracker_src36_h0.json").read_text())["runs"]
    v = e5["runs"]["kalman|S20|L0"]
    for r in v[1:4]:
        assert abs(r["top1"] - e2[f"{r['scenario']}|kalman|S20|L0"]["top1"]) < 0.03
    assert abs(v[4]["top1"] - v[0]["top1"]) < 0.02


def test_tracker_beats_static_on_every_off_domain_visit(e5):
    st, tr = e5["runs"]["static|S0|L0"], e5["runs"]["kalman|S20|L0"]
    for a, b in zip(st[1:4], tr[1:4]):
        assert b["top1"] > 2.5 * a["top1"] and b["apl_db"] > a["apl_db"] + 0.5
