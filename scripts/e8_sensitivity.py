"""E8: sensitivity of the tracker to every hand-set value, under the letter's settings (specification slope, no timing
correction, source-only offset). Each value moves on its own by about a factor of two either way, at two budgets:
local search only (L2), and a full sweep every 100 frames (S100). The 20-degree gate cannot fire at L2, where a reading
stays within a few beams of the map, so S100 is what tests it. The search width is a budget, moved at L2 only.
Writes results/e8_sensitivity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load  # noqa: E402
from v2v.geo import causal_features  # noqa: E402
from v2v.stream import run  # noqa: E402
from v2v.track import KalmanDelta  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
E2 = json.loads((OUT / "e2_tracker_letter_nolabelK_nolag.json").read_text())
K, D0 = E2["K"], np.radians(E2["delta0_deg"])
BASE = {"v_min": 2.0, "back": 5, "sigma": 0.7, "q": 0.5, "local_k": 2, "hold_max": 50, "reject_deg": 20.0, "P0": 5.0}
GRID = {"v_min": [1.0, 4.0], "back": [3, 10], "sigma": [0.35, 1.4], "q": [0.25, 1.0], "local_k": [1, 3], "hold_max": [25, 100],
        "reject_deg": [10.0, 40.0], "P0": [2.5, 10.0]}
BUDGETS = {"L2": 0, "S100": 100}


def one(s, cfg, budget):
    f = causal_features(s, back=cfg["back"], v_min=cfg["v_min"])
    m = KalmanDelta(K, delta0=D0, sigma_pos_m=cfg["sigma"], q_deg_per_sqrt_s=cfg["q"], hold_max_frames=cfg["hold_max"],
                    reject_deg=cfg["reject_deg"], P0_deg=cfg["P0"])
    return run(m, s, f, sweep_every=BUDGETS[budget], local_k=cfg["local_k"] if budget == "L2" else 0)


if __name__ == "__main__":
    res = {"base": BASE, "budgets": BUDGETS, "runs": {}}
    data = {k: load(k) for k in (36, 37, 38, 39)}
    for scen, s in data.items():
        for budget in BUDGETS:
            r = one(s, BASE, budget)
            res["runs"][f"{scen}|{budget}|base"] = r
            print(f"{scen} {budget} base top1 {r['top1']:.3f} APL {r['apl_db']:.2f}", flush=True)
            for p, vals in GRID.items():
                if p == "local_k" and budget != "L2":
                    continue
                for v in vals:
                    r = one(s, {**BASE, p: v}, budget)
                    res["runs"][f"{scen}|{budget}|{p}={v}"] = r
                    print(f"{scen} {budget} {p}={v}: top1 {r['top1']:.3f} APL {r['apl_db']:.2f}", flush=True)
    (OUT / "e8_sensitivity.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
