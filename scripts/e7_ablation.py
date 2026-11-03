"""E7: ablation (36 = source, where the design is chosen; 37-39 report it) of the tracker's three rules, under the letter's settings (label-free K, no timing correction,
source-only offset). Each rule is removed on its own, at four measurements per frame (local search only) and at a
full sweep every 100 frames. Writes results/e7_ablation.json.
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
VARIANTS = {"full": {}, "edge maxima used": {"require_interior": False}, "no agreement gate": {"agree_local": None},
            "no standstill hold": {"hold_still": False}, "edge maxima, no agreement gate": {"require_interior": False, "agree_local": None}}
SETTINGS = {"L2": (0, 2), "S100": (100, 0)}

if __name__ == "__main__":
    res = {"K": K, "delta0_deg": float(np.degrees(D0)), "runs": {}}
    scens = [int(a) for a in sys.argv[1:]] or [36, 37, 38, 39]
    for scen in scens:
        s = load(scen)
        f = causal_features(s)
        for name, kw in VARIANTS.items():
            for tag, (S, L) in SETTINGS.items():
                r = run(KalmanDelta(K, delta0=D0, **kw), s, f, sweep_every=S, local_k=L)
                res["runs"][f"{scen}|{name}|{tag}"] = r
                print(f"{scen} {name:20s} {tag:5s} top1 {r['top1']:.3f} APL {r['apl_db']:6.2f} still {r['stationary']['top1']:.3f} "
                      f"delta {r['delta_final_deg']:+.1f}", flush=True)
    # design choice made on the source recording only (rule 27); the deployment streams report what that choice gives
    (OUT / "e7_ablation.json").write_text(json.dumps(res, indent=1, default=float))
