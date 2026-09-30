"""E2: the tracker against the static map and the offset baselines, per scenario, as a stream.

Source of the fixed parameters: scenario 36 (intercity day) unless --source says otherwise. K is the joint
codebook slope from E1d; delta0 is 36's fitted front boresight. Streams: 37, 38, 39 (and 36 itself as the
in-domain reference), frames in time order, lag-corrected per E1d (--no-lag to switch off).
Grid: sweep_every in {0, 10, 20, 50, 100, 200} x local_k in {0, 2}. Horizon 0 (current frame) by default;
--horizon 5 reproduces the challenge's 500 ms look-ahead.
Writes results/e2_tracker_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load  # noqa: E402
from v2v.geo import apply_lag, causal_features  # noqa: E402
from v2v.stream import run  # noqa: E402
from v2v.track import METHODS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=36)
    ap.add_argument("--targets", type=int, nargs="+", default=[36, 37, 38, 39])
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--no-lag", action="store_true")
    ap.add_argument("--sweeps", type=int, nargs="+", default=[0, 10, 20, 50, 100, 200])
    ap.add_argument("--local", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    e1 = json.loads((OUT / "e1d_final.json").read_text())
    K = e1["K_joint"]
    lags = {int(k): v for k, v in e1["lag_frames"].items()}
    delta0 = np.radians(e1["scenarios"][str(a.source)]["boresight_deg"][0])
    tag = a.tag or f"src{a.source}_h{a.horizon}{'_nolag' if a.no_lag else ''}"
    res = {"K": K, "delta0_deg": float(np.degrees(delta0)), "source": a.source, "horizon": a.horizon, "lag": not a.no_lag, "runs": {}}
    for scen in a.targets:
        s = load(scen)
        if not a.no_lag:
            s = apply_lag(s, lags[scen])
        f = causal_features(s, horizon=a.horizon)
        for name in a.methods:
            for S in a.sweeps:
                for lk in a.local:
                    if name in ("static",) and (S or lk):
                        continue  # static ignores feedback
                    if name in ("beam-hold", "index-offset-ma") and not (S or lk):
                        continue  # nothing to hold or track without feedback
                    m = METHODS[name](K, delta0=delta0)
                    r = run(m, s, f, sweep_every=S, local_k=lk)
                    key = f"{scen}|{name}|S{S}|L{lk}"
                    res["runs"][key] = r
                    print(f"{key:34s} top1/3/5 {r['top1']:.3f}/{r['top3']:.3f}/{r['top5']:.3f} APL {r['apl_db']:6.2f} dB  "
                          f"moving {r.get('moving', {}).get('top1', float('nan')):.3f}  still {r.get('stationary', {}).get('top1', float('nan')):.3f}  "
                          f"sweeps {r['sweep_frac']:.3f}  delta {r['delta_final_deg']:+.1f}  {r['latency_ms']:.3f} ms", flush=True)
    (OUT / f"e2_tracker_{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
