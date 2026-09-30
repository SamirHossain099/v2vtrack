"""E5: one continual stream 36 -> 37 -> 38 -> 39 -> 36 (return visit), tracker state carried across.

Per visit: the scores as in E2, plus the recovery profile after each switch: top-1 and |delta - delta_own| in
the first 0-10 s, 10-30 s, 30-60 s and after 60 s of each visit, where delta_own is the scenario's own
offline boresight from E1d. Static and the offset baselines run on the same stream.
Writes results/e5_continual_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load  # noqa: E402
from v2v.geo import N_BEAMS, apply_lag, causal_features, wrap  # noqa: E402
from v2v.track import METHODS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
ORDER = [36, 37, 38, 39, 36]
WINDOWS = [(0, 10), (10, 30), (30, 60), (60, 1e9)]


def run_visit(method, s, f, sweep_every, local_k, delta_own):
    idx = np.flatnonzero(f["valid"])
    flat = s["pwr"].reshape(len(s["pwr"]), -1)
    hit1, apl, derr, tsec = [], [], [], []
    t0 = s["t"][idx[0]]
    for j, i in enumerate(idx):
        tgt = f["target"][i]
        cand = np.asarray(method.step(i, f)).ravel()
        served = int(cand[0])
        pw = flat[tgt]
        best = int(pw.argmax())
        hit1.append(served == best)
        apl.append(10 * np.log10(pw[served] / pw[best]))
        derr.append(abs(np.degrees(wrap(getattr(method, "delta", 0.0) - delta_own))))
        tsec.append(s["t"][tgt] - t0)
        fb = {"t": s["t"][tgt], "p_served": pw[served]}
        if sweep_every and j % sweep_every == 0:
            fb["sweep_beam"] = best
        if local_k:
            arr, bi = served // N_BEAMS, served % N_BEAMS
            lo, hi = max(0, bi - local_k), min(N_BEAMS - 1, bi + local_k)
            loc = np.arange(lo, hi + 1) + arr * N_BEAMS
            fb["local_best"] = int(loc[pw[loc].argmax()])
            fb["local_interior"] = bool(lo < fb["local_best"] % N_BEAMS < hi)
        method.feedback(i, f, fb)
    hit1, apl, derr, tsec = map(np.array, (hit1, apl, derr, tsec))
    out = {"n": int(len(idx)), "top1": float(hit1.mean()), "apl_db": float(apl.mean()), "delta_err_final_deg": float(derr[-1]),
           "recovery": {}}
    for lo, hi in WINDOWS:
        m = (tsec >= lo) & (tsec < hi)
        if m.any():
            out["recovery"][f"{lo}-{int(min(hi, 9999))}s"] = {"top1": float(hit1[m].mean()), "apl_db": float(apl[m].mean()),
                                                              "delta_err_deg_median": float(np.median(derr[m])), "n": int(m.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--local", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--methods", nargs="+", default=["static", "index-offset-ma", "robust-window", "kalman"])
    ap.add_argument("--tag", default="src36")
    a = ap.parse_args()
    e1 = json.loads((OUT / "e1d_final.json").read_text())
    K, lags = e1["K_joint"], {int(k): v for k, v in e1["lag_frames"].items()}
    own = {int(k): np.radians(v["boresight_deg"][0]) for k, v in e1["scenarios"].items()}
    data = {}
    for scen in set(ORDER):
        s = apply_lag(load(scen), lags[scen])
        data[scen] = (s, causal_features(s))
    res = {"order": ORDER, "runs": {}}
    for name in a.methods:
        for S in (a.sweeps if name != "static" else [0]):
            for L in (a.local if name != "static" else [0]):
                m = METHODS[name](K, delta0=own[36])
                visits = []
                for v, scen in enumerate(ORDER):
                    s, f = data[scen]
                    r = run_visit(m, s, f, S, L, own[scen])
                    r["scenario"] = scen
                    visits.append(r)
                key = f"{name}|S{S}|L{L}"
                res["runs"][key] = visits
                print(f"{key:26s} " + "  ".join(f"{scen}:{r['top1']:.2f}/{r['apl_db']:5.2f}" for scen, r in zip(ORDER, visits)) +
                      "   first 10 s top1: " + " ".join(f"{r['recovery'].get('0-10s', {}).get('top1', float('nan')):.2f}" for r in visits), flush=True)
    (OUT / f"e5_continual_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
