"""E1c: is the scenario-37 GPS/power lag constant through the recording, and is it in the index rows?

For each sequence, shift both GPS tracks by k frames (k in -4..20) against the power rows, score the
scenario's own pinned model, and take the k with the smallest robust rms. Also compares the row
timestamp against the GPS and power file timestamps per sequence.
Writes results/e1c_lag_per_seq.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v2v.data import load, wrap  # noqa: E402
from e1_offset_origin import fit_shared  # noqa: E402
from e1b_sync_transfer import features  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
KS = list(range(-4, 21))


def seq_rms(sub, p, k):
    phi, ok, _ = features(sub, k, k)
    if ok.sum() < 60:
        return np.nan
    r = sub["beam_in"] - (p[0] + p[1] * np.sin(wrap(p[2:6][sub["array"]] - phi)))
    return float(np.sqrt(np.mean(np.minimum(r[ok] ** 2, 100))))


if __name__ == "__main__":
    res = {}
    for scen in (36, 37, 38, 39):
        s = load(scen)
        phi, ok, _ = features(s)
        p = fit_shared(phi, s["array"], s["beam_in"], ok)
        rows = []
        for q in np.unique(s["seq"]):
            m = s["seq"] == q
            sub = {k: v[m] for k, v in s.items() if isinstance(v, np.ndarray) and v.shape[:1] == s["seq"].shape}
            curve = [seq_rms(sub, p, k) for k in KS]
            if np.all(np.isnan(curve)):
                continue
            kb = KS[int(np.nanargmin(curve))]
            rows.append({"seq": int(q), "best_k": kb, "rms0": curve[KS.index(0)], "rms_best": float(np.nanmin(curve)),
                         "gain": curve[KS.index(0)] - float(np.nanmin(curve)),
                         "stamp_gps1_minus_pwr": float(np.median(sub["t_gps1"] - sub["t_pwr"])),
                         "t_start_s": float(sub["t"][0]), "speed": float(np.nanmedian(np.hypot(np.diff(sub["e1"]), np.diff(sub["n1"])) * 10))})
        ks = np.array([r["best_k"] for r in rows])
        gains = np.array([r["gain"] for r in rows])
        res[scen] = {"n_seq": len(rows), "median_best_k": float(np.median(ks)),
                     "hist_best_k": {int(k): int((ks == k).sum()) for k in np.unique(ks)},
                     "frac_gain_over_0p3": float((gains > 0.3).mean()), "rows": rows}
        print(f"scenario {scen}: sequences {len(rows)}, median best lag {np.median(ks):+.0f} frames, "
              f"best-lag histogram {res[scen]['hist_best_k']}, gain>0.3 in {100*res[scen]['frac_gain_over_0p3']:.0f}% of sequences", flush=True)
        if scen == 37:
            big = [r for r in rows if r["gain"] > 0.3]
            print("   37 sequences with a real lag: best_k by start time (s):",
                  [(round(r["t_start_s"]), r["best_k"]) for r in big][:40])
    (OUT / "e1c_lag_per_seq.json").write_text(json.dumps(res, indent=1, default=float))
