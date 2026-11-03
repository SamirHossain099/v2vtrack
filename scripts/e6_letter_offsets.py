"""E6: the offset diagnostic for the letter, under the letter's rules (RESEARCH.md rules 27 and 28).

The codebook slope comes from the codebook specification alone (64 beams over +-45 deg, uniform in angle:
K = 63 / (pi/2) = 40.11 beams per radian), no recording's timing is corrected, and the quantity is called the
effective offset: the angle between the front array's look direction and the GPS course that best explains the
observed beams. It absorbs mounting, heading bias and timing error together.

This is a diagnostic that reads each recording's own beams; it motivates tracking and is not the evaluation, which
uses only the source recording (scripts/e2_tracker.py --K 40.107 --no-lag).
Writes results/e6_letter_offsets.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load  # noqa: E402
from v2v.geo import C_CENTRE, causal_features, wrap  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
K = 63 / (np.pi / 2)


def fit_offset(phi, beam_in, m, d0=0.0):
    r = least_squares(lambda d: C_CENTRE + K * np.sin(wrap(d[0] - phi[m])) - beam_in[m], [d0], loss="soft_l1", f_scale=3.0)
    return float(r.x[0])


if __name__ == "__main__":
    res = {"K": K, "scenarios": {}}
    for scen in (36, 37, 38, 39):
        s = load(scen)
        f = causal_features(s)
        base = f["valid"] & f["moving"] & (s["array"] == 0)
        whole = fit_offset(f["phi"], s["beam_in"], base)
        offs, starts = [], []
        for q in np.unique(s["seq"]):
            m = base & (s["seq"] == q)
            if m.sum() < 40:
                continue
            offs.append(fit_offset(f["phi"], s["beam_in"], m, whole))
            starts.append(float(s["t"][s["seq"] == q][0]))
        offs = np.degrees(np.array(offs))
        z = np.exp(1j * np.radians(offs))
        res["scenarios"][scen] = {
            "whole_deg": float(np.degrees(whole)), "n_seq": int(len(offs)),
            "circ_std_deg": float(np.degrees(np.sqrt(-2 * np.log(np.abs(z.mean()))))),
            "p10_p90_deg": np.percentile(offs, [10, 90]).round(2).tolist(),
            "offsets_deg": offs.round(2).tolist(), "start_s": starts,
        }
        r = res["scenarios"][scen]
        print(f"scenario {scen}: whole-recording effective offset {r['whole_deg']:+.2f} deg; per-sequence std {r['circ_std_deg']:.2f} deg, "
              f"p10-p90 {r['p10_p90_deg']} over {r['n_seq']} sequences", flush=True)
    w = [res["scenarios"][s]["whole_deg"] for s in (36, 37, 38, 39)]
    res["between_recordings_span_deg"] = float(max(w) - min(w))
    print(f"span of the whole-recording offsets across recordings: {res['between_recordings_span_deg']:.2f} deg")
    (OUT / "e6_letter_offsets.json").write_text(json.dumps(res, indent=1))
