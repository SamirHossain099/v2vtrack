"""E0: reproduce the 2023 challenge baseline (external/challenge2023/competition_benchmark.py) exactly.

Samples: inside each sequence, inputs x1..x5 = frames t-8, t-6, t-4, t-2, t (0.2 s apart at 10 Hz) and the
target y = frame t+5 (0.5 s after x5). That gives 200 - 13 = 187 samples per 200-frame sequence, i.e. 23,188
for scenario 36, which is the row count of the repo's SUBMISSION-EXAMPLE_prediction.csv.

Baseline steps, copied not paraphrased: linear extrapolation of lat/lon to the target time; heading from
(extrapolated - last input) in raw lat/lon (no cos-latitude scaling); relative orientation from (gps1 - gps2);
aoa = -(ori_rel - heading - pi/4) wrapped; nearest of 256 beams uniform over [-pi, pi]; then a constant
integer shift = -round(mean of signed circular error where |error| < 5), fitted on the same data.

Published baseline numbers (challenge page, read 2026-09-29): top-1/3/5 and APL
  36: 31.18 / 62.6 / 70.41, -4.81 dB   37: 24.05 / 58.6 / 70.47, -4.10 dB
  38: 10.69 / 32.2 / 45.57, -8.12 dB   39: 22.45 / 49.18 / 60.34, -8.02 dB
Writes results/e0_challenge_baseline.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load, topk_and_apl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "challenge2023"))
OUT = Path(__file__).resolve().parents[1] / "results"
PUBLISHED = {36: (31.18, 62.6, 70.41, -4.81), 37: (24.05, 58.6, 70.47, -4.10),
             38: (10.69, 32.2, 45.57, -8.12), 39: (22.45, 49.18, 60.34, -8.02)}
X_OFFS = np.array([-8, -6, -4, -2, 0])
Y_OFF = 5


def windows(seq):
    """Index of x5 (the last input) for every valid window, and the matching target index."""
    xs, ys = [], []
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        for k in range(8, len(idx) - Y_OFF):
            xs.append(idx[k])
            ys.append(idx[k + Y_OFF])
    return np.array(xs), np.array(ys)


def baseline_predict(s, xs, delta_input=0.2, delta_output=0.5, apply_shift=True, y_true=None):
    """Vectorised copy of competition_benchmark.py steps 1-3."""
    # Step 1: linear extrapolation (interp1d with fill_value='extrapolate' on 5 points is a least-squares-free
    # linear extension of the last segment, so extrapolating from x4,x5 reproduces it exactly).
    x = delta_input * np.arange(5)
    est = {}
    for u in (1, 2):
        lat = np.stack([s[f"lat{u}"][xs + o] for o in X_OFFS], 1)
        lon = np.stack([s[f"lon{u}"][xs + o] for o in X_OFFS], 1)
        slope_lat = (lat[:, -1] - lat[:, -2]) / delta_input
        slope_lon = (lon[:, -1] - lon[:, -2]) / delta_input
        est[u] = (lat[:, -1] + slope_lat * delta_output, lon[:, -1] + slope_lon * delta_output)
    # Step 2: heading from (estimated - last input), relative orientation from (gps1 - gps2).
    # With delta_output = 0 the estimate equals the last input, so use the last input segment instead.
    if delta_output > 0:
        heading = np.arctan2(est[1][0] - s["lat1"][xs], est[1][1] - s["lon1"][xs])
    else:
        heading = np.arctan2(s["lat1"][xs] - s["lat1"][xs - 2], s["lon1"][xs] - s["lon1"][xs - 2])
    ori_rel = np.arctan2(est[1][0] - est[2][0], est[1][1] - est[2][1])
    aoa = wrap(-(ori_rel - heading - np.pi / 4))
    # Step 3: nearest of 256 uniform beams
    beam_ori = np.arange(256) / 255 * 2 * np.pi - np.pi
    order = np.argsort(np.abs(aoa[:, None] - beam_ori[None, :]), axis=1)
    pred = order[:, 0].copy()
    shift = 0
    if apply_shift:
        d = circ_signed(pred, y_true)
        shift = -int(np.round(np.mean(d[np.abs(d) < 5])))
        pred = pred + shift
        pred[pred > 255] -= 255  # the script's own wrap, kept as written
        pred[pred < 0] += 255
    return pred, shift, heading, ori_rel, aoa


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circ_signed(a, b, l=256):
    d = (a % l) - (b % l)
    d = np.where(np.abs(d) > l / 2, l - np.abs(d), d)  # the script returns a positive value in that branch
    return d


def zero_heading_flags(s, xs):
    """The script's compute_ori_from_pos_delta returns 0 when both deltas are exactly zero."""
    return (s["lat1"][xs] == s["lat1"][xs - 2]) & (s["lon1"][xs] == s["lon1"][xs - 2])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    res = {}
    for scen in (36, 37, 38, 39):
        s = load(scen)
        xs, ys = windows(s["seq"])
        y = s["beam"][ys]
        pred, shift, heading, ori_rel, aoa = baseline_predict(s, xs, y_true=y)
        m = topk_and_apl(pred, s, ys)
        pred0, _, *_ = baseline_predict(s, xs, apply_shift=False)
        m0 = topk_and_apl(pred0, s, ys)
        # the same map on the current frame (no 500 ms look-ahead), to separate map error from prediction error
        pred_now, shift_now, *_ = baseline_predict(s, xs, delta_output=0.0, y_true=s["beam"][xs])
        m_now = topk_and_apl(pred_now, s, xs)
        speed = np.hypot(s["e1"][xs] - s["e1"][xs - 2], s["n1"][xs] - s["n1"][xs - 2]) / 0.2
        mov, still = speed > 2.0, speed <= 2.0
        r = {
            "n_windows": int(len(xs)), "shift": shift,
            "reproduced": m, "published": dict(zip(("top1", "top3", "top5", "apl_db"), PUBLISHED[scen])),
            "no_shift": m0, "current_frame_same_map": m_now, "shift_current_frame": shift_now,
            "moving_gt2mps": topk_and_apl(pred[mov], s, ys[mov]),
            "stationary_le2mps": topk_and_apl(pred[still], s, ys[still]),
            "frac_stationary": float(still.mean()),
        }
        res[scen] = r
        pub = PUBLISHED[scen]
        print(f"scenario {scen}: n={len(xs)} shift={shift:+d}  "
              f"top1/3/5 {100*m['top1']:.2f}/{100*m['top3']:.1f}/{100*m['top5']:.2f} APL {m['apl_db']:.2f} dB   "
              f"published {pub[0]}/{pub[1]}/{pub[2]} {pub[3]} dB   "
              f"| current-frame {100*m_now['top1']:.1f}/{100*m_now['top3']:.1f}/{100*m_now['top5']:.1f}  "
              f"| moving {100*r['moving_gt2mps']['top1']:.1f}  still {100*r['stationary_le2mps']['top1']:.1f} "
              f"({100*r['frac_stationary']:.0f}% still)", flush=True)
    (OUT / "e0_challenge_baseline.json").write_text(json.dumps(res, indent=1))
