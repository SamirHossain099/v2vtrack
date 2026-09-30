"""Gate check: does the best beam follow the relative bearing between the two vehicles?

Model-free. For each frame: bearing from unit1 (receiver, 4 arrays) to unit2 (transmitter) from the
two GPS fixes, unit1 heading as GPS course over ground (the dataset has no heading field), and
phi = bearing - heading. Then per scenario:
  - which array is best as a function of phi (should be four ~90 degree sectors),
  - per array, a fit  beam = c0 + K * sin(phi - theta)  on frames where that array is best,
  - top-1 / top-3 / top-5 of the two-stage geometric predictor, in-sample and across scenarios,
  - how the answer depends on speed (course over ground is undefined when stationary).
Writes results/sanity_geometry.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DATA = Path(r"N:\Datasets\DeepSense 6G\extracted")
OUT = Path(__file__).resolve().parents[1] / "results"
R = 6371000.0
SCENARIOS = [36, 37, 38, 39]
NOMINAL = np.radians([0.0, 90.0, 180.0, -90.0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(scen):
    d = pd.read_csv(DATA / f"scenario{scen}" / f"scenario{scen}.csv", low_memory=False)
    lat0, lon0 = np.radians(d["unit1_gps1_lat"].mean()), np.radians(d["unit1_gps1_lon"].mean())
    out = {"seq": d["seq_index"].to_numpy()}
    for u in (1, 2):
        lat, lon = np.radians(d[f"unit{u}_gps1_lat"].to_numpy()), np.radians(d[f"unit{u}_gps1_lon"].to_numpy())
        out[f"e{u}"] = R * (lon - lon0) * np.cos(lat0)
        out[f"n{u}"] = R * (lat - lat0)
    t = pd.to_datetime(d["timestamp"], format="%H-%M-%S.%f")
    out["t"] = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    out["beams"] = np.stack([d[f"unit1_pwr{a}_best-beam"].to_numpy() for a in (1, 2, 3, 4)], 1)
    out["maxpwr"] = np.stack([d[f"unit1_pwr{a}_max-pwr"].to_numpy() for a in (1, 2, 3, 4)], 1)
    out["array"] = np.nan_to_num(out["maxpwr"], nan=-1.0).argmax(1)
    out["beam"] = out["beams"][np.arange(len(d)), out["array"]]
    out["overall_csv"] = d["unit1_overall-beam"].to_numpy(dtype=float)
    out["overall"] = out["array"] * 64 + out["beam"]
    return out


def course(s, half=5):
    """Heading of unit1 from displacement over +-half frames inside a sequence, and speed."""
    n = len(s["t"])
    hd, sp = np.full(n, np.nan), np.full(n, np.nan)
    for q in np.unique(s["seq"]):
        idx = np.flatnonzero(s["seq"] == q)
        if len(idx) < 2 * half + 1:
            continue
        a, b = idx[:-2 * half], idx[2 * half:]
        de, dn = s["e1"][b] - s["e1"][a], s["n1"][b] - s["n1"][a]
        dt = s["t"][b] - s["t"][a]
        mid = idx[half:-half]
        hd[mid] = np.arctan2(de, dn)
        sp[mid] = np.hypot(de, dn) / np.where(dt > 0, dt, np.nan)
    return hd, sp


def features(s):
    hd, sp = course(s)
    brg = np.arctan2(s["e2"] - s["e1"], s["n2"] - s["n1"])
    s["phi"] = wrap(brg - hd)
    s["speed"] = sp
    s["dist"] = np.hypot(s["e2"] - s["e1"], s["n2"] - s["n1"])
    s["ok"] = np.isfinite(s["phi"]) & (sp > 2.0) & np.isfinite(s["maxpwr"]).all(1)
    return s


def fit_sector_centres(phi, arr):
    """Nominal mounting (front, right, back, left) plus one shared offset fitted as a circular mean.

    Per-array circular means are unusable for the side arrays: they are best in only a few percent
    of frames, many of them reflections, so their mean lands far from the true boresight.
    """
    delta = np.angle(np.exp(1j * wrap(phi - NOMINAL[arr])).mean())
    return wrap(NOMINAL + delta)


def fit_beam(phi, beam, centre):
    x = wrap(phi - centre)

    def res(p):
        return p[0] + p[1] * np.sin(x - p[2]) - beam

    r = least_squares(res, [32.0, -32.0, 0.0], loss="soft_l1", f_scale=3.0)
    return r.x


def predict(phi, centres, params):
    arr = np.abs(wrap(phi[:, None] - centres[None, :])).argmin(1)
    p = params[arr]
    beam = p[:, 0] + p[:, 1] * np.sin(wrap(phi - centres[arr]) - p[:, 2])
    return arr, np.clip(np.rint(beam), 0, 63)


def score(s, m, centres, params):
    phi, arr, beam = s["phi"][m], s["array"][m], s["beam"][m]
    pa, pb = predict(phi, centres, params)
    hit_a = pa == arr
    err = np.abs(pb - beam)
    return {
        "n": int(m.sum()),
        "array_acc": float(hit_a.mean()),
        "top1": float((hit_a & (err == 0)).mean()),
        "top3": float((hit_a & (err <= 1)).mean()),
        "top5": float((hit_a & (err <= 2)).mean()),
        "beam_mae_given_array": float(err[hit_a].mean()),
        "beam_within5_given_array": float((err[hit_a] <= 5).mean()),
    }


def fit(s, m):
    centres = fit_sector_centres(s["phi"][m], s["array"][m])
    params = np.zeros((4, 3))
    for a in range(4):
        k = m & (s["array"] == a)
        params[a] = fit_beam(s["phi"][k], s["beam"][k], centres[a])
    return centres, params


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    S = {k: features(load(k)) for k in SCENARIOS}
    res = {"scenarios": {}, "transfer": {}}
    fits = {}
    for k, s in S.items():
        m = s["ok"]
        centres, params = fit(s, m)
        fits[k] = (centres, params)
        consistent = float(np.nanmean(s["overall_csv"] == s["overall"]))
        r = {
            "frames": int(len(m)), "sequences": int(len(np.unique(s["seq"]))),
            "moving_frames": int(m.sum()),
            "frame_rate_hz": float(1 / np.median(np.diff(s["t"])[np.diff(s["t"]) > 0])),
            "dist_m_median": float(np.nanmedian(s["dist"])), "dist_m_p95": float(np.nanpercentile(s["dist"], 95)),
            "speed_mps_median": float(np.nanmedian(s["speed"])),
            "overall_beam_is_array_times_64_plus_beam": consistent,
            "array_share": [float((s["array"][m] == a).mean()) for a in range(4)],
            "sector_centre_deg": np.degrees(centres).round(1).tolist(),
            "beam_fit_c0_K_theta_deg": [[round(p[0], 2), round(p[1], 2), round(float(np.degrees(p[2])), 2)] for p in params],
            "in_sample": score(s, m, centres, params),
            "majority_array_acc": float(max((s["array"][m] == a).mean() for a in range(4))),
            "majority_beam_top1": float(np.bincount(s["overall"][m]).max() / m.sum()),
            "by_speed": {}, "by_distance": {},
        }
        for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 99)]:
            mm = np.isfinite(s["phi"]) & (s["speed"] >= lo) & (s["speed"] < hi)
            if mm.sum() > 50:
                r["by_speed"][f"{lo}-{hi}"] = score(s, mm, centres, params)
        for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 1e4)]:
            mm = m & (s["dist"] >= lo) & (s["dist"] < hi)
            if mm.sum() > 50:
                r["by_distance"][f"{lo}-{int(hi)}"] = score(s, mm, centres, params)
        res["scenarios"][k] = r
        print(f"\n== scenario {k}\n" + json.dumps(r, indent=1), flush=True)
    for a in SCENARIOS:
        for b in SCENARIOS:
            if a != b:
                res["transfer"][f"{a}->{b}"] = score(S[b], S[b]["ok"], *fits[a])
    print("\n== transfer (fit on a, score on b)")
    for k, v in res["transfer"].items():
        print(f"  {k}: array {v['array_acc']:.3f}  top1 {v['top1']:.3f}  top3 {v['top3']:.3f}  top5 {v['top5']:.3f}")
    (OUT / "sanity_geometry.json").write_text(json.dumps(res, indent=1, default=float))
