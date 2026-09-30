"""E1b: (i) the GPS-power time lag, per unit and per sequence, over a wider range; (ii) transfer of the
pinned physical model (c = 31.5, shared K, per-array boresight) between scenarios, and how much a
single refitted parameter recovers.
Writes results/e1b_sync_transfer.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v2v.data import bearing_1_to_2, course, load, topk_and_apl, wrap  # noqa: E402
from e1_offset_origin import C_CENTRE, NOMINAL, fit_shared  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
SCEN = (36, 37, 38, 39)


def shifted(x, seq, k):
    """x taken k frames later (k > 0) inside each sequence; NaN where unavailable."""
    y = np.full_like(x, np.nan)
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        if k > 0:
            y[idx[:len(idx) - k]] = x[idx[k:]]
        elif k < 0:
            y[idx[-k:]] = x[idx[:len(idx) + k]]
        else:
            y[idx] = x[idx]
    return y


def features(s, k1=0, k2=0):
    """phi with unit1's track shifted by k1 frames and unit2's by k2."""
    e1, n1 = shifted(s["e1"], s["seq"], k1), shifted(s["n1"], s["seq"], k1)
    e2, n2 = shifted(s["e2"], s["seq"], k2), shifted(s["n2"], s["seq"], k2)
    t = s["t"]
    hd, sp = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for q in np.unique(s["seq"]):
        idx = np.flatnonzero(s["seq"] == q)
        a, b, mid = idx[:-10], idx[10:], idx[5:-5]
        de, dn = e1[b] - e1[a], n1[b] - n1[a]
        hd[mid] = np.arctan2(dn, de)
        sp[mid] = np.hypot(de, dn) / (t[b] - t[a])
    phi = wrap(np.arctan2(n2 - n1, e2 - e1) - hd)
    ok = np.isfinite(phi) & (sp > 2.0)
    return phi, ok, sp


def rms_of(s, phi, ok):
    p = fit_shared(phi, s["array"], s["beam_in"], ok)
    r = s["beam_in"] - (p[0] + p[1] * np.sin(wrap(p[2:6][s["array"]] - phi)))
    return float(np.sqrt(np.mean(np.minimum(r[ok] ** 2, 100)))), p


def predict(s, phi, p):
    """Two-stage geometric prediction with parameters p = [c, K, b1..b4]: array by nearest boresight, beam by the map."""
    b = p[2:6]
    arr = np.abs(wrap(b[None, :] - phi[:, None])).argmin(1)
    beam_in = np.clip(np.rint(p[0] + p[1] * np.sin(wrap(b[arr] - phi))), 0, 63).astype(int)
    return arr * 64 + beam_in


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    S = {k: load(k) for k in SCEN}
    res = {"sync": {}, "transfer": {}}

    # (i) lag search: both units together, then each unit alone, over -20..+20 frames
    for scen, s in S.items():
        cur = {"both": {}, "unit1": {}, "unit2": {}}
        for k in range(-20, 21, 2):
            for tag, (k1, k2) in (("both", (k, k)), ("unit1", (k, 0)), ("unit2", (0, k))):
                phi, ok, _ = features(s, k1, k2)
                cur[tag][k] = rms_of(s, phi, ok)[0]
        best = {tag: min(c, key=c.get) for tag, c in cur.items()}
        res["sync"][scen] = {"best_shift_frames": best, "curve": cur}
        print(f"scenario {scen}: best lag both {best['both']:+d}  unit1 {best['unit1']:+d}  unit2 {best['unit2']:+d} frames;  "
              f"rms(both) at 0 {cur['both'][0]:.2f}, at best {cur['both'][best['both']]:.2f}", flush=True)

    # (ii) transfer of the pinned model
    fits, feats = {}, {}
    for scen, s in S.items():
        phi, ok, sp = features(s)
        feats[scen] = (phi, ok, sp)
        fits[scen] = rms_of(s, phi, ok)[1]
    for a in SCEN:
        for b in SCEN:
            s, (phi, ok, sp) = S[b], feats[b]
            idx = np.flatnonzero(ok)
            pred_src = predict(s, phi, fits[a])
            m_src = topk_and_apl(pred_src[idx], s, idx)
            # refit one parameter: a common rotation of all four boresights (the box), K and c kept from a
            p = fits[a].copy()
            f = lambda d: (C_CENTRE + p[1] * np.sin(wrap((p[2:6] + d[0])[s["array"]] - phi)) - s["beam_in"])[ok]
            d = least_squares(f, [0.0], loss="soft_l1", f_scale=3.0).x[0]
            p_rot = p.copy(); p_rot[2:6] = p[2:6] + d
            m_rot = topk_and_apl(predict(s, phi, p_rot)[idx], s, idx)
            m_own = topk_and_apl(predict(s, phi, fits[b])[idx], s, idx)
            res["transfer"][f"{a}->{b}"] = {"source": m_src, "rotation_refit": m_rot, "rotation_deg": float(np.degrees(d)),
                                             "own_fit": m_own, "K_src": float(fits[a][1]), "K_tgt": float(fits[b][1])}
            print(f"  {a}->{b}: source top1/3/5 {m_src['top1']:.3f}/{m_src['top3']:.3f}/{m_src['top5']:.3f} APL {m_src['apl_db']:.2f} | "
                  f"+rotation ({np.degrees(d):+.1f} deg) {m_rot['top1']:.3f}/{m_rot['top3']:.3f}/{m_rot['top5']:.3f} APL {m_rot['apl_db']:.2f} | "
                  f"own {m_own['top1']:.3f}/{m_own['top3']:.3f}/{m_own['top5']:.3f} APL {m_own['apl_db']:.2f}", flush=True)
    res["fits"] = {k: {"c": float(v[0]), "K": float(v[1]), "boresight_deg": np.degrees(wrap(v[2:6])).round(2).tolist()} for k, v in fits.items()}
    (OUT / "e1b_sync_transfer.json").write_text(json.dumps(res, indent=1, default=float))
