"""E1d: the offset analysis done properly. Per scenario: (1) GPS-to-power lag by a fine search; (2) with the
lag applied, one codebook slope K fitted jointly across all four scenarios (front array, per-scenario
boresight free); (3) per-array boresights with c and K pinned; (4) per-sequence front-array boresight
spread; (5) transfer table source / +one rotation / own; (6) the challenge baseline on lag-corrected rows.
Writes results/e1d_final.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v2v.data import load, topk_and_apl, wrap  # noqa: E402
from e1_offset_origin import C_CENTRE, NOMINAL  # noqa: E402
from e1b_sync_transfer import features, predict  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
SCEN = (36, 37, 38, 39)


def fit_b(phi, arr, beam_in, ok, K, b0=None, bound=45.0):
    """Per-array boresights with c, K pinned."""
    b0 = NOMINAL if b0 is None else b0
    f = lambda b: (C_CENTRE + K * np.sin(wrap(b[arr] - phi)) - beam_in)[ok]
    r = least_squares(f, b0, bounds=(b0 - np.radians(bound), b0 + np.radians(bound)), loss="soft_l1", f_scale=3.0)
    return r.x


def robust_rms(r):
    return float(np.sqrt(np.mean(np.minimum(r ** 2, 100))))


def lag_search(s, K):
    best, curve = None, {}
    for k in range(-4, 21):
        phi, ok, _ = features(s, k, k)
        b = fit_b(phi, s["array"], s["beam_in"], ok, K)
        r = s["beam_in"] - (C_CENTRE + K * np.sin(wrap(b[s["array"]] - phi)))
        curve[k] = robust_rms(r[ok])
    k = min(curve, key=curve.get)
    return k, curve


if __name__ == "__main__":
    S = {k: load(k) for k in SCEN}
    # pass 1: lag with a provisional K, then a joint K on the front array with lag applied
    K0 = 40.0
    lags = {k: lag_search(s, K0)[0] for k, s in S.items()}
    F = {k: features(s, lags[k], lags[k]) for k, s in S.items()}

    def joint_res(p):
        K, bs = p[0], p[1:]
        out = []
        for i, k in enumerate(SCEN):
            s, (phi, ok, _) = S[k], F[k]
            m = ok & (s["array"] == 0)
            out.append(C_CENTRE + K * np.sin(wrap(bs[i] - phi[m])) - s["beam_in"][m])
        return np.concatenate(out)
    rj = least_squares(joint_res, np.r_[K0, np.zeros(len(SCEN))], loss="soft_l1", f_scale=3.0)
    K = float(rj.x[0])
    # pass 2: lag again with the final K (cheap, and guards against K0 having steered it)
    lag_curves = {k: lag_search(s, K) for k, s in S.items()}
    lags = {k: v[0] for k, v in lag_curves.items()}
    F = {k: features(s, lags[k], lags[k]) for k, s in S.items()}
    res = {"K_joint": K, "c": C_CENTRE, "lag_frames": lags, "lag_rms_curve": {k: v[1] for k, v in lag_curves.items()}, "scenarios": {}, "transfer": {}}
    print(f"joint codebook slope K = {K:.2f} beams/rad ; lags {lags}")

    fits = {}
    for k, s in S.items():
        phi, ok, sp = F[k]
        b = fit_b(phi, s["array"], s["beam_in"], ok, K)
        fits[k] = np.r_[C_CENTRE, K, b]
        r = s["beam_in"] - (C_CENTRE + K * np.sin(wrap(b[s["array"]] - phi)))
        # per-sequence front boresight
        offs = []
        for q in np.unique(s["seq"]):
            m = ok & (s["seq"] == q) & (s["array"] == 0)
            if m.sum() < 40:
                continue
            f = lambda d: C_CENTRE + K * np.sin(wrap(d[0] - phi[m])) - s["beam_in"][m]
            offs.append(least_squares(f, [b[0]], loss="soft_l1", f_scale=3.0).x[0])
        offs = np.array(offs)
        idx = np.flatnonzero(ok)
        own = topk_and_apl(predict(s, phi, fits[k])[idx], s, idx)
        res["scenarios"][k] = {
            "boresight_deg": np.degrees(wrap(b)).round(2).tolist(),
            "back_minus_front_deg": float(np.degrees(wrap(b[2] - b[0] - np.pi))),
            "left_minus_front_deg": float(np.degrees(wrap(b[3] - b[0] - np.pi / 2))),
            "right_minus_front_deg": float(np.degrees(wrap(b[1] - b[0] + np.pi / 2))),
            "resid_rms": robust_rms(r[ok]), "resid_mad": float(np.median(np.abs(r[ok]))),
            "array_share": [float((s["array"][ok] == a).mean()) for a in range(4)],
            "per_seq_front": {"n": int(len(offs)), "mean_deg": float(np.degrees(np.angle(np.exp(1j * offs).mean()))),
                              "circ_std_deg": float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.exp(1j * offs).mean()))))),
                              "p10_p90_deg": np.degrees(np.percentile(offs, [10, 90])).round(2).tolist(),
                              "offsets_deg": np.degrees(offs).round(2).tolist()},
            "own_fit_metrics": own,
        }
        q = res["scenarios"][k]
        print(f"scenario {k} (lag {lags[k]:+d}): boresight {q['boresight_deg']}  back-front-180 {q['back_minus_front_deg']:+.2f}  "
              f"left-front-90 {q['left_minus_front_deg']:+.2f}  right-front+90 {q['right_minus_front_deg']:+.2f}  rms {q['resid_rms']:.2f}  "
              f"per-seq front mean {q['per_seq_front']['mean_deg']:+.2f} std {q['per_seq_front']['circ_std_deg']:.2f} p10-p90 {q['per_seq_front']['p10_p90_deg']}  "
              f"own top1/3/5 {own['top1']:.3f}/{own['top3']:.3f}/{own['top5']:.3f} APL {own['apl_db']:.2f}")

    for a in SCEN:
        for b_ in SCEN:
            if a == b_:
                continue
            s, (phi, ok, _) = S[b_], F[b_]
            idx = np.flatnonzero(ok)
            src = topk_and_apl(predict(s, phi, fits[a])[idx], s, idx)
            p = fits[a].copy()
            f = lambda d: (C_CENTRE + K * np.sin(wrap((p[2:6] + d[0])[s["array"]] - phi)) - s["beam_in"])[ok]
            d = float(least_squares(f, [0.0], loss="soft_l1", f_scale=3.0).x[0])
            p_rot = p.copy(); p_rot[2:6] = p[2:6] + d
            rot = topk_and_apl(predict(s, phi, p_rot)[idx], s, idx)
            own = res["scenarios"][b_]["own_fit_metrics"]
            res["transfer"][f"{a}->{b_}"] = {"source": src, "rotation_refit": rot, "rotation_deg": float(np.degrees(d)), "own": own}
            print(f"  {a}->{b_}: source {src['top1']:.3f}/{src['top3']:.3f}/{src['top5']:.3f} APL {src['apl_db']:.2f} | "
                  f"+rot {np.degrees(d):+.1f} deg {rot['top1']:.3f}/{rot['top3']:.3f}/{rot['top5']:.3f} APL {rot['apl_db']:.2f} | "
                  f"own {own['top1']:.3f}/{own['top3']:.3f}/{own['top5']:.3f} APL {own['apl_db']:.2f}")
    res["fits"] = {k: v.tolist() for k, v in fits.items()}
    (OUT / "e1d_final.json").write_text(json.dumps(res, indent=1, default=float))
