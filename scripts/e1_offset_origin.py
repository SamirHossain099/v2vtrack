"""E1: where does the per-array offset come from, and why does it not transfer between scenarios?

Physical model per array a:  beam_in = c + K * sin(phi - b_a),  phi = bearing(1->2) - heading(1),
with c and K shared (the codebook: 64 beams over +-45 deg) and b_a the array boresight relative to the
vehicle heading. Hypotheses for the scenario-to-scenario change in b_a seen at the gate:
  H1 remount / box rotation: b_a differs by a constant per recording, the same for all arrays, and is
     flat across sequences within a scenario.
  H2 sensor time offset: GPS and power rows are not simultaneous; shifting the GPS track by k frames
     relative to the power should minimise the residual at a non-zero k.
  H3 GPS-course lag/bias: course over ground lags the true heading during turns, so the residual should
     scale with yaw rate; the fitted "offset" then depends on how much a scenario turns.
  H4 identifiability: side arrays are best in a few percent of frames and their b_a is not estimable.
Writes results/e1_offset_origin.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import ARRAY_NAMES, bearing_1_to_2, course, load, wrap  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
NOMINAL = np.radians([0.0, -90.0, 180.0, 90.0])  # front, right, back, left; ccw-positive frame (east=0)
SCEN = (36, 37, 38, 39)
C_CENTRE = 31.5


def fit_shared(phi, arr, beam_in, v_ok):
    """Shared c, K and per-array boresight b_a by robust least squares on frames with the array best.

    Sign convention fixed by the data: the beam index falls as phi (ccw) rises, so beam = c + K sin(b_a - phi)
    with K > 0. Without that constraint and bounds the optimiser finds the mirrored solution (K < 0, b_a off
    by ~150 deg), which fits equally well and makes the per-sequence offsets meaningless as boresights.
    """
    # c is pinned at the codebook centre (64 beams over +-45 deg -> 31.5): a free c trades off against
    # every b_a and the "boresight" stops meaning the array normal. K stays free (the codebook slope).
    def res(p):
        K, b = p[0], p[1:5]
        return C_CENTRE + K * np.sin(wrap(b[arr] - phi)) - beam_in
    p0 = np.r_[40.0, NOMINAL]
    lo = np.r_[20.0, NOMINAL - np.radians(45)]
    hi = np.r_[60.0, NOMINAL + np.radians(45)]
    r = least_squares(lambda p: res(p)[v_ok], p0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
    return np.r_[C_CENTRE, r.x]


def per_seq_offset(phi, beam_in, seq, ok, c, K, b0):
    """Boresight of one array fitted per sequence with c, K frozen; returns (seq ids, offsets, counts)."""
    ids, offs, ns = [], [], []
    for q in np.unique(seq[ok]):
        m = ok & (seq == q)
        if m.sum() < 40:
            continue
        f = lambda b: c + K * np.sin(wrap(b - phi[m])) - beam_in[m]
        r = least_squares(f, [b0], bounds=([b0 - np.radians(45)], [b0 + np.radians(45)]), loss="soft_l1", f_scale=3.0)
        ids.append(int(q)); offs.append(float(r.x[0])); ns.append(int(m.sum()))
    return np.array(ids), np.array(offs), np.array(ns)


def circ_std_deg(a):
    return float(np.degrees(np.sqrt(-2 * np.log(np.abs(np.exp(1j * a).mean())))))


def analyse(scen, half=5):
    s = load(scen)
    hd, sp = course(s, half=half)
    brg = bearing_1_to_2(s)
    phi = wrap(brg - hd)
    ok_all = np.isfinite(phi) & (sp > 2.0)
    arr, beam_in, seq = s["array"], s["beam_in"], s["seq"]
    p = fit_shared(phi, arr, beam_in, ok_all)
    c, K, b = p[0], p[1], p[2:6]
    pred = c + K * np.sin(wrap(b[arr] - phi))
    resid = beam_in - pred
    out = {"c": float(c), "K": float(K), "boresight_deg": np.degrees(wrap(b)).round(2).tolist(),
           "array_share": [float((arr[ok_all] == a).mean()) for a in range(4)],
           "resid_mad_beams": float(np.median(np.abs(resid[ok_all])))}

    # H1: per-sequence offsets for front and back arrays
    for a in (0, 2):
        m = ok_all & (arr == a)
        ids, offs, ns = per_seq_offset(phi, beam_in, seq, m, c, K, b[a])
        out[f"per_seq_{ARRAY_NAMES[a]}"] = {
            "n_seq": int(len(ids)), "mean_deg": float(np.degrees(np.angle(np.exp(1j * offs).mean()))),
            "circ_std_deg": circ_std_deg(offs),
            "p10_p90_deg": [float(np.degrees(np.percentile(wrap(offs - offs.mean()), q)) + np.degrees(offs.mean())) for q in (10, 90)],
            "first_last_third_mean_deg": [float(np.degrees(np.angle(np.exp(1j * offs[:len(offs)//3]).mean()))),
                                          float(np.degrees(np.angle(np.exp(1j * offs[-(len(offs)//3):]).mean())))],
            "offsets_deg": np.degrees(offs).round(2).tolist(), "seq_ids": ids.tolist(),
        }

    # H2: time offset between GPS and power. Shift phi by k frames inside sequences and refit c,K,b.
    sync = {}
    for k in range(-8, 9):
        phi_k = np.full_like(phi, np.nan)
        for q in np.unique(seq):
            idx = np.flatnonzero(seq == q)
            if k >= 0:
                phi_k[idx[:len(idx) - k]] = phi[idx[k:]] if k else phi[idx]
            else:
                phi_k[idx[-k:]] = phi[idx[:len(idx) + k]]
        okk = np.isfinite(phi_k) & ok_all
        pk = fit_shared(phi_k, arr, beam_in, okk)
        rk = beam_in - (pk[0] + pk[1] * np.sin(wrap(pk[2:6][arr] - phi_k)))
        sync[k] = {"mad": float(np.median(np.abs(rk[okk]))), "rms": float(np.sqrt(np.mean(np.minimum(rk[okk] ** 2, 100)))),
                   "front_boresight_deg": float(np.degrees(wrap(pk[2])))}
    best_k = min(sync, key=lambda k: sync[k]["rms"])
    out["sync"] = {"best_shift_frames": best_k, "curve_rms": {k: v["rms"] for k, v in sync.items()},
                   "front_boresight_at_best_deg": sync[best_k]["front_boresight_deg"],
                   "file_stamp_gps1_minus_pwr_s_median": float(np.median(s["t_gps1"] - s["t_pwr"])),
                   "file_stamp_gps2_minus_pwr_s_median": float(np.median(s["t_gps2"] - s["t_pwr"])),
                   "file_stamp_row_minus_pwr_s_median": float(np.median(s["t"] - s["t_pwr"]))}

    # H3: residual (in degrees of bearing) against yaw rate and speed, front array only
    m = ok_all & (arr == 0)
    yaw = np.full(len(hd), np.nan)
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        d = wrap(np.diff(hd[idx]))
        yaw[idx[1:]] = d / np.diff(s["t"][idx])
    resid_deg = -np.degrees(resid / (K * np.cos(wrap(b[arr] - phi)) + 1e-9))  # local linearisation, degrees of phi
    mm = m & np.isfinite(yaw) & (np.abs(resid_deg) < 15)
    yr = np.degrees(yaw[mm])
    A = np.c_[yr, np.ones(mm.sum())]
    coef, *_ = np.linalg.lstsq(A, resid_deg[mm], rcond=None)
    out["yaw"] = {"slope_s_per_deg_resid_per_degps": float(coef[0]), "intercept_deg": float(coef[1]),
                  "corr": float(np.corrcoef(yr, resid_deg[mm])[0, 1]), "n": int(mm.sum()),
                  "resid_by_absyaw_bin_deg": {}}
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 90)):
        k = mm & (np.abs(np.degrees(yaw)) >= lo) & (np.abs(np.degrees(yaw)) < hi)
        if k.sum() > 50:
            out["yaw"]["resid_by_absyaw_bin_deg"][f"{lo}-{hi}"] = [float(np.median(resid_deg[k])), int(k.sum())]
    out["speed"] = {}
    for lo, hi in ((2, 5), (5, 10), (10, 15), (15, 30)):
        k = m & (sp >= lo) & (sp < hi) & (np.abs(resid_deg) < 15)
        if k.sum() > 50:
            out["speed"][f"{lo}-{hi}"] = [float(np.median(resid_deg[k])), int(k.sum())]

    # H4: bootstrap CI of each array's boresight
    rng = np.random.default_rng(0)
    boots = []
    idx_ok = np.flatnonzero(ok_all)
    for _ in range(30):
        take = rng.choice(idx_ok, len(idx_ok), replace=True)
        mask = np.zeros(len(phi), bool); mask[np.unique(take)] = True
        boots.append(wrap(fit_shared(phi, arr, beam_in, mask)[2:6]))
    boots = np.array(boots)
    out["boresight_boot_std_deg"] = [circ_std_deg(boots[:, a]) for a in range(4)]

    # heading window sensitivity
    out["half_window"] = {}
    for h in (1, 2, 5, 10):
        hd_h, sp_h = course(s, half=h)
        phi_h = wrap(brg - hd_h)
        okh = np.isfinite(phi_h) & (sp_h > 2.0)
        ph = fit_shared(phi_h, arr, beam_in, okh)
        rh = beam_in - (ph[0] + ph[1] * np.sin(wrap(ph[2:6][arr] - phi_h)))
        out["half_window"][h] = {"rms": float(np.sqrt(np.mean(np.minimum(rh[okh] ** 2, 100)))),
                                 "front_boresight_deg": float(np.degrees(wrap(ph[2])))}
    # codebook form: sin (DFT-like) against linear in angle, front array, same boresight and K free
    m = ok_all & (arr == 0)
    x = wrap(b[0] - phi[m])
    f_lin = lambda p: C_CENTRE + p[0] * x + p[1] - beam_in[m]
    f_sin = lambda p: C_CENTRE + p[0] * np.sin(x) + p[1] - beam_in[m]
    rl = least_squares(f_lin, [40.0, 0.0], loss="soft_l1", f_scale=3.0)
    rs = least_squares(f_sin, [40.0, 0.0], loss="soft_l1", f_scale=3.0)
    out["codebook_form"] = {"linear_rms": float(np.sqrt(np.mean(np.minimum(rl.fun ** 2, 100)))), "linear_K_per_rad": float(rl.x[0]),
                            "sin_rms": float(np.sqrt(np.mean(np.minimum(rs.fun ** 2, 100)))), "sin_K": float(rs.x[0])}
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    res = {}
    for scen in SCEN:
        r = analyse(scen)
        res[scen] = r
        print(f"\n== scenario {scen}: c {r['c']:.2f} K {r['K']:.2f} boresight {r['boresight_deg']} "
              f"boot std {np.round(r['boresight_boot_std_deg'], 2).tolist()} share {np.round(r['array_share'], 3).tolist()}")
        for a in ("front", "back"):
            q = r[f"per_seq_{a}"]
            print(f"   per-seq {a}: n {q['n_seq']} mean {q['mean_deg']:+.2f} std {q['circ_std_deg']:.2f} "
                  f"p10-p90 {np.round(q['p10_p90_deg'], 1).tolist()} first/last third {np.round(q['first_last_third_mean_deg'], 2).tolist()}")
        sy = r["sync"]
        print(f"   sync: best shift {sy['best_shift_frames']:+d} frames; rms by shift "
              + " ".join(f"{k}:{v:.2f}" for k, v in sy["curve_rms"].items() if abs(k) <= 4)
              + f" | file stamps gps1-pwr {sy['file_stamp_gps1_minus_pwr_s_median']:+.3f}s gps2-pwr {sy['file_stamp_gps2_minus_pwr_s_median']:+.3f}s row-pwr {sy['file_stamp_row_minus_pwr_s_median']:+.3f}s")
        print(f"   yaw: slope {r['yaw']['slope_s_per_deg_resid_per_degps']:+.3f} s, intercept {r['yaw']['intercept_deg']:+.2f} deg, corr {r['yaw']['corr']:+.2f}; "
              f"median resid by |yaw| bin {r['yaw']['resid_by_absyaw_bin_deg']}")
        print(f"   speed bins (median resid deg, n): {r['speed']}")
        print(f"   half-window: " + " ".join(f"{h}:rms {v['rms']:.2f}/b {v['front_boresight_deg']:+.1f}" for h, v in r["half_window"].items()))
    (OUT / "e1_offset_origin.json").write_text(json.dumps(res, indent=1, default=float))
