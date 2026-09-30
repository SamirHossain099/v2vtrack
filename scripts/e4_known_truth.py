"""E4: known-truth arm on DeepMIMO v4. Real data cannot say whether the tracked rotation is the array's rotation;
a ray-traced scene with a rotation applied by construction can.

Setup: one base station of a city scenario (28 GHz) with a 16-element horizontal ULA and a 64-beam codebook over
+-45 deg, mirroring one DeepSense V2V array. A "vehicle" drives along the user grid (rows of the grid, LoS users
only), giving a stream of positions; the BS array is rotated in azimuth by a piecewise-constant schedule of known
angles (0, +3, -2, +5 deg, ...), switching every T frames. At each frame the true best beam is the argmax of the
codebook gains on the ray-traced channel for the current rotation. The tracker sees the true bearing (or bearing
from positions with N(0, sigma) noise) and sparse sweeps, exactly as on the real data, and its delta estimate is
compared with the applied rotation.
Writes results/e4_known_truth.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.geo import C_CENTRE, wrap  # noqa: E402
from v2v.track import KalmanDelta  # noqa: E402

import deepmimo as dm  # noqa: E402

dm.config("scenarios_folder", r"N:\Datasets\DeepMIMO\scenarios")
OUT = Path(__file__).resolve().parents[1] / "results"
M = 16
N_BEAMS = 64
SCHEDULE_DEG = [0.0, 3.0, -2.0, 5.0, -4.0, 1.5]
T_SEG = 400  # frames per rotation segment
SWEEPS = [10, 20, 50, 100, 200]
SIGMAS = [0.0, 0.5, 1.0]  # metres of position noise on both ends
SEEDS = [0, 1, 2]


def codebook():
    ang = -45 + 90 * (np.arange(N_BEAMS) + 0.5) / N_BEAMS
    return np.stack([dm.steering_vec([M, 1], phi=a).squeeze() for a in ang], 0), ang


def best_beams(ds, rot_deg, cb):
    p = dm.ChannelParameters()
    p.bs_antenna.shape = np.array([M, 1])
    p.bs_antenna.rotation = np.array([0, 0, rot_deg])
    p.ue_antenna.shape = np.array([1, 1])
    p.num_paths = 10
    h = ds.compute_channels(p)[:, 0, :, 0]  # (n, M)
    g = np.abs(cb.conj() @ h.T) ** 2  # (B, n)
    return g.argmax(0), g.T


def main():
    scen = "city_3_houston_28"
    ds = dm.load(scen)
    ds = ds[0] if isinstance(ds, dm.MacroDataset) else ds
    los = np.asarray(ds.los) == 1
    tx = np.asarray(ds.tx_pos).reshape(-1, 3)[0]
    pos = ds.rx_pos
    bearing_all = np.arctan2(pos[:, 1] - tx[1], pos[:, 0] - tx[0])
    # the base station's nominal look direction: the median LoS bearing, so the codebook covers the users
    look = np.angle(np.exp(1j * bearing_all[los]).mean())
    base_rot = float(np.degrees(look))  # rotation slot 2 that points the array at the users
    # stream: LoS users inside the array's +-40 deg field of view, sorted along the grid (by y then x) so that
    # consecutive frames are neighbours; take a contiguous run
    infov = los & (np.abs(wrap(bearing_all - look)) < np.radians(40))
    order = np.lexsort((pos[infov, 0], pos[infov, 1]))
    users = np.flatnonzero(infov)[order]
    print(f"{los.sum()} LoS users, {infov.sum()} inside the field of view")
    n = len(SCHEDULE_DEG) * T_SEG
    users = users[:n] if len(users) >= n else np.resize(users, n)
    sub = ds.trim(idxs=users)
    cb, ang = codebook()
    # ground truth per segment
    # the tracker's phi is -phi0 (its frame has the beam index falling with ccw angle, as on the vehicle), so a ccw
    # array rotation of +d appears in its state as -d; the truth is compared in the tracker's frame
    truth_delta = -np.repeat(np.radians(SCHEDULE_DEG), T_SEG)
    beams = np.zeros(n, int)
    gains = np.zeros((n, N_BEAMS))
    for k, rd in enumerate(SCHEDULE_DEG):
        sl = slice(k * T_SEG, (k + 1) * T_SEG)
        b, g = best_beams(sub.trim(idxs=np.arange(n)[sl]) if False else sub, base_rot + rd, cb)
        beams[sl], gains[sl] = b[sl], g[sl]
    # map constant K from the codebook: beam = C + K sin(look - phi) -> fitted once on the truth at rotation 0
    phi0 = wrap(bearing_all[users] - np.radians(base_rot))  # angle off the unrotated boresight, ccw positive
    b0, _ = best_beams(sub, base_rot, cb)
    # tracker convention beam = C + K sin(b - phi_track); with phi_track = -phi0 and b = 0 this is K sin(phi0)
    K = float(np.polyfit(np.sin(phi0), b0 - C_CENTRE, 1)[0])
    resid = b0 - (C_CENTRE + K * np.sin(phi0))
    print(f"codebook fit: K {K:.2f}, residual rms {np.sqrt(np.mean(resid**2)):.2f} beams, beam range {b0.min()}-{b0.max()}")
    res = {"scenario": scen, "M": M, "K_fit": K, "base_rot_deg": base_rot, "schedule_deg": SCHEDULE_DEG, "T_seg": T_SEG,
           "sign_check_K_positive": bool(K > 0), "runs": {}}
    print(f"K from the codebook truth: {K:.2f} beams/rad (base rotation {base_rot:.1f} deg)")
    rel = np.array([0.0])  # one array
    for sigma in SIGMAS:
        for S in SWEEPS:
            errs, top1s = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                p1 = pos[users, :2] + rng.normal(0, sigma, (n, 2))
                p0 = tx[:2] + rng.normal(0, sigma, (n, 2))
                phi = wrap(np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0]) - np.radians(base_rot))
                f = {"phi": -phi, "moving": np.ones(n, bool), "range": np.hypot(*(p1 - p0).T)}
                # the tracker's convention: beam = C + K sin(b - phi_track) with b = delta; here phi_track = -phi
                trk = KalmanDelta(K, delta0=0.0, q_deg_per_sqrt_s=0.5, sigma_pos_m=max(sigma, 0.1), hold_still=False)
                est = np.zeros(n)
                hit = np.zeros(n, bool)
                for i in range(n):
                    cand = trk.step(i, f)
                    hit[i] = (cand[0] // N_BEAMS == 0) and (cand[0] % N_BEAMS == beams[i])
                    est[i] = trk.delta
                    fb = {"t": i * 0.1}
                    if i % S == 0:
                        fb["sweep_beam"] = int(beams[i])  # array 0 of the tracker's four = this array
                    trk.feedback(i, f, fb)
                # error after the first quarter of each segment (settling excluded)
                m = (np.arange(n) % T_SEG) >= T_SEG // 4
                errs.append(np.degrees(np.abs(wrap(est - truth_delta)))[m])
                top1s.append(hit.mean())
            e = np.concatenate(errs)
            if sigma == 0.5 and S == 50 and seed == SEEDS[-1]:
                res["trace"] = {"truth_deg": np.degrees(truth_delta).round(3).tolist(), "est_deg": np.degrees(est).round(3).tolist(),
                                "sigma": sigma, "S": S, "seed": seed}
            res["runs"][f"sigma{sigma}|S{S}"] = {"median_abs_err_deg": float(np.median(e)), "p90_abs_err_deg": float(np.percentile(e, 90)),
                                                  "top1": float(np.mean(top1s)), "beamwidth_deg": 90 / N_BEAMS}
            print(f"sigma {sigma:3.1f} m  sweep every {S:3d}: |delta err| median {np.median(e):.2f} deg, p90 {np.percentile(e, 90):.2f} deg "
                  f"(beam width {90 / N_BEAMS:.2f} deg), top-1 {np.mean(top1s):.3f}", flush=True)
    (OUT / "e4_known_truth.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
