"""E1e: verify the GPS-to-power lag with an independent sensor.

The radar on the receiver is recorded by the same unit as the beam power, so a GPS-to-radar lag measured
independently of the beams tests E1d's finding (11 frames in 37, ~0 elsewhere). Peak picking on the range
profile fails (the strongest return is usually clutter, not the transmitter car), so this uses trajectory
matching instead: for a lag k and a range-to-bin gain g, score(k, g) = mean over frames of the per-frame
z-scored log range profile at bin round(g * gps_range[t + k]), on the radar facing the transmitter (the radar
with the same index as the best power array). The transmitter is a real reflector at the GPS range, so the
score peaks at the true (k, g); clutter does not follow the GPS range and averages out.

Radar frames are raw FMCW cubes (4 rx, 256 samples, 128 chirps): range profile = |FFT over samples|^2, mean
over chirps and rx, one-sided, cached to results/radar_profiles_<scen>.npz (every `step`-th frame).
Usage: python e1e_radar_lag.py 36 37 [--step 2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import ROOT, load  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
LAGS = list(range(-15, 21))
GAINS = np.arange(1.0, 8.01, 0.05)  # bins per metre


def range_profile(path):
    d = scipy.io.loadmat(path)["data"]
    d = d - d.mean(axis=1, keepdims=True)
    win = np.hanning(d.shape[1])[None, :, None]
    p = np.abs(np.fft.fft(d * win, axis=1)) ** 2
    return p.mean(axis=(0, 2))[: d.shape[1] // 2].astype(np.float32)


def profiles(scen, s, step):
    npz = OUT / f"radar_profiles_{scen}.npz"
    if npz.exists():
        z = np.load(npz)
        if int(z["step"]) == step:
            return z["frames"], z["prof"]
    df = pd.read_csv(ROOT / f"scenario{scen}" / f"scenario{scen}.csv", low_memory=False,
                     usecols=[f"unit1_radar{a}" for a in (1, 2, 3, 4)])
    base = ROOT / f"scenario{scen}" / f"scenario{scen}"
    frames = np.arange(0, len(df), step)
    prof = np.zeros((len(frames), 128), np.float32)
    for j, i in enumerate(frames):
        a = s["array"][i]
        prof[j] = range_profile(base / df[f"unit1_radar{a + 1}"].iloc[i])
        if j % 2000 == 0:
            print(f"   {j}/{len(frames)} frames", flush=True)
    np.savez(npz, frames=frames, prof=prof, step=step)
    return frames, prof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", type=int, nargs="+")
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    res = {}
    for scen in a.scenarios:
        s = load(scen)
        frames, prof = profiles(scen, s, a.step)
        z = np.log10(prof + 1e-12)
        z[:, :4] = np.nan
        z = (z - np.nanmean(z, 1, keepdims=True)) / (np.nanstd(z, 1, keepdims=True) + 1e-9)
        z = np.nan_to_num(z, nan=0.0)
        gps_r = np.hypot(s["e2"] - s["e1"], s["n2"] - s["n1"])
        seq = s["seq"]
        n = len(seq)
        score = np.full((len(LAGS), len(GAINS)), np.nan)
        for ki, k in enumerate(LAGS):
            tgt = frames + k
            ok = (tgt >= 0) & (tgt < n)
            ok[ok] &= seq[np.clip(tgt, 0, n - 1)][ok] == seq[frames][ok]
            r = gps_r[np.clip(tgt, 0, n - 1)]
            ok &= (r > 5) & (r < 120)
            for gi, g in enumerate(GAINS):
                b = np.rint(g * r).astype(int)
                m = ok & (b >= 4) & (b < 128)
                if m.sum() < 500:
                    continue
                score[ki, gi] = z[np.flatnonzero(m), b[m]].mean()
        ki, gi = np.unravel_index(np.nanargmax(score), score.shape)
        best_k, best_g = LAGS[ki], float(GAINS[gi])
        by_lag = {k: float(np.nanmax(score[i])) for i, k in enumerate(LAGS)}
        # sharpness: how far the best lag beats the neighbours and lag 0
        res[scen] = {"best_lag_frames": best_k, "gain_bins_per_m": best_g, "range_res_m_per_bin": 1 / best_g,
                     "score_by_lag_at_best_gain": {k: float(score[i, gi]) for i, k in enumerate(LAGS)},
                     "best_score_by_lag": by_lag, "n_radar_frames": int(len(frames)), "step": a.step}
        print(f"scenario {scen}: best GPS-vs-radar lag {best_k:+d} frames at {1/best_g:.3f} m/bin; score by lag at that gain: "
              + " ".join(f"{k}:{score[i, gi]:.3f}" for i, k in enumerate(LAGS) if k % 2 == 0 or abs(k - best_k) <= 2), flush=True)
    (OUT / "e1e_radar_lag.json").write_text(json.dumps(res, indent=1, default=float))
