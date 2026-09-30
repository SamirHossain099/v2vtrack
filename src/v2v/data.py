"""Load a DeepSense V2V scenario index into arrays.

Everything downstream works from this dict:
  t        seconds since the first frame (from the index timestamp, one clock for the row)
  seq      sequence id (the dataset's seq_index; contiguous 10 Hz recordings of ~200 frames)
  lat1/lon1, lat2/lon2   receiver (unit1) and transmitter (unit2) GPS in degrees
  e1/n1, e2/n2           local east/north metres about the scenario centroid
  pwr      (N, 4, 64) received power, arrays in file order pwr1-4 = front, right, back, left
  beam     (N,) overall best beam 0-255 = 64 * array + beam (verified equal to the CSV column)
  array    (N,) argmax array, beam_in  (N,) argmax beam within it
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"N:\Datasets\DeepSense 6G\extracted")
R_EARTH = 6371000.0
N_ARR, N_BEAMS = 4, 64
ARRAY_NAMES = ("front", "right", "back", "left")


def _read_pwr(paths, base):
    out = np.empty((len(paths), N_BEAMS), dtype=np.float32)
    for i, p in enumerate(paths):
        out[i] = np.loadtxt(base / p, dtype=np.float32)
    return out


def load(scen: int, cache: bool = True) -> dict:
    d = ROOT / f"scenario{scen}"
    npz = d / f"scenario{scen}_index.npz"
    if cache and npz.exists():
        return dict(np.load(npz))
    df = pd.read_csv(d / f"scenario{scen}.csv", low_memory=False)
    t = pd.to_datetime(df["timestamp"], format="%H-%M-%S.%f")
    out = {
        "abs_index": df["abs_index"].to_numpy(),
        "t": (t - t.iloc[0]).dt.total_seconds().to_numpy(),
        "seq": df["seq_index"].to_numpy(),
    }
    for u in (1, 2):
        out[f"lat{u}"] = df[f"unit{u}_gps1_lat"].to_numpy()
        out[f"lon{u}"] = df[f"unit{u}_gps1_lon"].to_numpy()
    lat0, lon0 = np.radians(out["lat1"].mean()), np.radians(out["lon1"].mean())
    for u in (1, 2):
        out[f"e{u}"] = R_EARTH * (np.radians(out[f"lon{u}"]) - lon0) * np.cos(lat0)
        out[f"n{u}"] = R_EARTH * (np.radians(out[f"lat{u}"]) - lat0)
    # the dataset's pickle carries the 64-beam power vectors pre-loaded (same rows as the CSV)
    pk = pickle.load(open(d / f"scenario{scen}.p", "rb"))
    assert np.array_equal(pk["abs_index"], out["abs_index"]), "pickle rows differ from the CSV"
    pwr = np.stack([np.asarray(pk[f"unit1_pwr{a}"], dtype=np.float32) for a in range(1, N_ARR + 1)], 1)
    if scen == 36:  # spot-check the pickle against the raw power files once
        base = d / f"scenario{scen}"
        for i in (0, len(df) // 2, len(df) - 1):
            raw = np.loadtxt(base / df["unit1_pwr1"].iloc[i], dtype=np.float32)
            assert np.allclose(raw, pwr[i, 0], rtol=1e-5), "pickle power disagrees with the power file"
    out["pwr"] = pwr
    flat = pwr.reshape(len(df), -1)
    out["beam"] = flat.argmax(1)
    out["array"] = out["beam"] // N_BEAMS
    out["beam_in"] = out["beam"] % N_BEAMS
    csv_beam = df["unit1_overall-beam"].to_numpy(dtype=float)
    assert np.all(csv_beam == out["beam"]), "overall-beam column disagrees with the power files"
    # per-modality file timestamps, for the synchronisation check in E1
    for col, key in (("unit1_gps1", "t_gps1"), ("unit2_gps1", "t_gps2"), ("unit1_pwr1", "t_pwr")):
        stamp = df[col].str.extract(r"(\d{2}-\d{2}-\d{2}\.\d+)")[0]
        ts = pd.to_datetime(stamp, format="%H-%M-%S.%f")
        out[key] = (ts - t.iloc[0]).dt.total_seconds().to_numpy()
    if cache:
        np.savez(npz, **out)
    return out


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def bearing_1_to_2(s):
    """Compass-free bearing from unit1 to unit2 in the east-north frame: 0 = east, ccw positive."""
    return np.arctan2(s["n2"] - s["n1"], s["e2"] - s["e1"])


def course(s, half: int = 5, unit: int = 1):
    """Heading of a unit from its displacement over +-half frames inside one sequence, and speed (m/s)."""
    e, n, t, seq = s[f"e{unit}"], s[f"n{unit}"], s["t"], s["seq"]
    hd, sp = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        if len(idx) < 2 * half + 1:
            continue
        a, b, mid = idx[:-2 * half], idx[2 * half:], idx[half:-half]
        de, dn, dt = e[b] - e[a], n[b] - n[a], t[b] - t[a]
        hd[mid] = np.arctan2(dn, de)
        sp[mid] = np.hypot(de, dn) / np.where(dt > 0, dt, np.nan)
    return hd, sp


def topk_and_apl(pred_beam, s, idx=None):
    """Challenge metrics: top-k hit of the predicted beam among the k strongest true beams, and APL in dB."""
    idx = np.arange(len(pred_beam)) if idx is None else idx
    flat = s["pwr"].reshape(len(s["pwr"]), -1)[idx]
    order = np.argsort(-flat, axis=1)
    hits = order[:, :5] == pred_beam[:, None]
    top = {f"top{k}": float(hits[:, :k].any(1).mean()) for k in (1, 3, 5)}
    p_est = flat[np.arange(len(idx)), pred_beam]
    p_best = flat[np.arange(len(idx)), order[:, 0]]
    top["apl_db"] = float(np.mean(10 * np.log10(p_est / p_best)))
    top["n"] = int(len(idx))
    return top
