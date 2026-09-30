"""Causal geometry for the streaming tracker.

Frame conventions: east-north metres, angles ccw from east. phi = bearing(unit1 -> unit2) - heading(unit1).
Codebook map per array a: beam_in = C + K * sin(b_a - phi), b_a = boresight of array a relative to the heading,
nominal (front, right, back, left) = (0, -90, 180, +90) deg plus one box rotation delta.
"""
import numpy as np

C_CENTRE = 31.5
NOMINAL = np.radians([0.0, -90.0, 180.0, 90.0])
N_BEAMS = 64


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def apply_lag(s: dict, k: int) -> dict:
    """GPS taken k frames later than the power row, inside each sequence; frames without a partner are dropped."""
    if k == 0:
        return s
    n = len(s["seq"])
    keep = np.zeros(n, bool)
    src = np.zeros(n, int)
    for q in np.unique(s["seq"]):
        idx = np.flatnonzero(s["seq"] == q)
        if k > 0:
            keep[idx[:len(idx) - k]] = True
            src[idx[:len(idx) - k]] = idx[k:]
        else:
            keep[idx[-k:]] = True
            src[idx[-k:]] = idx[:len(idx) + k]
    out = {}
    for key, v in s.items():
        if not (isinstance(v, np.ndarray) and v.shape[:1] == (n,)):
            out[key] = v
        elif key in ("lat1", "lon1", "lat2", "lon2", "e1", "n1", "e2", "n2", "t_gps1", "t_gps2"):
            out[key] = v[src[keep]]
        else:
            out[key] = v[keep]
    return out


def causal_features(s: dict, back: int = 5, v_min: float = 2.0, horizon: int = 0) -> dict:
    """Per frame, from the past only: heading (course over the last `back` frames, held at standstill), speed, bearing,
    range, and phi. With horizon > 0, both positions are linearly extrapolated `horizon` frames ahead from the last
    `back` frames (the challenge baseline's recipe) so the prediction targets frame t + horizon.

    Returns arrays aligned with the frames whose target exists (t + horizon inside the sequence)."""
    n = len(s["seq"])
    e1, n1, e2, n2, t, seq = s["e1"], s["n1"], s["e2"], s["n2"], s["t"], s["seq"]
    heading = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    ee1, nn1, ee2, nn2 = e1.copy(), n1.copy(), e2.copy(), n2.copy()
    valid = np.zeros(n, bool)
    target = np.arange(n)
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        last = np.nan
        for j, i in enumerate(idx):
            if j >= back:
                a = idx[j - back]
                de, dn, dt = e1[i] - e1[a], n1[i] - n1[a], t[i] - t[a]
                sp = np.hypot(de, dn) / dt if dt > 0 else 0.0
                speed[i] = sp
                if sp > v_min:
                    last = np.arctan2(dn, de)
                if horizon:
                    f = horizon * (t[i] - t[idx[j - 1]]) / dt if dt > 0 else 0.0
                    ee1[i], nn1[i] = e1[i] + f * de, n1[i] + f * dn
                    ee2[i], nn2[i] = e2[i] + f * (e2[i] - e2[a]), n2[i] + f * (n2[i] - n2[a])
            heading[i] = last
            if j >= back and j + horizon < len(idx):
                valid[i] = True
                target[i] = idx[j + horizon]
    bearing = np.arctan2(nn2 - nn1, ee2 - ee1)
    rng = np.hypot(nn2 - nn1, ee2 - ee1)
    phi = wrap(bearing - heading)
    valid &= np.isfinite(phi)
    return {"phi": phi, "heading": heading, "speed": speed, "range": rng, "valid": valid, "target": target,
            "moving": speed > v_min}


def predict_beam(phi, delta, K, rel=NOMINAL):
    """Overall beam 0-255 from phi and the box rotation delta (array by nearest boresight)."""
    b = rel + delta
    arr = np.abs(wrap(b[None, :] - np.atleast_1d(phi)[:, None])).argmin(1)
    beam_in = np.clip(np.rint(C_CENTRE + K * np.sin(wrap(b[arr] - phi))), 0, N_BEAMS - 1).astype(int)
    return arr * N_BEAMS + beam_in


def rank_beams(phi, delta, K, rel=NOMINAL, k=5):
    """Top-k candidate overall beams from the map: the array's neighbours around the predicted beam."""
    p = predict_beam(phi, delta, K, rel)
    arr, bi = p // N_BEAMS, p % N_BEAMS
    offs = np.array([0, 1, -1, 2, -2, 3, -3])[:k]
    cand = np.clip(bi[:, None] + offs[None, :], 0, N_BEAMS - 1) + arr[:, None] * N_BEAMS
    return cand


def delta_from_beam(phi, beam, K, rel=NOMINAL):
    """Invert the map at a sweep: the box rotation implied by the true best beam. NaN where the sine is out of range."""
    arr, bi = beam // N_BEAMS, beam % N_BEAMS
    x = (bi + 0.0 - C_CENTRE) / K
    x = np.clip(x, -1.0, 1.0)
    # sin(b - phi) = x has two solutions; take the one nearer the array's nominal look direction
    a1 = np.arcsin(x)
    a2 = np.pi - a1
    cand = np.stack([a1, a2], -1) + np.atleast_1d(phi)[..., None] - rel[arr][..., None]
    cand = wrap(cand)
    pick = np.abs(cand).argmin(-1)
    out = np.take_along_axis(cand, pick[..., None], -1)[..., 0]
    return float(out.reshape(-1)[0]) if np.ndim(phi) == 0 else out
