"""Streaming evaluator: frames in time order; predict, score, then feed back what the receiver would observe.

Feedback policy per frame:
  sweep_every S   a full 64x4 sweep every S frames (S = 0: never) -> fb["sweep_beam"] = true best beam
  local k         the receiver also measures the k beams either side of the served one on the served array
                  (a local sweep, cheap) -> fb["local_best"] = best of those
Scoring follows the challenge: top-k = served beam among the k strongest true beams; APL on the served beam.
Stationary frames are scored too (the tracker's heading hold is what makes them predictable).
"""
import time

import numpy as np

from .geo import N_BEAMS


def run(method, s, f, sweep_every=0, local_k=0, idx=None):
    idx = np.flatnonzero(f["valid"]) if idx is None else idx
    flat = s["pwr"].reshape(len(s["pwr"]), -1)
    hits = np.zeros((len(idx), 3))
    apl = np.zeros(len(idx))
    served_all = np.zeros(len(idx), int)
    n_sweep, lat = 0, []
    for j, i in enumerate(idx):
        tgt = f["target"][i]
        t0 = time.perf_counter()
        cand = np.asarray(method.step(i, f)).ravel()
        served = int(cand[0])
        pw = flat[tgt]
        order = np.argsort(-pw)[:5]
        hits[j] = [served == order[0], served in order[:3], served in order[:5]]
        apl[j] = 10 * np.log10(pw[served] / pw[order[0]])
        served_all[j] = served
        fb = {"t": s["t"][tgt], "p_served": pw[served]}
        if sweep_every and (j % sweep_every == 0):
            fb["sweep_beam"] = int(order[0])
            n_sweep += 1
        if local_k:
            arr, bi = served // N_BEAMS, served % N_BEAMS
            lo, hi = max(0, bi - local_k), min(N_BEAMS - 1, bi + local_k)
            loc = np.arange(lo, hi + 1) + arr * N_BEAMS
            fb["local_best"] = int(loc[pw[loc].argmax()])
            # a maximum on the window edge says only "further that way"; interior maxima are true local optima
            fb["local_interior"] = bool(lo < fb["local_best"] % N_BEAMS < hi)
        method.feedback(i, f, fb)
        lat.append(time.perf_counter() - t0)
    mov = f["moving"][idx]
    out = {"n": int(len(idx)), "top1": float(hits[:, 0].mean()), "top3": float(hits[:, 1].mean()), "top5": float(hits[:, 2].mean()),
           "apl_db": float(apl.mean()), "sweeps": n_sweep, "sweep_frac": n_sweep / max(len(idx), 1),
           # beam measurements per frame beyond the served beam: a full sweep is 4 x 64, a local sweep 2k
           "meas_per_frame": float(4 * N_BEAMS * n_sweep / max(len(idx), 1) + 2 * local_k),
           "latency_ms": float(1000 * np.median(lat)), "frac_moving": float(mov.mean())}
    for tag, m in (("moving", mov), ("stationary", ~mov)):
        if m.any():
            out[tag] = {"n": int(m.sum()), "top1": float(hits[m, 0].mean()), "top3": float(hits[m, 1].mean()),
                        "top5": float(hits[m, 2].mean()), "apl_db": float(apl[m].mean())}
    out["delta_final_deg"] = float(np.degrees(getattr(method, "delta", 0.0)))
    return out
