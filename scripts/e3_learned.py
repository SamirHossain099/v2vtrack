"""E3: learned GPS-sequence baselines on the same streams as E2.

Train BeamNet on the source scenario (36 by default; sequences split 85/15 for early stopping), 3 seeds, with
and without the GPS course as an input. Evaluate as streams on 36-39 (lag-corrected) with source / norm / tent /
supft (supervised fine-tuning on sweep frames at sweep_every S). Same scoring as E2.
Writes results/e3_learned_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2v.data import load  # noqa: E402
from v2v.geo import apply_lag, causal_features  # noqa: E402
from v2v.learned import Learned, sequence_features, train  # noqa: E402
from v2v.stream import run  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=36)
    ap.add_argument("--targets", type=int, nargs="+", default=[36, 37, 38, 39])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sweeps", type=int, nargs="+", default=[0, 20, 100])
    ap.add_argument("--modes", nargs="+", default=["source", "norm", "tent", "supft"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4])
    ap.add_argument("--no-course", action="store_true")
    ap.add_argument("--no-lag", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    e1 = json.loads((OUT / "e1d_final.json").read_text())
    lags = {int(k): v for k, v in e1["lag_frames"].items()}
    tag = a.tag or f"src{a.source}{'_nocourse' if a.no_course else ''}{'_nolag' if a.no_lag else ''}"
    data = {}
    for scen in sorted(set(a.targets) | {a.source}):
        s = load(scen)
        if not a.no_lag:
            s = apply_lag(s, lags[scen])
        f = causal_features(s)
        X, valid = sequence_features(s, f, with_course=not a.no_course)
        f["valid"] = valid
        data[scen] = (s, f, X)
    s0, f0, X0 = data[a.source]
    v0 = np.flatnonzero(f0["valid"])
    res = {"source": a.source, "with_course": not a.no_course, "lag": not a.no_lag, "train": {}, "runs": {}}
    for seed in a.seeds:
        model, info = train(X0[v0], s0["beam"][v0], s0["seq"][v0], epochs=a.epochs, seed=seed)
        res["train"][seed] = info
        print(f"seed {seed}: val top-1 {info['val_top1']:.3f} on {info['n_val']} frames", flush=True)
        for scen in a.targets:
            s, f, X = data[scen]
            # the source scenario is scored on its held-out sequences only; the others are streamed in full
            idx = np.flatnonzero(f["valid"] & (np.isin(s["seq"], info["val_seqs"]) if scen == a.source else True))
            for mode in a.modes:
                for S in (a.sweeps if mode == "supft" else [0]):
                    for lr in (a.lrs if mode in ("tent", "supft") else [0.0]):
                        m = Learned(model, X, mode=mode, lr=lr or 1e-4, steps=a.steps)
                        r = run(m, s, f, sweep_every=S, idx=idx)
                        key = f"{scen}|{m.name}|S{S}|lr{lr:g}|seed{seed}"
                        res["runs"][key] = r
                        print(f"{key:40s} top1/3/5 {r['top1']:.3f}/{r['top3']:.3f}/{r['top5']:.3f} APL {r['apl_db']:6.2f} dB  "
                              f"moving {r.get('moving', {}).get('top1', float('nan')):.3f}  still {r.get('stationary', {}).get('top1', float('nan')):.3f}  "
                              f"{r['latency_ms']:.3f} ms", flush=True)
        (OUT / f"e3_learned_{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
