"""Learned baselines for E3: a GPS-sequence network (the 2022/2023 challenge family) and its online variants.

Input per frame t (causal): the last `back`+1 positions of both units, expressed in the receiver's local frame
(east-north metres relative to unit1 at t), optionally with the GPS course (cos, sin) and speed. Output: 256-way
beam logits. Trained on one scenario; evaluated as a stream on the others with 01's protocol (predict, then
feedback), with three online methods: source (frozen), norm (BN statistics from the stream), tent (entropy
minimisation on BN affine parameters) and supervised fine-tuning on sweep frames.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geo import N_BEAMS


def sequence_features(s, f, back=5, with_course=True):
    """(N, D) float32 features aligned with frames; rows where the history crosses a sequence start are marked invalid."""
    n = len(s["seq"])
    e1, n1, e2, n2, seq = s["e1"], s["n1"], s["e2"], s["n2"], s["seq"]
    cols, valid = [], np.ones(n, bool)
    for o in range(back, -1, -1):
        j = np.clip(np.arange(n) - o, 0, n - 1)
        valid &= seq[j] == seq
        cols += [e1[j] - e1, n1[j] - n1, e2[j] - e1, n2[j] - n1]
    X = np.stack(cols, 1)
    if with_course:
        hd = np.nan_to_num(f["heading"], nan=0.0)
        X = np.concatenate([X, np.cos(hd)[:, None], np.sin(hd)[:, None], np.nan_to_num(f["speed"], nan=0.0)[:, None] / 30.0], 1)
    valid &= f["valid"]
    return X.astype(np.float32), valid


class BeamNet(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, 4 * N_BEAMS))
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def train(X, y, seq, epochs=30, lr=1e-3, seed=0, device="cuda", val_frac=0.15):
    torch.manual_seed(seed); np.random.seed(seed)
    seqs = np.unique(seq)
    rng = np.random.default_rng(seed)
    val_seqs = rng.choice(seqs, max(1, int(val_frac * len(seqs))), replace=False)
    va = np.isin(seq, val_seqs)
    tr = ~va
    model = BeamNet(X.shape[1]).to(device)
    model.mu.copy_(torch.tensor(X[tr].mean(0))); model.sd.copy_(torch.tensor(X[tr].std(0) + 1e-6))
    Xt, yt = torch.tensor(X[tr], device=device), torch.tensor(y[tr], device=device)
    Xv, yv = torch.tensor(X[va], device=device), torch.tensor(y[va], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(model(Xt[b]), yt[b], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if acc > best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"val_top1": best, "n_train": int(tr.sum()), "n_val": int(va.sum()), "val_seqs": [int(q) for q in val_seqs]}


class Learned:
    """Online wrapper with the same step/feedback protocol as track.py. mode in {source, norm, tent, supft}."""
    name = "learned"

    def __init__(self, model, X, mode="source", momentum=0.05, lr=1e-4, device="cuda", steps=1, **kw):
        self.model = copy.deepcopy(model).to(device).eval()
        self.X = torch.tensor(X, device=device)
        self.mode, self.device, self.steps = mode, device, steps
        self.buf_x, self.sup_x, self.sup_y = [], [], []
        if mode in ("norm", "tent"):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.train(); m.momentum = momentum; m.track_running_stats = True
        if mode == "tent":
            params = [p for m in self.model.modules() if isinstance(m, nn.BatchNorm1d) for p in m.parameters()]
            self.opt = torch.optim.Adam(params, lr=lr)
        if mode == "supft":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.name = f"learned-{mode}"
        self._last_logits = None

    def step(self, i, f):
        x = self.X[i:i + 1]
        if self.mode in ("norm", "tent"):
            # 01's protocol: the adaptation batch is the last 16 consecutive frames (what a receiver has), and the
            # current frame is the last row of it; the batch statistics are those of the stream, not of one frame
            self.buf_x.append(self.X[i]); self.buf_x = self.buf_x[-16:]
            x = torch.stack(self.buf_x)
        if self.mode in ("norm", "tent") and len(self.buf_x) < 2:
            # batch statistics need at least two rows; the first frame of a stream is served by the source model
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.train()
            self._last_logits = logits[-1]
            return logits[-1].topk(5).indices.cpu().numpy()
        if self.mode == "tent":
            logits = self.model(x)
            p = logits.softmax(1)
            loss = -(p * logits.log_softmax(1)).sum(1).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            logits = logits.detach()
        else:
            with torch.no_grad():
                logits = self.model(x)
        self._last_logits = logits[-1]
        return logits[-1].topk(5).indices.cpu().numpy()

    def feedback(self, i, f, fb):
        if self.mode != "supft" or fb.get("sweep_beam") is None:
            return
        self.sup_x.append(self.X[i]); self.sup_y.append(int(fb["sweep_beam"]))
        self.sup_x, self.sup_y = self.sup_x[-64:], self.sup_y[-64:]
        if len(self.sup_y) < 8:
            return
        self.model.train()
        x = torch.stack(self.sup_x); y = torch.tensor(self.sup_y, device=self.device)
        for _ in range(self.steps):
            loss = F.cross_entropy(self.model(x), y)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.eval()
