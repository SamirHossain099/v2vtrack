"""Paper figures, one per file, 600 dpi PNG plus vector PDF, from results/*.json only.

Every series carries a redundant non-colour encoding (line style, marker or hatch) so the figures survive a
black-and-white print. Palette: the validated categorical set from the dataviz reference (blue, orange, aqua,
yellow, magenta, violet), assigned in fixed order per entity, never by rank.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7"}
# entity -> (colour, line style, marker); fixed for the whole paper
STYLE = {
    "static": (PALETTE["violet"], ":", "x"),
    "beam-hold": (PALETTE["yellow"], "--", "^"),
    "index-offset-ma": (PALETTE["orange"], "-.", "s"),
    "robust-window": (PALETTE["magenta"], (0, (3, 1, 1, 1)), "D"),
    "kalman": (PALETTE["blue"], "-", "o"),
    "learned": (PALETTE["aqua"], (0, (5, 2)), "v"),
}
LABEL = {"static": "static map", "beam-hold": "beam hold", "index-offset-ma": "index-offset MA", "robust-window": "windowed refit",
         "kalman": "tracked offset (ours)", "learned": "learned GPS net"}
SCEN_NAME = {36: "36 intercity day", 37: "37 intercity night", 38: "38 urban day", 39: "39 urban night"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                     "lines.linewidth": 1.4, "lines.markersize": 4})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_boresight_per_sequence():
    e1 = json.loads((RES / "e1d_final.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, (k, v) in zip(axes, sorted(e1["scenarios"].items())):
        offs = np.array(v["per_seq_front"]["offsets_deg"])
        ax.plot(np.arange(len(offs)), offs, color=PALETTE["blue"], marker="o", markersize=2.2, linewidth=0.8)
        ax.axhline(v["boresight_deg"][0], color="#52514e", linewidth=0.8, linestyle="--")
        ax.set_title(SCEN_NAME[int(k)], fontsize=8)
        ax.set_xlabel("sequence (time order)")
    axes[0].set_ylabel("front-array offset (deg)")
    save(fig, "fig_offset_per_sequence")


def fig_transfer_matrix():
    e1 = json.loads((RES / "e1d_final.json").read_text())
    sc = [36, 37, 38, 39]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3))
    for ax, key, title in zip(axes, ("source", "rotation_refit"), ("source parameters", "plus one refitted rotation")):
        M = np.zeros((4, 4))
        for i, a in enumerate(sc):
            for j, b in enumerate(sc):
                M[i, j] = e1["scenarios"][str(b)]["own_fit_metrics"]["top1"] if a == b else e1["transfer"][f"{a}->{b}"][key]["top1"]
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.35 else "#0b0b0b")
        ax.set_xticks(range(4)); ax.set_xticklabels(sc); ax.set_yticks(range(4)); ax.set_yticklabels(sc)
        ax.set_xlabel("scored on"); ax.set_title(title, fontsize=8); ax.grid(False)
    axes[0].set_ylabel("fitted on")
    fig.colorbar(im, ax=axes, shrink=0.8, label="top-1")
    save(fig, "fig_transfer_matrix")


def fig_budget_curve():
    runs = {}
    for tag in ("src36_h0", "extra_src36_h0"):
        runs.update(json.loads((RES / f"e2_tracker_{tag}.json").read_text())["runs"])
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.8), sharex=True)
    for col, scen in enumerate((37, 38, 39)):
        for m in ("beam-hold", "index-offset-ma", "robust-window", "kalman"):
            for L, filled in ((0, True), (2, False)):
                pts = []
                for S in (10, 20, 50, 100, 200, 500):
                    k = f"{scen}|{m}|S{S}|L{L}"
                    if k in runs:
                        r = runs[k]
                        pts.append((r["meas_per_frame"], r["top1"], r["apl_db"]))
                if not pts:
                    continue
                pts.sort()
                x, y1, y2 = zip(*pts)
                c, ls, mk = STYLE[m]
                kw = dict(color=c, linestyle=ls, marker=mk, markerfacecolor=c if filled else "white", markeredgecolor=c,
                          label=f"{LABEL[m]}{'' if L == 0 else ' + local ±2'}")
                axes[0, col].plot(x, y1, **kw)
                axes[1, col].plot(x, y2, **kw)
        st = runs[f"{scen}|static|S0|L0"]
        for row, key in ((0, "top1"), (1, "apl_db")):
            axes[row, col].axhline(st[key], color=STYLE["static"][0], linestyle=STYLE["static"][1], linewidth=1.0, label=LABEL["static"] if col == 0 else None)
        axes[0, col].set_title(SCEN_NAME[scen], fontsize=8)
        if col == 1:
            axes[1, col].set_xlabel("beam measurements per frame beyond the served beam (log scale)")
        for row in (0, 1):
            axes[row, col].set_xscale("log")
    axes[0, 0].set_ylabel("top-1")
    axes[1, 0].set_ylabel("average power loss (dB)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=7)
    save(fig, "fig_budget_curve")


def fig_recovery():
    e5 = json.loads((RES / "e5_continual_src36.json").read_text())
    wins = ["0-10s", "10-30s", "30-60s", "60-9999s"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
    for ax, visit in zip(axes, (1, 2, 3)):
        for key, m in (("static|S0|L0", "static"), ("index-offset-ma|S20|L0", "index-offset-ma"), ("robust-window|S20|L0", "robust-window"), ("kalman|S20|L0", "kalman")):
            r = e5["runs"][key][visit]["recovery"]
            y = [r.get(w, {}).get("top1", np.nan) for w in wins]
            c, ls, mk = STYLE[m]
            ax.plot(range(4), y, color=c, linestyle=ls, marker=mk, label=LABEL[m])
        ax.set_xticks(range(4)); ax.set_xticklabels(["0-10 s", "10-30 s", "30-60 s", "> 60 s"])
        ax.set_title(f"into {SCEN_NAME[e5['order'][visit]]}", fontsize=8)
        ax.set_xlabel("time since the switch")
    axes[0].set_ylabel("top-1")
    axes[0].legend(fontsize=6.5, loc="upper left")
    save(fig, "fig_recovery")


def fig_lag_histogram():
    e1c = json.loads((RES / "e1c_lag_per_seq.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.8), sharey=True)
    for ax, (k, v) in zip(axes, sorted(e1c.items(), key=lambda kv: int(kv[0]))):
        ks = np.array([r["best_k"] for r in v["rows"] if r["gain"] > 0.3])
        ax.hist(ks, bins=np.arange(-4.5, 21.5, 1), color=PALETTE["blue"], edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#52514e", linewidth=0.8, linestyle="--")
        ax.set_title(SCEN_NAME[int(k)], fontsize=8)
        ax.set_xlabel("best GPS lag (frames)")
    axes[0].set_ylabel("sequences with a lag effect")
    save(fig, "fig_lag_histogram")


def fig_known_truth():
    e4 = json.loads((RES / "e4_known_truth.json").read_text())
    if "trace" not in e4:
        return
    tr = e4["trace"]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    t = np.arange(len(tr["truth_deg"])) / 10
    ax.step(t, tr["truth_deg"], where="post", color="#52514e", linewidth=1.0, label="applied rotation")
    ax.plot(t, tr["est_deg"], color=PALETTE["blue"], linewidth=1.0, label="tracked (sweep every 5 s)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("rotation (deg)")
    ax.legend(fontsize=6.5)
    save(fig, "fig_known_truth")


if __name__ == "__main__":
    for f in (fig_boresight_per_sequence, fig_transfer_matrix, fig_budget_curve, fig_recovery, fig_lag_histogram, fig_known_truth):
        f()
        print("done", f.__name__)
