"""Fig. 3: decisions from a few examples at the instrument.
(a) MoS2 flakes lost in 30 selenization runs (20 seeds) with an in situ Raman stop rule, for the idealised
    probe (open circles, left column) and the probe redrawn from published in situ observations (filled
    squares, right column). Outcomes are whole flakes, so seeds with the same value are drawn side by side.
(b) Mean absolute error in sulfur-vacancy density when stopping Ar+ irradiation (20 seeds).
Black bars: mean with 95 % bootstrap interval over seeds (10,000 resamples).
Reads results/reflex/s4_insitu, s4_lit, s11_irradiation."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK

apply_style()


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "reflex", pat)))]


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def mean_ci(ax, x, v, half=0.09):
    lo, hi = boot_ci(v)
    ax.plot([x - half, x + half], [np.mean(v)] * 2, color=INK, lw=1.3, zorder=5)
    ax.plot([x, x], [lo, hi], color=INK, lw=0.9, zorder=5)


def stacked(ax, x, v, step=0.032, **kw):
    """Integer outcomes: seeds with equal value side by side, centred on x."""
    v = np.asarray(v)
    for val in np.unique(v):
        k = int((v == val).sum())
        ax.scatter(x + (np.arange(k) - (k - 1) / 2) * step, [val] * k, **kw)


s4 = load("s4_insitu/s4_s*.json"); s4l = load("s4_lit/s4l_x1_s*.json"); s11 = load("s11_irradiation/s11_s*.json")
fig, (a, b) = plt.subplots(1, 2, figsize=(IOP_FULL, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.42))
arms = [("gp", "Gaussian-process\nplanner only", C["gp"]), ("cbo", "constrained Bayesian\noptimisation", C["gp"]),
        ("gp+logreg", "+ logistic regression", C["lin"]), ("gp+knn", "+ nearest neighbour", C["knn"]),
        ("gp+fly_best", "+ connectome model", C["conn"])]
for i, (k, lab, col) in enumerate(arms):
    for data, mk, dx, open_ in ((s4, "o", -0.2, True), (s4l, "s", 0.2, False)):
        v = [d["arms"][k]["lost"] for d in data if k in d["arms"]]
        if not v: continue
        stacked(a, i + dx, v, step=0.03, s=4.5, marker=mk, lw=0.6, zorder=3,
                facecolors="white" if open_ else col, edgecolors=col if open_ else "white")
        mean_ci(a, i + dx, v)
a.set_xticks(range(len(arms))); a.set_xticklabels([l for _, l, _ in arms], rotation=35, ha="right", fontsize=6.8)
a.set_ylabel("flakes lost of 30"); a.set_ylim(0, 13); a.set_xlim(-0.6, len(arms) - 0.4)
a.scatter([], [], marker="o", facecolors="white", edgecolors="#555555", s=10, lw=0.7, label="idealised probe")
a.scatter([], [], marker="s", color="#555555", s=10, label="probe from published in situ data")
a.legend(loc="upper right", handletextpad=0.2, borderaxespad=0.1)
panel(a, "a")

ctl = [("dose", "ion counter", C["phys"], True), ("ratio", "LA(M)/A$_{1g}$ ratio", C["phys"], False),
       ("fly", "connectome model,\noutput neurons", C["conn"], True), ("ridge", "ridge regression", C["lin"], False),
       ("knn", "nearest spectrum", C["knn"], False), ("fly_lookup", "connectome model,\nKenyon-cell code", C["conn"], False)]
rng = np.random.default_rng(0)
for i, (k, lab, col, open_) in enumerate(ctl):
    v = 100 * np.array([d["summary"][k]["mae"] for d in s11])
    b.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=7, facecolors="white" if open_ else col, edgecolors=col,
              lw=0.7, alpha=0.8, zorder=3)
    mean_ci(b, i, v, half=0.2)
b.set_xticks(range(len(ctl))); b.set_xticklabels([l for _, l, _, _ in ctl], rotation=35, ha="right", fontsize=6.8)
b.set_ylabel("vacancy-density error (%)"); b.set_ylim(0, 35)
panel(b, "b")
fig.subplots_adjust(left=0.115, right=0.99, bottom=0.32, top=0.93)
save(fig, "fig_one_shot")
