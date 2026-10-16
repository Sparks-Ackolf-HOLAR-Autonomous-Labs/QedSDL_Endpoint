"""Fig. 8: which property of the FlyWire wiring matters. Measured wiring, degree-preserving shuffled
wiring and random sparse wiring with the same synapse count, paired by seed (20 seeds each).
(a) Janus line AUROC, (b) four-family facility AUROC, (d) dose error of the Kenyon-cell code lookup:
points are seeds, grey lines join one seed, black bars are means with 95 % bootstrap intervals.
(c) Selenization stop with the probe from published in situ data: flakes lost are whole numbers, so the
panel shows the paired difference per seed (measured minus shuffled, measured minus random), seeds with
the same value side by side, with the mean and its 95 % bootstrap interval.
Reads results/reflex/{s1_spectra,s1b_multi,s4_lit,s11_irradiation}."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, RES, IOP_FULL, C, INK

apply_style()


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "reflex", pat)))]


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def mean_ci(ax, x, v, half=0.25):
    lo, hi = boot_ci(v)
    ax.plot([x - half, x + half], [np.mean(v)] * 2, color=INK, lw=1.3, zorder=4)
    ax.plot([x, x], [lo, hi], color=INK, lw=0.9, zorder=4)


s1, s1b = load("s1_spectra/s1_s*.json"), load("s1b_multi/s1_s*.json")
s4l, s11 = load("s4_lit/s4l_x1_s*.json"), load("s11_irradiation/s11_s*.json")
sets = [
    ("Janus line", "AUROC", [[d["det"][k]["auroc"] for d in s1] for k in ("fly", "fly_shuf", "fly_rand")]),
    ("four-family facility", "AUROC", [[d["det"][k]["auroc"] for d in s1b] for k in ("fly", "fly_shuf", "fly_rand")]),
    ("selenization stop", "difference in flakes lost\n(measured minus other)",
     [[d["arms"][k]["lost"] for d in s4l] for k in ("gp+fly_best", "gp+fly_best_shuf", "gp+fly_best_rand")]),
    ("irradiation stop", "vacancy-density error (%)", [[100 * d["summary"][k]["mae"] for d in s11] for k in ("fly_lookup", "fly_shuf_lookup", "fly_rand_lookup")]),
]
fig, axs = plt.subplots(1, 4, figsize=(IOP_FULL, 2.4), gridspec_kw=dict(wspace=0.75))
labs, cols = ["measured", "shuffled", "random"], [C["conn"], C["shuf"], C["rand"]]
for k, (ax, (title, ylab, vals), L) in enumerate(zip(axs, sets, "abcd")):
    v = np.array(vals, float)
    if k == 2:
        for j, (i, lab) in enumerate(((1, "vs shuffled"), (2, "vs random"))):
            dlt = v[0] - v[i]
            for val in np.unique(dlt):
                n = int((dlt == val).sum())
                ax.scatter(j + (np.arange(n) - (n - 1) / 2) * 0.07, [val] * n, s=6, color=cols[i], lw=0, zorder=2)
            mean_ci(ax, j, dlt, half=0.3)
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["vs shuffled", "vs random"], rotation=35, ha="right")
        ax.set_xlim(-0.6, 1.6); ax.set_yticks([-1, 0, 1]); ax.set_ylim(-1.4, 1.4)
    else:
        for s in range(v.shape[1]): ax.plot(range(3), v[:, s], color="#BBBBBB", lw=0.4, zorder=1)
        for i in range(3):
            ax.scatter(np.full(v.shape[1], i), v[i], s=6, color=cols[i], zorder=2, lw=0)
            mean_ci(ax, i, v[i])
        ax.set_xticks(range(3)); ax.set_xticklabels(labs, rotation=35, ha="right"); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(ylab)
    ax.text(-0.45, 1.06, f"({L})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(-0.17, 1.06, title, transform=ax.transAxes, fontsize=8, va="bottom")
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.86)
save(fig, "fig_wiring")
