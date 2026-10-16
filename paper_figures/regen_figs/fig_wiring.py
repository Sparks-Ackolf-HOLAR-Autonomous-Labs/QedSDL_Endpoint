"""Fig. 4: which property of the FlyWire wiring matters. Measured wiring, degree-preserving shuffled
wiring and random sparse wiring with the same synapse count, paired by seed (20 seeds each).
(a) S1 Janus line AUROC, (b) S1b four-family facility AUROC, (c) S4L flakes lost, (d) S11 dose error
of the Kenyon-cell code lookup. Reads results/reflex/{s1_spectra,s1b_multi,s4_lit,s11_irradiation}."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK

apply_style()


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "reflex", pat)))]


s1, s1b = load("s1_spectra/s1_s*.json"), load("s1b_multi/s1_s*.json")
s4l, s11 = load("s4_lit/s4l_x1_s*.json"), load("s11_irradiation/s11_s*.json")
sets = [
    ("Janus line", "AUROC", [[d["det"][k]["auroc"] for d in s1] for k in ("fly", "fly_shuf", "fly_rand")]),
    ("four-family facility", "AUROC", [[d["det"][k]["auroc"] for d in s1b] for k in ("fly", "fly_shuf", "fly_rand")]),
    ("in-situ stop (S4L)", "flakes lost of 30", [[d["arms"][k]["lost"] for d in s4l] for k in ("gp+fly_best", "gp+fly_best_shuf", "gp+fly_best_rand")]),
    ("irradiation stop", "vacancy-density error (%)", [[100 * d["summary"][k]["mae"] for d in s11] for k in ("fly_lookup", "fly_shuf_lookup", "fly_rand_lookup")]),
]
fig, axs = plt.subplots(1, 4, figsize=(IOP_FULL, 2.3), gridspec_kw=dict(wspace=0.75))
labs, cols = ["measured", "shuffled", "random"], [C["conn"], C["shuf"], C["rand"]]
for ax, (title, ylab, vals), L in zip(axs, sets, "abcd"):
    v = np.array(vals, float)
    for s in range(v.shape[1]): ax.plot(range(3), v[:, s], color="#BBBBBB", lw=0.4, zorder=1)
    for i in range(3):
        ax.scatter(np.full(v.shape[1], i), v[i], s=6, color=cols[i], zorder=2, lw=0)
        ax.plot([i - 0.25, i + 0.25], [v[i].mean()] * 2, color=INK, lw=1.2, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, rotation=35, ha="right"); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(ylab)
    ax.text(-0.45, 1.06, f"({L})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.text(-0.17, 1.06, title, transform=ax.transAxes, fontsize=8, va="bottom")
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.25, top=0.86)
save(fig, "fig_wiring")
