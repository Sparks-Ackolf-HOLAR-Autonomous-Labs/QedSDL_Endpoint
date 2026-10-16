"""Fig. 2: decisions from a few examples at the instrument.
(a) S4 / S4L: MoS2 flakes lost in 30 selenization runs (20 seeds) with an in-situ Raman stop rule.
(b) S11: mean absolute error in sulfur-vacancy density when stopping Ar+ irradiation (20 seeds).
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


s4 = load("s4_insitu/s4_s*.json"); s4l = load("s4_lit/s4l_x1_s*.json"); s11 = load("s11_irradiation/s11_s*.json")
fig, (a, b) = plt.subplots(1, 2, figsize=(IOP_FULL, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.55))
arms = [("gp", "GP planner only", C["gp"]), ("cbo", "constrained BO", C["lin"]), ("gp+logreg", "GP + logistic reg.", C["z"]),
        ("gp+knn", "GP + nearest nbr.", C["knn"]), ("gp+fly_best", "GP + connectome model", C["conn"])]
rng = np.random.default_rng(0)
for i, (k, lab, col) in enumerate(arms):
    for j, (data, mk, dx) in enumerate([(s4, "o", -0.17), (s4l, "s", 0.17)]):
        v = [d["arms"][k]["lost"] for d in data if k in d["arms"]]
        if not v: continue
        x = i + dx
        a.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=5, color=col, alpha=0.45, marker=mk, lw=0)
        q = np.percentile(v, [25, 50, 75])
        a.plot([x - 0.1, x + 0.1], [np.mean(v)] * 2, color=INK, lw=1.2)
a.set_xticks(range(len(arms))); a.set_xticklabels([l for _, l, _ in arms], rotation=30, ha="right")
a.set_ylabel("flakes lost of 30"); a.set_ylim(0, 14)
a.scatter([], [], marker="o", color=GREY if False else "#777", s=8, label="idealised probe (S4)")
a.scatter([], [], marker="s", color="#777", s=8, label="literature-anchored probe (S4L)")
a.legend(loc="lower left", bbox_to_anchor=(0.25, 0.80), handletextpad=0.2)
panel(a, "a")

ctl = [("dose", "ion counter", C["lin"]), ("ratio", "LA(M)/A$_{1g}$ ratio", C["phys"]), ("fly", "connectome go/stop", C["rand"]),
       ("ridge", "ridge regression", C["z"]), ("knn", "nearest spectrum", C["knn"]), ("fly_lookup", "nearest KC code", C["conn"])]
for i, (k, lab, col) in enumerate(ctl):
    v = 100 * np.array([d["summary"][k]["mae"] for d in s11])
    b.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=col, alpha=0.5, lw=0)
    b.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=INK, lw=1.2)
b.set_xticks(range(len(ctl))); b.set_xticklabels([l for _, l, _ in ctl], rotation=30, ha="right")
b.set_ylabel("vacancy-density error (%)"); b.set_ylim(0, 35)
panel(b, "b")
fig.subplots_adjust(left=0.10, right=0.99, bottom=0.30, top=0.93)
save(fig, "fig_one_shot")
