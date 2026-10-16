"""Fig. 5: recovery rate sets the memory horizon of the connectome model over a week-long Raman stream (S5b).
(a) False alarms in the last block of 10^4 readings at the threshold set in the first block (design value 0.05).
(b) Drift-corrected return ratio of three rare phases seen once and again after 10^3, 10^4 and 6x10^4 readings
(each episode's score divided by the routine score in the 500 readings before it; below 1: the detector remembers the phase).
Top axis: memory span 1/r converted to mapping time at 1 s per spectrum (point-by-point confocal
Raman mapping, Nair et al. 2020; 10^5 readings = about 28 h).
Reads results/reflex/s5b_return/s5b_s*_*.json."""
import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK

apply_style()
R = defaultdict(list)
for f in glob.glob(os.path.join(RES, "reflex", "s5b_return", "s5b_s*_*.json")):
    d = json.load(open(f)); R[d["det"]].append(d)
rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]; names = ["fly_r1e-5", "fly_r3e-5", "fly_r1e-4", "fly_r3e-4", "fly_r1e-3"]
fig, (a, b) = plt.subplots(1, 2, figsize=(IOP_FULL, 2.7), gridspec_kw=dict(wspace=0.38))
SEC_PER_READING = 1.0


def span_axis(ax):
    """Top axis: memory span 1/r as mapping time at 1 s per spectrum."""
    top = ax.secondary_xaxis("top")
    top.set_xticks([1e-5, 1e-4, 1e-3])
    top.set_xticklabels(["28 h", "2.8 h", "17 min"])  # 1/r x 1 s: 27.8 h, 2.78 h, 16.7 min
    top.minorticks_off()
    top.set_xlabel("memory span $1/r$ (mapping at 1 s per spectrum)", fontsize=7)


def band(ax, ys, col, lab, ls="-", mk="o"):
    q = np.array([np.percentile(y, [25, 50, 75]) for y in ys])
    ax.fill_between(rates, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0)
    ax.plot(rates, q[:, 1], color=col, ls=ls, marker=mk, ms=3, label=lab)


band(a, [[r["blocks"][-1]["fa"] for r in R[n]] for n in names], C["conn"], "connectome model")
for k, col, lab in (("knn_budget", C["knn"], "nearest neighbour, equal memory"), ("zscore", C["z"], "per-band z-score")):
    v = np.median([r["blocks"][-1]["fa"] for r in R[k]]); a.axhline(v, color=col, ls="--", lw=0.9, label=lab)
a.axhline(0.05, color=INK, lw=0.5, ls=":"); a.text(1.1e-5, 0.049, "design value 0.05", fontsize=6.5, va="top")
a.set_xscale("log"); a.set_xlabel("recovery rate $r$ per reading"); a.set_ylabel("false alarms, last $10^4$ readings\n(routine spectra flagged as new)")
a.set_ylim(0, 0.085); a.legend(loc="upper left"); span_axis(a); a.text(-0.2, 1.27, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")

phases = [("alpha-MoO3", r"$\alpha$-MoO$_3$, back after $10^3$ readings (17 min)", "o"),
          ("GaSe", r"GaSe, back after $10^4$ (2.8 h)", "s"),
          ("SnS2", r"SnS$_2$, back after $6\times10^4$ (17 h)", "^")]
shades = ["#004D3A", "#009E73", "#7FD1B9"]
for (p, lab, mk), col in zip(phases, shades):
    band(b, [[r["rare"][p]["contrast_ratio"] for r in R[n]] for n in names], col, lab, mk=mk)
kn = np.median([[r["rare"][p]["contrast_ratio"] for p, _, _ in phases] for r in R["knn_budget"]])
b.axhline(kn, color=C["knn"], ls="--", lw=0.9, label="nearest neighbour, equal memory")
b.axhline(1.0, color=INK, lw=0.5, ls=":"); b.text(1.1e-5, 1.02, "1 = not remembered", fontsize=6.5, va="bottom")
for ax_ in (a, b):
    ax_.axvline(1e-4, color="#BBBBBB", lw=0.8, ls="-", zorder=0)
a.text(1.08e-4, 0.002, "setting used\nin the text", fontsize=6.5, va="bottom", color="#666666")
b.set_xscale("log"); b.set_xlabel("recovery rate $r$ per reading"); b.set_ylabel("return ratio, drift-corrected\n(lower = better remembered)")
b.set_ylim(0, 2.15); b.legend(loc="upper left", ncol=1, fontsize=6.5); span_axis(b); b.text(-0.2, 1.27, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.subplots_adjust(left=0.115, right=0.97, bottom=0.16, top=0.78)
save(fig, "fig_s5b")
