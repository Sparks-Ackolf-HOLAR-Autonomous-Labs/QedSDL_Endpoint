"""Fig. 4: one memory serves a four-family Raman facility when it is told the context (20 seeds).
AUROC of each detector of Table 1, with the numbers it stores at the end of the stream in its label: the
connectome model without and with a context input (one fixed quarter of the Kenyon cells per material
family), four quarter-size and four full-size connectome models routed by family, and the routed
equal-memory nearest neighbour and per-band z-score. (a) all new events; (b) the MoS2/WSe2 stack, a new
combination of two familiar spectra. Small points are seeds, bars the interquartile range, large markers the
median. Reads results/reflex/s8_context/s8_s*.json."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK

apply_style()
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "reflex", "s8_context", "s8_s*.json")))]
ARMS = [("fly", "connectome model,\nno context", C["conn"], "o", True),
        ("fly_ctx4", "connectome model,\ncontext input", C["conn"], "o", False),
        ("fam4_quarter", "four quarter-size\nmodels, routed", C["conn"], "s", True),
        ("fam4_full", "four full models,\nrouted", C["conn"], "^", True),
        ("knn_budget_fam", "nearest neighbour,\nequal memory, routed", C["knn"], "D", False),
        ("zscore_fam", "per-band z-score,\nrouted", C["z"], "v", False)]
fig, axs = plt.subplots(1, 2, figsize=(IOP_FULL, 2.9), sharey=True, gridspec_kw=dict(wspace=0.16))
rng = np.random.default_rng(0)
y = np.arange(len(ARMS))[::-1]
for ax, key, xlab in ((axs[0], "auroc", "AUROC, all new events"), (axs[1], "stack", "AUROC, MoS$_2$/WSe$_2$ stack")):
    for yi, (k, lab, col, mk, open_) in zip(y, ARMS):
        v = np.array([r["det"][k]["auroc"] if key == "auroc" else r["det"][k]["auroc_by_kind"]["MoS2/WSe2_stack"]
                      for r in R])
        ax.scatter(v, yi + rng.uniform(-0.17, 0.17, len(v)), s=5, color=col, alpha=0.4, lw=0)
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [yi, yi], color=col, lw=1.4)
        ax.plot(m, yi, mk, ms=5, color=col, mfc="white" if open_ else col, mew=1.0, zorder=4)
    ax.set_xlabel(xlab)
    ax.grid(axis="x", color="#E5E5E5", lw=0.5)
axs[0].set_yticks(y, [f"{lab}\n({R[0]['det'][k]['stored_numbers']:,} numbers)" for k, lab, *_ in ARMS], fontsize=6.5)
axs[0].set_xlim(0.82, 1.0); axs[1].set_xlim(0.5, 1.01)
axs[0].text(-0.02, 1.02, "(a)", transform=axs[0].transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
axs[1].text(-0.02, 1.02, "(b)", transform=axs[1].transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
fig.subplots_adjust(left=0.215, right=0.98, bottom=0.15, top=0.93)
save(fig, "fig_facility")
