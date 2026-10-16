"""Fig. 6: training the connectome model on its fixed wiring (S15) and the synapse device it needs (S16).
(a) S15 selenization stop: held-out AUROC against the number of training probes, median and interquartile
band over 20 seeds, for measured, shuffled, random sparse and dense wiring and two engineered baselines.
(b) S16 Janus line stream: AUROC against the number of conductance levels of the plastic output synapses
and of the fixed crossbar. (c) S16 four-family facility stream: AUROC against device-to-device spread and read noise.
Reads results/reflex/s15_trained/s15_s*.json and results/reflex/s16_device/s16_{janus,multi}_s*.json."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, RES, IOP_FULL, C, INK

apply_style()


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "reflex", pat)))]


s15, s16 = load("s15_trained/s15_s[0-9]*.json"), load("s16_device/s16_janus_s*.json")
s16m = load("s16_device/s16_multi_s*.json")
fig, axs = plt.subplots(1, 3, figsize=(IOP_FULL, 2.3), gridspec_kw=dict(wspace=0.45))


def band(ax, x, V, col, lab, ls="-", mk="o"):
    V = np.array(V, float); q1, m, q3 = np.percentile(V, [25, 50, 75], axis=0)
    ax.fill_between(x, q1, q3, color=col, alpha=0.18, lw=0)
    ax.plot(x, m, ls, color=col, marker=mk, ms=2.5, label=lab)


ax = axs[0]; NS = [int(n) for n in s15[0]["task_a"]["arms"]["fly"]["auroc_by_n"]]
for k, col, lab, ls in (("fly", C["conn"], "measured", "-"), ("shuf", C["shuf"], "shuffled", "-"), ("rand", C["rand"], "random sparse", "-"),
                        ("dense", C["lin"], "dense random", "--"), ("logreg", C["gp"], "logistic regression", ":"), ("knn", C["knn"], "nearest neighbour", ":")):
    band(ax, NS, [[d["task_a"]["arms"][k]["auroc_by_n"][str(n)] for n in NS] for d in s15], col, lab, ls)
ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS]); ax.minorticks_off()
ax.set_xlabel("training probes (Mo)"); ax.set_ylabel("held-out AUROC"); ax.set_ylim(0.75, 1.0)
ax.legend(loc="lower right", fontsize=6, handlelength=1.6, borderaxespad=0.2)

ax = axs[1]
ideal = [d["stream"]["ideal"]["auroc"] for d in s16]
for key, levels, col, lab in (("o_levels", [2, 4, 8, 16, 32, 64], C["conn"], "output synapses"), ("w_levels", [1, 2, 4, 8, 16], C["shuf"], "PN$\\to$KC crossbar")):
    band(ax, levels, [[d["stream"][f"{key}={L}"]["auroc"] for L in levels] for d in s16], col, lab)
ax.axhline(np.median(ideal), color=INK, lw=0.6, ls="--"); ax.text(64, np.median(ideal) + 0.003, "floating point", fontsize=6, color=INK, ha="right")
ax.axhline(np.median([d["stream"]["knn_budget"]["auroc"] for d in s16]), color=C["knn"], lw=0.6, ls=":")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "64"]); ax.minorticks_off()
ax.set_xlabel("conductance levels"); ax.set_ylabel("AUROC, Janus line"); ax.legend(loc="lower right", fontsize=6, handlelength=1.6)

ax = axs[2]
for key, vals, col, lab in (("spread", [0.05, 0.1, 0.2, 0.3, 0.5], C["phys"], "device-to-device spread"), ("read", [0.02, 0.05, 0.1, 0.2], C["z"], "read noise")):
    band(ax, [0] + vals, [[d["stream"]["ideal"]["auroc"]] + [d["stream"][f"{key}={v}"]["auroc"] for v in vals] for d in s16m], col, lab)
ax.set_xlabel("relative standard deviation"); ax.set_ylabel("AUROC, four-family facility"); ax.legend(loc="lower right", fontsize=6, handlelength=1.6)

for a_, L in zip(axs, "abc"):
    a_.text(-0.32, 1.04, f"({L})", transform=a_.transAxes, fontsize=9, fontweight="bold", va="bottom")
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.2, top=0.9)
save(fig, "fig_trained_device")
