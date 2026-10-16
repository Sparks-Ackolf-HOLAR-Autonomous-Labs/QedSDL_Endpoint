#!/usr/bin/env python3
"""Figure: (a) recipe map of mean Janus fidelity (best over time and Se supply) by strip and temperature;
(b) gap to the best recipe vs runs, single agents without/with prior; (c) three-party networks.
Bands = interquartile range over seeds. Usage: python plot.py SUMMARY_JSON OUT_PNG"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmc

S = json.load(open(sys.argv[1])); A = S["arms"]
T = np.load(os.path.join(os.path.dirname(__file__), "../../results/janus/truth_Mo.npy")); R = kmc.all_recipes()
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4), constrained_layout=True)
G = kmc.GRID; M = np.zeros((len(G["strip"]), len(G["T_C"])))
for i, s in enumerate(G["strip"]):
    for j, t in enumerate(G["T_C"]):
        M[i, j] = T[(R[:, 0] == s) & (R[:, 1] == t)].max()
im = ax[0].imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0.3, vmax=0.93,
                  extent=[225, 925, -0.125, 1.125])
ax[0].set_yticks(G["strip"]); ax[0].set_xlabel("Anneal temperature (C)"); ax[0].set_ylabel("Top-layer S stripped by H$_2$ plasma")
ax[0].set_title("(a) Janus fidelity, best over time and Se supply")
fig.colorbar(im, ax=ax[0], shrink=0.85)
x = np.arange(1, 31)
col = {"random": "#9e9e9e", "gp": "#1f77b4", "fly": "#d62728", "fly_shuf": "#ff9896", "llm": "#2ca02c"}
for a, c in col.items():
    for suf, ls in (("", "--"), ("_prior", "-")):
        k = a + suf
        if k not in A or a == "fly_shuf":
            continue
        ax[1].plot(x, A[k]["median_curve"], ls, color=c, lw=1.2, label=f"{a}{' + prior' if suf else ''}")
        if suf:
            ax[1].fill_between(x, A[k]["q25"], A[k]["q75"], color=c, alpha=0.15, lw=0)
ax[1].set_title("(b) single agents"); ax[1].set_xlabel("Experiments run"); ax[1].set_ylabel("Gap to best recipe (fidelity)")
ax[1].set_ylim(0, 0.45); ax[1].legend(ncol=2, frameon=False, handlelength=1.6)
nc = {"net_none": ("#7f7f7f", "network, no prior"), "net_split": ("#9467bd", "network, split expertise"),
      "net_all": ("#000000", "network, all prior")}
for k, (c, lab) in nc.items():
    if k in A:
        ax[2].plot(x, A[k]["median_curve"], color=c, lw=1.2, label=lab)
        ax[2].fill_between(x, A[k]["q25"], A[k]["q75"], color=c, alpha=0.12, lw=0)
for k, c in (("gp_prior", "#1f77b4"), ("llm_prior", "#2ca02c")):
    if k in A:
        ax[2].plot(x, A[k]["median_curve"], ":", color=c, lw=1.0, label=k.replace("_prior", " + prior, alone"))
ax[2].set_title("(c) three parties sharing a fleet"); ax[2].set_xlabel("Experiments run (all parties)")
ax[2].set_ylim(0, 0.45); ax[2].legend(frameon=False, handlelength=1.6)
fig.savefig(sys.argv[2], dpi=300)
