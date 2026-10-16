"""Fig.: memory against data and model size (S22).
(a) Bytes each method holds against the number of Raman readings it has seen, with the RAM of a
    microcontroller, a Raspberry Pi 5 and the DGX Spark. Measured for the detectors (S22a, median of seeds);
    Gaussian process = 16 n^2 bytes (measured to n = 3,000); language model = Qwen3.5-9B context memory
    (fixed recurrent state + key/value cache, 299 tokens per reading) up to its maximum context.
(b) AUROC of new vs normal readings in successive windows of the same 100,000-reading stream (S22a).
(c) AUROC against bytes held when the model is made larger (S22b, facility stream, median and IQR).
Reads results/reflex/s22_memory_scaling/{a,b}/*.json and llm.json."""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK, GREY

apply_style()
D = os.path.join(RES, "reflex", "s22_memory_scaling")
A = []
for f in sorted(glob.glob(os.path.join(D, "a", "s22a_s[0-9]*.json"))):
    s_ = json.load(open(f)); g = os.path.join(D, "a", f"s22a_rec_s{s_['seed']}.json")
    if os.path.exists(g): s_["det"].update(json.load(open(g))["det"])
    A.append(s_)
A = [s_ for s_ in A if "fly_rec" in s_["det"]] or A
B = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, "b", "s22b_s*.json")))]
Q = json.load(open(os.path.join(D, "llm.json")))["Qwen3.5-9B"]
ARMS = [("fly", "connectome model", C["conn"], "-"), ("fly_rec", "connectome model, slow recovery", C["conn"], "--"), ("zscore", "per-band z-score", C["z"], "-"),
        ("knn_budget", "nearest neighbour, equal memory", C["knn"], "-"), ("knn_full", "nearest neighbour, every reading", C["knn"], "--"),
        ("ae_res", "autoencoder, fixed replay buffer", C["net"], "-"), ("ae_full", "autoencoder, replays every reading", C["net"], "--")]
fig, (a, b, c) = plt.subplots(1, 3, figsize=(IOP_FULL, 3.4), gridspec_kw=dict(wspace=0.72))

N = np.array([r["N"] for r in A[0]["det"]["fly"]])
for k, lab, col, ls in ARMS:
    if k not in A[0]["det"]: continue
    a.plot(N, np.median([[r["bytes"] for r in s["det"][k]] for s in A], 0), color=col, ls=ls, marker="o", ms=2.5, label=lab)
n = np.logspace(2, 5, 30); a.plot(n, 16 * n ** 2 + 448 * n, color=C["gp"], ls=":", label="Gaussian process")
nl = np.logspace(2, np.log10(Q["max_context"] / Q["tokens_per_reading"]), 20)
a.plot(nl, Q["linear_state_bytes"] + Q["kv_bytes_per_token"] * Q["tokens_per_reading"] * nl, color=C["gp"], ls="-.",
       label="9-billion-parameter language model,\nreadings kept in its prompt")
a.plot([nl[-1]], [Q["linear_state_bytes"] + Q["kv_bytes_per_token"] * Q["max_context"]], marker="x", color=C["gp"], ms=4)
RAM = ((264e3, "Pico"), (8e9, "Pi 5"), (128e9, "DGX"))
for y, t in RAM:
    a.axhline(y, color=C["dev"], lw=0.6, ls=(0, (4, 2)))
ram_ax = a.twinx()
ram_ax.set_yscale("log"); ram_ax.set_ylim(100, 1e12)
ram_ax.set_yticks([y for y, _ in RAM], [t for _, t in RAM])
ram_ax.minorticks_off(); ram_ax.tick_params(axis="y", length=2, pad=1, labelsize=6.3)
ram_ax.spines["right"].set_visible(False)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(100, 1e5); a.set_ylim(100, 1e12)
a.set_xlabel("readings seen"); a.set_ylabel("memory held (bytes)"); panel(a, "a")

for k, lab, col, ls in ARMS:
    if k not in A[0]["det"]: continue
    y = np.array([[r["auroc_window"] for r in s["det"][k][1:]] for s in A])
    q = np.percentile(y, [25, 50, 75], 0)
    b.fill_between(N[1:], q[0], q[2], color=col, alpha=0.15, lw=0); b.plot(N[1:], q[1], color=col, ls=ls, marker="o", ms=2.5)
b.set_xscale("log"); b.set_xlabel("readings seen (window end)"); b.set_ylabel("AUROC in window"); b.set_ylim(0.6, 1.0); panel(b, "b")


def med_iqr(k):
    au = [r["det"][k]["auroc"] for r in B]; by = np.median([r["det"][k]["bytes"] for r in B])
    return by, np.median(au), np.percentile(au, 25), np.percentile(au, 75)


for keys, lab, col, mk in (([f"kc_x{s:g}" for s in (0.25, 0.5, 1, 2, 4, 8)], "connectome model resized,\n0.25 to 8 times the Kenyon cells", C["conn"], "o"),
                           ([f"knn_eq{s:g}" for s in (0.25, 0.5, 1, 2, 4, 8)], "nearest neighbour,\nsame plastic memory", C["knn"], "s"),
                           ([f"ae{h}" for h in (16, 64, 256, 1024)], "autoencoder, 16 to\n1,024 hidden units", C["net"], "^")):
    v = np.array([med_iqr(k) for k in keys])
    c.errorbar(v[:, 0], v[:, 1], yerr=[v[:, 1] - v[:, 2], v[:, 3] - v[:, 1]], color=col, marker=mk, ms=3, lw=0.9, capsize=1.5, label=lab)
v = med_iqr("knn_full"); c.errorbar([v[0]], [v[1]], yerr=[[v[1] - v[2]], [v[3] - v[1]]], color=C["knn"], marker="D", mfc="white", ms=3.5,
                                     capsize=1.5, ls="none", label="nearest neighbour,\nevery reading")
c.set_xscale("log"); c.set_xlabel("memory held (bytes)"); c.set_ylabel("AUROC"); c.set_ylim(0.7, 1.0); panel(c, "c")
h1, l1 = a.get_legend_handles_labels(); h3, l3 = c.get_legend_handles_labels()
fig.legend(h1, l1, loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=2, fontsize=6.3, columnspacing=1.2, handlelength=2.4,
           title="(a) and (b)", title_fontsize=6.5, alignment="left")
fig.legend(h3, l3, loc="lower left", bbox_to_anchor=(0.69, 0.035), ncol=1, fontsize=6.3, handlelength=2.0,
           title="(c)", title_fontsize=6.5, alignment="left")
fig.subplots_adjust(left=0.09, right=0.99, bottom=0.44, top=0.93, wspace=0.72)
save(fig, "fig_memory_scaling")
