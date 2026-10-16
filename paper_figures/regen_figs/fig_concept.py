"""Fig. 1: the measured connectome behind the model, and where the model sits in an autonomous
2D-semiconductor line. (a) FlyWire v783 skeletons (frontal view) of the right-hemisphere uniglomerular
olfactory projection neurons (one per glomerulus), 250 of the right-hemisphere Kenyon cells and the
two MBON alpha'3 (MBON14) output neurons, over the positions of a sample of all 139,000 neurons.
(b) Measured synapse counts from the uniglomerular projection neurons, grouped by their 56 glomeruli,
to the 5,177 Kenyon cells of both hemispheres, the input wiring of the model. (c) The model at the instrument; colours match (a) and (b). Reads results/anatomy/fig1_anatomy.npz (written by
src/reflex/fig1_anatomy_prep.py). Schematic coordinates in (c) are millimetres at print size."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, RES, IOP_FULL, C, INK, GREY

apply_style()
d = np.load(os.path.join(RES, "anatomy", "fig1_anatomy.npz"))
Q = float(d["q_um"])
COL = {"pn": "#E69F00", "kc": C["conn"], "mbon": "#0072B2"}
FS, FS_S = 7, 6.5
W_MM, H_MM = 153, 118
fig = plt.figure(figsize=(IOP_FULL, IOP_FULL * H_MM / W_MM))


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def label(x, y, s):
    fig.text(x / W_MM, y / H_MM, s, fontsize=9, fontweight="bold", va="top", ha="left", color=INK)


# (a) anatomy, frontal view, micrometres; FAFB y grows ventrally, so the axis is inverted
XL, YL = (470, 790), (320, 70)                     # crop around the right mushroom body
ax = ax_mm(3, 49, 90, 62)
o = d["outline"].astype(float)
ax.scatter(o[:, 0], o[:, 1], s=0.3, c="#DADADA", lw=0, rasterized=True, zorder=0)
for k, lw, al in [("kc", 0.15, 0.5), ("pn", 0.18, 0.75), ("mbon", 0.45, 1.0)]:
    lc = LineCollection(d[f"{k}_segs"] * Q, colors=COL[k], linewidths=lw, alpha=al, zorder=2)
    lc.set_rasterized(True); ax.add_collection(lc)
ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")
ax.plot([680, 730], [312, 312], color=INK, lw=1.0)
ax.text(705, 307, "50 µm", ha="center", va="bottom", fontsize=FS_S, color=INK)
pn = d["pn_segs"].reshape(-1, 2) * Q; kc = d["kc_segs"].reshape(-1, 2) * Q
mbo = d["mbon_segs"].reshape(-1, 2) * Q


def tag(pts, txt, xt, yt, col, ha="center"):
    xy = pts.mean(0)
    ax.annotate(txt, xy=xy, xytext=(xt, yt), fontsize=FS_S, color=col, ha=ha, va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=col, shrinkA=1, shrinkB=0), zorder=5)


tag(pn[(pn[:, 1] > 235) & (pn[:, 0] > 590)], "antennal lobe\n(projection-\nneuron dendrites)", 760, 248, COL["pn"])
tag(pn[(pn[:, 0] > 690) & (pn[:, 1] < 190)], "lateral horn", 765, 110, COL["pn"])
tag(kc[(kc[:, 0] > 640) & (kc[:, 1] < 175)], "calyx (Kenyon-cell\ndendrites)", 720, 80, COL["kc"])
tag(kc[kc[:, 0] < 560], "lobes", 500, 190, COL["kc"])
tag(mbo[mbo[:, 1] < 110], "MBON α′3\n(output)", 505, 85, COL["mbon"])
ins = ax_mm(3, 49.5, 25, 13.5)                     # whole brain, crop marked
ins.set_facecolor("#FFFFFF"); ins.patch.set_visible(True)
ins.scatter(o[:, 0], o[:, 1], s=0.12, c="#A6A6A6", lw=0, rasterized=True)
ins.add_patch(plt.Rectangle((XL[0], YL[1]), XL[1] - XL[0], YL[0] - YL[1], fill=False, ec=INK, lw=0.6))
ins.set_xlim(85, 905); ins.set_ylim(445, 45); ins.set_aspect("equal")
ins.set_xticks([]); ins.set_yticks([])
for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(GREY)
label(1, 116, "(a)")
fig.text(9 / W_MM, 116 / H_MM, "FlyWire connectome: right mushroom body, frontal view", fontsize=8, va="top", ha="left", color=INK)

# (b) measured glomerulus x Kenyon cell synapse counts
M = d["M"].astype(float)
sub = d["kc_sub"]; order_k = np.lexsort((-(M > 0).sum(0), sub))
order_g = np.argsort(-(M > 0).sum(1))
Ms = M[order_g][:, order_k]
bx = ax_mm(104, 61, 46, 48)
cmap = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", C["conn"], "#00402F"])
im = bx.imshow(np.minimum(Ms, 20), aspect="auto", cmap=cmap, interpolation="none", vmin=0, vmax=20, rasterized=True)
bx.set_xlabel("5,177 Kenyon cells (sorted by type)", fontsize=FS_S); bx.set_ylabel("56 glomeruli", fontsize=FS_S)
bx.set_xticks([]); bx.set_yticks([])
for s in bx.spines.values(): s.set_visible(True); s.set_linewidth(0.5)
nsub = [(n, (sub[order_k] == n).sum()) for n in ["KCab", "KCapbp", "KCg"]]
x = 0
for n, c in nsub:
    if c: bx.text(x + c / 2, -1.2, {"KCab": "αβ", "KCapbp": "α′β′", "KCg": "γ"}[n], ha="center", va="bottom", fontsize=FS_S)
    x += c; bx.axvline(x - 0.5, color=GREY, lw=0.4)
cb = fig.add_axes([104 / W_MM, 51 / H_MM, 20 / W_MM, 1.6 / H_MM])
fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[0, 10, 20]); cb.tick_params(labelsize=6, length=2, pad=1)
cb.set_xticklabels(["0", "10", "≥20"])
fig.text(126 / W_MM, 51.8 / H_MM, "synapses per pair", fontsize=6, va="center", color=INK)
label(96, 116, "(b)")
fig.text(103 / W_MM, 116 / H_MM, "Measured input wiring", fontsize=8, va="top", ha="left", color=INK)

# (c) schematic at the instrument, millimetres
cx_ = ax_mm(0, 0, W_MM, 46); cx_.set_xlim(0, W_MM); cx_.set_ylim(0, 46); cx_.axis("off")


def box(x, y, w, h, text=None, fc="#FFFFFF", ec=INK, lw=0.7, fs=FS, r=1.5):
    cx_.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text: cx_.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2)


def arrow(p0, p1, col=INK, ls="-"):
    cx_.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, ls=ls,
                                  shrinkA=0, shrinkB=0, zorder=5))


label(1, 46, "(c)")
box(2, 26, 30, 15, "Planner (workstation):\nGP, Bayesian\noptimisation or LLM\nagent; minutes\nper decision", fc="#F2F2F2", fs=6.2)
box(2, 1, 30, 15, "Instrument: furnace,\nAr$^+$ irradiation or\nRaman mapping;\na spectrum every few s", fc="#F2F2F2", fs=6.2)
arrow((8, 26), (8, 16)); cx_.text(9, 21, "command", fontsize=FS_S, va="center")
arrow((26, 16), (26, 26), ls=(0, (3, 2))); cx_.text(27, 21, "status", fontsize=FS_S, va="center")
box(38, 1, 78, 40, fc="#EAF6F1", ec=C["conn"], lw=1.0, r=2)
cx_.text(77, 38, "Edge computer at the instrument", ha="center", va="center", fontsize=FS, fontweight="bold")
steps = [(41, 15, "Raman\nspectrum,\n56 bands", INK), (59, 15, "56\nglomeruli,\none band\neach", COL["pn"]),
         (77, 17, "5,177 Kenyon\ncells, wiring\nof (b),\n5 % active", COL["kc"]),
         (97, 16, "familiarity\nscore\n(α′3-like\noutput)", COL["mbon"])]
for x, w, t, c in steps:
    box(x, 18, w, 16, t, ec=c, lw=0.9, fs=6.2)
for (x0, w0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
    arrow((x0 + w0, 26), (x1, 26))
box(41, 3, 72, 11, "Nerve-cord arrangement:\ncommand in, stop decided locally, status out", fs=6.2, ec=GREY, lw=0.5)
cx_.plot([32, 36], [8.5, 8.5], color=INK, lw=0.8); arrow((36, 8.5), (41, 8.5))
cx_.plot([36, 36, 39], [8.5, 26, 26], color=INK, lw=0.8); arrow((38.9, 26), (41, 26))
jobs = [(31.5, 7.5, "(i) flag a new phase\nor wrong sample"), (23, 7.5, "(ii) stop after one\nfailed run"),
        (14.5, 7.5, "(iii) read vacancy\ndensity"), (2, 10, "(iv) close the Se valve\nbefore the bottom\nlayer converts")]
for y, h, t in jobs:
    box(121, y, 31, h, t, fs=6.2)
for y, h, _ in jobs[:3]:
    arrow((113, 26), (120.6, y + h / 2), col=C["conn"])
arrow((113, 7), (120.6, 7), col=C["conn"])
save(fig, "fig_concept")
