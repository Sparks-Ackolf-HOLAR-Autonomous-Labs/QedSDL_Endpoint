"""Fig. 1: the measured connectome behind the model, and where the model sits in an autonomous
2D-semiconductor line. (a) FlyWire v783 skeletons (frontal view) of the right-hemisphere uniglomerular
olfactory projection neurons (one per glomerulus), 250 of the right-hemisphere Kenyon cells and the
two MBON alpha'3 (MBON14) output neurons, over the positions of a sample of all 139,000 neurons.
(b) Measured synapse counts from the uniglomerular projection neurons, grouped by their 56 glomeruli,
to the 5,177 Kenyon cells of both hemispheres, the input wiring of the model, with its degree structure
in the margins: glomeruli feeding each Kenyon cell (top) and Kenyon cells fed by each glomerulus
(right). (c) The model at the instrument; colours match (a) and (b). Reads results/anatomy/fig1_anatomy.npz
(written by src/reflex/fig1_anatomy_prep.py). Schematic coordinates in (c) are millimetres at print size;
every label is at least 7 pt."""
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
FS, FS_S = 7.5, 7
W_MM, H_MM = 153, 126
HC = 54                                            # height of panel (c), mm
OY = HC - 46                                       # panels (a) and (b) sit above (c)
fig = plt.figure(figsize=(IOP_FULL, IOP_FULL * H_MM / W_MM))


def ax_mm(x, y, w, h):
    return fig.add_axes([x / W_MM, y / H_MM, w / W_MM, h / H_MM])


def label(x, y, s):
    fig.text(x / W_MM, y / H_MM, s, fontsize=9, fontweight="bold", va="top", ha="left", color=INK)


# (a) anatomy, frontal view, micrometres; FAFB y grows ventrally, so the axis is inverted
XL, YL = (470, 790), (320, 70)                     # crop around the right mushroom body
ax = ax_mm(3, 49 + OY, 90, 62)
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
ins = ax_mm(3, 49.5 + OY, 25, 13.5)                # whole brain, crop marked
ins.set_facecolor("#FFFFFF"); ins.patch.set_visible(True)
ins.scatter(o[:, 0], o[:, 1], s=0.12, c="#A6A6A6", lw=0, rasterized=True)
ins.add_patch(plt.Rectangle((XL[0], YL[1]), XL[1] - XL[0], YL[0] - YL[1], fill=False, ec=INK, lw=0.6))
ins.set_xlim(85, 905); ins.set_ylim(445, 45); ins.set_aspect("equal")
ins.set_xticks([]); ins.set_yticks([])
for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color(GREY)
label(1, 116 + OY, "(a)")
fig.text(9 / W_MM, (116 + OY) / H_MM, "FlyWire connectome: right mushroom body, frontal view", fontsize=8, va="top",
         ha="left", color=INK)

# (b) measured glomerulus x Kenyon cell synapse counts, with the degree structure in the margins
M = d["M"].astype(float)
sub = d["kc_sub"]; order_k = np.lexsort((-(M > 0).sum(0), sub))
order_g = np.argsort(-(M > 0).sum(1))
Ms = M[order_g][:, order_k]
deg_kc = (Ms > 0).sum(0)                           # glomeruli feeding each Kenyon cell
deg_g = (Ms > 0).sum(1)                            # Kenyon cells fed by each glomerulus
BX, BY, BW, BH = 104, 60 + OY, 36, 39
bx = ax_mm(BX, BY, BW, BH)
cmap = LinearSegmentedColormap.from_list("w2g", ["#FFFFFF", C["conn"], "#00402F"])
im = bx.imshow(np.minimum(Ms, 20), aspect="auto", cmap=cmap, interpolation="none", vmin=0, vmax=20, rasterized=True)
bx.set_xlabel("5,177 Kenyon cells, by type", fontsize=FS_S, labelpad=2)
bx.set_ylabel("56 glomeruli", fontsize=FS_S, labelpad=2)
bx.set_xticks([]); bx.set_yticks([])
for s in bx.spines.values(): s.set_visible(True); s.set_linewidth(0.5)
tx = ax_mm(BX, BY + BH + 0.8, BW, 7)               # top margin: inputs per Kenyon cell
tx.fill_between(np.arange(len(deg_kc)), deg_kc, step="mid", color=C["conn"], lw=0)
tx.set_xlim(-0.5, len(deg_kc) - 0.5); tx.set_ylim(0, deg_kc.max() + 1)
tx.set_xticks([]); tx.set_yticks([0, int(deg_kc.max())]); tx.tick_params(labelsize=6.5, length=2, pad=1)
tx.spines["bottom"].set_visible(False)
rx = ax_mm(BX + BW + 0.8, BY, 8, BH)               # right margin: Kenyon cells per glomerulus
rx.barh(np.arange(len(deg_g)), deg_g, height=1.0, color=C["conn"], lw=0)
rx.set_ylim(len(deg_g) - 0.5, -0.5); rx.set_yticks([])
rx.set_xticks([1000]); rx.set_xticklabels(["1,000"]); rx.tick_params(labelsize=6.5, length=2, pad=1)
rx.spines["left"].set_visible(False)
nsub = [(n, (sub[order_k] == n).sum()) for n in ["KCab", "KCapbp", "KCg"]]
x = 0
for n, c in nsub:
    if c: tx.text(x + c / 2, deg_kc.max() + 1.3, {"KCab": "αβ", "KCapbp": "α′β′", "KCg": "γ"}[n], ha="center",
                  va="bottom", fontsize=FS_S)
    x += c; bx.axvline(x - 0.5, color=GREY, lw=0.4); tx.axvline(x - 0.5, color=GREY, lw=0.4)
cb = fig.add_axes([BX / W_MM, (BY - 9.5) / H_MM, 20 / W_MM, 1.6 / H_MM])
fig.colorbar(im, cax=cb, orientation="horizontal", ticks=[0, 10, 20]); cb.tick_params(labelsize=6.5, length=2, pad=1)
cb.set_xticklabels(["0", "10", "≥20"])
fig.text((BX + 22) / W_MM, (BY - 8.7) / H_MM, "synapses per pair", fontsize=6.5, va="center", color=INK)
label(96, 116 + OY, "(b)")
fig.text(103 / W_MM, (116 + OY) / H_MM, "Measured input wiring", fontsize=8, va="top", ha="left", color=INK)

# (c) schematic at the instrument, millimetres
cx_ = ax_mm(0, 0, W_MM, HC); cx_.set_xlim(0, W_MM); cx_.set_ylim(0, HC); cx_.axis("off")


def box(x, y, w, h, text=None, fc="#FFFFFF", ec=INK, lw=0.7, fs=FS_S, r=1.5):
    cx_.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if text: cx_.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.2)


def arrow(p0, p1, col=INK, ls="-"):
    cx_.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, ls=ls,
                                  shrinkA=0, shrinkB=0, zorder=5))


label(1, HC, "(c)")
box(1.5, 30, 33, 20, "Planner (workstation):\nGaussian process,\nBayesian optimisation\nor language-model\n"
    "agent; minutes per\ndecision", fc="#F2F2F2")
box(1.5, 1, 33, 20, "Instrument: furnace,\nAr$^+$ irradiation or\nRaman mapping; a\nspectrum every few\nseconds",
    fc="#F2F2F2")
arrow((5, 30), (5, 21)); cx_.text(6, 25.5, "command", fontsize=FS_S, va="center")
arrow((31, 21), (31, 30), ls=(0, (3, 2))); cx_.text(30, 25.5, "status", fontsize=FS_S, va="center", ha="right")
box(37.5, 1, 79.5, 48, fc="#EAF6F1", ec=C["conn"], lw=1.0, r=2)
cx_.text(77.25, 45.5, "Edge computer at the instrument", ha="center", va="center", fontsize=FS, fontweight="bold")
steps = [(39.5, 16, "Raman\nspectrum,\n56 bands", INK), (58.5, 16, "56\nglomeruli,\none band\neach", COL["pn"]),
         (77.5, 19, "5,177\nKenyon cells,\nwiring of (b),\n5 % active", COL["kc"]),
         (99.5, 16, "familiarity\nscore\n(α′3-like\noutput)", COL["mbon"])]
for x, w, t, c in steps:
    box(x, 20, w, 21, t, ec=c, lw=0.9)
for (x0, w0, *_), (x1, *_) in zip(steps[:-1], steps[1:]):
    arrow((x0 + w0, 30.5), (x1, 30.5))
box(41, 3, 72, 13, "Nerve-cord arrangement:\ncommand in, stop decided locally, status out", ec=GREY, lw=0.5)
cx_.plot([34.5, 36], [9.5, 9.5], color=INK, lw=0.8); arrow((36, 9.5), (41, 9.5))
cx_.plot([36, 36, 37.5], [9.5, 30.5, 30.5], color=INK, lw=0.8); arrow((37.4, 30.5), (39.5, 30.5))
jobs = [(40, 9, "(i) flag a new phase\nor wrong sample"), (30, 9, "(ii) stop after one\nfailed run"),
        (20, 9, "(iii) read vacancy\ndensity"), (1, 15, "(iv) close the Se\nvalve before the\nbottom layer converts")]
for y, h, t in jobs:
    box(119.5, y, 33, h, t)
for y, h, _ in jobs[:3]:
    arrow((115.5, 30.5), (119.3, y + h / 2), col=C["conn"])
arrow((113, 8.5), (119.3, 8.5), col=C["conn"])
save(fig, "fig_concept")
