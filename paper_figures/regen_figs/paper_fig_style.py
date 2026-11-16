"""Figure style for the paper (Advanced Intelligent Discovery, Wiley; 153 mm text width kept).

Same practice as the group's other drafts: each data figure is drawn by one script in this folder
from committed result files only, at its true print size (IOP text width 153 mm), and included in
the draft at that width, so a 7 pt tick label is 7 pt on the page.
"""
import os
import matplotlib
import matplotlib.text
import matplotlib.legend

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IOP_FULL = 153 / 25.4  # IOP text width, inches
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))

INK, GREY = "#333333", "#8C8C8C"
# One colour per method family, the same meaning in every figure (Okabe-Ito based,
# colour-vision-deficiency safe):
#   conn  green          connectome model, measured wiring ("connectome model, Kenyon-cell code")
#   shuf  sky blue       degree-preserving shuffled wiring (used for nothing else)
#   rand  reddish purple random sparse wiring (dashed: dense random expansion); used for nothing else
#   knn   blue           nearest neighbour
#   z     dark orange    per-band z-score
#   lin   grey           linear models (ridge, logistic regression), ion counter
#   net   brown          networks trained by gradient (perceptrons, autoencoders, LSTMs)
#   gp    near black     planners (Gaussian process, language model) and known-rate bound
#   phys  vermillion     physics-based rules (band ratio, rate rule)
#   dev   black          device non-idealities (line styles distinguish them)
C = {"conn": "#009E73", "shuf": "#56B4E9", "rand": "#CC79A7", "knn": "#0072B2", "z": "#C17D00",
     "lin": "#7F7F7F", "net": "#8C510A", "gp": "#333333", "phys": "#D55E00", "dev": "#000000"}
NAME_CONN = "connectome model"


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 600, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linestyle": "--", "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    })


MIN_PT = 6.5   # smallest text at print size (figures are drawn at their printed width)


def check_text(fig, name):
    """Report text smaller than MIN_PT, text outside the figure, and pairs of text boxes that overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    items = []
    ticklabels, drawn = set(), set()
    for ax in fig.axes:
        if not ax.axison:                                      # hidden axes (drawing canvases) print no ticks
            for axis in (ax.xaxis, ax.yaxis):
                for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                    ticklabels.update((id(tk.label1), id(tk.label2)))
            continue
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                ticklabels.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks():                       # only the ticks inside the view are drawn
                drawn.update(id(l) for l in (tk.label1, tk.label2) if l.get_visible())
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in ticklabels and id(t) not in drawn:
            continue
        bb = t.get_window_extent(r)
        if bb.width < 1:
            continue
        items.append((t, bb))
        if t.get_fontsize() < MIN_PT - 1e-6:
            print(f"[{name}] small text {t.get_fontsize():.1f} pt: {t.get_text()[:40]!r}")
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            print(f"[{name}] outside figure: {t.get_text()[:40]!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if items[i][0].get_text() == items[j][0].get_text() and abs(a.x0 - b.x0) < 1 and abs(a.y0 - b.y0) < 1:
                continue                                       # the same label drawn twice (twin axes)
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            rot = items[i][0].get_rotation() % 90 or items[j][0].get_rotation() % 90
            if rot:      # rotated labels: axis-aligned boxes overlap even when the glyphs do not; shrink the test
                ox -= 0.5 * min(a.width, b.width)
            if ox > 1.5 and oy > 1.5:
                print(f"[{name}] overlap: {items[i][0].get_text()[:30]!r} / {items[j][0].get_text()[:30]!r}")
    # text drawn over data: points along every plotted line, marker and error bar (in display space, where the
    # segments are straight on any axis scale) that fall inside a label's box
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    def dense(P):
        P = P[np.isfinite(P).all(1)]
        out = [P[-1:]]
        for p, q in zip(P[:-1], P[1:]):
            k = max(1, int(np.hypot(*(q - p)) / 2))
            out.append(p + (q - p) * (np.arange(k) / k)[:, None])
        return np.concatenate(out)

    paths = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        for art in ax.get_children():
            if isinstance(art, Line2D) and art.get_visible() and len(art.get_xdata()):
                paths.append((ax.bbox, dense(art.get_transform().transform(art.get_xydata()))))
            elif isinstance(art, LineCollection) and art.get_visible():
                for seg in art.get_segments():
                    if len(seg):
                        paths.append((ax.bbox, dense(art.get_transform().transform(seg))))
    boxed = set()                                              # legend text on an opaque frame hides the data under it
    for lg in fig.findobj(matplotlib.legend.Legend):
        fr = lg.get_frame()
        if lg.get_frame_on() and fr.get_alpha() in (None, 1) and fr.get_facecolor()[3] == 1:
            boxed.update(id(t) for t in lg.get_texts())
    for t, bb in items:
        if id(t) in ticklabels or id(t) in boxed:
            continue
        for clip, P in paths:
            inside = ((P[:, 0] > bb.x0 + 1) & (P[:, 0] < bb.x1 - 1) & (P[:, 1] > bb.y0 + 1) & (P[:, 1] < bb.y1 - 1)
                      & (P[:, 0] >= clip.x0) & (P[:, 0] <= clip.x1) & (P[:, 1] >= clip.y0) & (P[:, 1] <= clip.y1))
            if inside.any():
                print(f"[{name}] text over data: {t.get_text()[:30]!r}")
                break


def save(fig, name):
    check_text(fig, name)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)


def panel(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
