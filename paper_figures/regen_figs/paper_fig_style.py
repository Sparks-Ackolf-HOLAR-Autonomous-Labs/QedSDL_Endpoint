"""Figure style for the paper (Neuromorphic Computing and Engineering, IOP layout).

Same practice as the group's other drafts: each data figure is drawn by one script in this folder
from committed result files only, at its true print size (IOP text width 153 mm), and included in
the draft at that width, so a 7 pt tick label is 7 pt on the page.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IOP_FULL = 153 / 25.4  # IOP text width, inches
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.normpath(os.path.join(HERE, "..", "figures"))

INK, GREY = "#333333", "#8C8C8C"
# One colour per method family, the same in every figure (Okabe-Ito, colour-vision-deficiency safe).
C = {"conn": "#009E73", "shuf": "#56B4E9", "rand": "#CC79A7", "knn": "#0072B2", "z": "#C17D00",
     "lin": "#7F7F7F", "gp": "#333333", "phys": "#D55E00"}


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 600, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "legend.frameon": False,
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)


def panel(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
