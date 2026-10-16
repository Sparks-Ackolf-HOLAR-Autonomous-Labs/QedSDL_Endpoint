"""Fig. 2: the Raman spectra the model reads.
(a) Simulated in situ spectra through a MoS2 selenization run, drawn from the Lorentzian line set used in
    every test (src/reflex/s1_spectra.py: 2H MoS2 E 383, A1 403; Janus MoSSe A1 290, E 355; 2H MoSe2 A1 241,
    E 287 cm-1), without noise, background or intensity change. Each state is a mixture of the three phases
    for a given top-layer and bottom-layer Se fraction (independent layers); the goal state is the furnace
    keep window (top >= 0.90, bottom <= 0.10).
(b) The goal-state spectrum binned into the 56 bands, one per glomerulus, that are the model's input
    (src/reflex/s1_spectra.py bands()).
(c) Published in situ spectra of monolayer MoS2 after Ar+ irradiation, digitised from Daniel et al.,
    2D Mater. 13, 025033 (2026), figure 3(a), CC BY 4.0 (results/digitise_daniel2026/fig3a_spectra.csv);
    each trace normalised to its own maximum and offset.
No new computation: line positions and the digitised spectra are committed files."""
import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK, GREY  # noqa: E402
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "src", "reflex")))
os.environ["S1_MODE"] = "janus"
import s1_spectra as S  # noqa: E402

apply_style()
G = 56
STATES = [(0.0, 0.0, "MoS$_2$ (start)"), (0.5, 0.0, "top layer half\nconverted"),
          (0.92, 0.05, "goal: Janus\nMoSSe, stop"), (1.0, 0.35, "bottom layer\nconverting"),
          (1.0, 0.85, "over-selenized\n(to MoSe$_2$)")]
SHADES = ["#8C8C8C", "#7FC8B3", C["conn"], "#D9895B", C["phys"]]


def spectrum(xt, xb):
    f = {"MoS2": (1 - xt) * (1 - xb), "MoSSe": xt * (1 - xb) + xb * (1 - xt), "MoSe2": xt * xb}
    return sum(w * S.BASE[k] for k, w in f.items())


fig, (a, b, c) = plt.subplots(1, 3, figsize=(IOP_FULL, 2.75), gridspec_kw=dict(width_ratios=[1.25, 1, 0.95], wspace=0.45))
for i, ((xt, xb, lab), col) in enumerate(zip(STATES, SHADES)):
    off = 1.25 * (len(STATES) - 1 - i)
    y = spectrum(xt, xb)
    a.plot(S.NU, y + off, color=col, lw=1.1 if "goal" in lab else 0.8)
    a.text(22, off + 0.08, lab, fontsize=6.5, color=col if col != C["conn"] else "#00664A", va="bottom",
           fontweight="bold" if "goal" in lab else "normal", linespacing=1.05)
for pos, name, ha in ((241, "MoSe$_2$", "right"), (290, "MoSSe", "left"), (400, "MoS$_2$", "left")):
    a.text(pos, 6.45, name, fontsize=6.5, ha=ha, va="bottom", color=INK)
    a.axvline(pos, color="#DDDDDD", lw=0.5, zorder=0)
a.set_xlim(20, 500); a.set_xticks([100, 200, 300, 400, 500]); a.set_ylim(-0.1, 6.9); a.set_yticks([])
a.spines["bottom"].set_bounds(100, 500)
a.set_xlabel("Raman shift (cm$^{-1}$)"); a.set_ylabel("intensity (offset)")
panel(a, "a")

y = spectrum(*STATES[2][:2])
x = S.bands(y, G)
edges = np.linspace(100, 100 + (len(S.NU) // G) * G * 0.5, G + 1)
b.bar(edges[:-1], x, width=np.diff(edges), align="edge", color="#CDEBE1", edgecolor=C["conn"], lw=0.4)
b.plot(S.NU, y, color=INK, lw=0.7)
b.set_xlim(100, 500); b.set_ylim(0, 1.45)
b.set_xlabel("Raman shift (cm$^{-1}$)"); b.set_ylabel("intensity")
b.text(110, 1.38, "56 bands of about 7 cm$^{-1}$,\none per glomerulus", fontsize=6.5, va="top", color="#00664A")
panel(b, "b")

rows = list(csv.DictReader(open(os.path.join(RES, "digitise_daniel2026", "fig3a_spectra.csv"))))
traces = [("pristine", "pristine"), ("1e13", "$1\\times10^{13}$ cm$^{-2}$"), ("5e13", "$5\\times10^{13}$ cm$^{-2}$"),
          ("1e14", "$10^{14}$ cm$^{-2}$")]
grey = ["#333333", "#666666", "#8C8C8C", "#B0B0B0"]
for i, ((key, lab), col) in enumerate(zip(traces, grey)):
    cm = np.array([float(r["raman_shift_cm-1"]) for r in rows if r["trace"] == key])
    iv = np.array([float(r["intensity_px"]) for r in rows if r["trace"] == key])
    o = np.argsort(cm); cm, iv = cm[o], iv[o] / iv.max()
    off = 1.1 * (len(traces) - 1 - i)
    c.plot(cm, iv + off, color=C["phys"] if i == len(traces) - 1 else col, lw=0.8)
c.set_yticks([1.1 * (len(traces) - 1 - i) + 0.15 for i in range(len(traces))], [lab for _, lab in traces], fontsize=6.5)
c.tick_params(axis="y", length=0)
c.set_ylim(-0.1, 4.6)
c.set_xlabel("Raman shift (cm$^{-1}$)")
c.text(1.0, 1.0, "published, by Ar$^+$ fluence", transform=c.transAxes, fontsize=6.5, ha="right", va="bottom")
panel(c, "c")
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.9)
save(fig, "fig_spectra")
