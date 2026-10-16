"""Compare S4 (room-temperature probe lines) with S4L (literature-anchored probe), paired by seed.
Usage: python analyze_s4l.py  (from the folder root); writes results/reflex/s4_lit/summary.md"""
import json, glob
import numpy as np
from scipy.stats import wilcoxon
R = "results/reflex"
def load(pat):
    d = {}
    for f in glob.glob(pat):
        j = json.load(open(f)); d[j["seed"]] = j["arms"]
    return d
S4 = load(f"{R}/s4_insitu/s4_s*.json"); L1 = load(f"{R}/s4_lit/s4l_x1_s*.json"); L2 = load(f"{R}/s4_lit/s4l_x2_s*.json")
seeds = sorted(set(S4) & set(L1) & set(L2))
arms = [a for a in L1[seeds[0]] if a in S4[seeds[0]]]
def v(D, a, k): return np.array([D[s][a][k] for s in seeds], float)
def p(x, y):
    return float("nan") if np.allclose(x, y) else wilcoxon(x, y).pvalue
out = [f"# S4L: S4 with a literature-anchored in-situ Raman probe ({len(seeds)} seeds, paired with S4)", "",
       "GP alone loses %.2f flakes and constrained BO %.2f (no probe; identical to S4)." % (v(S4, "gp", "lost").mean(), v(S4, "cbo", "lost").mean()), "",
       "| Arm | flakes lost S4 / S4L / S4L x2 shift | false stops S4 / S4L / x2 | gap S4 / S4L / x2 | p lost S4 vs S4L |",
       "|---|---|---|---|---|"]
for a in arms:
    l = [v(D, a, "lost") for D in (S4, L1, L2)]; f = [v(D, a, "false_stops") for D in (S4, L1, L2)]; g = [v(D, a, "gap") for D in (S4, L1, L2)]
    out.append(f"| {a} | {' / '.join('%.2f' % x.mean() for x in l)} | {' / '.join('%.2f' % x.mean() for x in f)} | {' / '.join('%.3f' % x.mean() for x in g)} | {p(l[0], l[1]):.2g} |")
out.append("")
for D, name in ((L1, "S4L"), (L2, "S4L x2")):
    fb = v(D, "gp+fly_best", "lost")
    row = []
    for b in ("gp+knn", "gp+logreg", "gp+fly_best_shuf", "gp+fly_best_rand"):
        o = v(D, b, "lost"); row.append(f"vs {b} p = {p(fb, o):.2g} ({(fb - o).mean():+.2f})")
    row.append(f"vs gp p = {p(fb, v(S4, 'gp', 'lost')):.2g}; vs cbo p = {p(fb, v(S4, 'cbo', 'lost')):.2g}")
    out.append(f"{name}, paired on flakes lost, gp+fly_best " + "; ".join(row))
    out.append("")
open(f"{R}/s4_lit/summary.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
