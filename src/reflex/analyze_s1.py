#!/usr/bin/env python3
"""Summary table for S1 (src/reflex/s1_spectra.py): means, IQR and paired Wilcoxon tests over seeds."""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon

d = sys.argv[1] if len(sys.argv) > 1 else "results/reflex/s1_spectra"
R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s1_s*.json"))]
names = list(R[0]["det"]); kinds = list(R[0]["det"]["fly"]["recall_by_kind"])
g = lambda n, k: np.array([r["det"][n][k] for r in R])
q = lambda a: f"{np.percentile(a, 25):.3f} to {np.percentile(a, 75):.3f}"
L = [f"## S1 Raman spectra as odours ({len(R)} seeds, {R[0]['G']} glomeruli = spectral bands, {np.mean([r['n_new'] for r in R]):.0f} new events in 3700 scored readings)", "",
     "| Detector | AUROC mean | IQR | recall at 5 % false alarms | false alarms after laser change | ms per reading | stored numbers | p (AUROC vs fly) |",
     "|---|---|---|---|---|---|---|---|"]
for n in names:
    p = "" if n == "fly" else f"{wilcoxon(g(n, 'auroc'), g('fly', 'auroc')).pvalue:.2g}"
    L.append(f"| {n} | {g(n, 'auroc').mean():.3f} | {q(g(n, 'auroc'))} | {g(n, 'recall_at_5fpr').mean():.2f} | "
             f"{g(n, 'false_alarm_after_change').mean():.3f} | {g(n, 'ms_per_reading').mean():.2f} | {int(g(n, 'stored_numbers').mean())} | {p} |")
L += ["", "AUROC by kind of new event:", "", "| Detector | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
for n in names:
    L.append(f"| {n} | " + " | ".join(f"{np.mean([r['det'][n]['auroc_by_kind'][k] for r in R]):.3f}" for k in kinds) + " |")
fa = {n: g(n, "false_alarm_after_change") for n in names}
L += ["", "Paired Wilcoxon on false alarms after the laser change: " + "; ".join(
    f"{n} vs fly p = {wilcoxon(fa[n], fa['fly']).pvalue:.2g}" for n in names if n != "fly" and np.any(fa[n] != fa['fly']))]
open(f"{d}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
