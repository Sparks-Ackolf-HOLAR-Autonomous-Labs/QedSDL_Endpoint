#!/usr/bin/env python3
"""Summarise S5b (recovery-rate sweep with returning rare phases). Usage: analyze_s5b.py RESULT_DIR > summary.md
Per detector, mean over seeds: AUROC first/last block, false alarms in the last block, and per rare phase the
recall at first sighting / on return and the return ratio (return score / first score). Paired Wilcoxon on the
return ratio and last-block AUROC: each fly rate vs knn_budget."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

D = sys.argv[1]; R = defaultdict(dict)
for f in glob.glob(os.path.join(D, "s5b_s*_*.json")):
    d = json.load(open(f)); R[d["det"]][d["seed"]] = d
order = ["fly_r0", "fly_r1e-5", "fly_r3e-5", "fly_r1e-4", "fly_r3e-4", "fly_r1e-3", "knn_budget", "zscore"]
order = [o for o in order if o in R]; phases = ["alpha-MoO3", "GaSe", "SnS2"]
print(f"## S5b: recovery-rate sweep with returning rare phases ({min(len(R[o]) for o in order)}-{max(len(R[o]) for o in order)} seeds)\n")
print("| Detector | seeds | AUROC first / last block | false alarms last block | " + " | ".join(f"{p} recall first / return, ratio" for p in phases) + " | ms per reading |")
print("|---" * (5 + len(phases)) + "|")
summ = {}
for o in order:
    rs = list(R[o].values()); m = lambda f: float(np.mean([f(r) for r in rs]))
    row = dict(n=len(rs), a0=m(lambda r: r["blocks"][0]["auroc"]), a9=m(lambda r: r["blocks"][-1]["auroc"]), fa9=m(lambda r: r["blocks"][-1]["fa"]),
               ms=m(lambda r: r["ms"]))
    cells = []
    for p in phases:
        f1 = m(lambda r: r["rare"][p]["first"]["recall"]); f2 = m(lambda r: r["rare"][p]["return"]["recall"]); q = m(lambda r: r["rare"][p]["return_ratio"])
        c = m(lambda r: r["rare"][p].get("contrast_ratio", np.nan))
        row[p] = dict(first=f1, ret=f2, ratio=q, contrast=c); cells.append(f"{f1:.2f} / {f2:.2f}, {q:.2f} ({c:.2f})")
    summ[o] = row
    print(f"| {o} | {row['n']} | {row['a0']:.3f} / {row['a9']:.3f} | {row['fa9']:.3f} | " + " | ".join(cells) + f" | {row['ms']:.2f} |")
print("\nReturn ratio = mean score of the return episode / mean score of the first episode (below 1: remembered; about 1: "
      "treated as new). In brackets: contrast ratio, the same after dividing each episode's score by the mean routine score "
      "in the 500 readings before it (removes general drift).\n")
T = []
for o in order:
    if o == "knn_budget" or "knn_budget" not in R: continue
    s = sorted(set(R[o]) & set(R["knn_budget"]))
    if len(s) < 5: continue
    a = np.array([R[o][i]["blocks"][-1]["auroc"] for i in s]); b = np.array([R["knn_budget"][i]["blocks"][-1]["auroc"] for i in s])
    T.append((o, "last-block AUROC", a.mean() - b.mean(), int((a > b).sum()), len(s), wilcoxon(a, b).pvalue))
    for p in phases:
        a = np.array([R[o][i]["rare"][p]["contrast_ratio"] for i in s]); b = np.array([R["knn_budget"][i]["rare"][p]["contrast_ratio"] for i in s])
        T.append((o, f"{p} contrast ratio", a.mean() - b.mean(), int((a > b).sum()), len(s), wilcoxon(a, b).pvalue))
hp = np.minimum(1, np.maximum.accumulate((len(T) - np.arange(len(T))) * np.sort([t[-1] for t in T])))[np.argsort(np.argsort([t[-1] for t in T]))] if T else []
print(f"Paired Wilcoxon against the equal-memory kNN (seeds present in both), Holm over {len(T)} comparisons:\n")
print("| detector | measure | difference (detector minus kNN) | seeds detector higher | p | p (Holm) |\n|---|---|---|---|---|---|")
for t, h in zip(T, hp): print(f"| {t[0]} | {t[1]} | {t[2]:+.3f} | {t[3]} / {t[4]} | {t[5]:.2g} | {h:.2g} |")
summ["tests_vs_knn"] = [dict(det=t[0], measure=t[1], diff=float(t[2]), higher=t[3], n=t[4], p=float(t[5]), p_holm=float(h)) for t, h in zip(T, hp)]
json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)
