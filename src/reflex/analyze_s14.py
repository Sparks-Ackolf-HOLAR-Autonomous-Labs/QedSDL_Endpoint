#!/usr/bin/env python3
"""Summarise S14 (left-right decomposition): AUROC table and paired Wilcoxon tests, Holm-corrected per stream.
Usage: python analyze_s14.py RESULTS_DIR  (expects RESULTS_DIR/{janus,multi}/s14_s*.json)
"""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_s13 import holm  # noqa: E402

A = ("fafb", "banc", "malecns")
ARMS = ("left", "right", "left_nm", "right_nm", "left_gp", "right_gp", "LL", "LR", "RL", "RR")
STREAM = {"janus": "Janus MoSSe selenization line (S1 stream)", "multi": "shared facility, four material families (S1b stream)"}


def main(d):
    L = ["# S14: left versus right mushroom body, split into Kenyon-cell wiring and projection-neuron input", ""]
    out = {}
    for m, title in STREAM.items():
        R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{m}/s14_s*.json"))]
        au = {n: np.array([r["det"][n]["auroc"] for r in R]) for n in R[0]["det"]}
        L += [f"## {title}, {len(R)} seeds", "", "| Arm | " + " | ".join(A) + " |", "|---|" + "---|" * len(A)]
        for k in ARMS:
            L.append(f"| {k} | " + " | ".join(f"{au[f'{a}_{k}'].mean():.3f}" for a in A) + " |")
        L += ["", f"Per-band z-score: {au['zscore'].mean():.3f}", ""]
        tests = []
        for a in A:
            g = lambda k: au[f"{a}_{k}"]
            tests += [(a, "left - right (measured)", g("left") - g("right")),
                      (a, "left_nm - right_nm (multiglomerular PNs silenced)", g("left_nm") - g("right_nm")),
                      (a, "LL - left_nm (exact left wiring vs its side shuffle)", g("LL") - g("left_nm")),
                      (a, "RR - right_nm (exact right wiring vs its side shuffle)", g("RR") - g("right_nm")),
                      (a, "LL - RR (side shuffles)", g("LL") - g("RR")),
                      (a, "left_gp - right_gp (random band map, same for both sides)", g("left_gp") - g("right_gp")),
                      (a, "KC effect: (LL+LR)/2 - (RL+RR)/2", (g("LL") + g("LR") - g("RL") - g("RR")) / 2),
                      (a, "PN effect: (LL+RL)/2 - (LR+RR)/2", (g("LL") + g("RL") - g("LR") - g("RR")) / 2),
                      (a, "interaction: (LL-LR) - (RL-RR)", g("LL") - g("LR") - g("RL") + g("RR"))]
        ps = holm(np.array([wilcoxon(x).pvalue if np.any(x != 0) else 1.0 for _, _, x in tests]))
        L += ["| Animal | Contrast | mean | seeds > 0 | p (Holm) |", "|---|---|---|---|---|"]
        out[m] = []
        for (a, nm, x), p in zip(tests, ps):
            L.append(f"| {a} | {nm} | {x.mean():+.4f} | {(x > 0).sum()}/{len(x)} | {p:.2g} |")
            out[m].append(dict(animal=a, contrast=nm, mean=float(x.mean()), pos=int((x > 0).sum()), p_holm=float(p)))
        L.append("")
    open(f"{d}/summary.md", "w").write("\n".join(L)); json.dump(out, open(f"{d}/summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
