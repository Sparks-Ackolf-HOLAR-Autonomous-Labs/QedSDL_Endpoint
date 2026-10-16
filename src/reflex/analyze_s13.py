#!/usr/bin/env python3
"""Summarise S13 (cross-animal wiring): per-detector table and paired Wilcoxon tests (Holm-corrected per stream).
Usage: python analyze_s13.py RESULTS_DIR  (expects RESULTS_DIR/{janus,multi}/s13_s*.json and wiring_stats.json)
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

A = ("fafb", "banc", "malecns")
NAME = {"fafb": "FAFB v783 (female brain)", "banc": "BANC v888 (female brain + nerve cord)",
        "malecns": "MaleCNS v1.0 (male brain + nerve cord)"}
STREAM = {"janus": "Janus MoSSe selenization line (S1 stream)", "multi": "shared facility, four material families (S1b stream)"}


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def main(d):
    st = json.load(open(f"{d}/wiring_stats.json"))
    L = ["# S13: the connectome model built from three animals", "",
         f"Shared glomerulus list: G = {st['G']} glomeruli present as uniglomerular projection neurons in all three animals; "
         f"PN-KC pairs with at least {st['threshold']} synapses.", "",
         "| Wiring | KCs (left / right) | PNs | PN-KC pairs | synapses | cross-side pairs (synapses) | mean PNs per KC |",
         "|---|---|---|---|---|---|---|"]
    for a in A + ("fafb_raw",):
        s = st[a]; ks = s["n_kc_side"]
        L.append(f"| {a} | {s['n_kc']} ({ks.get('left', 0)} / {ks.get('right', 0)}) | {s['n_pn']} | {s['pairs']} | {s['synapses']} | "
                 f"{s['cross_side_pairs']} ({s['cross_side_synapses']}) | {s['mean_pn_per_kc']:.2f} |")
    summ = {}
    for mode in ("janus", "multi"):
        R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{mode}/s13_s*.json"))]
        if not R: continue
        dets = list(R[0]["det"])
        au = {n: np.array([r["det"][n]["auroc"] for r in R]) for n in dets}
        fa = {n: np.array([r["det"][n]["false_alarm_after_change"] for r in R]) for n in dets}
        stk = {n: np.array([r["det"][n]["auroc_by_kind"].get("MoS2/WSe2_stack", np.nan) for r in R]) for n in dets}
        L += ["", f"## {STREAM[mode]}, {len(R)} seeds", "",
              "| Detector | KCs | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |",
              "|---|---|---|---|---|---|---|---|"]
        for n in dets:
            q = np.percentile(au[n], [25, 75]); kc = R[0]["arms"].get(n, {}).get("n_kc", "")
            sk = f"{np.nanmean(stk[n]):.2f}" if mode == "multi" else ""
            L.append(f"| {n} | {kc} | {au[n].mean():.3f} | {q[0]:.3f} to {q[1]:.3f} | {fa[n].mean():.3f} | {sk} | "
                     f"{np.mean([r['det'][n]['ms_per_reading'] for r in R]):.2f} | {R[0]['det'][n]['stored_numbers']} |")
        pairs = [("fafb_both", "banc_both"), ("fafb_both", "malecns_both"), ("banc_both", "malecns_both")]
        for a in A:
            pairs += [(f"{a}_left", f"{a}_right"), (f"{a}_both", f"{a}_both_nox"), (f"{a}_both", f"{a}_shuf"),
                      (f"{a}_both", f"{a}_left"), (f"{a}_both", f"{a}_right"), (f"{a}_both", f"{a}_knn"), ("zscore", f"{a}_both")]
        pairs += [("fafb_raw_both", "fafb_both")]
        rows = []
        for x, y in pairs:
            da, df = au[x] - au[y], fa[x] - fa[y]
            pa = wilcoxon(au[x], au[y]).pvalue if np.any(da) else 1.0
            pf = wilcoxon(fa[x], fa[y]).pvalue if np.any(df) else 1.0
            rows.append([x, y, da.mean(), pa, df.mean(), pf, int((da > 0).sum())])
        ha = holm(np.array([r[3] for r in rows])); hf = holm(np.array([r[5] for r in rows]))
        L += ["", "Paired Wilcoxon signed-rank tests over seeds, Holm-corrected within this stream:", "",
              "| A vs B | mean AUROC A - B | seeds A > B | p (Holm) | mean false-alarm A - B | p (Holm) |", "|---|---|---|---|---|---|"]
        for r, a_, f_ in zip(rows, ha, hf):
            L.append(f"| {r[0]} vs {r[1]} | {r[2]:+.4f} | {r[6]}/{len(R)} | {a_:.2g} | {r[4]:+.3f} | {f_:.2g} |")
        summ[mode] = dict(n_seeds=len(R), auroc={n: float(au[n].mean()) for n in dets}, false_alarm={n: float(fa[n].mean()) for n in dets},
                          tests=[dict(a=r[0], b=r[1], d_auroc=float(r[2]), p_auroc_holm=float(x_), d_fa=float(r[4]), p_fa_holm=float(y_))
                                 for r, x_, y_ in zip(rows, ha, hf)])
    open(f"{d}/summary.md", "w").write("\n".join(L) + "\n")
    json.dump(summ, open(f"{d}/summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
