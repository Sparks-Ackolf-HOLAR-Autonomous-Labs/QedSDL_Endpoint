#!/usr/bin/env python3
"""Summarise the Janus campaigns: gap to the best recipe (true mean fidelity) after 10/20/30 runs,
share of seeds within 0.02 of the best, which route was found, paired Wilcoxon tests (Holm).
Usage: python analyze.py RUN_DIR OUT_JSON OUT_MD"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmc

run, oj, om = sys.argv[1:4]
R = kmc.all_recipes()
T = np.load(os.path.join(os.path.dirname(__file__), "../../results/janus/truth_Mo.npy"))
Fstar = T.max()
runs = {}
for f in glob.glob(f"{run}/*_s*.json"):
    d = json.load(open(f)); runs.setdefault(d["arm"], {})[d["seed"]] = d
ORDER = ["random", "random_prior", "gp", "gp_prior", "fly", "fly_prior", "fly_shuf", "fly_shuf_prior",
         "llm", "llm_prior", "net_none", "net_all", "net_split"]
arms = [a for a in ORDER if a in runs]
S = {}
for a in arms:
    seeds = sorted(runs[a]); C = np.array([runs[a][s]["best_true_curve"] for s in seeds])
    gap = Fstar - C
    best_idx = [runs[a][s]["idx"][int(np.argmax(runs[a][s]["true_F"]))] for s in seeds]
    lowT = float(np.mean([R[i][0] >= 0.99 for i in best_idx]))  # best recipe uses the full H2 strip
    llm = [e for s in seeds for e in runs[a][s]["llm_log"] if "llm_s" in e]
    S[a] = dict(n=len(seeds), seeds=seeds,
                gap={n: [float(np.median(gap[:, n - 1])), float(np.percentile(gap[:, n - 1], 25)),
                         float(np.percentile(gap[:, n - 1], 75))] for n in (10, 20, 30)},
                within_002=float((gap[:, -1] <= 0.02).mean()), best_is_lowT_route=lowT,
                median_curve=np.median(gap, 0).tolist(), q25=np.percentile(gap, 25, 0).tolist(),
                q75=np.percentile(gap, 75, 0).tolist(), final=gap[:, -1].tolist(),
                llm_calls=len(llm), llm_median_s=float(np.median([e["llm_s"] for e in llm])) if llm else None,
                llm_snapped=float(np.mean([e["snapped"] for e in llm])) if llm else None,
                llm_fallbacks=sum(1 for s in seeds for e in runs[a][s]["llm_log"] if "llm_fallback" in e))
    if a.startswith("net_"):
        share = {}
        for ag in ("gp", "fly", "llm"):
            hits = [runs[a][s]["who"][int(np.argmax(runs[a][s]["true_F"]))] == ag for s in seeds]
            share[ag] = float(np.mean(hits))
        S[a]["best_found_by"] = share
pairs = [("gp", "random"), ("fly", "random"), ("llm", "random"), ("fly", "fly_shuf"), ("gp", "fly"), ("gp", "llm"),
         ("fly", "llm"), ("gp_prior", "gp"), ("fly_prior", "fly"), ("llm_prior", "llm"), ("net_none", "gp"),
         ("net_none", "llm"), ("net_all", "net_none"), ("net_split", "net_none"), ("net_all", "llm_prior"),
         ("net_all", "gp_prior")]
tests = []
for a, b in pairs:
    if a in S and b in S:
        sh = sorted(set(S[a]["seeds"]) & set(S[b]["seeds"]))
        x = np.array([Fstar - max(runs[a][s]["true_F"]) for s in sh]); y = np.array([Fstar - max(runs[b][s]["true_F"]) for s in sh])
        auc_a = np.array([np.mean(Fstar - np.array(runs[a][s]["best_true_curve"])) for s in sh])
        auc_b = np.array([np.mean(Fstar - np.array(runs[b][s]["best_true_curve"])) for s in sh])
        p = wilcoxon(auc_a, auc_b).pvalue if np.any(auc_a != auc_b) else 1.0
        tests.append(dict(a=a, b=b, n=len(sh), median_auc_a=float(np.median(auc_a)), median_auc_b=float(np.median(auc_b)),
                          a_better_seeds=int((auc_a < auc_b).sum()), p=float(p)))
ps = np.array([t["p"] for t in tests]); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run_max = 0
for r, i in enumerate(o):
    run_max = max(run_max, min(1.0, (m - r) * ps[i])); adj[i] = run_max
for t, q in zip(tests, adj):
    t["p_holm"] = float(q)
json.dump(dict(F_star=float(Fstar), best_recipe=R[int(T.argmax())].tolist(), arms=S, tests=tests), open(oj, "w"), indent=1)
L = [f"Best recipe (true mean F = {Fstar:.3f}): strip, T_C, t_min, se = {R[int(T.argmax())].tolist()}", "",
     "| Arm | Seeds | Gap after 10 | after 20 | after 30 | Seeds within 0.02 | Best recipe uses full H2 strip |", "|---|---|---|---|---|---|---|"]
for a in arms:
    g = S[a]["gap"]; L.append(f"| {a} | {S[a]['n']} | " + " | ".join(f"{g[n][0]:.3f} [{g[n][1]:.3f}, {g[n][2]:.3f}]" for n in (10, 20, 30))
                              + f" | {S[a]['within_002']:.2f} | {S[a]['best_is_lowT_route']:.2f} |")
L += ["", "Paired Wilcoxon on the mean gap over the 30 runs (area under the gap curve), Holm-corrected:", "",
      "| A | B | n | median A | median B | seeds A better | p (Holm) |", "|---|---|---|---|---|---|---|"]
L += [f"| {t['a']} | {t['b']} | {t['n']} | {t['median_auc_a']:.3f} | {t['median_auc_b']:.3f} | {t['a_better_seeds']} | {t['p_holm']:.2g} |" for t in tests]
open(om, "w").write("\n".join(L) + "\n")
print("\n".join(L))
