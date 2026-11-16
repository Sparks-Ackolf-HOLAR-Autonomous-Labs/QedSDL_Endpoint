#!/usr/bin/env python3
"""Summarise S26: (a) martingale posterior neural networks as the Janus recipe planner, (b) a fast planner in the furnace
stop loop, (c) an early-exit network (BranchyNet) on the failing-run stop. Paired two-sided Wilcoxon signed-rank tests
by seed, Holm-corrected within each part. Writes SUMMARY.md and summary.json.
Usage: python analyze_s26.py RUNS_DIR RESULTS_DIR"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

runs, outd = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../../results")
os.makedirs(outd, exist_ok=True)
F_STAR = json.load(open(os.path.join(RES, "janus/summary.json")))["F_star"]
md, js = [], {}


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)


def q(v): return f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]"


# ---------------- (a) planner on the Janus map
A = {}
for f in glob.glob(f"{runs}/s26a/*_s*.json"):
    d = json.load(open(f)); A.setdefault(d["arm"], {})[d["seed"]] = d
seeds = sorted(set.intersection(*[set(v) for v in A.values()]))
md += ["# S26: martingale posterior networks and an early-exit network", "",
       f"## S26a: recipe planner on the Janus map ({len(seeds)} seeds, 30 experiments, F* = {F_STAR:.3f})", "",
       "| Arm | gap after 10 | after 20 | after 30 | seeds within 0.02 | mean gap over 30 (area) | proposal ms | update ms |",
       "|---|---|---|---|---|---|---|---|"]
area, ja = {}, {}
for arm in ["FLoRES", "LRKF", "FLoRES@0.01", "LRKF@0.01", "gp_rerun", "fly_rerun", "fly_shuf_rerun", "random_rerun"]:
    if arm not in A: continue
    cur = np.array([F_STAR - np.array(A[arm][s]["best_true_curve"]) for s in seeds])
    area[arm] = cur.mean(1)
    tp = [np.median(A[arm][s]["t_propose_s"][1:]) * 1e3 for s in seeds if len(A[arm][s]["t_propose_s"]) > 1]
    tu = [np.median(A[arm][s]["t_update_s"][1:]) * 1e3 for s in seeds if len(A[arm][s]["t_update_s"]) > 1]
    ja[arm] = dict(gap10=cur[:, 9].tolist(), gap20=cur[:, 19].tolist(), gap30=cur[:, 29].tolist(), area=area[arm].tolist(),
                   prop_ms=float(np.median(tp)) if tp else None, upd_ms=float(np.median(tu)) if tu else None)
    md.append(f"| {arm} | {q(cur[:, 9])} | {q(cur[:, 19])} | {q(cur[:, 29])} | {np.mean(cur[:, 29] <= 0.02):.2f} | "
              f"{np.median(area[arm]):.3f} | {np.median(tp) if tp else float('nan'):.1f} | {np.median(tu) if tu else float('nan'):.1f} |")
pairs = [("FLoRES", "gp_rerun"), ("LRKF", "gp_rerun"), ("FLoRES", "fly_rerun"), ("LRKF", "fly_rerun"),
         ("FLoRES", "random_rerun"), ("LRKF", "random_rerun"), ("FLoRES", "LRKF"), ("FLoRES@0.01", "gp_rerun"), ("LRKF@0.01", "gp_rerun")]
pairs = [p for p in pairs if p[0] in area and p[1] in area]
ps = [wil(area[a], area[b]) for a, b in pairs]; ph = holm(np.array(ps)) if ps else []
md += ["", "Paired Wilcoxon on the mean gap over the 30 experiments (lower is better), Holm over these comparisons:", "",
       "| A | B | median A | median B | seeds A better | p | p (Holm) |", "|---|---|---|---|---|---|---|"]
ja["tests"] = []
for (a, b), p, h in zip(pairs, ps, ph):
    nb = int(np.sum(area[a] < area[b]))
    md.append(f"| {a} | {b} | {np.median(area[a]):.3f} | {np.median(area[b]):.3f} | {nb} | {p:.2g} | {h:.2g} |")
    ja["tests"].append(dict(a=a, b=b, p=p, p_holm=float(h), a_better=nb))
md += ["", "Reproduction check: the re-run arms against results/janus/summary.md (gap after 30, median): gp 0.003, fly 0.047, "
       "fly_shuf 0.018, random 0.014.", ""]
js["s26a"] = ja

# ---------------- (b) furnace stop with a fast planner
B = [json.load(open(f)) for f in sorted(glob.glob(f"{runs}/s26b/s26b_s*.json"))]
if B:
    arms = list(B[0]["arms"])
    md += [f"## S26b: furnace stop with the planner's close delayed by D ({len(B)} seeds, 30 flakes each)", "",
           "| Arm | flakes lost (mean) | under | over |", "|---|---|---|---|"]
    jb = {}
    for a in arms:
        lost = [b["arms"][a]["lost"] for b in B]
        jb[a] = dict(lost=lost)
        md.append(f"| {a} | {np.mean(lost):.2f} | {np.mean([b['arms'][a]['under'] for b in B]) * 30:.2f} | "
                  f"{np.mean([b['arms'][a]['over'] for b in B]) * 30:.2f} |")
    ps = [wil([b["arms"][a]["lost"] for b in B], [b["arms"]["local_ridge_bands"]["lost"] for b in B]) for a in arms[1:]]
    ph = holm(np.array(ps))
    md += ["", "Paired Wilcoxon against the local stop (Holm): " + "; ".join(f"{a}: p = {p:.2g} (Holm {h:.2g})" for a, p, h in zip(arms[1:], ps, ph)), ""]
    js["s26b"] = jb

# ---------------- (c) early-exit network on the failing-run stop
C = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{runs}/s26c/s26c_s*.json")}
if C:
    S15 = {}
    for f in glob.glob(os.path.join(RES, "reflex/s15_trained/s15_s[0-9]*.json")):
        d = json.load(open(f)); S15[d["seed"]] = d
    M15 = {}
    for f in glob.glob(os.path.join(RES, "reflex/s15_mlp/s15mlp_s*.json")):
        d = json.load(open(f)); M15[d["seed"]] = d
    cs = sorted(set(C) & set(S15))
    md += [f"## S26c: early-exit network (BranchyNet) on the failing-run stop ({len(cs)} seeds; S15 task-a splits)", "",
           "Held-out AUROC, median [IQR]; exit-1 share at T = 0.2 bits (fixed in advance).", "",
           "| Training probes | BranchyNet, T = 0.2 | share leaving at exit 1 | BranchyNet, final exit only | two-layer perceptron (re-run / S15) | trained connectome model (S15) | logistic regression (S15) |",
           "|---|---|---|---|---|---|---|"]
    jc, tests = {}, []
    for n in ("25", "100", "400"):
        br = [C[s]["branchy"][n]["T=0.2"]["auroc"] for s in cs]; fr = [C[s]["branchy"][n]["T=0.2"]["exit1_frac"] for s in cs]
        bf = [C[s]["branchy"][n]["auroc_exit2"] for s in cs]; mr = [C[s]["mlp64x2"][n] for s in cs]
        mo = [M15[s]["arms"]["mlp64x2"]["auroc_by_n"][n] for s in cs if s in M15]
        fl = [S15[s]["task_a"]["arms"]["fly"]["auroc_by_n"][n] for s in cs]
        lg = [S15[s]["task_a"]["arms"]["logreg"]["auroc_by_n"][n] for s in cs]
        md.append(f"| {n} | {q(br)} | {np.median(fr):.2f} | {q(bf)} | {np.median(mr):.3f} / {np.median(mo):.3f} | {q(fl)} | {q(lg)} |")
        jc[n] = dict(branchy_T02=br, exit1_frac=fr, branchy_final=bf, mlp_rerun=mr, fly=fl, logreg=lg)
        for other, v in (("trained connectome model", fl), ("logistic regression", lg), ("BranchyNet final exit", bf)):
            tests.append((n, other, wil(br, v), int(np.sum(np.array(br) > np.array(v))), float(np.median(np.array(br) - np.array(v)))))
    ph = holm(np.array([t[2] for t in tests]))
    md += ["", "Paired Wilcoxon, BranchyNet at T = 0.2 against each (Holm over these 9):", "",
           "| probes | against | median difference | seeds BranchyNet higher | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for (n, o, p, k, dm), h in zip(tests, ph):
        md.append(f"| {n} | {o} | {dm:+.3f} | {k} | {p:.2g} | {h:.2g} |")
    sw = {T: np.median([C[s]["branchy"]["100"][f"T={T}"]["auroc"] for s in cs]) for T in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]}
    fw = {T: np.median([C[s]["branchy"]["100"][f"T={T}"]["exit1_frac"] for s in cs]) for T in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]}
    tm = {k: float(np.median([C[s]["timing"][k] for s in cs])) for k in ("early_us", "full_us")}
    md += ["", "Threshold sweep at 100 probes (AUROC / exit-1 share): " + "; ".join(f"T={T}: {sw[T]:.3f} / {fw[T]:.2f}" for T in sw),
           "", f"Time per probe, NumPy, one Grace core, batch 1: early path {tm['early_us']:.1f} us, full path {tm['full_us']:.1f} us.", ""]
    jc["timing_us"] = tm; jc["sweep_100"] = {str(T): [sw[T], fw[T]] for T in sw}
    js["s26c"] = jc

open(os.path.join(outd, "SUMMARY.md"), "w").write("\n".join(md) + "\n")
json.dump(js, open(os.path.join(outd, "summary.json"), "w"), indent=1)
print("\n".join(md))
