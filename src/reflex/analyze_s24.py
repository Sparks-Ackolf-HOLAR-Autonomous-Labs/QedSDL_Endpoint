#!/usr/bin/env python3
"""Tables for S24 (results/reflex/s24_fixed_memory): writes tables.md and summary.json there.
Paired two-sided Wilcoxon signed-rank tests on per-seed values (scipy defaults); not corrected for multiplicity.
Usage: python analyze_s24.py RESULTS_DIR
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

R = sys.argv[1]
TUNE = [101, 102, 103, 104, 105]


def load(mode):
    d = {}
    for f in glob.glob(f"{R}/a_{mode}/s24a_{mode}_s*.json"):
        j = json.load(open(f)); d[j["seed"]] = j
    return d


def kb(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else (f"{b / 1e3:.1f} kB" if b >= 1e3 else f"{b} B")


def pw(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b): return 1.0
    return float(wilcoxon(a, b).pvalue)


def fp(p): return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


TIM = json.load(open(f"{R}/timing.json")) if os.path.exists(f"{R}/timing.json") else None


def ms(arm):
    if TIM is None: return "-"
    base = arm.split("@")[0]
    t = TIM["detectors"].get(base)
    return f"{t['us_per_decision'] / 1e3:.3f}" if t else "-"


def table_a(D, seeds, ref="fly_rec", title=""):
    tune = [D[s] for s in TUNE if s in D]; test = [D[s] for s in seeds]
    B = test[0]["budgets"]; arms = list(test[0]["det"])
    au = lambda rs, a: np.array([r["det"][a]["auroc"] for r in rs])  # noqa: E731
    byt = lambda a: int(np.median([r["det"][a]["bytes"] for r in test]))  # noqa: E731
    # tuned within budget: choose on tuning seeds among the arms of a family whose bytes fit
    chosen = {}
    for fam, pre in (("knn", "knn_n"), ("knnfam", "knnfam_n"), ("ae", "ae")):
        for bn in ("pack", "run"):
            cand = [a for a in arms if a.startswith(pre) and not (fam == "knn" and a.startswith("knnfam")) and byt(a) <= B[bn]]
            if cand and tune:
                c = max(cand, key=lambda a: np.mean(au(tune, a))); chosen[f"{fam} tuned <= {bn}"] = c
    ref_v = au(test, ref)
    rows, js = [], {}
    order = ["fly", "fly_rec", "fly_ctx2", "fly_ctx2_k25", "zscore", "zscore_fam"] + \
        sorted([a for a in arms if a.startswith("knn_n")], key=lambda a: int(a[5:])) + \
        sorted([a for a in arms if a.startswith("knnfam_n")], key=lambda a: int(a[8:])) + \
        sorted([a for a in arms if a.startswith("ae")], key=lambda a: (int(a[2:a.index("_")]), int(a.split("_b")[1])))
    lab = {}
    for k, a in chosen.items(): lab.setdefault(a, []).append(k)
    for a in order:
        v = au(test, a); q = np.percentile(v, [25, 75]); routed = a.startswith(("knnfam", "zscore_fam", "fly_ctx"))
        rr = "fly_ctx2" if routed and a != "fly_ctx2" else ref
        rv = au(test, rr); p = pw(v, rv) if a != rr else None
        d = dict(bytes=byt(a), median=float(np.median(v)), iqr=[float(q[0]), float(q[1])], mean=float(v.mean()), ref=rr,
                 p_vs_ref=p, median_diff=float(np.median(v - rv)), seeds_above=int((v > rv).sum()),
                 p_vs_fly_rec=pw(v, ref_v) if a != ref else None, tuned_label=lab.get(a, []))
        js[a] = d
        name = a + (f" (chosen: {', '.join(lab[a])})" if a in lab else "")
        rows.append(f"| {name} | {kb(d['bytes'])} | {d['median']:.3f} [{q[0]:.3f}, {q[1]:.3f}] | "
                    + (f"{rr}: {d['median_diff']:+.3f}, {d['seeds_above']}/{len(seeds)}, p = {fp(p)}" if p is not None else "reference")
                    + f" | {fp(d['p_vs_fly_rec']) if d['p_vs_fly_rec'] is not None else '-'} | {ms(a)} |")
    head = [f"### {title}", "", f"Budgets: plastic weights {kb(B['eq'])}; packed model {kb(B['pack'])}; model as run {kb(B['run'])}. "
            f"Seeds {seeds[0]}-{seeds[-1]}; tuned rows chosen on seeds 101-105.", "",
            "| arm | bytes held | AUROC median [IQR] | vs reference: median diff, seeds above, p | p vs fly_rec | ms per decision |",
            "|---|---|---|---|---|---|"]
    return head + rows + [""], dict(budgets=B, seeds=seeds, chosen=chosen, arms=js)


def exp_b():
    J = {j: json.load(open(f"{R}/b_stack/s24b_job_{j}.json")) for j in ("i", "ii", "iii", "iv") if os.path.exists(f"{R}/b_stack/s24b_job_{j}.json")}
    L, out = [], {}
    test = [str(s) for s in range(1, 21)]
    if "i" in J:
        P = J["i"]["per_seed"]; L += ["### Job i: flag new phases and wrong samples (s6 streams, seeds 1-20, mean AUROC)", "",
                                      "| arm | new phase | wrong sample | MoS2/WSe2 stack | p new phase / wrong sample vs model |", "|---|---|---|---|---|"]
        o = {}
        for a in P[test[0]]:
            v = {m: np.array([P[s][a][m] for s in test]) for m in ("new_phase", "wrong_sample", "stack")}
            r = {m: np.array([P[s]["module_ctx"][m] for s in test]) for m in v}
            p = (pw(v["new_phase"], r["new_phase"]), pw(v["wrong_sample"], r["wrong_sample"])) if a != "module_ctx" else None
            o[a] = dict({m: float(x.mean()) for m, x in v.items()}, p=p)
            L.append(f"| {a} | {v['new_phase'].mean():.3f} | {v['wrong_sample'].mean():.3f} | {v['stack'].mean():.3f} | "
                     + (f"{fp(p[0])} / {fp(p[1])}" if p else "reference") + " |")
        out["i"] = o; L.append("")
    if "ii" in J:
        P = J["ii"]["per_seed"]; L += ["### Job ii: stop a failing run (s4_insitu, seeds 1-20, means of 30 runs)", "",
                                       "| arm | flakes lost | false stops | p lost vs plain model | p lost vs fly_best | ms per check (harness) |", "|---|---|---|---|---|---|"]
        o = {}
        for a in P[test[0]]:
            v = np.array([P[s][a]["lost"] for s in test]); fs = np.array([P[s][a]["false_stops"] for s in test])
            r1 = np.array([P[s]["fly"]["lost"] for s in test]); r2 = np.array([P[s]["fly_best"]["lost"] for s in test])
            t = np.mean([P[s][a]["ms_per_check"] for s in test])
            o[a] = dict(lost=float(v.mean()), false_stops=float(fs.mean()), p_vs_fly=pw(v, r1), p_vs_fly_best=pw(v, r2), ms_per_check=float(t))
            L.append(f"| {a} | {v.mean():.2f} | {fs.mean():.2f} | {fp(o[a]['p_vs_fly']) if a != 'fly' else '-'} | "
                     f"{fp(o[a]['p_vs_fly_best']) if a != 'fly_best' else '-'} | {t:.2f} |")
        out["ii"] = o; L.append("")
    if "iii" in J:
        P = J["iii"]["per_seed"]; L += ["### Job iii: end irradiation at a target dose (s11, seeds 1-20, 30 flakes each)", "",
                                        "| arm | share on target | mean abs error | p share / error vs fly_lookup |", "|---|---|---|---|"]
        o = {}
        for a in P[test[0]]:
            v = np.array([P[s][a]["success"] for s in test]); e = np.array([P[s][a]["mae"] for s in test])
            rv = np.array([P[s]["fly_lookup"]["success"] for s in test]); re = np.array([P[s]["fly_lookup"]["mae"] for s in test])
            o[a] = dict(success=float(v.mean()), mae=float(e.mean()), p_success=pw(v, rv), p_mae=pw(e, re))
            L.append(f"| {a} | {v.mean():.3f} | {100 * e.mean():.1f} % | " + (f"{fp(o[a]['p_success'])} / {fp(o[a]['p_mae'])}" if a != "fly_lookup" else "reference") + " |")
        out["iii"] = o; L.append("")
    if "iv" in J:
        P = J["iv"]["per_seed"]; arms = list(P[test[0]])
        tm = {a: np.mean([P[str(s)][a] for s in TUNE]) for a in arms}
        best = {g: min([a for a in arms if a.startswith(g + "_")], key=lambda a: tm[a]) for g in ("bands", "kc")}
        L += ["### Job iv: end point at the furnace (s18a local stop, seeds 1-20, flakes lost of 30)", "",
              f"Ridge penalty chosen on seeds 101-105: bands {best['bands']}, KC code {best['kc']}.", "",
              "| arm | flakes lost (test mean) | tuning-seed mean | p vs KC read-out (chosen) |", "|---|---|---|---|"]
        o = {}
        rk = np.array([P[s][best["kc"]] for s in test])
        for a in arms:
            v = np.array([P[s][a] for s in test]); o[a] = dict(lost=float(v.mean()), tune=float(tm[a]), p_vs_kc=pw(v, rk))
            L.append(f"| {a}{' (chosen)' if a in best.values() else ''}{' (default)' if a == 'bands_0.01' else ''} | {v.mean():.2f} | {tm[a]:.2f} | "
                     + (fp(o[a]['p_vs_kc']) if a != best['kc'] else "reference") + " |")
        out["iv"] = dict(arms=o, chosen=best); L.append("")
    if TIM:
        S = TIM["serve"]; bp = S["bytes_packed"]
        L += ["### One structure against the engineered stack: bytes and time", "",
              f"Time per spectrum serving all four jobs, one Grace core ({TIM['machine']}, NumPy {TIM['numpy']}), median over 2,000 S1b spectra.", "",
              "| system | packed bytes | of which | us per spectrum, all four jobs |", "|---|---|---|---|"]
        for k, lbl in (("connectome", "connectome model (one structure)"), ("stack", "engineered stack: routed z-score, logistic regression, ridge, ridge"),
                       ("stack_alt", "stack with routed nearest neighbour (184) and nearest calibration spectrum")):
            parts = ", ".join(f"{n} {kb(v)}" for n, v in bp[k].items() if n != "total")
            L.append(f"| {lbl} | {kb(bp[k]['total'])} | {parts} | {S[k]['us_per_spectrum']:.1f} |")
        L += ["", f"Learning costs: logistic-regression refit on 30 probes {S['logreg_refit_us'] / 1e3:.2f} ms (once per completed run); "
              f"the model's one-shot compartment write {S['fly_compartment_write_us']:.1f} us. Connectome model as run in NumPy: "
              f"{kb(S['bytes_as_run_numpy']['connectome']['total'])}.", ""]
        out["bytes_time"] = S
    return L, out


L = ["# S24 tables (generated by analyze_s24.py)", ""]
summ = {}
a = load("s1b")
if a:
    t, j = table_a(a, list(range(1, 11)), title="S24a, S22b setup: S1b facility stream, 10 seeds"); L += t; summ["s1b_10"] = j
    t, j = table_a(a, list(range(1, 21)), title="S24a, s1b_multi setup: S1b facility stream, 20 seeds"); L += t; summ["s1b_20"] = j
b = load("long")
if b:
    t, j = table_a(b, list(range(1, 11)), title="S24a, S22a long stream: 100,000 readings, no laser drop, AUROC 3,000-100,000 window-weighted, 10 seeds")
    L += t; summ["long_10"] = j
Lb, ob = exp_b(); L += ["## S24b", ""] + Lb; summ["s24b"] = ob
open(f"{R}/tables.md", "w").write("\n".join(L) + "\n")
json.dump(summ, open(f"{R}/summary.json", "w"), indent=1)
print("\n".join(L))
