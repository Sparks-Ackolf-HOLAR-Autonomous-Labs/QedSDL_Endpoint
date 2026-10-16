#!/usr/bin/env python3
"""Tables for S9, S10, S12 (s9_s12.py) and S11 (s11_irradiation.py): means, interquartile ranges and paired
Wilcoxon tests over seeds. Usage: python analyze_s9_s12.py RESULTS_REFLEX_DIR"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

D = sys.argv[1]


def p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float("nan") if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)


def fmt(x): return "n/a" if x != x else (f"{x:.2g}" if x < 0.01 else f"{x:.2f}")


out = []
for mode in ("janus", "multi"):
    fs = sorted(glob.glob(f"{D}/s9_s12/{mode}/s9_s*.json"))
    if not fs: continue
    R = [json.load(open(f)) for f in fs]; dets = list(R[0]["det"])
    g = lambda n, k: np.array([r["det"][n][k] for r in R])  # noqa: E731
    stack = lambda n: np.array([r["det"][n]["auroc_by_kind"].get("MoS2/WSe2_stack", np.nan) for r in R])  # noqa: E731
    title = "Janus MoSSe selenization line (S1 stream)" if mode == "janus" else "shared facility, four material families (S1b stream)"
    out += [f"## {title}, {len(R)} seeds\n", "| Detector | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |",
            "|---|---|---|---|---|---|---|"]
    for n in dets:
        a = g(n, "auroc"); q = np.percentile(a, [25, 75]); st = stack(n)
        out.append(f"| {n} | {a.mean():.3f} | {q[0]:.3f} to {q[1]:.3f} | {g(n, 'false_alarm_after_change').mean():.3f} | "
                   f"{'' if np.isnan(st).all() else f'{np.nanmean(st):.2f}'} | {g(n, 'ms_per_reading').mean():.2f} | {int(g(n, 'stored_numbers').mean())} |")
    mo = {w: {k: np.mean([r["mapping_objective"][w][k] for r in R]) for k in ("obj_arb", "obj_var", "obj_coact")} for w in ("fly", "fly_shuf", "fly_rand")}
    out += ["", "Co-activation match (sum over band pairs of warm-up correlation x Kenyon-cell co-sampling of their glomeruli), mean:", "",
            "| Wiring | name order | variance-matched | co-activation-matched |", "|---|---|---|---|"]
    out += [f"| {w} | {v['obj_arb']:.1f} | {v['obj_var']:.1f} | {v['obj_coact']:.1f} |" for w, v in mo.items()]
    pairs = [("fly_arb", "fly_shuf_arb"), ("fly_var", "fly_shuf_var"), ("fly_coact", "fly_shuf_coact"), ("fly_coact", "fly_rand_coact"),
             ("fly_var", "fly_arb"), ("fly_coact", "fly_arb"), ("fly_2comp", "fly_arb"), ("fly_2comp", "zscore_2t"), ("zscore_2t", "zscore"),
             ("fly_left", "fly_right"), ("fly_lr_vote", "fly_arb"), ("fly_left", "fly_arb"), ("knn_budget", "fly_arb"), ("zscore", "fly_arb")]
    out += ["", "Paired Wilcoxon tests (AUROC; false alarms after laser drop):", "", "| A vs B | mean AUROC A - B | p | mean false-alarm A - B | p |", "|---|---|---|---|---|"]
    for a_, b_ in pairs:
        if a_ in dets and b_ in dets:
            x, y = g(a_, "auroc"), g(b_, "auroc"); u, v = g(a_, "false_alarm_after_change"), g(b_, "false_alarm_after_change")
            out.append(f"| {a_} vs {b_} | {x.mean() - y.mean():+.4f} | {fmt(p(x, y))} | {u.mean() - v.mean():+.3f} | {fmt(p(u, v))} |")
    out.append("")

fs = sorted(glob.glob(f"{D}/s11_irradiation/s11_s*.json"))
if fs:
    R = [json.load(open(f)) for f in fs]; arms = list(R[0]["summary"])
    g = lambda n, k: np.array([r["summary"][n][k] for r in R])  # noqa: E731
    out += [f"## S11 stop Ar+ irradiation at a target vacancy density, {len(R)} seeds x 30 flakes\n",
            "| Controller | on target (+-15 %) | overshoot (flake lost) | stopped short | mean abs. error | p (on target vs fly) |", "|---|---|---|---|---|---|"]
    for n in arms:
        out.append(f"| {n} | {g(n, 'success').mean():.2f} | {g(n, 'lost').mean():.2f} | {g(n, 'under').mean():.2f} | {g(n, 'mae').mean():.2f} | "
                   f"{'' if n == 'fly' else fmt(p(g(n, 'success'), g('fly', 'success')))} |")
    out += ["", f"Model fit to the digitised Fig. 3(a) spectra (relative rms per trace): " +
            ", ".join(f"{k}: {v['rms']:.3f}" for k, v in R[0]["model_fit"].items())]
open(f"{D}/S9_S12_SUMMARY.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
