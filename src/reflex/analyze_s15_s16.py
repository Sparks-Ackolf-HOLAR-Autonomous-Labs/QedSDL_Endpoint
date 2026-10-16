#!/usr/bin/env python3
"""Summaries and paired tests for S15 (trained on fixed wiring) and S16 (device specification).
Usage: python analyze_s15_s16.py S15_DIR S16_DIR OUT_DIR"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def pw(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
    return dict(diff=float(d.mean()), wins=int((d > 0).sum()), n=len(d), p=p)


def tests(rows):
    """rows: list of (label, a, b); Holm-corrected within the list."""
    out = [dict(label=l, **pw(a, b)) for l, a, b in rows]
    for r, p in zip(out, holm([r["p"] for r in out])): r["p_holm"] = float(p)
    return out


def fmt_tests(T):
    s = ["| comparison | mean difference | seeds a > b | p (Holm) |", "|---|---|---|---|"]
    s += [f"| {t['label']} | {t['diff']:+.4f} | {t['wins']}/{t['n']} | {t['p_holm']:.2g} |" for t in T]
    return s


def s15(d):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s15_s[0-9]*.json"))]
    A = [r["task_a"]["arms"] for r in R]; arms = list(A[0]); NS = list(A[0]["fly"]["auroc_by_n"])
    g = lambda a, k, n=None: np.array([x[a][k][n] if n else x[a][k] for x in A])  # noqa: E731
    md = [f"# S15: connectome model trained on fixed wiring ({len(R)} seeds)", "",
          "## Task a, selenization stop: held-out Mo AUROC by number of training probes (mean)", "",
          "| arm | " + " | ".join(f"n={n}" for n in NS) + " | MoS2->WS2 zero-shot | fine-tuned on 25 W | 25 W from scratch |",
          "|---|" + "---|" * (len(NS) + 3)]
    for a in arms:
        md.append(f"| {a} | " + " | ".join(f"{g(a, 'auroc_by_n', n).mean():.3f}" for n in NS)
                  + f" | {g(a, 'w_zero_shot').mean():.3f} | {g(a, 'w_fine_tuned').mean():.3f} | {g(a, 'w_scratch').mean():.3f} |")
    steps = list(A[0]["fly"]["auroc_by_step"])
    md += ["", "## Training speed: held-out AUROC after k optimiser steps, n = 100 (mean)", "",
           "| arm | " + " | ".join(f"{s} steps" for s in steps) + " |", "|---|" + "---|" * len(steps)]
    for a in ["fly", "fly_fixmap", "shuf", "rand", "dense"]:
        md.append(f"| {a} | " + " | ".join(f"{g(a, 'auroc_by_step', s).mean():.3f}" for s in steps) + " |")
    rows = []
    for n in NS[:3]:
        for b in ("shuf", "rand", "dense", "fly_fixmap", "logreg", "knn"):
            rows.append((f"n={n}: fly vs {b}", g("fly", "auroc_by_n", n), g(b, "auroc_by_n", n)))
    for k in ("w_zero_shot", "w_fine_tuned"):
        for b in ("shuf", "rand", "dense", "logreg"):
            rows.append((f"{k}: fly vs {b}", g("fly", k), g(b, k)))
    rows.append(("fly: fine-tuned vs from scratch", g("fly", "w_fine_tuned"), g("fly", "w_scratch")))
    rows.append(("shuf: fine-tuned vs from scratch", g("shuf", "w_fine_tuned"), g("shuf", "w_scratch")))
    rows.append(("fly: step 25 vs shuf step 25", g("fly", "auroc_by_step", "25"), g("shuf", "auroc_by_step", "25")))
    Ta = tests(rows)
    md += ["", "## Paired tests, task a (Wilcoxon signed-rank, Holm over the table)", ""] + fmt_tests(Ta)
    out = dict(n_seeds=len(R), task_a={a: dict(auroc_by_n={n: float(g(a, "auroc_by_n", n).mean()) for n in NS},
                                              w_zero_shot=float(g(a, "w_zero_shot").mean()), w_fine_tuned=float(g(a, "w_fine_tuned").mean()),
                                              w_scratch=float(g(a, "w_scratch").mean())) for a in arms}, tests_a=Ta,
               fail_rate_mo=float(np.mean([r["task_a"]["fail_rate_mo"] for r in R])), fail_rate_w=float(np.mean([r["task_a"]["fail_rate_w"] for r in R])))
    out["task_b"] = {}
    for nc in ("1", "3"):
        B = [r["task_b"][nc] for r in R]; arms_b = list(B[0])
        gb = lambda a, k: np.array([x[a][k] for x in B])  # noqa: E731
        md += ["", f"## Task b, Ar+ dose stop, {nc} calibration flake(s): success (within 15 %), lost (overshoot), mean absolute error", "",
               "| arm | success | lost | error |", "|---|---|---|---|"]
        md += [f"| {a} | {gb(a, 'success').mean():.3f} | {gb(a, 'lost').mean():.3f} | {100 * gb(a, 'mae').mean():.1f} % |" for a in arms_b]
        Tb = tests([(f"error: {b} vs fly", gb(b, "mae"), gb("fly", "mae")) for b in ("shuf", "rand", "dense", "fly_fixmap", "ridge", "knn", "ratio")])
        md += ["", "Paired tests (positive = fly has the smaller error):", ""] + fmt_tests(Tb)
        out["task_b"][nc] = dict(arms={a: {k: float(gb(a, k).mean()) for k in ("success", "lost", "mae")} for a in arms_b}, tests=Tb)
    return md, out


def s16(d):
    md, out = [], {}
    for mode in ("janus", "multi"):
        R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s16_{mode}_s*.json"))]
        if not R: continue
        cf = list(R[0]["stream"]); g = lambda c: np.array([r["stream"][c]["auroc"] for r in R])  # noqa: E731
        md += [f"# S16 device specification, {mode} stream ({len(R)} seeds)", "",
               "| configuration | AUROC | change from ideal | seeds worse | p (Holm) | writes per output device per reading |", "|---|---|---|---|---|---|"]
        devs = [c for c in cf if c not in ("ideal", "knn_budget", "zscore")]
        T = tests([(c, g(c), g("ideal")) for c in devs])
        for c, t in zip(devs, T):
            wr = R[0]["stream"][c].get("writes_per_device_per_reading")
            wv = np.mean([r["stream"][c]["writes_per_device_per_reading"] for r in R]) if wr is not None else float("nan")
            md.append(f"| {c} | {g(c).mean():.3f} | {t['diff']:+.3f} | {t['n'] - t['wins']}/{t['n']} | {t['p_holm']:.2g} | {wv:.4f} |")
        md += [f"| ideal | {g('ideal').mean():.3f} | | | | {np.mean([r['stream']['ideal']['writes_per_device_per_reading'] for r in R]):.4f} |",
               f"| nearest neighbour, equal memory | {g('knn_budget').mean():.3f} | | | | |", f"| per-band z-score | {g('zscore').mean():.3f} | | | | |", ""]
        out[mode] = dict(n_seeds=len(R), auroc={c: float(g(c).mean()) for c in cf}, tests=T,
                         writes={c: float(np.mean([r["stream"][c]["writes_per_device_per_reading"] for r in R])) for c in cf if c not in ("knn_budget", "zscore")})
        if mode == "janus" and "trained" in R[0]:
            ct = list(R[0]["trained"]); gt = lambda c: np.array([r["trained"][c] for r in R])  # noqa: E731
            Tt = tests([(c, gt(c), gt("ideal")) for c in ct if c != "ideal"])
            md += [f"# S16 trained model (S15 task a, n = 400), quantised after training ({len(R)} seeds)", "",
                   "| configuration | AUROC | change from ideal | seeds worse | p (Holm) |", "|---|---|---|---|---|",
                   f"| ideal | {gt('ideal').mean():.3f} | | | |"]
            md += [f"| {t['label']} | {gt(t['label']).mean():.3f} | {t['diff']:+.3f} | {t['n'] - t['wins']}/{t['n']} | {t['p_holm']:.2g} |" for t in Tt]
            md.append("")
            out["trained"] = dict(auroc={c: float(gt(c).mean()) for c in ct}, tests=Tt)
    return md, out


if __name__ == "__main__":
    a, b, o = sys.argv[1:4]; os.makedirs(o, exist_ok=True)
    m1, j1 = s15(a); m2, j2 = s16(b)
    open(f"{o}/s15_summary.md", "w").write("\n".join(m1) + "\n"); json.dump(j1, open(f"{o}/s15_summary.json", "w"), indent=1)
    open(f"{o}/s16_summary.md", "w").write("\n".join(m2) + "\n"); json.dump(j2, open(f"{o}/s16_summary.json", "w"), indent=1)
    print("\n".join(m1 + [""] + m2))
