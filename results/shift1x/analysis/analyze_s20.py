#!/usr/bin/env python3
"""Summarise S20 (time-course stop): flakes lost per arm, paired Wilcoxon tests (Holm) against the LSTM."""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/reflex/s20_timecourse")
ARMS = ["now_ridge", "snap_ridge", "snap_mlp", "snap_kc", "seq_ridge", "lstm", "lstm_tuned", "seq_kc", "trend_rule", "oracle_k"]
out, lines, summ = {}, ["# S20: stop from the time course (slow purge), 20 seeds x 30 flakes", ""], {}
for nc in (3, 10):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/s20_cal{nc}_s*[0-9].json"))]
    if not R: continue
    for r in R:  # LSTM rerun with the setting chosen on tuning seeds 101-105 (lr 1e-2, 600 epochs, 16 units)
        t = json.load(open(f"{D}/s20_cal{nc}_s{r['seed']}_lstmtuned.json"))
        r["arms"]["lstm_tuned"] = t["arms"]["lstm"]; r["final_xt_mae"]["lstm_tuned"] = t["final_xt_mae"]["lstm"]
    lines += [f"## {nc} calibration flakes ({len(R)} seeds)", "",
              "| arm | lost of 30 | under-converted | over-selenized | lost among fast flakes | final x_t error (held out) |", "|---|---|---|---|---|---|"]
    s = {}
    for a in ARMS:
        L = np.array([r["arms"][a]["lost"] for r in R])
        s[a] = dict(lost=float(L.mean()), lost_sd=float(L.std(ddof=1)), under=float(np.mean([r["arms"][a]["under"] for r in R])),
                    over=float(np.mean([r["arms"][a]["over"] for r in R])),
                    lost_fast=float(np.sum([r["arms"][a]["lost_fast"] for r in R])), n_fast=int(np.sum([r["arms"][a]["n_fast"] for r in R])),
                    err=float(np.mean([r["final_xt_mae"][a] for r in R])) if a in R[0]["final_xt_mae"] else None, per_seed=L.tolist())
        e = "" if s[a]["err"] is None else f"{s[a]['err']:.3f}"
        lines.append(f"| {a} | {s[a]['lost']:.2f} | {s[a]['under']:.2f} | {s[a]['over']:.2f} | {s[a]['lost_fast']:.0f} of {s[a]['n_fast']} | {e} |")
    pairs = [("lstm_tuned", b) for b in ["snap_mlp", "snap_ridge", "seq_ridge", "now_ridge", "lstm"]] + [("seq_kc", b) for b in ["lstm_tuned", "snap_kc", "seq_ridge"]] + [("snap_kc", "snap_mlp")]
    tests = []
    for a, b in pairs:
        x, y = np.array(s[a]["per_seed"]), np.array(s[b]["per_seed"])
        p = 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)
        tests.append([a, b, float((x - y).mean()), int((x < y).sum()), int((x > y).sum()), p])
    order = np.argsort([t[-1] for t in tests]); m = len(tests); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * tests[j][-1])); tests[j].append(run)
    lines += ["", "Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:", "",
              "| arm | reference | difference | seeds better / worse | p (Holm) |", "|---|---|---|---|---|"]
    lines += [f"| {t[0]} | {t[1]} | {t[2]:+.2f} | {t[3]} / {t[4]} | {t[6]:.3g} |" for t in tests]
    lines.append("")
    summ[nc] = dict(arms=s, tests=[dict(arm=t[0], ref=t[1], diff=t[2], better=t[3], worse=t[4], p=t[5], p_holm=t[6]) for t in tests],
                    params=R[0]["params"], fit_s=float(np.mean([r["fit_s"] for r in R])))
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1)
open(f"{D}/summary.md", "w").write("\n".join(lines)); print("\n".join(lines))
