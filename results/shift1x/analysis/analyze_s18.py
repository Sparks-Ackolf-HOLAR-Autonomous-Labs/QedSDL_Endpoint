#!/usr/bin/env python3
"""Summarise S18a (furnace stop), S18b (status bit after a laser drop) and S18c (Pi timing).
Paired two-sided Wilcoxon signed-rank tests by seed, Holm-corrected within each table.
Usage: python analyze_s18.py RESULT_DIR   (expects s18a/, s18b/, s18c_pi.json, s18c_dgx.json)
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

d = sys.argv[1]


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def wx(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)


L = ["# S18a-c results", ""]
A = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s18a/s18a_s[0-9]*.json"))]
arms = list(A[0]["arms"])
L += [f"## S18a: selenization stop at the instrument ({len(A)} seeds x 30 flakes)", "",
      "| Arm | flakes lost of 30 (mean) | under-converted | over-selenized | Se valve open (s) | messages up per run | x_t estimate error |",
      "|---|---|---|---|---|---|---|"]
est = {"planner_D2": "bands", "planner_D10": "bands", "planner_D60": "bands", "local_ridge_bands": "bands",
       "cord_fly": "fly", "cord_shuf": "shuf", "cord_rand": "rand"}
for a in arms:
    g = lambda m: np.mean([r["arms"][a][m] for r in A])
    e = f"{np.mean([r['xt_mae'][est[a]] for r in A]):.3f}" if a in est else ""
    L.append(f"| {a} | {g('lost'):.2f} | {30 * g('under'):.2f} | {30 * g('over'):.2f} | {g('se_open_s'):.0f} | {g('up_msgs'):.0f} | {e} |")
pairs = [("local_ridge_bands", "open_loop"), ("local_ridge_bands", "planner_D2"), ("local_ridge_bands", "planner_D10"),
         ("local_ridge_bands", "planner_D60"), ("cord_fly", "local_ridge_bands"), ("cord_fly", "cord_shuf"), ("cord_fly", "cord_rand")]
ps = [wx([r["arms"][a]["lost"] for r in A], [r["arms"][b]["lost"] for r in A]) for a, b in pairs]
L += ["", "Paired tests on flakes lost (Holm over the 7 comparisons):", ""]
for (a, b), p, q in zip(pairs, ps, holm(np.array(ps))):
    diff = np.mean([r["arms"][a]["lost"] - r["arms"][b]["lost"] for r in A])
    L.append(f"- {a} - {b}: {diff:+.2f} flakes, p = {p:.2g}, Holm {q:.2g}")
L.append(f"\nNerve-cord features on the DGX: {np.mean([r['cord_ms_per_reading'] for r in A]):.2f} ms per reading (one thread).")

for mode in ("janus", "multi"):
    B = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s18b/s18b_{mode}_s[0-9]*.json"))]
    if not B: continue
    names = [n for n in B[0]["det"] if n != "t_bit"]
    L += ["", f"## S18b: status bit after the laser drop, {'Janus line' if mode == 'janus' else 'four-family facility'} ({len(B)} seeds)", "",
          "| Detector | AUROC | false alarms, readings 100-400 after the drop | recall of new events there |", "|---|---|---|---|"]
    for n in names:
        g = lambda m: np.nanmean([r["det"][n][m] for r in B])
        L.append(f"| {n} | {g('auroc'):.3f} | {g('false_alarm'):.3f} | {g('recall'):.2f} |")
    pr = [("fly_gate", "fly"), ("fly_ctx_gate", "fly_gate"), ("fly_gate", "zscore_reset_gate"), ("fly_gate", "knn_reset_gate"),
          ("fly_ctx_gate", "fly_shuf_ctx_gate")]
    for m in ("false_alarm", "recall"):
        ps = [wx([r["det"][a][m] for r in B], [r["det"][b][m] for r in B]) for a, b in pr]
        L += ["", f"Paired tests on {m.replace('_', ' ')} (Holm over {len(pr)}):", ""]
        for (a, b), p, q in zip(pr, ps, holm(np.array(ps))):
            diff = np.nanmean([r["det"][a][m] - r["det"][b][m] for r in B])
            L.append(f"- {a} - {b}: {diff:+.3f}, p = {p:.2g}, Holm {q:.2g}")

L += ["", "## S18c: the local stop on the Raspberry Pi 5 (monsterpi, nice 19) and a DGX Grace core", "",
      "| Machine | nerve-cord stop (ms per reading) | ridge on bands (ms) | synapse entries | list at 8-bit weights, 32-bit indices | peak memory |",
      "|---|---|---|---|---|---|"]
for name in ("s18c_pi.json", "s18c_dgx.json"):
    p = os.path.join(d, name)
    if os.path.exists(p):
        t = json.load(open(p))
        L.append(f"| {t['node']} ({t['machine']}) | {t['cord_stop_ms']:.2f} | {t['bands_stop_ms']:.4f} | {t['synapse_entries']:,} | "
                 f"{t['list_bytes_8bit_32idx'] / 1e6:.1f} MB | {t['maxrss_mb']:.0f} MB |")
open(os.path.join(d, "summary.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
