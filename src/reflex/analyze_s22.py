#!/usr/bin/env python3
"""Summarise S22 (memory against data and model size) into results/reflex/s22_memory_scaling/summary.md.

Reads a/s22a_s*.json (data growth), b/s22b_s*.json (model size), gp_probe.json, mace.json, llm.json and
pi.json from the result folder. Paired tests: two-sided Wilcoxon signed-rank over seeds, Holm-corrected
within each table.
Usage: python analyze_s22.py RESULT_DIR
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

D = sys.argv[1]
DEV = {"Raspberry Pi Pico (RP2040) RAM": 264e3, "Raspberry Pi 5 RAM": 8e9, "DGX Spark unified memory": 128e9}


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj


def fmt_b(b):
    for u, s in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= s: return f"{b / s:.3g} {u}"
    return f"{b:.0f} B"


L = []
A = []
for f in sorted(glob.glob(f"{D}/a/s22a_s[0-9]*.json")):
    a = json.load(open(f)); g = f"{D}/a/s22a_rec_s{a['seed']}.json"
    if os.path.exists(g): a["det"].update(json.load(open(g))["det"])
    A.append(a)
A = [a for a in A if "fly_rec" in a["det"]] or A
if A:
    dets = list(A[0]["det"]); Ns = [r["N"] for r in A[0]["det"]["fly"]]
    L += [f"## S22a: memory against readings seen ({len(A)} seeds, facility stream without laser drop)", "",
          "Bytes held / median us per reading (last 200) / AUROC in the window ending at N (median over seeds).", "",
          "| detector | " + " | ".join(f"N={n:,}" for n in Ns) + " |", "|---|" + "---|" * len(Ns)]
    for d in dets:
        cells = []
        for k in range(len(Ns)):
            b = np.median([a["det"][d][k]["bytes"] for a in A]); us = np.median([a["det"][d][k]["us_per_reading"] for a in A])
            au = [a["det"][d][k]["auroc_window"] for a in A]
            cells.append(f"{fmt_b(b)} / {us:.0f} / " + (f"{np.median(au):.3f}" if au[0] is not None else "-"))
        L.append(f"| {d} | " + " | ".join(cells) + " |")
    # AUROC over all windows after 3k (readings 3,000-100,000), paired against the connectome model
    def late(a, d):
        w = [r for r in a["det"][d] if r["N"] > 3000]; n = np.array([r["window"][1] - r["window"][0] for r in w])
        return float(np.sum(n * np.array([r["auroc_window"] for r in w])) / n.sum())
    L += ["", "AUROC over readings 3,000-100,000 (window-size-weighted), paired vs connectome model:", "",
          "| detector | median | vs model (median diff) | seeds above model | Holm p |", "|---|---|---|---|---|"]
    others = [d for d in dets if d != "fly"]
    diffs = [[late(a, d) - late(a, "fly") for a in A] for d in others]
    ps = [wilcoxon(x).pvalue if len(A) > 1 and np.any(np.array(x) != 0) else 1.0 for x in diffs]
    L.append(f"| fly | {np.median([late(a, 'fly') for a in A]):.3f} | | | |")
    for d, x, p in zip(others, diffs, holm(np.array(ps))):
        L.append(f"| {d} | {np.median([late(a, d) for a in A]):.3f} | {np.median(x):+.3f} | {sum(v > 0 for v in x)}/{len(x)} | {p:.3g} |")
    pairs = [("knn_full", "knn_budget"), ("ae_full", "ae_res"), ("knn_budget", "fly_rec"), ("fly_rec", "fly")]
    xs = [[late(a, u) - late(a, v) for a in A] for u, v in pairs]
    L += ["", "Growing against fixed memory (same method family):", "", "| comparison | median diff | seeds above | Holm p |", "|---|---|---|---|"]
    for (u, v), x, p in zip(pairs, xs, holm(np.array([wilcoxon(x).pvalue for x in xs]))):
        L.append(f"| {u} - {v} | {np.median(x):+.3f} | {sum(t > 0 for t in x)}/{len(x)} | {p:.3g} |")
    L += ["", f"With {len(A)} seeds the smallest two-sided Wilcoxon p is {2 / 2 ** len(A):.3g}.", ""]

B = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/b/s22b_s*.json"))]
if B:
    dets = list(B[0]["det"])
    L += [f"## S22b: detection against model size ({len(B)} seeds, facility stream S1b)", "",
          "| arm | bytes held | of which plastic | us per reading | AUROC median [IQR] |", "|---|---|---|---|---|"]
    for d in dets:
        v = [b["det"][d] for b in B]; au = [x["auroc"] for x in v]
        plastic = v[0]["n_kc"] * 4 if v[0]["n_kc"] else v[0]["bytes"]
        L.append(f"| {d} | {fmt_b(np.median([x['bytes'] for x in v]))} | {fmt_b(plastic)} | {np.median([x['us_per_reading'] for x in v]):.0f} | "
                 f"{np.median(au):.3f} [{np.percentile(au, 25):.3f}, {np.percentile(au, 75):.3f}] |")
    comps = [("kc_x8", "kc_x1"), ("kc_x0.25", "kc_x1"), ("kc_x1", "fly_measured"), ("knn_eq1", "kc_x1"), ("knn_eq8", "kc_x8"),
             ("knn_full", "knn_eq1"), ("ae1024", "ae16"), ("ae16", "kc_x1")]
    ps, rows = [], []
    for a, b in comps:
        x = [r["det"][a]["auroc"] - r["det"][b]["auroc"] for r in B]; ps.append(wilcoxon(x).pvalue); rows.append((a, b, np.median(x)))
    L += ["", "| comparison | median AUROC diff | Holm p |", "|---|---|---|"]
    L += [f"| {a} - {b} | {m:+.3f} | {p:.3g} |" for (a, b, m), p in zip(rows, holm(np.array(ps)))]
    L.append("")

if os.path.exists(f"{D}/gp_probe.json"):
    g = json.load(open(f"{D}/gp_probe.json"))
    L += ["## Gaussian-process planner (DGX Grace core)", "", "| stored readings | s per refit | bytes |", "|---|---|---|"]
    L += [f"| {n} | {v['s_per_refit']:.3g} | {fmt_b(v['bytes'])} |" for n, v in g.items()]
    for name, lim in DEV.items():
        L.append(f"- fills {name} at n = {int(np.sqrt(lim / 16)):,} readings (16 n^2 bytes)")
    L.append("")
if os.path.exists(f"{D}/pi.json"):
    p = json.load(open(f"{D}/pi.json"))
    L += ["## Raspberry Pi 5 (monsterpi), single thread", "", "| method | stored | ms per decision | bytes |", "|---|---|---|---|"]
    L += [f"| kNN, every reading | {n} | {v['ms']:.3g} | {fmt_b(v['bytes'])} |" for n, v in p["knn_full"].items()]
    L += [f"| GP refit | {n} | {1e3 * v['s']:.3g} | {fmt_b(v['bytes'])} |" for n, v in p["gp"].items()]
    L += [f"- kNN keeping every reading fills {k} after {int(v / 224):,} readings" for k, v in DEV.items()]
    L.append("")
if os.path.exists(f"{D}/mace.json"):
    m = json.load(open(f"{D}/mace.json"))
    L += ["## MACE-MP-0 medium on the DGX GPU (2H MoS2 supercells)", ""]
    for dt, v in m.items():
        r = v["runs"]; a = np.array([x["atoms"] for x in r]); b = np.array([x["peak_bytes"] for x in r])
        slope = np.polyfit(a[2:], b[2:], 1)[0]
        L.append(f"- {dt}: {v['parameters']:,} parameters ({fmt_b(v['weight_bytes'])}); peak memory per call "
                 + ", ".join(f"{x['atoms']} atoms {fmt_b(x['peak_bytes'])} ({x['s_call']:.2g} s)" for x in r)
                 + f"; slope {fmt_b(slope)} per atom; Pi 5 RAM holds about {int(8e9 / slope):,} atoms")
    L.append("")
if os.path.exists(f"{D}/llm.json"):
    q = json.load(open(f"{D}/llm.json"))
    L += ["## Language models (weights and per-sequence context memory)", "",
          "| model | weights | KV cache per token | fixed recurrent state | tokens per reading | readings in max context | context memory at max |",
          "|---|---|---|---|---|---|---|"]
    for k, v in q.items():
        L.append(f"| {k} | {fmt_b(v['weight_bytes'])} | {fmt_b(v['kv_bytes_per_token'])} | {fmt_b(v['linear_state_bytes'])} | {v['tokens_per_reading']} | "
                 f"{v['max_context'] // v['tokens_per_reading']:,} | {fmt_b(v['context_bytes'][str(v['max_context'])])} |")
    L.append("")
open(f"{D}/summary.md", "w").write("# S22 memory against data and model size\n\n" + "\n".join(L) + "\n")
print("\n".join(L))
