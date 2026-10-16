#!/usr/bin/env python3
"""Summarise T1-T4 (20 seeds each): means, interquartile ranges, paired Wilcoxon tests (fly vs each arm).
Usage: python analyze.py RESULTS_REFLEX_DIR  (writes summary.md and summary.json there)"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

D = sys.argv[1]; L = []; S = {}


def load(t): return [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{t}/*_s*.json"))]


def p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0


def row(name, v): return f"| {name} | {np.mean(v):.3f} | {np.percentile(v, 25):.3f} to {np.percentile(v, 75):.3f} |"


R = load("t1_janus_faults"); S["t1"] = {}
L += [f"## T1 fault flagging ({len(R)} seeds, mean {np.mean([r['n_effective'] for r in R]):.1f} effective faults per 160 scored runs)", "",
      "| Detector | AUROC mean | IQR | recall at 5 % false alarms | ms per reading | p (AUROC vs fly) |", "|---|---|---|---|---|---|"]
for n in R[0]["det"]:
    a = [r["det"][n]["auroc"] for r in R]; rc = [r["det"][n]["recall_at_5fpr"] for r in R]
    S["t1"][n] = dict(auroc=np.mean(a), recall=np.mean(rc), ms=np.mean([r["det"][n]["ms_per_reading"] for r in R]))
    L.append(f"| {n} | {np.mean(a):.3f} | {np.percentile(a,25):.3f} to {np.percentile(a,75):.3f} | {np.mean(rc):.3f} | {S['t1'][n]['ms']:.1f} | "
             f"{'' if n == 'fly' else f'{p(a, [r['det']['fly']['auroc'] for r in R]):.3g}'} |")
L += ["", "Recall at 5 % false alarms by fault kind (mean over seeds where the kind occurred):", "",
      "| Detector | " + " | ".join(R[0]["det"]["fly"]["recall_by_kind"]) + " |", "|---" * 5 + "|"]
for n in R[0]["det"]:
    L.append(f"| {n} | " + " | ".join(f"{np.mean([r['det'][n]['recall_by_kind'][k] for r in R if r['det'][n]['recall_by_kind'][k] is not None]):.2f}"
                                       for k in R[0]["det"]["fly"]["recall_by_kind"]) + " |")

R = load("t2_epochs"); S["t2"] = {}
L += ["", f"## T2 epochs, MoSSe then WSSe ({len(R)} seeds; RMSE of predicted fidelity over 1400 recipes)", "",
      "| Model | MoSSe after MoSSe | MoSSe after WSSe | forgetting | WSSe after WSSe |", "|---|---|---|---|---|"]
for n in R[0]["models"]:
    g = lambda k: [r["models"][n][k] for r in R]
    f = np.array(g("A_after_B")) - np.array(g("A_after_A"))
    S["t2"][n] = dict(A_A=np.mean(g("A_after_A")), A_B=np.mean(g("A_after_B")), B_B=np.mean(g("B_after_B")), forget=f.mean())
    L.append(f"| {n} | {np.mean(g('A_after_A')):.3f} | {np.mean(g('A_after_B')):.3f} | {f.mean():+.3f} (IQR {np.percentile(f,25):+.3f} to {np.percentile(f,75):+.3f}) | {np.mean(g('B_after_B')):.3f} |")
L.append(f"\nReference: predicting the MoSSe mean everywhere gives RMSE {R[0]['ref']['rmse_mean_predictor_Mo']:.3f}; "
         f"using the WSSe truth map for MoSSe gives {R[0]['ref']['rmse_W_truth_as_Mo']:.3f}. KC code overlap between the two epochs of one recipe: "
         f"{np.mean([r['models']['fly']['code_overlap_A_B'] for r in R]):.2f}.")

R = load("t3_avoid"); S["t3"] = {}
L += ["", f"## T3 failure avoidance ({len(R)} seeds, 30 runs, 29 % of grid recipes destroy the flake)", "",
      "| Arm | failed runs mean | IQR | gap mean | p fails vs gp | p gap vs gp |", "|---|---|---|---|---|---|"]
gpf = [r["arms"]["gp"]["fails"] for r in R]; gpg = [r["arms"]["gp"]["gap"] for r in R]
for a in R[0]["arms"]:
    fl = [r["arms"][a]["fails"] for r in R]; gp_ = [r["arms"][a]["gap"] for r in R]
    S["t3"][a] = dict(fails=np.mean(fl), gap=np.mean(gp_))
    L.append(f"| {a} | {np.mean(fl):.2f} | {np.percentile(fl,25):.0f} to {np.percentile(fl,75):.0f} | {np.mean(gp_):.4f} | "
             f"{'' if a == 'gp' else f'{p(fl, gpf):.3g}'} | {'' if a == 'gp' else f'{p(gp_, gpg):.3g}'} |")

R = load("t4_network"); S["t4"] = {}
L += ["", f"## T4 reflex layer in a three-party network ({len(R)} seeds; furnace 2 faulty from round 4; 36 runs)", "",
      "| Arm | faulty readings shared (furnace offset / Se empty) | repeats | furnace taken out (seeds) | gap mean |", "|---|---|---|---|---|"]
for a in R[0]["arms"]:
    fo = [r["arms"][a]["faulty_shared"] for r in R if r["arms"][a]["fault"] == "furnace_offset"]
    se = [r["arms"][a]["faulty_shared"] for r in R if r["arms"][a]["fault"] == "se_empty"]
    out = sum(r["arms"][a]["out_of_service_round"] is not None for r in R)
    S["t4"][a] = dict(shared_offset=np.mean(fo), shared_se=np.mean(se), out=out, gap=np.mean([r["arms"][a]["gap"] for r in R]))
    L.append(f"| {a} | {np.mean(fo):.1f} / {np.mean(se):.1f} | {np.mean([r['arms'][a]['repeats'] for r in R]):.1f} | {out} of {len(R)} | {S['t4'][a]['gap']:.4f} |")
fo = lambda a: [r["arms"][a]["faulty_shared"] for r in R]
L.append(f"\nPaired Wilcoxon on faulty readings shared: fly vs none p = {p(fo('fly'), fo('none')):.3g}; gp_resid vs none p = {p(fo('gp_resid'), fo('none')):.3g}; fly vs gp_resid p = {p(fo('fly'), fo('gp_resid')):.3g}.")
R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/t5_literature/runs/t5_s*.json"))]; S["t5"] = {}
if R:
    L += ["", f"## T5 literature values ({len(R)} seeds; mean {np.mean([r['n_err'] for r in R]):.1f} injected errors in {R[0]['n']} scored values)", "",
          "| Detector | AUROC mean | IQR | recall at 5 % false alarms | digit | unit | material | p (AUROC vs fly) |", "|---|---|---|---|---|---|---|---|"]
    for n in R[0]["det"]:
        a = [r["det"][n]["auroc"] for r in R]; rk = lambda k: np.mean([r["det"][n]["recall_by_kind"][k] for r in R if r["det"][n]["recall_by_kind"][k] is not None])
        S["t5"][n] = dict(auroc=np.mean(a), recall=np.mean([r["det"][n]["recall_at_5fpr"] for r in R]))
        L.append(f"| {n} | {np.mean(a):.3f} | {np.percentile(a,25):.3f} to {np.percentile(a,75):.3f} | {S['t5'][n]['recall']:.3f} | {rk('digit'):.2f} | {rk('unit'):.2f} | {rk('material'):.2f} | "
                 f"{'' if n == 'fly' else f'{p(a, [r['det']['fly']['auroc'] for r in R]):.3g}'} |")
open(f"{D}/summary.md", "w").write("\n".join(L) + "\n")
json.dump(S, open(f"{D}/summary.json", "w"), indent=1, default=float)
print("\n".join(L))
