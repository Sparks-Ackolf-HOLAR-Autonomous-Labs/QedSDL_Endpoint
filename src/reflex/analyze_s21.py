#!/usr/bin/env python3
"""S21 analysis: networks with memory (recurrent, transformer, memory-augmented) and four replay designs
against the connectome model on the new-phase flag (S1b), rare phases that return (S5b) and the
time-course stop (S20). Paired by seed; Wilcoxon signed-rank, Holm within each table.
Usage: python analyze_s21.py [RESULTS_DIR]   (default results/reflex/s21_memory_nets)"""
import glob, json, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__)); RR = os.path.join(H, "..", "..", "results", "reflex")
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RR, "s21_memory_nets")
NETS = {"ae16": "autoencoder", "lstm": "LSTM", "gru": "GRU", "peep": "peephole LSTM", "mog": "Mogrifier LSTM",
        "tfm": "transformer", "memae": "memory slots learned by gradient (MemAE)",
        "memfast": "fast slot memory written in one shot", "fastw": "fast weights (delta rule)"}
RP = {"none": "no replay", "res": "reservoir", "fifo": "recent (FIFO)", "prio": "surprise-prioritised"}
RARE = ["alpha-MoO3", "GaSe", "SnS2"]


def holm(P):
    P = np.array(P, float); o = np.argsort(P); h = np.empty_like(P); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(P) - r) * P[i])); h[i] = run
    return h


def wx(x, y):
    d = np.asarray(x) - np.asarray(y)
    return 1.0 if np.all(d == 0) else float(wilcoxon(x, y).pvalue)


L, out = ["# S21: networks with memory and four replay designs against the connectome model", ""], {}
# ---------------- S1b
s1 = {int(re.findall(r"_s(\d+)\.json", f)[0]): json.load(open(f))["det"] for f in glob.glob(f"{RR}/s1b_multi/s1_s[0-9]*.json")}
nn = {}
for f in glob.glob(f"{D}/s1b/s19_s1b_s*.json"):
    m = re.match(r"s19_s1b_s(\d+)_(\w+?)_(none|res|fifo|prio)\.json", os.path.basename(f))
    if m: nn.setdefault(int(m[1]), {})[(m[2], m[3])] = json.load(open(f))["det"][m[2]]
keys = [(n, r) for n in NETS for r in RP]
seeds = sorted(s for s in nn if s in s1 and all(k in nn[s] for k in keys))
if seeds:
    fly = np.array([s1[s]["fly"]["auroc"] for s in seeds]); knn = np.array([s1[s]["knn_budget"]["auroc"] for s in seeds])
    ffa = np.mean([s1[s]["fly"]["false_alarm_after_change"] for s in seeds])
    L += [f"## New-phase flag, S1b shared facility ({len(seeds)} seeds)", "",
          f"Connectome model: AUROC {fly.mean():.3f}, false alarms after the laser drop {ffa:.3f}, 5,177 stored numbers. "
          f"Equal-memory nearest neighbour: AUROC {knn.mean():.3f}.", "",
          "| network | " + " | ".join(f"AUROC, {v}" for v in RP.values()) + " | stored numbers (no replay / replay) | ms per reading (no replay) |",
          "|---" * (len(RP) + 3) + "|"]
    T = []; out["s1b"] = {"fly": float(fly.mean()), "knn_budget": float(knn.mean()), "fly_fa": float(ffa)}
    for n, lab in NETS.items():
        cells = []
        for r in RP:
            x = np.array([nn[s][(n, r)]["auroc"] for s in seeds]); cells.append(f"{x.mean():.3f}")
            fa = float(np.mean([nn[s][(n, r)]["false_alarm_after_change"] for s in seeds]))
            out["s1b"][f"{n}|{r}"] = dict(auroc=float(x.mean()), fa=fa, stored=nn[seeds[0]][(n, r)]["stored_numbers"],
                                          ms=float(np.mean([nn[s][(n, r)]["ms_per_reading"] for s in seeds])))
            T.append([f"{lab}, {RP[r]}", float((x - fly).mean()), int((x > fly).sum()), wx(x, fly)])
        st0, st1 = nn[seeds[0]][(n, "none")]["stored_numbers"], nn[seeds[0]][(n, "res")]["stored_numbers"]
        L.append(f"| {lab} | " + " | ".join(cells) + f" | {st0:,} / {st1:,} | {out['s1b'][n + '|none']['ms']:.2f} |")
    L += ["", "False alarms after the laser drop (share of routine readings over the pre-drop threshold):", "",
          "| network | " + " | ".join(RP.values()) + " |", "|---" * (len(RP) + 1) + "|"]
    for n, lab in NETS.items():
        L.append(f"| {lab} | " + " | ".join(f"{out['s1b'][f'{n}|{r}']['fa']:.3f}" for r in RP) + " |")
    hp = holm([t[-1] for t in T])
    L += ["", f"Paired against the connectome model (positive = network higher AUROC), Holm over {len(T)} comparisons:", "",
          "| network, replay | AUROC difference | seeds network higher | p (Holm) |", "|---|---|---|---|"]
    L += [f"| {t[0]} | {t[1]:+.3f} | {t[2]}/{len(seeds)} | {h:.3g} |" for t, h in zip(T, hp)]
    out["s1b_tests"] = [dict(det=t[0], diff=t[1], higher=t[2], p_holm=float(h)) for t, h in zip(T, hp)]
    # replay design effect pooled over networks
    L += ["", "Replay design, mean over the nine networks (AUROC / false alarms after the drop):", "",
          "| replay | AUROC | false alarms |", "|---|---|---|"]
    for r, lab in RP.items():
        L.append(f"| {lab} | {np.mean([out['s1b'][f'{n}|{r}']['auroc'] for n in NETS]):.3f} | "
                 f"{np.mean([out['s1b'][f'{n}|{r}']['fa'] for n in NETS]):.3f} |")
# ---------------- S5b
res = {}
for f in glob.glob(f"{D}/s5b/s5b_s*.json"):
    m = re.match(r"s5b_s(\d+)_(\w+?)_(none|res|fifo|prio)\.json", os.path.basename(f))
    if m: res.setdefault((m[2], m[3]), {})[int(m[1])] = json.load(open(f))
ref = {}
for f in glob.glob(f"{RR}/s5b_return/s5b_s*_fly_r*.json"):
    m = re.match(r"s5b_s(\d+)_fly_r([\de.-]+)\.json", os.path.basename(f))
    if m: ref.setdefault(m[2], {})[int(m[1])] = json.load(open(f))
if res:
    L += ["", "## Rare phases that return, S5b (10^5 readings; seeds per row in brackets)", "",
          "Contrast = episode mean score / mean score of the routine readings in the 500 readings before it; contrast ratio = "
          "return contrast / first contrast (below 1 = remembered, after removing general drift). Recall at first sighting "
          "at the threshold of the first block.", "",
          "| detector | AUROC first / last block | " + " | ".join(f"{k} contrast ratio" for k in RARE) + " | recall at first sighting |",
          "|---" * (3 + len(RARE)) + "|"]
    out["s5b"] = {}

    def row(lab, r):
        r = list(r.values()); first = np.mean([x["blocks"][0]["auroc"] for x in r]); last = np.mean([x["blocks"][-1]["auroc"] for x in r])
        cr = [np.mean([x["rare"][p]["contrast_ratio"] for x in r]) for p in RARE]
        rec = np.mean([x["rare"][p]["first"]["recall"] for x in r for p in RARE])
        L.append(f"| {lab} ({len(r)}) | {first:.3f} / {last:.3f} | " + " | ".join(f"{c:.2f}" for c in cr) + f" | {rec:.2f} |")
        return dict(auroc_first=float(first), auroc_last=float(last), contrast_ratio=dict(zip(RARE, map(float, cr))), recall_first=float(rec), n=len(r))
    for rate in ("0", "1e-5", "1e-4"):
        if rate in ref: out["s5b"][f"fly_r{rate}"] = row(f"connectome model, recovery {rate}", ref[rate])
    for n, lab in NETS.items():
        for r, rl in RP.items():
            if (n, r) in res: out["s5b"][f"{n}|{r}"] = row(f"{lab}, {rl}", res[(n, r)])
    # paired tests against the model at r = 1e-4 (the rate whose threshold held), same seeds
    if "1e-4" in ref:
        R4 = ref["1e-4"]; T = []
        mets = {"alpha-MoO3 contrast ratio": lambda x: x["rare"]["alpha-MoO3"]["contrast_ratio"],
                "recall at first sighting": lambda x: np.mean([x["rare"][p]["first"]["recall"] for p in RARE]),
                "AUROC last block": lambda x: x["blocks"][-1]["auroc"]}
        for n, lab in NETS.items():
            for r, rl in RP.items():
                if (n, r) not in res: continue
                sd = sorted(s for s in res[(n, r)] if s in R4)
                for mk, f in mets.items():
                    a = np.array([f(res[(n, r)][s]) for s in sd]); b = np.array([f(R4[s]) for s in sd])
                    T.append((f"{lab}, {rl}", mk, float((a - b).mean()), int((a < b).sum()), int((a > b).sum()), len(sd), wx(a, b)))
        hp = holm([t[-1] for t in T])
        L += ["", f"Paired against the connectome model at recovery 1e-4 (difference = network minus model), Holm over {len(T)} "
              f"comparisons over {len(R4)} seeds:", "",
              "| network, replay | measure | difference | seeds network lower / higher | p | p (Holm) |", "|---|---|---|---|---|---|"]
        L += [f"| {t[0]} | {t[1]} | {t[2]:+.3f} | {t[3]} / {t[4]} | {t[6]:.3g} | {h:.3g} |" for t, h in zip(T, hp)]
        out["s5b_tests"] = [dict(det=t[0], measure=t[1], diff=t[2], lower=t[3], higher=t[4], n=t[5], p=t[6], p_holm=float(h))
                            for t, h in zip(T, hp)]
        # replay design pooled over the nine networks (mean per seed)
        L += ["", "Replay design, mean over the nine networks (alpha-MoO3 contrast ratio / recall at first sighting / AUROC last block):", "",
              "| replay | alpha-MoO3 contrast ratio | recall at first sighting | AUROC last block |", "|---|---|---|---|"]
        out["s5b_replay"] = {}
        for r, rl in RP.items():
            v = [np.mean([f(x) for n in NETS if (n, r) in res for x in res[(n, r)].values()]) for f in mets.values()]
            out["s5b_replay"][r] = dict(zip(mets, map(float, v)))
            L.append(f"| {rl} | " + " | ".join(f"{x:.2f}" if i < 2 else f"{x:.3f}" for i, x in enumerate(v)) + " |")
# ---------------- S20
S20 = os.path.join(RR, "s20_timecourse"); NEW = ["net_gru", "net_peep", "net_mog", "net_tfm", "net_memnet"]
NL = dict(net_gru="GRU", net_peep="peephole LSTM", net_mog="Mogrifier LSTM", net_tfm="transformer",
          net_memnet="memory-augmented regressor (GRU keys, memory of calibration windows)")
tune = {}
for f in glob.glob(f"{D}/s20_tune/s20_cal10_s*_s21_lr*.json"):
    j = json.load(open(f)); lr = re.findall(r"_lr([\de.-]+)\.json", f)[0]
    for a in NEW: tune.setdefault((a, lr), []).append(j["arms"][a]["lost"])
if tune:
    L += ["", "## Time-course stop, S20 (flakes lost of 30)", "", "Learning rate per arm chosen on seeds 101-105 (10 calibration flakes, "
          "600 epochs, 16 units; mean flakes lost):", "", "| arm | lr 3e-3 | lr 1e-2 |", "|---|---|---|"]
    best = {}
    for a in NEW:
        v = {lr: np.mean(tune[(a, lr)]) for lr in ("3e-3", "1e-2") if (a, lr) in tune}
        L.append(f"| {NL[a]} | " + " | ".join(f"{v.get(lr, float('nan')):.2f}" for lr in ("3e-3", "1e-2")) + " |")
        if v: best[a] = min(v, key=v.get)
    out["s20_lr"] = best; out["s20"] = {}
    for nc in (3, 10):
        base = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{S20}/s20_cal{nc}_s*[0-9].json")}
        lt = {json.load(open(f))["seed"]: json.load(open(f)) for f in glob.glob(f"{S20}/s20_cal{nc}_s*_lstmtuned.json")}
        new = {}
        for a, lr in best.items():
            for f in glob.glob(f"{D}/s20/s20_cal{nc}_s*_s21_lr{lr}.json"):
                j = json.load(open(f)); new.setdefault(j["seed"], {})[a] = (j["arms"][a], j["final_xt_mae"][a], j["params"].get(a))
        seeds = sorted(s for s in base if s in lt and s in new and len(new[s]) == len(best))
        if not seeds: continue
        arms = {"snap_kc": "Kenyon-cell code, one spectrum", "seq_kc": "Kenyon-cell codes now and 14 s earlier (connectome model)",
                "snap_mlp": "perceptron, one spectrum", "lstm_tuned": "LSTM (S20, tuned)"}
        get = {a: (lambda a: lambda s: (base[s]["arms"][a], base[s]["final_xt_mae"][a]))(a) for a in ("snap_kc", "seq_kc", "snap_mlp")}
        get["lstm_tuned"] = lambda s: (lt[s]["arms"]["lstm"], lt[s]["final_xt_mae"]["lstm"])
        for a in best: arms[a] = NL[a]; get[a] = (lambda a: lambda s: new[s][a][:2])(a)
        L += ["", f"### {nc} calibration flakes ({len(seeds)} seeds)", "",
              "| arm | lost of 30 | under-converted | over-selenized | final x_t error (held out) | trained or stored numbers |", "|---|---|---|---|---|---|"]
        o = {}
        for a, lab in arms.items():
            v = [get[a](s) for s in seeds]; lost = np.array([x[0]["lost"] for x in v]); o[a] = lost
            npar = new[seeds[0]][a][2] if a in best else ""
            L.append(f"| {lab} | {lost.mean():.2f} | {np.mean([x[0]['under'] for x in v]):.2f} | {np.mean([x[0]['over'] for x in v]):.2f} | "
                     f"{np.mean([x[1] for x in v]):.3f} | {npar if npar == '' else f'{npar:,}'} |")
            out["s20"][f"cal{nc}|{a}"] = dict(lost=float(lost.mean()), err=float(np.mean([x[1] for x in v])))
        T = [(a, "seq_kc") for a in list(best) + ["lstm_tuned"]] + [(a, "lstm_tuned") for a in best]
        P = [wx(o[a], o[b]) for a, b in T]; hp = holm(P)
        L += ["", "Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:", "",
              "| arm | reference | difference | seeds better / worse | p (Holm) |", "|---|---|---|---|---|"]
        for (a, b), h in zip(T, hp):
            d = o[a] - o[b]; L.append(f"| {arms[a]} | {arms[b]} | {d.mean():+.2f} | {(d < 0).sum()} / {(d > 0).sum()} | {h:.3g} |")
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
open(f"{D}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
