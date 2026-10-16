"""Summaries for S15b (small networks on the Ar+ dose stop) and S19 (online networks on the familiarity jobs).
Paired by seed against the earlier arms (S15 task b, S1b, S5b); Wilcoxon signed-rank, Holm within each table."""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "reflex")
sid = lambda f: int(re.findall(r"_s(\d+)(?:_|\.json)", os.path.basename(f))[0])  # noqa: E731


def holm(P):
    P = np.array(P); o = np.argsort(P); h = np.empty_like(P); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(P) - r) * P[i])); h[i] = run
    return h


def wx(x, y):
    d = np.asarray(x) - np.asarray(y)
    return 1.0 if np.all(d == 0) else wilcoxon(x, y).pvalue


L = []
# ---- S15b: dose stop
mb = {sid(f): json.load(open(f)) for f in glob.glob(f"{R}/s15b_dose_mlp/s15b_s[0-9]*.json")}
s15 = {sid(f): json.load(open(f)) for f in glob.glob(f"{R}/s15_trained/s15_s[0-9]*.json")}
seeds = sorted(set(mb) & set(s15))
chk = max(abs(mb[s]["task_b"][n][a]["mae"] - s15[s]["task_b"][n][a]["mae"]) for s in seeds for n in ("1", "3") for a in ("ratio", "ridge", "knn"))
L += [f"# S15b: small networks on the Ar+ irradiation-dose stop ({len(seeds)} seeds x 30 test flakes)", "",
      f"Same flakes as S15 task b: largest difference in the recomputed ratio/ridge/knn errors = {chk:.1e}.", ""]
arms = ["ratio", "ridge", "knn", "fly", "shuf", "rand", "dense", "mlp32", "mlp64x2"]
for n in ("1", "3"):
    get = lambda a, k: np.array([(mb[s]["task_b"][n].get(a) or s15[s]["task_b"][n][a])[k] for s in seeds])  # noqa: E731
    L += [f"## {n} calibration flake(s)", "", "| arm | success (within 15 %) | lost (overshoot) | mean abs. error |", "|---|---|---|---|"]
    for a in arms: L.append(f"| {a} | {get(a,'success').mean():.3f} | {get(a,'lost').mean():.3f} | {100*get(a,'mae').mean():.1f} % |")
    T = []
    for m in ("mlp32", "mlp64x2"):
        for ref in ("fly", "ridge", "knn"):
            x, y = get(m, "mae"), get(ref, "mae"); T.append([m, ref, float((y - x).mean()), int((x < y).sum()), wx(x, y)])
    L += ["", "Paired error tests (positive = network has the smaller error), Holm over this table:", "",
          "| network | reference | error difference | seeds network better | p (Holm) |", "|---|---|---|---|---|"]
    for t, h in zip(T, holm([t[-1] for t in T])): L.append(f"| {t[0]} | {t[1]} | {t[2]:+.4f} | {t[3]}/{len(seeds)} | {h:.3g} |")
    L.append("")

# ---- S19 on S1b
NETS = ["ae16", "ae46", "lstm16", "lstmae16"]
s1 = {sid(f): json.load(open(f))["det"] for f in glob.glob(f"{R}/s1b_multi/s1_s[0-9]*.json")}
nn = {}
for f in glob.glob(f"{R}/s19_online_nets/s19_s1b_s*_*.json"):
    nn.setdefault(sid(f), {}).update(json.load(open(f))["det"])
seeds = sorted(s for s in set(s1) & set(nn) if all(k in nn[s] for k in NETS))
get = lambda a, k: np.array([(nn[s].get(a) or s1[s][a])[k] for s in seeds])  # noqa: E731
kinds = list(s1[seeds[0]]["fly"]["recall_by_kind"])
L += [f"# S19: online networks on the familiarity jobs", "", f"## New-phase flag, S1b shared-facility stream ({len(seeds)} seeds, 4,000 readings)", "",
      "| detector | stored numbers | AUROC | " + " | ".join(f"recall {k}" for k in kinds) + " | false alarms after laser drop | ms per reading |",
      "|---" * (5 + len(kinds)) + "|"]
for a in ["fly", "fly_shuf", "knn_budget", "zscore"] + NETS:
    rk = [np.mean([(nn[s].get(a) or s1[s][a])["recall_by_kind"][k] for s in seeds]) for k in kinds]
    L.append(f"| {a} | {int(get(a,'stored_numbers').mean()):,} | {get(a,'auroc').mean():.3f} | " + " | ".join(f"{v:.2f}" for v in rk)
             + f" | {get(a,'false_alarm_after_change').mean():.3f} | {get(a,'ms_per_reading').mean():.2f} |")
T = []
for m in NETS:
    for ref in ("fly", "knn_budget"):
        x, y = get(m, "auroc"), get(ref, "auroc"); T.append([m, ref, float((x - y).mean()), int((x > y).sum()), wx(x, y)])
L += ["", "Paired AUROC tests (network minus reference), Holm over this table:", "",
      "| network | reference | AUROC difference | seeds network higher | p (Holm) |", "|---|---|---|---|---|"]
for t, h in zip(T, holm([t[-1] for t in T])): L.append(f"| {t[0]} | {t[1]} | {t[2]:+.3f} | {t[3]}/{len(seeds)} | {h:.3g} |")

# ---- S19 on S5b
F = {}
for f in glob.glob(f"{R}/s5b_return/s5b_s*_*.json") + glob.glob(f"{R}/s19_online_nets/s5b/s5b_s*_*.json"):
    d = json.load(open(f)); F.setdefault(d["det"], {})[d["seed"]] = d
dets = ["fly_r0", "fly_r1e-4", "fly_r1e-5", "knn_budget", "zscore"] + NETS
seeds = sorted(set.intersection(*[set(F[d]) for d in dets]))
RARE = ["alpha-MoO3", "GaSe", "SnS2"]
L += ["", f"## Memory of rare phases, S5b week-long stream ({len(seeds)} seeds, 10^5 readings)", "",
      "Return ratio = mean score on return / mean score at first sighting: below 1 = remembered, about 1 = treated as new. "
      "Gaps before return: alpha-MoO3 1,000, GaSe 10,000, SnS2 60,000 readings.", "",
      "| detector | stored numbers | AUROC first / last block | false alarms last block | " + " | ".join(f"{k} recall first / return, ratio" for k in RARE) + " | ms per reading |",
      "|---" * (5 + len(RARE)) + "|"]
val = lambda d, f: np.array([f(F[d][s]) for s in seeds])  # noqa: E731
for d in dets:
    cells = []
    for k in RARE:
        rf = val(d, lambda x: x["rare"][k]["first"]["recall"]).mean(); rr = val(d, lambda x: x["rare"][k]["return"]["recall"]).mean()
        cells.append(f"{rf:.2f} / {rr:.2f}, {val(d, lambda x: x['rare'][k]['return_ratio']).mean():.2f}")
    L.append(f"| {d} | {int(val(d, lambda x: x['stored']).mean()):,} | {val(d, lambda x: x['blocks'][0]['auroc']).mean():.3f} / {val(d, lambda x: x['blocks'][-1]['auroc']).mean():.3f} | "
             f"{val(d, lambda x: x['blocks'][-1]['fa']).mean():.3f} | " + " | ".join(cells) + f" | {val(d, lambda x: x['ms']).mean():.2f} |")
T = []
for m in NETS:
    for ref in ("fly_r1e-4", "knn_budget"):
        for k in RARE:
            x, y = val(m, lambda z: z["rare"][k]["contrast_ratio"]), val(ref, lambda z: z["rare"][k]["contrast_ratio"])
            T.append([m, ref, k, float((x - y).mean()), int((x < y).sum()), wx(x, y)])
    x, y = val(m, lambda z: z["blocks"][-1]["auroc"]), val("fly_r1e-4", lambda z: z["blocks"][-1]["auroc"])
    T.append([m, "fly_r1e-4", "AUROC last block", float((x - y).mean()), int((x < y).sum()), wx(x, y)])
L += ["", "Paired tests (network minus reference) on the drift-corrected ratio (negative = network remembers better) "
      f"and on the last-block AUROC (negative = network detects less); {len(seeds)} seeds, Holm over {len(T)} comparisons.", "",
      "| network | reference | measure | difference | seeds network lower | p (Holm) |", "|---|---|---|---|---|---|"]
for t, h in zip(T, holm([t[-1] for t in T])): L.append(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]:+.3f} | {t[4]}/{len(seeds)} | {h:.3g} |")
L += ["", "Drift-corrected memory of the networks (contrast = episode score / score of the routine readings in the 500 readings before; "
      "contrast ratio = return contrast / first contrast; below 1 = remembered):", "",
      "| network | " + " | ".join(f"{k} contrast first / return, ratio" for k in RARE) + " |", "|---" * (1 + len(RARE)) + "|"]
for d in NETS:
    L.append(f"| {d} | " + " | ".join(f"{val(d, lambda x: x['rare'][k]['first']['contrast']).mean():.2f} / {val(d, lambda x: x['rare'][k]['return']['contrast']).mean():.2f}, "
                                     f"{val(d, lambda x: x['rare'][k]['contrast_ratio']).mean():.2f}" for k in RARE) + " |")
os.makedirs(f"{R}/s19_online_nets", exist_ok=True)
txt = "\n".join(L) + "\n"
open(f"{R}/s19_online_nets/summary.md", "w").write(txt); open(f"{R}/s15b_dose_mlp/summary.md", "w").write(txt.split("# S19")[0])
print(txt)
