"""S19b: the S19 online networks after a tuning sweep (learning rate, LSTM window; chosen on seeds 101-105,
never on the test seeds) and an autoencoder with a replay buffer, on S1b (20 seeds) and S5b (30 seeds),
paired by seed with the connectome model. S5b connectome runs were repeated with per-episode baselines
so that its return ratios can be drift-corrected the same way as the networks' (contrast ratio)."""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "reflex")
sid = lambda f: int(re.findall(r"_s(\d+)_", os.path.basename(f))[0])  # noqa: E731
ARMS = {"ae16_t_rp0": "autoencoder 16, tuned (lr 3e-3)", "ae46_t_rp0": "autoencoder 46, tuned (lr 1e-2)",
        "lstmae16_t_rp0": "LSTM reconstructing, tuned (lr 1e-2, window 4)",
        "ae16_t_rp59": "autoencoder 16 + replay of 59 readings (equal memory)", "ae16_t_rp500": "autoencoder 16 + replay of 500 readings"}


def holm(P):
    P = np.array(P); o = np.argsort(P); h = np.empty_like(P); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(P) - r) * P[i])); h[i] = run
    return h


def wx(x, y):
    d = np.asarray(x) - np.asarray(y)
    return 1.0 if np.all(d == 0) else wilcoxon(x, y).pvalue


L, out = ["# S19b: tuned online networks and replay", ""], {}
# tuning record
tun = {}
for f in glob.glob(f"{R}/s19_online_nets/tuning_seeds101-105/*.json"):
    det, lr, w = re.match(r"s19_s1b_s\d+_(\w+?)_lr([\de.-]+)_w(\d+)\.json", os.path.basename(f)).groups()
    tun.setdefault((det, lr, w), []).append(json.load(open(f))["det"][det]["auroc"])
L += ["## Tuning sweep (S1b, seeds 101-105; mean AUROC)", "", "| network | learning rate | window | AUROC |", "|---|---|---|---|"]
L += [f"| {k[0]} | {k[1]} | {k[2] if k[0].startswith('lstm') else '-'} | {np.mean(v):.3f} |" for k, v in sorted(tun.items())]
out["tuning"] = {f"{k[0]}|{k[1]}|{k[2]}": float(np.mean(v)) for k, v in tun.items()}
# S1b test seeds
s1 = {int(re.findall(r"_s(\d+)\.json", f)[0]): json.load(open(f))["det"] for f in glob.glob(f"{R}/s1b_multi/s1_s[0-9]*.json")}
nn = {}
for f in glob.glob(f"{R}/s19_online_nets/tuned/s19_s1b_s*.json"):
    j = json.load(open(f)); det = list(j["det"])[0]; tag = os.path.basename(f).split(f"_{det}")[1][:-5]
    nn.setdefault(sid(f), {})[det + tag] = j["det"][det]
old = {}
for f in glob.glob(f"{R}/s19_online_nets/s19_s1b_s*_*.json"):
    old.setdefault(sid(f), {}).update(json.load(open(f))["det"])
seeds = sorted(s for s in nn if len(nn[s]) == len(ARMS) and s in s1)
L += ["", f"## New-phase flag, S1b shared facility ({len(seeds)} test seeds)", "",
      "| detector | stored numbers | AUROC | AUROC before tuning | false alarms after laser drop |", "|---|---|---|---|---|"]
fly = np.array([s1[s]["fly"]["auroc"] for s in seeds]); knn = np.array([s1[s]["knn_budget"]["auroc"] for s in seeds])
L.append(f"| connectome model | 5,177 | {fly.mean():.3f} | | {np.mean([s1[s]['fly']['false_alarm_after_change'] for s in seeds]):.3f} |")
T = []; out["s1b"] = {"fly": float(fly.mean()), "knn_budget": float(knn.mean())}
for a, lab in ARMS.items():
    x = np.array([nn[s][a]["auroc"] for s in seeds]); base = a.split("_t_")[0]
    ob = np.mean([old[s][base]["auroc"] for s in seeds]) if a.endswith("rp0") else None
    L.append(f"| {lab} | {int(nn[seeds[0]][a]['stored_numbers']):,} | {x.mean():.3f} | {'' if ob is None else f'{ob:.3f}'} | "
             f"{np.mean([nn[s][a]['false_alarm_after_change'] for s in seeds]):.3f} |")
    T.append([lab, float((x - fly).mean()), int((x > fly).sum()), wx(x, fly)]); out["s1b"][a] = float(x.mean())
L += ["", "Paired against the connectome model (positive = network higher AUROC), Holm over this table:", "",
      "| detector | AUROC difference | seeds network higher | p (Holm) |", "|---|---|---|---|"]
for t, h in zip(T, holm([t[-1] for t in T])): L.append(f"| {t[0]} | {t[1]:+.3f} | {t[2]}/{len(seeds)} | {h:.3g} |")
out["s1b_tests"] = [dict(det=t[0], diff=t[1], higher=t[2], p_holm=float(h)) for t, h in zip(T, holm([t[-1] for t in T]))]
# S5b
D = os.path.join(R, "s19_online_nets", "s5b_tuned")
if glob.glob(f"{D}/*.json"):
    res = {}
    for f in glob.glob(f"{D}/s5b_s*.json"):
        j = json.load(open(f)); key = os.path.basename(f).split("_", 2)[2][:-5]
        if not key.startswith("fly_"): res.setdefault(key, {})[sid(f)] = j
    for f in glob.glob(f"{R}/s5b_return/s5b_s*_fly_r*.json"):  # connectome model reference, all seeds
        j = json.load(open(f)); key = os.path.basename(f).split("_", 2)[2][:-5]; res.setdefault(key, {})[sid(f)] = j
    order = ["fly_r0", "fly_r1e-5", "fly_r3e-5", "fly_r1e-4", "fly_r1e-3"] + list(ARMS)
    RARE = ["alpha-MoO3", "GaSe", "SnS2"]
    L += ["", "## Rare phases that return, S5b (seeds per row in brackets, 10^5 readings)", "",
          "Return ratio = score on return / score at first sighting; contrast ratio = the same after dividing each "
          "episode's score by the mean score of the routine readings in the 500 readings before it (removes general drift). "
          "Below 1 = remembered.", "",
          "| detector | AUROC first / last block | " + " | ".join(f"{k} return / contrast" for k in RARE) + " | recall at first sighting |",
          "|---" * (3 + len(RARE)) + "|"]
    out["s5b"] = {}
    for k in order:
        if k not in res: continue
        r = list(res[k].values()); first = np.mean([x["blocks"][0]["auroc"] for x in r]); last = np.mean([x["blocks"][-1]["auroc"] for x in r])
        cells, d = [], dict(auroc_first=float(first), auroc_last=float(last), n=len(r))
        for p in RARE:
            rr = np.mean([x["rare"][p]["return_ratio"] for x in r]); cr = np.mean([x["rare"][p]["contrast_ratio"] for x in r])
            cells.append(f"{rr:.2f} / {cr:.2f}"); d[p] = dict(return_ratio=float(rr), contrast_ratio=float(cr))
        rec = np.mean([x["rare"][p]["first"]["recall"] for x in r for p in RARE]); d["recall_first"] = float(rec)
        L.append(f"| {ARMS.get(k, 'connectome model, recovery ' + k[5:])} ({len(r)}) | {first:.3f} / {last:.3f} | " + " | ".join(cells) + f" | {rec:.2f} |")
        out["s5b"][k] = d
json.dump(out, open(f"{R}/s19_online_nets/summary_s19b.json", "w"), indent=1)
open(f"{R}/s19_online_nets/summary_s19b.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
