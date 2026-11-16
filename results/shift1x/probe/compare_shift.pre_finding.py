#!/usr/bin/env python3
"""Published (SHIFT_SCALE = argv[4], 6x for S18a, 5x for S20) next to the explicit 1x reruns in shift1x/.
Same statistics as P4 analyze_s18.py / analyze_s20.py (copied in this folder; P4 untouched):
two-sided paired Wilcoxon by seed, Holm within each table. Reads only; writes shift1x/compare/.
  python compare_shift.py            -> compare/compare_shift.md and compare/compare_shift.json
Each paired test is one claim ("first arm loses fewer flakes than the reference"). 'holds' says whether
the 1x rerun keeps the published direction and the Holm < 0.05 verdict.
"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # repo root
P4R = f"{ROOT}/results/reflex"
X = f"{ROOT}/results/shift1x"
PUB = dict(s18a=f"{P4R}/s18_control/s18a", s20=f"{P4R}/s20_timecourse", scale="S18a 6x, S20 5x")
NEW = dict(s18a=os.environ.get("NEW_S18A", f"{X}/s18a"), s20=os.environ.get("NEW_S20", f"{X}/s20"), scale="1x")  # env: self-test only
AT1X = json.load(open(f"{X}/probe/at1x.json")) if os.path.exists(f"{X}/probe/at1x.json") else []  # variants published at 1x: not rerun, read from P4


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def s18a(d):
    A = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s18a_s[0-9]*.json"))]
    est = {"planner_D2": "bands", "planner_D10": "bands", "planner_D60": "bands", "local_ridge_bands": "bands",
           "cord_fly": "fly", "cord_shuf": "shuf", "cord_rand": "rand"}
    arms = {a: dict(lost=float(np.mean([r["arms"][a]["lost"] for r in A])),
                    err=float(np.mean([r["xt_mae"][est[a]] for r in A])) if a in est else None) for a in A[0]["arms"]}
    pairs = [("local_ridge_bands", "open_loop"), ("local_ridge_bands", "planner_D2"), ("local_ridge_bands", "planner_D10"),
             ("local_ridge_bands", "planner_D60"), ("cord_fly", "local_ridge_bands"), ("cord_fly", "cord_shuf"), ("cord_fly", "cord_rand")]
    t = []
    for a, b in pairs:
        x, y = np.array([r["arms"][a]["lost"] for r in A]), np.array([r["arms"][b]["lost"] for r in A])
        t.append(dict(arm=a, ref=b, diff=float((x - y).mean()), better=int((x < y).sum()), worse=int((x > y).sum()),
                      p=1.0 if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)))
    for r, q in zip(t, holm(np.array([r["p"] for r in t]))): r["p_holm"] = float(q)
    return dict(n=len(A), arms=arms, tests=t)


def s20(d, nc, fallback=None):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s20_cal{nc}_s*[0-9].json"))]
    if not R: return None
    for r in R:  # tuned LSTM (lr 1e-2, 600 epochs, 16 units), as in analyze_s20.py
        f = f"{d}/s20_cal{nc}_s{r['seed']}_lstmtuned.json"
        if not os.path.exists(f) and fallback: f = f"{fallback}/s20_cal{nc}_s{r['seed']}_lstmtuned.json"
        t = json.load(open(f)); r["arms"]["lstm_tuned"] = t["arms"]["lstm"]; r["final_xt_mae"]["lstm_tuned"] = t["final_xt_mae"]["lstm"]
    ARMS = ["now_ridge", "snap_ridge", "snap_mlp", "snap_kc", "seq_ridge", "lstm", "lstm_tuned", "seq_kc", "trend_rule", "oracle_k"]
    arms = {a: dict(lost=float(np.mean([r["arms"][a]["lost"] for r in R])),
                    err=float(np.mean([r["final_xt_mae"][a] for r in R])) if a in R[0]["final_xt_mae"] else None) for a in ARMS}
    pairs = [("lstm_tuned", b) for b in ["snap_mlp", "snap_ridge", "seq_ridge", "now_ridge", "lstm"]] + \
            [("seq_kc", b) for b in ["lstm_tuned", "snap_kc", "seq_ridge"]] + [("snap_kc", "snap_mlp")]
    t = []
    for a, b in pairs:
        x, y = np.array([r["arms"][a]["lost"] for r in R]), np.array([r["arms"][b]["lost"] for r in R])
        t.append(dict(arm=a, ref=b, diff=float((x - y).mean()), better=int((x < y).sum()), worse=int((x > y).sum()),
                      p=1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)))
    for r, q in zip(t, holm(np.array([r["p"] for r in t]))): r["p_holm"] = float(q)
    return dict(n=len(R), arms=arms, tests=t)


def table(name, pub, new, L):
    if pub is None or new is None:
        L += [f"## {name}", "", "_missing on one side (pub: %s, 1x: %s)_" % (pub is not None, new is not None), ""]; return
    L += [f"## {name} (published {pub['n']} seeds, 1x {new['n']} seeds)", "",
          "| arm | lost of 30, published | lost of 30, 1x | change | x_t error, published | x_t error, 1x |", "|---|---|---|---|---|---|"]
    f = lambda v: "" if v is None else f"{v:.3f}"
    for a in pub["arms"]:
        p, n = pub["arms"][a], new["arms"].get(a, dict(lost=float("nan"), err=None))
        L.append(f"| {a} | {p['lost']:.2f} | {n['lost']:.2f} | {n['lost'] - p['lost']:+.2f} | {f(p['err'])} | {f(n['err'])} |")
    L += ["", "| claim (first arm vs reference, flakes lost) | published: diff, better/worse, Holm p | 1x: diff, better/worse, Holm p | holds at 1x? |",
          "|---|---|---|---|"]
    for p, n in zip(pub["tests"], new["tests"]):
        sig_p, sig_n = p["p_holm"] < 0.05, n["p_holm"] < 0.05
        same_dir = np.sign(p["diff"]) == np.sign(n["diff"]) or (abs(p["diff"]) < 1e-12 and abs(n["diff"]) < 1e-12)
        holds = "yes" if same_dir and sig_p == sig_n else ("direction kept, significance changed" if same_dir else "**no: direction flips**")
        L.append(f"| {p['arm']} vs {p['ref']} | {p['diff']:+.2f}, {p['better']}/{p['worse']}, {p['p_holm']:.3g} | "
                 f"{n['diff']:+.2f}, {n['better']}/{n['worse']}, {n['p_holm']:.3g} | {holds} |")
    L.append("")


out, L = {}, [f"# G1: published ({PUB['scale']}) vs explicit SHIFT_SCALE = 1x reruns", "",
              "Generated by `shift1x/analysis/compare_shift.py`. Statistics as in P4 `analyze_s18.py` / `analyze_s20.py`.", ""]
if AT1X: L += [f"Variants published at 1x already (not rerun; read from P4 on both sides): {', '.join(AT1X)}", ""]
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_lstm  # noqa: E402
pub_rank = pick_lstm.rank(f"{P4R}/s20_timecourse/lstm_tuning_seeds101-105")
new_dir = os.environ.get("NEW_TUNE", f"{X}/s20_tune")
new_rank = pick_lstm.rank(new_dir) if glob.glob(f"{new_dir}/s20_cal10_s*_*.json") else None
L += ["## LSTM tuning pick (S20, cal10, seeds 101-105; rule: fewest flakes lost in total, tie -> lower held-out x_t error)", ""]
if new_rank is None:
    L += [f"_1x sweep not found in {new_dir}_", ""]
else:
    nr = {r["cfg"]: (i + 1, r) for i, r in enumerate(new_rank)}
    L += [f"**Pick: published {pub_rank[0]['cfg']}, 1x {new_rank[0]['cfg']}"
          f" ({'same' if pub_rank[0]['cfg'] == new_rank[0]['cfg'] else 'CHANGED'}).**", "",
          "| setting (lr, epochs, hidden) | rank, published | lost (5 seeds), published | x_t error, published | rank, 1x | lost, 1x | x_t error, 1x |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pub_rank):
        j, q = nr[r["cfg"]]
        L.append(f"| {r['cfg']} | {i + 1} | {r['lost']:.0f} | {r['mae']:.4f} | {j} | {q['lost']:.0f} | {q['mae']:.4f} |")
    L.append("")
out["lstm_pick"] = dict(pub=pub_rank, new=new_rank)
out["s18a"] = dict(pub=s18a(PUB["s18a"]), new=s18a(NEW["s18a"]) if glob.glob(f"{NEW['s18a']}/s18a_s*.json") else None)
table("S18a: selenization stop at the instrument", out["s18a"]["pub"], out["s18a"]["new"], L)
for nc in (3, 10):
    fb = PUB["s20"] if f"s20_cal{nc}_lstmtuned" in AT1X else None
    out[f"s20_cal{nc}"] = dict(pub=s20(PUB["s20"], nc), new=s20(NEW["s20"], nc, fb) if os.path.isdir(NEW["s20"]) else None)
    table(f"S20: time-course stop, {nc} calibration flakes", out[f"s20_cal{nc}"]["pub"], out[f"s20_cal{nc}"]["new"], L)
SG = [json.load(open(f)) for f in sorted(glob.glob(f"{X}/signed/signed_s[0-9]*.json"))]
if SG:  # block C: no published counterpart, 1x only
    L += [f"## Signed cord, 1x only ({len(SG)} seeds; no published run)", "", "| arm | lost of 30 (mean) |", "|---|---|"]
    L += [f"| {a} | {np.mean([r['arms'][a]['lost'] for r in SG]):.2f} |" for a in SG[0]["arms"]]
    L.append(""); out["signed"] = {a: float(np.mean([r["arms"][a]["lost"] for r in SG])) for a in SG[0]["arms"]}
os.makedirs(f"{X}/compare", exist_ok=True)
json.dump(out, open(f"{X}/compare/compare_shift.json", "w"), indent=1)
open(f"{X}/compare/compare_shift.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
