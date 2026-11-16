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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
P4R = f"{ROOT}/results/reflex"
X = f"{ROOT}/results/shift1x"
PUB = dict(s18a=f"{P4R}/s18_control/s18a", s20=f"{P4R}/s20_timecourse", scale="S18a 6x, S20 5x, S20 tuned LSTM 10x, LSTM tuning sweep 5x")
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
    if not all(os.path.exists(f"{d}/s20_cal{nc}_s{r['seed']}_lstmtuned.json") or fallback for r in R):
        return None  # tuned-LSTM block not finished yet
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
        tie = (abs(p["diff"]) < 1e-12) != (abs(n["diff"]) < 1e-12)  # zero difference on exactly one side
        holds = ("yes" if same_dir and sig_p == sig_n else "direction kept, significance changed" if same_dir else
                 ("tie on one side, n.s. on both" if not (sig_p or sig_n) else "**no: tie on one side, significance changed**") if tie
                 else "**no: direction flips**")
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
try:
    new_rank = pick_lstm.rank(new_dir)
except SystemExit as e:  # sweep missing or still running
    new_rank, why = None, str(e)
L += ["## LSTM tuning pick (S20, cal10, seeds 101-105; rule: fewest flakes lost in total, tie -> lower held-out x_t error)", ""]
if new_rank is None:
    L += [f"_1x sweep not complete: {why}_", ""]
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
L += ["## Finding: the published LSTM was tuned at 5x but run at 10x", "",
      "- The published tuning sweep (`lstm_tuning_seeds101-105/`) ran at SHIFT_SCALE 5x: seed 101, setting 1e-2,600,16,"
      " matches it exactly at 5 and at no other scale tried (1-4, 6, 8, 10; `shift1x/probe/tune_probe.py`, `t_x*.log`).",
      "- The published tuned runs (`s20_cal{3,10}_s*_lstmtuned`) ran at 10x (exact seed-1 match at 10 only).",
      "- So the setting picked on the tuning seeds was then used at a different scale from the one it was tuned at."
      " In the 1x reruns, tuning and the tuned runs share one scale (1x), and the pick is preregistered"
      " (`shift1x/PREREG_lstm_pick.md`).", ""]
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
# ---------------- S21 extra networks on the S20 stop (Table S11, Note S3); same pairing and tests as P4 analyze_s21.py
import pick_s21  # noqa: E402
S21P = f"{P4R}/s21_memory_nets"; S21N = os.environ.get("NEW_S21", f"{X}/s21"); S20N = os.environ.get("NEW_S21_S20", NEW["s20"])  # env: self-test only
NL21 = dict(snap_kc="Kenyon-cell code, one spectrum", seq_kc="Kenyon-cell codes now and 14 s earlier (connectome model)",
            snap_mlp="perceptron, one spectrum", lstm_tuned="LSTM (S20, tuned)", net_gru="GRU", net_peep="peephole LSTM",
            net_mog="Mogrifier LSTM", net_tfm="transformer", net_memnet="memory-augmented regressor (GRU keys, memory of calibration windows)")
NOTE_S3 = {("net_gru", "seq_kc"): "two codes vs GRU (p<=0.005)", ("net_peep", "seq_kc"): "two codes vs gated LSTMs (p<=0.005)",
           ("net_mog", "seq_kc"): "two codes vs gated LSTMs (p<=0.005)", ("net_tfm", "seq_kc"): "two codes vs transformer (p=0.09)",
           ("net_tfm", "lstm_tuned"): "transformer vs LSTM (p=0.03)"}


def s21(nets, s20d, best, nc):
    J = lambda f: json.load(open(f))  # noqa: E731
    base = {J(f)["seed"]: J(f) for f in glob.glob(f"{s20d}/s20_cal{nc}_s*[0-9].json")}
    lt = {J(f)["seed"]: J(f) for f in glob.glob(f"{s20d}/s20_cal{nc}_s*_lstmtuned.json")}
    new = {}
    for a, lr in best.items():
        for f in glob.glob(f"{nets}/s20/s20_cal{nc}_s*_s21_lr{lr}.json"):
            j = J(f); new.setdefault(j["seed"], {})[a] = (j["arms"][a], j["final_xt_mae"][a], j["params"].get(a))
    seeds = sorted(x for x in base if x in lt and x in new and len(new[x]) == len(best))
    if not seeds: return None
    get = {a: (lambda a: lambda x: (base[x]["arms"][a], base[x]["final_xt_mae"][a], None))(a) for a in ("snap_kc", "seq_kc", "snap_mlp")}
    get["lstm_tuned"] = lambda x: (lt[x]["arms"]["lstm"], lt[x]["final_xt_mae"]["lstm"], None)
    for a in best: get[a] = (lambda a: lambda x: new[x][a])(a)
    arms, o = {}, {}
    for a, g in get.items():
        v = [g(x) for x in seeds]; o[a] = np.array([q[0]["lost"] for q in v])
        arms[a] = dict(lost=float(o[a].mean()), under=float(np.mean([q[0]["under"] for q in v])), over=float(np.mean([q[0]["over"] for q in v])),
                       err=float(np.mean([q[1] for q in v])), params=v[0][2])
    T = [(a, "seq_kc") for a in list(best) + ["lstm_tuned"]] + [(a, "lstm_tuned") for a in best]
    P = [1.0 if np.all(o[a] == o[b]) else float(wilcoxon(o[a], o[b]).pvalue) for a, b in T]
    t = [dict(arm=a, ref=b, diff=float((o[a] - o[b]).mean()), better=int((o[a] < o[b]).sum()), worse=int((o[a] > o[b]).sum()), p=q, p_holm=float(h))
         for (a, b), q, h in zip(T, P, holm(np.array(P)))]
    return dict(n=len(seeds), arms=arms, tests=t)


r21p = pick_s21.rank(f"{S21P}/s20_tune"); best_p = {a: r21p[a][0]["lr"] for a in pick_s21.NETS}
try:
    r21n = pick_s21.rank(f"{S21N}/s20_tune"); best_n = {a: r21n[a][0]["lr"] for a in pick_s21.NETS}
except SystemExit as e:
    r21n, best_n, why21 = None, None, str(e)
L += ["## S21 extra networks on the S20 stop (Table S11, Note S3)", "",
      "Published: networks at 4x (`s21_memory_nets/run_s20.sh` passed WORKERS=4 as argv[4]; seed-1 exact match at 4, not 1 or 5),"
      " paired with the S20 reference arms from `s20_timecourse/`: Kenyon-cell and perceptron arms at 5x, tuned LSTM at 10x."
      " So the published Table S11 pairs three scales. 1x: every arm at 1x; nets from `shift1x/s21/`, reference arms from `shift1x/s20/`.", "",
      "### Learning rate per network (seeds 101-105, cal10; rule as PREREG, per network: fewest flakes lost in total, tie -> lower x_t error)", ""]
if r21n is None:
    L += [f"_1x S21 sweep not complete: {why21}_", ""]
L += ["| network | lost lr 3e-3 / 1e-2, published (4x) | pick, published | lost lr 3e-3 / 1e-2, 1x | pick, 1x |", "|---|---|---|---|---|"]
for a in pick_s21.NETS:
    lp = {r["lr"]: r["lost"] for r in r21p[a]}; ln = {r["lr"]: r["lost"] for r in r21n[a]} if r21n else {}
    L.append(f"| {NL21[a]} | {lp['3e-3']:.0f} / {lp['1e-2']:.0f} | {best_p[a]} | "
             + (f"{ln['3e-3']:.0f} / {ln['1e-2']:.0f} | {best_n[a]}" + ("" if best_n[a] == best_p[a] else " (CHANGED)") if r21n else " | ") + " |")
L.append("")
out["s21_pick"] = dict(pub=best_p, new=best_n)
for nc in (10, 3):
    pub = s21(S21P, PUB["s20"], best_p, nc); new = s21(S21N, S20N, best_n, nc) if best_n else None
    out[f"s21_cal{nc}"] = dict(pub=pub, new=new)
    if pub is None or new is None:
        L += [f"### {nc} calibration flakes", "", "_missing on one side (pub: %s, 1x: %s)_" % (pub is not None, new is not None), ""]; continue
    L += [f"### {nc} calibration flakes (published {pub['n']} seeds, 1x {new['n']} seeds)", "",
          "| arm | lost of 30, published | lost of 30, 1x | change | under / over, published | under / over, 1x | x_t error, published | x_t error, 1x | trained or stored numbers |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in pub["arms"]:
        p, n = pub["arms"][a], new["arms"][a]
        npar = "" if p["params"] is None else f"{p['params']:,}"
        L.append(f"| {NL21[a]} | {p['lost']:.2f} | {n['lost']:.2f} | {n['lost'] - p['lost']:+.2f} | {p['under']:.2f} / {p['over']:.2f} | "
                 f"{n['under']:.2f} / {n['over']:.2f} | {p['err']:.3f} | {n['err']:.3f} | {npar} |")
    L += ["", "| claim (first arm vs reference, flakes lost) | Note S3 | published: diff, better/worse, Holm p | 1x: diff, better/worse, Holm p | holds at 1x? |",
          "|---|---|---|---|---|"]
    for p, n in zip(pub["tests"], new["tests"]):
        sig_p, sig_n = p["p_holm"] < 0.05, n["p_holm"] < 0.05
        same_dir = np.sign(p["diff"]) == np.sign(n["diff"])
        tie = (abs(p["diff"]) < 1e-12) != (abs(n["diff"]) < 1e-12)
        holds = ("yes" if same_dir and sig_p == sig_n else "direction kept, significance changed" if same_dir else
                 ("tie on one side, n.s. on both" if not (sig_p or sig_n) else "**no: tie on one side, significance changed**") if tie
                 else "**no: direction flips**")
        ns3 = NOTE_S3.get((p["arm"], p["ref"]), "") if nc == 10 else ""
        L.append(f"| {NL21[p['arm']]} vs {NL21[p['ref']]} | {ns3} | {p['diff']:+.2f}, {p['better']}/{p['worse']}, {p['p_holm']:.3g} | "
                 f"{n['diff']:+.2f}, {n['better']}/{n['worse']}, {n['p_holm']:.3g} | {holds} |")
    L.append("")
os.makedirs(f"{X}/compare", exist_ok=True)
json.dump(out, open(f"{X}/compare/compare_shift.json", "w"), indent=1)
open(f"{X}/compare/compare_shift.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
