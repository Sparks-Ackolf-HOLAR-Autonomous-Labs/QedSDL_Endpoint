"""sme-stats re-analysis for G1 (sme-stats review items 1-8).

Paired effect sizes with 95% CIs (Hodges-Lehmann + Wilcoxon-inversion CI; paired bootstrap of the mean), Wilcoxon p,
Holm within each family, and TOST equivalence (two one-sided Wilcoxon tests; also the 90% HL CI inside +-delta).
Data: release worktree results/reflex (read only). Output: results/stats_ci/ci_reanalysis.{json,md}
"""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon

P4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.path.join(P4, "results", "reflex")
OUT = os.path.join(P4, "results", "stats_ci")
RNG = np.random.default_rng(20261022)
NB = 10000
J = lambda *p: json.load(open(os.path.join(R, *p)))
DELTA = {"auroc": 0.01, "flakes": 0.3, "dose_success": 0.05, "dose_mae": 0.5, "janus_gap": 0.01}


def seeds(pattern):
    out = {}
    for f in glob.glob(os.path.join(R, pattern)):
        m = re.search(r"_s(\d+)\.json$", f)
        if m:
            out[int(m.group(1))] = json.load(open(f))
    return dict(sorted(out.items()))


def signed_rank_null(n):
    # exact null distribution of W+ for n untied nonzero differences
    c = np.zeros(n * (n + 1) // 2 + 1); c[0] = 1
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k].copy()
    return np.cumsum(c) / c.sum()


def hl_ci(d, conf=0.95):
    d = np.asarray(d, float); n = len(d)
    w = np.sort([(d[i] + d[j]) / 2 for i in range(n) for j in range(i, n)])
    cdf = signed_rank_null(n)
    k = int(np.searchsorted(cdf, (1 - conf) / 2, side="right"))  # number of W+ values with P <= alpha/2
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[len(w) - 1 - k])


def wp(d, alt="two-sided"):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative=alt).pvalue)
    except ValueError:
        return 1.0


def boot(d):
    d = np.asarray(d, float)
    m = d[RNG.integers(0, len(d), (NB, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def compare(name, a, b, unit, higher_better, family, note=""):
    """a, b: dict seed -> value. Difference a - b."""
    S = sorted(set(a) & set(b))
    x = np.array([a[s] for s in S], float); y = np.array([b[s] for s in S], float); d = x - y
    hl, lo, hi = hl_ci(d)
    _, lo90, hi90 = hl_ci(d, 0.90)
    blo, bhi = boot(d)
    dl = DELTA.get(unit)
    r = dict(name=name, family=family, unit=unit, n=len(S), mean_a=float(x.mean()), mean_b=float(y.mean()),
             mean_diff=float(d.mean()), hl=hl, hl_ci95=[lo, hi], boot_ci95=[blo, bhi], p=wp(d),
             higher_better=higher_better, note=note)
    if dl is not None:
        p_tost = max(wp(d + dl, "greater"), wp(d - dl, "less"))
        r.update(delta=dl, p_tost=p_tost, hl_ci90=[lo90, hi90], equiv_ci90=bool(-dl < lo90 and hi90 < dl))
    return r


def holm(rows, key="p", out="p_holm"):
    ps = np.array([r[key] for r in rows]); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    for r, v in zip(rows, adj):
        r[out] = float(v)


RES, PREC = [], []
fams = {}


def add(fam, *rows):
    fams.setdefault(fam, []).extend(rows)


# ---- job ii, s4 in situ stop (20 seeds), flakes lost --------------------------------------------------------------
s4 = seeds("s4_insitu/s4_s*.json")
L = lambda arm: {s: v["arms"][arm]["lost"] for s, v in s4.items() if arm in v["arms"]}
for b in ("gp", "cbo", "gp+knn", "gp+logreg", "gp+fly"):
    add("job2_s4_vs_baselines", compare(f"s4 fly_best - {b} (lost)", L("gp+fly_best"), L(b), "flakes", False,
                                        "job2_s4_vs_baselines"))
for b in ("gp+fly_best_shuf", "gp+fly_best_rand"):
    add("wiring_tost", compare(f"s4 fly_best - {b} (lost)", L("gp+fly_best"), L(b), "flakes", False, "wiring_tost"))

# ---- job ii, equal tuning (S23 t1, j1; test seeds 1-20), flakes lost -----------------------------------------------
t1 = J("s23_fair_tuning", "fly_fair_t1.json")["test"]
j1 = J("s23_fair_tuning", "nn_fair_j1.json")
nnk = [k for k in j1["test"] if k != "logreg"][0]
T = lambda dd, f="lost": {int(s): v[f] for s, v in dd.items()}
add("job2_equal_tuning",
    compare("tuned fly - tuned logreg (lost)", T(t1["fly"]), T(t1["logreg"]), "flakes", False, "job2_equal_tuning"),
    compare(f"tuned fly - tuned NN {nnk} (lost)", T(t1["fly"]), T(j1["test"][nnk]), "flakes", False,
            "job2_equal_tuning"),
    compare("tuned NN - tuned logreg, j1 (lost)", T(j1["test"][nnk]), T(j1["test"]["logreg"]), "flakes", False,
            "job2_equal_tuning"))

# ---- precision at one look 0.25: seed-cluster paired bootstrap ----------------------------------------------------
pt = J("s23_fair_tuning", "probe_time.json")["per_run"]
pf = J("s23_fair_tuning", "probe_time_flybest.json")["per_run"]


def per_seed(rows, arm, frac=0.25):
    return {r["seed"]: r for r in rows if r["arm"] == arm and abs(r["frac"] - frac) < 1e-9}


CAND = {"fly_best (paper)": per_seed(pf, "gp+fly_best"), "logreg": per_seed(pt, "gp+logreg"),
        "knn": per_seed(pt, "gp+knn"), "fly plain": per_seed(pt, "gp+fly"),
        "small NN (j1 test)": {int(s): v for s, v in j1["test"][nnk].items()}}


def pooled(rows, S):
    sv = sum(rows[s]["saved"] for s in S); fs = sum(rows[s]["false_stops"] for s in S)
    return sv / max(sv + fs, 1)


A = CAND["fly_best (paper)"]
for k, B in CAND.items():
    if k.startswith("fly_best"):
        continue
    S = np.array(sorted(set(A) & set(B)))
    pa, pb = pooled(A, S), pooled(B, S)
    bs = []
    for _ in range(NB):
        Sb = S[RNG.integers(0, len(S), len(S))]
        bs.append(pooled(A, Sb) - pooled(B, Sb))
    bs = np.array(bs)
    p_boot = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    ra = {s: A[s]["saved"] / max(A[s]["saved"] + A[s]["false_stops"], 1) for s in S}
    rb = {s: B[s]["saved"] / max(B[s]["saved"] + B[s]["false_stops"], 1) for s in S}
    PREC.append(dict(name=f"precision fly_best - {k}", n=len(S), prec_a=pa, prec_b=pb, diff=pa - pb,
                     boot_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_boot=p_boot,
                     per_seed_wilcoxon_p=wp([ra[s] - rb[s] for s in S]),
                     note="stopped-run precision, saved/(saved+false stops), runs that no rule may stop excluded "
                          "(README table includes them, so its values differ)"))
holm(PREC, "p_boot", "p_holm")

# ---- dose (S11; S23 j3), share on target -------------------------------------------------------------------------
s11 = seeds("s11_irradiation/s11_s*.json")
D = lambda arm, f="success": {s: v["summary"][arm][f] for s, v in s11.items() if arm in v["summary"]}
for b in ("knn", "ridge", "dose"):
    if all(b in v["summary"] for v in s11.values()):
        add("job3_dose", compare(f"s11 fly_lookup - {b} (success)", D("fly_lookup"), D(b), "dose_success", True,
                                 "job3_dose"))
for b in ("fly_shuf_lookup", "fly_rand_lookup"):
    add("wiring_tost", compare(f"s11 fly_lookup - {b} (success)", D("fly_lookup"), D(b), "dose_success", True,
                               "wiring_tost"))
j3 = J("s23_fair_tuning", "nn_fair_j3.json")["test"]
nn3 = [k for k in j3 if k not in ("ridge", "fly_lookup")][0]
add("job3_equal_tuning",
    compare("tuned fly_lookup - tuned ridge (success)", T(j3["fly_lookup"], "success"), T(j3["ridge"], "success"),
            "dose_success", True, "job3_equal_tuning"),
    compare(f"tuned fly_lookup - tuned NN {nn3} (success)", T(j3["fly_lookup"], "success"), T(j3[nn3], "success"),
            "dose_success", True, "job3_equal_tuning"))

# ---- facility stream with sample log (S8), AUROC -----------------------------------------------------------------
s8 = seeds("s8_context/s8_s*.json")
C8 = lambda arm: {s: v["det"][arm]["auroc"] for s, v in s8.items() if arm in v["det"]}
for b in ("fam4_quarter", "fam4_full", "zscore_fam", "knn_budget_fam", "fly"):
    add("job1_facility", compare(f"s8 fly_ctx4 - {b}", C8("fly_ctx4"), C8(b), "auroc", True, "job1_facility"))
for b in ("fly_ctx4_shuf", "fly_ctx4_rand"):
    add("wiring_tost", compare(f"s8 fly_ctx4 - {b}", C8("fly_ctx4"), C8(b), "auroc", True, "wiring_tost"))

# ---- Janus flagging (S2), AUROC ----------------------------------------------------------------------------------
for tag in ("janus", "multi"):
    s2 = seeds(f"s2_s3/s2_{tag}_s*.json")
    C2 = lambda arm: {s: v["det"][arm]["auroc"] for s, v in s2.items() if arm in v["det"]}
    for b in ("zscore", "knn_budget"):
        add("flagging_s2", compare(f"s2 {tag} fly - {b}", C2("fly"), C2(b), "auroc", True, "flagging_s2"))
    for b in ("fly_best_shuf", "fly_best_rand"):
        add("wiring_tost", compare(f"s2 {tag} fly_best - {b}", C2("fly_best"), C2(b), "auroc", True, "wiring_tost"))

# ---- cross-animal wiring (S13), AUROC ----------------------------------------------------------------------------
for tag in ("janus", "multi"):
    s13 = seeds(f"s13_cross_animal/{tag}/s13_s*.json")
    if not s13:
        continue
    C13 = lambda arm: {s: v["det"][arm]["auroc"] for s, v in s13.items() if arm in v["det"]}
    for X in ("fafb", "banc", "malecns"):
        add("wiring_tost", compare(f"s13 {tag} {X}_both - {X}_shuf", C13(f"{X}_both"), C13(f"{X}_shuf"), "auroc", True,
                                   "wiring_tost"))

# ---- furnace end point (S18a), flakes lost -----------------------------------------------------------------------
s18 = seeds("s18_control/s18a/s18a_s*.json")
F = lambda arm: {s: v["arms"][arm]["lost"] for s, v in s18.items() if arm in v["arms"]}
for b in ("planner_D2", "planner_D10", "planner_D60", "open_loop"):
    add("job4_furnace", compare(f"s18a local_ridge_bands - {b} (lost)", F("local_ridge_bands"), F(b), "flakes", False,
                                "job4_furnace"))
add("job4_furnace", compare("s18a cord_fly - local_ridge_bands (lost)", F("cord_fly"), F("local_ridge_bands"),
                            "flakes", False, "job4_furnace"))
for b in ("cord_shuf", "cord_rand"):
    add("wiring_tost", compare(f"s18a cord_fly - {b} (lost)", F("cord_fly"), F(b), "flakes", False, "wiring_tost"))

# ---- time course (S20 cal10), flakes lost ------------------------------------------------------------------------
s20 = seeds("s20_timecourse/s20_cal10_s*.json")
TC = lambda arm: {s: v["arms"][arm]["lost"] for s, v in s20.items() if arm in v["arms"]}
arms20 = sorted(set.intersection(*[set(v["arms"]) for v in s20.values()]))
for b in [a for a in arms20 if a in ("lstm", "now_ridge", "trend_rule") or a.startswith("snap_")]:
    add("timecourse_s20", compare(f"s20 seq_kc - {b} (lost)", TC("seq_kc"), TC(b), "flakes", False, "timecourse_s20"))

# ---- memory scaling (S22b), AUROC; Holm family of 8 as in summary.md ----------------------------------------------
s22 = seeds("s22_memory_scaling/b/s22b_s*.json")
C22 = lambda arm: {s: v["det"][arm]["auroc"] for s, v in s22.items() if arm in v["det"]}
for a, b in (("kc_x8", "kc_x1"), ("kc_x0.25", "kc_x1"), ("kc_x1", "fly_measured"), ("knn_eq1", "kc_x1"),
             ("knn_eq8", "kc_x8"), ("knn_full", "knn_eq1"), ("ae1024", "ae16"), ("ae16", "kc_x1")):
    add("scaling_s22b", compare(f"s22b {a} - {b}", C22(a), C22(b), "auroc", True, "scaling_s22b"))

# ---- memory need (S25), fly frozen vs online, AUROC --------------------------------------------------------------
for st in ("s1b", "long", "shift"):
    s25 = seeds(f"s25_memory_need/a/s25a_{st}_s*.json")
    C25 = lambda arm: {s: v["det"][arm]["auroc"] for s, v in s25.items() if arm in v["det"]}
    add("memory_s25", compare(f"s25 {st} fly|frozen - fly|online", C25("fly|frozen"), C25("fly|online"), "auroc", True,
                              "memory_s25"))

# ---- Janus planner (MINOR 7): final gap per seed -----------------------------------------------------------------
jz = json.load(open(os.path.join(R, "..", "janus", "summary.json")))
JF = {a: {i: float(v) for i, v in enumerate(jz["arms"][a]["final"])} for a in jz["arms"] if "final" in jz["arms"][a]}
PLAN = {a: dict(n=len(v), median=float(np.median(list(v.values()))),
                iqr=[float(np.percentile(list(v.values()), 25)), float(np.percentile(list(v.values()), 75))],
                mean=float(np.mean(list(v.values())))) for a, v in JF.items()}
for b in [a for a in JF if a != "random"]:
    add("planner_janus", compare(f"janus {b} - random (final gap)", JF[b], JF["random"], "janus_gap", False,
                                 "planner_janus"))
if "fly" in JF and "gp" in JF:
    add("planner_janus_extra", compare("janus fly - gp (final gap)", JF["fly"], JF["gp"], "janus_gap", False,
                                       "planner_janus_extra"))
if "fly_shuf" in JF:
    add("wiring_tost", compare("janus fly - fly_shuf (final gap)", JF["fly"], JF["fly_shuf"], "janus_gap", False,
                               "wiring_tost"))

for fam, rows in fams.items():
    holm(rows)
    RES.extend(rows)

os.makedirs(OUT, exist_ok=True)
json.dump(dict(comparisons=RES, precision=PREC, planner=PLAN, delta=DELTA, n_boot=NB), open(
    os.path.join(OUT, "ci_reanalysis.json.tmp"), "w"), indent=1)
os.replace(os.path.join(OUT, "ci_reanalysis.json.tmp"), os.path.join(OUT, "ci_reanalysis.json"))
f3 = lambda v: f"{v:+.3f}"
for r in RES:
    eq = ""
    if "delta" in r:
        eq = f" | TOST d={r['delta']} p={r['p_tost']:.3f} 90%CI[{f3(r['hl_ci90'][0])},{f3(r['hl_ci90'][1])}]" + \
             (" EQUIV" if r["p_tost"] < 0.05 else "")
    print(f"[{r['family']}] {r['name']}: n={r['n']} {r['mean_a']:.3f} vs {r['mean_b']:.3f} mean {f3(r['mean_diff'])} "
          f"HL {f3(r['hl'])} [{f3(r['hl_ci95'][0])},{f3(r['hl_ci95'][1])}] boot[{f3(r['boot_ci95'][0])},"
          f"{f3(r['boot_ci95'][1])}] p={r['p']:.4f} holm={r['p_holm']:.4f}{eq}")
for r in PREC:
    print("PREC", json.dumps(r))
print("PLAN", json.dumps(PLAN))
