"""P55 econ re-analysis (sme-econ review MAJOR 1, m1, m2). Existing outputs only, release worktree read only.

L = c_lost*lost + c_false*false_stops + c_time*furnace_runs, c_lost = 1 (units: flakes lost).
Variant 'reanneal': each false stop also costs one extra anneal (c_time * 1).
Endpoints on the never-stop (gp) trajectory, per seed:
  never-stop = gp arm; stop-all at 0.25 = (0, 30 - lost_gp, 7.5); oracle hindsight = (0, 0, 30 - 0.75 lost_gp).
Output: results/s23_econ/econ_reanalysis.{json,md} and V6_econ.{pdf,png}
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

P4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.path.join(P4, "results", "reflex")
OUT = os.path.join(P4, "results", "s23_econ")
SEEDS = list(range(1, 21))
K = ("lost", "false_stops", "furnace_runs")

# ---- per-seed arms, one look at 0.25 (S4, the paper's table)
S4 = {}
for s in SEEDS:
    d = json.load(open(os.path.join(R, "s4_insitu", f"s4_s{s}.json")))
    for a, v in d["arms"].items():
        S4.setdefault(a, {})[s] = tuple(float(v[k]) for k in K)
gp = S4["gp"]
S4["stop-all (gp traj.)"] = {s: (0.0, 30.0 - gp[s][0], 7.5) for s in SEEDS}
ORACLE = {s: (0.0, 0.0, 30.0 - 0.75 * gp[s][0]) for s in SEEDS}
FLY = "gp+fly_best"

# ---- other families
def per_run(fname, key="frac"):
    d = json.load(open(os.path.join(R, "s23_fair_tuning", fname)))
    out = {}
    for r in d.get("per_seed") or d.get("per_run"):
        name = r["arm"] + (f"@{r[key]}" if r.get(key) is not None else "")
        out.setdefault(name, {})[r["seed"]] = tuple(float(r[k]) for k in K)
    return out

PROBE = {**per_run("probe_time.json"), **per_run("probe_time_flybest.json")}
ANY = {**per_run("stop_anytime.json"), **per_run("stop_anytime_flybest.json")}
TUNED = {}
for f in sorted(glob.glob(os.path.join(R, "s23_fair_tuning", "fly_fair_t*.json")) +
                glob.glob(os.path.join(R, "s23_fair_tuning", "nn_fair_j*.json"))):
    d = json.load(open(f))
    tag = os.path.basename(f)[:-5]
    for m, bys in d.get("test", {}).items():
        try:
            TUNED[f"{tag}:{m}"] = {int(s): tuple(float(v[k]) for k in K) for s, v in bys.items()}
        except (KeyError, TypeError, AttributeError, ValueError):
            pass
for f in sorted(glob.glob(os.path.join(R, "s23_fair_tuning", "nn_e*.json"))):
    d = json.load(open(f))
    rows = d.get("per_seed", d)
    try:
        for s, arms in (rows.items() if isinstance(rows, dict) else []):
            for m, v in arms.items():
                TUNED.setdefault(f"{os.path.basename(f)[:-5]}:{m}", {})[int(s)] = tuple(float(v[k]) for k in K)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass


def L(t, cf, ct, re=False):
    lost, fs, fu = t
    return lost + cf * fs + ct * (fu + (fs if re else 0.0))


def mean(arm):
    return tuple(float(np.mean([arm[s][i] for s in SEEDS if s in arm])) for i in range(3))


def Lseeds(arm, cf, ct, re=False):
    return np.array([L(arm[s], cf, ct, re) for s in SEEDS])


def rank(fam, cf, ct, re=False):
    return sorted(((float(Lseeds(a, cf, ct, re).mean()), n) for n, a in fam.items()
                   if all(s in a for s in SEEDS)))


def ptest(fam, a, b, cf, ct, re=False):
    x, y = Lseeds(fam[a], cf, ct, re), Lseeds(fam[b], cf, ct, re)
    dlt = x - y
    p = 1.0 if np.allclose(dlt, 0) else float(wilcoxon(x, y).pvalue)
    return float(dlt.mean()), int((dlt < 0).sum()), int((dlt > 0).sum()), p


# ---- grids
CF = np.round(np.arange(0, 2.0001, 0.02), 4)
CT = np.round(np.arange(0, 1.0001, 0.02), 4)
LEARN = {n: a for n, a in S4.items() if n not in ("stop-all (gp traj.)",)}


def winner_map(fam, re=False):
    names = list(fam)
    W = np.empty((len(CT), len(CF)), dtype=object)
    for i, ct in enumerate(CT):
        for j, cf in enumerate(CF):
            W[i, j] = rank(fam, cf, ct, re)[0][1]
    return W


res = {"definitions": __doc__, "means_s4": {n: mean(a) for n, a in S4.items()}, "oracle_hindsight": mean(ORACLE)}
md = ["# P55 econ re-analysis, G1 (existing outputs, 20 seeds, one look at 0.25 unless stated)", "",
      "L = lost + c_false * false stops + c_time * furnace runs (c_lost = 1). 'reanneal' adds one anneal per false stop.",
      "Endpoints use the never-stop (gp) trajectory: stopping changes what the planner sees next, so stop-all and the",
      "oracle are bounds, not runs. Stop-all also gives the planner no fidelity data.", ""]

md += ["## Frontier (means over 20 seeds)", "", "| arm | lost | false stops | furnace runs |", "|---|---|---|---|"]
for n, a in sorted(S4.items(), key=lambda kv: mean(kv[1])[1]):
    m = mean(a); md.append(f"| {n} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
m = mean(ORACLE); md.append(f"| oracle hindsight (lower bound) | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
pts = {n: mean(a)[:2] for n, a in S4.items()}
pareto = [n for n, (l, f) in pts.items() if not any((l2 <= l and f2 <= f and (l2, f2) != (l, f)) for l2, f2 in pts.values())]
md += ["", f"Pareto set on (lost, false stops): {', '.join(sorted(pareto))}", ""]
res["pareto_lost_false"] = pareto

# ---- crossover vs each competitor (mean-based, closed form along each axis)
def cross(a, b, re=False):
    """Region where b beats FLY: returns c_false threshold at c_time=0 and c_time threshold at c_false=0."""
    fa, fb = mean(S4[a]), mean(S4[b])
    dl, df, du = fb[0] - fa[0], fb[1] - fa[1], fb[2] - fa[2] + ((fb[1] - fa[1]) if re else 0)
    return dict(d_lost=dl, d_false=df, d_furnace=du)

md += ["## Who beats the connectome model (gp+fly_best) and where", "",
       "Differences are other arm minus gp+fly_best (means). Other arm wins where d_lost + c_false d_false + c_time d_furn < 0.", "",
       "| arm | d_lost | d_false | d_furnace | wins at c_time=0 if c_false | wins at c_false=0 if c_time |", "|---|---|---|---|---|---|"]
res["cross"] = {}
for b in S4:
    if b == FLY:
        continue
    c = cross(FLY, b); res["cross"][b] = c
    dl, df, du = c["d_lost"], c["d_false"], c["d_furnace"]
    def thr(lin, const):
        # const + lin * x < 0
        if lin == 0:
            return "always" if const < 0 else "never"
        x = -const / lin
        return (f"< {x:.3f}" if lin > 0 else f"> {x:.3f}") if x > 0 else ("always" if (const < 0) else "never")
    md.append(f"| {b} | {dl:+.2f} | {df:+.2f} | {du:+.2f} | {thr(df, dl)} | {thr(du, dl)} |")

# ---- named ratio points
POINTS = [("c_false 0.096 (Janus share), c_time 0", 0.096, 0.0, False),
          ("c_false 0.096, c_time 0.1, reanneal", 0.096, 0.1, True),
          ("c_false 0.25, c_time 0", 0.25, 0.0, False),
          ("c_false 0.5, c_time 0", 0.5, 0.0, False),
          ("c_false 1, c_time 0", 1.0, 0.0, False),
          ("c_false 0.5, c_time 0.5", 0.5, 0.5, False),
          ("c_false 0, c_time 1", 0.0, 1.0, False),
          ("c_false 1, c_time 1", 1.0, 1.0, False)]
md += ["", "## Ranking at named ratios (S4 arms + stop-all)", ""]
res["points"] = []
for lab, cf, ct, re in POINTS:
    r = rank(S4, cf, ct, re)
    tests = {n: ptest(S4, FLY, n, cf, ct, re) for _, n in r if n != FLY}
    res["points"].append(dict(label=lab, cf=cf, ct=ct, reanneal=re, ranking=r, fly_vs=tests))
    md.append(f"**{lab}**: " + "; ".join(f"{n} {v:.2f}" for v, n in r[:5]) + f" (oracle {float(np.mean([L(ORACLE[s], cf, ct, re) for s in SEEDS])):.2f})")
    bad = [f"{n} (fly {d:+.2f}, {w}/{l} seeds, p={p:.3g})" for n, (d, w, l, p) in tests.items() if d > 0]
    md.append("  fly_best behind: " + (", ".join(bad) if bad else "none"))
    md.append("")

# ---- winner maps
for fam_name, fam in (("with stop-all", S4), ("learned stoppers and never-stop only", LEARN)):
    for re in (False, True):
        W = winner_map(fam, re)
        u, cnt = np.unique(W, return_counts=True)
        share = {str(k): round(float(v) / W.size, 3) for k, v in zip(u, cnt)}
        key = f"winner_share[{fam_name}][reanneal={re}]"
        res[key] = share
        res[key.replace("share", "map")] = W.tolist()
        md.append(f"Winner share of the grid c_false 0-2 x c_time 0-1 ({fam_name}, reanneal={re}): " +
                  ", ".join(f"{k} {v:.0%}" for k, v in sorted(share.items(), key=lambda kv: -kv[1])))
md.append("")

# ---- other look times / any-time / tuned: best per method at named points (post hoc pick flagged)
for fam_name, fam in (("probe time (one look at f)", PROBE), ("any-time rule", ANY), ("tuned variants", TUNED)):
    if not fam:
        continue
    fam = {**fam, "gp (never stop, S4)": S4["gp"], "stop-all (gp traj.)": S4["stop-all (gp traj.)"]}
    md += [f"## {fam_name}", "", "| arm | lost | false | furnace |", "|---|---|---|---|"]
    for n, a in sorted(fam.items()):
        if all(s in a for s in SEEDS):
            m = mean(a); md.append(f"| {n} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} |")
    md.append("")
    for lab, cf, ct, re in POINTS:
        r = rank(fam, cf, ct, re)
        md.append(f"{lab}: " + "; ".join(f"{n} {v:.2f}" for v, n in r[:4]))
    md.append("")
    res[f"family:{fam_name}"] = {n: mean(a) for n, a in fam.items() if all(s in a for s in SEEDS)}

# ---- m2: value of information about the purge rate (slow purge, S20)
s20 = os.path.join(R, "s20_timecourse", "summary.md")
txt = open(s20, encoding="utf-8").read()
VOI = {}
for cal in ("3", "10"):
    sec = txt.split(f"## {cal} calibration flakes")[1].split("Paired tests")[0]
    for line in sec.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 6 and c[0] not in ("arm",) and not c[0].startswith("---"):
            VOI.setdefault(c[0], {})[cal] = float(c[1])
assert VOI["oracle_k"] == {"3": 0.0, "10": 0.0}
md += ["## m2: value of knowing the purge rate (S20 slow purge, 20 seeds x 30 flakes)", "",
       "oracle_k knows the rate and loses 0, so it is an oracle, not a method. VoI = rule's losses - oracle's losses.", "",
       "| rule | VoI, 3 calibration flakes | VoI, 10 calibration flakes |", "|---|---|---|"]
for n, v in sorted(VOI.items(), key=lambda kv: kv[1]["10"]):
    if n != "oracle_k":
        md.append(f"| {n} | {v['3'] - VOI['oracle_k']['3']:.2f} | {v['10'] - VOI['oracle_k']['10']:.2f} |")
res["voi_purge_rate"] = VOI

md += ["", "## m1: optimal stopping", "",
       "The one-look rule stops at the first checkpoint whose score crosses a threshold: a myopic one-step rule.",
       "Under L the problem is a finite-horizon optimal stopping problem (Wald 1947; Chow, Robbins & Siegmund 1971),",
       "solved by backward induction on the posterior that the flake is failing. The any-time and probe-time files",
       "only sample this; none of the arms is the Bayes-optimal stopper, and the oracle row bounds what any can reach."]

os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "econ_reanalysis.json"), "w"), indent=1, default=str)
open(os.path.join(OUT, "econ_reanalysis.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))

# ---- figure (standalone, not in the paper)
sys.path.insert(0, os.path.join(P4, "draft", "regen_figs"))
sys.path.insert(0, os.path.join(P4, "paper_figures", "regen_figs"))  # public release layout
import paper_fig_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
S.OUT = OUT
S.apply_style()
FS = 6.8
G = S.C["conn"]
LAB = {"gp": "never stop", "cbo": "CBO", "gp+fly": "connectome, default", FLY: "connectome (paper)",
       "gp+fly_best_shuf": "shuffled wiring", "gp+fly_best_rand": "random wiring", "gp+knn": "nearest neighbour",
       "gp+logreg": "logistic regression", "stop-all (gp traj.)": "stop every run"}
fig, (a, b) = plt.subplots(1, 2, figsize=(S.IOP_FULL, 2.9),
                           gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.32, left=0.08, right=0.98, bottom=0.17, top=0.91))
for n, (l, f) in pts.items():
    fly = n.startswith("gp+fly")
    a.plot([f], [l], "o", ms=4.6 if n == FLY else 3.6, color=G if fly else S.INK, mfc=G if n == FLY else ("white" if fly else S.INK))
    a.annotate(LAB.get(n, n), (f, l), xytext=(4, 2), textcoords="offset points", fontsize=FS - 0.6,
               color=G if fly else S.INK)
o = mean(ORACLE)
a.plot([o[1]], [o[0]], "*", ms=6, color=S.GREY)
a.annotate("hindsight oracle", (o[1], o[0]), xytext=(4, 2), textcoords="offset points", fontsize=FS - 0.6, color="#555555")
pf = sorted(pts[n] for n in pareto)
a.plot([p[1] for p in sorted(pf, key=lambda p: p[1])], [p[0] for p in sorted(pf, key=lambda p: p[1])],
       color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=0)
a.set_xlabel("false stops of 30", fontsize=FS); a.set_ylabel("flakes lost of 30", fontsize=FS)
a.set_xlim(-1, 24.5); a.set_ylim(-0.5, 8.6)
S.panel(a, "a")

W = np.array(res["winner_map[with stop-all][reanneal=False]"], dtype=object)
names = sorted(set(W.ravel()), key=lambda n: (n != FLY, n))
cols = {FLY: G, "stop-all (gp traj.)": "#bbbbbb", "gp+knn": "#555555", "gp+logreg": "#888888",
        "gp+fly_best_shuf": "#9fc9b0", "gp+fly_best_rand": "#d6e9dd", "gp": "#eeeeee"}
idx = np.vectorize(lambda n: names.index(n))(W)
b.imshow(idx, origin="lower", aspect="auto", extent=[CF[0], CF[-1], CT[0], CT[-1]],
         cmap=ListedColormap([cols.get(n, "#dddddd") for n in names]), vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
b.axvline(0.096, color=S.INK, lw=0.6, ls=(0, (1.2, 1.4)))
b.text(0.11, 0.97, "c_false from the\npaper's outcome shares", fontsize=FS - 0.6, va="top", ha="left", color=S.INK)
b.set_xlabel("cost of a false stop (flakes)", fontsize=FS); b.set_ylabel("cost of one anneal (flakes)", fontsize=FS)
h = [plt.Rectangle((0, 0), 1, 1, color=cols.get(n, "#dddddd")) for n in names]
b.legend(h, [LAB.get(n, n) for n in names], loc="upper right", fontsize=FS - 0.6, frameon=True, framealpha=0.9)
S.panel(b, "b")
S.save(fig, "V6_econ")
print("fig ok")
