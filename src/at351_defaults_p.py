"""AT-351: call (ii) p values for the defaults arm (gp+fly, 3.55), same tests as stats_ci.py (read only)."""
import json, os, sys
sys.argv = ["x"]
import numpy as np
from scipy.stats import wilcoxon

P4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
R = os.path.join(P4, "results", "reflex")
J = lambda *p: json.load(open(os.path.join(R, *p)))


def wp(d):
    d = np.asarray(d, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(ps):
    ps = np.array(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1.0, (m - i) * ps[k])); adj[k] = run
    return adj


s4 = {s: J("s4_insitu", f"s4_s{s}.json") for s in range(1, 21)}
L = lambda arm: np.array([s4[s]["arms"][arm]["lost"] for s in range(1, 21)], float)
t1 = J("s23_fair_tuning", "fly_fair_t1.json")["test"]
j1 = J("s23_fair_tuning", "nn_fair_j1.json")
nnk = [k for k in j1["test"] if k != "logreg"][0]
T = lambda dd: np.array([dd[str(s)]["lost"] for s in range(1, 21)], float)

for head in ("gp+fly", "gp+fly_best"):
    a = L(head)
    rows = [(b, a.mean(), L(b).mean(), wp(a - L(b))) for b in ("gp", "cbo", "gp+knn", "gp+logreg",
            "gp+fly_best" if head == "gp+fly" else "gp+fly")]
    adj = holm([r[3] for r in rows])
    print(f"== family of 5, {head} ({a.mean():.2f})")
    for r, h in zip(rows, adj):
        print(f"  vs {r[0]:12s} {r[1]:.2f} vs {r[2]:.2f} raw p={r[3]:.3g} holm5={h:.3g}")
a = L("gp+fly")
print("== other comparisons, defaults arm")
print(f"  s4 gp+fly == t1 fly test? {np.array_equal(a, T(t1['fly']))}  t1 fly mean {T(t1['fly']).mean():.2f}")
print(f"  vs tuned logreg (t1, {T(t1['logreg']).mean():.2f}) raw p={wp(T(t1['fly']) - T(t1['logreg'])):.3g}")
nn = T(j1["test"][nnk])
print(f"  vs small NN {nnk} ({nn.mean():.2f}) raw p={wp(a - nn):.3g}  (default = tuned: {j1['best']})")
print(f"  gp+fly_best vs small NN raw p={wp(L('gp+fly_best') - nn):.3g}")
print("  t1 keys:", list(t1))
