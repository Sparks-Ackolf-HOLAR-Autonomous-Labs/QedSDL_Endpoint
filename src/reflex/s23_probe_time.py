#!/usr/bin/env python3
"""S23: when should the in-situ stop rule look? s4_insitu fixes one probe at PROBE_FRAC = 0.25 of each anneal.

This re-runs the job 1 stop test (s4_insitu.run_arm, unchanged) with the probe taken at 0.10, 0.25, 0.50 and
0.75 of the anneal, for the fly stop rule (plain default, the setting tuned in s23_fair_fly t1), logistic
regression and nearest neighbour; the BO planner alone (no probe) once as the reference. A stop at fraction f
uses f of an anneal. Seeds 1-20, 30 recipes per campaign.
S23_ARMS=gp+fly_best runs only the listed arms (any arm of s4_insitu.arms).
Usage: CUDA_VISIBLE_DEVICES="" python s23_probe_time.py OUT_JSON [WORKERS]
"""
import json, os, sys
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FRACS = (0.10, 0.25, 0.50, 0.75)
ARMS = tuple(os.environ.get("S23_ARMS", "gp+fly,gp+logreg,gp+knn").split(","))


def one(args):
    seed, f, arm = args
    import s4_insitu as S4
    S4.PROBE_FRAC = f
    Wd, Min, G = S4.H.wirings(seed)
    mk = S4.arms(Wd, Min)[arm]
    r = S4.run_arm(arm, mk, seed, G)
    return seed, f, arm, r


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(s, f, a) for s in range(1, 21) for f in FRACS for a in ARMS] + ([(s, 0.25, "gp") for s in range(1, 21)] if "S23_ARMS" not in os.environ else [])
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
        res = list(ex.map(one, jobs))
    keys = [k for k in res[0][3] if isinstance(res[0][3][k], (int, float))]
    summ = {}
    for f in FRACS:
        for a in ARMS:
            rs = [r for s, ff, aa, r in res if ff == f and aa == a]
            summ[f"{a}@{f}"] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    rs = [r for s, ff, aa, r in res if aa == "gp"]
    if rs: summ["gp"] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    json.dump(dict(summary=summ, per_run=[dict(seed=s, frac=f, arm=a, **{k: r[k] for k in keys}) for s, f, a, r in res]), open(out, "w"), indent=1)
    for k, v in summ.items():
        print(k, {kk: round(x, 2) for kk, x in v.items()}, flush=True)
