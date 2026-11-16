#!/usr/bin/env python3
"""S26b: the furnace stop of S18a with a fast planner in the loop. A martingale-posterior network decides in tens of
milliseconds (S26a: proposal plus update on one Grace core), so a planner built on it and placed at the instrument
sends its "close" after D = 0.02 s; D = 0.2 s adds a local-network round trip. Same simulator, seeds, calibration,
estimator (ridge on 56 bands) and stop rule as s18a_furnace.py; only the delay differs. Reference arms are re-run.
Usage: python s26b_fast_planner.py OUT_DIR SEEDS"""
import json, os, sys
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s18a_furnace as F  # noqa: E402

DELAYS = {"local_ridge_bands": 0.0, "planner_D0.02": 0.02, "planner_D0.2": 0.2, "planner_D2": 2.0, "planner_D10": 10.0,
          "planner_D60": 60.0}


def run(seed):
    rng = np.random.default_rng(18000 + seed)
    ks_cal = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_CAL)); ks = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_RUN))
    Xc, Yc = F.calib(np.random.default_rng(18100 + seed), ks_cal)
    r = F.ridge(Xc, Yc)
    out = {}
    for name, d in DELAYS.items():
        hist = []

        def ctrl(x):
            p = r(x); hist.append(p[0] >= F.XT_STOP or p[1] >= F.XB_STOP)
            return len(hist) >= F.N_CONSEC and all(hist[-F.N_CONSEC:])
        prng = np.random.default_rng(18400 + seed)  # same probe noise sequence as S18a
        rs = []
        for k in ks:
            hist.clear(); rs.append(F.run_flake(k, prng, ctrl, d)[0])
        out[name] = {m: float(np.mean([q[m] for q in rs])) for m in rs[0]}
        out[name]["lost"] = float(sum(not q["ok"] for q in rs))
    return dict(seed=seed, arms=out)


if __name__ == "__main__":
    outd, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(outd, exist_ok=True)
    for s in seeds:
        res = run(s); json.dump(res, open(f"{outd}/s26b_s{s}.json", "w"), indent=1)
        print(s, {a: d["lost"] for a, d in res["arms"].items()}, flush=True)
