"""Small neural networks on jobs 1-3 under one protocol: settings chosen on seeds 101-105, then tested on seeds 1-20,
inside the paper's own harnesses (src/reflex, imported read-only). Outputs go to results/reflex/s23_fair_tuning/nn_fair_<job>.json.

  j1  in situ stop of selenization (s4_insitu): MLP classifier stop rule, trained on completed runs only.
      Grid: hidden (16,), (32,), (64, 64) x L2 alpha 1e-4, 1e-2, 1, 10. Score: flakes lost (lower is better).
      Check arm: logistic regression (paper 3.90).
  j2  shared Raman station (s8_context streams): the paper's online autoencoder (s19 Online, ae46 = equal memory),
      unrouted and routed by the sample log (one per family, as the paper's routed z-score and nearest neighbour).
      Grid: Adam learning rate 1e-3, 3e-3, 1e-2, 3e-2. Score: AUROC (higher is better).
  j3  Ar+ irradiation dose (s11_irradiation): MLP regressor from 56 bands to fluence, trained on the 3 calibration
      flakes like the ridge arm. Same grid as j1. Score: share of flakes stopped within tolerance.
      Check arms: ridge, fly_lookup (paper 0.95).
Usage: python nn_fair.py j1|j2|j3
"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC); os.chdir(SRC)
import numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor

TUNE, TEST = list(range(101, 106)), list(range(1, 21))
GRID = [(h, a) for h in [(16,), (32,), (64, 64)] for a in [1e-4, 1e-2, 1.0, 10.0]]
LRS = [1e-3, 3e-3, 1e-2, 3e-2]


# ---------------------------------------------------------------- job 1
def j1_one(args):
    seed, cfg = args
    os.environ["S1_MODE"] = "janus"
    import s4_insitu as S4
    from sklearn.neural_network import MLPClassifier

    class MlpStop(S4.KnnStop):
        def stop(self, x):
            if len({f for _, f in self.mem}) < 2: return False
            X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
            c = MLPClassifier(hidden_layer_sizes=cfg[0], alpha=cfg[1], max_iter=2000, random_state=0).fit(X * 50, y)
            return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
    Wd, Min, G = S4.H.wirings(seed)
    if cfg == "logreg":
        return seed, "logreg", S4.run_arm("gp+logreg", S4.LogStop, seed, G)
    return seed, str(cfg), S4.run_arm("gp+mlp", MlpStop, seed, G)


# ---------------------------------------------------------------- job 2
def j2_one(args):
    seed, lr, routed = args
    os.environ["S1_MODE"] = "multi"
    import s1_spectra as S
    import s7_swarm as S7
    import s2_s3 as H
    import s19_online_nets as N
    H.MODE = "multi"
    N.LR = lr

    def arms(Wd, Min, G):
        if routed:
            return {"ae46_fam": S7.Route([N.Online("ae46", G, seed * 10 + j) for j in range(len(S7.FAMS))])}
        return {"ae46": N.Online("ae46", G, seed)}
    r = H.run(seed, arms)
    d = list(r["det"].values())[0]
    return seed, (lr, routed), dict(auroc=d["auroc"], by_kind=d["auroc_by_kind"], stored=d["stored_numbers"])


# ---------------------------------------------------------------- job 3
def j3_one(seed):
    import s11_irradiation as M
    from sklearn.neural_network import MLPRegressor
    S = M.S
    M.MODEL = M.fit_digitised()
    rng = np.random.default_rng(seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    fly = M.FlyStop(W, Min)
    newflake = lambda: dict(e=float(np.exp(rng.normal(0, M.E_SD))), dpos=float(rng.normal(0, M.POS_SD)))  # noqa: E731
    calF, calY = [], []
    for _ in range(M.N_CAL):
        fl = newflake()
        for k in range(int(10 / M.STEP) + 1):
            Fe = fl["e"] * k * M.STEP; calF.append(Fe); calY.append(M.spectrum(Fe, fl, rng))
    calF = np.array(calF); calB = np.array([M.norm(S.bands(y, G)) for y in calY])
    Xr = np.c_[calB, np.ones(len(calB))]
    beta = np.linalg.solve(Xr.T @ Xr + 1e-4 * np.eye(Xr.shape[1]), Xr.T @ calF)
    calC = fly.codes([S.bands(y, G) for y in calY])
    mu, sd = calB.mean(0), calB.std(0) + 1e-12
    mlps = {str(c): MLPRegressor(hidden_layer_sizes=c[0], alpha=c[1], max_iter=3000, random_state=0).fit((calB - mu) / sd, calF) for c in GRID}
    arms = ["ridge", "fly_lookup"] + list(mlps)
    res = {a_: [] for a_ in arms}
    for t in range(M.N_TEST):
        fl = newflake(); Ft = float(rng.uniform(2, 6)); srng = np.random.default_rng([seed, t])
        done = {a_: None for a_ in arms}; k = 0
        while k * M.STEP <= M.FMAX and any(v is None for v in done.values()):
            k += 1; Fn = k * M.STEP; Fe = fl["e"] * Fn
            y = M.spectrum(Fe, fl, srng); bands = S.bands(y, G); bn = M.norm(bands)
            vote = {"ridge": float(np.r_[bn, 1] @ beta) >= Ft}
            cn = fly.f.code(bands).astype(float)
            vote["fly_lookup"] = float(calF[np.argmax(calC @ cn)]) >= Ft
            z = ((bn - mu) / sd)[None]
            for n_, m_ in mlps.items():
                vote[n_] = float(m_.predict(z)[0]) >= Ft
            for a_ in arms:
                if done[a_] is None and (vote[a_] or k * M.STEP >= M.FMAX): done[a_] = Fe
        for a_ in arms:
            res[a_].append((done[a_] - Ft) / Ft)
    summ = {a_: dict(success=float(np.mean([abs(e) <= M.TOL for e in v])), lost=float(np.mean([e > M.TOL for e in v])),
                     under=float(np.mean([e < -M.TOL for e in v])), mae=float(np.mean(np.abs(v)))) for a_, v in res.items()}
    return seed, summ


def mean(xs): return float(np.mean(xs))


if __name__ == "__main__":
    job = sys.argv[1]
    out = {}
    with ProcessPoolExecutor(20) as ex:
        if job == "j1":
            tune = list(ex.map(j1_one, [(s, c) for c in GRID for s in TUNE]))
            agg = {}
            for s, c, r in tune: agg.setdefault(c, []).append(r["lost"])
            best = min(agg, key=lambda c: mean(agg[c])); cfg = eval(best)
            test = list(ex.map(j1_one, [(s, cfg) for s in TEST] + [(s, "logreg") for s in TEST]))
            out = dict(tune={c: mean(v) for c, v in agg.items()}, best=best,
                       test={c: {str(s): r for s, c2, r in test if c2 == c} for c in (best, "logreg")})
            for c in (best, "logreg"):
                rs = list(out["test"][c].values())
                print(c, {k: round(mean([r[k] for r in rs]), 3) for k in ("lost", "saved", "false_stops", "furnace_runs", "ms_per_check")})
        elif job == "j2":
            tune = list(ex.map(j2_one, [(s, lr, rt) for rt in (False, True) for lr in LRS for s in TUNE]))
            agg = {}
            for s, (lr, rt), r in tune: agg.setdefault((lr, rt), []).append(r["auroc"])
            best = {rt: max(LRS, key=lambda lr: mean(agg[(lr, rt)])) for rt in (False, True)}
            test_all = list(ex.map(j2_one, [(s, lr, rt) for rt in (False, True) for lr in LRS for s in TEST]))
            test = [t for t in test_all if t[1][0] == best[t[1][1]]]
            out = dict(tune={str(k): mean(v) for k, v in agg.items()}, best={str(k): v for k, v in best.items()},
                       test={("routed" if rt else "unrouted"): {str(s): r for s, (lr, rt2), r in test if rt2 == rt} for rt in (False, True)},
                       test_every_setting={str(k): mean([r["auroc"] for s, k2, r in test_all if k2 == k]) for k in {k2 for _, k2, _ in test_all}})
            for k_, v in out["test"].items():
                rs = list(v.values()); kinds = rs[0]["by_kind"].keys()
                print(k_, "lr", best[k_ == "routed"], "auroc", round(mean([r["auroc"] for r in rs]), 3),
                      {kk: round(mean([r["by_kind"][kk] for r in rs]), 3) for kk in kinds}, "stored", rs[0]["stored"])
        else:
            res = dict(ex.map(j3_one, TUNE + TEST))
            best = max([str(c) for c in GRID], key=lambda c: mean([res[s][c]["success"] for s in TUNE]))
            out = dict(tune={str(c): mean([res[s][str(c)]["success"] for s in TUNE]) for c in GRID}, best=best,
                       test={a_: {str(s): res[s][a_] for s in TEST} for a_ in ("ridge", "fly_lookup", best)},
                       test_every_setting={a_: mean([res[s][a_]["success"] for s in TEST]) for a_ in res[TEST[0]]})
            for a_ in ("ridge", "fly_lookup", best):
                print(a_, {k: round(mean([res[s][a_][k] for s in TEST]), 3) for k in ("success", "under", "lost", "mae")})
    json.dump(out, open(os.path.join(SRC, "../../results/reflex/s23_fair_tuning", f"nn_fair_{job}.json"), "w"), indent=1, default=str)
