"""The same tuning protocol for the fly model and the classical baselines as nn_fair.py gave the small networks:
settings chosen on seeds 101-105, then tested on seeds 1-20, inside the paper's harnesses (imported read-only).
Outputs: results/reflex/s23_fair_tuning/fly_fair_<job>.json

  t1  in situ stop (s4_insitu): fly stop rule, sparsity {2.5, 5, 10 %} x gain {off, beta 0.5, beta 1.0};
      logistic regression, C in {0.01, 0.1, 1, 10, 100}.                         score: flakes lost (lower)
  t2  shared station (s8 streams): fly with sample-log context, sparsity x context gain {1, 2, 4}.  score: AUROC
  t3  irradiation dose (s11): fly code lookup, sparsity x gain; ridge, lambda in {1e-6, 1e-4, 1e-2, 1, 100}.
                                                                                    score: share on target
  t4  furnace local stop (s18a): ridge readout lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1} for the bands and for the
      fly nerve-cord wiring.                                                       score: flakes lost (lower)
Usage: python fly_fair.py t1|t2|t3|t4
"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC); os.chdir(SRC)
import numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor

TUNE, TEST = list(range(101, 106)), list(range(1, 21))
FRACS = [0.025, 0.05, 0.10]
GAINS = {"off": {}, "g0.5": dict(gain=True, beta=0.5, alpha=1.0), "g1.0": dict(gain=True, beta=1.0, alpha=1.0)}
FLY_GRID = [(f, g) for f in FRACS for g in GAINS]
CS = [0.01, 0.1, 1.0, 10.0, 100.0]


def mean(x): return float(np.mean(x))


def make_fly(W, Min, frac, extra):
    from fly2 import Fly2
    f = Fly2(W, Min, **extra)
    f.k = int(frac * W.shape[0])
    return f


# ---------------------------------------------------------------- t1
def t1_one(args):
    seed, kind, cfg = args
    os.environ["S1_MODE"] = "janus"
    import s4_insitu as S4
    Wd, Min, G = S4.H.wirings(seed)
    if kind == "fly":
        frac, g = cfg

        def mk():
            d = S4.FlyStop(Wd["fly"], Min, {})
            d.f = make_fly(Wd["fly"], Min, frac, GAINS[g])
            return d
        return seed, ("fly", str(cfg)), S4.run_arm("gp+fly", mk, seed, G)
    C = cfg

    class Log(S4.KnnStop):
        def stop(self, x):
            if len({f for _, f in self.mem}) < 2: return False
            from sklearn.linear_model import LogisticRegression
            X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
            c = LogisticRegression(C=C, max_iter=500).fit(X * 50, y)
            return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
    return seed, ("logreg", str(C)), S4.run_arm("gp+logreg", Log, seed, G)


# ---------------------------------------------------------------- t2
def t2_one(seed):
    os.environ["S1_MODE"] = "multi"
    import s2_s3 as H
    H.MODE = "multi"

    def arms(Wd, Min, G):
        return {f"fly_ctx_{fr}_{cg}": make_fly(Wd["fly"], Min, fr, dict(ctx=True, ctx_gain=cg)) for fr in FRACS for cg in (1.0, 2.0, 4.0)}
    r = H.run(seed, arms)
    return seed, {n: dict(auroc=d["auroc"], stack=d["auroc_by_kind"]["MoS2/WSe2_stack"]) for n, d in r["det"].items()}


# ---------------------------------------------------------------- t3
def t3_one(seed):
    import s11_irradiation as M
    S = M.S
    M.MODEL = M.fit_digitised()
    rng = np.random.default_rng(seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    flies = {str(c): make_fly(W, Min, c[0], GAINS[c[1]]) for c in FLY_GRID}
    newflake = lambda: dict(e=float(np.exp(rng.normal(0, M.E_SD))), dpos=float(rng.normal(0, M.POS_SD)))  # noqa: E731
    calF, calY = [], []
    for _ in range(M.N_CAL):
        fl = newflake()
        for k in range(int(10 / M.STEP) + 1):
            Fe = fl["e"] * k * M.STEP; calF.append(Fe); calY.append(M.spectrum(Fe, fl, rng))
    calF = np.array(calF); calB = np.array([M.norm(S.bands(y, G)) for y in calY])
    Xr = np.c_[calB, np.ones(len(calB))]
    betas = {f"ridge_{l}": np.linalg.solve(Xr.T @ Xr + l * np.eye(Xr.shape[1]), Xr.T @ calF) for l in (1e-6, 1e-4, 1e-2, 1.0, 100.0)}
    calC = {n: np.array([f.code(S.bands(y, G)) for y in calY]).astype(float) for n, f in flies.items()}
    arms = list(betas) + [f"fly_{n}" for n in flies]
    res = {a_: [] for a_ in arms}
    for t in range(M.N_TEST):
        fl = newflake(); Ft = float(rng.uniform(2, 6)); srng = np.random.default_rng([seed, t])
        done = {a_: None for a_ in arms}; k = 0
        while k * M.STEP <= M.FMAX and any(v is None for v in done.values()):
            k += 1; Fn = k * M.STEP; Fe = fl["e"] * Fn
            y = M.spectrum(Fe, fl, srng); bands = S.bands(y, G); bn = M.norm(bands)
            vote = {n: float(np.r_[bn, 1] @ b) >= Ft for n, b in betas.items()}
            for n, f in flies.items():
                vote[f"fly_{n}"] = float(calF[np.argmax(calC[n] @ f.code(bands).astype(float))]) >= Ft
            for a_ in arms:
                if done[a_] is None and (vote[a_] or k * M.STEP >= M.FMAX): done[a_] = Fe
        for a_ in arms:
            res[a_].append((done[a_] - Ft) / Ft)
    return seed, {a_: float(np.mean([abs(e) <= M.TOL for e in v])) for a_, v in res.items()}


# ---------------------------------------------------------------- t4
def t4_one(seed):
    import s18a_furnace as F
    F.CACHE = os.environ.get("S23_BANC_CACHE", os.path.expanduser("work/p4_s18/banc_cord.npz"))   # the module default is relative
    rng = np.random.default_rng(18000 + seed)
    ks_cal = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_CAL)); ks = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_RUN))
    Xc, Yc = F.calib(np.random.default_rng(18100 + seed), ks_cal)
    feats = {"bands": lambda x: x}
    c = F.Cord(F.load_cord("fly", np.random.default_rng(18200 + seed)), np.random.default_rng(18300 + seed))
    feats["fly"] = c.features
    Fc = {n: np.array([f(x) for x in Xc]) for n, f in feats.items()}
    out = {}
    for n, f in feats.items():
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            r = F.ridge(Fc[n], Yc, lam=lam)
            hist = []

            def ctrl(x, r=r, f=f, hist=hist):
                p = r(f(x)); hist.append(p[0] >= F.XT_STOP or p[1] >= F.XB_STOP)
                return len(hist) >= F.N_CONSEC and all(hist[-F.N_CONSEC:])
            prng = np.random.default_rng(18400 + seed); lost = 0
            for k in ks:
                hist.clear(); lost += not F.run_flake(k, prng, ctrl, 0.0, None)[0]["ok"]
            out[f"{n}_{lam}"] = float(lost)
    return seed, out


if __name__ == "__main__":
    job = sys.argv[1]
    with ProcessPoolExecutor(20) as ex:
        if job == "t1":
            jobs = [(s, "fly", c) for c in FLY_GRID for s in TUNE] + [(s, "logreg", C) for C in CS for s in TUNE]
            tune = list(ex.map(t1_one, jobs))
            agg = {}
            for s, key, r in tune: agg.setdefault(key, []).append(r["lost"])
            best = {k: min((kk for kk in agg if kk[0] == k), key=lambda kk: mean(agg[kk])) for k in ("fly", "logreg")}
            parse = lambda kk: (eval(kk[1]) if kk[0] == "fly" else float(kk[1]))  # noqa: E731
            test = list(ex.map(t1_one, [(s, k, parse(best[k])) for k in ("fly", "logreg") for s in TEST]))
            out = dict(tune={str(k): mean(v) for k, v in agg.items()}, best={k: str(v) for k, v in best.items()},
                       test={k: {str(s): r for s, kk, r in test if kk[0] == k} for k in ("fly", "logreg")})
            for k in ("fly", "logreg"):
                rs = list(out["test"][k].values())
                print(k, best[k], {m: round(mean([r[m] for r in rs]), 3) for m in ("lost", "saved", "false_stops", "furnace_runs")})
        else:
            fn = {"t2": t2_one, "t3": t3_one, "t4": t4_one}[job]
            res = dict(ex.map(fn, TUNE + TEST))
            arms = list(res[TEST[0]])
            val = (lambda s, a: res[s][a]["auroc"]) if job == "t2" else (lambda s, a: res[s][a])
            better = max if job in ("t2", "t3") else min
            groups = sorted({a.rsplit("_", 1)[0] if job == "t4" else ("ridge" if a.startswith("ridge") else "fly") for a in arms})
            out = dict(tune={a: mean([val(s, a) for s in TUNE]) for a in arms}, test={a: mean([val(s, a) for s in TEST]) for a in arms})
            for g in groups:
                cand = [a for a in arms if (a.rsplit("_", 1)[0] == g if job == "t4" else (("ridge" if a.startswith("ridge") else "fly") == g))]
                b = better(cand, key=lambda a: out["tune"][a])
                extra = {"stack": round(mean([res[s][b]["stack"] for s in TEST]), 3)} if job == "t2" else {}
                print(g, "best", b, "tune", round(out["tune"][b], 3), "TEST", round(out["test"][b], 3), extra)
            out["full"] = {str(s): res[s] for s in res}
    json.dump(out, open(os.path.join(SRC, "../../results/reflex/s23_fair_tuning", f"fly_fair_{job}.json"), "w"), indent=1, default=str)
