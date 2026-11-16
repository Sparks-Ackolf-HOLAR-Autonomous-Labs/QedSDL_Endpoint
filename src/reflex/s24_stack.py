#!/usr/bin/env python3
"""S24b: one connectome model against the cheapest engineered stack that covers the same four jobs.

Each job runs in the paper's own harness (imported unchanged), test seeds 1-20 (tuning seeds 101-105 only
for the new arm in job iv). The connectome model is the plain measured-wiring model at its defaults
(5 % active, no gain control), one structure: one PN->KC drive per spectrum, read out per job.
  i    flag new phases and wrong samples (s6_one_module streams and mislabel draw, 3 % of flakes logged
       under the wrong family). Model: context input from the sample log, gain 2 (module_ctx). Stack: per-band
       z-score routed by family (zscore_fam); alternative nearest neighbour routed by family (92 and 184
       spectra in all). Score: AUROC new phase / wrong sample / MoS2-WSe2 stack.
  ii   stop a failing run from one failure (s4_insitu.run_arm, GP-UCB planner, probe at 0.25). Model: avoid
       and approach compartments on the KC code (gp+fly; the paper's gain-control setting gp+fly_best also
       run). Stack: logistic regression (C = 1 default, C = 10 tuned in S23) or nearest neighbour.
       Score: flakes lost of 30.
  iii  end irradiation at a target dose (s11_irradiation.run). Model: lookup of the calibration spectrum with
       the most overlapping KC code (fly_lookup). Stack: ridge on 56 bands (lambda 1e-4); alternative
       nearest calibration spectrum (knn). Score: share on target (+-15 %), mean |error|.
  iv   end point at the furnace (s18a_furnace, local stop, 3 calibration flakes). Stack: ridge on the 56
       bands (lambda 1e-2 default). Model: the same ridge read-out from the model's KC code (new arm: one
       structure for all four jobs), lambda chosen on seeds 101-105 from {1e-4, 1e-3, 1e-2, 1e-1, 1}.
       The paper's nerve-cord read-out is a different wiring (BANC) and is not part of this one structure.
       Score: flakes lost of 30.
Usage: python s24_stack.py i|ii|iii|iv OUT_DIR [WORKERS]
"""
import json, os, sys
JOB = sys.argv[1] if __name__ == "__main__" else "i"
os.environ["S1_MODE"] = "multi" if JOB == "i" else "janus"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import numpy as np, warnings  # noqa: E402
warnings.filterwarnings("ignore")
import s24_wcache  # noqa: E402,F401  (S24_WCACHE: wiring from one small npz)

TUNE, TEST = list(range(101, 106)), list(range(1, 21))
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def job_i(seed):
    from sklearn.metrics import roc_auc_score
    import s1_spectra as S, s2_s3 as H, s7_swarm as S7, s22_data_growth as D
    from fly2 import Fly2
    Wd, Min, G = H.wirings(seed); W = Wd["fly"]; nf = len(S7.FAMS)
    dets = {"module_ctx": Fly2(W, Min, ctx=True, ctx_gain=2.0),
            "zscore_fam": S7.Route([S.Z(G) for _ in range(nf)]),
            "knnfam_n92": S7.Route([S.KNN(True, 92 // nf, np.random.default_rng(3 + j)) for j in range(nf)]),
            "knnfam_n184": S7.Route([S.KNN(True, 184 // nf, np.random.default_rng(3 + j)) for j in range(nf)])}
    st = S7.stream(seed, G); rng = np.random.default_rng(500 + seed)   # s6_one_module mislabel draw
    logged, mis = [], []
    for x, k, f in st:
        m = k is None and rng.random() < 0.03
        logged.append(int((f + rng.integers(1, nf)) % nf) if m else f); mis.append(m)
    mis = np.array(mis); kinds = np.array([k or "" for _, k, _ in st]); new = kinds != ""
    post = np.arange(len(st)) >= S.WARM; ok = post & ~new & ~mis
    au = lambda pos, x: float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(ok.sum())], np.r_[x[pos], x[ok]]))  # noqa: E731
    out = {}
    for n, d in dets.items():
        x = np.array([d.step(xx, lf) for (xx, _, _), lf in zip(st, logged)])
        out[n] = dict(new_phase=au(post & new, x), wrong_sample=au(post & mis, x),
                      stack=au(post & (kinds == "MoS2/WSe2_stack"), x), stored_numbers=int(d.stored()))
    return seed, out


def job_ii(seed):
    import s4_insitu as S4
    Wd, Min, G = S4.H.wirings(seed)

    def logreg(C):
        class Log(S4.KnnStop):
            def stop(self, x):
                if len({f for _, f in self.mem}) < 2: return False
                from sklearn.linear_model import LogisticRegression
                X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
                c = LogisticRegression(C=C, max_iter=500).fit(X * 50, y)
                return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
        return Log
    arms = {"fly": lambda: S4.FlyStop(Wd["fly"], Min, {}),
            "fly_best": lambda: S4.FlyStop(Wd["fly"], Min, S4.H.default_full_cfg()),
            "logreg_C1": logreg(1.0), "logreg_C10": logreg(10.0), "knn": S4.KnnStop}
    return seed, {a: S4.run_arm("gp+" + a, mk, seed, G) for a, mk in arms.items()}


def job_iii(seed):
    import s11_irradiation as M
    r = M.run(seed)
    return seed, {a: r["summary"][a] for a in ("fly_lookup", "ridge", "knn", "dose", "fly")}


def job_iv(seed):
    import s18a_furnace as F, s1_spectra as S
    from fly2 import Fly2
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn)
    fly = Fly2(W, Min)
    rng = np.random.default_rng(18000 + seed)   # t4_one / s18a.run draws
    ks_cal = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_CAL)); ks = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_RUN))
    Xc, Yc = F.calib(np.random.default_rng(18100 + seed), ks_cal)
    feats = {"bands": lambda x: x, "kc": lambda x: fly.code(x).astype(float)}
    Fc = {n: np.array([f(x) for x in Xc]) for n, f in feats.items()}
    out = {}
    for n, f in feats.items():
        for lam in LAMS:
            r = F.ridge(Fc[n], Yc, lam=lam); hist = []

            def ctrl(x, r=r, f=f, hist=hist):
                p = r(f(x)); hist.append(p[0] >= F.XT_STOP or p[1] >= F.XB_STOP)
                return len(hist) >= F.N_CONSEC and all(hist[-F.N_CONSEC:])
            prng = np.random.default_rng(18400 + seed); lost = 0
            for k in ks:
                hist.clear(); lost += not F.run_flake(k, prng, ctrl, 0.0, None)[0]["ok"]
            out[f"{n}_{lam:g}"] = float(lost)
    return seed, out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    out_dir = sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
    fn = dict(i=job_i, ii=job_ii, iii=job_iii, iv=job_iv)[JOB]
    seeds = TUNE + TEST if JOB == "iv" else TEST
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 10) as ex:
        res = {str(s): r for s, r in ex.map(fn, seeds)}
        for s in seeds: print(JOB, s, res[str(s)], flush=True)
    json.dump(dict(job=JOB, seeds=seeds, per_seed=res), open(f"{out_dir}/s24b_job_{JOB}.json", "w"), indent=1)
