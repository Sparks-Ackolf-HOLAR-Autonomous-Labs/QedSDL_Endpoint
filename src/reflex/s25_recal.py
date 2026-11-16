#!/usr/bin/env python3
"""S25b-recal: after the laser drop (intensity x 1/3), recalibrate from scratch on post-drop data only.

Engineered specialists (E) and the generic connectome model (G; P = per-job tuned connectome model where the
harness has it) discard their old calibration or memory at the drop and refit or relearn from n post-drop
examples only. Seeds 1-20, harnesses as S25.
  dose     new calibration flakes measured at the new intensity (3 flakes x 41 spectra, own generator); refit on
           n = 5, 10, 20, 41 (one flake), 82, 123 of them; test flakes at the new intensity. Arms: E ridge (lambda 1e-4),
           G lookup (5 % code), P lookup (10 %), nearest calibration spectrum. References: before the drop (old
           calibration, old intensity) and no recalibration (old calibration, new intensity).
  furnace  the same with 3 new calibration flakes x 180 readings; n = 10, 20, 45, 90, 180, 360, 540. Arms: E ridge on
           bands (lambda 1e-2), G ridge on the Kenyon-cell code (lambda 0.1).
  stop     drop from probe 15 of 30; at the drop the memory is cleared and the method learns only from the first
           C completed runs after it (C = 3, 6, 10, all 15). Arms: E logistic regression C 10, G model (plain),
           P model (gain control), nearest neighbour. References: no drop; drop with memory kept.
  flag     S1b stream (drop at reading 2,000). recal = memory reset at the drop, then online as deployed; norecal =
           online throughout. The alarm threshold stays at its pre-drop value (95th percentile of the 300 normal
           readings before the drop). Windows of 400 readings starting n = 0, 5, 10, 20, 41, 82, 180, 400, 1,000
           readings after the drop: false alarms (normal readings over threshold), recall (new events over threshold),
           AUROC. Pre-drop reference: readings 1,600-2,000. Arms: E routed z-score, E routed nearest neighbour (184),
           G model told the family (5 %), P the same at 2.5 %.
Usage: python s25_recal.py dose|furnace|stop|flag OUT_DIR [WORKERS]
"""
import json, os, sys, time
JOB = sys.argv[1] if __name__ == "__main__" else "dose"
os.environ["S1_MODE"] = "multi" if JOB == "flag" else "janus"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
import numpy as np, warnings  # noqa: E402
warnings.filterwarnings("ignore")
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401
from s5_long import auroc  # noqa: E402
DROP = 1 / 3   # laser power after the drop, as S25


def subset(n, per, nfl):
    """indices of n labelled spectra: evenly spaced within the first flake, then whole flakes (s25_generic.subset)."""
    if n <= per: return np.unique(np.linspace(0, per - 1, n).round().astype(int))
    return np.arange(min(n // per, nfl) * per)

TEST = list(range(1, 21))


def dose(seed):
    import s11_irradiation as M
    M.MODEL = M.fit_digitised()

    def spec(F, flake, rng, g):   # s11.spectrum with the intensity scaled by g (same draws; g = 1 is s11.spectrum)
        p = M.params(F); s = flake["dpos"]
        y = (M.lor(S.NU, p["cA"] + s, p["wA"]) + p["E_rel"] * M.lor(S.NU, p["cE"] + s, p["wE"])
             + p["L_rel"] * M.lor(S.NU, p["cL"], p["wL"]) + p["O_rel"] * M.lor(S.NU, p["cO"], p["wO"]))
        scale = g * np.exp(rng.uniform(np.log(0.6), np.log(1.6)))
        bg = rng.uniform(0, 0.05) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (y + bg)
        return np.maximum(y + rng.normal(0, 0.01 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
    rng = np.random.default_rng(seed)
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn)
    fG = S.Fly(W, Min); fP = S.Fly(W, Min); fP.k = int(0.10 * W.shape[0])
    newflake = lambda r: dict(e=float(np.exp(r.normal(0, M.E_SD))), dpos=float(r.normal(0, M.POS_SD)))  # noqa: E731
    per = int(10 / M.STEP) + 1
    calF, calY = [], []
    for _ in range(M.N_CAL):
        fl = newflake(rng)
        for k in range(per): Fe = fl["e"] * k * M.STEP; calF.append(Fe); calY.append(spec(Fe, fl, rng, 1.0))
    test = []
    for t in range(M.N_TEST): fl = newflake(rng); test.append((fl, float(rng.uniform(2, 6))))
    xr = np.random.default_rng(93_000 + seed); xF, xY = [], []
    for _ in range(M.N_CAL):   # new calibration flakes at the new intensity
        fl = newflake(xr)
        for k in range(per): Fe = fl["e"] * k * M.STEP; xF.append(Fe); xY.append(spec(Fe, fl, xr, DROP))
    fit_s = {}

    def build(F, Y):
        t = time.perf_counter(); B = np.array([M.norm(S.bands(y, G)) for y in Y]); Xr = np.c_[B, np.ones(len(B))]
        beta = np.linalg.solve(Xr.T @ Xr + 1e-4 * np.eye(Xr.shape[1]), Xr.T @ np.array(F)); fit_s["E_ridge"] = time.perf_counter() - t
        t = time.perf_counter(); CG = np.array([fG.code(S.bands(y, G)) for y in Y]).astype(float); fit_s["G_lookup"] = time.perf_counter() - t
        CP = np.array([fP.code(S.bands(y, G)) for y in Y]).astype(float)
        return dict(F=np.array(F), B=B, beta=beta, CG=CG, CP=CP)

    def evaluate(C, g, tag):
        res = {a: [] for a in ("G_lookup", "P_lookup", "E_ridge", "knn")}
        for t, (fl, Ft) in enumerate(test):
            srng = np.random.default_rng([seed, t]); done = {a: None for a in res}; k = 0
            while k * M.STEP <= M.FMAX and any(v is None for v in done.values()):
                k += 1; Fe = fl["e"] * k * M.STEP; y = spec(Fe, fl, srng, g); b = S.bands(y, G); bn = M.norm(b)
                est = dict(G_lookup=C["F"][np.argmax(C["CG"] @ fG.code(b).astype(float))],
                           P_lookup=C["F"][np.argmax(C["CP"] @ fP.code(b).astype(float))],
                           E_ridge=float(np.r_[bn, 1] @ C["beta"]), knn=C["F"][np.argmin(((C["B"] - bn) ** 2).sum(1))])
                for a in res:
                    if done[a] is None and (est[a] >= Ft or k * M.STEP >= M.FMAX): done[a] = Fe
            for a in res: res[a].append((done[a] - Ft) / Ft)
        return {f"{a}|{tag}": dict(success=float(np.mean([abs(e) <= M.TOL for e in v])), mae=float(np.mean(np.abs(v)))) for a, v in res.items()}
    old = build(calF, calY)
    out = evaluate(old, 1.0, "before"); out.update(evaluate(old, DROP, "norecal"))
    for n in (5, 10, 20, 41, 82, 123):
        i = subset(n, per, M.N_CAL); C = build([xF[j] for j in i], [xY[j] for j in i]); out.update(evaluate(C, DROP, f"recal{n}"))
        out[f"fit_seconds|recal{n}"] = dict(fit_s)
    return seed, out


def furnace(seed):
    import s18a_furnace as F
    import s4_lit as L
    from fly2 import Fly2
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn); fly = Fly2(W, Min)
    orig = F.spectrum

    def raman_lit_gain(st, rng, G_, T_C, g):   # s4_lit.raman_lit with the intensity scaled by g (same draws)
        dT = T_C - 25.0; br = (1 + 3 * st["d_lat"]) * (1 + dT / 600.0); xt = st["x_top"]; s = np.zeros_like(S.NU)
        for ph, fr in (("MoS2", st["f_mos2"]), ("MoSSe", st["f_janus"]), ("MoSe2", st["f_mose2"])):
            ls = []
            for c, a, w in L.LINES[ph]:
                c0 = 280 + 10 * xt if (ph, c) == ("MoSSe", 290) else c
                dw = (L.DW_SE if (ph, c) in L.SE_SIDE else L.DW_S) * dT * L.SHIFT_SCALE
                ls.append((c0 + dw, a, w * br))
            s += fr * S.lines(ls)
        s += 2.0 * st["d_lat"] * S.lines([(227 + L.DW_S * dT * L.SHIFT_SCALE, 0.6, 12 * (1 + dT / 600.0))])
        s += 0.8 * st["theta"] * S.lines([(250, 1.0, 30)])
        s += 0.3 * (max(dT, 0) / 875.0) ** 4 * (S.NU - 100) / 400.0
        scale = g * np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        return S.bands(y, G_)

    def sp_drop(xt, xb, rng):
        st = dict(d_lat=0.0, x_top=xt, f_mos2=1 - xt, f_janus=xt * (1 - xb), f_mose2=xt * xb, theta=0.0)
        b = raman_lit_gain(st, rng, G, F.T_C, DROP); return b / (b.sum() + 1e-12)
    rng = np.random.default_rng(18000 + seed)
    ks_cal = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_CAL)); ks = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_RUN))
    Xc, Yc = F.calib(np.random.default_rng(18100 + seed), ks_cal); per = len(Xc) // F.N_CAL
    kx = F.K0 * np.exp(np.random.default_rng(94_000 + seed).normal(0, F.SIG_K, F.N_CAL))
    F.spectrum = sp_drop; Xx, Yx = F.calib(np.random.default_rng(95_000 + seed), kx); F.spectrum = orig
    feats = {"G_kc": (lambda x: fly.code(x).astype(float), 0.1), "E_bands": (lambda x: x, 1e-2)}
    Fc = {a: np.array([f(x) for x in Xc]) for a, (f, _) in feats.items()}
    Fx = {a: np.array([f(x) for x in Xx]) for a, (f, _) in feats.items()}

    def lost(a, Xf, Y, drop):
        f, lam = feats[a]; t = time.perf_counter(); r = F.ridge(Xf, Y, lam=lam); fit = time.perf_counter() - t; hist = []

        def ctrl(x):
            p = r(f(x)); hist.append(p[0] >= F.XT_STOP or p[1] >= F.XB_STOP)
            return len(hist) >= F.N_CONSEC and all(hist[-F.N_CONSEC:])
        F.spectrum = sp_drop if drop else orig
        try:
            prng = np.random.default_rng(18400 + seed); n = 0
            for k in ks: hist.clear(); n += not F.run_flake(k, prng, ctrl, 0.0, None)[0]["ok"]
        finally: F.spectrum = orig
        return float(n), fit
    out = {}
    for a in feats:
        out[f"{a}|before"] = lost(a, Fc[a], Yc, False)[0]; out[f"{a}|norecal"] = lost(a, Fc[a], Yc, True)[0]
        for n in (10, 20, 45, 90, 180, 360, 540):
            i = subset(n, per, F.N_CAL); v, fit = lost(a, Fx[a][i], Yx[i], True)
            out[f"{a}|recal{n}"] = v; out[f"fit_seconds|{a}|recal{n}"] = fit
    return seed, out


def stop(seed):
    import s4_insitu as S4
    from sklearn.linear_model import LogisticRegression
    Wd, Min, G = S4.H.wirings(seed)

    def mk_raman(on):
        cnt = [0]

        def r_gain(st, rng, G_):   # S4.raman with the intensity scaled by DROP from the 16th probe on (same draws)
            cnt[0] += 1; g = DROP if (on and cnt[0] > 15) else 1.0
            br = 1 + 3 * st["d_lat"]; s = np.zeros_like(S.NU)
            for ph, fr in (("MoS2", st["f_mos2"]), ("MoSSe", st["f_janus"]), ("MoSe2", st["f_mose2"])):
                s += fr * S.lines([(c, a, w * br) for c, a, w in S4.LINES[ph]])
            s += 2.0 * st["d_lat"] * S.lines([(227, 0.6, 12)]) + 0.8 * st["theta"] * S.lines([(250, 1.0, 30)])
            scale = g * np.exp(rng.uniform(np.log(0.2), np.log(5)))
            bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
            y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
            return S.bands(y, G_)
        return r_gain, cnt

    class Recal:   # memory cleared at the drop; then learns from the first C completed runs only
        def __init__(self, mk, cnt, C): self.mk, self.d, self.cnt, self.C, self.done, self.nl = mk, mk(), cnt, C, False, 0
        def stop(self, x):
            if not self.done and self.cnt[0] > 15: self.d = self.mk(); self.done = True
            return self.d.stop(x)
        def learn(self, x, fail):
            if self.done:
                if self.C is not None and self.nl >= self.C: return
                self.nl += 1
            self.d.learn(x, fail)

    class Log10(S4.KnnStop):
        def stop(self, x):
            if len({f for _, f in self.mem}) < 2: return False
            X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
            return LogisticRegression(C=10.0, max_iter=500).fit(X * 50, y).predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
    M = {"G_fly": lambda: S4.FlyStop(Wd["fly"], Min, {}), "P_fly_best": lambda: S4.FlyStop(Wd["fly"], Min, S4.H.default_full_cfg()),
         "E_logreg_C10": Log10, "knn": S4.KnnStop}
    orig = S4.raman; out = {}
    for a, mk in M.items():
        for cond in ("nodrop", "kept", "recal3", "recal6", "recal10", "recal_all"):
            r, cnt = mk_raman(cond != "nodrop"); S4.raman = r
            C = None if cond in ("nodrop", "kept", "recal_all") else int(cond[5:])
            det = (lambda mk=mk, cnt=cnt, C=C: Recal(mk, cnt, C)) if cond.startswith("recal") else mk
            try: out[f"{a}|{cond}"] = S4.run_arm("gp+" + a, det, seed, G)
            finally: S4.raman = orig
    out["gp_alone"] = S4.run_arm("gp", None, seed, G)
    return seed, out


OFFS = [0, 5, 10, 20, 41, 82, 180, 400, 1000]


def flag(seed):
    import s7_swarm as S7, s25_memory as A
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    mk = {"G_fly_ctx": lambda: A.Fly(W, Min, ctx=2.0), "P_fly_ctx25": lambda: A.Fly(W, Min, ctx=2.0, frac=0.025),
          "E_zscore_fam": lambda: A.Route([A.Z(G) for _ in range(4)]),
          "E_knnfam": lambda: A.Route([A.KNN(G, 46, 3 + seed + 17 * j) for j in range(4)])}
    dets = {f"{a}|{c}": m() for a, m in mk.items() for c in ("norecal", "recal")}
    st = S7.stream(seed, G); n = len(st); h = n // 2
    sc = {k: np.zeros(n) for k in dets}
    for i, (x, kind, f) in enumerate(st):
        for k, d in dets.items():
            if i == h and k.endswith("|recal"): d.reset()
            sc[k][i] = d.step(x, f)
    new = np.array([k is not None for _, k, _ in st]); idx = np.arange(n)
    out = {}
    for k, x in sc.items():
        thr = np.quantile(x[~new & (idx >= h - 300) & (idx < h)], 0.95)
        pre = (idx >= h - 400) & (idx < h)
        r = dict(pre=dict(fa=float(np.mean(x[pre & ~new] > thr)), recall=float(np.mean(x[pre & new] > thr)), auroc=float(auroc(x[pre & new], x[pre & ~new]))))
        for o in OFFS:
            w = (idx >= h + o) & (idx < h + o + 400)
            r[f"n{o}"] = dict(fa=float(np.mean(x[w & ~new] > thr)), recall=float(np.mean(x[w & new] > thr)) if (w & new).any() else None,
                              auroc=float(auroc(x[w & new], x[w & ~new])) if (w & new).sum() > 1 else None)
        out[k] = r
    return seed, out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    out_dir = sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
    fn = dict(dose=dose, furnace=furnace, stop=stop, flag=flag)[JOB]
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 6) as ex:
        res = {str(s): r for s, r in ex.map(fn, TEST)}
    json.dump(dict(job=JOB, seeds=TEST, per_seed=res), open(f"{out_dir}/s25r_{JOB}.json", "w"), indent=1)
    print(JOB, "done", flush=True)
