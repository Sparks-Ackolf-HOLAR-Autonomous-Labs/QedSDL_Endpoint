#!/usr/bin/env python3
"""S25b: one generic connectome model against specialist models at an edge station.

Three options per station: G = one generic connectome model at its defaults (one PN->KC drive, plain 5 % code,
read out per job; S24); P = one tuned connectome model per job (S23 settings: flag 2.5 % active with context,
stop with gain control beta 1, dose 10 % active, furnace 5 % with ridge lambda 0.1); E = engineered specialists
(routed z-score, logistic regression, ridge, ridge; S24). Harnesses imported unchanged; test seeds 1-20.

  dose      cost to set up or add the dose job: share on target and mean |error| against the number of labelled
            calibration spectra n = 5, 10, 20, 41 (one flake, evenly spaced steps), 82 (two flakes), 123 (three; the
            paper). Arms: G code lookup (5 %), P code lookup (10 %), E ridge (lambda 1e-4), nearest calibration spectrum.
            Laser drop: test spectra at 1/3 intensity, then recalibration with n_new = 0, 5, 10, 20, 41 spectra of one
            extra calibration flake measured at the new intensity (added to the original 123).
  furnace   the same for the furnace end point: n = 10, 20, 45, 90, 180 (one flake), 360, 540 (three; the paper)
            labelled readings. Arms: G/P ridge on the 5 % KC code (lambda 0.1, chosen in S24), E ridge on bands
            (lambda 1e-2). Laser drop as for dose, n_new = 0, 10, 45, 180.
  family    cost to add a material family to the facility flagger: S1b-type stream (no drop); MoTe2 enters at
            reading 1,000 as a fifth routine family (20 % of later readings, its own family in the log); 8 % of MoTe2
            readings carry a new event (cosmic spike or 1T' MoS2 mixed in at 0.4-0.8). AUROC of new vs normal MoTe2
            readings by how many MoTe2 readings the method had seen (bins 0-10, 10-20, 20-40, ..., 640-1,280), pooled
            over seeds. Arms: G fly with context (5 %, gain 2; the new family gets a new random quarter of cells),
            P same at 2.5 %, E z-score routed (a new fifth z-score), nearest neighbour routed (a new 46-spectrum reservoir),
            pooled fly and pooled z-score.
  stopdrop  laser drop in the failing-run stop: probe spectra at 1/3 intensity from run 15 of 30 on; arms keep their
            memory (drop) or clear it at the drop (drop_reset); flakes lost against no drop.
  flagspec  job i (s6 streams, mislabels) for P: context model at 2.5 % (the generic G at 5 % is S24's module_ctx).
Usage: python s25_generic.py dose|furnace|family|stopdrop|flagspec OUT_DIR [WORKERS]
"""
import json, os, sys
JOB = sys.argv[1] if __name__ == "__main__" else "dose"
os.environ["S1_MODE"] = "multi" if JOB in ("family", "flagspec") else "janus"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
SRC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SRC)
import numpy as np, warnings  # noqa: E402
warnings.filterwarnings("ignore")
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401
from s5_long import auroc  # noqa: E402

TEST = list(range(1, 21)); DROP = 1 / 3


def subset(n, per, nfl):
    """indices of n labelled spectra: evenly spaced within the first flake, then whole flakes."""
    if n <= per: return np.unique(np.linspace(0, per - 1, n).round().astype(int))
    return np.arange(min(n // per, nfl) * per)


# ------------------------------------------------------------------ dose
def dose(seed):
    import s11_irradiation as M
    M.MODEL = M.fit_digitised()
    base_spec = M.spectrum

    def spec_gain(F, flake, rng, g):   # M.spectrum with the intensity scaled by g (same random draws)
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
    calF, calY = [], []
    for _ in range(M.N_CAL):   # s11.run draw order
        fl = newflake(rng)
        for k in range(int(10 / M.STEP) + 1):
            Fe = fl["e"] * k * M.STEP; calF.append(Fe); calY.append(base_spec(Fe, fl, rng))
    per = int(10 / M.STEP) + 1
    test = []
    for t in range(M.N_TEST):
        fl = newflake(rng); Ft = float(rng.uniform(2, 6)); test.append((fl, Ft))
    # extra calibration flake measured after the laser drop (own generator)
    xr = np.random.default_rng(90_000 + seed); xfl = newflake(xr)
    xF = [xfl["e"] * k * M.STEP for k in range(per)]; xY = [spec_gain(F, xfl, xr, DROP) for F in xF]

    def build(F, Y):
        B = np.array([M.norm(S.bands(y, G)) for y in Y]); Xr = np.c_[B, np.ones(len(B))]
        beta = np.linalg.solve(Xr.T @ Xr + 1e-4 * np.eye(Xr.shape[1]), Xr.T @ F)
        return dict(F=np.array(F), B=B, beta=beta, CG=np.array([fG.code(S.bands(y, G)) for y in Y]).astype(float),
                    CP=np.array([fP.code(S.bands(y, G)) for y in Y]).astype(float))

    def evaluate(C, g, tag):
        res = {a: [] for a in ("G_lookup", "P_lookup", "E_ridge", "knn")}
        for t, (fl, Ft) in enumerate(test):
            srng = np.random.default_rng([seed, t]); done = {a: None for a in res}; k = 0
            while k * M.STEP <= M.FMAX and any(v is None for v in done.values()):
                k += 1; Fe = fl["e"] * k * M.STEP
                y = spec_gain(Fe, fl, srng, g); b = S.bands(y, G); bn = M.norm(b)
                est = dict(G_lookup=C["F"][np.argmax(C["CG"] @ fG.code(b).astype(float))],
                           P_lookup=C["F"][np.argmax(C["CP"] @ fP.code(b).astype(float))],
                           E_ridge=float(np.r_[bn, 1] @ C["beta"]), knn=C["F"][np.argmin(((C["B"] - bn) ** 2).sum(1))])
                for a in res:
                    if done[a] is None and (est[a] >= Ft or k * M.STEP >= M.FMAX): done[a] = Fe
            for a in res: res[a].append((done[a] - Ft) / Ft)
        return {f"{a}|{tag}": dict(success=float(np.mean([abs(e) <= M.TOL for e in v])), mae=float(np.mean(np.abs(v))))
                for a, v in res.items()}
    out = {}
    for n in (5, 10, 20, 41, 82, 123):
        i = subset(n, per, M.N_CAL); out.update(evaluate(build([calF[j] for j in i], [calY[j] for j in i]), 1.0, f"n{n}"))
    for nn in (0, 5, 10, 20, 41):
        j = subset(nn, per, 1) if nn else np.array([], int)
        out.update(evaluate(build(calF + [xF[q] for q in j], calY + [xY[q] for q in j]), DROP, f"drop_new{nn}"))
    return seed, out


# ------------------------------------------------------------------ furnace
def furnace(seed):
    import s18a_furnace as F
    import s4_lit as L
    from fly2 import Fly2
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn); fly = Fly2(W, Min)
    orig_spectrum = F.spectrum

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

    def spectrum_gain(g):
        def sp(xt, xb, rng):   # F.spectrum at intensity g
            st = dict(d_lat=0.0, x_top=xt, f_mos2=1 - xt, f_janus=xt * (1 - xb), f_mose2=xt * xb, theta=0.0)
            b = raman_lit_gain(st, rng, G, F.T_C, g); return b / (b.sum() + 1e-12)
        return sp
    rng = np.random.default_rng(18000 + seed)
    ks_cal = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_CAL)); ks = F.K0 * np.exp(rng.normal(0, F.SIG_K, F.N_RUN))
    Xc, Yc = F.calib(np.random.default_rng(18100 + seed), ks_cal); per = len(Xc) // F.N_CAL
    kx = F.K0 * np.exp(np.random.default_rng(91_000 + seed).normal(0, F.SIG_K, 1))
    F.spectrum = spectrum_gain(DROP); Xx, Yx = F.calib(np.random.default_rng(92_000 + seed), kx); F.spectrum = orig_spectrum
    feats = {"GP_kc": (lambda x: fly.code(x).astype(float), 0.1), "E_bands": (lambda x: x, 1e-2)}
    Fc = {a: np.array([f(x) for x in Xc]) for a, (f, _) in feats.items()}
    Fx = {a: np.array([f(x) for x in Xx]) for a, (f, _) in feats.items()}

    def lost(a, Xf, Y, g):
        f, lam = feats[a]; r = F.ridge(Xf, Y, lam=lam); hist = []

        def ctrl(x):
            p = r(f(x)); hist.append(p[0] >= F.XT_STOP or p[1] >= F.XB_STOP)
            return len(hist) >= F.N_CONSEC and all(hist[-F.N_CONSEC:])
        F.spectrum = orig_spectrum if g == 1.0 else spectrum_gain(g)
        try:
            prng = np.random.default_rng(18400 + seed); n = 0
            for k in ks: hist.clear(); n += not F.run_flake(k, prng, ctrl, 0.0, None)[0]["ok"]
        finally: F.spectrum = orig_spectrum
        return float(n)
    out = {}
    for a in feats:
        for n in (10, 20, 45, 90, 180, 360, 540):
            i = subset(n, per, F.N_CAL); out[f"{a}|n{n}"] = lost(a, Fc[a][i], Yc[i], 1.0)
        for nn in (0, 10, 45, 180):
            j = subset(nn, per, 1) if nn else np.array([], int)
            out[f"{a}|drop_new{nn}"] = lost(a, np.r_[Fc[a], Fx[a][j]], np.r_[Yc, Yx[j]], DROP)
    return seed, out


# ------------------------------------------------------------------ new material family
BINS = [0, 10, 20, 40, 80, 160, 320, 640, 1280]


def family(seed):
    import s25_memory as A
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    dets = {"G_fly_ctx": A.Fly(W, Min, ctx=2.0, nf=5), "P_fly_ctx25": A.Fly(W, Min, ctx=2.0, nf=5, frac=0.025),
            "E_zscore_fam": A.Route([A.Z(G) for _ in range(5)]),
            "E_knnfam": A.Route([A.KNN(G, 46, 3 + seed + 17 * j) for j in range(5)]),
            "fly_pooled": A.Fly(W, Min), "zscore_pooled": A.Z(G)}
    rng = np.random.default_rng(60_000 + seed); fams = list(S.FAM)
    mote2 = S.lines([(171, 1.0, 4), (234, 1.2, 4), (289, 0.4, 5)]); n5 = 0
    rows = []   # (n MoTe2 readings seen before, new?, scores)
    for i in range(8_000):
        if i >= 1000 and rng.random() < 0.2:
            f = 4; s = mote2 * rng.uniform(0.8, 1.2); new = rng.random() < 0.08
            if new:
                if rng.random() < 0.5: s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
                else: fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS["1T'_MoS2"]
        else:
            f = int(rng.integers(4)); comp = S.FAM[fams[f]]
            w = rng.dirichlet([0.6] * len(comp)); s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); new = False
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        x = S.bands(y, G)
        sc = {a: d.step(x, f) for a, d in dets.items()}
        if f == 4: rows.append((n5, bool(new), sc)); n5 += 1
    return seed, rows


# ------------------------------------------------------------------ laser drop in the failing-run stop
def stopdrop(seed):
    import s4_insitu as S4
    Wd, Min, G = S4.H.wirings(seed); orig = S4.raman

    def mk_raman(on):
        cnt = [0]

        def r_gain(st, rng, G_):   # S4.raman with the intensity scaled by DROP from the 15th probe on (same draws)
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

    class ResetAt:
        def __init__(self, mk, cnt): self.mk, self.d, self.cnt, self.done = mk, mk(), cnt, False
        def stop(self, x):
            if not self.done and self.cnt[0] > 15: self.d = self.mk(); self.done = True
            return self.d.stop(x)
        def learn(self, x, fail): self.d.learn(x, fail)
    from sklearn.linear_model import LogisticRegression

    class Log10(S4.KnnStop):
        def stop(self, x):
            if len({f for _, f in self.mem}) < 2: return False
            X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
            return LogisticRegression(C=10.0, max_iter=500).fit(X * 50, y).predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
    M = {"G_fly": lambda: S4.FlyStop(Wd["fly"], Min, {}), "P_fly_best": lambda: S4.FlyStop(Wd["fly"], Min, S4.H.default_full_cfg()),
         "E_logreg_C10": Log10, "knn": S4.KnnStop}
    out = {}
    for a, mk in M.items():
        for cond in ("nodrop", "drop", "drop_reset"):
            r, cnt = mk_raman(cond != "nodrop"); S4.raman = r
            try:
                det_mk = (lambda: ResetAt(mk, cnt)) if cond == "drop_reset" else mk
                out[f"{a}|{cond}"] = S4.run_arm("gp+" + a, det_mk, seed, G)
            finally: S4.raman = orig
    return seed, out


# ------------------------------------------------------------------ job i for the specialist P
def flagspec(seed):
    from sklearn.metrics import roc_auc_score
    import s2_s3 as H, s7_swarm as S7
    from fly2 import Fly2
    Wd, Min, G = H.wirings(seed); W = Wd["fly"]; nf = len(S7.FAMS)
    d = Fly2(W, Min, ctx=True, ctx_gain=2.0); d.k = int(0.025 * W.shape[0])
    st = S7.stream(seed, G); rng = np.random.default_rng(500 + seed)
    logged, mis = [], []
    for x, k, f in st:
        m = k is None and rng.random() < 0.03
        logged.append(int((f + rng.integers(1, nf)) % nf) if m else f); mis.append(m)
    mis = np.array(mis); kinds = np.array([k or "" for _, k, _ in st]); new = kinds != ""
    post = np.arange(len(st)) >= S.WARM; ok = post & ~new & ~mis
    x = np.array([d.step(xx, lf) for (xx, _, _), lf in zip(st, logged)])
    au = lambda pos: float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(ok.sum())], np.r_[x[pos], x[ok]]))  # noqa: E731
    return seed, dict(P_ctx25=dict(new_phase=au(post & new), wrong_sample=au(post & mis), stack=au(post & (kinds == "MoS2/WSe2_stack"))))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    out_dir = sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
    fn = dict(dose=dose, furnace=furnace, family=family, stopdrop=stopdrop, flagspec=flagspec)[JOB]
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        res = {str(s): r for s, r in ex.map(fn, TEST)}
    json.dump(dict(job=JOB, seeds=TEST, per_seed=res), open(f"{out_dir}/s25b_{JOB}.json", "w"), indent=1)
    print(JOB, "done", flush=True)
