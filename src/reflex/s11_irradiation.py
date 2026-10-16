#!/usr/bin/env python3
"""S11: stop Ar+ irradiation of monolayer MoS2 at a target sulfur-vacancy density from the in situ
Raman time course, anchored on Daniel et al., 2D Mater. 13, 025033 (2026), doi:10.1088/2053-1583/ae6b2b.

Task. Defect engineering by ion irradiation (vacancy-rich MoS2 for catalysis, single-photon emitters,
doping) needs a set vacancy density. The ion counter gives the nominal fluence, but the vacancy density
per ion differs from flake to flake (beam current calibration, substrate, sputter yield). The run is
dosed in steps of 2.5e12 cm-2 with an in situ spectrum after each step (as in Daniel et al.); after
each spectrum the controller decides "dose again" or "stop". Overshooting cannot be undone.

Spectrum model, built from the paper and nothing else:
  * A1g and E12g position and width at fluence 0, 1e13, 5e13, 1e14 cm-2: Tables 2 and 3 (in situ rows),
    linearly interpolated in effective fluence.
  * Amplitudes of E12g, the defect-activated LA(M) band near 227 cm-1 and the LO(M) band near 360 cm-1,
    relative to A1g, with the LA(M) and LO(M) centre and width: fitted here to the four in situ spectra
    digitised from Fig. 3(a) (results/digitise_daniel2026/fig3a_spectra.csv), then interpolated.
Assumptions (not from the paper): vacancies per nominal ion vary between flakes by a log-normal factor
(sd 0.25); flake-to-flake strain/doping shifts both lines by N(0, 0.4) cm-1; intensity varies x0.6-1.6
between spectra (focus); background and noise as in S1.

Each seed: 3 calibration flakes (dosed to 1e14 cm-2, vacancy density known from STEM at every step,
as the paper did) and 30 test flakes, each with its own target in 2-6e13 cm-2 effective fluence.
Success: stop within +-15 % of the target. Overshoot beyond +15 %: flake lost.
Controllers (all trained on the same 3 calibration flakes; all decide "stop" at the first step whose
estimate reaches the target):
  dose         nominal fluence from the ion counter (no spectrum)
  ratio        physicist's route: LA(M)/A1g area ratio from the full spectrum, linear calibration
               (the paper's Fig. 3(d) relation), inverted
  ridge        ridge regression from the band vector (56 bands, area-normalised) to fluence
  knn          fluence of the nearest calibration band vector
  centroid     nearest of two centroids (calibration spectra below / at or above the target) on the
               area-normalised band vector: the same decision rule as the fly, without the model
  fly          band vector into the FlyWire antennal lobe (one band per glomerulus) and Kenyon cells
               (top 5 %); two output neurons learn the "go" and "stop" codes of the calibration spectra
               (dopamine-labelled by whether the spectrum is below or past the target); stop when the
               current code overlaps the stop memory more than the go memory
  fly_shuf / fly_rand   same with degree-preserving shuffled / random sparse wiring
  fly_near     three output neurons: "go" (below 85 % of target), "near" (within +-15 %), "past"; stop when
               the current code overlaps the near or past memory more than the go memory
  fly_lookup   diagnostic, not a fly mechanism: fluence of the calibration spectrum whose Kenyon-cell code
               overlaps most with the current code (tests whether the code keeps the fluence information);
               fly_shuf_lookup / fly_rand_lookup: same with the control wirings
  fly_trend    fly decision required on two consecutive spectra (uses the time course; the dose
               then stops one step later, so the target is lowered by one expected step)
  ratio_trend  same two-step rule on the ratio estimate
Usage: python s11_irradiation.py OUT_DIR SEEDS
"""
import csv, json, os, sys
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIG = os.path.join(HERE, "../../results/digitise_daniel2026/fig3a_spectra.csv")
TABLE = {0.0: dict(A1g=(404.0, 5.2), E2g=(383.2, 3.8)), 1.0: dict(A1g=(404.3, 6.2), E2g=(382.3, 4.1)),
         5.0: dict(A1g=(405.2, 10.5), E2g=(380.6, 6.3)), 10.0: dict(A1g=(405.5, 13.5), E2g=(376.7, 9.7))}
FLU = {"pristine": 0.0, "1e13": 1.0, "5e13": 5.0, "1e14": 10.0}
STEP, FMAX, TOL, N_CAL, N_TEST = 0.25, 12.0, 0.15, 3, 30
E_SD, POS_SD = 0.25, 0.4


def lor(x, c, w): return (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)


def fit_digitised():
    from scipy.optimize import curve_fit
    rows = list(csv.DictReader(open(DIG)))
    out = {}
    for lab, F in FLU.items():
        x = np.array([float(r["raman_shift_cm-1"]) for r in rows if r["trace"] == lab])
        y = np.array([float(r["intensity_px"]) for r in rows if r["trace"] == lab])
        (cA, wA), (cE, wE) = TABLE[F]["A1g"], TABLE[F]["E2g"]

        def f(x, aA, aE, aL, cL, wL, aO, cO, wO, b0, b1):
            return aA * lor(x, cA, wA) + aE * lor(x, cE, wE) + aL * lor(x, cL, wL) + aO * lor(x, cO, wO) + b0 + b1 * (x - 300)
        p0 = [y.max(), y.max() / 2, y.max() / 10, 227, 15, y.max() / 10, 362, 25, 0, 0]
        lo = [0, 0, 0, 215, 5, 0, 345, 10, -50, -1]; hi = [1e4, 1e4, 1e4, 240, 40, 1e4, 378, 60, 50, 1]
        p, _ = curve_fit(f, x, y, p0=p0, bounds=(lo, hi), maxfev=40000)
        out[F] = dict(E_rel=p[1] / p[0], L_rel=p[2] / p[0], cL=p[3], wL=p[4], O_rel=p[5] / p[0], cO=p[6], wO=p[7],
                      rms=float(np.sqrt(np.mean((f(x, *p) - y) ** 2)) / p[0]))
    return out


MODEL = None


def params(F):
    Fs = np.array(sorted(TABLE)); g = lambda v: float(np.interp(F, Fs, v))  # noqa: E731
    d = dict(cA=g([TABLE[k]["A1g"][0] for k in Fs]), wA=g([TABLE[k]["A1g"][1] for k in Fs]),
             cE=g([TABLE[k]["E2g"][0] for k in Fs]), wE=g([TABLE[k]["E2g"][1] for k in Fs]))
    for key in ("E_rel", "L_rel", "cL", "wL", "O_rel", "cO", "wO"):
        d[key] = g([MODEL[k][key] for k in Fs])
    # beyond 1e14 cm-2 keep growing linearly with the 5e13-1e14 slope (only reached by overshooting runs)
    if F > 10:
        for key, k2 in (("wA", "A1g"), ("wE", "E2g")):
            d[key] += (F - 10) * (TABLE[10.0][k2][1] - TABLE[5.0][k2][1]) / 5
        d["L_rel"] += (F - 10) * (MODEL[10.0]["L_rel"] - MODEL[5.0]["L_rel"]) / 5
    return d


def spectrum(F, flake, rng):
    p = params(F); s = flake["dpos"]
    y = (lor(S.NU, p["cA"] + s, p["wA"]) + p["E_rel"] * lor(S.NU, p["cE"] + s, p["wE"])
         + p["L_rel"] * lor(S.NU, p["cL"], p["wL"]) + p["O_rel"] * lor(S.NU, p["cO"], p["wO"]))
    scale = np.exp(rng.uniform(np.log(0.6), np.log(1.6)))
    bg = rng.uniform(0, 0.05) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
    y = scale * (y + bg)
    return np.maximum(y + rng.normal(0, 0.01 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)


def ratio(y):
    def area(a, b):
        m = (S.NU >= a) & (S.NU <= b); x, v = S.NU[m], y[m]
        base = np.interp(x, [x[0], x[-1]], [v[:3].mean(), v[-3:].mean()])
        return float(np.trapezoid(v - base, x))
    return area(205, 250) / max(area(392, 418), 1e-6)


def norm(b): return b / (b.sum() + 1e-12)


class FlyStop:
    def __init__(self, W, Min):
        self.f = S.Fly(W, Min)
    def codes(self, B): return np.array([self.f.code(b) for b in B]).astype(float)
    def fit(self, C, above):
        self.go = C[~above].mean(0) if (~above).any() else np.zeros(C.shape[1])
        self.st = C[above].mean(0) if above.any() else np.zeros(C.shape[1])
    def stop(self, c): return float(c @ self.st) > float(c @ self.go)
    def fit3(self, C, F, Ft):
        g, nr, pa = F < (1 - TOL) * Ft, np.abs(F - Ft) <= TOL * Ft, F > (1 + TOL) * Ft
        m = lambda s: C[s].mean(0) if s.any() else np.zeros(C.shape[1])  # noqa: E731
        self.g3, self.n3, self.p3 = m(g), m(nr), m(pa)
    def stop3(self, c): return max(float(c @ self.n3), float(c @ self.p3)) > float(c @ self.g3)


def run(seed):
    global MODEL
    MODEL = fit_digitised()
    rng = np.random.default_rng(seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    r, c = np.nonzero(W); w = W[r, c]; crng = np.random.default_rng(1000 + seed)
    Ws = np.zeros_like(W); np.add.at(Ws, (r, crng.permutation(c)), w)
    Wr = np.zeros_like(W); np.add.at(Wr, (crng.integers(0, W.shape[0], len(w)), crng.integers(0, W.shape[1], len(w))), w)
    flies = {"fly": FlyStop(W, Min), "fly_shuf": FlyStop(Ws, Min), "fly_rand": FlyStop(Wr, Min)}
    newflake = lambda: dict(e=float(np.exp(rng.normal(0, E_SD))), dpos=float(rng.normal(0, POS_SD)))  # noqa: E731
    # calibration flakes: every step to 1e14 nominal, effective fluence known (STEM)
    calF, calY = [], []
    for _ in range(N_CAL):
        fl = newflake()
        for k in range(int(10 / STEP) + 1):
            Fe = fl["e"] * k * STEP; calF.append(Fe); calY.append(spectrum(Fe, fl, rng))
    calF = np.array(calF); calB = np.array([norm(S.bands(y, G)) for y in calY])
    calR = np.array([ratio(y) for y in calY])
    a, b = np.polyfit(calF, calR, 1)[::-1]
    Xr = np.c_[calB, np.ones(len(calB))]; lam = 1e-4
    beta = np.linalg.solve(Xr.T @ Xr + lam * np.eye(Xr.shape[1]), Xr.T @ calF)
    calC = {n: f.codes([S.bands(y, G) for y in calY]) for n, f in flies.items()}
    arms = ["dose", "ratio", "ratio_trend", "ridge", "knn", "centroid"] + list(flies) + ["fly_near", "fly_lookup", "fly_shuf_lookup", "fly_rand_lookup", "fly_trend"]
    res = {a_: [] for a_ in arms}
    for t in range(N_TEST):
        fl = newflake(); Ft = float(rng.uniform(2, 6)); srng = np.random.default_rng([seed, t])
        above = calF >= Ft
        for n, f in flies.items(): f.fit(calC[n], above)
        flies["fly"].fit3(calC["fly"], calF, Ft)
        cg = calB[~above].mean(0); cs = calB[above].mean(0)
        Ft2 = Ft - STEP * 1.0  # trend rules stop one step after the first vote
        done = {a_: None for a_ in arms}; prev = {"fly_trend": False, "ratio_trend": False}
        k = 0
        while k * STEP <= FMAX and any(v is None for v in done.values()):
            k += 1; Fn = k * STEP; Fe = fl["e"] * Fn
            y = spectrum(Fe, fl, srng); bands = S.bands(y, G); bn = norm(bands)
            est = dict(dose=Fn, ratio=(ratio(y) - a) / b, ridge=float(np.r_[bn, 1] @ beta),
                       knn=float(calF[np.argmin(((calB - bn) ** 2).sum(1))]))
            vote = {a_: est[a_] >= Ft for a_ in est}
            vote["centroid"] = ((bn - cs) ** 2).sum() < ((bn - cg) ** 2).sum()
            for n, f in flies.items():
                cn = f.f.code(bands).astype(float); vote[n] = f.stop(cn)
                vote[n + "_lookup"] = float(calF[np.argmax(calC[n] @ cn)]) >= Ft
            cf = flies["fly"].f.code(bands).astype(float)
            fv2 = flies["fly"].stop(cf)  # same as vote["fly"]
            vote["fly_near"] = flies["fly"].stop3(cf)
            rv2 = est["ratio"] >= Ft2
            vote["fly_trend"] = prev["fly_trend"] and fv2; prev["fly_trend"] = fv2
            vote["ratio_trend"] = prev["ratio_trend"] and rv2; prev["ratio_trend"] = rv2
            for a_ in arms:
                if done[a_] is None and (vote[a_] or k * STEP >= FMAX): done[a_] = Fe
        for a_ in arms:
            e = (done[a_] - Ft) / Ft
            res[a_].append(dict(target=Ft, stop=done[a_], rel_err=e))
    summ = {a_: dict(success=float(np.mean([abs(r_["rel_err"]) <= TOL for r_ in v])),
                     lost=float(np.mean([r_["rel_err"] > TOL for r_ in v])),
                     under=float(np.mean([r_["rel_err"] < -TOL for r_ in v])),
                     mae=float(np.mean([abs(r_["rel_err"]) for r_ in v]))) for a_, v in res.items()}
    return dict(seed=seed, G=G, cal_ratio_fit=[float(a), float(b)], model_fit={str(k): v for k, v in MODEL.items()},
                summary=summ, runs=res)


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s11_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["success"], 2) for n, d in r["summary"].items()}, flush=True)
