#!/usr/bin/env python3
"""S4: avoid memory keyed on an in-situ Raman spectrum, in a Janus MoSSe selenization campaign.

Materials setting: MoS2 monolayers are converted to Janus MoSSe in a Se-vapour furnace (kmc.py, the
same lattice model and parameters as the Janus campaign and T3). A run destroys the flake when the bottom
S layer converts too (toward MoSe2, x_bot_Se > 0.3) or the lattice or surface degrades (defects > 0.3).
T3 keyed the fly's avoid memory on recipe settings and did not beat the GP. Here a Raman probe is taken
a quarter of the way through each anneal; the furnace run can be stopped there, which saves the flake
(it can be re-used) and three quarters of the furnace time, but gives no fidelity result.

Probe spectrum (simulated from the lattice state at step NSTEP/4 of the same trajectory, 100-500 cm^-1,
literature line positions as in S1): MoS2 (E 383, A1 403) for sites with neither layer converted,
Janus MoSSe (A1 290, E 355) for one layer converted, MoSe2 (A1 241, E 287) for both, line widths
broadened by lattice defects, defect-activated LA(M) band near 227 cm^-1 (defective MoS2), and a broad
amorphous-Se band near 250 cm^-1 for the condensed Se adlayer. Unknown intensity (x0.2-x5), sloped
background, noise, binned one band per FlyWire glomerulus.
Policy: GP-UCB proposes the recipe (as T3); the detector sees the probe and may stop the run. Detectors
learn only from completed runs (an aborted run has no outcome label), and need >= 1 failure and >= 1
success before they may stop a run.
  gp            no probe check (T3 baseline)
  cbo           constrained BO on recipe settings (T3 reference; no probe)
  gp+fly*       KC code of the probe; avoid compartment written one-shot by failed runs, approach
                compartment by successful ones; stop if avoid > approach. * = config (plain / all)
                and wiring (measured, shuffled, random)
  gp+knn        stop if the nearest failed probe (area-normalised) is closer than the nearest successful one
  gp+logreg     logistic regression on area-normalised probes; stop if p(fail) > 0.5
Measures (30 recipes per campaign): flakes lost, runs stopped that would have failed (saved) or
succeeded (false stops), furnace time in full-run units, gap = best true F on the grid minus best true F
among completed runs.
Usage: python s4_insitu.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import s1_spectra as S  # noqa: E402
import s2_s3 as H  # noqa: E402
from fly2 import Fly2  # noqa: E402
from t3_avoid import gp_ucb, p_success, BUDGET, N0, TB, TD  # noqa: E402
kmc = K.kmc
T = K.TRUTH["Mo"]
PROBE_FRAC = 0.25
LINES = dict(MoS2=[(383, 1.0, 4), (403, 1.2, 4)], MoSSe=[(290, 1.2, 5), (355, 0.7, 6)], MoSe2=[(241, 1.3, 4), (287, 0.4, 5)])


def simulate_probe(recipe, p, rng, L=24):
    """kmc.simulate with the lattice state also returned at step NSTEP*PROBE_FRAC (same trajectory)."""
    strip, T_C, t_min, se = recipe
    Sx, SE, V, D = 0, 1, 2, 3
    top = np.zeros((L, L), np.int8); bot = np.zeros((L, L), np.int8)
    top[rng.random((L, L)) < strip] = V
    top[(top == V) & (rng.random((L, L)) < p["plasma_dmg"] * strip ** 3)] = D
    kT = kmc.KB * (T_C + 273.15); dmu = kT * np.log(se)
    k_fill = kmc.NU_ADS * se * np.exp(-p["Ea_fill"] / kT)
    k_ext = kmc.NU * np.exp(-(p["Ea_ex"] + 0.5 * p["dE_top"] - 0.5 * dmu) / kT)
    k_exb = kmc.NU * np.exp(-(p["Ea_ex"] + p["dEa_bot"] + 0.5 * p["dE_bot"] - 0.5 * dmu) / kT)
    k_lt = kmc.NU * np.exp(-p["Ea_loss"] / kT); k_lb = kmc.NU * np.exp(-(p["Ea_loss"] + 0.3) / kT)
    k_dmg = kmc.NU * np.exp(-(p["Ea_hop"] + p["dEa_dmg"]) / kT)
    dt = t_min * 60.0 / kmc.NSTEP
    P = lambda k: 1.0 - np.exp(-np.minimum(k * dt, 50.0))
    k_des = kmc.NU * np.exp(-p["E_des"] / kT); theta = p["nu_cond"] * se / (p["nu_cond"] * se + k_des)

    def state():
        n = L * L
        conv_t = top == SE; conv_b = bot == SE
        d_lat = ((top == V) | (top == D)).sum() / n * 0.5 + (bot == V).sum() / n * 0.5
        return dict(x_top=conv_t.mean(), x_bot=conv_b.mean(), d_lat=d_lat, theta=theta,
                    f_mos2=(~conv_t & ~conv_b).mean(), f_janus=(conv_t ^ conv_b).mean(), f_mose2=(conv_t & conv_b).mean())
    probe = None
    for step in range(kmc.NSTEP):
        if step == int(kmc.NSTEP * PROBE_FRAC): probe = state()
        u = rng.random((6, L, L))
        nv = kmc._nb(((top == V) | (top == D)).astype(np.float32))
        tv, ts, bs, bv = top == V, top == Sx, bot == Sx, bot == V
        new_top = top.copy(); new_bot = bot.copy()
        kd = np.where(nv >= 4, k_dmg, 0.0); ev = tv & (u[0] < P(k_fill + kd))
        dmg = ev & (u[1] < kd / (k_fill + kd))
        new_top[ev & ~dmg] = SE; new_top[dmg] = D
        new_top[ts & (u[2] < P(k_ext * (1 + p["c_v"] * nv)))] = SE
        new_top[((top == Sx) | (top == SE)) & (u[3] < P(k_lt))] = V
        new_bot[bs & (u[4] < P(k_exb * (1 + p["c_v"] * nv)))] = SE
        lb = (bot != V) & (u[5] < P(k_lb)); new_bot[lb] = V
        new_bot[bv & (u[5] < P(0.01 * k_fill))] = SE
        top, bot = new_top, new_bot
    fin = state()
    defects = 1 - (1 - fin["d_lat"]) * (1 - theta)
    F = fin["x_top"] * (1 - fin["x_bot"]) * (1 - defects)
    return probe, dict(F=float(F), x_top_Se=float(fin["x_top"]), x_bot_Se=float(fin["x_bot"]), defects=float(defects))


def raman(st, rng, G):
    br = 1 + 3 * st["d_lat"]
    s = np.zeros_like(S.NU)
    for ph, fr in (("MoS2", st["f_mos2"]), ("MoSSe", st["f_janus"]), ("MoSe2", st["f_mose2"])):
        s += fr * S.lines([(c, a, w * br) for c, a, w in LINES[ph]])
    s += 2.0 * st["d_lat"] * S.lines([(227, 0.6, 12)]) + 0.8 * st["theta"] * S.lines([(250, 1.0, 30)])
    scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
    bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
    y = scale * (s + bg)
    y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
    return S.bands(y, G)


class FlyStop:
    def __init__(self, W, Min, cfg):
        self.f = Fly2(W, Min, **cfg); n = W.shape[0]; self.av = np.zeros(n); self.ap = np.zeros(n); self.nf = self.ns = 0

    def stop(self, x):
        c = self.last = self.f.code(x)  # computed once per probe (adaptation state advances once)
        if not (self.nf and self.ns): return False
        return self.av[c].mean() > self.ap[c].mean()

    def learn(self, x, fail):
        c = self.last
        if fail: self.av[c] = 1.0; self.nf += 1
        else: self.ap[c] = 1.0; self.ns += 1


class KnnStop:
    def __init__(self): self.mem = []

    def stop(self, x):
        if len({f for _, f in self.mem}) < 2: return False
        v = x / x.sum(); d = {f: min(np.linalg.norm(v - m) for m, g in self.mem if g == f) for f in (True, False)}
        return d[True] < d[False]

    def learn(self, x, fail): self.mem.append((x / x.sum(), fail))


class LogStop(KnnStop):
    def stop(self, x):
        if len({f for _, f in self.mem}) < 2: return False
        from sklearn.linear_model import LogisticRegression
        X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
        c = LogisticRegression(C=1.0, max_iter=500).fit(X * 50, y)
        return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5


def arms(Wd, Min):
    full = H.default_full_cfg()
    return {"gp": None, "cbo": None,
            "gp+fly": lambda: FlyStop(Wd["fly"], Min, {}),
            "gp+fly_best": lambda: FlyStop(Wd["fly"], Min, full),
            "gp+fly_best_shuf": lambda: FlyStop(Wd["shuf"], Min, full),
            "gp+fly_best_rand": lambda: FlyStop(Wd["rand"], Min, full),
            "gp+knn": KnnStop, "gp+logreg": LogStop}


def run_arm(arm, mk, seed, G):
    rng = np.random.default_rng(9000 + seed); prng = np.random.default_rng(7000 + seed)
    start = list(np.random.default_rng(seed).choice(K.NC, N0, replace=False))
    det = mk() if mk else None
    idx, y, fl, done = [], [], [], []
    lost = saved = false_stop = 0; furnace = 0.0; t_det = 0.0
    for n in range(BUDGET):
        if n < N0: j = int(start[n])
        else:
            s = gp_ucb(idx, y)
            if arm == "cbo": s = (s - s.min()) * p_success(idx, fl)
            s = s.astype(float); s[idx] = -np.inf; j = int(np.argmax(s))
        probe, m = simulate_probe(K.R[j], K.PAR["Mo"], rng)
        m = {k: float(np.clip(v + rng.normal(0, 0.02), 0, 1)) for k, v in m.items()}
        fail = m["x_bot_Se"] > TB or m["defects"] > TD
        x = raman(probe, prng, G)
        t0 = time.perf_counter(); stop = det is not None and det.stop(x) and n >= N0; t_det += time.perf_counter() - t0
        idx.append(j)
        if stop:
            furnace += PROBE_FRAC; saved += fail; false_stop += (not fail)
            y.append(0.0); fl.append(True)  # the GP sees a stopped run as a poor recipe
        else:
            furnace += 1.0; lost += fail; y.append(m["F"]); fl.append(fail); done.append(j)
            if det is not None: det.learn(x, fail)
    return dict(lost=int(lost), saved=int(saved), false_stops=int(false_stop), furnace_runs=float(furnace),
                gap=float(T.max() - T[done].max()) if done else float(T.max()), ms_per_check=1e3 * t_det / (BUDGET - N0))


def run(seed):
    Wd, Min, G = H.wirings(seed)
    return dict(seed=seed, arms={a: run_arm(a, mk, seed, G) for a, mk in arms(Wd, Min).items()})


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s4_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {a: (d["lost"], d["saved"], d["false_stops"], round(d["gap"], 3)) for a, d in r["arms"].items()}, flush=True)
