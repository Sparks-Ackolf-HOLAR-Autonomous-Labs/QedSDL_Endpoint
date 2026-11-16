#!/usr/bin/env python3
"""S23: a stop rule that can stop at any point of the anneal, not only at one fixed probe.

s4_insitu takes one in-situ spectrum at PROBE_FRAC = 0.25 of each anneal and may stop the run there. Here the same
run (same recipe choice, same kinetic Monte Carlo trajectory and outcome, same Raman model) is probed at
CHECKS = 0.1, 0.2, ..., 0.9 of the anneal, and the run is stopped at the first checkpoint where the rule says
"fail"; a run stopped at f uses f of an anneal. Each checkpoint has its own memory (one detector per
checkpoint), written only from completed runs with the spectrum taken at that checkpoint, so a 10 % spectrum is
only compared with other 10 % spectra. Detectors need >= 1 failure and >= 1 success at a checkpoint before it
may stop a run, and none stop in the first N0 runs (as s4_insitu).
Arms: gp (BO planner alone), gp+fly (plain fly stop rule, the setting chosen in s23_fair_fly t1), gp+logreg,
gp+knn, gp+fly_best (the paper's fly setting, H.default_full_cfg), gp+mlp (MLP 64x64, alpha 1e-4, the setting chosen in s23_fair_nn j1). Seeds 1-20, 30 recipes.
Outputs per arm: flakes lost, saved, false stops, furnace time, and the stop-time distribution (checkpoint of
every stop, split by whether the run would have failed).
S23_ARMS=gp+fly_best runs only the listed arms.
Usage: CUDA_VISIBLE_DEVICES="" python s23_stop_anytime.py OUT_JSON [WORKERS]
"""
import json, os, sys, warnings
os.environ.setdefault("S1_MODE", "janus")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
CHECKS = tuple(round(0.1 * k, 1) for k in range(1, 10))
ARMS = tuple(os.environ.get("S23_ARMS", "gp,gp+fly,gp+logreg,gp+knn,gp+mlp").split(","))


def simulate_probes(S4, recipe, p, rng, L=24):
    """s4_insitu.simulate_probe with the lattice state recorded at every checkpoint (same draws, same trajectory)."""
    kmc = S4.kmc
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
    P = lambda k: 1.0 - np.exp(-np.minimum(k * dt, 50.0))  # noqa: E731
    k_des = kmc.NU * np.exp(-p["E_des"] / kT); theta = p["nu_cond"] * se / (p["nu_cond"] * se + k_des)

    def state():
        n = L * L
        conv_t = top == SE; conv_b = bot == SE
        d_lat = ((top == V) | (top == D)).sum() / n * 0.5 + (bot == V).sum() / n * 0.5
        return dict(x_top=conv_t.mean(), x_bot=conv_b.mean(), d_lat=d_lat, theta=theta,
                    f_mos2=(~conv_t & ~conv_b).mean(), f_janus=(conv_t ^ conv_b).mean(), f_mose2=(conv_t & conv_b).mean())
    at = {int(kmc.NSTEP * f): i for i, f in enumerate(CHECKS)}
    probes = [None] * len(CHECKS)
    for step in range(kmc.NSTEP):
        if step in at: probes[at[step]] = state()
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
    return probes, dict(F=float(F), x_top_Se=float(fin["x_top"]), x_bot_Se=float(fin["x_bot"]), defects=float(defects))


def one(args):
    seed, arm = args
    import s4_insitu as S4
    from t3_avoid import gp_ucb, BUDGET, N0, TB, TD
    K = S4.K
    from sklearn.neural_network import MLPClassifier

    class MlpStop(S4.KnnStop):
        def stop(self, x):
            if len({f for _, f in self.mem}) < 2: return False
            X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
            c = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=2000, random_state=0).fit(X * 50, y)
            return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
    Wd, Min, G = S4.H.wirings(seed)
    mk = {"gp": None, "gp+fly": lambda: S4.FlyStop(Wd["fly"], Min, {}), "gp+fly_best": lambda: S4.FlyStop(Wd["fly"], Min, S4.H.default_full_cfg()), "gp+logreg": S4.LogStop, "gp+knn": S4.KnnStop, "gp+mlp": MlpStop}[arm]
    rng = np.random.default_rng(9000 + seed); prng = np.random.default_rng(7000 + seed)
    start = list(np.random.default_rng(seed).choice(K.NC, N0, replace=False))
    dets = [mk() for _ in CHECKS] if mk else None
    T = K.TRUTH["Mo"]
    idx, y, done, stops = [], [], [], []
    lost = saved = false_stop = 0; furnace = 0.0
    for n in range(BUDGET):
        if n < N0: j = int(start[n])
        else:
            s = gp_ucb(idx, y).astype(float); s[idx] = -np.inf; j = int(np.argmax(s))
        probes, m = simulate_probes(S4, K.R[j], K.PAR["Mo"], rng)
        m = {k: float(np.clip(v + rng.normal(0, 0.02), 0, 1)) for k, v in m.items()}
        fail = m["x_bot_Se"] > TB or m["defects"] > TD
        xs = [S4.raman(pr, prng, G) for pr in probes]        # all checkpoint spectra drawn up front (same noise for every arm)
        idx.append(j)
        stop_at = None
        if dets is not None:
            for i, f in enumerate(CHECKS):
                if dets[i].stop(xs[i]) and n >= N0:
                    stop_at = f; break
        if stop_at is not None:
            furnace += stop_at; saved += fail; false_stop += (not fail); stops.append((stop_at, bool(fail)))
            y.append(0.0)                                      # the GP sees a stopped run as a poor recipe (as s4_insitu)
        else:
            furnace += 1.0; lost += fail; y.append(m["F"]); done.append(j)
            if dets is not None:
                for i in range(len(CHECKS)):
                    dets[i].learn(xs[i], fail)
    return seed, arm, dict(lost=int(lost), saved=int(saved), false_stops=int(false_stop), furnace_runs=float(furnace),
                           gap=float(T.max() - T[done].max()) if done else float(T.max()), stops=stops)


if __name__ == "__main__":
    out = sys.argv[1]
    jobs = [(s, a) for s in range(1, 21) for a in ARMS]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as ex:
        res = list(ex.map(one, jobs))
    summ = {}
    for a in ARMS:
        rs = [r for s, aa, r in res if aa == a]
        st = [x for r in rs for x in r["stops"]]
        summ[a] = dict(lost=float(np.mean([r["lost"] for r in rs])), saved=float(np.mean([r["saved"] for r in rs])),
                       false_stops=float(np.mean([r["false_stops"] for r in rs])), furnace_runs=float(np.mean([r["furnace_runs"] for r in rs])),
                       gap=float(np.mean([r["gap"] for r in rs])),
                       stop_hist_saved={str(f): sum(1 for g, fl in st if g == f and fl) for f in CHECKS},
                       stop_hist_false={str(f): sum(1 for g, fl in st if g == f and not fl) for f in CHECKS})
    json.dump(dict(checks=CHECKS, summary=summ, per_seed=[dict(seed=s, arm=a, **r) for s, a, r in res]), open(out, "w"), indent=1)
    for a, v in summ.items():
        print(a, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
