#!/usr/bin/env python3
"""S18a: command down, local stop at the instrument, on a simulated selenization furnace.

S18 found that the fly nerve cord is a controller: a few descending commands, a local reflex driven by
the cord's own sensors, and a compressed status report sent up. S18a asks whether that arrangement helps
an autonomous furnace, and whether the measured nerve-cord wiring helps as the local reflex.

Furnace (reduced kinetics; all rates are assumptions, not MACE-MP-0 or kMC outputs):
  one MoS2 flake per run at 450 C; Se valve u in {0, 1}; the chamber Se level relaxes to u with
  time constant TAU_SE (purge after closing); top-layer exchange dx_t/dt = k_t * se * (1 - x_t);
  bottom-layer exchange starts as the top completes, dx_b/dt = RB * k_t * se * x_t**3 * (1 - x_b).
  k_t varies from flake to flake (log-normal, s.d. SIG_K) and is unknown to every controller.
  Success: x_t >= 0.90 and x_b <= 0.10 when the chamber is purged. Readings every DT s.
In situ Raman: the literature-anchored probe of S4L (s4_lit.raman_lit) at 450 C, 56 bands.

Arms (all use the same estimator of x_t, x_b from the spectrum except where noted):
  open_loop        planner sets the valve time from the median rate, no spectra
  planner_D{2,10,60}  every spectrum goes to the planner; its "close" arrives D s later
                   (D = 2-60 s: a planner turn from a hosted to a local language model)
  local_ridge_bands  the same stop rule run at the instrument (no delay); one status message per run;, estimator = ridge on 56 normalised bands (engineered reference)
  cord_{fly,shuf,rand}  as local, estimator = ridge on the motor-neuron activity of the BANC v888
                   nerve-cord graph (VNC sensory -> VNC intrinsic -> motor neurons, plus descending
                   command input), measured / degree-preserving shuffled / random sparse wiring
Estimators are fitted on 3 calibration flakes per seed (x_t, x_b known at every reading).
Stop rule (cold-start default): close when N_CONSEC readings in a row have estimated x_t >= XT_STOP
or x_b >= XB_STOP. Planner arms apply the same rule on the workstation, then wait D s.
Usage: python s18a_furnace.py BANC_DIR OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_lit as L  # noqa: E402

DT, T_MAX, T_C = 2.0, 600.0, 450.0
K0, SIG_K, RB, TAU_SE = 0.025, 0.4, 0.03, 10.0
XT_OK, XB_OK = 0.90, 0.10
XT_STOP, XB_STOP = 0.95, 0.06
N_CAL, N_RUN, G = 3, 30, 56
N_CONSEC = 3
CACHE = os.path.expanduser("work/p4_s18/banc_cord.npz")


# ---------------------------------------------------------------- nerve-cord graph
def build_cord(bd):
    import pandas as pd
    m = pd.read_feather(f"{bd}/banc_888_meta.feather").dropna(subset=["banc_888_id"])
    m["id"] = m.banc_888_id.astype("int64")
    sc, reg = m.super_class.astype(str), m.region.astype(str)
    role = pd.Series("o", index=m.index)
    role[sc == "motor"] = "M"
    role[sc.isin(["descending", "sensory_descending"])] = "D"
    role[(sc == "sensory") & (reg == "ventral_nerve_cord")] = "S"
    role[sc == "ventral_nerve_cord_intrinsic"] = "I"
    R = dict(zip(m.id, role))
    e = pd.read_parquet(f"{bd}/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet",
                        columns=["pre_root_id", "post_root_id", "num_synapses"])
    e.columns = ["pre", "post", "n"]
    e["a"], e["b"] = e.pre.map(R), e.post.map(R)
    idx = {r: {i: j for j, i in enumerate(sorted(m.id[role == r]))} for r in "SIMD"}
    out = {f"n_{r}": len(idx[r]) for r in "SIMD"}
    for a, b in (("S", "I"), ("D", "I"), ("I", "M"), ("S", "M"), ("D", "M"), ("I", "I")):
        x = e[(e.a == a) & (e.b == b)]
        out[f"{a}{b}_pre"] = x.pre.map(idx[a]).to_numpy(np.int32)
        out[f"{a}{b}_post"] = x.post.map(idx[b]).to_numpy(np.int32)
        out[f"{a}{b}_w"] = x.n.to_numpy(np.float32)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)


def load_cord(kind, rng):
    z = dict(np.load(CACHE))
    if kind != "fly":
        for blk in ("SI", "DI", "IM", "SM", "DM", "II"):
            post, w = z[f"{blk}_post"], z[f"{blk}_w"]
            if kind == "shuf":  # keeps every pre and post neuron's number of entries
                z[f"{blk}_post"] = rng.permutation(post)
            else:  # same number of entries placed uniformly
                nb = z[f"n_{blk[1]}"]; na = z[f"n_{blk[0]}"]
                z[f"{blk}_pre"] = rng.integers(0, na, len(w)).astype(np.int32)
                z[f"{blk}_post"] = rng.integers(0, nb, len(w)).astype(np.int32)
    return z


class Cord:
    """Two synaptic steps through the nerve cord: VNC sensory -> intrinsic -> motor, with direct
    sensory -> motor and descending (command) -> intrinsic/motor input. Rectified, mean-normalised."""

    def __init__(self, z, rng):
        self.z = z
        self.band = rng.integers(0, G, z["n_S"])  # each sensory neuron reads one Raman band (fixed map)
        self.cmd = rng.random(z["n_D"]) < 0.05    # descending neurons carrying the planner's command

    def _prop(self, blk, x, n_out):
        z = self.z
        return np.bincount(z[f"{blk}_post"], weights=z[f"{blk}_w"] * x[z[f"{blk}_pre"]], minlength=n_out)

    def features(self, xb, command=1.0):
        z = self.z
        s = xb[self.band]; d = command * self.cmd.astype(float)
        h = np.maximum(self._prop("SI", s, z["n_I"]) + self._prop("DI", d, z["n_I"]), 0)
        h /= h.mean() + 1e-12
        mn = np.maximum(self._prop("IM", h, z["n_M"]) + self._prop("SM", s, z["n_M"]) + self._prop("DM", d, z["n_M"]), 0)
        return mn / (mn.mean() + 1e-12)


# ---------------------------------------------------------------- furnace and probe
def spectrum(xt, xb, rng):
    st = dict(d_lat=0.0, x_top=xt, f_mos2=1 - xt, f_janus=xt * (1 - xb), f_mose2=xt * xb, theta=0.0)
    b = L.raman_lit(st, rng, G, T_C)
    return b / (b.sum() + 1e-12)


def run_flake(k, rng, ctrl, delay=0.0, t_open=None):
    """ctrl(band_vector) -> True to close. Returns outcome dict."""
    xt = xb = se = 0.0; u = 1.0; t = 0.0; close_at = None; msgs = 0; open_s = 0.0; traj = []
    while t < T_MAX + 120:
        if u > 0: open_s += DT
        if t_open is not None and t >= t_open: u = 0.0
        if close_at is not None and t >= close_at: u = 0.0
        se += (u - se) * (1 - np.exp(-DT / TAU_SE))
        xt += k * se * (1 - xt) * DT
        xb += RB * k * se * xt ** 3 * (1 - xb) * DT
        t += DT
        if u == 0 and se < 0.01: break
        if t >= T_MAX: u = 0.0
        if ctrl is not None and u > 0 and close_at is None:
            x = spectrum(xt, xb, rng); traj.append((x, xt, xb)); msgs += 1 if delay > 0 else 0
            if ctrl(x): close_at = t + delay
    ok = xt >= XT_OK and xb <= XB_OK
    return dict(ok=bool(ok), under=bool(xt < XT_OK), over=bool(xb > XB_OK), xt=float(xt), xb=float(xb),
                t=float(t), se_open_s=float(open_s), up_msgs=int(msgs) + 1), traj


def calib(rng, ks):
    X, Y = [], []
    for k in ks:  # calibration flakes: valve open to T_MAX, every reading labelled
        xt = xb = se = 0.0; t = 0.0
        while t < 360:
            se += (1 - se) * (1 - np.exp(-DT / TAU_SE)); xt += k * se * (1 - xt) * DT
            xb += RB * k * se * xt ** 3 * (1 - xb) * DT; t += DT
            X.append(spectrum(xt, xb, rng)); Y.append((xt, xb))
    return np.array(X), np.array(Y)


def ridge(F, Y, lam=1e-2):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    B = np.linalg.solve(A, Z.T @ (Y - Y.mean(0)))
    return lambda f: ((f - mu) / sd) @ B + Y.mean(0)


def run(args):
    seed, bd = args
    rng = np.random.default_rng(18000 + seed)
    ks_cal = K0 * np.exp(rng.normal(0, SIG_K, N_CAL)); ks = K0 * np.exp(rng.normal(0, SIG_K, N_RUN))
    Xc, Yc = calib(np.random.default_rng(18100 + seed), ks_cal)
    feats = {"bands": lambda x: x}
    for kind in ("fly", "shuf", "rand"):
        c = Cord(load_cord(kind, np.random.default_rng(18200 + seed)), np.random.default_rng(18300 + seed))
        feats[kind] = c.features
    est = {n: (f, ridge(np.array([f(x) for x in Xc]), Yc)) for n, f in feats.items()}

    def stop_rule(name):
        f, r = est[name]
        hist = []

        def ctrl(x):  # debounced: all of the last N_CONSEC estimates past the stop line
            p = r(f(x)); hist.append(p[0] >= XT_STOP or p[1] >= XB_STOP)
            if len(hist) > 400: hist.clear()
            return len(hist) >= N_CONSEC and all(hist[-N_CONSEC:])
        ctrl.reset = hist.clear
        return ctrl
    t_nom = -np.log(1 - 0.97) / K0 + TAU_SE * 0  # planner's fixed time from the median rate
    arms = {"open_loop": dict(ctrl=None, t_open=t_nom),
            "planner_D2": dict(ctrl=stop_rule("bands"), delay=2.0),
            "planner_D10": dict(ctrl=stop_rule("bands"), delay=10.0),
            "planner_D60": dict(ctrl=stop_rule("bands"), delay=60.0),
            "local_ridge_bands": dict(ctrl=stop_rule("bands")),
            "cord_fly": dict(ctrl=stop_rule("fly")),
            "cord_shuf": dict(ctrl=stop_rule("shuf")),
            "cord_rand": dict(ctrl=stop_rule("rand"))}
    out = {}
    for name, a in arms.items():
        prng = np.random.default_rng(18400 + seed)  # same probe noise sequence for every arm
        rs = []
        for k in ks:
            if a.get("ctrl") is not None: a["ctrl"].reset()
            rs.append(run_flake(k, prng, a.get("ctrl"), a.get("delay", 0.0), a.get("t_open"))[0])
        out[name] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0]}
        out[name]["lost"] = float(sum(not r["ok"] for r in rs))
    # estimator error on held-out calibration-style flakes (same for every arm that shares the estimator)
    Xh, Yh = calib(np.random.default_rng(18500 + seed), K0 * np.exp(np.random.default_rng(18600 + seed).normal(0, SIG_K, 3)))
    err = {n: float(np.mean(np.abs(np.array([r(f(x)) for x in Xh])[:, 0] - Yh[:, 0]))) for n, (f, r) in est.items()}
    t0 = time.perf_counter(); [feats["fly"](x) for x in Xh[:200]]; ms = (time.perf_counter() - t0) / 200 * 1e3
    return dict(seed=seed, arms=out, xt_mae=err, cord_ms_per_reading=ms, k_runs=ks.tolist())


if __name__ == "__main__":
    bd, outd, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(outd, exist_ok=True)
    if not os.path.exists(CACHE): build_cord(bd)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex:
        for r in ex.map(run, [(s, bd) for s in seeds]):
            json.dump(r, open(f"{outd}/s18a_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {a: (d["lost"], round(d["se_open_s"])) for a, d in r["arms"].items()},
                  {n: round(v, 3) for n, v in r["xt_mae"].items()}, flush=True)
