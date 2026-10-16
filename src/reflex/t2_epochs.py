#!/usr/bin/env python3
"""T2 (reflex layer): keep what was learned in an earlier epoch while learning a new one.

Epoch A: 150 noisy furnace runs on MoS2 -> Janus MoSSe (kmc, Mo parameters), random recipes.
Epoch B: 150 runs on WS2 -> Janus WSSe (W parameters). Every model is told the epoch (context input).
After each epoch, every model predicts Janus fidelity F for all 1400 recipes of both systems;
error = RMSE against the 8-replicate truth maps.
Models:
  fly / fly_shuf / fly_rand  KC code of (recipe, epoch) on measured / shuffled / random PN->KC
                   wiring; KC->MBON readout learned online, one pass, by a delta rule (dopamine-style
                   prediction error), only on KCs with MBON partners
  mlp_seq          small neural network (2 x 64) trained online on the stream with the epoch input;
                   the standard case for forgetting
  gp_ctx           Gaussian process on all runs with an epoch feature (refit; upper reference)
  gp_pool          Gaussian process on all runs without the epoch feature
  gp_new           Gaussian process refit on the new epoch only (earlier epoch dropped)
Measures: RMSE on A after A, RMSE on A after B (forgetting = increase), RMSE on B after B.
Usage: python t2_epochs.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor

NE = 150
LR = 400.0  # readout step size, tuned on seed 0 (not a test seed): 1, 20, 100, 400 -> RMSE A 0.40, 0.26, 0.23, 0.21


def gp(A, y):
    k = ConstantKernel(1.0) * Matern([0.3] * A.shape[1], (0.03, 30), nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(A, y)


def rmse(p, M): return float(np.sqrt(np.mean((p - K.TRUTH[M]) ** 2)))


def run(seed):
    rng = np.random.default_rng(5000 + seed)
    stream = []
    for e, M in enumerate(["Mo", "W"]):
        for j in rng.integers(0, K.NC, NE):
            stream.append((e, int(j), K.kmc.measure(K.R[j], K.PAR[M], rng)["F"]))
    Xc = lambda e: np.hstack([K.X, np.full((K.NC, 1), float(e))])
    res = dict(seed=seed, models={}); mask = K.mbon_mask()
    for name, W in K.wirings(np.random.default_rng(seed)).items():
        t0 = time.perf_counter(); cd = [K.codes(W, 0), K.codes(W, 1)]; w = np.zeros(len(mask)); out = {}
        ov = float(np.mean((cd[0] * cd[1]).sum(1) / cd[0].sum(1)))
        for i, (e, j, y) in enumerate(stream):
            c = cd[e][j]; pred = (c * w).sum() / c.sum()
            w += 0.5 * (y - pred) * c * mask / c.sum() * LR
            if i in (NE - 1, 2 * NE - 1):
                P = [(cd[k] * w).sum(1) / cd[k].sum(1) for k in (0, 1)]
                out["A_after_A" if i < NE else "A_after_B"] = rmse(P[0], "Mo")
                if i >= NE: out["B_after_B"] = rmse(P[1], "W")
        out.update(code_overlap_A_B=ov, s=time.perf_counter() - t0); res["models"][name] = out
    # neural network, online
    t0 = time.perf_counter(); nn = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=3e-3, random_state=seed); out = {}
    for i in range(0, 2 * NE, 10):
        b = stream[i:i + 10]
        for _ in range(20):
            nn.partial_fit(np.array([np.r_[K.X[j], e] for e, j, _ in b]), np.array([y for *_, y in b]))
        if i + 10 in (NE, 2 * NE):
            out["A_after_A" if i < NE else "A_after_B"] = rmse(nn.predict(Xc(0)), "Mo")
            if i >= NE: out["B_after_B"] = rmse(nn.predict(Xc(1)), "W")
    out["s"] = time.perf_counter() - t0; res["models"]["mlp_seq"] = out
    A = np.array([np.r_[K.X[j], e] for e, j, _ in stream]); y = np.array([v for *_, v in stream])
    t0 = time.perf_counter(); gA = gp(A[:NE], y[:NE]); a_after_a = rmse(gA.predict(Xc(0)), "Mo")
    g = gp(A, y)
    res["models"]["gp_ctx"] = dict(A_after_A=a_after_a, A_after_B=rmse(g.predict(Xc(0)), "Mo"),
                                   B_after_B=rmse(g.predict(Xc(1)), "W"), s=time.perf_counter() - t0)
    gA4 = gp(A[:NE, :4], y[:NE]); gp4 = gp(A[:, :4], y); pp = gp4.predict(K.X)
    res["models"]["gp_pool"] = dict(A_after_A=rmse(gA4.predict(K.X), "Mo"), A_after_B=rmse(pp, "Mo"), B_after_B=rmse(pp, "W"))
    gB = gp(A[NE:, :4], y[NE:]); pb = gB.predict(K.X)
    res["models"]["gp_new"] = dict(A_after_A=rmse(gA4.predict(K.X), "Mo"), A_after_B=rmse(pb, "Mo"), B_after_B=rmse(pb, "W"))
    res["ref"] = dict(rmse_mean_predictor_Mo=rmse(np.full(K.NC, K.TRUTH["Mo"].mean()), "Mo"),
                      rmse_W_truth_as_Mo=rmse(K.TRUTH["W"], "Mo"))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/t2_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: [round(v, 3) for k, v in d.items() if k[0] in "AB"] for n, d in r["models"].items()}, flush=True)
