#!/usr/bin/env python3
"""T4 (reflex layer): the fly familiarity check as a reflex layer under three GP parties.

Network: three parties (GP upper confidence bound, kappa 1, 2, 3) share every accepted result of a
Janus MoSSe campaign (kmc, Mo parameters) and three furnaces; 36 runs in total, one run per furnace
per round. From round 4 furnace 2 has an unannounced, persistent fault, drawn per seed:
thermocouple reading 100 C low (the flake sees set point + 100 C) or an exhausted Se source
(supply 0.1 of nominal). The run is reported under the nominal recipe.
Reflex policy (same for every detector): each reading is scored before it is shared; a score above
the 90th percentile of earlier scores holds the reading back and queues a repeat of the same recipe
on another furnace. If the repeat differs by more than 0.08 in any readout channel the held reading
is discarded and its furnace gets a strike; two strikes take the furnace out of service. Repeats
count against the 36-run budget.
Every detector first reads the facility's history: 60 earlier fault-free runs on random recipes
(the same history in every arm of a seed); the parties' models start from the 3 campaign starts only.
Arms: none (every reading shared), fly / fly_shuf / fly_rand (T1 detector on measured, shuffled,
random PN->KC wiring), gp_resid (T1 GP residual detector), oracle (furnace 2 out of service at onset).
Measures: faulty readings shared, runs spent on the faulty furnace after onset, repeats,
round the furnace left service, gap = best true F on the grid minus best true F among shared recipes.
Usage: python t4_network.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import t1_janus_faults as T1  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

BUDGET, N0, ONSET, BAD, KAPPA, NHIST = 36, 3, 4, 2, (1.0, 2.0, 3.0), 60
T = K.TRUTH["Mo"]
ARMS = ["none", "fly", "fly_shuf", "fly_rand", "gp_resid", "oracle"]


def ucb(idx, y, kappa):
    k = ConstantKernel(1.0) * Matern([0.3] * 4, (0.03, 30), nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e-1))
    g = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(K.X[idx], y)
    m, s = g.predict(K.X, return_std=True); return m + kappa * s


def run_arm(arm, seed, dets):
    rng = np.random.default_rng(7000 + seed); fault = ["furnace_offset", "se_empty"][seed % 2]
    start = list(np.random.default_rng(seed).choice(K.NC, N0, replace=False))
    shared, mem, scores = [], [], []      # shared: (recipe, F, readout); mem: detector memory
    held, repeats_q, strikes, active = {}, [], {0: 0, 1: 0, 2: 0}, [0, 1, 2]
    hrng = np.random.default_rng(seed + 50000)
    if arm in dets or arm == "gp_resid":
        for j in hrng.integers(0, K.NC, NHIST):
            m = K.kmc.measure(K.R[j], K.PAR["Mo"], hrng); yv = np.array([m["x_top_Se"], m["x_bot_Se"], m["defects"]])
            if arm in dets: sc, item = dets[arm].score(mem, int(j), yv)
            else: item = (int(j), yv); sc = T1.gp_score(mem, int(j), yv) if len(mem) >= 5 else 0.0
            scores.append(sc); mem.append(item)
    used = rnd = bad_shared = bad_runs = n_rep = 0; out_round = None; tested = set()
    while used < BUDGET:
        rnd += 1
        if arm == "oracle" and rnd == ONSET and BAD in active: active.remove(BAD); out_round = rnd
        jobs = []
        for f in active:
            if repeats_q:
                key = repeats_q.pop(0)
                if key[1] == f and len(active) > 1: repeats_q.insert(0, key); jobs.append((f, None)); continue
                jobs.append((f, key)); continue
            jobs.append((f, None))
        props = []
        for p, (f, key) in enumerate(jobs):
            if used >= BUDGET: break
            if key is not None: j = key[0]
            elif len(shared) + len(props) < N0 and len(tested) < N0: j = int(start[len(tested)])
            else:
                idx = [s[0] for s in shared]; y = [s[1] for s in shared]
                s = ucb(idx, y, KAPPA[(rnd + p) % 3]) if len(idx) >= 2 else rng.random(K.NC)
                s[list(tested | set(props))] = -np.inf; j = int(np.argmax(s))
            props.append(j); tested.add(j); used += 1
            act = K.R[j].copy(); faulty = (f == BAD and rnd >= ONSET)
            if faulty:
                bad_runs += 1
                if fault == "furnace_offset": act[1] += 100
                else: act[3] = max(0.01, act[3] * 0.1)
            m = K.kmc.measure(act, K.PAR["Mo"], rng); yv = np.array([m["x_top_Se"], m["x_bot_Se"], m["defects"]])
            if key is not None:  # this run is a repeat of a held reading
                n_rep += 1; h = held.pop(key)
                if np.abs(h["y"] - yv).max() > 0.08:
                    strikes[key[1]] += 1
                    if strikes[key[1]] >= 2 and key[1] in active and len(active) > 1:
                        active.remove(key[1]); out_round = out_round or rnd
                else:
                    shared.append((j, h["F"], h["y"])); bad_shared += h["faulty"]
                shared.append((j, m["F"], yv)); bad_shared += faulty
                if arm in dets: mem.append(dets[arm].score(mem, j, yv)[1])
                elif arm == "gp_resid": mem.append((j, yv))
                continue
            flag = False
            if arm in dets or arm == "gp_resid":
                if arm in dets: sc, item = dets[arm].score(mem, j, yv)
                else: item = (j, yv); sc = T1.gp_score(mem, j, yv) if len(mem) >= 5 else 0.0
                flag = len(scores) >= 10 and sc > np.quantile(scores, 0.9); scores.append(sc)
                if not flag: mem.append(item)
            if flag:
                key = (j, f, rnd); held[key] = dict(F=m["F"], y=yv, faulty=faulty); repeats_q.append(key)
            else:
                shared.append((j, m["F"], yv)); bad_shared += faulty
    best = max(T[s[0]] for s in shared)
    return dict(fault=fault, faulty_shared=int(bad_shared), runs_on_faulty=int(bad_runs), repeats=int(n_rep),
                out_of_service_round=out_round, rounds=rnd, gap=float(T.max() - best), n_shared=len(shared))


def run(seed):
    Wd = K.wirings(np.random.default_rng(seed))
    dets = {a: T1.FlyDet(Wd[a], np.random.default_rng(7)) for a in ("fly", "fly_shuf", "fly_rand")}
    return dict(seed=seed, arms={a: run_arm(a, seed, dets) for a in ARMS})


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/t4_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {a: (d["faulty_shared"], d["runs_on_faulty"], d["out_of_service_round"], round(d["gap"], 3)) for a, d in r["arms"].items()}, flush=True)
