#!/usr/bin/env python3
"""Timing of the fly reflex layer on an edge controller (Raspberry Pi 5) against the DGX.
Needs only NumPy and the wiring file from t5_edge_export.py; no FlyWire download, no scikit-learn.
Measures, single thread:
  load_s            read the wiring file and build the dense PN->KC matrix
  fault_ms          T1 familiarity score of one reading against a memory of N earlier readings
  avoid_codes_s     KC codes of all 1400 Janus recipes (done once per campaign)
  avoid_ms          T3 avoid value of all 1400 recipes + one avoid-memory update
  peak_rss_MB       peak resident memory
Usage: OPENBLAS_NUM_THREADS=1 python edge_timing.py WIRING_NPZ OUT_JSON
"""
import json, platform, resource, sys, time
import numpy as np

t0 = time.perf_counter(); z = np.load(sys.argv[1])
W = np.zeros(tuple(z["shape"]), np.float32); W[z["r"], z["c"]] = z["w"]; mask = z["mask"].astype(np.float32)
load_s = time.perf_counter() - t0
rng = np.random.default_rng(0); npn = W.shape[1]; k = int(0.05 * W.shape[0])
Er = rng.normal(size=(npn, 4)).astype(np.float32); Ej = rng.normal(size=(npn, 7)).astype(np.float32)
Ea = rng.normal(size=(npn, 5)).astype(np.float32)


def code(V, E):
    d = np.maximum(((np.atleast_2d(V) - 0.5) / 0.3) @ E.T, 0) @ W.T
    thr = np.partition(d, -k, axis=1)[:, -k][:, None]; return d >= thr


def score(mem, x, y):
    cr = code(x, Er)[0]; cj = code(np.r_[x, y], Ej)[0]
    if not mem: return 1.0, (cr, cj)
    CR = np.array([m[0] for m in mem]); CJ = np.array([m[1] for m in mem])
    near = np.argsort(-(CR & cr).sum(1))[:10]
    return 1 - (CJ[near] & cj).sum(1).max() / k, (cr, cj)


res = dict(machine=platform.machine(), node=platform.node(), numpy=np.__version__, n_PN=npn, n_KC=W.shape[0], load_s=load_s, fault_ms={})
for N in (50, 200, 1000):
    mem = []
    for i in range(N):
        mem.append(score(mem, rng.random(4), rng.random(3))[1])
    ts = []
    for _ in range(30):
        t = time.perf_counter(); score(mem, rng.random(4), rng.random(3)); ts.append(time.perf_counter() - t)
    res["fault_ms"][str(N)] = 1e3 * float(np.median(ts))
Xg = rng.random((1400, 5)).astype(np.float32)
t = time.perf_counter(); cd = code(Xg, Ea).astype(np.float32)
res["avoid_codes_s"] = time.perf_counter() - t
av = np.zeros(W.shape[0], np.float32); ts = []
for _ in range(30):
    t = time.perf_counter(); val = (cd * av).sum(1) / cd.sum(1); j = int(rng.integers(1400)); av += (1 - val[j]) * cd[j] * mask
    ts.append(time.perf_counter() - t)
res["avoid_ms"] = 1e3 * float(np.median(ts))
res["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(res, open(sys.argv[2], "w"), indent=1); print(res)
