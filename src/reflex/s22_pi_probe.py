#!/usr/bin/env python3
"""S22d: time per decision of the growing methods on the Raspberry Pi 5 (NumPy only, single thread).

Nearest neighbour that keeps every reading: search over n stored 56-band readings (float32),
n = 1e3, 1e4, 1e5, 1e6. Gaussian-process planner: refit + predict with n = 100, 300, 1,000, 2,000.
The fixed-size methods (connectome model, z-score, reservoir kNN) were timed on the Pi in edge_compare.
Usage: python3 s22_pi_probe.py OUT_JSON
"""
import json, os, platform, sys, time
import numpy as np

rng = np.random.default_rng(0); res = dict(machine=platform.machine(), node=platform.node(), knn_full={}, gp={})
for n in (1000, 10_000, 100_000, 1_000_000):
    M = rng.random((n, 56), dtype=np.float32); M /= M.sum(1, keepdims=True); ts = []
    for i in range(30 if n < 1_000_000 else 8):
        v = M[i] + 1e-4; t = time.perf_counter(); float(np.sqrt(((M - v) ** 2).sum(1)).min()); ts.append(time.perf_counter() - t)
    res["knn_full"][n] = dict(ms=1e3 * float(np.median(ts)), bytes=int(M.nbytes)); print("knn", n, res["knn_full"][n], flush=True)
    del M
for n in (100, 300, 1000, 2000):
    X = rng.random((n, 56)); X /= X.sum(1, keepdims=True); y = rng.random(n); v = X[0]; ts = []
    for _ in range(3 if n >= 1000 else 10):
        t = time.perf_counter()
        d = ((X[:, None, :] - X[None]) ** 2).sum(-1); K = np.exp(-d / 0.002) + 1e-4 * np.eye(n)
        L = np.linalg.cholesky(K); a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        ks = np.exp(-((X - v) ** 2).sum(1) / 0.002); _ = ks @ a; np.linalg.solve(L, ks); ts.append(time.perf_counter() - t)
    res["gp"][n] = dict(s=float(np.median(ts)), bytes=int(X.nbytes + 2 * n * n * 8)); print("gp", n, res["gp"][n], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
