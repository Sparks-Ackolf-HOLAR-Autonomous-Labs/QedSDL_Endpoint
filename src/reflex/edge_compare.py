#!/usr/bin/env python3
"""Edge-compute specification of the fly reflex layer against the engineered tools used in this work.

Runs on any machine with NumPy (no scikit-learn, no SciPy, no FlyWire download): the DGX and the
Raspberry Pi 5 (monsterpi). Input: the wiring file from s5_export.py and a file of 2,000 Raman band
vectors (56 bands) from the S1b stream (edge_inputs.npz, written by --make-inputs on the DGX).
For each method, single thread, it measures the time of one decision (median of 500 readings after
warm-up), the bytes the method must keep in memory, and the bytes it would need on a microcontroller
(8-bit weights, 16-bit indices). Methods:
  fly_dense     connectome model as in S1-S8: dense float32 PN->KC matrix (5,177 x 685)
  fly_sparse    same model, only the measured synapses stored (index lists); identical output
  knn_budget    equal-memory nearest neighbour (5,152 stored numbers), as in S1b
  zscore        per-band running mean and variance (112 numbers)
  logreg        logistic regression on 56 bands, predict + one online gradient step
  ridge         ridge regression predict (S11)
  gp_n          Gaussian process (RBF kernel) refit and predict with n stored readings, n = 30, 100, 300
                (the S4 / Janus planner's cost per decision grows as n^3)
Usage: python edge_compare.py WIRING_NPZ INPUTS_NPZ OUT_JSON
       python edge_compare.py --make-inputs INPUTS_NPZ        (DGX, needs s1_spectra)
"""
import json, os, platform, resource, sys, time
import numpy as np


def med_us(f, X, reps=500, warm=200):
    for x in X[:warm]: f(x)
    ts = []
    for x in X[warm:warm + reps]:
        t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
    return 1e6 * float(np.median(ts))


class FlyDense:
    def __init__(self, W, Min):
        self.W, self.M = W, Min; self.k = int(0.05 * W.shape[0]); self.w = np.ones(W.shape[0], np.float32)
    def code(self, x):
        d = self.W @ (self.M @ x); return d >= np.partition(d, -self.k)[-self.k]
    def step(self, x):
        c = self.code(x); s = self.w[c].sum() / self.k; self.w[c] *= 0.7; return s
    def nbytes(self): return self.W.nbytes + self.M.nbytes + self.w.nbytes
    def mcu_bytes(self): return None  # dense matrix (3.5 M entries) does not fit a microcontroller


class FlySparse(FlyDense):
    def __init__(self, r, c, w, n_kc, Min):
        self.r, self.c, self.v, self.n = r.astype(np.int32), c.astype(np.int32), w.astype(np.float32), n_kc
        self.M = Min; self.k = int(0.05 * n_kc); self.w = np.ones(n_kc, np.float32)
        # glomerulus -> PN map is sparse too: each PN reads one band (or the mean of three)
        pr, pc = np.nonzero(Min); self.pr, self.pc, self.pv = pr.astype(np.int32), pc.astype(np.int32), Min[pr, pc]
        self.npn = Min.shape[0]
    def code(self, x):
        p = np.bincount(self.pr, weights=self.pv * x[self.pc], minlength=self.npn)
        d = np.bincount(self.r, weights=self.v * p[self.c], minlength=self.n)
        return d >= np.partition(d, -self.k)[-self.k]
    def nbytes(self): return self.r.nbytes + self.c.nbytes + self.v.nbytes + self.pr.nbytes + self.pc.nbytes + self.pv.nbytes + self.w.nbytes
    def mcu_bytes(self):  # uint16 KC index, uint16 PN index, uint8 synapse count; uint8 KC->MBON weight; PN map
        return int(len(self.r) * 5 + self.n + len(self.pr) * 5)


class KNNB:
    def __init__(self, G, budget):
        self.mem = np.zeros((budget // G, G), np.float32); self.n = 0; self.rng = np.random.default_rng(3); self.seen = 0
    def step(self, x):
        v = x / (x.sum() + 1e-9)
        s = float(np.sqrt(((self.mem[:max(self.n, 1)] - v) ** 2).sum(1)).min())
        self.seen += 1
        if self.n < len(self.mem): self.mem[self.n] = v; self.n += 1
        else:
            j = self.rng.integers(self.seen)
            if j < len(self.mem): self.mem[j] = v
        return s
    def nbytes(self): return self.mem.nbytes
    def mcu_bytes(self): return self.mem.size * 2  # float16


class Z:
    def __init__(self, G): self.n, self.m, self.v = 0, np.zeros(G), np.zeros(G)
    def step(self, x):
        v = x / (x.sum() + 1e-9)
        s = float(np.max(np.abs(v - self.m) / np.sqrt(self.v / max(self.n - 1, 1) + 1e-12))) if self.n > 2 else 0.0
        self.n += 1; d = v - self.m; self.m += d / self.n; self.v += d * (v - self.m); return s
    def nbytes(self): return self.m.nbytes + self.v.nbytes
    def mcu_bytes(self): return self.m.size * 8


class LogReg:
    def __init__(self, G): self.b = np.zeros(G + 1)
    def step(self, x):
        v = np.r_[x / (x.sum() + 1e-9) * 50, 1.0]; p = 1 / (1 + np.exp(-v @ self.b))
        self.b += 0.01 * ((p < 0.5) - p) * v; return p
    def nbytes(self): return self.b.nbytes
    def mcu_bytes(self): return self.b.size * 4


class Ridge(LogReg):
    def step(self, x): return float(np.r_[x / (x.sum() + 1e-9), 1.0] @ self.b)


class GP:
    def __init__(self, X, y): self.X, self.y = X, y
    def step(self, x):
        X = self.X / (self.X.sum(1, keepdims=True) + 1e-9); v = x / (x.sum() + 1e-9)
        d = ((X[:, None, :] - X[None]) ** 2).sum(-1); K = np.exp(-d / 0.002) + 1e-4 * np.eye(len(X))
        L = np.linalg.cholesky(K); a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        ks = np.exp(-((X - v) ** 2).sum(1) / 0.002); mu = ks @ a; w = np.linalg.solve(L, ks)
        return mu + 2 * np.sqrt(max(1 - w @ w, 0))
    def nbytes(self): n = len(self.X); return self.X.nbytes + 2 * n * n * 8
    def mcu_bytes(self): n = len(self.X); return self.X.size * 4 + n * n * 4


def make_inputs(path):
    os.environ["S1_MODE"] = "multi"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import s1_spectra as S
    S.N = 2000
    X = np.array([x for x, _ in S.stream(1, 56)], np.float32)
    np.savez_compressed(path, X=X)


if __name__ == "__main__":
    if sys.argv[1] == "--make-inputs":
        make_inputs(sys.argv[2]); sys.exit()
    z = np.load(sys.argv[1]); X = np.load(sys.argv[2])["X"].astype(np.float64); G = X.shape[1]
    shape = tuple(z["shape"]); Min = z["Min"].astype(np.float32)
    W = np.zeros(shape, np.float32); W[z["r"], z["c"]] = z["w"]
    rng = np.random.default_rng(0)
    m = {"fly_dense": FlyDense(W, Min), "fly_sparse": FlySparse(z["r"], z["c"], z["w"], shape[0], Min.astype(np.float64)),
         "knn_budget": KNNB(G, 5152), "zscore": Z(G), "logreg": LogReg(G), "ridge": Ridge(G)}
    for n in (30, 100, 300):
        m[f"gp_{n}"] = GP(X[rng.choice(len(X), n, replace=False)], rng.random(n))
    # identical output check for the two fly implementations
    same = all((m["fly_dense"].code(x) == m["fly_sparse"].code(x.astype(np.float64))).all() for x in X[:50])
    res = dict(machine=platform.machine(), node=platform.node(), numpy=np.__version__, python=platform.python_version(),
               n_synapse_entries=int(len(z["r"])), fly_dense_equals_sparse=bool(same), methods={})
    for n, d in m.items():
        reps = 30 if n.startswith("gp") else 500
        us = med_us(d.step, X, reps=reps, warm=20 if n.startswith("gp") else 200)
        res["methods"][n] = dict(us_per_decision=us, model_bytes=int(d.nbytes()), mcu_bytes=d.mcu_bytes())
        print(n, round(us, 1), "us", d.nbytes(), flush=True)
    res["peak_rss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    json.dump(res, open(sys.argv[3], "w"), indent=1)
