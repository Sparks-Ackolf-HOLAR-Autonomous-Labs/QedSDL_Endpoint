#!/usr/bin/env python3
"""S22a: how the memory of each detector grows with the data it has seen.

One long S1b-type Raman stream (four material families, 5 % new events after warm-up, no laser drop so
that every window is drawn from the same conditions), 100,000 readings. At checkpoints N = 300, 1k, 3k,
10k, 30k, 100k each detector reports the bytes it holds (float32 storage), the median time per reading
over the last 200 readings, and the AUROC of new vs normal readings inside the window that ends at N.
Detectors (single thread):
  fly          connectome model as in S1 (measured PN->KC synapses as index lists, 5,177 KC weights)
  fly_rec      same with slow synaptic recovery r = 1e-4 per reading (the S5 setting)
  zscore       per-band running mean and variance
  knn_budget   nearest neighbour, reservoir with the fly's number of stored values
  knn_full     nearest neighbour that keeps every reading (memory and search grow with N)
  ae_res       online autoencoder (56-64-56), reservoir replay of 92 readings (fixed size)
  ae_full      same autoencoder, replay buffer keeps every reading (grows with N)
The Gaussian-process planner is measured separately (gp_probe): refit and predict time and memory with
n = 100, 300, 1,000, 3,000 stored readings; beyond that the n^2 memory is computed, not measured.
Seeds 1-5 ran before fly_rec was added; fly_rec for them was run alone (S22_ONLY=fly_rec S22_TAG=rec); the
stream and the other detectors do not depend on which detectors run.
Usage: python s22_data_growth.py OUT_DIR SEED   |   python s22_data_growth.py --gp OUT_JSON
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["S1_MODE"] = "multi"
import s1_spectra as S  # noqa: E402
from s5_long import auroc  # noqa: E402

NTOT, WARM, P_NEW = 100_000, 300, 0.05
CHECK = [300, 1000, 3000, 10_000, 30_000, 100_000]


def stream(seed, G):
    """S1b generator without the instrument change (S.stream drops the laser power at N/2)."""
    rng = np.random.default_rng(seed); fams = list(S.FAM)
    for i in range(NTOT):
        comp = S.FAM[fams[rng.integers(len(fams))]]
        w = rng.dirichlet([0.6] * len(comp)); s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); kind = None
        if i >= WARM and rng.random() < P_NEW:
            kind = S.KINDS[rng.integers(len(S.KINDS))]
            if kind == "MoS2/WSe2_stack": s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike": s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
            else:
                f = rng.uniform(0.4, 0.8); s = (1 - f) * s + f * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        yield S.bands(y, G), kind


class FlySparse:
    def __init__(self, W, Min, eta=0.3, rec=0.0):
        self.rec = rec  # slow synaptic recovery per reading, as in S5 (w += rec * (1 - w))
        r, c = np.nonzero(W); self.r, self.c, self.v = r.astype(np.int32), c.astype(np.int32), W[r, c].astype(np.float32)
        self.n = W.shape[0]; self.k = int(S.K_FRAC * self.n); self.w = np.ones(self.n, np.float32); self.eta = eta
        pr, pc = np.nonzero(Min); self.pr, self.pc, self.pv = pr.astype(np.int32), pc.astype(np.int32), Min[pr, pc].astype(np.float32)
        self.npn = Min.shape[0]

    def step(self, x):
        p = np.bincount(self.pr, weights=self.pv * x[self.pc], minlength=self.npn)
        d = np.bincount(self.r, weights=self.v * p[self.c], minlength=self.n)
        a = d >= np.partition(d, -self.k)[-self.k]; s = float(self.w[a].sum() / self.k); self.w[a] *= 1 - self.eta
        if self.rec: self.w += self.rec * (1 - self.w)
        return s

    def nbytes(self):  # wiring (int32 indices + float32 counts) and the KC weights; fixed
        return sum(a.nbytes for a in (self.r, self.c, self.v, self.pr, self.pc, self.pv, self.w))


class Z:
    def __init__(self, G): self.n, self.m, self.v = 0, np.zeros(G), np.zeros(G)

    def step(self, x):
        v = x / (x.sum() + 1e-9)
        s = float(np.max(np.abs(v - self.m) / np.sqrt(self.v / max(self.n - 1, 1) + 1e-12))) if self.n > 2 else 0.0
        self.n += 1; d = v - self.m; self.m += d / self.n; self.v += d * (v - self.m); return s

    def nbytes(self): return 2 * self.m.size * 4


class KNN:
    """Nearest neighbour on area-normalised spectra; budget None keeps every reading (array doubles as it fills)."""
    def __init__(self, G, budget=None, seed=3):
        self.budget = budget; self.mem = np.zeros((budget or 1024, G), np.float32); self.n = 0; self.seen = 0
        self.rng = np.random.default_rng(seed)

    def step(self, x):
        v = (x / (x.sum() + 1e-9)).astype(np.float32)
        s = float(np.sqrt(((self.mem[:self.n] - v) ** 2).sum(1)).min()) if self.n else 1.0
        self.seen += 1
        if self.budget is None:
            if self.n == len(self.mem): self.mem = np.concatenate([self.mem, np.zeros_like(self.mem)])
            self.mem[self.n] = v; self.n += 1
        elif self.n < self.budget: self.mem[self.n] = v; self.n += 1
        else:
            j = self.rng.integers(self.seen)
            if j < self.budget: self.mem[j] = v
        return s

    def nbytes(self): return self.n * self.mem.shape[1] * 4


class AE:
    """56-64-56 online autoencoder (Adam, lr 3e-3), 8 replayed readings per step."""
    def __init__(self, G, replay, seed, h=64, lr=3e-3):
        import torch
        torch.manual_seed(seed); self.t = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(G, h), torch.nn.ReLU(), torch.nn.Linear(h, G))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = replay; self.buf = np.zeros((replay or 1024, G), np.float32); self.n = 0; self.seen = 0
        self.rng = np.random.default_rng(seed + 777)

    def step(self, x):
        t = self.t; v = (x / (x.mean() + 1e-12)).astype(np.float32); vt = t.from_numpy(v)
        loss = ((self.net(vt[None])[0] - vt) ** 2).mean(); s = float(loss.detach())
        if self.n:
            idx = self.rng.choice(self.n, min(8, self.n), replace=False); R = t.from_numpy(self.buf[idx])
            loss = loss + ((self.net(R) - R) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step(); self.seen += 1
        if self.replay is None:
            if self.n == len(self.buf): self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)])
            self.buf[self.n] = v; self.n += 1
        elif self.n < self.replay: self.buf[self.n] = v; self.n += 1
        else:
            j = int(self.rng.integers(self.seen))
            if j < self.replay: self.buf[j] = v
        return s

    def nbytes(self):  # weights + Adam moments (3 copies) + replay buffer
        p = sum(q.numel() for q in self.net.parameters())
        return 3 * p * 4 + self.n * self.buf.shape[1] * 4


def run(seed, out):
    import torch
    torch.set_num_threads(1)
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn)
    fly = FlySparse(W, Min); budget = W.shape[0] // G
    dets = {"fly": fly, "fly_rec": FlySparse(W, Min, rec=1e-4), "zscore": Z(G), "knn_budget": KNN(G, budget),
            "knn_full": KNN(G), "ae_res": AE(G, budget, seed), "ae_full": AE(G, None, seed)}
    only = os.environ.get("S22_ONLY")  # comma-separated subset, written to s22a_<tag>_s<seed>.json
    if only: dets = {k: v for k, v in dets.items() if k in only.split(",")}
    sc = {n: np.zeros(NTOT, np.float32) for n in dets}; tt = {n: np.zeros(NTOT, np.float32) for n in dets}
    new = np.zeros(NTOT, bool); rec = {n: [] for n in dets}; c0 = 0; t_start = time.time()
    for i, (x, kind) in enumerate(stream(seed, G)):
        new[i] = kind is not None
        for n, d in dets.items():
            t0 = time.perf_counter(); sc[n][i] = d.step(x); tt[n][i] = time.perf_counter() - t0
        if i + 1 in CHECK:
            lo = max(c0, WARM); m = np.arange(NTOT) < i + 1; m[:lo] = False
            for n, d in dets.items():
                a = auroc(sc[n][m & new], sc[n][m & ~new]) if (m & new).sum() > 5 else None
                rec[n].append(dict(N=i + 1, bytes=int(d.nbytes()), us_per_reading=float(1e6 * np.median(tt[n][max(0, i - 199):i + 1])),
                                   auroc_window=a, window=[int(lo), i + 1]))
            c0 = i + 1
            print(seed, i + 1, {n: (rec[n][-1]["bytes"], round(rec[n][-1]["us_per_reading"]), round(rec[n][-1]["auroc_window"] or 0, 3)) for n in dets},
                  f"{time.time() - t_start:.0f}s", flush=True)
    json.dump(dict(seed=seed, N=NTOT, budget_spectra=int(budget), G=int(G), det=rec), open(f"{out}/s22a_{os.environ['S22_TAG'] + '_' if only else ''}s{seed}.json", "w"), indent=1)


def gp_probe(path):
    """GP planner (RBF, as in edge_compare): refit + predict on n stored readings; memory = X + K + L (float64)."""
    rng = np.random.default_rng(0); res = {}
    for n in (100, 300, 1000, 3000):
        X = rng.random((n, 56)); X /= X.sum(1, keepdims=True); y = rng.random(n); v = X[0]
        ts = []
        for _ in range(3 if n >= 1000 else 10):
            t = time.perf_counter()
            d = ((X[:, None, :] - X[None]) ** 2).sum(-1); K = np.exp(-d / 0.002) + 1e-4 * np.eye(n)
            L = np.linalg.cholesky(K); a = np.linalg.solve(L.T, np.linalg.solve(L, y))
            ks = np.exp(-((X - v) ** 2).sum(1) / 0.002); _ = ks @ a; w = np.linalg.solve(L, ks)
            ts.append(time.perf_counter() - t)
        res[n] = dict(s_per_refit=float(np.median(ts)), bytes=int(X.nbytes + 2 * n * n * 8))
        print("gp", n, res[n], flush=True)
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--gp":
        gp_probe(sys.argv[2]); sys.exit()
    os.makedirs(sys.argv[1], exist_ok=True); run(int(sys.argv[2]), sys.argv[1])
