#!/usr/bin/env python3
"""S24a: flagging with every fixed-memory baseline given the connectome model's full memory.

S22 gave the "equal memory" nearest neighbour as many values as the connectome model has plastic weights
(5,177 KC weights = 92 stored spectra = 20.6 kB). The model also holds its wiring: 0.37 MB as run in NumPy
(int32 indices, float32 counts) or about 152 kB packed for a microcontroller (uint16 KC and PN indices,
uint8 synapse counts, uint8 KC weights; edge_compare.FlySparse.mcu_bytes). Here each fixed-memory baseline
gets that whole budget.
Streams (identical draws to the paper's):
  s1b   S1b / S7 facility stream, 4,000 readings, four families, laser drop at N/2 (S22b used seeds 1-10,
        s1b_multi seeds 1-20). AUROC new vs normal over readings 300-4,000.
  long  S22a stream, 100,000 readings, no laser drop. AUROC window-weighted over 3,000-100,000 (as S22a).
Both carry the family from the sample log (same draw order), used only by the routed arms.
Detectors (score = novelty, higher = less familiar):
  fly, fly_rec           connectome model (measured wiring), recovery r = 0 / 1e-4 per reading
  fly_ctx2, fly_ctx2_k25 the model told the family (S8 context input, gain 2), 5 % / 2.5 % active (S23 tuned)
  knn_n{n}               nearest neighbour on area-normalised spectra, reservoir of n spectra (float32)
  knnfam_n{n}            one reservoir of n/4 spectra per family, reading routed by the sample log
  ae{h}_b{n}             online autoencoder 56-h-56 (Adam lr 3e-3), reservoir replay buffer of n spectra
  zscore, zscore_fam     per-band running z-score (448 B), one per family (1,792 B)
Budgets (bytes, float32 storage for every baseline as in S22): eq = 20.6 kB, pack = 152 kB, run = 0.37 MB;
n chosen so the detector's nbytes() fits the budget; knn_n1357 = 152 kB at float16 (the packed accounting).
Usage: python s24_fixed_memory.py s1b|long OUT_DIR SEED
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["S1_MODE"] = "multi"
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401  (S24_WCACHE: wiring from one small npz)
import s7_swarm as S7  # noqa: E402
from s5_long import auroc  # noqa: E402
import s22_data_growth as D  # noqa: E402

NF = len(S.FAM)


def stream_long(seed, G):
    """s22_data_growth.stream with the family index kept (same draw order)."""
    rng = np.random.default_rng(seed); fams = list(S.FAM)
    for i in range(D.NTOT):
        f = int(rng.integers(len(fams))); comp = S.FAM[fams[f]]
        w = rng.dirichlet([0.6] * len(comp)); s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); kind = None
        if i >= D.WARM and rng.random() < D.P_NEW:
            kind = S.KINDS[rng.integers(len(S.KINDS))]
            if kind == "MoS2/WSe2_stack": s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike": s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
            else:
                fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        yield S.bands(y, G), kind, f


class FlyCtx(D.FlySparse):
    """FlySparse with the S8 context input (fly2.Fly2 ctx=True, same quarter masks)."""
    def __init__(self, W, Min, ctx_gain=2.0, frac=0.05, seed=0):
        super().__init__(W, Min); self.g = ctx_gain; self.k = int(frac * self.n)
        q = np.array_split(np.random.default_rng(seed).permutation(self.n), NF)
        self.cm = np.zeros((NF, self.n), np.float32)
        for j, i in enumerate(q): self.cm[j, i] = 1

    def step(self, x, f=0):
        p = np.bincount(self.pr, weights=self.pv * x[self.pc], minlength=self.npn)
        d = np.bincount(self.r, weights=self.v * p[self.c], minlength=self.n)
        d = d + self.g * d.std() * self.cm[f]
        a = d >= np.partition(d, -self.k)[-self.k]; s = float(self.w[a].sum() / self.k); self.w[a] *= 1 - self.eta
        return s

    def nbytes(self): return super().nbytes() + self.n // 4  # 2-bit family mask per KC


class One:
    def __init__(self, d): self.d = d
    def step(self, x, f): return self.d.step(x)
    def nbytes(self): return int(self.d.nbytes())


class Route:
    def __init__(self, parts): self.p = parts
    def step(self, x, f): return self.p[f].step(x)
    def nbytes(self): return int(sum(d.nbytes() for d in self.p))


def mcu_bytes(fly):  # edge_compare.FlySparse.mcu_bytes
    return int(len(fly.r) * 5 + fly.n + len(fly.pr) * 5)


def budgets(W, Min, G):
    f = D.FlySparse(W, Min)
    run_b = int(f.nbytes()); pack_b = mcu_bytes(f); eq_b = int(f.n * 4)
    spec = G * 4
    ae_par = {h: 3 * 4 * (G * h + h + h * G + G) for h in (16, 64)}
    knn = sorted({23, 46, 92, 184, 368, eq_b // spec, pack_b // spec, pack_b // (G * 2), run_b // spec})
    ae = {f"ae{h}_b{(b - ae_par[h]) // spec}": (h, (b - ae_par[h]) // spec) for h in (16, 64) for b in (pack_b, run_b)}
    ae["ae64_b92"] = (64, 92)  # S22a ae_res
    ae["ae16_b0"] = (16, 0)    # S22b ae16 (no replay)
    return dict(run=run_b, pack=pack_b, eq=eq_b), knn, ae


def detectors(W, Min, G, seed):
    B, knn, ae = budgets(W, Min, G)
    d = {"fly": One(D.FlySparse(W, Min)), "fly_rec": One(D.FlySparse(W, Min, rec=1e-4)),
         "fly_ctx2": FlyCtx(W, Min, 2.0, 0.05), "fly_ctx2_k25": FlyCtx(W, Min, 2.0, 0.025),
         "zscore": One(D.Z(G)), "zscore_fam": Route([D.Z(G) for _ in range(NF)])}
    for n in knn:
        d[f"knn_n{n}"] = One(D.KNN(G, n, seed=3 + seed))
        d[f"knnfam_n{n}"] = Route([D.KNN(G, max(1, n // NF), seed=3 + seed + 17 * j) for j in range(NF)])
    for name, (h, b) in ae.items():
        d[name] = One(D.AE(G, b, seed, h=h))
    return d, B


def run(mode, seed, out):
    import torch
    torch.set_num_threads(1)
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    dets, B = detectors(W, Min, G, seed)
    only = os.environ.get("S24_ONLY")
    if only: dets = {k: v for k, v in dets.items() if any(k.startswith(p) for p in only.split(","))}
    st = S7.stream(seed, G) if mode == "s1b" else stream_long(seed, G)
    n_tot = S.N if mode == "s1b" else D.NTOT
    sc = {n: np.zeros(n_tot, np.float32) for n in dets}; new = np.zeros(n_tot, bool); t0 = time.time()
    for i, (x, kind, f) in enumerate(st):
        new[i] = kind is not None
        for n, d in dets.items(): sc[n][i] = d.step(x, f)
        if mode == "long" and (i + 1) % 10000 == 0: print(seed, i + 1, f"{time.time() - t0:.0f}s", flush=True)
    res = dict(mode=mode, seed=seed, budgets=B, det={})
    idx = np.arange(n_tot)
    for n, d in dets.items():
        x = sc[n]; r = dict(bytes=int(d.nbytes()))
        if mode == "s1b":
            m = idx >= S.WARM; r["auroc"] = float(auroc(x[m & new], x[m & ~new]))
        else:
            wins = [(3000, 10000), (10000, 30000), (30000, 100000)]
            a = [auroc(x[(idx >= lo) & (idx < hi) & new], x[(idx >= lo) & (idx < hi) & ~new]) for lo, hi in wins]
            r["auroc_windows"] = [float(v) for v in a]
            r["auroc"] = float(np.average(a, weights=[hi - lo for lo, hi in wins]))
            m = idx >= 3000; r["auroc_pooled"] = float(auroc(x[m & new], x[m & ~new]))
        res["det"][n] = r
    tag = os.environ.get("S24_TAG", "")
    json.dump(res, open(f"{out}/s24a_{mode}{'_' + tag if tag else ''}_s{seed}.json", "w"), indent=1)
    print(mode, seed, f"{time.time() - t0:.0f}s", {n: round(v["auroc"], 3) for n, v in res["det"].items()}, flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[2], exist_ok=True); run(sys.argv[1], int(sys.argv[3]), sys.argv[2])
