#!/usr/bin/env python3
"""S25a: does each job need memory? Every method in three conditions.

  frozen    learns during calibration only, then scores without updates
            (flagging: the first WARM = 300 readings; failing-run stop: the first C completed runs, C = 3, 6, 10, 15)
  reset_k   memory cleared every k readings (flagging: k = 100, 1,000; failing-run stop: every 10 completed runs,
            since one campaign has only 30 probes and k = 100 or 1,000 equals online)
  online    full online memory, as in the paper
Dose and furnace end point (s11, s18a) learn only from their calibration flakes; no labelled data arrives
after calibration, so both are frozen by construction (reported, not re-run).

Flagging streams (four families, sample log as the family label):
  s1b    S1b / S7 stream, 4,000 readings, laser drop at N/2 (seeds 1-20). AUROC new vs normal after warm-up; also
         false alarms after the drop: threshold = 95th percentile of the 300 normal readings before the drop; the
         alarm share in successive blocks of 100 normal readings after it; readings until a block is at or below 7.5 %.
  long   S22a stream, 100,000 readings, no drop (seeds 1-10). AUROC over readings 3,000-100,000, window-weighted as S22a.
  shift  new here, 70,000 readings, built from the S1b generator (seeds 1-10):
           genuinely new: cosmic spike or MoS2/WSe2 stack, 2 % of readings after warm-up;
           accepted phase: MoTe2 enters at reading 5,000 as a fifth routine material (10 % of later readings, its
             own family in the log); its first 30 readings are new, those after reading 15,000 are accepted routine;
           rare phase: 1T' MoS2 episodes of 10 readings (one every 5, mixed 0.6-0.9 into a routine spectrum) first
             seen at reading 2,000 and returning 1,000, 10,000 and 60,000 readings later;
           slow drift: the Raman axis shifts linearly from 0 to 3 cm-1 over the stream.
         Threshold: 95th percentile of routine (four-family) scores in readings 300-5,000. Reported: AUROC
         genuinely new vs routine; alarm share on accepted MoTe2, on each rare episode, and on routine readings
         early (300-5,000) and late (60,000-70,000, after 2.6-3 cm-1 of drift).
Methods: connectome model (fly; fly_rec r = 1e-4; fly_ctx2 told the family), per-band z-score pooled and routed,
reservoir nearest neighbour (92 and 184 spectra) pooled and routed (184), small NN (online autoencoder 56-16-56,
Adam 3e-3, no replay: S22b ae16).
Failing-run stop (s4_insitu.run_arm, seeds 1-20): fly (plain), fly_best (paper setting), logistic regression C = 1
and 10, nearest neighbour, small NN (MLP 64x64, alpha 1e-4: S23 j1 tuned).
Usage: python s25_memory.py s1b|long|shift OUT_DIR SEED   |   python s25_memory.py stop OUT_DIR SEED
"""
import copy, json, os, sys, time
MODE = sys.argv[1] if __name__ == "__main__" else "s1b"
os.environ["S1_MODE"] = "janus" if MODE == "stop" else "multi"
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401  (S24_WCACHE: wiring from one small npz)
from s5_long import auroc  # noqa: E402

WARM = 300


# ------------------------------------------------------------------ detectors with score-only and reset
class Fly:
    def __init__(self, W, Min, rec=0.0, ctx=None, nf=4, frac=0.05, eta=0.3):
        r, c = np.nonzero(W); self.r, self.c, self.v = r.astype(np.int32), c.astype(np.int32), W[r, c].astype(np.float32)
        pr, pc = np.nonzero(Min); self.pr, self.pc, self.pv = pr.astype(np.int32), pc.astype(np.int32), Min[pr, pc].astype(np.float32)
        self.n, self.npn = W.shape[0], Min.shape[0]; self.k = int(frac * self.n); self.eta, self.rec, self.ctx = eta, rec, ctx
        if ctx:
            q = np.array_split(np.random.default_rng(0).permutation(self.n), 4)   # fly2.Fly2 masks (seed 0)
            self.cm = np.zeros((max(nf, 4), self.n), np.float32)
            for j, i in enumerate(q): self.cm[j, i] = 1
            for j in range(4, nf):   # a family added later gets its own random quarter
                self.cm[j, np.random.default_rng(100 + j).choice(self.n, self.n // 4, replace=False)] = 1
        self.reset()

    def reset(self): self.w = np.ones(self.n, np.float32)

    def step(self, x, f=0, learn=True):
        p = np.bincount(self.pr, weights=self.pv * x[self.pc], minlength=self.npn)
        d = np.bincount(self.r, weights=self.v * p[self.c], minlength=self.n)
        if self.ctx: d = d + self.ctx * d.std() * self.cm[f]
        a = d >= np.partition(d, -self.k)[-self.k]; s = float(self.w[a].sum() / self.k)
        if learn:
            self.w[a] *= 1 - self.eta
            if self.rec: self.w += self.rec * (1 - self.w)
        return s


class Z:
    def __init__(self, G): self.G = G; self.reset()
    def reset(self): self.n, self.m, self.v = 0, np.zeros(self.G), np.zeros(self.G)

    def step(self, x, f=0, learn=True):
        v = x / (x.sum() + 1e-9)
        s = float(np.max(np.abs(v - self.m) / np.sqrt(self.v / max(self.n - 1, 1) + 1e-12))) if self.n > 2 else 0.0
        if learn: self.n += 1; d = v - self.m; self.m += d / self.n; self.v += d * (v - self.m)
        return s


class KNN:
    def __init__(self, G, budget, seed): self.G, self.budget, self.seed = G, budget, seed; self.reset()
    def reset(self): self.mem = np.zeros((self.budget, self.G), np.float32); self.n = 0; self.seen = 0; self.rng = np.random.default_rng(self.seed)

    def step(self, x, f=0, learn=True):
        v = (x / (x.sum() + 1e-9)).astype(np.float32)
        s = float(np.sqrt(((self.mem[:self.n] - v) ** 2).sum(1)).min()) if self.n else 1.0
        if learn:
            self.seen += 1
            if self.n < self.budget: self.mem[self.n] = v; self.n += 1
            else:
                j = self.rng.integers(self.seen)
                if j < self.budget: self.mem[j] = v
        return s


class AE:
    """56-16-56 online autoencoder, Adam 3e-3, no replay (s22_data_growth.AE(G, 0, seed, h=16))."""
    def __init__(self, G, seed, h=16, lr=3e-3): self.G, self.seed, self.h, self.lr = G, seed, h, lr; self.reset()

    def reset(self):
        import torch
        torch.manual_seed(self.seed); self.t = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(self.G, self.h), torch.nn.ReLU(), torch.nn.Linear(self.h, self.G))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def step(self, x, f=0, learn=True):
        t = self.t; vt = t.from_numpy((x / (x.mean() + 1e-12)).astype(np.float32))
        if not learn:
            with t.no_grad(): return float(((self.net(vt[None])[0] - vt) ** 2).mean())
        loss = ((self.net(vt[None])[0] - vt) ** 2).mean(); s = float(loss.detach())
        self.opt.zero_grad(); loss.backward(); self.opt.step(); return s


class Route:
    def __init__(self, parts): self.p = parts
    def reset(self): [d.reset() for d in self.p]
    def step(self, x, f=0, learn=True): return self.p[f].step(x, 0, learn)


class Cond:
    def __init__(self, det, cond):
        self.d, self.cond = det, cond; self.k = int(cond[5:]) if cond.startswith("reset") else None

    def step(self, i, x, f):
        if self.k and i and i % self.k == 0: self.d.reset()
        return self.d.step(x, f, learn=(i < WARM) if self.cond == "frozen" else True)


CONDS = ["frozen", "reset100", "reset1000", "online"]


def methods(W, Min, G, seed, nf):
    return {"fly": lambda: Fly(W, Min), "fly_rec": lambda: Fly(W, Min, rec=1e-4),
            "fly_ctx2": lambda: Fly(W, Min, ctx=2.0, nf=nf),
            "zscore": lambda: Z(G), "zscore_fam": lambda: Route([Z(G) for _ in range(nf)]),
            "knn_n92": lambda: KNN(G, 92, 3 + seed), "knn_n184": lambda: KNN(G, 184, 3 + seed),
            "knnfam_n184": lambda: Route([KNN(G, 46, 3 + seed + 17 * j) for j in range(nf)]),
            "ae16": lambda: AE(G, seed)}


# ------------------------------------------------------------------ streams
def stream_long(seed, G):
    import s24_fixed_memory as A
    return ((x, "new" if k else "", f) for x, k, f in A.stream_long(seed, G))


SH_N, SH_MOTE2, SH_ACCEPT, SH_RARE, SH_DRIFT = 70_000, 5_000, 15_000, [2_000, 3_000, 12_000, 62_000], 3.0
RARE_LAB = {2_000: "rare_first", 3_000: "rare_ret_1e3", 12_000: "rare_ret_1e4", 62_000: "rare_ret_6e4"}


def stream_shift(seed, G):
    rng = np.random.default_rng(50_000 + seed); fams = list(S.FAM); mote2 = S.lines([(171, 1.0, 4), (234, 1.2, 4), (289, 0.4, 5)])
    t1 = S.NEWS["1T'_MoS2"]; rare = {}
    for t0 in SH_RARE:
        for j in range(10): rare[t0 + 5 * j] = RARE_LAB[t0]
    n_mote2 = 0
    for i in range(SH_N):
        lab = ""
        if i not in rare and i >= SH_MOTE2 and rng.random() < 0.10:
            f = 4; s = mote2 * rng.uniform(0.8, 1.2); n_mote2 += 1
            lab = "new" if n_mote2 <= 30 else ("accepted" if i >= SH_ACCEPT else "mote2_mid")
        else:
            f = int(rng.integers(4)); comp = S.FAM[fams[f]]
            w = rng.dirichlet([0.6] * len(comp)); s = sum(wi * S.BASE[c] for wi, c in zip(w, comp))
            if i in rare:   # episode readings are never MoTe2 or another new event
                fr = rng.uniform(0.6, 0.9); s = (1 - fr) * s + fr * t1; lab = rare[i]
            elif i >= WARM and rng.random() < 0.02:
                lab = "new"
                if rng.random() < 0.5: s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
                else: s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        d = SH_DRIFT * i / SH_N
        if d: y = np.interp(S.NU - d, S.NU, y)
        yield S.bands(y, G), lab, f


def stream_s1b(seed, G):
    import s7_swarm as S7
    return ((x, "new" if k else "", f) for x, k, f in S7.stream(seed, G))


def run_flag(mode, seed, out):
    import torch
    torch.set_num_threads(1)
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    nf = 5 if mode == "shift" else 4
    M = methods(W, Min, G, seed, nf)
    dets = {f"{m}|{c}": Cond(mk(), c) for m, mk in M.items() for c in CONDS}
    st = dict(s1b=stream_s1b, long=stream_long, shift=stream_shift)[mode](seed, G)
    n = dict(s1b=S.N, long=100_000, shift=SH_N)[mode]
    sc = {k: np.zeros(n, np.float32) for k in dets}; lab = np.empty(n, object); t0 = time.time()
    for i, (x, l, f) in enumerate(st):
        lab[i] = l
        for k, d in dets.items(): sc[k][i] = d.step(i, x, f)
        if (i + 1) % 10_000 == 0: print(mode, seed, i + 1, f"{time.time() - t0:.0f}s", flush=True)
    idx = np.arange(n); res = dict(mode=mode, seed=seed, det={})
    for k in dets:
        x = sc[k]; r = {}
        if mode == "s1b":
            new = lab == "new"; post = idx >= WARM; r["auroc"] = float(auroc(x[post & new], x[post & ~new]))
            h = n // 2; nor = ~new
            thr = np.quantile(x[nor & (idx >= h - 300) & (idx < h)], 0.95)
            after = np.where(nor & (idx >= h))[0]; blocks = [float(np.mean(x[after[j:j + 100]] > thr)) for j in range(0, len(after) - 99, 100)]
            r["fa_blocks_after_drop"] = blocks
            ok = [j for j, b in enumerate(blocks) if b <= 0.075]
            r["readings_to_recover"] = int(after[ok[0] * 100] - h) if ok else None
        elif mode == "long":
            new = lab == "new"; wins = [(3000, 10000), (10000, 30000), (30000, 100000)]
            a = [auroc(x[(idx >= lo) & (idx < hi) & new], x[(idx >= lo) & (idx < hi) & ~new]) for lo, hi in wins]
            r["auroc"] = float(np.average(a, weights=[hi - lo for lo, hi in wins]))
            r["auroc_blocks_500"] = [float(auroc(x[(idx >= lo) & (idx < lo + 500) & new], x[(idx >= lo) & (idx < lo + 500) & ~new]))
                                     for lo in range(WARM, 10_300, 500)]
        else:
            rout = lab == ""; post = idx >= WARM; new = lab == "new"
            r["auroc"] = float(auroc(x[post & new], x[post & rout]))
            thr = np.quantile(x[rout & post & (idx < SH_MOTE2)], 0.95)
            for name, m in (("alarm_accepted", lab == "accepted"), ("alarm_routine_early", rout & post & (idx < SH_MOTE2)),
                            ("alarm_routine_late", rout & (idx >= 60_000)), ("recall_new_late", new & (idx >= 50_000))):
                r[name] = float(np.mean(x[m] > thr))
            for l in RARE_LAB.values(): r["alarm_" + l] = float(np.mean(x[lab == l] > thr))
        res["det"][k] = r
    json.dump(res, open(f"{out}/s25a_{mode}_s{seed}.json", "w"), indent=1)
    print(mode, seed, f"{time.time() - t0:.0f}s done", flush=True)


# ------------------------------------------------------------------ failing-run stop
STOP_CONDS = ["frozen3", "frozen6", "frozen10", "frozen15", "reset10", "online"]


class CondStop:
    def __init__(self, mk, cond):
        self.mk, self.d, self.nl = mk, mk(), 0
        self.C = int(cond[6:]) if cond.startswith("frozen") else None
        self.R = int(cond[5:]) if cond.startswith("reset") else None

    def stop(self, x): return self.d.stop(x)

    def learn(self, x, fail):
        if self.C is not None and self.nl >= self.C: return
        self.d.learn(x, fail); self.nl += 1
        if self.R and self.nl % self.R == 0: self.d = self.mk()


def run_stop(seed, out):
    import s4_insitu as S4
    Wd, Min, G = S4.H.wirings(seed)

    def clf(make):
        class Clf(S4.KnnStop):
            def stop(self, x):
                if len({f for _, f in self.mem}) < 2: return False
                X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
                return make().fit(X * 50, y).predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5
        return Clf
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    M = {"fly": lambda: S4.FlyStop(Wd["fly"], Min, {}),
         "fly_best": lambda: S4.FlyStop(Wd["fly"], Min, S4.H.default_full_cfg()),
         "logreg_C1": clf(lambda: LogisticRegression(C=1.0, max_iter=500)),
         "logreg_C10": clf(lambda: LogisticRegression(C=10.0, max_iter=500)),
         "knn": S4.KnnStop,
         "mlp": clf(lambda: MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=2000, random_state=0))}
    res = dict(mode="stop", seed=seed, det={})
    for m, mk in M.items():
        for c in STOP_CONDS:
            res["det"][f"{m}|{c}"] = S4.run_arm("gp+" + m, lambda mk=mk, c=c: CondStop(mk, c), seed, G)
    res["det"]["gp_alone"] = S4.run_arm("gp", None, seed, G)
    json.dump(res, open(f"{out}/s25a_stop_s{seed}.json", "w"), indent=1)
    print("stop", seed, {k: v["lost"] for k, v in res["det"].items()}, flush=True)


if __name__ == "__main__":
    out = sys.argv[2]; os.makedirs(out, exist_ok=True); seed = int(sys.argv[3])
    run_stop(seed, out) if MODE == "stop" else run_flag(MODE, seed, out)
