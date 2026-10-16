#!/usr/bin/env python3
"""S5: long Raman streams, 10^5 readings, at a shared 2D-semiconductor facility (S1b families).

A Raman mapping stage on a wafer-scale TMD line produces of order 10^5 spectra per week. Does the fly
circuit keep its memory, time per reading and detection quality flat over that many readings, without
a refit? Stream as S1b (four families, 5 % new events: 1T' MoS2, MoTe2, cosmic spike, MoS2/WSe2 stack),
laser drop to 1/3 at reading 50,000. Measures per block of 10^4 readings: AUROC, false alarms on normal
readings at the threshold of the first scored block, ms per reading; stored numbers at the end.
Detectors: fly (plain), fly_best (mechanisms that passed the S2/S3 tuning rule), fly_recover (plain + slow synaptic recovery,
w += 1e-4 (1 - w) per reading), knn_budget (reservoir of equal memory), zscore.
Needs NumPy only plus the wiring file from s5_export (so it also runs on the Raspberry Pi 5).
Usage: python s5_long.py WIRING_NPZ OUT_JSON SEED [N] [DETECTORS]
"""
import json, os, platform, resource, sys, time
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402  (numpy only at import)
from fly2 import Fly2  # noqa: E402

FAMS = list(S.FAM)


class Recover(Fly2):
    def step(self, x, f=0):
        s = super().step(x, f); self.w += 1e-4 * (1 - self.w); return s


def gen(seed, G, N):
    rng = np.random.default_rng(seed)
    for i in range(N):
        comp = S.FAM[FAMS[rng.integers(len(FAMS))]]; w = rng.dirichlet([0.6] * len(comp))
        s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); kind = None
        if i >= S.WARM and rng.random() < S.P_NEW:
            kind = S.KINDS[rng.integers(len(S.KINDS))]
            if kind == "MoS2/WSe2_stack": s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike": s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
            else: fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        if i >= N // 2: scale /= 3
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        yield S.bands(y, G), kind


def auroc(pos, neg):
    """Mann-Whitney AUROC without scikit-learn."""
    a = np.r_[pos, neg]; r = np.empty(len(a)); o = a.argsort(kind="mergesort"); r[o] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True); cs = np.cumsum(cnt)
    r = ((cs - cnt + 1 + cs) / 2)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    z = np.load(sys.argv[1]); out = sys.argv[2]; seed = int(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
    W = np.zeros(tuple(z["shape"]), np.float32); W[z["r"], z["c"]] = z["w"]; Min = z["Min"].astype(np.float32)
    full = json.loads(str(z["full_cfg"])); G = Min.shape[1]
    allowed = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    mk = {"fly": lambda: Fly2(W, Min), "fly_best": lambda: Fly2(W, Min, **full), "fly_recover": lambda: Recover(W, Min),
          "knn_budget": lambda: S.KNN(True, int(W.shape[0] // G), np.random.default_rng(3)), "zscore": lambda: S.Z(G)}
    dets = {n: f() for n, f in mk.items() if allowed is None or n in allowed}
    B = 10000; sc = {n: [] for n in dets}; ms = {n: [] for n in dets}; kinds = []; tb = {n: 0.0 for n in dets}
    for i, (x, kind) in enumerate(gen(seed, G, N)):
        kinds.append(kind or "")
        for n, d in dets.items():
            t0 = time.perf_counter(); sc[n].append(d.step(x)); tb[n] += time.perf_counter() - t0
        if (i + 1) % B == 0:
            for n in dets: ms[n].append(1e3 * tb[n] / B); tb[n] = 0.0
            print(i + 1, {n: round(ms[n][-1], 2) for n in dets}, flush=True)
    kinds = np.array(kinds); new = kinds != ""; res = dict(seed=seed, N=N, host=platform.node(), machine=platform.machine(),
                                                       peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, det={})
    for n in dets:
        x = np.array(sc[n]); blocks = []
        thr = np.quantile(x[S.WARM:B][~new[S.WARM:B]], 0.95)
        for b in range(N // B):
            sl = slice(max(b * B, S.WARM), (b + 1) * B); xb, nb = x[sl], new[sl]
            blocks.append(dict(auroc=auroc(xb[nb], xb[~nb]), fa=float(np.mean(xb[~nb] > thr)), ms=ms[n][b]))
        res["det"][n] = dict(blocks=blocks, stored_numbers=int(dets[n].stored()))
    json.dump(res, open(out, "w"), indent=1)
