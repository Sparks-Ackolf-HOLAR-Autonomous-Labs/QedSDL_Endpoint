#!/usr/bin/env python3
"""S5b: recovery-rate sweep with rare phases that return after a set gap.

Same week-long Raman stream as S5 (four 2D-semiconductor families, 5 % new events, laser drop to 1/3
at reading 50,000, 10^5 readings). Added: three rare phases that each appear in a first episode and
return once after a fixed gap (in readings). Line positions are literature Raman lines; intensities
and widths are assumed.
  alpha-MoO3 (oxidised Mo source): first at 10,000, returns after   1,000 readings
  GaSe (foreign flake on a shared tray): first at 15,000, returns after 10,000 readings
  SnS2 (contaminated selenium source): first at 20,000, returns after 60,000 readings
An episode is 20 readings, one every 5 readings, each a 0.6-0.9 mix of the phase into a routine spectrum.

Detectors: connectome model with slow synaptic recovery w += r (1 - w) per reading, r in RATES
(r = 0 is the plain model), equal-memory nearest neighbour (reservoir), per-band z-score.
Measures per detector: AUROC and false alarms per block of 10^4 readings (threshold fixed in the
first block, as in S5); per rare phase: recall at first sighting and on return at that threshold, and
the return ratio = mean score of the return episode / mean score of the first episode
(below 1 = the detector remembers the phase; 1 = it treats the phase as new again).
Usage: python s5b_return.py WIRING_NPZ OUT_JSON SEED DETECTOR   (DETECTOR: fly_r<rate>, knn_budget, zscore)
"""
import json, os, platform, sys, time
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
from fly2 import Fly2  # noqa: E402
from s5_long import auroc  # noqa: E402

FAMS = list(S.FAM)
RARE = {  # name: (lines (cm-1, amplitude, width), first reading, gap)
    "alpha-MoO3": ([(158, 0.7, 6), (283, 0.9, 6), (291, 0.6, 6), (338, 0.5, 6), (472, 0.4, 8)], 10000, 1000),
    "GaSe": ([(134, 1.0, 4), (213, 0.5, 5), (307, 0.8, 4)], 15000, 10000),
    "SnS2": ([(205, 0.3, 5), (315, 1.2, 4)], 20000, 60000),
}
EP, STEP = 20, 5
RS = {k: S.lines(v[0]) for k, v in RARE.items()}


def schedule():
    m = {}
    for k, (_, t0, gap) in RARE.items():
        for e, start in (("first", t0), ("return", t0 + gap)):
            for j in range(EP): m[start + STEP * j] = (k, e)
    return m


class Recover(Fly2):
    def __init__(self, W, Min, rate):
        super().__init__(W, Min); self.rate = rate

    def step(self, x, f=0):
        s = super().step(x, f)
        if self.rate: self.w += self.rate * (1 - self.w)
        return s


def gen(seed, G, N):
    rng = np.random.default_rng(seed); sch = schedule()
    for i in range(N):
        comp = S.FAM[FAMS[rng.integers(len(FAMS))]]; w = rng.dirichlet([0.6] * len(comp))
        s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); kind = None; rare = sch.get(i)
        if rare:
            fr = rng.uniform(0.6, 0.9); s = (1 - fr) * s + fr * RS[rare[0]]
        elif i >= S.WARM and rng.random() < S.P_NEW:
            kind = S.KINDS[rng.integers(len(S.KINDS))]
            if kind == "MoS2/WSe2_stack": s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike": s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
            else: fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        if i >= N // 2: scale /= 3
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        yield S.bands(y, G), kind, rare


if __name__ == "__main__":
    z = np.load(sys.argv[1]); out, seed, name = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    N = int(os.environ.get("S5B_N", 100000))
    W = np.zeros(tuple(z["shape"]), np.float32); W[z["r"], z["c"]] = z["w"]; Min = z["Min"].astype(np.float32)
    G = Min.shape[1]
    if name.startswith("fly_r"): det = Recover(W, Min, float(name[5:]))
    elif name == "knn_budget": det = S.KNN(True, int(W.shape[0] // G), np.random.default_rng(3))
    else: det = S.Z(G)
    sc, kinds, rares, t = [], [], [], 0.0
    for i, (x, kind, rare) in enumerate(gen(seed, G, N)):
        t0 = time.perf_counter(); sc.append(det.step(x)); t += time.perf_counter() - t0
        kinds.append(kind or ""); rares.append(rare)
        if (i + 1) % 10000 == 0: print(i + 1, round(1e3 * t / (i + 1), 2), flush=True)
    x = np.array(sc); new = np.array(kinds) != ""; isr = np.array([r is not None for r in rares])
    B = 10000; normal = ~new & ~isr
    thr = float(np.quantile(x[S.WARM:B][normal[S.WARM:B]], 0.95))
    blocks = []
    for b in range(N // B):
        sl = slice(max(b * B, S.WARM), (b + 1) * B); xb, nb, ob = x[sl], new[sl], normal[sl]
        blocks.append(dict(auroc=auroc(xb[nb], xb[ob]), fa=float(np.mean(xb[ob] > thr))))
    rare_res = {}
    for k in RARE:
        r = {}
        for e in ("first", "return"):
            idx = [i for i, v in enumerate(rares) if v == (k, e)]
            if idx:
                lo = max(idx[0] - 500, 0)  # routine readings just before the episode (added 14 Oct 2026, as S19)
                base = float(x[lo:idx[0]][normal[lo:idx[0]]].mean())
                r[e] = dict(recall=float(np.mean(x[idx] > thr)), mean_score=float(x[idx].mean()), baseline=base,
                            contrast=float(x[idx].mean()) / max(base, 1e-12))
        if "return" in r:
            r["return_ratio"] = r["return"]["mean_score"] / max(r["first"]["mean_score"], 1e-12)
            r["contrast_ratio"] = r["return"]["contrast"] / max(r["first"]["contrast"], 1e-12)
        rare_res[k] = r
    json.dump(dict(seed=seed, N=N, det=name, host=platform.node(), thr=thr, ms=1e3 * t / N,
                   stored=int(det.stored()), blocks=blocks, rare=rare_res), open(out, "w"), indent=1)
