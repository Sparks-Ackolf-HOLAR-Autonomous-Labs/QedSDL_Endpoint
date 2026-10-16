#!/usr/bin/env python3
"""S1: Raman spectra of selenized MoS2 flakes fed to the mushroom body the way an
odour arrives, one spectral band per glomerulus, over a long stream with fixed memory.

Why: T1-T5 gave the model 3-7 hand-made numbers through a dense random projection, stored every past
code in a list (a nearest-neighbour search in disguise) and ran 100-200 readings. S1 tests the regime the
olfactory circuit is built for: a many-channel, positive signal whose overall strength is unknown (odour
concentration ~ laser power and focus), mixtures of a few known components, rare new components, and a
memory that does not grow with the number of samples.

Spectra (simulated, Lorentzian lines, 100-500 cm^-1, typical literature positions):
  2H MoS2 E 383, A1 403; Janus MoSSe A1 290, E 355; 2H MoSe2 A1 241, E 287.
  Normal reading: Dirichlet mixture of the three phases (partly selenized flake), overall intensity
  log-uniform x0.2-x5, sloped fluorescence background, noise.
New events (5 %, after warm-up): 1T' MoS2 (J1 156, J2 226, J3 330 + A1 403), WS2 contamination
  (E 356, A1 418), cosmic-ray spike (one narrow line), all at the same random intensity.
Instrument change: from reading N/2 the laser power drops to 1/3 (every later reading is weaker).

Input map: each uniglomerular FlyWire projection neuron (PN) reads the band of its glomerulus (bands
ordered by glomerulus name); a multiglomerular PN reads the mean of 3 random bands. KC code = top 5 % of
W @ PN (APL-like winner-take-all). Familiarity memory = one weight per KC onto a novelty MBON,
depressed by each code it sees (bounded: 5,177 numbers whatever the stream length).
Detectors:
  fly / fly_shuf / fly_rand  measured, degree-preserving shuffled, random sparse PN->KC wiring
  fly_dense                  measured wiring but the T1-T5 input path (centred values, dense random projection)
  knn_raw                    distance to the nearest stored raw spectrum (memory grows)
  knn_norm                   same on area-normalised spectra (memory grows)
  knn_budget                 area-normalised, reservoir with the same number of stored values as the fly
  zscore                     running per-band mean and variance of area-normalised spectra, max |z|
Measures (readings after warm-up): AUROC new vs normal, recall at 5 % false alarms per kind,
false-alarm rate on normal readings in the 300 readings after the instrument change (threshold from the
300 before), ms per reading, stored numbers at the end.
Usage: [S1_MODE=multi] python s1_spectra.py OUT_DIR SEEDS
"""
import json, os, sys, time
import numpy as np

N, WARM, P_NEW, K_FRAC = 4000, 300, 0.05, 0.05
NU = np.linspace(100, 500, 801)
PH = {"MoS2": [(383, 1.0, 4), (403, 1.2, 4)], "MoSSe": [(290, 1.2, 5), (355, 0.7, 6)],
      "MoSe2": [(241, 1.3, 4), (287, 0.4, 5)]}
NEW = {"1T'_MoS2": [(156, 0.8, 8), (226, 1.0, 8), (330, 0.8, 8), (403, 0.5, 4)],
       "WS2": [(356, 1.0, 5), (418, 1.0, 4)], "spike": None}
MODE = os.environ.get("S1_MODE", "janus")
# multi mode (S1b): a shared facility scanning four material families in one stream; per-band statistics
# then describe a blend of families rather than any one of them.
FAM = {"MoSSe": ["MoS2", "MoSSe", "MoSe2"], "WSSe": ["WS2", "WSe2"], "BP": ["BP"], "ReS2": ["ReS2"]}
if MODE == "multi":
    PH.update({"WS2": [(356, 1.0, 5), (418, 1.0, 4)], "WSe2": [(250, 1.4, 4), (260, 0.4, 6)],
               "BP": [(362, 0.8, 4), (439, 1.0, 4), (467, 1.1, 4)],
               "ReS2": [(151, 0.6, 3), (161, 0.8, 3), (212, 1.0, 3), (236, 0.6, 3), (307, 0.5, 3), (418, 0.4, 3)]})
    NEW = {"1T'_MoS2": NEW["1T'_MoS2"], "MoTe2": [(171, 1.0, 4), (234, 1.2, 4), (289, 0.4, 5)], "spike": None,
           "MoS2/WSe2_stack": None}  # new combination of familiar components: both families' lines at once
KINDS = list(NEW)


def lines(ls):
    s = np.zeros_like(NU)
    for c, a, w in ls:
        s += a * (w / 2) ** 2 / ((NU - c) ** 2 + (w / 2) ** 2)
    return s


BASE = {k: lines(v) for k, v in PH.items()}
NEWS = {k: lines(v) for k, v in NEW.items() if v}


def glom_map(rng):
    import pandas as pd
    d = os.path.expanduser("~/flywire_v783")
    a = pd.read_csv(f"{d}/neuron_annotations.tsv", sep="\t", low_memory=False)
    p = a[a.cell_class == "ALPN"].copy(); p["root_id"] = p.root_id.astype(np.int64)
    p = p.drop_duplicates("root_id").sort_values("root_id")
    ug = p.cell_sub_class == "uniglomerular"
    glom = p.cell_type.fillna("").str.split("_").str[0]
    names = sorted(set(glom[ug]))
    G = len(names); gi = {g: i for i, g in enumerate(names)}
    M = np.zeros((len(p), G))
    for r, (u, g) in enumerate(zip(ug, glom)):
        if u: M[r, gi[g]] = 1
        else: M[r, rng.choice(G, 3, replace=False)] = 1 / 3
    return p.root_id.values, M, G


def load_w(pn_ids):
    import pandas as pd
    d = os.path.expanduser("~/flywire_v783")
    a = pd.read_csv(f"{d}/neuron_annotations.tsv", sep="\t", low_memory=False)
    con = pd.read_parquet(f"{d}/Connectivity_783.parquet")
    KC = sorted(set(a.root_id[a.cell_class == "Kenyon_Cell"].astype(np.int64)))
    pi = {f: i for i, f in enumerate(pn_ids)}; ki = {f: i for i, f in enumerate(KC)}
    pk = con[con.Presynaptic_ID.isin(pi) & con.Postsynaptic_ID.isin(ki)]
    W = np.zeros((len(KC), len(pn_ids)))
    np.add.at(W, (pk.Postsynaptic_ID.map(ki).values, pk.Presynaptic_ID.map(pi).values), pk.Connectivity.values)
    return W


def bands(s, G):
    return s[: (len(s) // G) * G].reshape(G, -1).mean(1)


def stream(seed, G):
    rng = np.random.default_rng(seed); out = []
    for i in range(N):
        comp = FAM[list(FAM)[rng.integers(len(FAM))]] if MODE == "multi" else FAM["MoSSe"]
        w = rng.dirichlet([0.6] * len(comp))
        s = sum(wi * BASE[c] for wi, c in zip(w, comp))
        kind = None
        if i >= WARM and rng.random() < P_NEW:
            kind = KINDS[rng.integers(len(KINDS))]
            if kind == "MoS2/WSe2_stack":
                s = 0.5 * BASE["MoS2"] + 0.5 * BASE["WSe2"]
            elif kind == "spike":
                c = rng.uniform(110, 490); s = s + lines([(c, 4.0, 1.0)])
            else:
                f = rng.uniform(0.4, 0.8); s = (1 - f) * s + f * NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        if i >= N // 2: scale /= 3
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (NU - 300) / 200)
        y = scale * (s + bg)
        y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        out.append((bands(y, G), kind))
    return out


class Fly:
    def __init__(self, W, Min, dense=None, eta=0.3):
        self.W, self.M, self.dense = W, Min, dense
        self.k = int(K_FRAC * W.shape[0]); self.w = np.ones(W.shape[0]); self.eta = eta

    def code(self, x):
        if self.dense is not None:
            pn = np.maximum(((x / (x.mean() + 1e-9) - 1.0)) @ self.dense.T, 0)
        else:
            pn = self.M @ x
        d = self.W @ pn
        return d >= np.partition(d, -self.k)[-self.k]

    def step(self, x):
        c = self.code(x); s = self.w[c].sum() / self.k  # undepressed weight left on the active KCs = novelty
        self.w[c] *= 1 - self.eta
        return s

    def stored(self):
        return self.w.size


class KNN:
    def __init__(self, norm, budget=None, rng=None):
        self.norm, self.budget, self.rng, self.mem, self.seen = norm, budget, rng, [], 0

    def step(self, x):
        v = x / (x.sum() + 1e-9) if self.norm else x
        s = float(np.sqrt(((np.array(self.mem) - v) ** 2).sum(1)).min()) if self.mem else 1.0
        self.seen += 1
        if self.budget is None or len(self.mem) < self.budget: self.mem.append(v)
        else:
            j = self.rng.integers(self.seen)
            if j < self.budget: self.mem[j] = v
        return s

    def stored(self):
        return len(self.mem) * (len(self.mem[0]) if self.mem else 0)


class Z:
    def __init__(self, G): self.n, self.m, self.v = 0, np.zeros(G), np.zeros(G)

    def step(self, x):
        v = x / (x.sum() + 1e-9)
        s = float(np.max(np.abs(v - self.m) / np.sqrt(self.v / max(self.n - 1, 1) + 1e-12))) if self.n > 2 else 0.0
        self.n += 1; d = v - self.m; self.m += d / self.n; self.v += d * (v - self.m)
        return s

    def stored(self): return 2 * self.m.size


def run(seed):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1000 + seed)
    pn_ids, Min, G = glom_map(np.random.default_rng(5))
    W = load_w(pn_ids)
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); np.add.at(Wr, (rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))), w)
    dense = np.random.default_rng(7).normal(size=(W.shape[1], G))
    budget = int(W.shape[0] // G)
    dets = {"fly": Fly(W, Min), "fly_shuf": Fly(Ws, Min), "fly_rand": Fly(Wr, Min), "fly_dense": Fly(W, Min, dense),
            "knn_raw": KNN(False), "knn_norm": KNN(True), "knn_budget": KNN(True, budget, np.random.default_rng(3)),
            "zscore": Z(G)}
    S = stream(seed, G)
    sc = {n: [] for n in dets}; ms = {n: 0.0 for n in dets}
    for i, (x, kind) in enumerate(S):
        for n, d in dets.items():
            t0 = time.perf_counter(); s = d.step(x)
            if i >= WARM: ms[n] += (time.perf_counter() - t0) * 1e3
            sc[n].append(s)  # every detector returns a novelty score (higher = less familiar)
    kinds = np.array([k if k else "" for _, k in S]); new = kinds != ""
    idx = np.arange(N); post = idx >= WARM; h = N // 2
    pre_win = post & ~new & (idx >= h - 300) & (idx < h); post_win = ~new & (idx >= h) & (idx < h + 300)
    res = dict(seed=seed, G=G, n_new=int(new[post].sum()), budget_spectra=budget, det={})
    for n in dets:
        x = np.array(sc[n]); thr = np.quantile(x[post & ~new], 0.95)
        thr_pre = np.quantile(x[pre_win], 0.95)
        res["det"][n] = dict(
            auroc=float(roc_auc_score(new[post], x[post])),
            recall_at_5fpr=float(np.mean(x[post & new] > thr)),
            recall_by_kind={k: float(np.mean(x[post & (kinds == k)] > thr)) for k in KINDS},
            auroc_by_kind={k: float(roc_auc_score(np.r_[np.ones((post & (kinds == k)).sum()), np.zeros((post & ~new).sum())],
                                                  np.r_[x[post & (kinds == k)], x[post & ~new]])) for k in KINDS},
            false_alarm_after_change=float(np.mean(x[post_win] > thr_pre)),
            ms_per_reading=ms[n] / (N - WARM), stored_numbers=int(dets[n].stored()))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s1_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
