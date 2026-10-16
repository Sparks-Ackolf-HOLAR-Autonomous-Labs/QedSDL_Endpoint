#!/usr/bin/env python3
"""S7: does the connectome model work better as several models (a swarm or a team) than as one?

Same four-family Raman stream as S1b (identical random draws per seed), plus the material family of
each scanned flake, which a facility knows from its sample log (an anomaly sits on the flake that was
loaded). Arms, with stored numbers relative to one model (5,177 Kenyon-cell weights):
  fly            one model, measured wiring                                   (1x)
  ens4_wiring    four models, measured + 3 degree-preserving shuffles, mean score  (4x)
  ens4_quarter   the measured model cut into 4 disjoint quarters of Kenyon cells, mean score (1x)
  fam4_full      one full model per material family, reading routed by family (4x)
  fam4_quarter   one quarter model per family (1x)
  zscore / zscore_fam        per-band running z-score, pooled / one per family
  knn_budget / knn_budget_fam  equal-memory nearest neighbour, pooled / budget split across families
Usage: python s7_swarm.py OUT_DIR SEEDS
"""
import json, os, sys, time
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402

FAMS = list(S.FAM)


def stream(seed, G):
    """Copy of S.stream with the family index kept (same draw order, so the stream is identical)."""
    rng = np.random.default_rng(seed); out = []
    for i in range(S.N):
        f = rng.integers(len(FAMS)); comp = S.FAM[FAMS[f]]
        w = rng.dirichlet([0.6] * len(comp))
        s = sum(wi * S.BASE[c] for wi, c in zip(w, comp))
        kind = None
        if i >= S.WARM and rng.random() < S.P_NEW:
            kind = S.KINDS[rng.integers(len(S.KINDS))]
            if kind == "MoS2/WSe2_stack":
                s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike":
                c = rng.uniform(110, 490); s = s + S.lines([(c, 4.0, 1.0)])
            else:
                fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        if i >= S.N // 2: scale /= 3
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg)
        y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        out.append((S.bands(y, G), kind, int(f)))
    return out


class Mean:
    def __init__(self, parts): self.p = parts
    def step(self, x, f): return float(np.mean([d.step(x) for d in self.p]))
    def stored(self): return sum(d.stored() for d in self.p)


class Route:
    def __init__(self, parts): self.p = parts
    def step(self, x, f): return self.p[f].step(x)
    def stored(self): return sum(d.stored() for d in self.p)


class One:
    def __init__(self, d): self.d = d
    def step(self, x, f): return self.d.step(x)
    def stored(self): return self.d.stored()


def shuffle(W, rng):
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w); return Ws


def run(seed):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2000 + seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    q = np.array_split(rng.permutation(W.shape[0]), 4)
    budget = int(W.shape[0] // G)
    nf = len(FAMS)
    dets = {
        "fly": One(S.Fly(W, Min)),
        "ens4_wiring": Mean([S.Fly(W, Min)] + [S.Fly(shuffle(W, rng), Min) for _ in range(3)]),
        "ens4_quarter": Mean([S.Fly(W[i], Min) for i in q]),
        "fam4_full": Route([S.Fly(W, Min) for _ in range(nf)]),
        "fam4_quarter": Route([S.Fly(W[i], Min) for i in q]),
        "zscore": One(S.Z(G)), "zscore_fam": Route([S.Z(G) for _ in range(nf)]),
        "knn_budget": One(S.KNN(True, budget, np.random.default_rng(3))),
        "knn_budget_fam": Route([S.KNN(True, budget // nf, np.random.default_rng(3 + j)) for j in range(nf)]),
    }
    st = stream(seed, G)
    sc = {n: [] for n in dets}; ms = {n: 0.0 for n in dets}
    for i, (x, kind, f) in enumerate(st):
        for n, d in dets.items():
            t0 = time.perf_counter(); s = d.step(x, f)
            if i >= S.WARM: ms[n] += (time.perf_counter() - t0) * 1e3
            sc[n].append(s)
    kinds = np.array([k if k else "" for _, k, _ in st]); new = kinds != ""
    idx = np.arange(S.N); post = idx >= S.WARM; h = S.N // 2
    pre_win = post & ~new & (idx >= h - 300) & (idx < h); post_win = ~new & (idx >= h) & (idx < h + 300)
    res = dict(seed=seed, det={})
    for n in dets:
        x = np.array(sc[n]); thr = np.quantile(x[post & ~new], 0.95); thr_pre = np.quantile(x[pre_win], 0.95)
        res["det"][n] = dict(
            auroc=float(roc_auc_score(new[post], x[post])),
            recall_at_5fpr=float(np.mean(x[post & new] > thr)),
            auroc_by_kind={k: float(roc_auc_score(np.r_[np.ones((post & (kinds == k)).sum()), np.zeros((post & ~new).sum())],
                                                  np.r_[x[post & (kinds == k)], x[post & ~new]])) for k in S.KINDS},
            false_alarm_after_change=float(np.mean(x[post_win] > thr_pre)),
            ms_per_reading=ms[n] / (S.N - S.WARM), stored_numbers=int(dets[n].stored()))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s7_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
