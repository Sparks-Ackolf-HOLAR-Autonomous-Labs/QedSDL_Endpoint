#!/usr/bin/env python3
"""S9, S10, S12: three tests that address the misses of tests T1-T5.

All three reuse the S1 Raman streams unchanged (S1_MODE=janus: Janus MoSSe selenization line;
S1_MODE=multi: shared facility with four material families), so every arm is paired seed by seed with
S1/S1b and with the engineered baselines (per-band z-score, equal-memory nearest neighbour).

S9  input mapping. In S1 the spectral bands were assigned to glomeruli in name order, so the measured
    wiring could not carry meaning beyond its degree structure. Three mappings, each computed from the
    300 unlabelled warm-up readings only (no new events occur before reading 300):
      arb    name order (as S1)
      var    bands ranked by their warm-up coefficient of variation matched to glomeruli ranked by
             Kenyon-cell drive (out-degree): the most variable bands go to the best-connected glomeruli
      coact  bands that rise and fall together (warm-up correlation) placed on glomeruli that the
             wiring co-samples (correlation of their Kenyon-cell drive columns): quadratic assignment,
             spectral-order start plus pairwise-swap search, done separately for each wiring
    Each mapping is run with measured, degree-preserving shuffled and random sparse wiring. The test:
    does a statistics-matched mapping make the measured wiring beat the shuffled one?
S10 two memory compartments. The mushroom body has about 15 output compartments with different
    learning and forgetting rates. Here: a slow compartment (as S1, no recovery) and a fast one whose
    learns in one shot (depression 0.9 instead of 0.3) and whose weights recover towards 1 at rate
    r_fast per reading; novelty = min(slow, fast), so a pattern is
    new only if both compartments find it new. After an instrument change the fast compartment learns
    the new normal within about 1/r_fast readings, while a rare new phase stays new for it.
    Result on tuning seeds: min(slow, fast) is always the slow score (the slow weights never recover, so
    they are never above the fast ones); the arm is kept as fly_2comp_min. Second form, fly_2comp:
    the fast compartment tracks the recent level of the slow output (an output that adapts with rate
    r_fast), and novelty = slow - fast, so a lasting rise of novelty after an instrument change fades
    while a single new phase stands out. Engineered control zscore_2t: the same subtraction on the z-score.
    Earlier engineered control (min rule) on a slow and a fast (exponentially weighted) z-score.
    r_fast for both is chosen on tuning seeds 100-103 (rule: lowest false alarms after the laser drop
    with AUROC within 0.01 of the one-compartment version on both streams), then frozen for seeds 1-20.
S12 the two hemispheres. FlyWire holds both mushroom bodies of one animal. The S1 model pooled them
    (5,177 Kenyon cells). Arms: left only, right only, both pooled (S1), left and right as two
    circuits whose scores are averaged (a "group" of two real circuits).
Usage: S1_MODE=janus|multi python s9_s12.py OUT_DIR SEEDS [R_FAST] [R_FAST_Z]
       S1_MODE=... python s9_s12.py --tune OUT_DIR
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402

R_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


# ---------------------------------------------------------------- S9 mappings
def warm_stats(Xw):
    V = Xw / (Xw.sum(1, keepdims=True) + 1e-12)
    cv = V.std(0) / (V.mean(0) + 1e-12)
    R = np.nan_to_num(np.corrcoef(V.T))
    np.fill_diagonal(R, 0)
    return cv, R


def glom_stats(W, Min):
    A = W @ Min                      # KC x glomerulus drive
    deg = A.sum(0)
    C = np.nan_to_num(np.corrcoef(A.T)); np.fill_diagonal(C, 0)
    return deg, C


def spectral_order(M):
    L = np.diag(np.abs(M).sum(1)) - M
    _, v = np.linalg.eigh(L)
    return np.argsort(v[:, 1])


def qap(R, C, rng, passes=6):
    """perm[b] = glomerulus of band b, maximising sum_bb' R[b,b'] C[perm b, perm b']."""
    G = len(R); p = np.empty(G, int)
    p[spectral_order(np.maximum(R, 0))] = spectral_order(np.maximum(C, 0))
    for _ in range(passes):
        improved = False
        for a in rng.permutation(G):
            for b in range(G):
                if a == b: continue
                pa, pb = p[a], p[b]
                # change in objective when bands a and b swap glomeruli (R, C symmetric, zero diagonal)
                ra, rb = R[a].copy(), R[b].copy(); ra[b] = 0; rb[a] = 0
                d = 2 * (ra @ (C[pb, p] - C[pa, p]) + rb @ (C[pa, p] - C[pb, p]))
                if d > 1e-9:
                    p[a], p[b] = pb, pa; improved = True
        if not improved: break
    return p


def obj(R, C, p): return float((R * C[np.ix_(p, p)]).sum())


def var_map(cv, deg):
    p = np.empty(len(cv), int); p[np.argsort(-cv)] = np.argsort(-deg); return p


class Mapped:
    """Feeds the model the band vector re-ordered so that band b drives glomerulus perm[b]."""
    def __init__(self, det, perm):
        self.d = det; self.inv = np.argsort(perm)
    def step(self, x, f=0): return self.d.step(x[self.inv])
    def stored(self): return self.d.stored()


# ---------------------------------------------------------------- S10 compartments
class TwoComp(S.Fly):
    def __init__(self, W, Min, r_fast, eta=0.3, eta_fast=0.9):
        super().__init__(W, Min, eta=eta); self.wf = np.ones(W.shape[0]); self.r = r_fast; self.ef = eta_fast
    def step(self, x, f=0):
        c = self.code(x)
        s_slow = self.w[c].sum() / self.k; s_fast = self.wf[c].sum() / self.k
        self.w[c] *= 1 - self.eta; self.wf[c] *= 1 - self.ef
        self.wf += self.r * (1 - self.wf)
        return min(s_slow, s_fast)
    def stored(self): return 2 * self.w.size


class Adapt:
    """Output adaptation: score minus its own exponentially weighted recent level (fast compartment)."""
    def __init__(self, det, r): self.d, self.r, self.b = det, r, None
    def step(self, x, f=0):
        s = self.d.step(x)
        if self.b is None: self.b = s; return 0.0
        o = s - self.b; self.b += self.r * (s - self.b); return o
    def stored(self): return self.d.stored() + 1


class Z2:
    """Two-timescale per-band z-score: slow = running (all data), fast = exponentially weighted."""
    def __init__(self, G, r):
        self.z = S.Z(G); self.r = r; self.m = None; self.v = None
    def step(self, x, f=0):
        s_slow = self.z.step(x); v = x / (x.sum() + 1e-9)
        if self.m is None:
            self.m, self.v = v.copy(), np.full_like(v, 1e-4); return s_slow
        s_fast = float(np.max(np.abs(v - self.m) / np.sqrt(self.v + 1e-12)))
        d = v - self.m; self.m += self.r * d; self.v = (1 - self.r) * (self.v + self.r * d * d)
        return min(s_slow, s_fast)
    def stored(self): return 4 * self.m.size


class Mean:
    def __init__(self, parts): self.p = parts
    def step(self, x, f=0): return float(np.mean([d.step(x) for d in self.p]))
    def stored(self): return sum(d.stored() for d in self.p)


# ---------------------------------------------------------------- helpers
def sides(n_kc_ids):
    import pandas as pd
    a = pd.read_csv(os.path.expanduser("~/flywire_v783/neuron_annotations.tsv"), sep="\t", low_memory=False)
    k = a[a.cell_class == "Kenyon_Cell"].copy(); k["root_id"] = k.root_id.astype(np.int64)
    side = dict(zip(k.root_id, k.side))
    import pandas as pd  # noqa
    KC = sorted(set(k.root_id))
    return np.array([side.get(r, "") for r in KC])


def controls(W, rng):
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); np.add.at(Wr, (rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))), w)
    return Ws, Wr


def score(dets, Sx, G):
    from sklearn.metrics import roc_auc_score
    sc = {n: [] for n in dets}; ms = {n: 0.0 for n in dets}
    for i, (x, kind) in enumerate(Sx):
        for n, d in dets.items():
            t0 = time.perf_counter(); sc[n].append(d.step(x))
            if i >= S.WARM: ms[n] += (time.perf_counter() - t0) * 1e3
    N = len(Sx); kinds = np.array([k or "" for _, k in Sx]); new = kinds != ""
    idx = np.arange(N); post = idx >= S.WARM; h = N // 2
    pre_win = post & ~new & (idx >= h - 300) & (idx < h); post_win = ~new & (idx >= h) & (idx < h + 300)
    out = {}
    for n in dets:
        x = np.array(sc[n]); thr = np.quantile(x[post & ~new], 0.95); thr_pre = np.quantile(x[pre_win], 0.95)
        out[n] = dict(auroc=float(roc_auc_score(new[post], x[post])),
                      recall_at_5fpr=float(np.mean(x[post & new] > thr)),
                      auroc_by_kind={k: float(roc_auc_score(np.r_[np.ones((post & (kinds == k)).sum()), np.zeros((post & ~new).sum())],
                                                            np.r_[x[post & (kinds == k)], x[post & ~new]])) for k in S.KINDS},
                      false_alarm_after_change=float(np.mean(x[post_win] > thr_pre)),
                      ms_per_reading=ms[n] / (N - S.WARM), stored_numbers=int(dets[n].stored()))
    return out


def setup(seed):
    rng = np.random.default_rng(1000 + seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    Ws, Wr = controls(W, rng)  # same draw order as S1, so fly_shuf / fly_rand equal S1's
    return Min, G, W, Ws, Wr


def run(args):
    seed, r_fast, r_z = args
    Min, G, W, Ws, Wr = setup(seed)
    Sx = S.stream(seed, G)
    Xw = np.array([x for x, _ in Sx[:S.WARM]])
    cv, R = warm_stats(Xw)
    rng = np.random.default_rng(77 + seed)
    dets, maps = {}, {}
    for wn, Wx in (("fly", W), ("fly_shuf", Ws), ("fly_rand", Wr)):
        deg, C = glom_stats(Wx, Min)
        pa = np.arange(G); pv = var_map(cv, deg); pc = qap(R, C, rng)
        maps[wn] = dict(obj_arb=obj(R, C, pa), obj_var=obj(R, C, pv), obj_coact=obj(R, C, pc))
        for mn, p in (("arb", pa), ("var", pv), ("coact", pc)):
            dets[f"{wn}_{mn}"] = Mapped(S.Fly(Wx, Min), p)
    # S10
    dets["fly_2comp_min"] = TwoComp(W, Min, r_fast)
    dets["fly_2comp"] = Adapt(S.Fly(W, Min), r_fast)
    dets["fly_2comp_shuf"] = Adapt(S.Fly(Ws, Min), r_fast)
    dets["zscore"] = S.Z(G); dets["zscore_2t"] = Adapt(S.Z(G), r_z)
    dets["knn_budget"] = S.KNN(True, int(W.shape[0] // G), np.random.default_rng(3))
    # S12
    sd = sides(W.shape[0])
    L, Rt = W[sd == "left"], W[sd == "right"]
    dets["fly_left"] = S.Fly(L, Min); dets["fly_right"] = S.Fly(Rt, Min)
    dets["fly_lr_vote"] = Mean([S.Fly(L, Min), S.Fly(Rt, Min)])
    res = dict(seed=seed, mode=S.MODE, G=G, r_fast=r_fast, r_fast_z=r_z, n_kc=dict(left=int((sd == "left").sum()),
               right=int((sd == "right").sum()), all=int(W.shape[0])), mapping_objective=maps, det=score(dets, Sx, G))
    return res


def tune_one(args):
    seed, r = args
    Min, G, W, Ws, Wr = setup(seed)
    Sx = S.stream(seed, G)
    d = {"fly": S.Fly(W, Min), "zscore": S.Z(G), f"fly_2comp_{r}": Adapt(S.Fly(W, Min), r), f"zscore_2t_{r}": Adapt(S.Z(G), r)}
    return seed, r, score(d, Sx, G)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    if sys.argv[1] == "--tune":
        out = sys.argv[2]; os.makedirs(out, exist_ok=True)
        jobs = [(s, r) for s in (100, 101, 102, 103) for r in R_GRID]
        res = {}
        with ProcessPoolExecutor(4) as ex:
            for s, r, o in ex.map(tune_one, jobs):
                res[f"{s}_{r}"] = o; print(s, r, {n: (round(v["auroc"], 3), round(v["false_alarm_after_change"], 3)) for n, v in o.items()}, flush=True)
        json.dump(res, open(f"{out}/tune_{S.MODE}.json", "w"), indent=1)
    else:
        out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
        rf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03; rz = float(sys.argv[4]) if len(sys.argv) > 4 else rf
        os.makedirs(out, exist_ok=True)
        with ProcessPoolExecutor(min(4, len(seeds))) as ex:
            for r in ex.map(run, [(s, rf, rz) for s in seeds]):
                json.dump(r, open(f"{out}/s9_s{r['seed']}.json", "w"), indent=1)
                print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
