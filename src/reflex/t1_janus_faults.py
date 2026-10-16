#!/usr/bin/env python3
"""T1 (reflex layer): flag faulty Janus MoSSe furnace runs, one reading at a time.

Stream: 200 selenization runs per seed (recipes drawn at random from the 1400-recipe grid),
each simulated by the lattice kMC (../janus/kmc.py) and read out as top Se fraction, bottom Se
fraction and defect fraction (Raman/XPS-type readout, noise 0.02).
Faults injected in 10 % of runs (labels known):
  furnace_offset  thermocouple reads low: the flake sees set point + 100 C
  se_empty        Se source exhausted: actual supply 0.1 of nominal / floor 0.1
  short_anneal    run aborted after 1/5 of the set time
  readout_glitch  readout replaced by a random triple (mis-focused Raman, wrong flake)
Detectors (all online; a reading is scored before it enters memory; faulty readings are
not added to memory once flagged in the top 10 % of scores, normal ones always are):
  fly        mushroom body, FlyWire 783 PN->KC wiring; the joint vector (recipe, readout)
             is encoded on the PNs, top-5 % KC code; score = 1 - max overlap with stored codes
             of runs whose recipe code overlaps most (familiarity of the readout given the recipe)
  fly_shuf   same, PN->KC partners shuffled (degree preserved)
  fly_rand   same, random sparse PN->KC wiring with the same number of synapses
  gp         GP (Matern 5/2) per readout fitted on memory, standardised residual (max over 3)
  iforest    isolation forest on the joint vector, refit on memory
  knn        distance to the 3rd nearest neighbour in readout space among the 10 nearest recipes
Effective fault: the readout differs by > 0.08 in any channel from a fault-free run of the same recipe;
faults that leave the flake unchanged cannot be seen by any detector and are left out of the scores.
Measures: AUROC (effective faults vs clean runs) over runs 41-200 (first 40 = warm-up), recall at 5 % false alarms, ms per reading.
Usage: python t1_janus_faults.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../janus"))
import kmc  # noqa: E402

N, WARM, FAULT_P = 200, 40, 0.10
KINDS = ["furnace_offset", "se_empty", "short_anneal", "readout_glitch"]
R = kmc.all_recipes(); XR = kmc.features(R); P_MO = kmc.system_params("Mo")


def stream(seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(R), N); out = []
    for i, j in enumerate(idx):
        rec = R[j].copy(); kind = None
        if i >= 5 and rng.random() < FAULT_P:
            kind = KINDS[rng.integers(4)]
        act = rec.copy()
        if kind == "furnace_offset": act[1] = rec[1] + 100
        if kind == "se_empty": act[3] = max(0.1, rec[3] * 0.1) if rec[3] > 0.1 else 0.01
        if kind == "short_anneal": act[2] = rec[2] / 5
        m = kmc.measure(act, P_MO, rng)
        y = np.array([m["x_top_Se"], m["x_bot_Se"], m["defects"]])
        if kind == "readout_glitch": y = rng.random(3)
        if kind is not None:  # does the fault change what the flake looks like? (same rng stream offset)
            ref = kmc.measure(rec, P_MO, np.random.default_rng(seed * 1000 + i))
            yr = np.array([ref["x_top_Se"], ref["x_bot_Se"], ref["defects"]])
            eff = bool(np.abs(y - yr).max() > 0.08)
        else:
            eff = False
        out.append((int(j), y, kind, eff))
    return out


def load_w():
    import pandas as pd
    d = os.path.expanduser("~/flywire_v783")
    a = pd.read_csv(f"{d}/neuron_annotations.tsv", sep="\t", low_memory=False)
    con = pd.read_parquet(f"{d}/Connectivity_783.parquet")
    PN = sorted(set(a.root_id[a.cell_class == "ALPN"].astype(np.int64)))
    KC = sorted(set(a.root_id[a.cell_class == "Kenyon_Cell"].astype(np.int64)))
    pi = {f: i for i, f in enumerate(PN)}; ki = {f: i for i, f in enumerate(KC)}
    pk = con[con.Presynaptic_ID.isin(pi) & con.Postsynaptic_ID.isin(ki)]
    W = np.zeros((len(KC), len(PN)))
    W[pk.Postsynaptic_ID.map(ki), pk.Presynaptic_ID.map(pi)] = pk.Connectivity.values
    return W


class FlyDet:
    def __init__(self, W, rng, frac=0.05):
        npn = W.shape[1]
        self.Er = rng.normal(size=(npn, 4)); self.Ej = rng.normal(size=(npn, 7))
        self.W, self.k = W, int(frac * W.shape[0])

    def code(self, v, E):
        d = np.maximum(((v - 0.5) / 0.3) @ E.T, 0) @ self.W.T
        thr = np.partition(d, -self.k)[-self.k]
        return d >= thr

    def score(self, mem, j, y):
        cr = self.code(XR[j], self.Er); cj = self.code(np.r_[XR[j], y], self.Ej)
        if not mem: return 1.0, (cr, cj)
        CR = np.array([m[0] for m in mem]); CJ = np.array([m[1] for m in mem])
        near = np.argsort(-(CR & cr).sum(1))[:10]           # runs with the most similar recipes
        fam = (CJ[near] & cj).sum(1).max() / self.k            # best readout match among them
        return 1 - fam, (cr, cj)


def gp_score(mem, j, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    A = XR[[m[0] for m in mem]]; Y = np.array([m[1] for m in mem]); s = 0
    for c in range(3):
        k = ConstantKernel(1.0) * Matern(0.3, (0.03, 30), nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e-1))
        g = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(A, Y[:, c])
        mu, sd = g.predict(XR[j][None], return_std=True)
        s = max(s, abs(y[c] - mu[0]) / (sd[0] + 0.02))
    return s


def run(seed):
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    W = load_w(); rng = np.random.default_rng(1000 + seed)
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); Wr[rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))] = w
    dets = {"fly": FlyDet(W, np.random.default_rng(7)), "fly_shuf": FlyDet(Ws, np.random.default_rng(7)),
            "fly_rand": FlyDet(Wr, np.random.default_rng(7))}
    S = stream(seed); names = list(dets) + ["gp", "iforest", "knn"]
    mem = {n: [] for n in names}; sc = {n: [] for n in names}; ms = {n: 0.0 for n in names}
    for i, (j, y, kind, eff) in enumerate(S):
        for n in names:
            t0 = time.perf_counter()
            if n in dets:
                s, item = dets[n].score(mem[n], j, y)
            else:
                item = (j, y)
                if len(mem[n]) < 5: s = 0.0
                elif n == "gp": s = gp_score(mem[n], j, y)
                elif n == "iforest":
                    Z = np.array([np.r_[XR[a], b] for a, b in mem[n]])
                    s = -IsolationForest(n_estimators=100, random_state=0).fit(Z).score_samples(np.r_[XR[j], y][None])[0]
                else:
                    A = XR[[m[0] for m in mem[n]]]; Y = np.array([m[1] for m in mem[n]])
                    near = np.argsort(((A - XR[j]) ** 2).sum(1))[:10]
                    s = np.sort(np.sqrt(((Y[near] - y) ** 2).sum(1)))[min(2, len(near) - 1)]
            if i >= WARM: ms[n] += (time.perf_counter() - t0) * 1e3
            sc[n].append(s)
            past = np.array(sc[n][:-1]) if len(sc[n]) > 10 else None
            if past is None or s < np.quantile(past, 0.9):
                mem[n].append(item)
    lab = np.array([k is not None for _, _, k, _ in S])[WARM:]
    effl = np.array([e for *_, e in S])[WARM:]
    keep = ~lab | effl  # score effective faults against clean runs; faults that change nothing are left out
    kinds = [k if e else None for _, _, k, e in S][WARM:]
    res = dict(seed=seed, n_faults=int(lab.sum()), n_effective=int(effl.sum()), det={})
    for n in names:
        x0 = np.array(sc[n][WARM:]); fpr_thr = np.quantile(x0[~lab], 0.95)
        x, labk = x0[keep], effl[keep]
        rec = {k: float(np.mean(x0[[kk == k for kk in kinds]] > fpr_thr)) if k in kinds else None for k in KINDS}
        res["det"][n] = dict(auroc=float(roc_auc_score(labk, x)), recall_at_5fpr=float(np.mean(x[labk] > fpr_thr)),
                             recall_by_kind=rec, ms_per_reading=ms[n] / (N - WARM))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    wk = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(wk) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/t1_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
