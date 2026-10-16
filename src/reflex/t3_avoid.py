#!/usr/bin/env python3
"""T3 (reflex layer): avoid recipes that destroy the flake, in a Janus MoSSe campaign.

Campaign: 30 furnace runs on MoS2 -> Janus MoSSe (kmc, Mo parameters), 3 shared random starts.
A run fails (flake lost) when the readout shows the bottom layer converted (x_bot_Se > 0.3,
toward MoSe2) or the lattice damaged (defects > 0.3); 29 % of the 1400 grid recipes fail.
Arms:
  random            uniform over untested recipes
  gp                GP upper confidence bound on fidelity F (failures enter with their low F)
  cbo               constrained BO: UCB x probability of success from a GP classifier on the failure label
  fly               mushroom-body approach memory only (as in the Janus campaign)
  fly_avoid         + a separate avoid compartment: KC->MBON weights written only by failures,
                    no decay; a candidate whose avoid value exceeds AVOID_THR is vetoed
  gp+fly_avoid      GP-UCB chooses, the fly avoid compartment vetoes (reflex layer under the GP)
  *_shuf, *_rand    the same avoid compartment on shuffled / random PN->KC wiring
Measures: failed runs out of 30; gap = best true F on the grid minus best true F among tested recipes.
Usage: python t3_avoid.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

BUDGET, N0, TB, TD, AVOID_THR = 30, 3, 0.3, 0.3, 0.1  # threshold set on tuning seeds 100-103 (0.1, 0.2, 0.3 tried), not on test seeds
T = K.TRUTH["Mo"]


def failed(m): return m["x_bot_Se"] > TB or m["defects"] > TD


def gp_ucb(idx, y, kappa=2.0):
    k = ConstantKernel(1.0) * Matern([0.3] * 4, (0.03, 30), nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e-1))
    g = GaussianProcessRegressor(k, normalize_y=True, random_state=0).fit(K.X[idx], y)
    m, s = g.predict(K.X, return_std=True); return m + kappa * s


def p_success(idx, fl):
    if len(set(fl)) < 2: return np.full(K.NC, 1.0 if not any(fl) else 0.5)
    c = GaussianProcessClassifier(1.0 * Matern(0.3, (0.03, 30), nu=2.5), random_state=0).fit(K.X[idx], np.array(fl, int))
    return c.predict_proba(K.X)[:, list(c.classes_).index(0)]


class Mem:
    """Approach and avoid compartments on one KC code; approach relearned each step (as in the campaign agent)."""
    def __init__(self, W, mask):
        self.cd = K.codes(W, 0); self.mask = mask; self.av = np.zeros(len(mask))

    def avoid_value(self):
        return (self.cd * self.av).sum(1) / self.cd.sum(1)

    def learn_avoid(self, j, fail):  # written only by failures, one shot; successes weaken it a little
        c = self.cd[j] * self.mask
        self.av += (1.0 - self.avoid_value()[j]) * c if fail else -0.2 * self.av * c

    def approach_scores(self, idx, y, eta=0.5, beta=0.6):
        w = np.zeros(len(self.mask)); mu, sd = np.mean(y), np.std(y) + 1e-9
        for j, v in zip(idx, y):
            c = self.cd[j]; w += eta * ((v - mu) / sd - (c * w).sum() / c.sum()) * c * self.mask / c.sum()
        val = (self.cd * w).sum(1) / self.cd.sum(1)
        nov = 1 - (self.cd @ self.cd[idx].T).max(1) / self.cd.sum(1)
        return val + beta * nov


def run_arm(arm, seed, Wd, mask):
    rng = np.random.default_rng(9000 + seed); start = list(np.random.default_rng(seed).choice(K.NC, N0, replace=False))
    wkey = "fly_shuf" if arm.endswith("_shuf") else "fly_rand" if arm.endswith("_rand") else "fly"
    mem = Mem(Wd[wkey], mask) if "fly" in arm else None
    idx, y, fl, vetoed = [], [], [], 0
    t_dec = 0.0
    for n in range(BUDGET):
        t0 = time.perf_counter()
        if n < N0: j = int(start[n])
        else:
            if arm == "random": s = rng.random(K.NC)
            elif arm.startswith("gp") or arm == "cbo":
                s = gp_ucb(idx, y)
                if arm == "cbo": s = (s - s.min()) * p_success(idx, fl)
            else: s = mem.approach_scores(idx, y)
            s = s.astype(float); s[idx] = -np.inf
            if "avoid" in arm:
                bad = mem.avoid_value() > AVOID_THR; bad[idx] = False
                if (~bad & np.isfinite(s)).any(): vetoed += int(np.sum(bad & np.isfinite(s))); s[bad] = -np.inf
            j = int(np.argmax(s))
        t_dec += time.perf_counter() - t0
        m = K.kmc.measure(K.R[j], K.PAR["Mo"], rng); f = failed(m)
        idx.append(j); y.append(m["F"]); fl.append(f)
        if mem is not None: mem.learn_avoid(j, f)
    return dict(fails=int(sum(fl)), fails_after_start=int(sum(fl[N0:])), gap=float(T.max() - T[idx].max()),
                ms_per_decision=1e3 * t_dec / (BUDGET - N0), candidates_vetoed_total=vetoed)


ARMS = ["random", "gp", "cbo", "fly", "fly_avoid", "gp+fly_avoid", "gp+fly_avoid_shuf", "gp+fly_avoid_rand", "fly_avoid_shuf", "fly_avoid_rand"]


def run(seed):
    Wd = K.wirings(np.random.default_rng(seed)); mask = K.mbon_mask()
    return dict(seed=seed, arms={a: run_arm(a, seed, Wd, mask) for a in ARMS})


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/t3_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {a: (d["fails"], round(d["gap"], 3)) for a, d in r["arms"].items()}, flush=True)
