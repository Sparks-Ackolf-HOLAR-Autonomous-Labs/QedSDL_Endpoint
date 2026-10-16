#!/usr/bin/env python3
"""S6: one fly module, several jobs, at a shared 2D-semiconductor Raman facility.

Jobs on one Kenyon-cell code (one model, one configuration):
  J1 new-phase flag: 1T' MoS2, MoTe2, cosmic spike, MoS2/WSe2 stack (S1b stream), AUROC
  J2 wrong-sample flag: 3 % of normal flakes are logged under the wrong family (a mislabelled crystal or
     a swapped chip), AUROC of mislabelled vs correctly logged normal readings
  J3 lost-flake stop in a Janus MoSSe selenization run from the in-situ probe (S4; reported from the
     S4 run of the same configuration)
The module takes the logged family as context input (S8), so a spectrum that does not belong to its
logged family is unfamiliar in that context: J1 and J2 are read from the same familiarity output.
Separate tools for comparison: per-family z-score (J1, J2), nearest family centroid ratio (J2:
distance to the logged family's running centroid over the distance to the nearest centroid), and
for J3 the best probe classifier of S4 (nearest neighbour or logistic regression).
Pass rule (set before the run): each job within 0.02 AUROC (J3: within one lost flake) of the best
separate tool.
Usage: python s6_one_module.py OUT_DIR SEEDS CTX_GAIN [WORKERS]
"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s2_s3 as H  # noqa: E402
import s7_swarm as S7  # noqa: E402
from fly2 import Fly2  # noqa: E402
P_MIS = 0.03
CTX = float(sys.argv[3]) if __name__ == "__main__" else 4.0


class Centroid:
    def __init__(self, G, nf): self.m = np.zeros((nf, G)); self.n = np.zeros(nf)

    def step(self, x, f):
        v = x / (x.sum() + 1e-9)
        d = np.linalg.norm(self.m - v, axis=1); ok = self.n > 5
        s = float(d[f] / (d[ok].min() + 1e-9)) if ok[f] and ok.sum() > 1 else 1.0
        self.n[f] += 1; self.m[f] += (v - self.m[f]) / self.n[f]
        return s

    def stored(self): return self.m.size


def run(seed):
    from sklearn.metrics import roc_auc_score
    Wd, Min, G = H.wirings(seed); W = Wd["fly"]; nf = len(S7.FAMS); full = H.default_full_cfg()
    q = np.array_split(np.random.default_rng(11).permutation(W.shape[0]), nf)
    dets = {"module_ctx": Fly2(W, Min, ctx=True, ctx_gain=CTX),
            "module_best_ctx": Fly2(W, Min, ctx=True, ctx_gain=CTX, **full),
            "module_ctx_shuf": Fly2(Wd["shuf"], Min, ctx=True, ctx_gain=CTX),
            "module_ctx_rand": Fly2(Wd["rand"], Min, ctx=True, ctx_gain=CTX),
            "fam4_quarter": S7.Route([Fly2(W[i], Min) for i in q]),
            "zscore_fam": S7.Route([S.Z(G) for _ in range(nf)]),
            "centroid": Centroid(G, nf)}
    st = S7.stream(seed, G); rng = np.random.default_rng(500 + seed)
    logged, mis = [], []
    for x, k, f in st:
        m = k is None and rng.random() < P_MIS
        logged.append(int((f + rng.integers(1, nf)) % nf) if m else f); mis.append(m)
    mis = np.array(mis); kinds = np.array([k or "" for _, k, _ in st]); new = kinds != ""
    post = np.arange(len(st)) >= S.WARM; ok = post & ~new & ~mis
    res = dict(seed=seed, ctx_gain=CTX, n_mislabelled=int((mis & post).sum()), det={})
    for n, d in dets.items():
        x = np.array([d.step(xx, lf) for (xx, _, _), lf in zip(st, logged)])
        res["det"][n] = dict(
            j1_new_phase=float(roc_auc_score(np.r_[np.ones((post & new).sum()), np.zeros(ok.sum())], np.r_[x[post & new], x[ok]])),
            j2_wrong_sample=float(roc_auc_score(np.r_[np.ones((post & mis).sum()), np.zeros(ok.sum())], np.r_[x[post & mis], x[ok]])),
            stack=float(roc_auc_score(np.r_[np.ones((post & (kinds == "MoS2/WSe2_stack")).sum()), np.zeros(ok.sum())],
                                      np.r_[x[post & (kinds == "MoS2/WSe2_stack")], x[ok]])),
            stored_numbers=int(d.stored()))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s6_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: (round(d["j1_new_phase"], 3), round(d["j2_wrong_sample"], 3)) for n, d in r["det"].items()}, flush=True)
