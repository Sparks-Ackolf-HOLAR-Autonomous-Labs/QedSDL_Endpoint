#!/usr/bin/env python3
"""S23: small neural networks on the S6 wrong-sample flag (3 % of normal flakes logged under the wrong family).

Same streams, mislabel draw and AUROC as s6_one_module.run; arms: the fly module with context (reproduction,
gain 2 as in results/reflex/s6_one_module), the small network ae46 unrouted and routed by the sample log (one
per family), at the learning rate chosen in s23_fair_nn j2 (1e-2 for both), and the per-family z-score.
S23_ONLY=fly_plain runs one arm (written to OUT_JSON).
Usage: CUDA_VISIBLE_DEVICES="" python s23_wrong_nn.py OUT_JSON SEEDS [WORKERS]
"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(seed):
    os.environ["S1_MODE"] = "multi"
    from sklearn.metrics import roc_auc_score
    import s1_spectra as S
    import s2_s3 as H
    import s7_swarm as S7
    import s19_online_nets as N
    from fly2 import Fly2
    H.MODE = "multi"; N.LR = 1e-2
    Wd, Min, G = H.wirings(seed); W = Wd["fly"]; nf = len(S7.FAMS)
    dets = {"module_ctx": Fly2(W, Min, ctx=True, ctx_gain=2.0),
            "nn_unrouted": S7.One(N.Online("ae46", G, seed)),
            "nn_routed": S7.Route([N.Online("ae46", G, seed * 10 + j) for j in range(nf)]),
            "zscore_fam": S7.Route([S.Z(G) for _ in range(nf)]),
            "fly_plain": Fly2(W, Min)}                     # the fly model without the sample log (context off)
    only = os.environ.get("S23_ONLY")
    if only:
        dets = {k: v for k, v in dets.items() if k in only.split(",")}
    st = S7.stream(seed, G); rng = np.random.default_rng(500 + seed)
    logged, mis = [], []
    for x, k, f in st:
        m = k is None and rng.random() < 0.03
        logged.append(int((f + rng.integers(1, nf)) % nf) if m else f); mis.append(m)
    mis = np.array(mis); kinds = np.array([k or "" for _, k, _ in st]); new = kinds != ""
    post = np.arange(len(st)) >= S.WARM; ok = post & ~new & ~mis
    res = dict(seed=seed, det={})
    for n, d in dets.items():
        x = np.array([d.step(xx, lf) for (xx, _, _), lf in zip(st, logged)])
        au = lambda pos: float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(ok.sum())], np.r_[x[pos], x[ok]]))  # noqa: E731
        res["det"][n] = dict(j1_new_phase=au(post & new), j2_wrong_sample=au(post & mis),
                             stack=au(post & (kinds == "MoS2/WSe2_stack")), stored_numbers=int(d.stored()))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        rs = list(ex.map(run, seeds))
    mean = {n: {k: float(np.mean([r["det"][n][k] for r in rs])) for k in rs[0]["det"][n]} for n in rs[0]["det"]}
    json.dump(dict(mean=mean, per_seed=rs), open(out, "w"), indent=1)
    for n, v in mean.items():
        print(n, {k: round(x, 3) for k, x in v.items()}, flush=True)
