#!/usr/bin/env python3
"""S8: one connectome model told the material family (context), against the four-model team of S7.

Setting: a shared Raman facility for 2D semiconductors scans four families in one stream (S1b/S7:
Mo-S-Se including Janus MoSSe, W-S-Se, black phosphorus, ReS2); the sample log says which crystal is
loaded. S7 found that one quarter model per family (routing) lifts AUROC from 0.86 to 0.98 at equal
memory, but so does a per-family z-score. In the fly, division of labour happens inside one mushroom
body: Kenyon cells receive non-olfactory context in the accessory calyx, and dopamine neurons gate
separate output compartments. S8 tests both forms on one model (fly2.py options ctx and comp).
Arms (memory relative to one model): fly (1x); fly_ctx1/2/4 (context drive 1, 2, 4 x std of the
olfactory drive on one quarter of the Kenyon cells per family, 1x); fly_comp (one dopamine-gated
compartment per family on one KC code, 4x); fly_ctx2_comp (both, 4x); fam4_quarter and fam4_full
(S7 teams); zscore_fam, knn_budget_fam (references). Same streams and measures as S7.
Usage: python s8_context.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s7_swarm as S7  # noqa: E402
import s2_s3 as H  # noqa: E402
from fly2 import Fly2  # noqa: E402
H.MODE = "multi"


def arms(Wd, Min, G):
    W = Wd["fly"]; nf = len(S7.FAMS); budget = int(W.shape[0] // G)
    q = np.array_split(np.random.default_rng(11).permutation(W.shape[0]), nf)
    return {
        "fly": Fly2(W, Min),
        "fly_ctx1": Fly2(W, Min, ctx=True, ctx_gain=1.0),
        "fly_ctx2": Fly2(W, Min, ctx=True, ctx_gain=2.0),
        "fly_ctx4": Fly2(W, Min, ctx=True, ctx_gain=4.0),
        "fly_ctx4_shuf": Fly2(Wd["shuf"], Min, ctx=True, ctx_gain=4.0),
        "fly_ctx4_rand": Fly2(Wd["rand"], Min, ctx=True, ctx_gain=4.0),
        "fly_comp": Fly2(W, Min, comp=True),
        "fly_ctx2_comp": Fly2(W, Min, ctx=True, ctx_gain=2.0, comp=True),
        "fam4_quarter": S7.Route([Fly2(W[i], Min) for i in q]),
        "fam4_full": S7.Route([Fly2(W, Min) for _ in range(nf)]),
        "zscore_fam": S7.Route([S.Z(G) for _ in range(nf)]),
        "knn_budget_fam": S7.Route([S.KNN(True, budget // nf, np.random.default_rng(3 + j)) for j in range(nf)]),
    }


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(H.run, seeds, [arms] * len(seeds)):
            json.dump(r, open(f"{out}/s8_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: (round(d["auroc"], 3), round(d["auroc_by_kind"]["MoS2/WSe2_stack"], 2)) for n, d in r["det"].items()}, flush=True)
