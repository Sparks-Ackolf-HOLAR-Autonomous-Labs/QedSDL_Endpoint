#!/usr/bin/env python3
"""S18b: an ascending-style status report from the instrument, used after a laser-power drop.

In the fly, ascending neurons send a compressed report of body state up to the brain (S18). Here the
instrument sends one status bit: "the laser power changed", raised when the power telemetry departs by
more than 20 % from its running mean (simulated telemetry, 2 % noise). The S1/S7 Raman streams have
the laser drop to one third at reading N/2; every detector normalises intensity, so the drop acts
through signal-to-noise and background, which is what made the S2 mechanisms fail (annex table A3).

Uses of the status bit (cold-start defaults):
  gate      hold alarms for GATE readings after the bit and re-set the alarm threshold from them
            (95th percentile of all scores in the gate window; no labels used)
  ctx       connectome model only: the bit is a context input (Fly2 ctx), so readings after the change
            build a fresh familiarity memory in another quarter of the Kenyon cells; with the gate
  reset     z-score and nearest neighbour: clear the running statistics or memory at the bit; with the gate
Without the bit, the threshold is the 95th percentile of all scores in the 300 readings before the drop.
Measures in readings GATE..GATE+300 after the drop: false alarms on familiar readings, recall of new
events at the same threshold; AUROC over all readings after warm-up.
Usage: python s18b_status.py janus|multi OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys
MODE = sys.argv[1] if __name__ == "__main__" else os.environ.get("S1_MODE", "janus")
os.environ["S1_MODE"] = MODE
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s2_s3 as B  # noqa: E402
from fly2 import Fly2  # noqa: E402

GATE, WIN = 100, 300


def status_bits(seed):
    rng = np.random.default_rng(31000 + seed)
    p = np.where(np.arange(S.N) >= S.N // 2, 1 / 3, 1.0) * (1 + rng.normal(0, 0.02, S.N))
    ema, bits = p[0], np.zeros(S.N, bool)
    for i in range(1, S.N):
        bits[i] = abs(p[i] / ema - 1) > 0.2
        ema = p[i] if bits[i] else 0.95 * ema + 0.05 * p[i]
    return bits


def run(seed):
    from sklearn.metrics import roc_auc_score
    Wd, Min, G = B.wirings(seed)
    st = B.stream(seed, G); bits = status_bits(seed)
    nf = 4 if MODE == "multi" else 1
    det = {"fly": Fly2(Wd["fly"], Min), "fly_gate": Fly2(Wd["fly"], Min),
           "fly_ctx_gate": Fly2(Wd["fly"], Min, ctx=True, n_fam=2 * nf),
           "fly_shuf_ctx_gate": Fly2(Wd["shuf"], Min, ctx=True, n_fam=2 * nf),
           "zscore": S.Z(G), "zscore_gate": S.Z(G), "zscore_reset_gate": S.Z(G),
           "knn": S.KNN(True, int(Wd["fly"].shape[0] // G), np.random.default_rng(3)),
           "knn_gate": S.KNN(True, int(Wd["fly"].shape[0] // G), np.random.default_rng(3)),
           "knn_reset_gate": S.KNN(True, int(Wd["fly"].shape[0] // G), np.random.default_rng(3))}
    sc = {n: [] for n in det}; state = 0; t_bit = None
    for i, (x, kind, f) in enumerate(st):
        if bits[i] and t_bit is None:
            t_bit = i; state = 1
            for n in ("zscore_reset_gate",): det[n] = S.Z(G)
            det["knn_reset_gate"] = S.KNN(True, int(Wd["fly"].shape[0] // G), np.random.default_rng(4))
        for n, d in det.items():
            if n.startswith("fly"):
                fam = (f + nf * state) if "ctx" in n else 0
                sc[n].append(d.step(x, fam))
            else:
                sc[n].append(d.step(x))
    kinds = np.array([k if k else "" for _, k, _ in st]); new = kinds != ""
    idx = np.arange(S.N); h = S.N // 2; post = idx >= S.WARM
    ev = (idx >= h + GATE) & (idx < h + GATE + WIN)
    out = dict(t_bit=int(t_bit) if t_bit is not None else None)
    for n in det:
        s = np.array(sc[n])
        if "gate" in n:
            thr = np.quantile(s[(idx >= t_bit) & (idx < t_bit + GATE)], 0.95)
        else:
            thr = np.quantile(s[(idx >= h - WIN) & (idx < h)], 0.95)
        out[n] = dict(auroc=float(roc_auc_score(new[post], s[post])),
                      false_alarm=float(np.mean(s[ev & ~new] > thr)),
                      recall=float(np.mean(s[ev & new] > thr)) if (ev & new).any() else float("nan"))
    return dict(seed=seed, mode=MODE, det=out)


if __name__ == "__main__":
    outd, seeds = sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(outd, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{outd}/s18b_{MODE}_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], r["det"]["t_bit"] if False else "", {n: (round(d["false_alarm"], 3), round(d["recall"], 2))
                  for n, d in r["det"].items() if n != "t_bit"}, flush=True)
