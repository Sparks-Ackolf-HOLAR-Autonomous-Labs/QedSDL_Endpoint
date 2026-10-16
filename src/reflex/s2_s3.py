#!/usr/bin/env python3
"""S2 (antennal-lobe gain control, receptor adaptation) and S3 (claw coincidence) on the 2D-semiconductor
Raman streams of S1 (Janus MoSSe selenization line) and S1b (shared facility, four families: Mo-S-Se,
W-S-Se, black phosphorus, ReS2). Same streams, warm-up, laser drop and measures as S1/S7.

Arms: fly (S1 baseline), fly_gain, fly_adapt, fly_coinc, fly_gain_coinc, fly_all (adapt+gain+coinc);
fly_best (only the mechanisms that passed the tuning rule) with fly_best_shuf / fly_best_rand (degree-preserving shuffled / random sparse wiring); zscore; knn_budget.
Parameters of the new mechanisms were chosen on tuning seeds 100-103 (s2_tune.py), not on test seeds.
Pass rules (set before the run):
  S2  false alarms after the laser drop no higher than the per-band z-score's
  S3  MoS2/WSe2 stack AUROC of at least 0.85 (S1b)
Usage: python s2_s3.py janus|multi OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, time
MODE = sys.argv[1] if __name__ == "__main__" else os.environ.get("S1_MODE", "multi")
os.environ["S1_MODE"] = MODE
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
from fly2 import Fly2  # noqa: E402

PAR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_params.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_params.json")) else {}


def stream(seed, G):
    if MODE == "multi":
        import s7_swarm as S7
        return S7.stream(seed, G)
    return [(x, k, 0) for x, k in S.stream(seed, G)]


def wirings(seed):
    rng = np.random.default_rng(1000 + seed)
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    W = S.load_w(pn_ids)
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); np.add.at(Wr, (rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))), w)
    return dict(fly=W, shuf=Ws, rand=Wr), Min, G


def default_full_cfg():
    """Best configuration: only the mechanisms that passed the tuning rule (s2_params.json 'passed')."""
    cfg = {}
    for m in PAR.get("passed", []): cfg.update({m: True}, **PAR[m])
    return cfg


def default_arms(Wd, Min, G):
    g = PAR.get("gain", {}); a = PAR.get("adapt", {}); c = PAR.get("coinc", {})
    full = dict(adapt=True, gain=True, coinc=True, **a, **g, **c); best = default_full_cfg()
    return {
        "fly": Fly2(Wd["fly"], Min),
        "fly_gain": Fly2(Wd["fly"], Min, gain=True, **g),
        "fly_adapt": Fly2(Wd["fly"], Min, adapt=True, **a),
        "fly_coinc": Fly2(Wd["fly"], Min, coinc=True, **c),
        "fly_gain_coinc": Fly2(Wd["fly"], Min, gain=True, coinc=True, **g, **c),
        "fly_all": Fly2(Wd["fly"], Min, **full),
        "fly_best": Fly2(Wd["fly"], Min, **best),
        "fly_best_shuf": Fly2(Wd["shuf"], Min, **best),
        "fly_best_rand": Fly2(Wd["rand"], Min, **best),
        "zscore": S.Z(G),
        "knn_budget": S.KNN(True, int(Wd["fly"].shape[0] // G), np.random.default_rng(3)),
    }


def score_stream(dets, st):
    from sklearn.metrics import roc_auc_score
    import inspect
    sc = {n: [] for n in dets}; ms = {n: 0.0 for n in dets}
    two = {n: len(inspect.signature(d.step).parameters) >= 2 for n, d in dets.items()}
    for i, (x, kind, f) in enumerate(st):
        for n, d in dets.items():
            t0 = time.perf_counter()
            s = d.step(x, f) if two[n] else d.step(x)
            if i >= S.WARM: ms[n] += (time.perf_counter() - t0) * 1e3
            sc[n].append(s)
    N = len(st); kinds = np.array([k if k else "" for _, k, _ in st]); new = kinds != ""
    idx = np.arange(N); post = idx >= S.WARM; h = N // 2
    pre_win = post & ~new & (idx >= h - 300) & (idx < h); post_win = ~new & (idx >= h) & (idx < h + 300)
    out = {}
    for n in dets:
        x = np.array(sc[n]); thr = np.quantile(x[post & ~new], 0.95); thr_pre = np.quantile(x[pre_win], 0.95)
        out[n] = dict(
            auroc=float(roc_auc_score(new[post], x[post])),
            recall_at_5fpr=float(np.mean(x[post & new] > thr)),
            auroc_by_kind={k: float(roc_auc_score(np.r_[np.ones((post & (kinds == k)).sum()), np.zeros((post & ~new).sum())],
                                                  np.r_[x[post & (kinds == k)], x[post & ~new]])) for k in S.KINDS},
            false_alarm_after_change=float(np.mean(x[post_win] > thr_pre)),
            ms_per_reading=ms[n] / (N - S.WARM), stored_numbers=int(dets[n].stored()))
    return out


def run(seed, arms_fn=default_arms):
    Wd, Min, G = wirings(seed)
    return dict(seed=seed, mode=MODE, params=PAR, det=score_stream(arms_fn(Wd, Min, G), stream(seed, G)))


if __name__ == "__main__":
    out, seeds = sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s2_{MODE}_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: (round(d["auroc"], 3), round(d["false_alarm_after_change"], 3),
                                  round(d["auroc_by_kind"].get("MoS2/WSe2_stack", float("nan")), 2))
                              for n, d in r["det"].items()}, flush=True)

