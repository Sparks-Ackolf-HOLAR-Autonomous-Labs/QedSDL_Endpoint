#!/usr/bin/env python3
"""S22b: detection quality against memory when the model is made larger (more compute per reading).

Standard S1b stream (4,000 readings, four material families, laser drop at N/2), one seed per call.
Arms, each reporting AUROC (new vs normal), bytes held (float32 values, int32 indices) and us per reading:
  kc_x{s}     connectome-shaped model with s x 5,177 Kenyon cells, s = 0.25, 0.5, 1, 2, 4, 8. Each KC copies
              the in-degree and synapse counts of a randomly drawn measured KC; its PNs are drawn in proportion
              to measured PN out-degree (degree-preserving resampling; S1 showed measured = shuffled wiring).
              kc_x1 also has the measured wiring itself (fly_measured) as a check.
  knn_eq{s}   nearest-neighbour reservoir holding as many values as kc_x{s} holds KC weights
  knn_full    nearest neighbour keeping every reading (reference: unbounded memory)
  ae{h}       online autoencoder 56-h-56 (Adam lr 3e-3, no replay), h = 16, 64, 256, 1024
Usage: python s22_model_size.py OUT_DIR SEED
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["S1_MODE"] = "multi"
import s1_spectra as S  # noqa: E402
from s5_long import auroc  # noqa: E402
from s22_data_growth import FlySparse, KNN, AE  # noqa: E402

SCALES = [0.25, 0.5, 1, 2, 4, 8]


def resample(W, s, rng):
    """Degree-preserving connectome-shaped wiring with round(s * n_KC) Kenyon cells."""
    n_kc, n_pn = W.shape; n = int(round(s * n_kc))
    pn_p = (W > 0).sum(0).astype(float); pn_p /= pn_p.sum()
    rows = [np.nonzero(W[k])[0] for k in range(n_kc)]
    Wn = np.zeros((n, n_pn), np.float32)
    for j, k in enumerate(rng.integers(0, n_kc, n)):
        cnt = W[k, rows[k]]
        if len(cnt) == 0: continue
        cols = rng.choice(n_pn, len(cnt), replace=False, p=pn_p)
        Wn[j, cols] = cnt
    return Wn


def run(seed, out):
    import torch
    torch.set_num_threads(1)
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    rng = np.random.default_rng(2200 + seed)
    dets = {"fly_measured": FlySparse(W, Min)}
    for s in SCALES:
        f = FlySparse(resample(W, s, rng), Min); dets[f"kc_x{s:g}"] = f
        dets[f"knn_eq{s:g}"] = KNN(G, max(1, f.n // G), seed=3 + seed)
    dets["knn_full"] = KNN(G)
    for h in (16, 64, 256, 1024):
        dets[f"ae{h}"] = AE(G, 0, seed, h=h)
    st = S.stream(seed, G)
    sc = {n: [] for n in dets}; tt = {n: [] for n in dets}
    for x, _ in st:
        for n, d in dets.items():
            t0 = time.perf_counter(); sc[n].append(d.step(x)); tt[n].append(time.perf_counter() - t0)
    new = np.array([k is not None for _, k in st]); post = np.arange(S.N) >= S.WARM
    res = dict(seed=seed, det={})
    for n, d in dets.items():
        x = np.array(sc[n])
        res["det"][n] = dict(auroc=auroc(x[post & new], x[post & ~new]), bytes=int(d.nbytes()),
                             us_per_reading=float(1e6 * np.median(tt[n][S.WARM:])),
                             n_kc=int(getattr(d, "n", 0)) if n.startswith(("kc", "fly")) else None,
                             synapse_entries=int(len(d.r)) if n.startswith(("kc", "fly")) else None)
    json.dump(res, open(f"{out}/s22b_s{seed}.json", "w"), indent=1)
    print(seed, {n: (round(v["auroc"], 3), v["bytes"]) for n, v in res["det"].items()}, flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True); run(int(sys.argv[2]), sys.argv[1])
