#!/usr/bin/env python3
"""S16: what a hardware synapse must do to run the connectome model.

The model is mapped onto two device arrays, as a 2D-semiconductor memristor or ferroelectric-transistor
build would be:
  * PN -> KC crossbar (fixed after programming; 23,516-27,848 occupied cross points of 685 x 5,177),
    synapse count stored as a conductance;
  * KC -> output (MBON) array: 5,177 plastic weights, depressed by each code they see (the familiarity
    memory) and read every spectrum.
Non-idealities swept one at a time from the ideal (floating-point) model, plus two combined points:
  w_levels   conductance levels of the crossbar (every occupied cross point keeps at least level 1,
             so connectivity is preserved); 'f' = floating point
  o_levels   levels of the plastic output weights; depression w -> w (1 - eta) and recovery are applied
             with stochastic rounding to the level grid (expected value preserved)
  spread     device-to-device variation: every device has a fixed log-normal gain (sd in log units)
  read       cycle-to-cycle read noise on each crossbar device, Gaussian, relative sd (added at the KC
             drive with the exact variance sum_j (W_ij x_j)^2 sd^2)
  retention  each output weight relaxes toward its unwritten value 1 at rate rho per reading (loss of a
             written state; equivalent to the S5 recovery rate)
Measures: AUROC for new versus normal readings on the S1 Janus line and S1b facility streams (4,000
readings, 20 seeds, measured wiring), and the number of writes (level changes) per output device per
reading, which sets the endurance a device needs. Reference detectors (no devices): equal-memory nearest
neighbour and per-band z-score, from the same streams.
Second part (trained model, S15 task a): the fly model trained on 400 Mo probes is quantised after
training (read-out weights to signed levels, crossbar to w_levels, device spread) and scored on held-out
probes.
Usage: S1_MODE=janus|multi python s16_device.py OUT_DIR SEEDS POOL_DIR
"""
import json, os, sys, time, zlib
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IDEAL = dict(w_levels="f", o_levels="f", spread=0.0, read=0.0, retention=0.0)
SWEEP = dict(w_levels=[1, 2, 4, 8, 16], o_levels=[2, 4, 8, 16, 32, 64], spread=[0.05, 0.1, 0.2, 0.3, 0.5],
             read=[0.02, 0.05, 0.1, 0.2], retention=[1e-5, 1e-4, 1e-3, 1e-2])
COMBINED = {"typical_2D_memristor": dict(w_levels=8, o_levels=16, spread=0.2, read=0.05, retention=1e-5),
            "poor": dict(w_levels=4, o_levels=4, spread=0.3, read=0.1, retention=1e-4)}
ETA, KF = 0.3, 0.05


def configs():
    out = {"ideal": dict(IDEAL)}
    for k, vs in SWEEP.items():
        for v in vs: out[f"{k}={v}"] = dict(IDEAL, **{k: v})
    out.update({k: dict(v) for k, v in COMBINED.items()})
    return out


def quant_w(W, L):
    if L == "f": return W.copy()
    mx = W.max(); q = np.where(W > 0, np.maximum(1, np.round(W / mx * L)), 0)
    return q / L * mx


def codes(X, Min, Wdev, read, rng):
    """KC codes (top 5 %) of all readings at once on the GPU."""
    k = int(KF * Wdev.shape[0])
    P = torch.tensor(X @ Min.T, dtype=torch.float32, device=DEV)
    Wt = torch.tensor(Wdev, dtype=torch.float32, device=DEV)
    D = P @ Wt.T
    if read > 0:
        sd = torch.sqrt((P ** 2) @ (Wt ** 2).T) * read
        g = torch.Generator(device=DEV); g.manual_seed(int(rng.integers(1 << 31)))
        D = D + sd * torch.randn(D.shape, generator=g, device=DEV)
    return torch.topk(D, k, dim=1).indices.cpu().numpy()


def familiarity(C, n_kc, cfg, rng):
    L = cfg["o_levels"]; rho = cfg["retention"]
    gain = np.exp(rng.normal(0, cfg["spread"], n_kc)) if cfg["spread"] else np.ones(n_kc)
    w = np.ones(n_kc); writes = np.zeros(n_kc); sc = np.empty(len(C)); k = C.shape[1]
    for i, c in enumerate(C):
        if rho:
            if L == "f": w += rho * (1 - w)
            else:
                step = 1.0 / (L - 1); up = rng.random(n_kc) < rho * (1 - w) / step
                w = np.minimum(1.0, w + up * step); writes += up
        sc[i] = (w[c] * gain[c]).sum() / k
        t = w[c] * (1 - ETA)
        if L != "f":
            step = 1.0 / (L - 1); lo = np.floor(t / step) * step
            t = lo + step * (rng.random(k) < (t - lo) / step)
            writes[c] += t != w[c]
        else:
            writes[c] += 1
        w[c] = t
    return sc, writes


def stream_part(seed, mode, Min, W, G):
    import s1_spectra as S
    from sklearn.metrics import roc_auc_score
    st = S.stream(seed, G)
    X = np.array([x for x, _ in st]); new = np.array([k is not None for _, k in st])
    post = np.arange(len(st)) >= S.WARM
    res = {}
    for name, cfg in configs().items():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        Wq = quant_w(W, cfg["w_levels"])
        Wdev = Wq * (np.exp(rng.normal(0, cfg["spread"], Wq.shape)) if cfg["spread"] else 1.0)
        C = codes(X, Min, Wdev, cfg["read"], rng)
        sc, writes = familiarity(C, W.shape[0], cfg, rng)
        res[name] = dict(auroc=float(roc_auc_score(new[post], sc[post])),
                         writes_per_device_per_reading=float(writes.mean() / len(st)),
                         max_writes_per_device_per_reading=float(writes.max() / len(st)))
    budget = int(W.shape[0] // G)
    for name, det in (("knn_budget", S.KNN(True, budget, np.random.default_rng(3))), ("zscore", S.Z(G))):
        sc = np.array([det.step(x) for x in X]); res[name] = dict(auroc=float(roc_auc_score(new[post], sc[post])))
    return res


def trained_part(seed, Min, W, pool):
    import s15_trained as T
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    P = dict(np.load(f"{pool}/pool_Mo.npz")); y = P["fail"].astype(float)
    tr, te = T.split(P, rng, 400); te = rng.choice(te, min(1200, len(te)), replace=False)
    sub = T.draw(tr, y, 400, rng); X = T.prep(P["bands"])
    m = T.Model(W, Min, True, seed); T.train(m, X[sub], y[sub], 400)
    r0 = m.r.detach().clone(); W0 = m.W.clone(); Wn = W / W[W > 0].mean()
    res = {}
    cf = {"ideal": dict(w_levels="f", r_levels="f", spread=0.0)}
    for L in (1, 2, 4, 8, 16): cf[f"w_levels={L}"] = dict(w_levels=L, r_levels="f", spread=0.0)
    for L in (3, 5, 9, 17, 33): cf[f"r_levels={L}"] = dict(w_levels="f", r_levels=L, spread=0.0)
    for s in (0.05, 0.1, 0.2, 0.3, 0.5): cf[f"spread={s}"] = dict(w_levels="f", r_levels="f", spread=s)
    cf["typical_2D_memristor"] = dict(w_levels=8, r_levels=17, spread=0.2)
    cf["poor"] = dict(w_levels=4, r_levels=5, spread=0.3)
    for name, c in cf.items():
        g = np.random.default_rng([seed, zlib.crc32(name.encode())])
        Wq = quant_w(Wn, c["w_levels"]) * (np.exp(g.normal(0, c["spread"], Wn.shape)) if c["spread"] else 1.0)
        r = r0.cpu().numpy()
        if c["r_levels"] != "f":
            h = (c["r_levels"] - 1) // 2; mx = np.abs(r).max(); r = np.round(r / mx * h) / h * mx
        if c["spread"]: r = r * np.exp(g.normal(0, c["spread"], r.shape))
        with torch.no_grad():
            m.W.copy_(torch.tensor(Wq, dtype=torch.float32, device=DEV)); m.r.copy_(torch.tensor(r, dtype=torch.float32, device=DEV))
            res[name] = float(roc_auc_score(y[te], m(X[te]).cpu().numpy()))
            m.W.copy_(W0); m.r.copy_(r0)
    return res


def run(seed, pool):
    import s1_spectra as S
    t0 = time.time()
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn_ids)
    r = dict(seed=seed, n_kc=int(W.shape[0]), occupied=int((W > 0).sum()),
             mode=S.MODE, stream=stream_part(seed, S.MODE, Min, W, G))
    if S.MODE == "janus": r["trained"] = trained_part(seed, Min, W, pool)
    r["seconds"] = time.time() - t0
    return r


if __name__ == "__main__":
    out, seeds, pool = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    os.makedirs(out, exist_ok=True); torch.set_num_threads(2)
    for s in seeds:
        r = run(s, pool)
        json.dump(r, open(f"{out}/s16_{r['mode']}_s{s}.json", "w"), indent=1)
        print(s, round(r["seconds"]), {k: round(v["auroc"], 3) for k, v in r["stream"].items()}, {k: round(v, 3) for k, v in r.get("trained", {}).items()}, flush=True)
