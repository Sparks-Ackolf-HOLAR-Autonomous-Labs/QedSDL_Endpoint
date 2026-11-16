#!/usr/bin/env python3
"""S24 timing and bytes, one thread (run pinned to one idle Grace core with taskset; the Pi 5 was not reachable).

A  per-decision time of every S24a detector on the S1b stream, seed 1: median over readings 2,000-4,000,
   after every reservoir has filled (largest reservoir 1,664 spectra).
B  time per spectrum to serve all four jobs, and packed bytes (microcontroller accounting of
   edge_compare: uint16 indices, uint8 synapse counts and KC weights, float16 stored spectra,
   float32 regression weights; binary memories as bits):
     connectome model  one PN->KC drive; plain top-5 % code for jobs ii-iv, context code for job i;
                       i novelty weights (update), ii avoid/approach compartments, iii overlap lookup over
                       the 123 calibration codes (3 flakes x 41 steps), iv ridge read-out from the code
     engineered stack  area normalisation; i per-family z-score (update); ii logistic-regression predict;
                       iii ridge on bands; iv ridge on bands. Alternatives: routed nearest neighbour (184
                       spectra) for i, nearest calibration spectrum for iii.
   Learning costs reported separately: logistic-regression refit on 30 labelled probes (scikit-learn,
   as in s4_insitu; done once per completed run), the model's one-shot compartment write.
Usage: python s24_timing.py OUT_JSON
"""
import json, os, platform, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["S1_MODE"] = "multi"
import numpy as np  # noqa: E402
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401  (S24_WCACHE: wiring from one small npz)
import s7_swarm as S7  # noqa: E402
import s24_fixed_memory as A  # noqa: E402


def med_us(ts): return 1e6 * float(np.median(ts))


def part_a(W, Min, G, st):
    import torch
    torch.set_num_threads(1)
    dets, B = A.detectors(W, Min, G, 1)
    out = {}
    for n, d in dets.items():
        ts = []
        for i, (x, _, f) in enumerate(st):
            t = time.perf_counter(); d.step(x, f); dt = time.perf_counter() - t
            if i >= 2000: ts.append(dt)
        out[n] = dict(us_per_decision=med_us(ts), bytes=int(d.nbytes())); print("A", n, out[n], flush=True)
    return out, B


def part_b(W, Min, G, st):
    import s22_data_growth as D
    rng = np.random.default_rng(0)
    fly = A.FlyCtx(W, Min, 2.0, 0.05); n, k = fly.n, fly.k
    av = np.zeros(n, np.float32); ap = np.zeros(n, np.float32)
    av[rng.random(n) < 0.1] = 1; ap[rng.random(n) < 0.1] = 1
    NCAL = 123
    calC = np.zeros((NCAL, n), np.float32)
    for j in range(NCAL): calC[j, rng.choice(n, k, replace=False)] = 1
    calF = rng.random(NCAL).astype(np.float32)
    Bk = rng.normal(size=(n, 2)).astype(np.float32); b0 = np.zeros(2, np.float32)

    def serve_fly(x, f, Ft=0.5):
        p = np.bincount(fly.pr, weights=fly.pv * x[fly.pc], minlength=fly.npn)
        d = np.bincount(fly.r, weights=fly.v * p[fly.c], minlength=n)
        c = d >= np.partition(d, -k)[-k]                                   # plain code (jobs ii-iv)
        dc = d + fly.g * d.std() * fly.cm[f]; cc = dc >= np.partition(dc, -k)[-k]   # context code (job i)
        s = float(fly.w[cc].sum() / k); fly.w[cc] *= 1 - fly.eta           # i  novelty
        stop = av[c].mean() > ap[c].mean()                                 # ii failing-run stop
        dose = float(calF[np.argmax(calC[:, c].sum(1))]) >= Ft             # iii dose lookup
        fz = Bk[c].sum(0) + b0                                              # iv furnace end point
        return s, stop, dose, fz

    zf = [D.Z(G) for _ in range(4)]
    wl = rng.normal(size=G).astype(np.float64); bl = 0.0
    beta = rng.normal(size=G + 1); Bb = rng.normal(size=(G, 2)); mu = rng.random(G); sd = rng.random(G) + 0.5; ym = np.zeros(2)

    def serve_stack(x, f, Ft=0.5):
        v = x / (x.sum() + 1e-9)
        s = zf[f].step(x)                                                   # i  routed z-score
        stop = 1 / (1 + np.exp(-(50 * v @ wl + bl))) > 0.5                 # ii logistic regression
        dose = float(np.r_[v, 1] @ beta) >= Ft                              # iii ridge
        fz = ((v - mu) / sd) @ Bb + ym                                      # iv ridge
        return s, stop, dose, fz

    knnf = S7.Route([S.KNN(True, 184 // 4, np.random.default_rng(3 + j)) for j in range(4)])
    calB = rng.random((NCAL, G)).astype(np.float32); calB /= calB.sum(1, keepdims=True)

    def serve_stack_alt(x, f, Ft=0.5):
        v = x / (x.sum() + 1e-9)
        s = knnf.step(x, f)                                                 # i  routed nearest neighbour
        stop = 1 / (1 + np.exp(-(50 * v @ wl + bl))) > 0.5
        dose = float(calF[np.argmin(((calB - v) ** 2).sum(1))]) >= Ft      # iii nearest calibration spectrum
        fz = ((v - mu) / sd) @ Bb + ym
        return s, stop, dose, fz

    out = {}
    for name, fn in (("connectome", serve_fly), ("stack", serve_stack), ("stack_alt", serve_stack_alt)):
        ts = []
        for i, (x, _, f) in enumerate(st):
            t = time.perf_counter(); fn(x, f); dt = time.perf_counter() - t
            if i >= 2000: ts.append(dt)
        out[name] = dict(us_per_spectrum=med_us(ts)); print("B", name, out[name], flush=True)
    # learning costs
    from sklearn.linear_model import LogisticRegression
    X = rng.random((30, G)); X /= X.sum(1, keepdims=True); y = (rng.random(30) < 0.3).astype(int); y[:2] = [0, 1]
    ts = []
    for _ in range(50):
        t = time.perf_counter(); LogisticRegression(C=1.0, max_iter=500).fit(X * 50, y); ts.append(time.perf_counter() - t)
    out["logreg_refit_us"] = med_us(ts)
    c = calC[0] > 0; ts = []
    for _ in range(2000):
        t = time.perf_counter(); av[c] = 1.0; ts.append(time.perf_counter() - t)
    out["fly_compartment_write_us"] = med_us(ts)
    # bytes
    nsyn, npm = len(fly.r), len(fly.pr)
    pk = dict(wiring=nsyn * 5 + npm * 5, i_novelty_weights=n, i_context_mask=n // 4, ii_compartments_bits=2 * n // 8,
              iii_lookup_codes=NCAL * k * 2 + NCAL * 4, iv_ridge_on_code=2 * (n + 1) * 4)
    pk["total"] = sum(pk.values())
    run_b = dict(wiring=int(sum(a.nbytes for a in (fly.r, fly.c, fly.v, fly.pr, fly.pc, fly.pv))), i_novelty_weights=n * 4,
                 i_context_mask=int(fly.cm.nbytes), ii_compartments=2 * n * 8, iii_lookup_codes=int(calC.nbytes) + NCAL * 8,
                 iv_ridge_on_code=2 * (n + 1) * 8)
    run_b["total"] = sum(run_b.values())
    sk = dict(i_zscore_fam=4 * G * 2 * 4, ii_logreg_weights=(G + 1) * 4, ii_logreg_training_probes=30 * G * 4,
              iii_ridge=(G + 1) * 4, iv_ridge=2 * (G + 1) * 4)
    sk["total"] = sum(sk.values())
    sk_alt = dict(i_knn_fam_184=184 * G * 2, ii_logreg_weights=(G + 1) * 4, ii_logreg_training_probes=30 * G * 4,
                  iii_knn_calibration=NCAL * G * 2 + NCAL * 4, iv_ridge=2 * (G + 1) * 4)
    sk_alt["total"] = sum(sk_alt.values())
    out["bytes_packed"] = dict(connectome=pk, stack=sk, stack_alt=sk_alt)
    out["bytes_as_run_numpy"] = dict(connectome=run_b)
    out["notes"] = ("connectome iii stores the active-cell indices of each calibration code (uint16); iv weights float32 "
                    "with the standardisation folded in; stack ii keeps the labelled probes to refit (online SGD would need "
                    "only the 228 B of weights); stack_alt stores spectra as float16")
    return out


if __name__ == "__main__":
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
    st = S7.stream(1, G)
    a, B = part_a(W, Min, G, st)
    b = part_b(W, Min, G, st)
    res = dict(machine=platform.machine(), node=platform.node(), numpy=np.__version__,
               cpu_affinity=sorted(os.sched_getaffinity(0)), budgets=B, detectors=a, serve=b)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
