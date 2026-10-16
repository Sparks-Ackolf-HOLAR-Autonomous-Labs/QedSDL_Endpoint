#!/usr/bin/env python3
"""S15: train the connectome model on fixed wiring.

Question: once the band-to-glomerulus input map and the read-out are learned from data, does the measured
FlyWire projection-neuron (PN) to Kenyon-cell (KC) wiring train faster, need fewer examples, or transfer
better to a new material than a degree-preserving shuffle, a random sparse graph or a dense random
expansion of the same size? S1-S14 set the input map by hand (alphabetical glomerulus order, an order with
no physical meaning, S14), so they could not answer this.

Model (wiring fixed, everything else trained by gradient descent, PyTorch):
  x       56-band in-situ Raman probe, area-normalised (mean 1 per band)
  glom    softplus(A x + b): A is a 56 x 56 band-to-glomerulus map, initialised to the alphabetical map
          of S1-S14 (identity); one set of parameters per glomerulus type, shared by every PN of that type
  PN      Min @ glom (685 PNs; uniglomerular PNs read their glomerulus, multiglomerular the mean of 3)
  KC      W @ PN with W the fixed PN->KC synapse counts; top 5 % kept (APL-like threshold, recomputed
          for every spectrum), activity normalised
  output  KC @ r + c, one read-out weight per KC (the KC -> output-neuron synapses)
Trained: A, b, r, c (3,192 + 5,178 numbers). Wiring arms: fly (measured), shuf (degree-preserving
shuffle), rand (random sparse, same synapse count), dense (Gaussian all-to-all, same KC count). Arm
fly_fixmap keeps A at the alphabetical map (only the read-out learns: the S1-S14 setting with a trained
read-out). Baselines on the same area-normalised bands: logistic regression and 5-nearest-neighbour.

Task a (selenization stop, S4/S4L): label = the anneal will destroy the flake. Pools from s15_pool.py.
  * data efficiency: train on n = 25, 50, 100, 200, 400 Mo probes; AUROC on 1,200 held-out Mo probes
    (held-out recipes, so no recipe appears in both sets)
  * training speed: held-out AUROC after 25, 50, 100, 200, 400 optimiser steps at n = 100
  * transfer MoS2 -> WS2 selenization (W kMC parameters, W lines): zero-shot AUROC of the n = 400 Mo
    model on W probes; then fine-tuned with 25 W probes (200 steps); against the same model trained on the
    25 W probes alone
Task b (Ar+ dose stop, S11, Daniel et al. 2026): regression of effective fluence from the in-situ band
  vector, trained on 1 or 3 calibration flakes; stop at the first step whose estimate reaches the target;
  success = stop within +-15 %. 30 test flakes per seed, the S11 flake model unchanged.
Usage: python s15_trained.py OUT_DIR SEEDS POOL_DIR
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s11_irradiation as I  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = [25, 50, 100, 200, 400]
STEPS = [25, 50, 100, 200, 400]
ARMS = ["fly", "fly_fixmap", "shuf", "rand", "dense"]


def wirings(seed, W):
    rng = np.random.default_rng(1000 + seed)
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); np.add.at(Wr, (rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))), w)
    Wd = np.abs(rng.normal(size=W.shape)) * (w.sum() / W.size)
    return dict(fly=W, fly_fixmap=W, shuf=Ws, rand=Wr, dense=Wd)


class Model(torch.nn.Module):
    def __init__(self, W, Min, learn_map=True, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        G = Min.shape[1]
        self.W = torch.tensor(W / W[W > 0].mean(), dtype=torch.float32, device=DEV)
        self.M = torch.tensor(Min, dtype=torch.float32, device=DEV)
        self.A = torch.nn.Parameter(torch.eye(G, device=DEV) + 0.01 * torch.randn(G, G, device=DEV), requires_grad=learn_map)
        self.b = torch.nn.Parameter(torch.zeros(G, device=DEV), requires_grad=learn_map)
        self.r = torch.nn.Parameter(torch.zeros(W.shape[0], device=DEV)); self.c = torch.nn.Parameter(torch.zeros(1, device=DEV))
        self.k = int(S.K_FRAC * W.shape[0]); self.learn_map = learn_map

    def kc(self, x):
        g = torch.nn.functional.softplus(x @ self.A.T + self.b) if self.learn_map else x
        d = (g @ self.M.T) @ self.W.T
        th = torch.topk(d, self.k, dim=1).values[:, -1:].detach()
        a = torch.relu(d - th + 1e-6)
        return a / (a.sum(1, keepdim=True) + 1e-9) * self.k

    def forward(self, x): return self.kc(x) @ self.r + self.c

    def reg(self):
        G = self.A.shape[0]
        return 1e-3 * (self.r ** 2).mean() + (1e-3 * ((self.A - torch.eye(G, device=DEV)) ** 2).sum() if self.learn_map else 0.0)


def prep(B):
    B = np.asarray(B, np.float64); return torch.tensor(B / (B.mean(1, keepdims=True) + 1e-12), dtype=torch.float32, device=DEV)


def train(m, X, y, steps, loss="bce", lr=3e-3, eval_at=None, Xt=None, yt=None):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    yt_ = torch.tensor(y, dtype=torch.float32, device=DEV)
    if loss == "bce":
        pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32, device=DEV)
        lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        lf = torch.nn.MSELoss()
    curve = {}
    for s in range(1, steps + 1):
        opt.zero_grad(); L = lf(m(X), yt_) + m.reg(); L.backward(); opt.step()
        if eval_at and s in eval_at: curve[s] = auroc(m, Xt, yt)
    return curve


def auroc(m, X, y):
    from sklearn.metrics import roc_auc_score
    with torch.no_grad(): return float(roc_auc_score(y, m(X).cpu().numpy()))


def base_scores(Xtr, ytr, Xte, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr).decision_function(Xte)
    k = min(5, len(ytr)); return KNeighborsClassifier(k).fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def split(P, rng, n_test_rec):
    rec = np.unique(P["recipe"]); te = set(rng.choice(rec, n_test_rec, replace=False))
    m = np.array([r in te for r in P["recipe"]]); return np.where(~m)[0], np.where(m)[0]


def draw(idx, y, n, rng):
    while True:
        s = rng.choice(idx, n, replace=False)
        if 0 < y[s].sum() < n: return s


def task_a(seed, Wd, Min, pool):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    Pm = dict(np.load(f"{pool}/pool_Mo.npz")); Pw = dict(np.load(f"{pool}/pool_W.npz"))
    ym, yw = Pm["fail"].astype(float), Pw["fail"].astype(float)
    tr_m, te_m = split(Pm, rng, 400); tr_w, te_w = split(Pw, rng, 400)
    te_m = rng.choice(te_m, min(1200, len(te_m)), replace=False)
    Xm, Xw = prep(Pm["bands"]), prep(Pw["bands"])
    Bm = Pm["bands"] / Pm["bands"].mean(1, keepdims=True); Bw = Pw["bands"] / Pw["bands"].mean(1, keepdims=True)
    out = {a: {} for a in ARMS + ["logreg", "knn"]}
    subs = {n: draw(tr_m, ym, n, rng) for n in NS}
    few_w = draw(tr_w, yw, 25, rng)
    for a in ARMS:
        eff = {}
        for n in NS:
            m = Model(Wd[a], Min, a != "fly_fixmap", seed); train(m, Xm[subs[n]], ym[subs[n]], 400)
            eff[n] = auroc(m, Xm[te_m], ym[te_m])
            if n == 400:
                zero = auroc(m, Xw[te_w], yw[te_w])
                train(m, Xw[few_w], yw[few_w], 200); fine = auroc(m, Xw[te_w], yw[te_w])
        m = Model(Wd[a], Min, a != "fly_fixmap", seed)
        curve = train(m, Xm[subs[100]], ym[subs[100]], max(STEPS), eval_at=set(STEPS), Xt=Xm[te_m], yt=ym[te_m])
        m = Model(Wd[a], Min, a != "fly_fixmap", seed); train(m, Xw[few_w], yw[few_w], 400)
        out[a] = dict(auroc_by_n={str(k): v for k, v in eff.items()}, auroc_by_step={str(k): v for k, v in curve.items()},
                      w_zero_shot=zero, w_fine_tuned=fine, w_scratch=auroc(m, Xw[te_w], yw[te_w]))
    for b in ("logreg", "knn"):
        eff = {str(n): float(roc_auc_score(ym[te_m], base_scores(Bm[subs[n]], ym[subs[n]], Bm[te_m], b))) for n in NS}
        zero = float(roc_auc_score(yw[te_w], base_scores(Bm[subs[400]], ym[subs[400]], Bw[te_w], b)))
        Xc = np.r_[Bm[subs[400]], Bw[few_w]]; yc = np.r_[ym[subs[400]], yw[few_w]]
        fine = float(roc_auc_score(yw[te_w], base_scores(Xc, yc, Bw[te_w], b)))
        scr = float(roc_auc_score(yw[te_w], base_scores(Bw[few_w], yw[few_w], Bw[te_w], b)))
        out[b] = dict(auroc_by_n=eff, w_zero_shot=zero, w_fine_tuned=fine, w_scratch=scr)
    return dict(fail_rate_mo=float(ym.mean()), fail_rate_w=float(yw.mean()), n_test_mo=int(len(te_m)), n_test_w=int(len(te_w)), arms=out)


def task_b(seed, Wd, Min, G, n_cal):
    I.MODEL = I.MODEL or I.fit_digitised()
    rng = np.random.default_rng([seed, n_cal])
    newflake = lambda: dict(e=float(np.exp(rng.normal(0, I.E_SD))), dpos=float(rng.normal(0, I.POS_SD)))  # noqa: E731
    calF, calY = [], []
    for _ in range(n_cal):
        fl = newflake()
        for k in range(int(10 / I.STEP) + 1):
            Fe = fl["e"] * k * I.STEP; calF.append(Fe); calY.append(I.spectrum(Fe, fl, rng))
    calF = np.array(calF); calB = np.array([S.bands(y, G) for y in calY]); calN = np.array([I.norm(b) for b in calB])
    calR = np.array([I.ratio(y) for y in calY]); a_, b_ = np.polyfit(calF, calR, 1)[::-1]
    Xr = np.c_[calN, np.ones(len(calN))]; beta = np.linalg.solve(Xr.T @ Xr + 1e-4 * np.eye(Xr.shape[1]), Xr.T @ calF)
    models = {}
    for a in ARMS:
        m = Model(Wd[a], Min, a != "fly_fixmap", seed); train(m, prep(calB), calF / 10.0, 400, loss="mse"); models[a] = m
    arms = ["ratio", "ridge", "knn"] + ARMS
    res = {a: [] for a in arms}
    for t in range(I.N_TEST):
        fl = newflake(); Ft = float(rng.uniform(2, 6)); srng = np.random.default_rng([seed, n_cal, t])
        done = {a: None for a in arms}; k = 0
        while k * I.STEP <= I.FMAX and any(v is None for v in done.values()):
            k += 1; Fe = fl["e"] * k * I.STEP
            y = I.spectrum(Fe, fl, srng); bands = S.bands(y, G); bn = I.norm(bands)
            est = dict(ratio=(I.ratio(y) - a_) / b_, ridge=float(np.r_[bn, 1] @ beta),
                       knn=float(calF[np.argmin(((calN - bn) ** 2).sum(1))]))
            with torch.no_grad():
                Xb = prep(bands[None])
                for a in ARMS: est[a] = 10.0 * float(models[a](Xb)[0])
            for a in arms:
                if done[a] is None and (est[a] >= Ft or k * I.STEP >= I.FMAX): done[a] = Fe
        for a in arms: res[a].append((done[a] - Ft) / Ft)
    return {a: dict(success=float(np.mean(np.abs(v) <= I.TOL)), lost=float(np.mean(np.array(v) > I.TOL)),
                    mae=float(np.mean(np.abs(v)))) for a, v in res.items()}


def run(seed, pool):
    t0 = time.time()
    pn_ids, Min, G = S.glom_map(np.random.default_rng(5))
    Wd = wirings(seed, S.load_w(pn_ids))
    r = dict(seed=seed, G=G, device=DEV, task_a=task_a(seed, Wd, Min, pool),
             task_b={str(n): task_b(seed, Wd, Min, G, n) for n in (1, 3)})
    r["seconds"] = time.time() - t0
    return r


if __name__ == "__main__":
    out, seeds, pool = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    os.makedirs(out, exist_ok=True); torch.set_num_threads(2)
    for s in seeds:
        r = run(s, pool)
        json.dump(r, open(f"{out}/s15_s{s}.json", "w"), indent=1)
        A = r["task_a"]["arms"]
        print(s, round(r["seconds"]), {a: (round(d["auroc_by_n"]["100"], 3), round(d["w_zero_shot"], 3), round(d["w_fine_tuned"], 3)) for a, d in A.items()},
              {a: round(d["success"], 2) for a, d in r["task_b"]["3"].items()}, flush=True)
