#!/usr/bin/env python3
"""S15b: small trained neural networks on the Ar+ irradiation-dose stop (S11 / S15 task b).

Question: S15-MLP added small perceptrons to the selenization stop (S15 task a)
but not to the irradiation-dose task. This script runs the two S15-MLP networks as fluence regressors on
exactly the S15 task-b flakes (same seeds, same random-number sequence, same 1 or 3 calibration flakes,
same 30 test flakes per seed, same stop rule):
  mlp32     56 bands -> 32 ReLU -> 1               (1,857 trained numbers)
  mlp64x2   56 bands -> 64 ReLU -> 64 ReLU -> 1    (7,873 trained numbers; trained connectome model 8,370)
Training as for the connectome model in S15 task b: area-normalised bands, target fluence / 10, MSE,
full batch, Adam (lr 3e-3, weight decay 1e-3), 400 steps. The ratio, ridge and knn arms are recomputed as
a check that the flakes are identical to S15 (their numbers must match s15_s<seed>.json).
Usage: python s15b_dose_mlp.py OUT_DIR SEEDS
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s11_irradiation as I  # noqa: E402
from s15_trained import prep, DEV  # noqa: E402
from s15_mlp import MLP, NETS  # noqa: E402


def fit_reg(m, X, y, steps=400):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV); lf = torch.nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad(); lf(m(X), yt).backward(); opt.step()


def task_b(seed, G, n_cal):
    I.MODEL = I.MODEL or I.fit_digitised()
    rng = np.random.default_rng([seed, n_cal])              # same call order as s15_trained.task_b
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
    for a, hid in NETS.items():
        m = MLP(hid, seed); fit_reg(m, prep(calB), calF / 10.0); models[a] = m
    arms = ["ratio", "ridge", "knn"] + list(NETS)
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
                for a in NETS: est[a] = 10.0 * float(models[a](Xb)[0])
            for a in arms:
                if done[a] is None and (est[a] >= Ft or k * I.STEP >= I.FMAX): done[a] = Fe
        for a in arms: res[a].append((done[a] - Ft) / Ft)
    return {a: dict(success=float(np.mean(np.abs(v) <= I.TOL)), lost=float(np.mean(np.array(v) > I.TOL)),
                    mae=float(np.mean(np.abs(v)))) for a, v in res.items()}


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True); torch.set_num_threads(2)
    _, _, G = S.glom_map(np.random.default_rng(5))
    for s in seeds:
        t0 = time.time(); r = dict(seed=s, G=G, device=DEV, task_b={str(n): task_b(s, G, n) for n in (1, 3)})
        r["seconds"] = time.time() - t0
        json.dump(r, open(f"{out}/s15b_s{s}.json", "w"), indent=1)
        print(s, round(r["seconds"]), {a: round(d["mae"], 3) for a, d in r["task_b"]["3"].items()}, flush=True)
