#!/usr/bin/env python3
"""S26c: an early-exit network (BranchyNet; Teerapittayanon, McDanel and Kung, arXiv 1709.01686) on the S15 failing-run
stop, on exactly the S15 task-a splits (same seeds, pools and random-number sequence as s15_mlp.py).
  branchy   56 bands -> 64 ReLU -> exit 1 (linear) ; -> 64 ReLU -> exit 2 (linear); 8,002 trained numbers
            Both exits trained jointly (sum of the two class-weighted BCE losses, BranchyNet's joint training with
            equal weights); otherwise as s15_mlp (Adam 3e-3, 400 steps, weight decay 1e-3, full batch).
  mlp64x2   re-run of s15_mlp's two-layer perceptron as a reproduction check.
Inference: a probe leaves at exit 1 when the binary entropy (bits) of exit 1 is below T, else it goes on to exit 2.
T = 0.2 bits is fixed in advance as the main setting; a sweep over T is recorded. Score for AUROC: the logit of the
exit taken. Time per probe: NumPy, one thread, batch 1, on the machine that runs the script.
Usage: python s26c_branchynet.py OUT_DIR SEEDS POOL_DIR"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s15_trained import prep, split, draw, NS, DEV  # noqa: E402
from s15_mlp import MLP, fit, auc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

T_MAIN, T_SWEEP = 0.2, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


class Branchy(torch.nn.Module):
    def __init__(self, seed):
        super().__init__(); torch.manual_seed(seed)
        self.l1, self.e1 = torch.nn.Linear(56, 64), torch.nn.Linear(64, 1)
        self.l2, self.e2 = torch.nn.Linear(64, 64), torch.nn.Linear(64, 1)
        self.to(DEV)

    def forward(self, x):
        h = torch.relu(self.l1(x)); g = torch.relu(self.l2(h))
        return self.e1(h).squeeze(1), self.e2(g).squeeze(1)


def fit_branchy(m, X, y, steps=400):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32, device=DEV)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw); yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    for _ in range(steps):
        opt.zero_grad(); z1, z2 = m(X); (lf(z1, yt) + lf(z2, yt)).backward(); opt.step()


def h_bits(z):
    p = 1 / (1 + np.exp(-z)); p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def evaluate(m, X, y):
    with torch.no_grad(): z1, z2 = (t.cpu().numpy() for t in m(X))
    res = {"auroc_exit1": float(roc_auc_score(y, z1)), "auroc_exit2": float(roc_auc_score(y, z2))}
    for T in T_SWEEP:
        early = h_bits(z1) < T; s = np.where(early, z1, z2)
        res[f"T={T}"] = {"auroc": float(roc_auc_score(y, s)), "exit1_frac": float(early.mean())}
    return res


def timing(m):
    """Per-probe time of the early path and the full path in NumPy, one thread, batch 1."""
    W = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    x = np.random.default_rng(0).random(56).astype(np.float32)

    def early(x): h = np.maximum(W["l1.weight"] @ x + W["l1.bias"], 0); return W["e1.weight"] @ h + W["e1.bias"], h

    def full(x):
        z1, h = early(x); g = np.maximum(W["l2.weight"] @ h + W["l2.bias"], 0); return W["e2.weight"] @ g + W["e2.bias"]
    out = {}
    for name, f in (("early_us", early), ("full_us", full)):
        for _ in range(200): f(x)
        t0 = time.perf_counter(); [f(x) for _ in range(5000)]; out[name] = (time.perf_counter() - t0) / 5000 * 1e6
    return out


def task_a(seed, pool):
    rng = np.random.default_rng(seed)                       # same call order as s15_trained.task_a / s15_mlp.task_a
    Pm = dict(np.load(f"{pool}/pool_Mo.npz")); Pw = dict(np.load(f"{pool}/pool_W.npz"))
    ym, yw = Pm["fail"].astype(float), Pw["fail"].astype(float)
    tr_m, te_m = split(Pm, rng, 400); tr_w, te_w = split(Pw, rng, 400)
    te_m = rng.choice(te_m, min(1200, len(te_m)), replace=False)
    Xm = prep(Pm["bands"])
    subs = {n: draw(tr_m, ym, n, rng) for n in NS}
    draw(tr_w, yw, 25, rng)                                 # keep the random-number sequence of s15_mlp
    out = {"branchy": {}, "mlp64x2": {}}
    for n in NS:
        b = Branchy(seed); fit_branchy(b, Xm[subs[n]], ym[subs[n]]); out["branchy"][str(n)] = evaluate(b, Xm[te_m], ym[te_m])
        m = MLP([64, 64], seed); fit(m, Xm[subs[n]], ym[subs[n]], 400); out["mlp64x2"][str(n)] = auc(m, Xm[te_m], ym[te_m])
        if n == 400: out["timing"] = timing(b)
    out["n_params"] = int(sum(p.numel() for p in Branchy(seed).parameters()))
    return out


if __name__ == "__main__":
    outd, seeds, pool = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    os.makedirs(outd, exist_ok=True); torch.set_num_threads(2)
    for s in seeds:
        r = dict(seed=s, device=DEV, **task_a(s, pool))
        json.dump(r, open(f"{outd}/s26c_s{s}.json", "w"), indent=1)
        b = r["branchy"]
        print(s, {n: (round(b[n]["auroc_exit2"], 3), round(b[n][f"T={T_MAIN}"]["auroc"], 3), round(b[n][f"T={T_MAIN}"]["exit1_frac"], 2),
                      round(r["mlp64x2"][n], 3)) for n in ("25", "100", "400")}, {k: round(v, 2) for k, v in r["timing"].items()}, flush=True)
