#!/usr/bin/env python3
"""S15-MLP: small trained neural networks on the S15 selenization-stop task.

Question: is a small conventional neural network compared with the connectome model?
S15 compared the trained connectome model with logistic regression, 5-nearest-neighbour and a dense random
expansion with a trained read-out, but not with a small multilayer perceptron trained end to end. This
script adds two, on exactly the S15 task-a splits (same seeds, same random-number sequence, same pools):
  mlp32     56 bands -> 32 ReLU -> 1               (1,857 trained numbers)
  mlp64x2   56 bands -> 64 ReLU -> 64 ReLU -> 1    (7,873 trained numbers, about the 8,370 of the
            trained connectome model)
Training as for the connectome model in S15: full batch, Adam (lr 3e-3), 400 steps, class-weighted
binary cross-entropy, weight decay 1e-3; transfer MoS2 -> WS2 as in S15 (zero shot, fine-tune 200 steps
on 25 W probes, scratch on those 25). Inputs are the area-normalised bands used by every S15 arm.
Usage: python s15_mlp.py OUT_DIR SEEDS POOL_DIR
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s15_trained import prep, split, draw, NS, DEV  # noqa: E402

NETS = {"mlp32": [32], "mlp64x2": [64, 64]}


class MLP(torch.nn.Module):
    def __init__(self, hidden, seed):
        super().__init__(); torch.manual_seed(seed)
        layers, d = [], 56
        for h in hidden: layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]; d = h
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers).to(DEV)

    def forward(self, x): return self.net(x).squeeze(1)


def fit(m, X, y, steps):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    pw = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32, device=DEV)
    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw); yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    for _ in range(steps):
        opt.zero_grad(); lf(m(X), yt).backward(); opt.step()


def auc(m, X, y):
    from sklearn.metrics import roc_auc_score
    with torch.no_grad(): return float(roc_auc_score(y, m(X).cpu().numpy()))


def task_a(seed, pool):
    rng = np.random.default_rng(seed)                       # same call order as s15_trained.task_a
    Pm = dict(np.load(f"{pool}/pool_Mo.npz")); Pw = dict(np.load(f"{pool}/pool_W.npz"))
    ym, yw = Pm["fail"].astype(float), Pw["fail"].astype(float)
    tr_m, te_m = split(Pm, rng, 400); tr_w, te_w = split(Pw, rng, 400)
    te_m = rng.choice(te_m, min(1200, len(te_m)), replace=False)
    Xm, Xw = prep(Pm["bands"]), prep(Pw["bands"])
    subs = {n: draw(tr_m, ym, n, rng) for n in NS}
    few_w = draw(tr_w, yw, 25, rng)
    out = {}
    for name, hid in NETS.items():
        eff = {}
        for n in NS:
            m = MLP(hid, seed); fit(m, Xm[subs[n]], ym[subs[n]], 400); eff[str(n)] = auc(m, Xm[te_m], ym[te_m])
            if n == 400:
                zero = auc(m, Xw[te_w], yw[te_w]); fit(m, Xw[few_w], yw[few_w], 200); fine = auc(m, Xw[te_w], yw[te_w])
        m = MLP(hid, seed); fit(m, Xw[few_w], yw[few_w], 400)
        out[name] = dict(n_params=int(sum(p.numel() for p in m.parameters())), auroc_by_n=eff,
                         w_zero_shot=zero, w_fine_tuned=fine, w_scratch=auc(m, Xw[te_w], yw[te_w]))
    return out


if __name__ == "__main__":
    out, seeds, pool = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    os.makedirs(out, exist_ok=True); torch.set_num_threads(2)
    for s in seeds:
        t0 = time.time(); r = dict(seed=s, device=DEV, arms=task_a(s, pool)); r["seconds"] = time.time() - t0
        json.dump(r, open(f"{out}/s15mlp_s{s}.json", "w"), indent=1)
        print(s, round(r["seconds"]), {a: (d["auroc_by_n"]["25"], d["auroc_by_n"]["100"]) for a, d in r["arms"].items()}, flush=True)
