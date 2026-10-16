#!/usr/bin/env python3
"""S19: small neural networks trained online on the familiarity jobs (new-phase flag, rare-phase memory).

Question: S15-MLP and S15b compared the connectome model with small networks on
supervised stops. The familiarity jobs have no labels: a detector sees one Raman reading at a time and
must say how unfamiliar it is, with a memory that does not grow. The standard neural answer is a network
trained online on the stream whose prediction error is the novelty score. Two kinds, each with about the
5,177 stored numbers of the connectome model (one KC -> output weight per Kenyon cell):
  ae16     autoencoder 56 -> 16 -> 56 (ReLU), 1,864 weights
  ae46     autoencoder 56 -> 46 -> 56 (ReLU), 5,254 weights (equal memory)
  lstm16   LSTM, 16 hidden units, reads the previous 8 readings and predicts the current one (the usual
           LSTM forecasting detector), 5,688 weights; short-term memory = hidden state, long-term = weights
  lstmae16 same LSTM, but the window ends with the current reading, which it reconstructs (an LSTM
           autoencoder with the recent readings as context), 5,688 weights
Per reading: area-normalised bands (mean 1), score = mean squared error before the update, then one
Adam step (lr 1e-3) on that reading. No pre-training, no replay buffer, no labels.
Streams, unchanged and with the same seeds:
  s1b  S1b shared-facility stream (four families, 4,000 readings, 5 % new events, laser drop at 2,000);
       measures as s1_spectra.run (compare with results/reflex/s1b_multi/s1_s<seed>.json)
  s5b  S5b week-long stream (10^5 readings, three rare phases returning after 1e3 / 1e4 / 6e4 readings);
       measures as s5b_return (compare with results/reflex/s5b_return/s5b_s<seed>_*.json); added for the
       networks: contrast = episode mean score / mean score of the routine readings in the 500 readings
       before it, and contrast_ratio = return contrast / first contrast. A network trained online lowers
       its error on everything as it trains, so its plain return ratio mixes memory of the phase with
       that general drift; the contrast ratio removes the drift.
Options (added after the first S19 run), set in the environment; defaults reproduce S19:
  S19_LR      Adam learning rate (default 1e-3)       S19_WIN  LSTM window in readings (default 8)
  S19_REPLAY  replay buffer of B past readings, filled by reservoir sampling; each step also trains on
              a minibatch of 8 of them (default 0 = no replay). The buffer counts as stored numbers.
  S19_TAG     suffix of the output file (default none)
Usage: python s19_online_nets.py MODE OUT_DIR SEED DETECTOR     (MODE s1b | s5b; DETECTOR ae16 | ae46 | lstm16 | lstmae16)
"""
import json, os, platform, sys, time
from collections import deque
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s5b_return as R  # noqa: E402  (sets S1_MODE=multi before s1_spectra is imported)
import s1_spectra as S  # noqa: E402
from s5_long import auroc  # noqa: E402

torch.set_num_threads(1)
LR = float(os.environ.get("S19_LR", 1e-3)); WIN = int(os.environ.get("S19_WIN", 8))
REPLAY = int(os.environ.get("S19_REPLAY", 0)); TAG = os.environ.get("S19_TAG", "")


class Online:
    def __init__(self, kind, G, seed):
        torch.manual_seed(seed); self.kind = kind; self.buf_r, self.seen = [], 0
        self.rrng = np.random.default_rng(seed + 777)
        if kind.startswith("ae"):
            h = int(kind[2:])
            self.net = torch.nn.Sequential(torch.nn.Linear(G, h), torch.nn.ReLU(), torch.nn.Linear(h, G))
            self.mods = [self.net]
        else:
            h = int(kind.lstrip("lstmae")); self.T = WIN; self.buf = deque(maxlen=self.T)
            self.lstm = torch.nn.LSTM(G, h, batch_first=True); self.head = torch.nn.Linear(h, G)
            self.mods = [self.lstm, self.head]
        self.opt = torch.optim.Adam([p for m in self.mods for p in m.parameters()], lr=LR)

    def step(self, x):
        v = torch.tensor(x / (x.mean() + 1e-12), dtype=torch.float32)
        if self.kind.startswith("ae"):
            pred = self.net(v[None])[0]
        else:
            if self.kind.startswith("lstmae"): self.buf.append(v)
            elif not self.buf: self.buf.append(v); return 1.0
            o, _ = self.lstm(torch.stack(list(self.buf))[None]); pred = self.head(o[0, -1])
            if not self.kind.startswith("lstmae"): self.buf.append(v)
        loss = ((pred - v) ** 2).mean(); score = float(loss.detach())
        if REPLAY and self.kind.startswith("ae") and self.buf_r:
            idx = self.rrng.choice(len(self.buf_r), min(8, len(self.buf_r)), replace=False)
            R = torch.stack([self.buf_r[i] for i in idx]); loss = loss + ((self.net(R) - R) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if REPLAY and self.kind.startswith("ae"):  # reservoir sampling keeps a uniform sample of the stream
            self.seen += 1
            if len(self.buf_r) < REPLAY: self.buf_r.append(v)
            else:
                j = int(self.rrng.integers(self.seen))
                if j < REPLAY: self.buf_r[j] = v
        return score

    def stored(self):
        return int(sum(p.numel() for m in self.mods for p in m.parameters())) + REPLAY * (self.net[0].in_features if self.kind.startswith("ae") else 0)


def run_s1b(seed, name, out):
    from sklearn.metrics import roc_auc_score
    _, _, G = S.glom_map(np.random.default_rng(5)); st = S.stream(seed, G); det = Online(name, G, seed)
    sc, ms = [], 0.0
    for i, (x, _) in enumerate(st):
        t0 = time.perf_counter(); sc.append(det.step(x))
        if i >= S.WARM: ms += (time.perf_counter() - t0) * 1e3
    N, WARM, KINDS = S.N, S.WARM, S.KINDS
    kinds = np.array([k if k else "" for _, k in st]); new = kinds != ""
    idx = np.arange(N); post = idx >= WARM; h = N // 2
    pre_win = post & ~new & (idx >= h - 300) & (idx < h); post_win = ~new & (idx >= h) & (idx < h + 300)
    x = np.array(sc); thr = np.quantile(x[post & ~new], 0.95); thr_pre = np.quantile(x[pre_win], 0.95)
    d = dict(auroc=float(roc_auc_score(new[post], x[post])), recall_at_5fpr=float(np.mean(x[post & new] > thr)),
             recall_by_kind={k: float(np.mean(x[post & (kinds == k)] > thr)) for k in KINDS},
             auroc_by_kind={k: float(roc_auc_score(np.r_[np.ones((post & (kinds == k)).sum()), np.zeros((post & ~new).sum())],
                                                   np.r_[x[post & (kinds == k)], x[post & ~new]])) for k in KINDS},
             false_alarm_after_change=float(np.mean(x[post_win] > thr_pre)),
             ms_per_reading=ms / (N - WARM), stored_numbers=det.stored())
    json.dump(dict(seed=seed, G=G, det={name: d}), open(f"{out}/s19_s1b_s{seed}_{name}{TAG}.json", "w"), indent=1)


def run_s5b(seed, name, out):
    N = int(os.environ.get("S5B_N", 100000))
    _, _, G = S.glom_map(np.random.default_rng(5)); det = Online(name, G, seed)
    sc, kinds, rares, t = [], [], [], 0.0
    for i, (x, kind, rare) in enumerate(R.gen(seed, G, N)):
        t0 = time.perf_counter(); sc.append(det.step(x)); t += time.perf_counter() - t0
        kinds.append(kind or ""); rares.append(rare)
        if (i + 1) % 10000 == 0: print(name, seed, i + 1, round(1e3 * t / (i + 1), 2), flush=True)
    x = np.array(sc); new = np.array(kinds) != ""; isr = np.array([r is not None for r in rares])
    B = 10000; normal = ~new & ~isr
    thr = float(np.quantile(x[S.WARM:B][normal[S.WARM:B]], 0.95))
    blocks = []
    for b in range(N // B):
        sl = slice(max(b * B, S.WARM), (b + 1) * B); xb, nb, ob = x[sl], new[sl], normal[sl]
        blocks.append(dict(auroc=auroc(xb[nb], xb[ob]), fa=float(np.mean(xb[ob] > thr))))
    rare_res = {}
    for k in R.RARE:
        r = {}
        for e in ("first", "return"):
            idx = [i for i, v in enumerate(rares) if v == (k, e)]
            if idx:
                base = float(x[max(idx[0] - 500, 0):idx[0]][normal[max(idx[0] - 500, 0):idx[0]]].mean())
                r[e] = dict(recall=float(np.mean(x[idx] > thr)), mean_score=float(x[idx].mean()), baseline=base,
                            contrast=float(x[idx].mean()) / max(base, 1e-12))
        if "return" in r:
            r["return_ratio"] = r["return"]["mean_score"] / max(r["first"]["mean_score"], 1e-12)
            r["contrast_ratio"] = r["return"]["contrast"] / max(r["first"]["contrast"], 1e-12)
        rare_res[k] = r
    json.dump(dict(seed=seed, N=N, det=name, host=platform.node(), thr=thr, ms=1e3 * t / N, stored=det.stored(),
                   blocks=blocks, rare=rare_res), open(f"{out}/s5b_s{seed}_{name}{TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    mode, out, seed, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    (run_s1b if mode == "s1b" else run_s5b)(seed, name, out)
