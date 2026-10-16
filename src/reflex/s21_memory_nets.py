#!/usr/bin/env python3
"""S21: memory-augmented and recurrent networks trained online on the familiarity jobs, with four
replay designs.

Question (from S19b): compare the connectome model with a
network that has a separate fast memory (memory-augmented network) and with other architectures that
carry memory (GRU, peephole LSTM, Mogrifier LSTM, transformer), each with several replay designs, to see
where the connectome model's memory helps and where it does not.

Detectors (memnets.py; each sees one reading at a time, scores it by its reconstruction error before the
update, then takes one Adam step; no pre-training, no labels; 16 hidden units or a 16-number code):
  ae16      plain autoencoder (S19b baseline)                       lstm      standard LSTM (S19b baseline)
  gru       GRU                                                    peep      peephole LSTM
  mog       Mogrifier LSTM                                         tfm       one-layer transformer
  memae     autoencoder reading a 200-slot memory learned by gradient (MemAE)
  memfast   autoencoder reading a 200-slot memory written in one shot (separate fast memory)
  fastw     autoencoder plus delta-rule fast weights (forgetting factor LAM)
Sequence detectors read the last WIN = 4 readings, ending with the current one, and reconstruct the
current reading (the S19b "LSTM reconstructing" form; forecasting the next reading failed in S19).
Replay designs (each step also trains on a minibatch of 8 items drawn from a buffer of B = 64 items;
an item is one reading, or one window for the sequence detectors; the buffer counts as stored numbers):
  none      no replay
  res       reservoir sampling: a uniform sample of everything seen (S19b design)
  fifo      the most recent B items (short-term rehearsal)
  prio      surprise-prioritised: items enter by their novelty score, scores decay by 0.999 per reading,
            the lowest-priority item is evicted, draws are proportional to priority (after Schaul et al. 2016)
Options (environment): S21_LR learning rate (default 3e-3), S21_TAU fast-slot write threshold (0.9),
S21_LAM fast-weight forgetting factor (0.999), S21_TAG output suffix.
Streams and measures: as S19 (s19_online_nets.run_s1b / run_s5b), unchanged, same seeds.
Usage: python s21_memory_nets.py MODE OUT_DIR SEED DETECTOR REPLAY   (MODE s1b | s5b)
"""
import os, sys
from collections import deque
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s19_online_nets as B  # noqa: E402
import memnets as MN  # noqa: E402

torch.set_num_threads(1)
LR = float(os.environ.get("S21_LR", 3e-3)); TAU = float(os.environ.get("S21_TAU", 0.9))
LAM = float(os.environ.get("S21_LAM", 0.999)); WIN, BUF, MB, H, K = 4, 64, 8, 16, 200


class Online21:
    def __init__(self, kind, G, seed, replay="none"):
        torch.manual_seed(seed); self.kind, self.replay, self.G = kind, replay, G
        self.rng = np.random.default_rng(seed + 777); self.items, self.prio, self.seen = [], [], 0
        self.seq = kind in MN.SEQ
        if self.seq:
            self.cell = MN.SEQ[kind](G, H); self.head = torch.nn.Linear(H, G); self.win = deque(maxlen=WIN)
            self.mods = [self.cell, self.head]
        elif kind == "ae16":
            self.net = torch.nn.Sequential(torch.nn.Linear(G, H), torch.nn.ReLU(), torch.nn.Linear(H, G)); self.mods = [self.net]
        elif kind in ("memae", "memfast"):
            self.net = MN.MemAE(G, H, K, fast=kind == "memfast", tau=TAU); self.mods = [self.net]
        elif kind == "fastw":
            self.net = MN.FastWeight(G, H, lam=LAM); self.mods = [self.net]
        else:
            raise ValueError(kind)
        self.opt = torch.optim.Adam([p for m in self.mods for p in m.parameters()], lr=LR)

    def predict(self, X):  # X: (B, G) or (B, T, G); returns reconstruction of the last reading
        if self.seq: return self.head(self.cell(X))
        if self.kind == "ae16": return self.net(X)
        return self.net(X)[0]

    def step(self, x):
        v = torch.tensor(x / (x.mean() + 1e-12), dtype=torch.float32)
        if self.seq:
            self.win.append(v); w = list(self.win); w = [w[0]] * (WIN - len(w)) + w; item = torch.stack(w)
        else:
            item = v
        out = self.net(item[None]) if self.kind in ("memae", "memfast", "fastw") else None
        pred = out[0][0] if out is not None else self.predict(item[None])[0]
        loss = ((pred - v) ** 2).mean(); score = float(loss.detach())
        if self.replay != "none" and self.items:
            n = min(MB, len(self.items))
            if self.replay == "prio":
                p = np.asarray(self.prio); idx = self.rng.choice(len(self.items), n, replace=False, p=p / p.sum())
            else:
                idx = self.rng.choice(len(self.items), n, replace=False)
            Rb = torch.stack([self.items[i] for i in idx]); tgt = Rb[:, -1] if self.seq else Rb
            loss = loss + ((self.predict(Rb) - tgt) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.kind == "memfast": self.net.write(out[1][0])
        elif self.kind == "fastw": self.net.write(v, (out[1][0][0], out[1][1][0]))
        self._store(item, score)
        return score

    def _store(self, item, score):
        if self.replay == "none": return
        self.seen += 1
        if self.replay == "fifo":
            self.items.append(item); self.items = self.items[-BUF:]
        elif self.replay == "res":
            if len(self.items) < BUF: self.items.append(item)
            else:
                j = int(self.rng.integers(self.seen))
                if j < BUF: self.items[j] = item
        else:  # prio
            self.prio = [p * 0.999 for p in self.prio]
            if len(self.items) < BUF: self.items.append(item); self.prio.append(score + 1e-6)
            else:
                j = int(np.argmin(self.prio))
                if score > self.prio[j]: self.items[j] = item; self.prio[j] = score + 1e-6

    def stored(self):
        n = sum(p.numel() for m in self.mods for p in m.parameters())
        if self.kind == "memfast": n += K * H
        if self.kind == "fastw": n += self.G * H
        if self.replay != "none": n += BUF * self.G * (WIN if self.seq else 1)
        return int(n)


if __name__ == "__main__":
    mode, out, seed, name, rp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    os.makedirs(out, exist_ok=True)
    B.Online = lambda kind, G, s: Online21(kind, G, s, rp)
    B.TAG = f"_{rp}" + os.environ.get("S21_TAG", "")
    (B.run_s1b if mode == "s1b" else B.run_s5b)(seed, name, out)
