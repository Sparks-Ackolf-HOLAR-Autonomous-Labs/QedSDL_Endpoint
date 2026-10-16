"""Small sequence and memory-augmented networks for S21, all in PyTorch, CPU, single thread.

Sequence cells (input: a window of T readings, output: the hidden state after the last reading):
  GRU        torch.nn.GRU (Cho et al. 2014)
  Peephole   LSTM whose input, forget and output gates also see the cell state (Gers and Schmidhuber 2000)
  Mogrifier  LSTM whose input and hidden state gate each other for R = 4 rounds before the usual
             LSTM step (Melis, Kocisky and Blunsom 2020)
  Transformer one pre-norm encoder layer, d_model 16, 2 heads, learned position embeddings, causal
             mask, read-out from the last position (Vaswani et al. 2017)
Memory-augmented autoencoders (encoder 56 -> 16, decoder 16 -> 56, ReLU):
  MemAE      memory of K slots learned by gradient; the code z addresses the slots by softmax
             attention with hard shrinkage and the decoder reconstructs from the read-out (Gong et al. 2019).
             The memory is slow: it changes only through gradient steps.
  FastSlot   the same read path, but the K slots are a separate fast memory written in one shot, not
             trained: after each reading the code is written to the most similar slot if its cosine
             similarity exceeds TAU (running average), otherwise to the least-used slot (usage decays),
             as in the least-recently-used write of memory-augmented networks (Santoro et al. 2016)
             and the episodic store of Pritzel et al. 2017. Slow weights (encoder, decoder) train by gradient.
  FastWeight slow autoencoder plus a fast matrix A (56 x 16) that learns the slow decoder's residual
             with the delta rule, A <- LAM A + ETA (x - dec(z) - A k) k^T, k = z/|z|; the prediction is
             dec(z) + A k (fast weights: Ba et al. 2016; delta rule: Schlag, Irie and Schmidhuber 2021).
             LAM sets the forgetting horizon (about 1/(1-LAM) readings), as the recovery rate does for the
             connectome model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Peephole(nn.Module):
    def __init__(self, G, h):
        super().__init__(); self.h = h
        self.W = nn.Linear(G + h, 4 * h); self.p = nn.Parameter(torch.zeros(3, h))  # peepholes i, f, o

    def forward(self, X):  # X: (B, T, G)
        B = X.shape[0]; h = X.new_zeros(B, self.h); c = X.new_zeros(B, self.h)
        for t in range(X.shape[1]):
            a = self.W(torch.cat([X[:, t], h], 1)); i, f, g, o = a.chunk(4, 1)
            i = torch.sigmoid(i + self.p[0] * c); f = torch.sigmoid(f + self.p[1] * c)
            c = f * c + i * torch.tanh(g); o = torch.sigmoid(o + self.p[2] * c); h = o * torch.tanh(c)
        return h


class Mogrifier(nn.Module):
    def __init__(self, G, h, rounds=4, rank=8):
        super().__init__(); self.h, self.R = h, rounds
        self.W = nn.Linear(G + h, 4 * h)
        # low-rank Q (h -> G) and R (G -> h) as in Melis et al. 2020
        self.Q = nn.ModuleList([nn.Sequential(nn.Linear(h, rank, bias=False), nn.Linear(rank, G)) for _ in range((rounds + 1) // 2)])
        self.Rm = nn.ModuleList([nn.Sequential(nn.Linear(G, rank, bias=False), nn.Linear(rank, h)) for _ in range(rounds // 2)])

    def forward(self, X):
        B = X.shape[0]; h = X.new_zeros(B, self.h); c = X.new_zeros(B, self.h)
        for t in range(X.shape[1]):
            x = X[:, t]
            for r in range(self.R):
                if r % 2 == 0: x = 2 * torch.sigmoid(self.Q[r // 2](h)) * x
                else: h = 2 * torch.sigmoid(self.Rm[r // 2](x)) * h
            i, f, g, o = self.W(torch.cat([x, h], 1)).chunk(4, 1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g); h = torch.sigmoid(o) * torch.tanh(c)
        return h


class GRUc(nn.Module):
    def __init__(self, G, h):
        super().__init__(); self.g = nn.GRU(G, h, batch_first=True)

    def forward(self, X): return self.g(X)[0][:, -1]


class LSTMc(nn.Module):
    def __init__(self, G, h):
        super().__init__(); self.g = nn.LSTM(G, h, batch_first=True)

    def forward(self, X): return self.g(X)[0][:, -1]


class Tfm(nn.Module):
    def __init__(self, G, d=16, T=8, heads=2):
        super().__init__(); self.inp = nn.Linear(G, d); self.pos = nn.Parameter(torch.zeros(T, d))
        self.layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True)

    def forward(self, X):
        T = X.shape[1]; z = self.inp(X) + self.pos[-T:]
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        return self.layer(z, src_mask=mask)[:, -1]


SEQ = dict(lstm=LSTMc, gru=GRUc, peep=Peephole, mog=Mogrifier, tfm=lambda G, h: Tfm(G, h))


class MemAE(nn.Module):
    """Autoencoder whose decoder reads from a slot memory. fast=False: slots are parameters (MemAE);
    fast=True: slots are a buffer written in one shot by write()."""

    def __init__(self, G, h=16, K=100, fast=False, tau=0.9, shrink=None):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(G, h), nn.ReLU()); self.dec = nn.Linear(h, G)
        self.fast, self.tau, self.K = fast, tau, K
        self.shrink = 1.0 / K if shrink is None else shrink
        if fast:
            self.register_buffer("M", torch.zeros(K, h)); self.register_buffer("use", torch.zeros(K)); self.n = 0
        else:
            self.M = nn.Parameter(torch.randn(K, h) / h ** 0.5)

    def read(self, z):
        M = self.M[: self.n] if self.fast else self.M
        if self.fast and self.n == 0: return z, None
        s = F.normalize(z, dim=-1) @ F.normalize(M, dim=-1).T
        w = torch.softmax((s * 10.0) if self.fast else (z @ M.T), -1)
        if not self.fast:  # hard shrinkage of Gong et al. 2019
            w = F.relu(w - self.shrink) * w / (torch.abs(w - self.shrink) + 1e-12); w = w / (w.sum(-1, keepdim=True) + 1e-12)
        return w @ M, s

    def forward(self, x):
        z = self.enc(x); r, _ = self.read(z); return self.dec(r), z

    @torch.no_grad()
    def write(self, z):
        if not self.fast: return
        z = z.detach(); self.use *= 0.99
        if self.n:
            s = F.normalize(z, dim=-1) @ F.normalize(self.M[: self.n], dim=-1).T; j = int(s.argmax())
            if float(s[j]) > self.tau:
                self.M[j] = 0.8 * self.M[j] + 0.2 * z; self.use[j] += 1; return
        j = self.n if self.n < self.K else int(self.use.argmin())
        self.M[j] = z; self.use[j] = 1.0; self.n = min(self.n + 1, self.K)


class FastWeight(nn.Module):
    def __init__(self, G, h=16, lam=0.999, eta=0.5):
        super().__init__(); self.enc = nn.Sequential(nn.Linear(G, h), nn.ReLU()); self.dec = nn.Linear(h, G)
        self.register_buffer("A", torch.zeros(G, h)); self.lam, self.eta = lam, eta

    def forward(self, x):
        z = self.enc(x); k = F.normalize(z, dim=-1).detach(); slow = self.dec(z)
        return slow + k @ self.A.T, (slow, k)

    @torch.no_grad()
    def write(self, x, aux):
        slow, k = aux; x = x.reshape(-1); k = k.reshape(-1); slow = slow.detach().reshape(-1)
        self.A.mul_(self.lam).add_(self.eta * torch.outer(x - slow - self.A @ k, k))
