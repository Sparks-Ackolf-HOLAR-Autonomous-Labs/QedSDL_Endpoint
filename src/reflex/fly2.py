"""Fly mushroom-body familiarity circuit with the antennal-lobe and calyx features tested in S2-S8.

Builds on s1_spectra.Fly (one spectral band per glomerulus, top-5 % Kenyon-cell code, one depressible
KC->MBON weight per Kenyon cell). Options, each named after the fly mechanism it stands for:
  gain    antennal-lobe gain control (local-neuron inhibition): PN = max(p - beta*mean(p), 0), then
          divisive normalisation PN^n / (PN^n + (alpha*mean(p))^n). Scale-free, so laser power,
          focus and flake size cancel; the subtractive part removes the flat fluorescence floor.
  adapt   receptor adaptation: each PN response is divided by its own slow running mean (time
          constant tau readings), so a lasting change in the instrument fades from the code.
  coinc   claw coincidence: a Kenyon cell is driven by the synapse-weighted count of its ACTIVE inputs
          (PNs in the top act_frac of the current pattern), not by their summed strength. A cell that
          samples lines from two materials is then strongly driven only when both are present.
  ctx     non-olfactory context input (accessory calyx): the material family from the sample log adds
          ctx_gain * std(drive) to a fixed quarter of the Kenyon cells, one quarter per family.
  comp    dopamine-gated compartments: one KC->MBON weight vector per family on the same KC code
          (memory x number of families).
"""
import numpy as np

K_FRAC = 0.05


class Fly2:
    def __init__(self, W, Min, gain=False, adapt=False, coinc=False, ctx=False, comp=False, n_fam=4,
                 beta=0.5, alpha=1.0, n=1.5, act_frac=0.3, tau=200.0, ctx_gain=2.0, eta=0.3, seed=0):
        self.W, self.M = W, Min
        self.gain, self.adapt, self.coinc, self.ctx, self.comp = gain, adapt, coinc, ctx, comp
        self.beta, self.alpha, self.n, self.act_frac, self.tau, self.ctx_gain = beta, alpha, n, act_frac, tau, ctx_gain
        self.k = int(K_FRAC * W.shape[0]); self.eta = eta
        self.w = np.ones((n_fam if comp else 1, W.shape[0]))
        self.run = None
        q = np.array_split(np.random.default_rng(seed).permutation(W.shape[0]), n_fam)
        self.cmask = np.zeros((n_fam, W.shape[0]))
        for f, i in enumerate(q): self.cmask[f, i] = 1

    def pn(self, x):
        p = self.M @ x
        if self.adapt:
            p = p / (p.sum() + 1e-12)
            self.run = p.copy() if self.run is None else self.run + (p - self.run) / self.tau
            p = p / (self.run + 1e-6)
        if self.gain:
            m = p.mean() + 1e-12
            p = np.maximum(p - self.beta * m, 0)
            s = self.alpha * m
            p = p ** self.n / (p ** self.n + s ** self.n)
        return p

    def code(self, x, f=0):
        p = self.pn(x)
        if self.coinc:
            act = (p >= np.quantile(p, 1 - self.act_frac)) & (p > 0)
            lin = self.W @ p
            d = self.W @ act.astype(float) + 1e-3 * lin / (lin.max() + 1e-12)
        else:
            d = self.W @ p
        if self.ctx:
            d = d + self.ctx_gain * d.std() * self.cmask[f]
        return d >= np.partition(d, -self.k)[-self.k]

    def step(self, x, f=0):
        c = self.code(x, f); w = self.w[f if self.comp else 0]
        s = w[c].sum() / self.k
        w[c] *= 1 - self.eta
        return s

    def stored(self):
        return self.w.size
