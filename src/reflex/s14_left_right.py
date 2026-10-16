#!/usr/bin/env python3
"""S14: why the left mushroom body beats the right (S12, S13): Kenyon-cell wiring or projection-neuron input?

Wirings from s13_prep.py (FAFB v783, BANC v888, MaleCNS v1.0; 55 shared glomeruli, >= 3 synapses per pair).
Per animal A and side s in {left, right} (left KCs read left PNs, as A_left / A_right in S13):
  A_left, A_right        measured wiring, as S13 (reference)
  A_left_nm, A_right_nm  measured wiring, multiglomerular PNs silenced. In S12-S13 each multiglomerular PN read
                         3 random bands, drawn independently for the two sides, so part of a side difference
                         could come from that draw rather than from the animal.
  A_XY (X, Y in L, R)    transplant (configuration model), multiglomerular PNs silenced:
                         every KC of side X keeps its own number of PN partners and its own synapse counts;
                         its partners are drawn from the PNs of side Y with probability proportional to each
                         PN's total synapses onto side-Y KCs (no PN twice per KC); PN glomeruli from side Y.
                         A_LL and A_RR are the per-side degree-preserving controls; A_LR puts left KC wiring
                         on right PN input, A_RL the reverse.
  A_left_gp, A_right_gp   measured wiring with the glomerulus-to-band map permuted at random per seed (the same
                         permutation for both sides; separate generator, other arms unchanged). In S1-S13 the
                         map is alphabetical, so a side difference could come from how each side's convergence
                         pattern lines up with that arbitrary band order.
Two-way split: KC effect = mean(LL, LR) - mean(RL, RR); PN effect = mean(LL, RL) - mean(LR, RR).
Stream, KC code, memory and measures as S1/S12/S13; every detector of a seed sees the same stream.
Usage: [S1_MODE=multi] python s14_left_right.py WIRING_DIR OUT_DIR SEEDS
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
from s9_s12 import score  # noqa: E402

ANIMALS = ("fafb", "banc", "malecns")
SIDE = {"L": "left", "R": "right"}


def silence_multi(Min):
    M = Min.copy(); M[(M > 0).sum(1) > 1] = 0.0
    return M


def transplant(W, ks, ps, x, y, rng):
    """KC rows of side x (partner count and synapse counts kept), partners drawn from side-y PNs by strength."""
    Wx = W[np.ix_(ks == x, ps == x)]; Wy = W[np.ix_(ks == y, ps == y)]
    p = Wy.sum(0); p = p / p.sum(); live = np.flatnonzero(p > 0)
    out = np.zeros((Wx.shape[0], Wy.shape[1]))
    for i, row in enumerate(Wx):
        w = row[row > 0]
        if not len(w): continue
        j = rng.choice(live, size=min(len(w), len(live)), replace=False, p=p[live] / p[live].sum())
        out[i, j] = rng.permutation(w)[:len(j)]
    return out


def arms(Wd, rng, perm):
    W, Min, ks, ps = Wd["W"], Wd["Min"], Wd["kc_side"], Wd["pn_side"]
    Mn = silence_multi(Min); out = {}
    for s in ("left", "right"):
        out[s] = (W[np.ix_(ks == s, ps == s)], Min[ps == s])
        out[f"{s}_nm"] = (W[np.ix_(ks == s, ps == s)], Mn[ps == s])
    for s in ("left", "right"):
        out[f"{s}_gp"] = (W[np.ix_(ks == s, ps == s)], Min[ps == s][:, perm])
    for X in "LR":
        for Y in "LR":
            out[X + Y] = (transplant(W, ks, ps, SIDE[X], SIDE[Y], rng), Mn[ps == SIDE[Y]])
    return out


def run(wdir, seed):
    dets, info = {}, {}
    for a in ANIMALS:
        Wd = dict(np.load(f"{wdir}/wiring_{a}.npz")); G = Wd["Min"].shape[1]
        rng = np.random.default_rng(2000 + seed)
        perm = np.random.default_rng(3000 + seed).permutation(G)
        for k, (W, M) in arms(Wd, rng, perm).items():
            dets[f"{a}_{k}"] = S.Fly(W, M)
            info[f"{a}_{k}"] = dict(n_kc=int(W.shape[0]), n_pn=int(W.shape[1]), pairs=int((W > 0).sum()),
                                    synapses=int(W.sum()), kc_with_input=int(((W @ M).sum(1) > 0).sum()))
    dets["zscore"] = S.Z(G)
    return dict(seed=seed, mode=S.MODE, G=G, arms=info, det=score(dets, S.stream(seed, G), G))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    wdir, out, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        for r in ex.map(run, [wdir] * len(seeds), seeds):
            json.dump(r, open(f"{out}/s14_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
