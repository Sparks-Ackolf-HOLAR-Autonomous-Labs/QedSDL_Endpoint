#!/usr/bin/env python3
"""S13: the same connectome model built from three animals, per side and with or without cross-side wiring.

Wirings from s13_prep.py (FAFB v783 female brain, BANC v888 female brain and nerve cord, MaleCNS v1.0 male
brain and nerve cord; one shared list of 55 glomeruli, >= 3 synapses per PN-KC pair). Per animal A:
  A_left     left KCs, left PNs only
  A_right    right KCs, right PNs only
  A_both     all KCs, all PNs, every connection (cross-side pairs included)
  A_both_nox all KCs, all PNs, cross-side pairs removed (two hemispheres side by side, one readout)
  A_shuf     A_both with degree-preserving shuffle of the PN partners (same as fly_shuf in S1-S12)
  A_knn      equal-memory reservoir kNN with the stored-number budget of A_both (n_KC // G spectra)
plus fafb_raw_both (FAFB without the synapse threshold, the S1-S12 wiring on this glomerulus list) and the
per-band z-score. Stream, KC code (top 5 %), bounded familiarity memory and measures are those of S1/S12;
only the wiring changes, and every detector of a seed sees the same stream.
Usage: [S1_MODE=multi] python s13_cross_animal.py WIRING_DIR OUT_DIR SEEDS
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
from s9_s12 import score  # noqa: E402

ANIMALS = ("fafb", "banc", "malecns")


def arms(Wd, rng):
    W, Min, ks, ps = Wd["W"], Wd["Min"], Wd["kc_side"], Wd["pn_side"]
    out = {}
    for s in ("left", "right"):
        out[s] = (W[np.ix_(ks == s, ps == s)], Min[ps == s])
    cross = (ks[:, None] != ps[None, :]) & (ks[:, None] != "") & (ps[None, :] != "")
    out["both"] = (W, Min); out["both_nox"] = (np.where(cross, 0.0, W), Min)
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    out["shuf"] = (Ws, Min)
    return out


def run(wdir, seed):
    G = None; dets = {}; info = {}
    for a in ANIMALS + ("fafb_raw",):
        Wd = dict(np.load(f"{wdir}/wiring_{a}.npz"))
        G = Wd["Min"].shape[1]
        rng = np.random.default_rng(1000 + seed)
        for k, (W, M) in arms(Wd, rng).items():
            if a == "fafb_raw" and k != "both": continue
            dets[f"{a}_{k}"] = S.Fly(W, M); info[f"{a}_{k}"] = dict(n_kc=int(W.shape[0]), n_pn=int(W.shape[1]),
                                                                    pairs=int((W > 0).sum()), synapses=int(W.sum()))
        if a != "fafb_raw":
            dets[f"{a}_knn"] = S.KNN(True, int(Wd["W"].shape[0] // G), np.random.default_rng(3))
    dets["zscore"] = S.Z(G)
    Sx = S.stream(seed, G)
    return dict(seed=seed, mode=S.MODE, G=G, arms=info, det=score(dets, Sx, G))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    wdir, out, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(min(4, len(seeds))) as ex:
        for r in ex.map(run, [wdir] * len(seeds), seeds):
            json.dump(r, open(f"{out}/s13_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
