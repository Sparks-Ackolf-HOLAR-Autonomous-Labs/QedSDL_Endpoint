"""Shared pieces for the reflex tests T2-T4: FlyWire mushroom-body wiring and its controls."""
import os, sys, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../janus"))
warnings.filterwarnings("ignore")
import kmc  # noqa: E402
import campaign as C  # noqa: E402  (FlyWire loader, kc_codes)

R, X, NC = C.R, C.X, C.NC
TRUTH = {"Mo": np.load(os.path.join(C.RES, "truth_Mo.npy")), "W": np.load(os.path.join(C.RES, "truth_W.npy"))}
PAR = {"Mo": C.P_MO, "W": C.P_W}


def wirings(rng):
    """Measured PN->KC wiring, degree-preserving shuffle, and random sparse wiring with the same synapse count."""
    C.FlyWire.load(); W = C.FlyWire.W
    r, c = np.nonzero(W); w = W[r, c]
    Ws = np.zeros_like(W); np.add.at(Ws, (r, rng.permutation(c)), w)
    Wr = np.zeros_like(W); Wr[rng.integers(0, W.shape[0], len(w)), rng.integers(0, W.shape[1], len(w))] = w
    return {"fly": W, "fly_shuf": Ws, "fly_rand": Wr}


def mbon_mask():
    C.FlyWire.load(); return (C.FlyWire.M.sum(1) > 0).astype(float)


def codes(W, ctx):
    """KC codes (top 5 %) of all 1400 recipes with a context input (0 = Mo epoch, 1 = W epoch)."""
    C.FlyWire.load()
    return C.kc_codes(W, np.hstack([X, np.full((NC, 1), float(ctx))]))
