#!/usr/bin/env python3
"""S23: outcome classes for every recipe of the Janus selenization grid (s4_insitu.simulate_probe, unchanged).

For each of the 1,400 recipes, N_REP kinetic Monte Carlo replicates: destroyed (x_bot_Se > 0.3 or defects > 0.3,
the s4_insitu loss rule), and for kept flakes the top-layer conversion x_top_Se. Writes the share destroyed and the
share kept with x_top_Se below 0.5 and below 0.9 (under-selenized), plus the mean x_top_Se of kept flakes.
Usage: python s23_under.py OUT_NPZ [N_REP] [WORKERS]
"""
import os, sys
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def one(args):
    j, nrep = args
    import s4_insitu as S4
    K = S4.K; rng = np.random.default_rng(31000 + j)
    out = []
    for _ in range(nrep):
        _, m = S4.simulate_probe(K.R[j], K.PAR["Mo"], rng)
        out.append((m["x_bot_Se"] > S4.TB or m["defects"] > S4.TD, m["x_top_Se"], m["F"]))
    return j, out


if __name__ == "__main__":
    out, nrep = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    import s4_insitu as S4
    n = len(S4.K.R)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        res = dict(ex.map(one, [(j, nrep) for j in range(n)]))
    lost = np.array([np.mean([a for a, _, _ in res[j]]) for j in range(n)])
    u50 = np.array([np.mean([(not a) and x < 0.5 for a, x, _ in res[j]]) for j in range(n)])
    u90 = np.array([np.mean([(not a) and x < 0.9 for a, x, _ in res[j]]) for j in range(n)])
    xk = np.array([np.mean([x for a, x, _ in res[j] if not a]) if any(not a for a, _, _ in res[j]) else np.nan for j in range(n)])
    np.savez(out, R=S4.K.R, lost=lost, under50=u50, under90=u90, xtop_kept=xk)
    print("destroyed %.3f  kept & x_top<0.5 %.3f  kept & x_top<0.9 %.3f  kept & x_top>=0.9 %.3f" % (lost.mean(), u50.mean(), u90.mean(), 1 - lost.mean() - u90.mean()))
