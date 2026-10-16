#!/usr/bin/env python3
"""Mean Janus fidelity of every grid recipe (8 kMC replicates, no readout noise), for regret.
Usage: python truth.py M OUT_NPY"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmc
M, out = sys.argv[1:3]
from multiprocessing import Pool
p = kmc.system_params(M); R = kmc.all_recipes()


def one(i):
    rng = np.random.default_rng(2026 + i)
    return np.mean([kmc.simulate(R[i], p, rng)["F"] for _ in range(8)])


with Pool(8) as pool:
    F = np.array(pool.map(one, range(len(R))))
np.save(out, F)
o = np.argsort(-F)
print(M, "max", F.max(), "median", np.median(F), "n>=max-0.02", int((F >= F.max() - 0.02).sum()))
for i in o[:10]:
    print(R[i], round(F[i], 3))
for T in range(250, 901, 50):
    m = R[:, 1] == T; print(T, round(F[m].max(), 3), end="; ")
print()
for s in (0, 0.25, 0.5, 0.75, 1.0):
    m = R[:, 0] == s; print(s, round(F[m].max(), 3), end="; ")
print()
