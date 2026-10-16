#!/usr/bin/env python3
"""S18c: time the S18a local stop (nerve-cord features + ridge read-out, and ridge on bands) on the Pi 5.

NumPy only, one thread. Copies of banc_cord.npz (built by s18a_furnace.py) and this file go to the Pi.
Usage: python3 s18c_pi_timing.py banc_cord.npz OUT_JSON [N]
"""
import json, os, platform, resource, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

z = dict(np.load(sys.argv[1])); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
G = 56; rng = np.random.default_rng(0)
band = rng.integers(0, G, int(z["n_S"])); cmd = (rng.random(int(z["n_D"])) < 0.05).astype(float)


def prop(blk, x, n):
    return np.bincount(z[blk + "_post"], weights=z[blk + "_w"] * x[z[blk + "_pre"]], minlength=int(n))


def cord(xb):
    s = xb[band]
    h = np.maximum(prop("SI", s, z["n_I"]) + prop("DI", cmd, z["n_I"]), 0); h /= h.mean() + 1e-12
    m = np.maximum(prop("IM", h, z["n_M"]) + prop("SM", s, z["n_M"]) + prop("DM", cmd, z["n_M"]), 0)
    return m / (m.mean() + 1e-12)


Bc = rng.normal(size=(int(z["n_M"]), 2)); Bb = rng.normal(size=(G, 2))
X = rng.random((N, G)); X /= X.sum(1, keepdims=True)
res = {}
for name, f in (("cord_stop", lambda x: cord(x) @ Bc), ("bands_stop", lambda x: x @ Bb)):
    for x in X[:50]: f(x)
    t0 = time.perf_counter()
    for x in X: f(x)
    res[name + "_ms"] = (time.perf_counter() - t0) / N * 1e3
n_syn = sum(len(v) for k, v in z.items() if k.endswith("_w"))
res.update(synapse_entries=int(n_syn), list_bytes_8bit_32idx=int(n_syn * (1 + 4 + 4)),
           maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, machine=platform.machine(),
           node=platform.node(), numpy=np.__version__, n=N)
json.dump(res, open(sys.argv[2], "w"), indent=1); print(res)
