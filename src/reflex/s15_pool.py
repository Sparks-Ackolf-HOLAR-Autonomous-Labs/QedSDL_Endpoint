#!/usr/bin/env python3
"""S15 data pool: in-situ Raman probes of MoS2 -> MoSSe and WS2 -> WSSe selenization, with outcome labels.

Each recipe of the Janus campaign grid (1,400: strip fraction, T_C, time, Se pressure) is run DRAWS times through the
S4 lattice kMC (s4_insitu.simulate_probe) with the Mo or W parameter set of the Janus campaign. The probe
state a quarter of the way through the anneal becomes an in-situ Raman spectrum with the S4L
temperature model (s4_lit.py: lines red-shift and broaden with T_C, Janus A1 moves with the top-layer Se
fraction, thermal emission background). Label: the run fails (bottom layer converts, x_bot_Se > 0.3, or
defects > 0.3), as in S4/S4L.

Mo lines: as S4/S4L. W lines (Harris et al., ACS Nano 2023, 10.1021/acsnano.2c09952, measured on WS2 ->
WSSe -> WSe2 at 450 C and room temperature): WS2 E' 352.3 and A1 418; Janus WSSe A1 276 cm^-1 moving to
285 as the top layer completes; WSe2 A1 249.5 and 2LA 260. Assumed, not from that paper: WSSe E mode at
330 cm^-1, defect-activated LA(M) of WS2 at 176 cm^-1. Temperature coefficients are the measured W values
(-0.0038 and -0.0104 cm^-1/K for S-side and Se-side modes).
Output: pool_<mat>.npz with bands (one band per FlyWire glomerulus, G = 56), fail, F, recipe index.
Usage: python s15_pool.py OUT_DIR [DRAWS] [WORKERS]
"""
import os, sys
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
import s1_spectra as S  # noqa: E402
import s4_insitu as B  # noqa: E402

DW_S, DW_SE = -0.0038, -0.0104
MAT = {
    "Mo": dict(lines={"A": [(383, 1.0, 4), (403, 1.2, 4)], "J": [(290, 1.2, 5), (355, 0.7, 6)], "B": [(241, 1.3, 4), (287, 0.4, 5)]},
               janus=(290, 280, 10), se_side={("B", 241), ("B", 287), ("J", 290)}, la=227),
    "W": dict(lines={"A": [(352.3, 1.0, 4), (418, 1.2, 4)], "J": [(285, 1.2, 5), (330, 0.7, 6)], "B": [(249.5, 1.3, 4), (260, 0.4, 6)]},
              janus=(285, 276, 9), se_side={("B", 249.5), ("B", 260), ("J", 285)}, la=176),
}


def raman_mat(st, rng, G, T_C, mat):
    m = MAT[mat]; dT = T_C - 25.0
    br = (1 + 3 * st["d_lat"]) * (1 + dT / 600.0); xt = st["x_top"]
    s = np.zeros_like(S.NU)
    for ph, fr in (("A", st["f_mos2"]), ("J", st["f_janus"]), ("B", st["f_mose2"])):
        ls = []
        for c, a, w in m["lines"][ph]:
            c0 = m["janus"][1] + m["janus"][2] * xt if (ph, c) == ("J", m["janus"][0]) else c
            dw = (DW_SE if (ph, c) in m["se_side"] else DW_S) * dT
            ls.append((c0 + dw, a, w * br))
        s += fr * S.lines(ls)
    s += 2.0 * st["d_lat"] * S.lines([(m["la"] + DW_S * dT, 0.6, 12 * (1 + dT / 600.0))])
    s += 0.8 * st["theta"] * S.lines([(250, 1.0, 30)])
    s += 0.3 * (max(dT, 0) / 875.0) ** 4 * (S.NU - 100) / 400.0
    scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
    bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
    y = scale * (s + bg)
    y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
    return S.bands(y, G)


def one(args):
    mat, j, d = args
    rng = np.random.default_rng([11 if mat == "Mo" else 12, j, d])
    probe, m = B.simulate_probe(K.R[j], K.PAR[mat], rng)
    m = {k: float(np.clip(v + rng.normal(0, 0.02), 0, 1)) for k, v in m.items()}
    fail = m["x_bot_Se"] > B.TB or m["defects"] > B.TD
    return raman_mat(probe, rng, 56, float(K.R[j][1]), mat), fail, m["F"], j


if __name__ == "__main__":
    out = sys.argv[1]; draws = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 8) as ex:
        for mat in MAT:
            r = list(ex.map(one, [(mat, j, d) for d in range(draws) for j in range(K.NC)], chunksize=50))
            np.savez_compressed(f"{out}/pool_{mat}.npz", bands=np.array([a for a, *_ in r], np.float32),
                                fail=np.array([b for _, b, *_ in r]), F=np.array([c for *_, c, _ in r]),
                                recipe=np.array([e for *_, e in r]))
            print(mat, len(r), "fail rate", np.mean([b for _, b, *_ in r]), flush=True)
