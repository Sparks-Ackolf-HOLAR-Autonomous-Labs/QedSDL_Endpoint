#!/usr/bin/env python3
"""S23: how a shifted Raman axis and unequal band gains change the connectome model's Kenyon-cell code.

The nine phases of the multi-family streams (s1_spectra, S1_MODE=multi: MoS2, MoSSe, MoSe2, WS2, WSe2, black phosphorus,
ReS2, and the new phases 1T'-MoS2 and MoTe2) are coded by the plain connectome model (fly2.Fly2 on the measured FlyWire
wiring and glom_map of s2_s3.wirings). A reference code is written at the true axis; a reading whose lines are shifted by
d cm^-1 (drift between writing the memory and reading against it) is compared with every reference: overlap with its own
reference (shared active cells / k) and whether its own reference is the closest. Three inputs: the 56 equal-width bands
of the model (about 7.1 cm^-1 each); broad overlapping Gaussian bands centred on the same positions (sigma 7 and 12
cm^-1); and the 56 bands with an offset search (the reading is re-coded at corrections of -20 to +20 cm^-1 in 1 cm^-1
steps and the most familiar correction kept). Per-band gain error: a fixed log-normal gain per band (s.d. 5 to 30 %),
the same for every reading, applied to readings but not to the references, 20 draws.
Usage: python s23_raman_axis.py OUT_JSON
"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s2_s3 as H  # noqa: E402
from fly2 import Fly2  # noqa: E402

LS = {**S.PH, "1T'_MoS2": S.NEW["1T'_MoS2"], "MoTe2": S.NEW["MoTe2"]}
NAMES = list(LS)
OFFS = [0, 1, 2, 3, 4, 5, 7, 10, 15]


def main(out):
    Wd, Min, G = H.wirings(1)
    f = Fly2(Wd["fly"], Min)
    k = f.k
    cen = S.NU[: (len(S.NU) // G) * G].reshape(G, -1).mean(1)
    gw = {sg: np.exp(-(S.NU[None] - cen[:, None]) ** 2 / (2 * sg ** 2)) for sg in (7.0, 12.0)}

    def spec(name, d=0.0):
        return S.lines([(c + d, a, w) for c, a, w in LS[name]]) + 0.03

    def x_of(s, sg=None):
        x = S.bands(s, G) if sg is None else (gw[sg] @ s) / gw[sg].sum(1)
        return x / x.max()
    code = lambda x: f.code(x)  # noqa: E731
    res = {}
    for label, sg in (("bands", None), ("broad_7", 7.0), ("broad_12", 12.0)):
        ref = {n: code(x_of(spec(n), sg)) for n in NAMES}
        closest_other = float(np.mean([max((ref[n] & ref[m]).sum() / k for m in NAMES if m != n) for n in NAMES]))
        rows = []
        for d in OFFS:
            ov, ok = [], 0
            for n in NAMES:
                c = code(x_of(spec(n, d), sg)); o = {m: (c & ref[m]).sum() / k for m in NAMES}
                ov.append(o[n]); ok += max(o, key=o.get) == n
            rows.append(dict(offset=d, self_overlap=float(np.mean(ov)), identified=ok, of=len(NAMES)))
        res[label] = dict(closest_other_material=closest_other, by_offset=rows)
    ref = {n: code(x_of(spec(n))) for n in NAMES}
    rows = []
    for d in OFFS:
        ov, ok = [], 0
        for n in NAMES:
            best = max(((m, (code(x_of(spec(n, d - e))) & ref[m]).sum() / k) for e in range(-20, 21) for m in NAMES), key=lambda t: t[1])
            ok += best[0] == n; ov.append(best[1] if best[0] == n else 0.0)
        rows.append(dict(offset=d, self_overlap=float(np.mean(ov)), identified=ok, of=len(NAMES)))
    res["bands_offset_search"] = dict(corrections=41, by_offset=rows)
    rng = np.random.default_rng(0); rows = []
    for g in (0.05, 0.1, 0.2, 0.3):
        ov, ok, tot = [], 0, 0
        for _ in range(20):
            gain = np.exp(rng.normal(0, g, G))
            for n in NAMES:
                c = code(x_of(spec(n)) * gain); o = {m: (c & ref[m]).sum() / k for m in NAMES}
                ov.append(o[n]); ok += max(o, key=o.get) == n; tot += 1
        rows.append(dict(gain_sd=g, self_overlap=float(np.mean(ov)), identified=ok, of=tot))
    res["band_gain_error"] = rows
    res["band_width_cm1"] = float(400 / G)
    json.dump(res, open(out, "w"), indent=1)
    for kk, v in res.items():
        print(kk, json.dumps(v)[:600])


if __name__ == "__main__":
    main(sys.argv[1])
