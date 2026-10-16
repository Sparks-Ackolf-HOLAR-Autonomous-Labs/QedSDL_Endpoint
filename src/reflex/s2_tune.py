#!/usr/bin/env python3
"""Parameter choice for S2/S3 on tuning seeds 100-103 (both streams). Writes s2_params.json and a log.
Choice rule: gain (beta, alpha) and adapt (tau) minimise false alarms after the laser drop subject to
AUROC no lower than the plain fly's minus 0.01; coinc act_frac maximises S1b stack AUROC.
Usage: python s2_tune.py janus|multi OUT_JSON"""
import itertools, json, os, sys
MODE = sys.argv[1]; os.environ["S1_MODE"] = MODE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import s2_s3 as H  # noqa: E402
from fly2 import Fly2  # noqa: E402

GAIN = [dict(beta=b, alpha=a) for b, a in itertools.product([0.0, 0.5, 1.0], [0.5, 1.0, 2.0])]
ADAPT = [dict(tau=t) for t in [50.0, 200.0, 1000.0]]
COINC = [dict(act_frac=f) for f in [0.15, 0.3, 0.5]]


def arms(Wd, Min, G):
    d = {"fly": Fly2(Wd["fly"], Min)}
    for i, p in enumerate(GAIN): d[f"gain{i}"] = Fly2(Wd["fly"], Min, gain=True, **p)
    for i, p in enumerate(ADAPT): d[f"adapt{i}"] = Fly2(Wd["fly"], Min, adapt=True, **p)
    for i, p in enumerate(COINC): d[f"coinc{i}"] = Fly2(Wd["fly"], Min, coinc=True, **p)
    return d


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        R = list(ex.map(H.run, [100, 101, 102, 103], [arms] * 4))
    m = lambda n, k: float(np.mean([r["det"][n][k] for r in R]))
    st = lambda n: float(np.mean([r["det"][n]["auroc_by_kind"].get("MoS2/WSe2_stack", np.nan) for r in R]))
    log = {n: dict(auroc=m(n, "auroc"), fa=m(n, "false_alarm_after_change"), stack=st(n)) for n in R[0]["det"]}
    json.dump(dict(mode=MODE, grid=dict(gain=GAIN, adapt=ADAPT, coinc=COINC), log=log), open(sys.argv[2], "w"), indent=1)
    for n, v in log.items(): print(n, {k: round(x, 3) for k, x in v.items()})
