#!/usr/bin/env python3
"""S7b: three Raman stations in a 2D-semiconductor lab network, each with its own connectome model,
with and without shared familiarity memory.

Stations (interleaved, 2,000 readings each, 6,000 in all; warm-up 300 readings in total):
  A  Janus line: Mo-S-Se 70 %, W-S-Se 30 %
  B  W-S-Se 60 %, black phosphorus 40 %
  C  ReS2 50 %, Mo-S-Se 50 %
From each station's 1,000th reading, 25 % of its flakes come from a family it has never scanned but
another station has (A gets black P, B gets Mo-S-Se, C gets W-S-Se): new to the station, known to the
network ("network-known first sightings"). These are normal readings; flagging them is a false alarm.
New events (5 % of readings after warm-up): 1T' MoS2, cosmic-ray spike, MoS2/WSe2 stack, at any station.
Spreading contamination: MoTe2 (from a shared Te-contaminated precursor) appears at station A from its
600th reading and at station B from its 1,400th, each with probability 0.06 per reading. At B it is a new
problem for B; shared memory may already have made it familiar ("hidden").
Arms: private (one model per station), shared (one memory written and read by all three), weighted
(each station's own memory, other stations' readings written at 0.3 of the rate: social cue weighting,
3x memory); for the plain fly and for the S2/S3 configuration (fly_best); zscore and knn_budget private /
shared. Measures: AUROC on new events (all stations, excluding MoTe2); false alarms on network-known
first sightings (threshold = each station's 95 % quantile of its other normal readings); recall at 5 %
false alarms of MoTe2 at A and at B.
Usage: python s7b_stations.py OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s2_s3 as H  # noqa: E402
from fly2 import Fly2  # noqa: E402

FAMS = list(S.FAM)
MIX = {0: {"MoSSe": 0.7, "WSSe": 0.3}, 1: {"WSSe": 0.6, "BP": 0.4}, 2: {"ReS2": 0.5, "MoSSe": 0.5}}
LATE = {0: "BP", 1: "MoSSe", 2: "WSSe"}
NPER, P_LATE, P_TE = 2000, 0.25, 0.06
TE_START = {0: 600, 1: 1400}
EVENTS = ["1T'_MoS2", "spike", "MoS2/WSe2_stack"]


def stream(seed, G):
    rng = np.random.default_rng(seed); out = []; cnt = [0, 0, 0]
    for i in range(3 * NPER):
        st = i % 3; k = cnt[st]; cnt[st] += 1
        late = k >= NPER // 2 and rng.random() < P_LATE
        if late: fam = LATE[st]
        else:
            fs, ps = zip(*MIX[st].items()); fam = fs[rng.choice(len(fs), p=ps)]
        comp = S.FAM[fam]; w = rng.dirichlet([0.6] * len(comp))
        s = sum(wi * S.BASE[c] for wi, c in zip(w, comp)); kind = None
        if st in TE_START and k >= TE_START[st] and rng.random() < P_TE:
            kind = "MoTe2"; fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS["MoTe2"]
        elif i >= S.WARM and rng.random() < S.P_NEW:
            kind = EVENTS[rng.integers(len(EVENTS))]
            if kind == "MoS2/WSe2_stack": s = 0.5 * S.BASE["MoS2"] + 0.5 * S.BASE["WSe2"]
            elif kind == "spike": s = s + S.lines([(rng.uniform(110, 490), 4.0, 1.0)])
            else: fr = rng.uniform(0.4, 0.8); s = (1 - fr) * s + fr * S.NEWS[kind]
        scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
        bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
        y = scale * (s + bg); y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
        out.append(dict(x=S.bands(y, G), kind=kind, st=st, late=bool(late)))
    return out


class Private:
    def __init__(self, mk): self.d = [mk() for _ in range(3)]
    def step(self, x, st): return self.d[st].step(x)
    def stored(self): return sum(d.stored() for d in self.d)


class Shared:
    def __init__(self, mk): self.d = mk()
    def step(self, x, st): return self.d.step(x)
    def stored(self): return self.d.stored()


class Weighted:
    """Each station reads its own fly memory; other stations' readings depress it at rate eta*share."""
    def __init__(self, mk, share=0.3):
        self.d = [mk() for _ in range(3)]; self.share = share

    def step(self, x, st):
        s = self.d[st].step(x)
        for o in range(3):
            if o != st:
                f = self.d[o]; c = f.code(x); f.w[0][c] *= 1 - f.eta * self.share
        return s

    def stored(self): return sum(d.stored() for d in self.d)


def run(seed):
    from sklearn.metrics import roc_auc_score
    Wd, Min, G = H.wirings(seed); W = Wd["fly"]; full = H.default_full_cfg(); budget = int(W.shape[0] // G)
    plain = lambda: Fly2(W, Min); allc = lambda: Fly2(W, Min, **full)
    dets = {"fly_private": Private(plain), "fly_shared": Shared(plain), "fly_weighted": Weighted(plain),
            "fly_best_private": Private(allc), "fly_best_shared": Shared(allc), "fly_best_weighted": Weighted(allc),
            "zscore_private": Private(lambda: S.Z(G)), "zscore_shared": Shared(lambda: S.Z(G)),
            "knn_budget_private": Private(lambda: S.KNN(True, budget // 3, np.random.default_rng(3))),
            "knn_budget_shared": Shared(lambda: S.KNN(True, budget, np.random.default_rng(3)))}
    stm = stream(seed, G)
    sc = {n: np.array([d.step(r["x"], r["st"]) for r in stm]) for n, d in dets.items()}
    kind = np.array([r["kind"] or "" for r in stm]); st = np.array([r["st"] for r in stm]); late = np.array([r["late"] for r in stm])
    post = np.arange(len(stm)) >= S.WARM; ev = np.isin(kind, EVENTS); norm = (kind == "") & post
    res = dict(seed=seed, det={})
    for n, x in sc.items():
        thr = {s: np.quantile(x[norm & ~late & (st == s)], 0.95) for s in range(3)}
        above = np.array([x[i] > thr[st[i]] for i in range(len(x))])
        res["det"][n] = dict(
            auroc=float(roc_auc_score(ev[post & (ev | norm)], x[post & (ev | norm)])),
            fa_network_known=float(above[norm & late].mean()), fa_station_normal=float(above[norm & ~late].mean()),
            fa_network_known_first50=float(np.mean([above[np.where(norm & late & (st == s))[0][:50]].mean() for s in range(3)])),
            mote2_recall_A=float(above[(kind == "MoTe2") & (st == 0)].mean()),
            mote2_recall_B=float(above[(kind == "MoTe2") & (st == 1)].mean()),
            mote2_recall_B_first5=float(above[np.where((kind == "MoTe2") & (st == 1))[0][:5]].mean()),
            stored_numbers=int(dets[n].stored()))
    return res


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s7b_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {n: (round(d["auroc"], 3), round(d["fa_network_known"], 2), round(d["mote2_recall_B"], 2)) for n, d in r["det"].items()}, flush=True)
