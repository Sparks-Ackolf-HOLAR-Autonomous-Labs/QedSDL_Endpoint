#!/usr/bin/env python3
"""T5 (reflex layer): flag wrong values while reading Janus MoSSe literature, one value at a time.

Data: reported Janus MoSSe values (Raman A1 and E modes, PL energy, computed gap, lattice
constant, dipole) compiled from papers and public databases with their sources
(results/reflex/t5_literature/janus_lit_values.csv), plus MoS2 and MoSe2 values from the same
sources. A reading agent (a VLM in the full pipeline) passes each value on in random order.
Errors injected into 20 % of the stream (labels known), the kinds a figure or table reader makes:
  digit      one digit changed (288 -> 388), decimal shift x10 or /10
  unit       PL given in nm instead of eV (1240/E), lattice constant in nm instead of Angstrom
  material   an MoS2 or MoSe2 value reported as Janus MoSSe
Detectors (online; a value is scored against values of the same quantity read before it, then
stored unless it scored in the top 10 % so far):
  fly / fly_shuf / fly_rand  mushroom-body familiarity of (quantity, log value) on measured /
                   shuffled / random PN->KC wiring
  robust_z         |value - median| / (1.4826 MAD) of earlier values of the quantity
  knn              distance in log value to the nearest earlier value of the quantity
First 15 values are warm-up. Measures: AUROC, recall at 5 % false alarms, recall per error kind.
Usage: python t5_literature.py CSV OUT_DIR SEEDS
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402
from sklearn.metrics import roc_auc_score

WARM, P_ERR = 15, 0.2


def load(path):
    rows = [r for r in csv.DictReader(open(path))]
    for r in rows: r["value"] = float(r["value"])
    n = {}
    for r in rows:
        if r["material"] == "MoSSe": n[r["quantity"]] = n.get(r["quantity"], 0) + 1
    # quantities with fewer than 8 Janus values (the dipole, reported on different cell bases) are left out
    return [r for r in rows if n.get(r["quantity"], 0) >= 8]


def corrupt(r, rng, others):
    q, v = r["quantity"], r["value"]; kinds = ["digit"]
    if q.startswith("pl") or q.startswith("a_"): kinds.append("unit")
    if others.get(q): kinds.append("material")
    k = kinds[rng.integers(len(kinds))]
    if k == "digit":
        if rng.random() < 0.5: v = v * (10 if rng.random() < 0.5 else 0.1)
        else:
            s = f"{v:g}"; pos = [i for i, ch in enumerate(s) if ch.isdigit()]; i = pos[rng.integers(len(pos))]
            d = (int(s[i]) + int(rng.integers(1, 10))) % 10; s2 = s[:i] + str(d) + s[i + 1:]
            v = float(s2) if float(s2) > 0 else v * 10
    elif k == "unit":
        v = 1239.84 / v if q.startswith("pl") else v / 10
    else:
        v = others[q][rng.integers(len(others[q]))]
    return dict(r, value=float(v)), k


class Fly:
    def __init__(self, W, quantities, rng):
        self.W, self.k, self.q = W, int(0.05 * W.shape[0]), {q: i for i, q in enumerate(quantities)}
        self.E = rng.normal(size=(W.shape[1], len(quantities) + 1)); self.mem = {}

    def code(self, q, v, scale):
        x = np.zeros(len(self.q) + 1); x[self.q[q]] = 1.0; x[-1] = (np.log(v) - scale[0]) / scale[1]
        d = np.maximum((x - 0.5) / 0.3 @ self.E.T, 0) @ self.W.T
        return d >= np.partition(d, -self.k)[-self.k]

    def score(self, q, v, scale):
        c = self.code(q, v, scale); M = self.mem.get(q, [])
        return (1.0 if not M else 1 - max((m & c).sum() for m in M) / self.k), c

    def store(self, q, c): self.mem.setdefault(q, []).append(c)


def run(rows, seed):
    rng = np.random.default_rng(seed)
    janus = [r for r in rows if r["material"] == "MoSSe"]
    others = {}
    for r in rows:
        if r["material"] != "MoSSe": others.setdefault(r["quantity"], []).append(r["value"])
    quantities = sorted({r["quantity"] for r in janus})
    # broad prior scale per quantity from the order of magnitude only (what a reader knows before reading)
    stream = []
    for r in [janus[i] for i in rng.permutation(len(janus))] * 2:   # each value read twice, from two papers citing it
        if rng.random() < P_ERR: r2, k = corrupt(r, rng, others); stream.append((r2["quantity"], r2["value"], k))
        else: stream.append((r["quantity"], r["value"], None))
    scale = {q: (np.log(np.median([r["value"] for r in janus if r["quantity"] == q])), 1.0) for q in quantities}
    Wd = K.wirings(np.random.default_rng(seed))
    dets = {n: Fly(Wd[n], quantities, np.random.default_rng(7)) for n in Wd}
    names = list(dets) + ["robust_z", "knn"]; sc = {n: [] for n in names}; mem = {n: {} for n in names}
    for q, v, k in stream:
        for n in names:
            if n in dets:
                s, c = dets[n].score(q, v, scale[q]); item = c
            else:
                past = np.log(mem[n].get(q, [])); item = v
                if len(past) < 3: s = 0.0
                elif n == "robust_z": s = abs(np.log(v) - np.median(past)) / (1.4826 * np.median(abs(past - np.median(past))) + 1e-3)
                else: s = float(np.min(abs(past - np.log(v))))
            prev = sc[n]; sc[n].append(s)
            if len(prev) < 10 or s < np.quantile(prev[:-1], 0.9):
                (dets[n].store(q, item) if n in dets else mem[n].setdefault(q, []).append(item))
    lab = np.array([k is not None for *_, k in stream])[WARM:]; kinds = [k for *_, k in stream][WARM:]
    res = dict(seed=seed, n=len(stream) - WARM, n_err=int(lab.sum()), det={})
    for n in names:
        x = np.array(sc[n][WARM:]); thr = np.quantile(x[~lab], 0.95)
        res["det"][n] = dict(auroc=float(roc_auc_score(lab, x)), recall_at_5fpr=float(np.mean(x[lab] > thr)),
                             recall_by_kind={kk: (float(np.mean(x[[a == kk for a in kinds]] > thr)) if kk in kinds else None)
                                             for kk in ("digit", "unit", "material")})
    return res


if __name__ == "__main__":
    rows = load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    for s in [int(x) for x in sys.argv[3].split(",")]:
        r = run(rows, s); json.dump(r, open(f"{out}/t5_s{s}.json", "w"), indent=1)
        print(s, r["n_err"], {n: round(d["auroc"], 3) for n, d in r["det"].items()}, flush=True)
