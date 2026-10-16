#!/usr/bin/env python3
"""Janus MoSSe active-learning campaign: algorithm-only vs connectome model vs LLM agents,
alone and as three parties sharing one furnace fleet, with and without prior information.

Task: choose selenization recipes (H2 pre-strip, temperature, time, Se supply; 1400 on a grid)
for MoS2 -> Janus MoSSe; each experiment is one kinetic Monte Carlo run (kmc.py) with readout
noise. Goal: highest Janus fidelity F within 30 experiments.

Prior information (arm suffix "_prior"):
  data        30 recipes run earlier on WS2 -> WSSe (same simulator, W parameters)
  literature  a short text summary of the two published routes (given to the LLM only)
Agents:
  random      uniform over untested recipes
  gp          Gaussian process (Matern 5/2) with upper confidence bound; prior data enter as
              extra points with a system indicator feature (Mo = 0, W = 1)
  fly         mushroom body on the FlyWire 783 PN->KC wiring (as in ../mb_agent.py): recipe ->
              PN activity -> top-5 % KC code -> MBON value learned by a dopamine-style
              prediction error, plus KC novelty bonus; prior data pre-train the MBON weights
  fly_shuf    same, PN->KC partners shuffled (degree-preserving): wiring control
  llm         Qwen3.6-35B-A3B (FP8, local vLLM, no reasoning mode) reads the history table
              (fidelity and the three readouts) and returns the next recipe as JSON
Network arms ("net_*"): parties gp, fly and llm share all results; each round every party
proposes one recipe and the three run in parallel on three furnaces (10 rounds = 30 runs).
  net_none   no party has prior information
  net_all    every party has all prior information
  net_split  expertise split: gp has the WSSe data, llm has the literature, fly has neither
All arms of a seed start from the same 3 random recipes.
Usage: python campaign.py OUT_DIR ARM[,ARM...] SEEDS [WORKERS]
"""
import json, os, re, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kmc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../../results/janus")
R = kmc.all_recipes(); X = kmc.features(R); NC = len(R)
P_MO, P_W = kmc.system_params("Mo"), kmc.system_params("W")
TRUTH = np.load(os.path.join(RES, "truth_Mo.npy"))  # mean F per recipe, 8 replicates
BUDGET, N0, N_PRIOR = 30, 3, 30
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8721/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")

LITERATURE = (
    "Published routes to Janus MoSSe from monolayer MoS2: (1) thermal selenization with Se vapour, "
    "Janus MoSSe forms near 800 C; at higher temperature the bottom S layer also exchanges and the "
    "flake converts toward MoSe2 (Zhang et al., ACS Nano 2017). (2) H2-plasma stripping of the top S "
    "layer at room temperature, then selenization at about 350 C gives Janus MoSSe with a sharp "
    "interface (Lu et al., Nat. Nanotechnol. 2017). Over-etching by the plasma damages the lattice.")


# ---------------------------------------------------------------- mushroom body
class FlyWire:
    _lock = threading.Lock(); W = None; M = None; ENC = None

    @classmethod
    def load(cls, data=os.path.expanduser("~/flywire_v783")):
        with cls._lock:
            if cls.W is not None:
                return
            import pandas as pd
            a = pd.read_csv(f"{data}/neuron_annotations.tsv", sep="\t", low_memory=False)
            con = pd.read_parquet(f"{data}/Connectivity_783.parquet")
            ids = lambda m: set(a.root_id[m].astype(np.int64))
            PN, KC, MB = ids(a.cell_class == "ALPN"), ids(a.cell_class == "Kenyon_Cell"), ids(a.cell_class == "MBON")
            PNl, KCl, MBl = sorted(PN), sorted(KC), sorted(MB)
            pi, ki, mi = ({f: i for i, f in enumerate(l)} for l in (PNl, KCl, MBl))
            pk = con[con.Presynaptic_ID.isin(PN) & con.Postsynaptic_ID.isin(KC)]
            km = con[con.Presynaptic_ID.isin(KC) & con.Postsynaptic_ID.isin(MB)]
            W = np.zeros((len(KCl), len(PNl)))
            W[pk.Postsynaptic_ID.map(ki), pk.Presynaptic_ID.map(pi)] = pk.Connectivity.values
            M = np.zeros((len(KCl), len(MBl)))
            M[km.Presynaptic_ID.map(ki), km.Postsynaptic_ID.map(mi)] = 1.0
            cls.ENC = np.random.default_rng(12345).normal(size=(len(PNl), 5))
            cls.W, cls.M = W, M


def kc_codes(W, Xf, frac=0.05):
    Z = (Xf - 0.5) / 0.3
    d = np.maximum(Z @ FlyWire.ENC.T, 0.0) @ W.T
    k = int(frac * W.shape[0]); thr = np.partition(d, -k, axis=1)[:, -k][:, None]
    return (d >= thr).astype(np.float32)


class Fly:
    def __init__(self, rng, shuffled=False, eta=0.5, beta=0.6):
        FlyWire.load(); W = FlyWire.W
        if shuffled:
            r, c = np.nonzero(W); w = W[r, c]; W = np.zeros_like(W); np.add.at(W, (r, rng.permutation(c)), w)
        Xs = np.hstack([X, np.zeros((NC, 1))]); Xw = np.hstack([X, np.ones((NC, 1))])
        self.codes = kc_codes(W, Xs); self.codes_w = kc_codes(W, Xw)
        self.w = np.zeros(FlyWire.M.shape[1]); self.eta, self.beta = eta, beta
        self.Mrow = FlyWire.M.sum(1)  # number of MBON partners per KC

    def train(self, codes, ys):
        mu, sd = np.mean(ys), np.std(ys) + 1e-9
        for c, y in zip(codes, ys):
            pred = (c * self.kcw).sum() / c.sum()
            err = (y - mu) / sd - pred  # dopamine-style reward prediction error
            self.kcw += self.eta * err * c * (self.Mrow > 0) / c.sum()

    def propose(self, hist_idx, hist_y, prior):
        self.kcw = np.zeros(len(self.Mrow))
        if prior is not None:
            self.train(self.codes_w[prior["idx"]], prior["F"])
        self.train(self.codes[hist_idx], hist_y)
        vis = self.codes[hist_idx]
        val = (self.codes * self.kcw).sum(1) / self.codes.sum(1)
        nov = 1 - (self.codes @ vis.T).max(1) / self.codes.sum(1)
        s = val + self.beta * nov; s[hist_idx] = -np.inf
        return int(np.argmax(s))


# ---------------------------------------------------------------- Gaussian process
def gp_propose(hist_idx, hist_y, prior, kappa=2.0):
    Xm = np.hstack([X, np.zeros((NC, 1))])
    A, y = Xm[hist_idx], np.array(hist_y)
    if prior is not None:
        A = np.vstack([A, np.hstack([X[prior["idx"]], np.ones((len(prior["idx"]), 1))])])
        y = np.concatenate([y, prior["F"]])
    k = ConstantKernel(1.0) * Matern(length_scale=[0.3] * 5, length_scale_bounds=(0.03, 30), nu=2.5) + WhiteKernel(1e-3, (1e-5, 1e-1))
    g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(A, y)
    m, s = g.predict(Xm, return_std=True); u = m + kappa * s; u[hist_idx] = -np.inf
    return int(np.argmax(u))


# ---------------------------------------------------------------- LLM
def _fmt(r):
    return f"strip={r[0]:.2f}, T_C={int(r[1])}, t_min={int(r[2])}, se={r[3]:g}"


def llm_propose(hist_idx, hist_obs, prior, use_lit, use_data, log, tries=3):
    lines = ["You are the decision agent of a self-driving laboratory that makes Janus MoSSe monolayers "
             "by selenizing monolayer MoS2 in a furnace. The goal is the highest Janus fidelity F "
             "(0-1): Se fully replacing the top S layer while the bottom S layer stays S and the lattice "
             "stays intact. Each furnace run tests one recipe:",
             "  strip: fraction of top-layer S removed by H2 plasma before annealing, one of 0, 0.25, 0.5, 0.75, 1",
             "  T_C: anneal temperature in C, one of 250, 300, ..., 900 (steps of 50)",
             "  t_min: anneal time in minutes, one of 5, 15, 30, 60, 120",
             "  se: relative Se vapour supply, one of 0.1, 1, 10, 100",
             "Each run reports F and three readouts: x_top_Se (Se fraction in the top layer), "
             "x_bot_Se (Se fraction in the bottom layer, should be 0) and defects (vacancy/damage fraction). "
             "Readouts carry about 0.02 noise. The campaign has 30 runs in total."]
    if use_lit:
        lines += ["", "Literature: " + LITERATURE]
    if use_data:
        lines += ["", "Earlier campaign on WS2 -> Janus WSSe (a related but different material, same furnace):"]
        lines += [f"  {_fmt(R[i])} -> F={o['F']:.2f} x_top_Se={o['x_top_Se']:.2f} x_bot_Se={o['x_bot_Se']:.2f} defects={o['defects']:.2f}"
                  for i, o in zip(prior["idx"], prior["obs"])]
    lines += ["", f"MoS2 runs so far ({len(hist_idx)} of 30):"]
    lines += [f"  {_fmt(R[i])} -> F={o['F']:.2f} x_top_Se={o['x_top_Se']:.2f} x_bot_Se={o['x_bot_Se']:.2f} defects={o['defects']:.2f}"
              for i, o in zip(hist_idx, hist_obs)]
    lines += ["", "Choose the next recipe (not one already run). Answer with one JSON object only, e.g. "
              '{"strip": 0.5, "T_C": 600, "t_min": 30, "se": 10}']
    prompt = "\n".join(lines)
    body = json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 200, "temperature": 0.7,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    for k in range(tries):
        t0 = time.time()
        try:
            req = urllib.request.Request(LLM_URL, body, {"Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=300))
            txt = out["choices"][0]["message"]["content"]
            m = re.search(r"\{[^{}]*\}", txt); d = json.loads(m.group(0))
            v = np.array([float(d["strip"]), float(d["T_C"]), float(d["t_min"]), float(d["se"])])
        except Exception as e:  # noqa: BLE001
            log.append({"llm_error": repr(e)[:200], "try": k}); continue
        f = kmc.features(v)[0]
        dist = ((X - f) ** 2).sum(1); dist[hist_idx] = np.inf
        i = int(np.argmin(dist))
        log.append({"llm_s": round(time.time() - t0, 2), "tokens_in": out["usage"]["prompt_tokens"],
                    "tokens_out": out["usage"]["completion_tokens"], "asked": v.tolist(),
                    "snapped": bool(dist[i] > 1e-9)})
        return i
    log.append({"llm_fallback": "random"})
    rest = np.setdiff1d(np.arange(NC), hist_idx)
    return int(np.random.default_rng(len(hist_idx)).choice(rest))


# ---------------------------------------------------------------- campaign
def make_prior(seed):
    rng = np.random.default_rng(10_000 + seed)
    idx = rng.choice(NC, N_PRIOR, replace=False)
    obs = [kmc.measure(R[i], P_W, rng) for i in idx]
    return {"idx": idx, "obs": obs, "F": np.array([o["F"] for o in obs])}


def run(arm, seed):
    base, has_prior = arm.replace("_prior", ""), arm.endswith("_prior")
    rng = np.random.default_rng(seed)
    prior = make_prior(seed)
    hist_idx = list(rng.choice(NC, N0, replace=False))
    erng = np.random.default_rng(50_000 + seed)
    hist_obs = [kmc.measure(R[i], P_MO, erng) for i in hist_idx]
    who = ["init"] * N0; log = []
    flies = {}

    def propose(agent, pr_data, pr_lit, taken):
        hi = hist_idx + taken  # recipes already queued this round count as taken (mean placeholder value)
        hy = [o["F"] for o in hist_obs]
        if agent == "random":
            rest = np.setdiff1d(np.arange(NC), hi); return int(rng.choice(rest))
        if agent == "gp":
            return gp_propose(hi, hy + [np.mean(hy)] * len(taken), prior if pr_data else None)
        if agent in ("fly", "fly_shuf"):
            if agent not in flies:
                flies[agent] = Fly(np.random.default_rng(777 + seed), shuffled=(agent == "fly_shuf"))
            return flies[agent].propose(hi, hy + [np.mean(hy)] * len(taken), prior if pr_data else None)
        if agent == "llm":
            obs = hist_obs + [{"F": float("nan"), "x_top_Se": float("nan"), "x_bot_Se": float("nan"), "defects": float("nan")}] * len(taken)
            return llm_propose(hi, obs, prior, pr_lit, pr_data, log)

    while len(hist_idx) < BUDGET:
        if base.startswith("net_"):
            cfg = {"net_none": dict(gp=(0, 0), fly=(0, 0), llm=(0, 0)),
                   "net_all": dict(gp=(1, 1), fly=(1, 1), llm=(1, 1)),
                   "net_split": dict(gp=(1, 0), fly=(0, 0), llm=(0, 1))}[base]
            taken = []
            for ag in ("gp", "fly", "llm"):
                if len(hist_idx) + len(taken) >= BUDGET:
                    break
                d, l = cfg[ag]
                taken.append(propose(ag, d, l, taken)); who.append(ag)
            for i in taken:
                hist_idx.append(i); hist_obs.append(kmc.measure(R[i], P_MO, erng))
        else:
            i = propose(base, has_prior, has_prior, [])
            hist_idx.append(i); hist_obs.append(kmc.measure(R[i], P_MO, erng)); who.append(base)
    tr = TRUTH[hist_idx]
    return {"arm": arm, "seed": seed, "idx": [int(i) for i in hist_idx], "who": who, "obs": hist_obs,
            "true_F": tr.tolist(), "best_true_curve": np.maximum.accumulate(tr).tolist(), "llm_log": log}


if __name__ == "__main__":
    out, arms, seeds = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(out, exist_ok=True)
    jobs = [(a, s) for a in arms for s in seeds if not os.path.exists(f"{out}/{a}_s{s}.json")]

    def go(j):
        a, s = j; t0 = time.time(); r = run(a, s); r["wall_s"] = time.time() - t0
        json.dump(r, open(f"{out}/{a}_s{s}.json", "w"), default=float)
        print(a, s, f"best={max(r['true_F']):.3f}", f"{r['wall_s']:.0f}s", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(go, jobs))
