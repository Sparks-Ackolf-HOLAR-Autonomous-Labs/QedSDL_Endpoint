#!/usr/bin/env python3
"""S26a: martingale posterior neural networks (Duran-Martin, Sanchez-Betancourt, Cartea and Murphy, NeurIPS 2025) as the
recipe planner on the Janus selenization map, run with the authors' code and settings
(github.com/gerdm/martingale-posterior-neural-networks @1a1e518, experiments/bayesopt: agents.toml, Thompson sampling over
a grid). Protocol of src/janus/campaign.py: 1,400-recipe grid, the same 3 initial recipes and the same measurement noise
stream per seed, 30 experiments, true fidelity from truth_Mo.npy.
Arms: FLoRES (low-rank last layer) and LRKF (low-rank filter over all weights), the paper's two martingale-posterior
networks. Their BO settings take obs_noise = 0 (noiseless test functions); on the noisy fidelity readings
(kmc.measure adds Gaussian noise of s.d. 0.02) that makes the first Kalman update divide by zero (every score NaN), so
obs_noise is set to the known readout variance 0.02**2 = 4e-4; arm "NAME@x" uses obs_noise = x instead (sensitivity).
Arms "gp_rerun", "fly_rerun", "fly_shuf_rerun", "random_rerun" re-run campaign.py's arms for paired tests.
Records the wall time of every proposal (sample a function, score the grid, take the best untested recipe) and update.
Usage: python s26_mpnn_janus.py OUT_DIR ARM[,ARM...] SEEDS"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.join(os.environ.get("WORKSPACE", os.path.expanduser("~")), "ext/martingale-posterior-neural-networks/experiments")
sys.path.insert(0, os.path.join(HERE, "../janus"))
import kmc  # noqa: E402
import campaign  # noqa: E402

R, X, NC, TRUTH = campaign.R, campaign.X, campaign.NC, campaign.TRUTH
P_MO, BUDGET, N0 = campaign.P_MO, campaign.BUDGET, campaign.N0


def start(seed):
    """Initial recipes and measurement stream exactly as campaign.run."""
    rng = np.random.default_rng(seed)
    campaign.make_prior(seed)
    hist_idx = list(rng.choice(NC, N0, replace=False))
    erng = np.random.default_rng(50_000 + seed)
    hist_obs = [kmc.measure(R[i], P_MO, erng) for i in hist_idx]
    return rng, erng, hist_idx, hist_obs


def run_mpnn(arm, seed):
    name, _, nz = arm.partition("@")
    sys.path.insert(0, MP)
    import jax, jax.numpy as jnp, toml
    from bayesopt import agents
    cfg = dict(toml.load(os.path.join(MP, "bayesopt/agents.toml"))[name])
    cfg["obs_noise"] = float(nz) if nz else 0.02 ** 2
    agent, init_fn = agents.AGENTS[name](jnp.zeros(X.shape[1]), **cfg)
    bel = init_fn(jax.random.PRNGKey(seed))
    _, erng, hist_idx, hist_obs = start(seed)
    Xg = jnp.asarray(X, dtype=jnp.float32)
    upd = jax.jit(agent.update)
    for i, o in zip(hist_idx, hist_obs):
        bel = upd(bel, jnp.asarray(o["F"], jnp.float32), Xg[i])
    key = jax.random.PRNGKey(100_000 + seed)
    t_prop, t_upd = [], []
    score = jax.jit(lambda k, b: agent.sample_fn(k, b)(Xg).squeeze())
    t = 0
    while len(hist_idx) < BUDGET:
        t0 = time.perf_counter()
        s = np.array(score(jax.random.fold_in(key, t), bel), dtype=float)
        assert not np.isnan(s).any(), "NaN scores"
        s[hist_idx] = -np.inf
        i = int(np.argmax(s)); t_prop.append(time.perf_counter() - t0)
        o = kmc.measure(R[i], P_MO, erng); hist_idx.append(i); hist_obs.append(o)
        t0 = time.perf_counter(); bel = upd(bel, jnp.asarray(o["F"], jnp.float32), Xg[i])
        jax.block_until_ready(bel); t_upd.append(time.perf_counter() - t0); t += 1
    return hist_idx, hist_obs, t_prop, t_upd


def run_rerun(arm, seed):
    base = arm.replace("_rerun", "")
    t0 = time.perf_counter(); r = campaign.run(base, seed); wall = time.perf_counter() - t0
    tp = []
    if base == "gp":   # time one proposal at the full history, as the planner would face it late in the campaign
        hy = [o["F"] for o in r["obs"]]
        for _ in range(3):
            t1 = time.perf_counter(); campaign.gp_propose(r["idx"][:-1], hy[:-1], None); tp.append(time.perf_counter() - t1)
    return r["idx"], r["obs"], tp, [wall]


if __name__ == "__main__":
    out, arms, seeds = sys.argv[1], sys.argv[2].split(","), [int(s) for s in sys.argv[3].split(",")]
    os.makedirs(out, exist_ok=True)
    for arm in arms:
        for seed in seeds:
            f = f"{out}/{arm}_s{seed}.json"
            if os.path.exists(f):
                continue
            idx, obs, tp, tu = (run_rerun if arm.endswith("_rerun") else run_mpnn)(arm, seed)
            tr = TRUTH[idx]
            json.dump({"arm": arm, "seed": seed, "idx": [int(i) for i in idx], "obs": obs, "true_F": tr.tolist(),
                       "best_true_curve": np.maximum.accumulate(tr).tolist(), "t_propose_s": tp, "t_update_s": tu},
                      open(f, "w"), default=float)
            print(arm, seed, f"gap30={campaign.TRUTH.max() - tr.max():.3f}",
                  f"prop={np.median(tp[1:]) * 1e3 if len(tp) > 1 else float('nan'):.1f}ms", flush=True)
