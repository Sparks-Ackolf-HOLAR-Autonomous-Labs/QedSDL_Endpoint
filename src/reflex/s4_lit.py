"""S4L: S4 with the in-situ Raman probe re-drawn from what published in-situ studies show.

No published in-situ (at temperature, in Se vapour) Raman of MoS2 thermal selenization was found
(results/reflex/s4_insitu/INSITU_RAMAN_LIT.md). The closest is Harris et al., ACS Nano 2023,
10.1021/acsnano.2c09952: WS2 -> Janus WSSe -> WSe2 by Se implantation at 450 C, one spectrum per pulse.
Its observations, carried over to the Mo-S-Se line as assumptions, change the S4 probe in four ways:

  1. Spectra are taken at the anneal temperature, so lines red-shift with T_C. Harris measured the
     S-side mode 1.6 cm^-1 and the Se-side A1 4.4 cm^-1 lower at 450 C than at room temperature,
     i.e. about -0.0038 and -0.0104 cm^-1/K (W analogue, assumed linear, applied to Mo lines).
     T_C spans 250-900 C, so the shift varies with the recipe the detector is judging.
  2. The Janus A1 line moves with the top-layer Se fraction instead of sitting at its final position:
     it appears about 10 cm^-1 low and moves up as the top layer completes (Harris Fig. 4d;
     Guo et al., PNAS 2021, 10.1073/pnas.2106124118, ex situ for MoSSe). Here 280 + 10 x_top cm^-1.
  3. Lines broaden with temperature (assumed factor 1 + (T_C - 25)/600).
  4. Thermal emission adds a background that rises with T_C and with Raman shift (assumed form).
A stress arm doubles the temperature shift (mode coefficients of MoS2 monolayers reported in the
range -0.01 to -0.016 cm^-1/K would be larger than the W values for the S-side lines).

Everything else (kMC, recipes, failure rule, detectors, seeds) is S4 unchanged, so the result is
paired with S4 seed by seed.
Usage: python s4_lit.py OUT_DIR SEEDS [WORKERS] [SHIFT_SCALE]
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_insitu as B  # noqa: E402
from s4_insitu import K, S, H, gp_ucb, p_success, BUDGET, N0, TB, TD, LINES  # noqa: E402

DW_S, DW_SE = -0.0038, -0.0104  # cm^-1 per K, W analogue (Harris 2023), assumed for Mo lines
SE_SIDE = {("MoSe2", 241), ("MoSe2", 287), ("MoSSe", 290)}
SHIFT_SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0


def raman_lit(st, rng, G, T_C):
    dT = T_C - 25.0
    br = (1 + 3 * st["d_lat"]) * (1 + dT / 600.0)
    xt = st["x_top"]
    s = np.zeros_like(S.NU)
    for ph, fr in (("MoS2", st["f_mos2"]), ("MoSSe", st["f_janus"]), ("MoSe2", st["f_mose2"])):
        ls = []
        for c, a, w in LINES[ph]:
            c0 = 280 + 10 * xt if (ph, c) == ("MoSSe", 290) else c
            dw = (DW_SE if (ph, c) in SE_SIDE else DW_S) * dT * SHIFT_SCALE
            ls.append((c0 + dw, a, w * br))
        s += fr * S.lines(ls)
    s += 2.0 * st["d_lat"] * S.lines([(227 + DW_S * dT * SHIFT_SCALE, 0.6, 12 * (1 + dT / 600.0))])
    s += 0.8 * st["theta"] * S.lines([(250, 1.0, 30)])
    s += 0.3 * (max(dT, 0) / 875.0) ** 4 * (S.NU - 100) / 400.0  # thermal emission (assumed form)
    scale = np.exp(rng.uniform(np.log(0.2), np.log(5)))
    bg = rng.uniform(0, 0.15) * (1 + rng.uniform(-1, 1) * (S.NU - 300) / 200)
    y = scale * (s + bg)
    y = np.maximum(y + rng.normal(0, 0.02 + 0.02 * np.sqrt(np.maximum(y, 0))), 0)
    return S.bands(y, G)


def run_arm(arm, mk, seed, G):
    rng = np.random.default_rng(9000 + seed); prng = np.random.default_rng(7000 + seed)
    start = list(np.random.default_rng(seed).choice(K.NC, N0, replace=False))
    det = mk() if mk else None
    idx, y, fl, done = [], [], [], []
    lost = saved = false_stop = 0; furnace = 0.0
    for n in range(BUDGET):
        if n < N0: j = int(start[n])
        else:
            s = gp_ucb(idx, y)
            if arm == "cbo": s = (s - s.min()) * p_success(idx, fl)
            s = s.astype(float); s[idx] = -np.inf; j = int(np.argmax(s))
        probe, m = B.simulate_probe(K.R[j], K.PAR["Mo"], rng)
        m = {k: float(np.clip(v + rng.normal(0, 0.02), 0, 1)) for k, v in m.items()}
        fail = m["x_bot_Se"] > TB or m["defects"] > TD
        x = raman_lit(probe, prng, G, float(K.R[j][1]))
        stop = det is not None and det.stop(x) and n >= N0
        idx.append(j)
        if stop:
            furnace += B.PROBE_FRAC; saved += fail; false_stop += (not fail)
            y.append(0.0); fl.append(True)
        else:
            furnace += 1.0; lost += fail; y.append(m["F"]); fl.append(fail); done.append(j)
            if det is not None: det.learn(x, fail)
    T = B.T
    return dict(lost=int(lost), saved=int(saved), false_stops=int(false_stop), furnace_runs=float(furnace),
                gap=float(T.max() - T[done].max()) if done else float(T.max()))


def run(seed):
    Wd, Min, G = H.wirings(seed)
    return dict(seed=seed, shift_scale=SHIFT_SCALE,
                arms={a: run_arm(a, mk, seed, G) for a, mk in B.arms(Wd, Min).items() if a not in ("gp", "cbo")})


if __name__ == "__main__":
    out, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
    os.makedirs(out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as ex:
        for r in ex.map(run, seeds):
            json.dump(r, open(f"{out}/s4l_x{SHIFT_SCALE:g}_s{r['seed']}.json", "w"), indent=1)
            print(r["seed"], {a: (d["lost"], d["saved"], d["false_stops"], round(d["gap"], 3)) for a, d in r["arms"].items()}, flush=True)
