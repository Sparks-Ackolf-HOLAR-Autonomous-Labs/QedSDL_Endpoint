#!/usr/bin/env python3
"""Lattice kinetic Monte Carlo of top-layer selenization: MS2 monolayer -> Janus MSSe.

Two chalcogen layers (top, bottom) on a triangular L x L lattice; site states S, Se,
vacancy, damaged (top only). A recipe is one furnace run:
  strip   fraction of top S removed by an H2-plasma pre-treatment at room temperature
          (the H-stripping route of Lu et al. 2017), 0-1
  T_C     anneal temperature (deg C), Se vapour present throughout
  t_min   anneal time (min)
  se      relative Se supply (vapour pressure relative to a reference), 0.1-100
Processes (Arrhenius, nu = 1e13 s^-1 unless noted), integrated with fixed time steps
(NSTEP per anneal, per-site probability 1 - exp(-k dt), neighbour states updated each step):
  top vacancy filled by Se          nu_ads * se * exp(-Ea_fill/kT)
  top S -> Se exchange              nu * exp(-(Ea_ex + 0.5 dE_top - 0.5 dmu)/kT) * (1 + c_v * n_vac_nb)
  bottom S -> Se exchange           nu * exp(-(Ea_ex + dEa_bot + 0.5 dE_bot - 0.5 dmu)/kT) * (1 + c_v * n_vac_top_nb)
  top chalcogen loss (vacancy)      nu * exp(-Ea_loss/kT); bottom loss with Ea_loss + 0.3 eV
  bottom vacancy refilled by Se     0.01 * fill rate
  plasma over-etching               a share plasma_dmg * strip^3 of stripped sites starts damaged
  excess Se adlayer                 coverage nu_cond*se / (nu_cond*se + nu*exp(-E_des/kT)) at the end, counted as defect
  top vacancy cluster -> damage     nu * exp(-(Ea_hop + dEa_dmg)/kT) if >= 4 vacant/damaged neighbours
dmu = kT ln(se) is the Se chemical-potential shift from the supply.
Force-field inputs (MACE-MP-0, ff_energetics.json): dE_top, dE_bot. Ea_hop (vacancy migration
precedes clustering) is the DFT (PBE, Quantum ESPRESSO) climbing-image NEB barrier of the Mo system
(PBE_HOP, dft_neb/), which MACE-MP-0 under-predicts by about 1.3 eV; W has no PBE run and takes its
MACE-MP-0 barriers shifted by the same Mo offset. dEa_dmg was 1.30 eV on the MACE-MP-0 barriers; it
is re-tuned by that offset, so Ea_hop + dEa_dmg, and with it every process window, is unchanged.
The other barriers are model assumptions set so that the two published
process windows are reproduced (thermal selenization near 800 C without stripping, Zhang et al.
2017; H-stripped selenization near 350 C, Lu et al. 2017); see ASSUMED below.
Readouts (as from Raman/XPS on the flake): top Se fraction, bottom Se fraction, defect fraction.
Objective: Janus fidelity F = x_top_Se * (1 - x_bot_Se) * (1 - defects).
"""
import json, os
import numpy as np

KB = 8.617333e-5
NU, NU_ADS = 1e13, 1e6
NSTEP = 300
_FF_JSON = os.path.join(os.path.dirname(__file__), "../../results/janus/ff_energetics.json")
# PBE vacancy-hop barriers (eV), S layer of MoS2 and Se layer of Janus MoSSe, 4x4 cell (dft_neb/).
PBE_HOP = dict(Mo=dict(S=2.3155, Se=2.0876))
_ff_mo = json.load(open(_FF_JSON))["Mo"]
# mean PBE - MACE-MP-0 hop offset of the Mo system (eV)
HOP_OFFSET = 0.5 * (PBE_HOP["Mo"]["S"] + PBE_HOP["Mo"]["Se"]) - 0.5 * (_ff_mo["Ea_hop_S"] + _ff_mo["Ea_hop_Se"])
DEA_DMG_MACE = 1.30  # damage increment as first tuned on the MACE-MP-0 hop barriers
ASSUMED = dict(Ea_fill=1.00, Ea_ex=3.55, dEa_bot=0.35, Ea_loss=3.70, dEa_dmg=DEA_DMG_MACE - HOP_OFFSET, c_v=2.0, plasma_dmg=0.06, E_des=1.10, nu_cond=100.0)
# W-S bonds are stronger than Mo-S: assumed +0.15 eV on exchange and loss barriers for the W system.
W_SHIFT = 0.15

GRID = dict(strip=[0.0, 0.25, 0.5, 0.75, 1.0],
            T_C=list(range(250, 901, 50)),
            t_min=[5, 15, 30, 60, 120],
            se=[0.1, 1.0, 10.0, 100.0])
NAMES = list(GRID)


def all_recipes():
    g = np.meshgrid(*[np.array(GRID[k], float) for k in NAMES], indexing="ij")
    return np.stack([x.ravel() for x in g], 1)  # (1400, 4)


def features(R):
    """Scaled recipe features in [0, 1] (log for time and supply)."""
    R = np.atleast_2d(R)
    return np.stack([R[:, 0], (R[:, 1] - 250) / 650, np.log(R[:, 2] / 5) / np.log(24),
                     (np.log10(R[:, 3]) + 1) / 3], 1)


def system_params(M, ff_json=_FF_JSON):
    ff = json.load(open(ff_json))[M]
    p = dict(ASSUMED)
    if M in PBE_HOP:
        hop = 0.5 * (PBE_HOP[M]["S"] + PBE_HOP[M]["Se"])
    else:
        hop = 0.5 * (ff["Ea_hop_S"] + ff["Ea_hop_Se"]) + HOP_OFFSET
    p.update(dE_top=ff["dE_top"], dE_bot=ff["dE_bot"], Ea_hop=hop)
    if M == "W":
        p["Ea_ex"] += W_SHIFT; p["Ea_loss"] += W_SHIFT
    return p


def _nb(x):
    return (np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)
            + np.roll(np.roll(x, 1, 0), -1, 1) + np.roll(np.roll(x, -1, 0), 1, 1))


def simulate(recipe, p, rng, L=24):
    strip, T_C, t_min, se = recipe
    S, SE, V, D = 0, 1, 2, 3
    top = np.zeros((L, L), np.int8); bot = np.zeros((L, L), np.int8)
    top[rng.random((L, L)) < strip] = V
    # plasma over-etching: a share of stripped sites (growing as strip^3) is damaged outright
    top[(top == V) & (rng.random((L, L)) < p["plasma_dmg"] * strip ** 3)] = D
    kT = KB * (T_C + 273.15); dmu = kT * np.log(se)
    k_fill = NU_ADS * se * np.exp(-p["Ea_fill"] / kT)
    k_ext = NU * np.exp(-(p["Ea_ex"] + 0.5 * p["dE_top"] - 0.5 * dmu) / kT)
    k_exb = NU * np.exp(-(p["Ea_ex"] + p["dEa_bot"] + 0.5 * p["dE_bot"] - 0.5 * dmu) / kT)
    k_lt = NU * np.exp(-p["Ea_loss"] / kT); k_lb = NU * np.exp(-(p["Ea_loss"] + 0.3) / kT)
    k_dmg = NU * np.exp(-(p["Ea_hop"] + p["dEa_dmg"]) / kT)
    dt = t_min * 60.0 / NSTEP
    P = lambda k: 1.0 - np.exp(-np.minimum(k * dt, 50.0))
    for _ in range(NSTEP):
        u = rng.random((6, L, L))
        nv = _nb(((top == V) | (top == D)).astype(np.float32))
        tv, ts, bs, bv = top == V, top == S, bot == S, bot == V
        new_top = top.copy(); new_bot = bot.copy()
        # a top vacancy is either filled by Se or (if clustered) collapses into damage: competing events
        kd = np.where(nv >= 4, k_dmg, 0.0); ev = tv & (u[0] < P(k_fill + kd))
        dmg = ev & (u[1] < kd / (k_fill + kd))
        new_top[ev & ~dmg] = SE; new_top[dmg] = D
        new_top[ts & (u[2] < P(k_ext * (1 + p["c_v"] * nv)))] = SE
        new_top[((top == S) | (top == SE)) & (u[3] < P(k_lt))] = V
        new_bot[bs & (u[4] < P(k_exb * (1 + p["c_v"] * nv)))] = SE
        lb = (bot != V) & (u[5] < P(k_lb)); new_bot[lb] = V
        new_bot[bv & (u[5] < P(0.01 * k_fill))] = SE
        top, bot = new_top, new_bot
    n = L * L
    x_top = (top == SE).sum() / n; x_bot = (bot == SE).sum() / n
    d_lat = ((top == V) | (top == D)).sum() / n * 0.5 + (bot == V).sum() / n * 0.5
    # excess Se condensed on the surface at the end of the anneal (physisorbed adlayer, steady state
    # between arrival nu_cond*se and desorption nu*exp(-E_des/kT)); it counts as a defect
    k_des = NU * np.exp(-p["E_des"] / kT); theta = p["nu_cond"] * se / (p["nu_cond"] * se + k_des)
    defects = 1 - (1 - d_lat) * (1 - theta)
    F = x_top * (1 - x_bot) * (1 - defects)
    return dict(F=float(F), x_top_Se=float(x_top), x_bot_Se=float(x_bot), defects=float(defects))


def measure(recipe, p, rng, noise=0.02):
    r = simulate(recipe, p, rng)
    return {k: float(np.clip(v + rng.normal(0, noise), 0, 1)) for k, v in r.items()}
