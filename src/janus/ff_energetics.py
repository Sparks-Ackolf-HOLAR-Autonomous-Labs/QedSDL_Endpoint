#!/usr/bin/env python3
"""Force-field energetics for the Janus selenization campaign.

MACE-MP-0 (medium, float64) on 4x4 2H monolayers of MS2, MSe2 and Janus MSSe
(M = Mo, W). Quantities passed to the kinetic Monte Carlo model (kmc.py):
  a_*            in-plane lattice constants from an E(a) scan (A)
  E_mix_janus    E(MSSe) - [E(MS2) + E(MSe2)]/2 per formula unit (eV)
  dE_top         one Se replacing a top-layer S in MS2, referenced to the end members (eV)
  dE_bot         one Se replacing a bottom-layer S in Janus MSSe, same reference (eV)
  Ea_hop_S       in-plane chalcogen-vacancy hop barrier, S layer of MS2 (NEB, eV)
  Ea_hop_Se      same, Se layer of Janus MSSe (NEB, eV)
Reference for substitution energies: mu_Se - mu_S = [E(MSe2) - E(MS2)]/2 per formula unit.
Usage: python ff_energetics.py OUT_JSON
"""
import json, sys, time
import numpy as np
from ase.build import mx2
from ase.optimize import FIRE, BFGS
from ase.mep import NEB
from mace.calculators import mace_mp

calc = mace_mp(model="medium", device="cuda", default_dtype="float64")
N = 4  # 4x4 supercell, 16 formula units


def cell(M, top, bot, a=3.2, n=N):
    at = mx2(f"{M}S2", "2H", a=a, thickness=3.2, size=(n, n, 1), vacuum=10.0)
    z = at.positions[:, 2]; zm = z[at.numbers == at[0].number].mean() if at[0].symbol == M else None
    zM = z[[s == M for s in at.get_chemical_symbols()]].mean()
    sym = at.get_chemical_symbols()
    for i, s in enumerate(sym):
        if s == "S":
            sym[i] = top if z[i] > zM else bot
    at.set_chemical_symbols(sym)
    return at


def a_eq(M, top, bot, a0):
    """In-plane lattice constant from a quadratic fit of E(a) on the 1x1 cell (atoms relaxed at each a).
    (A cell filter stopped early on these slab cells, so the scan is used instead.)"""
    A = a0 + np.linspace(-0.08, 0.08, 9); E = []
    for a in A:
        at = cell(M, top, bot, a, n=1); at.calc = calc
        BFGS(at, logfile=None).run(fmax=0.005, steps=300); E.append(at.get_potential_energy())
    p = np.polyfit(A, E, 2)
    return float(-p[1] / (2 * p[0]))


def relax(at, cellopt=False, fmax=0.01):
    at.calc = calc
    BFGS(at, logfile=None).run(fmax=fmax, steps=600)
    return at.get_potential_energy()


def layer_idx(at, which):
    sym = np.array(at.get_chemical_symbols()); z = at.positions[:, 2]
    zM = z[(sym == "Mo") | (sym == "W")].mean()
    return [i for i in range(len(at)) if sym[i] in ("S", "Se") and ((z[i] > zM) == (which == "top"))]


def neb_hop(at0, which):
    """Vacancy in the given chalcogen layer hops to a nearest-neighbour site of the same layer."""
    at = at0.copy(); at.calc = calc
    idx = layer_idx(at, which)
    v = idx[0]
    d = at.get_distances(v, idx[1:], mic=True); nb = idx[1:][int(np.argmin(d))]
    target = at.positions[v].copy()
    del at[v]
    nb = nb - 1 if nb > v else nb
    ini = at.copy(); ini.calc = calc; BFGS(ini, logfile=None).run(fmax=0.02, steps=500)
    fin = ini.copy(); fin.positions[nb] = target + (fin.positions[nb] - at.positions[nb]) * 0  # move neighbour into vacancy
    fin.calc = calc; BFGS(fin, logfile=None).run(fmax=0.02, steps=500)
    imgs = [ini] + [ini.copy() for _ in range(5)] + [fin]
    for im in imgs:
        im.calc = calc
    neb = NEB(imgs, climb=True, allow_shared_calculator=True)
    neb.interpolate(mic=True)
    FIRE(neb, logfile=None).run(fmax=0.05, steps=800)
    E = []
    for im in imgs:
        E.append(calc.get_potential_energy(im))
    return float(max(E) - E[0]), [float(e - E[0]) for e in E]


out = {"model": "MACE-MP-0 medium (2023-12-03), float64", "supercell": f"{N}x{N} 2H monolayer"}
t0 = time.time()
for M in ("Mo", "W"):
    r = {}
    r["a_MS2"], r["a_MSe2"], r["a_MSSe"] = a_eq(M, "S", "S", 3.18), a_eq(M, "Se", "Se", 3.31), a_eq(M, "Se", "S", 3.25)
    sS, sSe, sJ = cell(M, "S", "S", r["a_MS2"]), cell(M, "Se", "Se", r["a_MSe2"]), cell(M, "Se", "S", r["a_MSSe"])
    E_S, E_Se, E_J = relax(sS), relax(sSe), relax(sJ)
    nfu = N * N
    r["E_mix_janus"] = (E_J - 0.5 * (E_S + E_Se)) / nfu
    dmu = (E_Se - E_S) / nfu / 2.0
    t = sS.copy(); i = layer_idx(t, "top")[0]; t[i].symbol = "Se"; t.calc = calc
    r["dE_top"] = relax(t, cellopt=False) - E_S - dmu
    b = sJ.copy(); i = layer_idx(b, "bot")[0]; b[i].symbol = "Se"; b.calc = calc
    r["dE_bot"] = relax(b, cellopt=False) - E_J - dmu
    r["Ea_hop_S"], r["neb_S"] = neb_hop(sS, "top")
    r["Ea_hop_Se"], r["neb_Se"] = neb_hop(sJ, "top")
    out[M] = r
    print(M, {k: v for k, v in r.items() if not k.startswith("neb")}, f"{time.time()-t0:.0f}s", flush=True)
out["wall_s"] = time.time() - t0
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
