#!/usr/bin/env python3
"""Quantum ESPRESSO inputs for the DFT check of the two MACE-MP-0 hop barriers.

Same geometry and hop as ff_energetics.py: a 4x4 2H monolayer at the MACE lattice constant,
one top-layer chalcogen removed (the first top-layer site), and its nearest top-layer
neighbour moved into the empty site.
  MoS2   vacancy hop in the S layer of MoS2        (MACE-MP-0: 0.89 eV)
  MoSSe  vacancy hop in the Se layer of Janus MoSSe (MACE-MP-0: 0.85 eV)
Writes, per system: <sys>.ini.in and <sys>.fin.in (pw.x fixed-cell relax of the end states)
and <sys>.neb.tmpl (neb.x input; run_neb.py fills in the relaxed end states).
Settings: PBE, GBRV 1.5 ultrasoft pseudopotentials, 40/320 Ry, cold smearing 0.015 Ry,
3x3x1 k-points, 15 A vacuum, 5 images (3 moving), climbing image, path_thr 0.05 eV/A.
Usage: python make_inputs.py   (needs ASE)
"""
import json
from pathlib import Path

import numpy as np
from ase.build import mx2

HERE = Path(__file__).resolve().parent
OUT = HERE / "inputs"
FF = json.loads((HERE.parents[2] / "results" / "janus" / "ff_energetics.json").read_text())
N = 4
SPECIES = {"Mo": ("95.95", "mo_pbe_v1.uspp.F.UPF"), "S": ("32.06", "s_pbe_v1.4.uspp.F.UPF"),
           "Se": ("78.97", "se_pbe_v1.uspp.F.UPF")}
SYSTEMS = {"MoS2": ("S", "S", FF["Mo"]["a_MS2"], FF["Mo"]["Ea_hop_S"]),
           "MoSSe": ("Se", "S", FF["Mo"]["a_MSSe"], FF["Mo"]["Ea_hop_Se"])}


def cell(top, bot, a):
    at = mx2("MoS2", "2H", a=a, thickness=3.2, size=(N, N, 1), vacuum=7.5)
    z = at.positions[:, 2]
    sym = at.get_chemical_symbols()
    zM = z[[s == "Mo" for s in sym]].mean()
    at.set_chemical_symbols([(top if z[i] > zM else bot) if s == "S" else s for i, s in enumerate(sym)])
    return at


def header(prefix, calc, at):
    sp = sorted(set(at.get_chemical_symbols()))
    c = at.cell
    return f"""&CONTROL
  calculation = '{calc}'
  prefix = '{prefix}'
  pseudo_dir = './pseudos'
  outdir = './out_{prefix}'
  disk_io = 'low'
  forc_conv_thr = 1.0d-3
  etot_conv_thr = 1.0d-5
  nstep = 150
/
&SYSTEM
  ibrav = 0
  nat = {len(at)}
  ntyp = {len(sp)}
  ecutwfc = 40.0
  ecutrho = 320.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.015
/
&ELECTRONS
  mixing_beta = 0.3
  conv_thr = 1.0d-7
  electron_maxstep = 150
/
&IONS
/
ATOMIC_SPECIES
""" + "".join(f"  {s} {SPECIES[s][0]} {SPECIES[s][1]}\n" for s in sp) + \
        "CELL_PARAMETERS angstrom\n" + "".join(f"  {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n" for v in c)


def pos(at):
    return "\n".join(f"  {s}  {x:.8f}  {y:.8f}  {z:.8f}" for s, (x, y, z) in
                     zip(at.get_chemical_symbols(), at.positions))


KPTS = "K_POINTS automatic\n  3 3 1 0 0 0\n"


def main():
    OUT.mkdir(exist_ok=True)
    man = {}
    for name, (top, bot, a, ea_mace) in SYSTEMS.items():
        at = cell(top, bot, a)
        sym = np.array(at.get_chemical_symbols()); z = at.positions[:, 2]
        zM = z[sym == "Mo"].mean()
        idx = [i for i in range(len(at)) if sym[i] != "Mo" and z[i] > zM]
        v = idx[0]
        d = at.get_distances(v, idx[1:], mic=True); nb = idx[1:][int(np.argmin(d))]
        ini = at.copy(); del ini[v]
        nb2 = nb - 1 if nb > v else nb
        step = at.get_distance(nb, v, mic=True, vector=True)  # shortest periodic image of the hop
        fin = ini.copy(); p = fin.positions; p[nb2] = at.positions[nb] + step; fin.positions = p
        for st, x in (("ini", ini), ("fin", fin)):
            (OUT / f"{name}.{st}.in").write_text(header(f"{name}_{st}", "relax", x)
                                                 + "ATOMIC_POSITIONS angstrom\n" + pos(x) + "\n" + KPTS)
        eng = header(f"{name}_neb", "scf", ini).replace("  calculation = 'scf'\n", "").replace("&IONS\n/\n", "")
        (OUT / f"{name}.neb.tmpl").write_text(f"""BEGIN
BEGIN_PATH_INPUT
&PATH
  restart_mode = '@RESTART@'
  string_method = 'neb'
  nstep_path = 150
  num_of_images = 5
  opt_scheme = 'broyden'
  CI_scheme = 'auto'
  path_thr = 0.05
/
END_PATH_INPUT
BEGIN_ENGINE_INPUT
{eng}BEGIN_POSITIONS
FIRST_IMAGE
ATOMIC_POSITIONS angstrom
@FIRST@
LAST_IMAGE
ATOMIC_POSITIONS angstrom
@LAST@
END_POSITIONS
{KPTS}END_ENGINE_INPUT
END
""")
        man[name] = {"layer": top, "nat": len(ini), "a_A": a, "hop_distance_A": float(d.min()),
                     "c_A": float(at.cell[2, 2]), "Ea_mace_eV": ea_mace}
        print(name, len(ini), round(a, 4), round(float(d.min()), 3), round(float(at.cell[2, 2]), 2))
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
