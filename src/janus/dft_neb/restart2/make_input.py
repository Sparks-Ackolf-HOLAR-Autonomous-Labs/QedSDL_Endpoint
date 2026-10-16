#!/usr/bin/env python3
"""Build the second MoSSe NEB restart: fresh start from the iteration-13 path.

Reads ../MoSSe.neb.in (settings and atom labels) and ../MoSSe_neb.path (the five images of
iteration 13, Cartesian bohr) in the CHPC run folder and writes MoSSe_r2.neb.in with
restart_mode 'from_scratch', the three inner images as INTERMEDIATE_IMAGE, a new prefix and
outdir (fresh charge densities), conjugate-gradient diagonalization and a longer Broyden
mixing history. Smearing and every other setting stay as in the MoS2 run.
Usage (in <run>/r2): python3 make_input.py [restart]
"""
import re
import sys
from pathlib import Path

src = Path("../MoSSe.neb.in").read_text()
first = src.split("FIRST_IMAGE")[1].split("ATOMIC_POSITIONS angstrom")[1].split("LAST_IMAGE")[0]
labels = [l.split()[0] for l in first.strip().splitlines()]
assert len(labels) == int(re.search(r"nat = (\d+)", src).group(1)), len(labels)

lines = Path("../MoSSe_neb.path").read_text().splitlines()
imgs, i = [], 0
while i < len(lines):
    if lines[i].startswith("Image:"):
        imgs.append([[float(x) for x in l.split()[:3]] for l in lines[i + 2:i + 2 + len(labels)]])
        i += 2 + len(labels)
    else:
        i += 1
assert len(imgs) == 5, len(imgs)


def block(xyz):
    return "ATOMIC_POSITIONS bohr\n" + "\n".join(
        f"  {s}  {x:.10f}  {y:.10f}  {z:.10f}" for s, (x, y, z) in zip(labels, xyz))


pos = ("BEGIN_POSITIONS\nFIRST_IMAGE\n" + block(imgs[0]) + "\n"
       + "".join("INTERMEDIATE_IMAGE\n" + block(m) + "\n" for m in imgs[1:4])
       + "LAST_IMAGE\n" + block(imgs[4]) + "\nEND_POSITIONS")
out = re.sub(r"BEGIN_POSITIONS.*END_POSITIONS", lambda _: pos, src, flags=re.S)
mode = "restart" if len(sys.argv) > 1 and sys.argv[1] == "restart" else "from_scratch"
out = re.sub(r"restart_mode = '\w+'", f"restart_mode = '{mode}'", out)
out = (out.replace("prefix = 'MoSSe_neb'", "prefix = 'MoSSe_r2'")
          .replace("outdir = './out_MoSSe_neb'", "outdir = './out_MoSSe_r2'")
          .replace("pseudo_dir = './pseudos'", "pseudo_dir = '../pseudos'"))
out = re.sub(r"&ELECTRONS.*?\n/", "&ELECTRONS\n  diagonalization = 'cg'\n  mixing_beta = 0.1\n"
             "  mixing_mode = 'local-TF'\n  mixing_ndim = 16\n  conv_thr = 1.0d-7\n"
             "  electron_maxstep = 300\n/", out, flags=re.S)
Path("MoSSe_r2.neb.in").write_text(out)
print(mode, "images", len(imgs), "atoms", len(labels))
