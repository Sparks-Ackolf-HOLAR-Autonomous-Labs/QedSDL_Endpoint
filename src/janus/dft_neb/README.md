# DFT check of the two MACE-MP-0 hop barriers

The Janus selenization model takes two chalcogen-vacancy hop barriers from MACE-MP-0
(`../ff_energetics.py`): 0.89 eV in the S layer of MoS2 and 0.85 eV in the Se layer of Janus
MoSSe (4x4 monolayer, climbing-image NEB). This folder recomputes both with PBE in
Quantum ESPRESSO 7.4 on the same cell and the same hop.

| File | Role |
|---|---|
| `make_inputs.py` | writes `inputs/` (end-state relaxations and the NEB template) |
| `run_neb.py` | one array task: relax both end states, then `neb.x`; resume-safe after preemption |
| `neb.sbatch`, `submit.sh` | SLURM array (2 tasks, 32 cores each) on the University of Utah CHPC `notchpeak-guest` partition |
| `compare.py` | compares the PBE barriers in `results/janus/dft_neb/` with the MACE-MP-0 values |
| `restart2/` | inputs of the second MoSSe restart |

Settings: PBE, GBRV 1.5 ultrasoft pseudopotentials (Mo, S, Se), 40/320 Ry, cold smearing
0.015 Ry, 3x3x1 k-points, 15 A vacuum, cell fixed at the MACE-MP-0 lattice constant,
5 images with climbing image, path threshold 0.05 eV/A. No dipole correction (the Janus
layer's dipole is nearly the same along the path).

Runs (12-13 Oct 2026, SLURM array 16199490, task 0 = MoS2, task 1 = MoSSe). The MoS2 NEB
converged. The MoSSe NEB stopped at iteration 13 when the SCF of image 2 did not converge
(climbing image 0.11 eV/A, barrier 2.088 eV and still falling). Two restarts followed:

1. Restart from `MoSSe_neb.path` (iteration 13) with `mixing_beta = 0.1`,
   `mixing_mode = 'local-TF'` and `electron_maxstep = 400` (job 16204496_1): the image-2 SCF
   plateaued at 2e-4 to 5e-3 Ry for 400 steps.
2. A fresh `from_scratch` NEB (`restart2/`) using the five iteration-13 images, a new prefix and
   output directory, `diagonalization = 'cg'`, `mixing_ndim = 16`, `mixing_beta = 0.1`
   local-TF, `electron_maxstep = 300`, smearing kept at 0.015 Ry for comparability with MoS2
   (job 16208048): image 2 did not reach SCF convergence in NEB iteration 1 after 9 h.

The reported MoSSe value is therefore 2.09 eV (not converged, climbing image 0.11 eV/A); see
`results/janus/dft_neb/README.md`. `kmc.PBE_HOP["Mo"]["Se"]` holds the barrier used by the
lattice model (`dEa_dmg` follows from it).
