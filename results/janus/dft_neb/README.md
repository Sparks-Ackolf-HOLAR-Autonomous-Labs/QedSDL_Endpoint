# DFT (PBE) check of the MACE-MP-0 chalcogen-vacancy hop barriers

SLURM array 16199490 on notchpeak-guest, both tasks COMPLETED (12 Oct 2026; wall time 6.5 h and 6.0 h
including end-state relaxations).

| System | Hop layer | MACE-MP-0 (eV) | PBE (eV) | PBE - MACE (eV) | Saddle force (eV/A) | Status |
|---|---|---|---|---|---|---|
| MoS2 | S | 0.89 | 2.32 | +1.42 | 0.045 | converged (threshold 0.05) |
| MoSSe | Se | 0.85 | 2.09 | +1.24 | 0.111 | stopped at NEB iteration 13: SCF did not converge on image 2 |

- MoS2: 2.32 eV matches the literature PBE value for an S-vacancy hop in monolayer MoS2 (about 2.3 eV)
  and an earlier PBE run on 2H MoS2 in a different cell (2.14 eV).
- MoSSe: the barrier was still falling slowly when the run stopped (2.119, 2.100, 2.088 eV over
  iterations 11-13), with the climbing image at 0.11 eV/A. The converged value is likely within about
  0.05 eV of 2.09 eV, but it is not converged.
- Both barriers are symmetric (the end states are equivalent by symmetry), so forward = backward.
- Conclusion: MACE-MP-0 under-predicts both hop barriers by 1.2-1.4 eV. The ordering (Se-layer hop
  lower than S-layer hop) is preserved.

What it changes in the model: the hop barrier enters the kMC only through the damage channel
(Ea_hop + assumed dEa_dmg = 1.30 eV, tuned to process windows). Replacing the MACE value with PBE
without re-tuning dEa_dmg would raise the damage barrier from about 2.2 to about 3.6 eV and make
damage negligible at process temperatures; re-tuning dEa_dmg to keep the same process window leaves the kinetics unchanged.

Files: `MoS2.json`, `MoSSe.json` (per-task summary), `summary.md` (written by `compare.py`),
`outputs/` (NEB output, energy profile `*_neb.dat`, task logs). The full NEB scratch directories are not included.

## MoSSe restart (job 16204496_1), 12 Oct 2026

Settings changed for the restart: `mixing_beta = 0.1` (was 0.3), `mixing_mode = 'local-TF'`,
`electron_maxstep = 400`; `neb.x` restarted from iteration 13 (`MoSSe_neb.path`). The job ran on
notch026 for 1 h 55 min (08:11-10:06 MDT) and stopped again at
iteration 13 with "scf convergence NOT achieved on image 2".

- Image 2 SCF: 400 steps, estimated accuracy fell from 2.7 Ry to 3e-4 Ry within about 50 steps and
  then stayed between 2e-4 and 5e-3 Ry (median 1.8e-3 Ry) for the remaining 350 steps; the
  threshold is 1e-7 Ry. Lower mixing did not fix it. The plateau with oscillation points to
  occupation switching of near-degenerate vacancy states at this image, not to charge sloshing.
- No NEB iteration completed, so the path, the profile and the barrier are unchanged:
  MoSSe Se hop 2.09 eV, climbing image 0.11 eV/A, not converged. The values in the paper and in
  `kmc.PBE_HOP` (2.0876 eV) stay as they are.

Files: `restart1/` (task summary `MoSSe.json`, `MoSSe.neb.in`, `MoSSe.neb.out`, task log,
`image2_scf_accuracy_Ry.txt` = the 400 SCF accuracy values of image 2). `MoSSe.json` and
`summary.md` in this folder still describe run 1.

## MoSSe restart 2 (job 16208048), 12-13 Oct 2026: not converged, 2.09 eV is the reported value

Second and last restart: fresh `from_scratch` NEB from the iteration-13 images, `cg`
diagonalization, `mixing_ndim = 16`, `mixing_beta = 0.1` local-TF, `electron_maxstep = 300`.

- The job was preempted twice on notchpeak-guest (notch026 after 2 min, notch500 after 2 min) and
  requeued; the third start (notch491, 17:52 UTC 12 Oct) ran 9 h 08 min and ended at 03:00 UTC
  13 Oct (21:00 MDT 12 Oct) with "scf convergence NOT achieved on image 2" in NEB iteration 1.
  Image 1 converged; image 2 took 3,654 s of CPU and still did not reach the threshold.
- No NEB iteration completed, so the path and the barrier are unchanged: MoSSe Se hop 2.09 eV,
  climbing image 0.11 eV/A, not converged. Three runs with three SCF settings stopped at the same
  image, which supports the near-degenerate vacancy-state reading in the restart-1 note.
- No further restarts were run. `kmc.PBE_HOP` keeps 2.0876 eV, and the paper reports the MoSSe
  barrier as not converged.

Files: `restart2/` (`MoSSe_r2.neb.in`, `MoSSe_r2.neb.out`, `MoSSe_r2.path`, job log).
