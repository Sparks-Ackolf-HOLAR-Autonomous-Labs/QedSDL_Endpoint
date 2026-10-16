#!/bin/bash -l
#SBATCH --job-name=p4_neb_r2
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=neb_r2_%j.log
# Second (last) restart of the MoSSe Se-vacancy NEB (P4). Fresh start from the iteration-13
# path, cg diagonalization; a requeue after preemption continues from MoSSe_r2.path.
set -euo pipefail
module load openmpi/5.0.3 quantum-espresso/7.4 python/3.10.3
export OMP_NUM_THREADS=1
cd "${P4_NEB_ROOT:?}/r2"
echo "restart=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME start=$(date -u +%FT%TZ)"
if [ -s MoSSe_r2.path ]; then python3 make_input.py restart; else python3 make_input.py; fi
mpirun -np "$SLURM_NTASKS" neb.x -nk 2 -inp MoSSe_r2.neb.in >> MoSSe_r2.neb.out 2>&1 || echo "neb.x rc=$?"
grep -E "activation energy \(->\)|convergence achieved" MoSSe_r2.neb.out | tail -3
echo "end=$(date -u +%FT%TZ)"
