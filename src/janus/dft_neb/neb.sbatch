#!/bin/bash -l
#SBATCH --job-name=p4_neb
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=neb_%A_%a.log
# DFT (PBE) check of the two MACE-MP-0 chalcogen-vacancy hop barriers used by the Janus
# selenization model (P4). Free guest partition; account/partition/qos passed by submit.sh.
set -euo pipefail
module load openmpi/5.0.3 quantum-espresso/7.4 python/3.10.3
export OMP_NUM_THREADS=1
cd "${P4_NEB_ROOT:?}"
echo "task=$SLURM_ARRAY_TASK_ID restart=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME"
python3 run_neb.py
