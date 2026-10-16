#!/bin/bash
# Stage and submit the second MoSSe NEB restart. Needs the Duo-approved ControlMaster.
set -euo pipefail
cd "$(dirname "$0")"
SSHOPT="-o ControlPath=$HOME/.ssh/cm/%r@%h:%p"
G="$CHPC_USERNAME@notchpeak.chpc.utah.edu"
R() { ssh $SSHOPT "$G" "bash -l -s" <<<"$*"; }
D=$(R 'echo /scratch/general/vast/$USER/p4_neb')
R "mkdir -p $D/r2"
scp -q $SSHOPT make_input.py neb_r2.sbatch "$G:$D/r2/"
R "cd $D/r2 && python3 make_input.py && grep -c INTERMEDIATE_IMAGE MoSSe_r2.neb.in"
ARGS="--account=owner-guest --partition=notchpeak-guest --qos=notchpeak-guest --export=ALL,P4_NEB_ROOT=$D"
R "cd $D/r2 && sbatch --test-only $ARGS neb_r2.sbatch"
if [ "${P4_NEB_SUBMIT:-0}" = "1" ]; then
  R "cd $D/r2 && sbatch $ARGS neb_r2.sbatch" | tee .jobid
  R "squeue -u \$USER -n p4_neb_r2 -o '%.18i %.9P %.2t %.10M %R'"
fi
