#!/bin/bash
# S25b-recal: run from connectome_model/ on the DGX. Same environment as run_a.sh; 6 workers per job.
R=results/reflex/s25_memory_need
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HOME=$WORKSPACE/s24_home
export S24_WCACHE=$WORKSPACE/s24_home/s24_wiring.npz
for j in flag stop furnace dose; do nice -n 10 python src/reflex/s25_recal.py $j $R/recal 6 >> $R/recal.log 2>&1; done
date -u > $R/recal_done.txt
