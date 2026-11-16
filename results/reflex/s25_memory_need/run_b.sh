#!/bin/bash
# S25b: run from connectome_model/. Same environment as run_a.sh; 4 workers per job.
R=results/reflex/s25_memory_need
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HOME=$WORKSPACE/s24_home
export S24_WCACHE=$WORKSPACE/s24_home/s24_wiring.npz
for j in family flagspec stopdrop furnace dose; do nice -n 10 python src/reflex/s25_generic.py $j $R/b 4 >> $R/b.log 2>&1; done
date -u > $R/b_done.txt
