#!/bin/bash
# S25a: run from connectome_model/. Wiring from the S24 cache; ~/venv (mat_theory lacks torch/sklearn/pandas).
R=results/reflex/s25_memory_need
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HOME=$WORKSPACE/s24_home
export S24_WCACHE=$WORKSPACE/s24_home/s24_wiring.npz
xargs -P 8 -L 1 nice -n 10 python src/reflex/s25_memory.py < $R/a_jobs.txt >> $R/a.log 2>&1
date -u > $R/a_done.txt
