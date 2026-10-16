#!/bin/bash
# Detached runner: S5b long stream, first-setting online nets (lr 1e-3, window 8), seeds 7-30 plus a seed-1 reproducibility check.
R=work/p4_s5b_first; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u > start.txt
xargs -P 10 -I{} sh -c '{}' < jobs.txt > xargs.log 2>&1; date -u > done.txt
