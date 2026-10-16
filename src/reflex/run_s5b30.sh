#!/bin/bash
# Detached runner for the S5b 30-seed extension on the DGX (12 lanes, nice 19, one thread per run).
R=work/p4_s5b30; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -u > start.txt
xargs -P 12 -I{} sh -c '{}' < jobs.txt > xargs.log 2>&1; date -u > done.txt
