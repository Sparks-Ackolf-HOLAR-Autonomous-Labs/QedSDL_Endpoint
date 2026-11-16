#!/bin/bash
# S24a: run from connectome_model/. FlyWire v783 wiring from one small npz (s24_wcache.py, built from ~/flywire_v783).
# Relaunched 20:45 MDT after the 20:00 out-of-memory crash: 10 workers, about 0.8 GB each; s1b seeds 1-5 had finished.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HOME=$WORKSPACE/s24_home
export S24_WCACHE=$WORKSPACE/s24_home/s24_wiring.npz
xargs -P 10 -L 1 nice -n 10 python src/reflex/s24_fixed_memory.py < results/reflex/s24_fixed_memory/a_jobs_rest.txt >> results/reflex/s24_fixed_memory/a.log 2>&1
date -u > results/reflex/s24_fixed_memory/a_done.txt
