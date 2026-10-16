#!/bin/bash
cd work/p4_reflex
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/src/reflex/t1_janus_faults.py out/t1 $(seq -s, 1 20) 5 > t1.log 2>&1
echo $? > t1_done.txt
