#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
B=work/p4_s13; cd $B
for m in janus multi; do for s in $(seq 1 20); do echo "$m $s"; done; done |
 xargs -P 8 -n 2 sh -c 'S1_MODE=$0 nice -n 19 python s13_cross_animal.py wiring out/$0 $1 > log_$0_$1.txt 2>&1'
echo done > $B/done.txt
