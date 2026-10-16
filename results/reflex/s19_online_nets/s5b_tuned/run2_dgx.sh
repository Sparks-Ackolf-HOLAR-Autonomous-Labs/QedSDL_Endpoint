#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd work/p4_s19b
for s in 1 2 3 4 5 6; do for r in 1e-5 3e-5 1e-3; do echo "$s $r"; done; done | xargs -P 12 -L 1 sh -c 'nice -n 19 python src/reflex/s5b_return.py work/p4_s2s8/wiring_s5.npz out/s5b_s$0_fly_r$1.json $0 fly_r$1 > out/log_$0_r$1.txt 2>&1'
echo done > work/p4_s19b/done2.txt
