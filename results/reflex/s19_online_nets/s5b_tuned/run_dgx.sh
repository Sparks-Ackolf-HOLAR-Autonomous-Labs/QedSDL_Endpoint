#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd work/p4_s19b
{ for s in 1 2 3 4 5 6; do
  echo "$s fly r0 x x x"; echo "$s fly r1e-4 x x x"
  echo "$s net ae16 3e-3 8 0"; echo "$s net ae46 1e-2 8 0"; echo "$s net lstmae16 1e-2 4 0"
  echo "$s net ae16 3e-3 8 59"; echo "$s net ae16 3e-3 8 500"
done; } | xargs -P 10 -L 1 sh -c 'if [ $1 = fly ]; then nice -n 19 python src/reflex/s5b_return.py work/p4_s2s8/wiring_s5.npz out/s5b_s$0_fly_$2.json $0 fly_$2 > out/log_$0_$2.txt 2>&1; else S19_LR=$3 S19_WIN=$4 S19_REPLAY=$5 S19_TAG=_t_rp$5 nice -n 19 python src/reflex/s19_online_nets.py s5b out $0 $2 > out/log_$0_$2_$5.txt 2>&1; fi'
echo done > work/p4_s19b/done.txt
