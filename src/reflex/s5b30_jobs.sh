#!/bin/bash
# S5b extension to 30 seeds: writes the job list for seeds 7-30 of every S5b arm
# (connectome model at six recovery rates, equal-memory kNN and z-score; the S19b tuned online nets;
# the nine S21 networks with four replay designs), plus seeds 1-6 of the S5b arms whose files were written
# before the contrast correction (model r=3e-4, kNN, z-score). One line per run, grouped by seed so that a
# partial collection still gives complete seeds. Usage: s5b30_jobs.sh RUN_DIR S21_MAIN_TXT > jobs.txt
R=$1; M=$2; PY=python
ret() { echo "nice -n 19 $PY $R/src/s5b_return.py work/p4_s2s8/wiring_s5.npz $R/s5b_return/s5b_s$1_$2.json $1 $2 > $R/log/ret_$1_$2.txt 2>&1"; }
for s in 1 2 3 4 5 6; do for d in fly_r3e-4 knn_budget zscore; do ret $s $d; done; done
for s in $(seq 7 30); do
  for d in fly_r0 fly_r1e-5 fly_r3e-5 fly_r1e-4 fly_r3e-4 fly_r1e-3 knn_budget zscore; do ret $s $d; done
  for c in "ae16 3e-3 8 0" "ae46 1e-2 8 0" "lstmae16 1e-2 4 0" "ae16 3e-3 8 59" "ae16 3e-3 8 500"; do
    set -- $c
    echo "S19_LR=$2 S19_WIN=$3 S19_REPLAY=$4 S19_TAG=_t_rp$4 nice -n 19 $PY $R/src/s19_online_nets.py s5b $R/s19 $s $1 > $R/log/s19_${s}_$1_$4.txt 2>&1"
  done
  grep "s5b work/p4_s21/s5b 1 " "$M" | sed -e "s# 1 \([a-z0-9]*\) \([a-z]*\)\$# $s \1 \2 > $R/log/s21_${s}_\1_\2.txt 2>\&1#" \
      -e "s#work/p4_s21/s5b#$R/s21#" -e "s#src/s21_memory_nets.py#$R/src/s21_memory_nets.py#" -e "s#\$PY#$PY#" \
      -e "s#python#nice -n 19 $PY#"
done
