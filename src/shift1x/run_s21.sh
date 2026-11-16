#!/usr/bin/env bash
# G1: S21 extra networks on the S20 time-course stop (Table S11, Note S3) at an explicit SHIFT_SCALE (default 1).
# Same commands as P4 results/reflex/s21_memory_nets/run_s20.sh, which passed WORKERS=4 as argv[4] and so ran at 4x,
# but through shift1x/pin_shift.py --shift-scale. D-018: <= 4 workers, nice + ionice, nohup, logs end rc=.
#   cd $RA && SHIFT_SCALE=1 BLOCKS="s21_tune s21_test" nohup bash shift1x/run_s21.sh > shift1x/launch_s21.nohup 2>&1 < /dev/null &
# BLOCKS picks any of:
#   s21_tune  lr 3e-3 and 1e-2 (600 epochs, 16 units), seeds 101-105, cal10, the 5 nets
#             -> shift1x/s21/s20_tune/s20_cal10_s{seed}_s21_lr{lr}.json, then the per-net pick -> shift1x/s21_pick.{txt,json}
#   s21_test  lr 1e-2 and 3e-3, cal10 and cal3, seeds 1-20, the 5 nets (both lrs, as published; the analysis uses the pick)
#             -> shift1x/s21/s20/s20_cal{10,3}_s{seed}_s21_lr{lr}.json
# The S20 reference arms (seq_kc, snap_kc, snap_mlp, tuned LSTM) come from the 1x runs already in shift1x/s20/.
set -u
RA=${RA:?set RA to the working folder that holds shift1x/}
PY=${PY:-python3}
O=$RA/shift1x
D=${OUT_ROOT:-$O}   # output root (smoke tests only; default shift1x/)
SEEDS=${SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
TSEEDS=${TUNE_SEEDS:-101,102,103,104,105}   # override for smoke tests only
W=${WORKERS:-4}
BLOCKS=${BLOCKS:-"s21_tune s21_test"}
SCALE=${SHIFT_SCALE:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 S20_ARMS=net_gru,net_peep,net_mog,net_tfm,net_memnet
mkdir -p $D/logs
[ "$W" -le 4 ] || { echo "D-018: WORKERS must be <= 4"; exit 2; }
blk() {
  local name=$1; shift; local log=$D/logs/$name.log t0=$(date +%s)
  echo "$(date +%FT%T) START $name shift_scale=$SCALE workers=$W args=$* S20_ARMS=$S20_ARMS S20_TAG=${S20_TAG-} S20_LSTM=${S20_LSTM-} load=$(cut -d' ' -f1 /proc/loadavg)" > $log
  nice -n 10 ionice -c3 $PY $O/pin_shift.py --shift-scale $SCALE "$@" >> $log 2>&1; local rc=$?
  echo "$(date +%FT%T) END $name wall_s=$(( $(date +%s) - t0 ))" >> $log
  echo "rc=$rc" >> $log
  echo "$name rc=$rc"
  return $rc
}
for b in $BLOCKS; do
  case $b in
    s21_tune) for lr in 3e-3 1e-2; do S20_LSTM=$lr,600,16 S20_TAG=_s21_lr$lr blk s21_tune_lr$lr s20 $D/s21/s20_tune $TSEEDS 10 $W || exit 3; done
              $PY $O/analysis/pick_s21.py $D/s21/s20_tune $D > $D/logs/s21_pick.log 2>&1; r=$?; echo "rc=$r" >> $D/logs/s21_pick.log; tail -n 2 $D/logs/s21_pick.log; [ $r -eq 0 ] ;;
    s21_test) for lr in 1e-2 3e-3; do for n in 10 3; do S20_LSTM=$lr,600,16 S20_TAG=_s21_lr$lr blk s21_cal${n}_lr$lr s20 $D/s21/s20 $SEEDS $n $W || exit 3; done; done ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac || { echo "block $b failed; stopping"; exit 3; }
done
echo "S21_DONE"
