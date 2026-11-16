#!/usr/bin/env bash
# G1 reruns with s4_lit.SHIFT_SCALE set explicitly (shift1x/pin_shift.py --shift-scale; SHIFT_SCALE env, default 1). D-018: <= 4 workers, nice + ionice, nohup, logs end rc=.
#   cd $RA && SHIFT_SCALE=1 BLOCKS="s18a s20 ..." nohup bash shift1x/run_shift1x.sh > shift1x/launch.nohup 2>&1 < /dev/null &
# BLOCKS picks any of (20 seeds each):
#   s18a         S18a job iv (tab:control)                                  -> shift1x/s18a/
#   s20          S20 time course, 10 calibration flakes, all arms           -> shift1x/s20/s20_cal10_s*.json
#   s20_cal3     S20, 3 calibration flakes, all arms                        -> shift1x/s20/s20_cal3_s*.json
#   s20_tune     LSTM tuning sweep at this SHIFT_SCALE: 7 settings x seeds 101-105, cal10, lstm arm only
#                -> shift1x/s20_tune/s20_cal10_s{101..105}_{lr,epochs,hidden}.json, then the pick -> shift1x/lstm_pick.{txt,json}
#   s20_cal10_lt S20 LSTM with the setting in shift1x/lstm_pick.txt (run s20_tune first), cal10 -> shift1x/s20/s20_cal10_s*_lstmtuned.json
#   s20_cal3_lt  same, cal3                                                 -> shift1x/s20/s20_cal3_s*_lstmtuned.json
#   signed       signed_cord (neuro_reruns), booked separately              -> shift1x/signed/
# All S20 blocks share shift1x/s20/ so the analyzer copy (shift1x/analysis/analyze_s20.py) reads them as P4 does.
set -u
RA=${RA:?set RA to the working folder that holds shift1x/}
PY=${PY:-python3}
O=$RA/shift1x
D=${OUT_ROOT:-$O}   # output root (smoke tests only; default shift1x/)
SEEDS=${SEEDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
W=${WORKERS:-4}
BLOCKS=${BLOCKS:-"s18a s20"}
SCALE=${SHIFT_SCALE:-1}
BANC=NA   # cord caches exist; BANC_DIR is only read to rebuild them
LSTM_DEF=3e-3,600,16   # s20_timecourse.py default
TUNE_GRID=${TUNE_GRID:-"1e-2,2000,16 1e-2,2000,32 1e-2,600,16 1e-3,600,16 3e-3,2000,16 3e-3,600,16 3e-3,600,32"}   # as published; override for smoke tests only (lstm_tuning_seeds101-105)
pick() { [ -s $D/lstm_pick.txt ] || { echo "no $D/lstm_pick.txt: run BLOCKS=s20_tune first" >&2; return 1; }; head -1 $D/lstm_pick.txt; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $D/logs
[ "$W" -le 4 ] || { echo "D-018: WORKERS must be <= 4"; exit 2; }
for f in $HOME/monster_blood/p4_s18/banc_cord.npz $WORKSPACE/g1_visuals/neuro_reruns/cache/banc_cord_signed.npz; do
  [ -s "$f" ] || { echo "missing cache $f"; exit 2; }
done
blk() {
  local name=$1; shift; local log=$D/logs/$name.log t0=$(date +%s)
  echo "$(date +%FT%T) START $name shift_scale=$SCALE workers=$W seeds=$SEEDS S20_ARMS=${S20_ARMS-} S20_TAG=${S20_TAG-} S20_LSTM=${S20_LSTM-} load=$(cut -d' ' -f1 /proc/loadavg)" > $log
  nice -n 10 ionice -c3 $PY $O/pin_shift.py --shift-scale $SCALE "$@" >> $log 2>&1; local rc=$?
  echo "$(date +%FT%T) END $name wall_s=$(( $(date +%s) - t0 ))" >> $log
  echo "rc=$rc" >> $log
  echo "$name rc=$rc"
  return $rc
}
for b in $BLOCKS; do
  case $b in
    s18a)         blk s18a s18a $BANC $D/s18a $SEEDS $W ;;
    s20)          S20_ARMS= S20_TAG= S20_LSTM=$LSTM_DEF blk s20_cal10 s20 $D/s20 $SEEDS 10 $W ;;
    s20_cal3)     S20_ARMS= S20_TAG= S20_LSTM=$LSTM_DEF blk s20_cal3 s20 $D/s20 $SEEDS 3 $W ;;
    s20_tune)     for c in $TUNE_GRID; do SEEDS=101,102,103,104,105 S20_ARMS=lstm S20_TAG=_$c S20_LSTM=$c blk s20_tune_$c s20 $D/s20_tune 101,102,103,104,105 10 $W || exit 3; done
                  $PY $O/analysis/pick_lstm.py $D/s20_tune $D > $D/logs/lstm_pick.log 2>&1; r=$?; echo "rc=$r" >> $D/logs/lstm_pick.log; tail -1 $D/logs/lstm_pick.log; [ $r -eq 0 ] ;;
    s20_cal10_lt) P=$(pick) || exit 2; S20_ARMS=lstm S20_TAG=_lstmtuned S20_LSTM=$P blk s20_cal10_lstmtuned s20 $D/s20 $SEEDS 10 $W ;;
    s20_cal3_lt)  P=$(pick) || exit 2; S20_ARMS=lstm S20_TAG=_lstmtuned S20_LSTM=$P blk s20_cal3_lstmtuned s20 $D/s20 $SEEDS 3 $W ;;
    signed)       blk signed signed $BANC $D/signed $SEEDS $W ;;
    *) echo "unknown block $b"; exit 2 ;;
  esac || { echo "block $b failed; stopping"; exit 3; }
done
echo "SHIFT1X_DONE"
