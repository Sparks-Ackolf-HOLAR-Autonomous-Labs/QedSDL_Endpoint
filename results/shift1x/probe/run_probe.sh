#!/usr/bin/env bash
# 18 seed-1 probes: base cal3, lstmtuned cal3, lstmtuned cal10 at argv4 = 1..6. nice 10 + ionice idle, 3 at a time.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-python3}
one() { set -- $1; if [ $1 = lstmtuned ]; then export S20_ARMS=lstm S20_LSTM=1e-2,600,16; fi; nice -n 10 ionice -c3 $PY s20_scale_probe.py $1 $2 $3 > p_$1_cal$2_x$3.log 2>&1; echo "rc=$?" >> p_$1_cal$2_x$3.log; }
export -f one; export PY
for s in 1 2 3 4 5 6; do echo "base 3 $s"; echo "lstmtuned 3 $s"; echo "lstmtuned 10 $s"; done | xargs -P 3 -I{} bash -c "one \"{}\""
echo PROBE_DONE
