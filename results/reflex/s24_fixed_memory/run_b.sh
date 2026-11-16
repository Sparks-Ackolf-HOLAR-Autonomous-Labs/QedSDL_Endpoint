#!/bin/bash
# S24b: run from connectome_model/; same HOME as run_a.sh
export HOME=$WORKSPACE/s24_home S24_WCACHE=$WORKSPACE/s24_home/s24_wiring.npz
for j in iii i iv ii; do nice -n 10 python src/reflex/s24_stack.py $j results/reflex/s24_fixed_memory/b_stack 4 >> results/reflex/s24_fixed_memory/b.log 2>&1; done
date -u > results/reflex/s24_fixed_memory/b_done.txt
