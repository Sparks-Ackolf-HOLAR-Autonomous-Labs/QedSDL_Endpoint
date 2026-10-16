# S5b extended to 30 seeds

With 6 seeds no S5b difference could pass Holm correction over the
108 paired comparisons (the smallest two-sided Wilcoxon p with 6 pairs is 0.031, above 0.05/108 = 4.6e-4).
With 30 pairs a difference that holds in every seed reaches p = 1.9e-9, and one that holds in 24 of 30
seeds about 1e-3 before correction, so consistent differences can now pass.

Runs (DGX Spark, 14 Oct 2026, 12 lanes, nice 19, one thread each; job list `jobs.txt`, written by
`src/reflex/s5b30_jobs.sh`):
- seeds 7-30: connectome model at recovery 0, 1e-5, 3e-5, 1e-4, 3e-4 and 1e-3, equal-memory kNN, z-score
  (`s5b_return.py`); the S19b tuned online nets (`s19_online_nets.py`, same settings as `s5b_tuned`); the nine
  S21 networks with four replay designs (`s21_memory_nets.py`, same learning rates as `s21_memory_nets/main.txt`)
- seeds 1-6: model r = 3e-4, kNN and z-score again, because those files were written before the
  drift-corrected contrast was added (the code is deterministic per seed, so the other values must match)

1,194 runs, started 06:53 UTC on 14 Oct 2026, about 50 lane-hours on 12 lanes.

## Run checks

- All 1,194 runs finished (run log `done.txt`: 10:39 UTC). 11 S21 runs (seeds 16, 17, 25, 26; transformer, memory slots,
  fast weights, LSTM, peephole) had stopped with a CUDA out-of-memory error while 12 lanes shared the GPU; they were rerun
  alone with the same command and finished without error.
- Seeds 1-6 rerun of model r = 3e-4, kNN and z-score matched the earlier files value for value; only the drift-corrected
  contrast fields were added. The seed 1-6 model files at the other rates in `s5b_return/` were replaced by the copies in
  `s19_online_nets/s5b_tuned/`, which carry the contrast fields and otherwise match.
- The result files were merged into `../s5b_return/` (240), `../s19_online_nets/s5b_tuned/` (150 network files) and
  `../s21_memory_nets/s5b/` (1,080) and are not kept twice here.
- Analyses rerun: `analyze_s5b.py` (now with drift-corrected ratios and Holm over 28 tests against the kNN),
  `analyze_s19b.py` (model reference read from `s5b_return/`, all 30 seeds) and `analyze_s21.py` (Holm over 108 tests
  against the model at r = 1e-4; 95 significant).
