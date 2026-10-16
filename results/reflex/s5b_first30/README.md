# S5b first-setting online networks extended to 30 seeds

The four online networks with the first setting (Adam learning rate 1e-3,
LSTM window 8: `ae16`, `ae46`, `lstm16` forecasting, `lstmae16` reconstructing) had only seeds 1-6 on the long
stream. Seeds 7-30 were run with the unchanged `s19_online_nets.py s5b` (DGX Spark, 14 Oct 2026,
10 lanes, nice 19, one thread each; `jobs.txt`, `src/reflex/run_s5b_first.sh`), 12:06-12:13 UTC.

- 96 of 96 runs finished, no errors. Seed 1 of all four networks was rerun as a check and matched the committed
  files value for value (blocks and rare-phase fields).
- Result files merged into `../s19_online_nets/s5b/` (now 120 files, 30 seeds x 4 networks).
- `analyze_s19.py` rerun: the S5b section of `../s19_online_nets/summary.md` now has 30 seeds and paired
  Wilcoxon tests on the drift-corrected ratio and the last-block AUROC, Holm over 28 comparisons.
