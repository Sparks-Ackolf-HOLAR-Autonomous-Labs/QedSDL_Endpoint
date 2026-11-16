# PREREG: learning-rate pick for the S21 extra networks, S20 1x reruns (G1)

Frozen by the analyst at **2026-10-22T22:10:25-06:00**, on g1-fly's instruction, **before the sweep is registered with cm or launched**. The pick is made once, by the script below, and never revised.

## Rule (as `PREREG_lstm_pick.md`, applied to each network separately)
- Networks: net_gru (GRU), net_peep (peephole LSTM), net_mog (Mogrifier LSTM), net_tfm (transformer), net_memnet (memory-augmented regressor).
- For each network and each learning rate, sum the network's flakes lost over the five tuning seeds (cal10).
- Pick, per network, the learning rate with the **fewest flakes lost in total**.
- Break ties with the **lower mean held-out x_t error** (final_xt_mae of that network) over the same seeds.
- Published `analyze_s21.py` used mean flakes lost only (tie -> 3e-3, listed first). On the published sweep (4x) the two rules agree, with no ties: GRU 1e-2, peephole 3e-3, Mogrifier 3e-3, transformer 1e-2, memory-augmented 3e-3 (`probe/pub_s21_pick/s21_pick.json`, sha256 c44541e3f81da6f8d242d710b396f195ea349c81e857c226cb3c62a8c165ab46).

## Sweep (as P4 `results/reflex/s21_memory_nets/run_s20.sh`, but at SHIFT_SCALE 1)
- Seeds: **101, 102, 103, 104, 105** (separate from the test seeds 1-20).
- N_CAL 10; S20_ARMS=net_gru,net_peep,net_mog,net_tfm,net_memnet (all five in one run); 600 epochs, 16 units.
- Grid: lr **3e-3** and **1e-2** (S20_LSTM=$lr,600,16, S20_TAG=_s21_lr$lr). 4 workers.
- Outputs: `shift1x/s21/s20_tune/s20_cal10_s{seed}_s21_lr{lr}.json` (10 files). An incomplete sweep is refused; no pick is written.
- Test runs (seeds 1-20, cal10 and cal3) are run at **both** learning rates, as published, so they do not depend on the pick; the analysis uses the pick.

## Frozen code (both now read-only)
| File | sha256 | mtime |
|---|---|---|
| `shift1x/analysis/pick_s21.py` | 13e68604c3c2575078bc3601280c9c8b257d8c503fac77e1bab51f41f2524cac | 2026-10-22T22:02:33 |
| `shift1x/run_s21.sh` (launcher; its s21_tune block calls pick_s21) | ffe5cdc77384901e53491f1ef1053ba2e0a0edb6c74e7e6672f8a11a7f5c41bf | 2026-10-22T22:02:33 |
| `shift1x/pin_shift.py` (sets SHIFT_SCALE) | 82f86a3766b0e907c478f5ec63bd61a8a94e9cd7395751b2ab4c1045fbc48dfa | 2026-10-22T21:15:40 |

## Output
- `shift1x/s21_pick.txt`: one line per network, `net lr`.
- `shift1x/s21_pick.json`: the pick, the rule and the full ranking.
- Log: `shift1x/logs/s21_pick.log`, ending `PICK ...` and `rc=0`.

## Timing
- At freeze time **0 of 10 sweep outputs existed** (the 1x S21 sweep had not been registered or launched) and no pick had been made.
- Smoke tests (`shift1x/smoke5/`, seed 101 / seed 1 only) ran before the freeze; they are not part of the sweep.
