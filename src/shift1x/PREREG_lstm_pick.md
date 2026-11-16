# PREREG: LSTM tuning pick for the S20 1x reruns (G1, D-033 step 2)

Frozen by the analyst at **2026-10-22T21:35:06-06:00**, on g1-fly's instruction. The pick is made once, by the script below, and never revised.

## Rule
- For each setting, sum the LSTM arm's flakes lost over the five tuning seeds (cal10).
- Pick the setting with the **fewest flakes lost in total**.
- Break ties with the **lower mean held-out x_t error** (final_xt_mae, LSTM arm) over the same seeds.
- This is the same metric as the s23 equal-tuning criterion (g1-fly).
- On the published sweep (5x) this rule gives 1e-2,600,16, the published pick (`probe/pub_pick/lstm_pick.json`, sha256 2d05c0b45f58f1e64577902ae65c293d3c1579e616c3593655babc8656fefc56).

## Sweep
- Seeds: **101, 102, 103, 104, 105** (separate from the evaluation seeds 1-20).
- N_CAL 10, S20_ARMS=lstm, SHIFT_SCALE 1, 4 workers.
- Grid, 7 settings (lr,epochs,hidden), as published in `lstm_tuning_seeds101-105/`:
  1. 1e-2,2000,16
  2. 1e-2,2000,32
  3. 1e-2,600,16
  4. 1e-3,600,16
  5. 3e-3,2000,16
  6. 3e-3,600,16
  7. 3e-3,600,32
- Outputs: `shift1x/s20_tune/s20_cal10_s{seed}_{setting}.json` (35 files). An incomplete sweep is refused; no pick is written.

## Frozen code
| File | sha256 | mtime |
|---|---|---|
| `shift1x/analysis/pick_lstm.py` (now read-only) | 78903ec2f16fdc63e13e3640deed092f965cdbaa12f8c7bcc3ba116959da2be2 | 2026-10-22 21:27:22 |
| `shift1x/run_shift1x.sh` (launcher; its s20_tune block calls pick_lstm) | baf6817204efa22fec89ee615fe2cd1c3907c228bf95ce99cd1ae16ae9ecbd70 | |

## Output
- `shift1x/lstm_pick.txt`: the pick, one line. Read by the s20_cal10_lt and s20_cal3_lt blocks (D-033 step 3).
- `shift1x/lstm_pick.json`: the pick, the rule and the full ranking.
- Log: `shift1x/logs/lstm_pick.log`.

## Timing
- The rule and code were written and smoke-tested before the sweep (smoke 4); pick_lstm.py is unchanged since its mtime above.
- cm launched the sweep (pid 1535197; first START 2026-10-22T21:34:24) shortly before this file was written. At freeze time **0 of 35 sweep outputs existed** and no pick had been made.
