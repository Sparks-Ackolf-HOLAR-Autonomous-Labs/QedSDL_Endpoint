# S20: stop from the time course (slow purge), 20 seeds x 30 flakes

## 3 calibration flakes (20 seeds)

| arm | lost of 30 | under-converted | over-selenized | lost among fast flakes | final x_t error (held out) |
|---|---|---|---|---|---|
| now_ridge | 4.60 | 0.00 | 4.60 | 73 of 89 |  |
| snap_ridge | 3.15 | 2.20 | 0.95 | 19 of 89 | 0.056 |
| snap_mlp | 3.00 | 2.70 | 0.30 | 6 of 89 | 0.049 |
| snap_kc | 2.45 | 2.20 | 0.25 | 5 of 89 | 0.033 |
| seq_ridge | 7.25 | 6.45 | 0.80 | 18 of 89 | 0.055 |
| lstm | 3.95 | 3.35 | 0.60 | 13 of 89 | 0.046 |
| lstm_tuned | 4.15 | 3.65 | 0.50 | 10 of 89 | 0.043 |
| seq_kc | 2.35 | 2.25 | 0.10 | 2 of 89 | 0.027 |
| trend_rule | 12.45 | 12.45 | 0.00 | 29 of 89 |  |
| oracle_k | 0.00 | 0.00 | 0.00 | 0 of 89 |  |

Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:

| arm | reference | difference | seeds better / worse | p (Holm) |
|---|---|---|---|---|
| lstm_tuned | snap_mlp | +1.15 | 5 / 13 | 0.089 |
| lstm_tuned | snap_ridge | +1.00 | 1 / 11 | 0.0664 |
| lstm_tuned | seq_ridge | -3.10 | 16 / 2 | 0.0034 |
| lstm_tuned | now_ridge | -0.45 | 12 / 7 | 1 |
| lstm_tuned | lstm | +0.20 | 7 / 9 | 1 |
| seq_kc | lstm_tuned | -1.80 | 14 / 2 | 0.0101 |
| seq_kc | snap_kc | -0.10 | 5 / 7 | 1 |
| seq_kc | seq_ridge | -4.90 | 19 / 1 | 0.00103 |
| snap_kc | snap_mlp | -0.55 | 10 / 3 | 0.209 |

## 10 calibration flakes (20 seeds)

| arm | lost of 30 | under-converted | over-selenized | lost among fast flakes | final x_t error (held out) |
|---|---|---|---|---|---|
| now_ridge | 3.35 | 0.00 | 3.35 | 55 of 88 |  |
| snap_ridge | 1.90 | 1.50 | 0.40 | 8 of 88 | 0.055 |
| snap_mlp | 2.10 | 1.80 | 0.30 | 6 of 88 | 0.038 |
| snap_kc | 1.90 | 1.65 | 0.25 | 5 of 88 | 0.034 |
| seq_ridge | 4.65 | 3.95 | 0.70 | 15 of 88 | 0.045 |
| lstm | 3.80 | 3.50 | 0.30 | 6 of 88 | 0.039 |
| lstm_tuned | 2.90 | 2.65 | 0.25 | 5 of 88 | 0.038 |
| seq_kc | 1.00 | 1.00 | 0.00 | 2 of 88 | 0.025 |
| trend_rule | 10.90 | 10.90 | 0.00 | 27 of 88 |  |
| oracle_k | 0.00 | 0.00 | 0.00 | 0 of 88 |  |

Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:

| arm | reference | difference | seeds better / worse | p (Holm) |
|---|---|---|---|---|
| lstm_tuned | snap_mlp | +0.80 | 3 / 9 | 0.0926 |
| lstm_tuned | snap_ridge | +1.00 | 3 / 12 | 0.0329 |
| lstm_tuned | seq_ridge | -1.75 | 16 / 2 | 0.0109 |
| lstm_tuned | now_ridge | -0.45 | 11 / 7 | 0.867 |
| lstm_tuned | lstm | -0.90 | 13 / 4 | 0.0964 |
| seq_kc | lstm_tuned | -1.90 | 16 / 0 | 0.00307 |
| seq_kc | snap_kc | -0.90 | 11 / 2 | 0.0437 |
| seq_kc | seq_ridge | -3.65 | 19 / 0 | 0.00108 |
| snap_kc | snap_mlp | -0.20 | 6 / 4 | 0.867 |
