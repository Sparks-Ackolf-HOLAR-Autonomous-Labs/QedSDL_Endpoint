# S18a-c results

## S18a: selenization stop at the instrument (20 seeds x 30 flakes)

| Arm | flakes lost of 30 (mean) | under-converted | over-selenized | Se valve open (s) | messages up per run | x_t estimate error |
|---|---|---|---|---|---|---|
| open_loop | 8.60 | 3.65 | 4.95 | 144 | 1 |  |
| planner_D2 | 0.15 | 0.15 | 0.00 | 147 | 73 | 0.051 |
| planner_D10 | 0.45 | 0.15 | 0.30 | 155 | 73 | 0.051 |
| planner_D60 | 10.65 | 0.10 | 10.55 | 205 | 73 | 0.051 |
| local_ridge_bands | 0.15 | 0.15 | 0.00 | 145 | 1 | 0.051 |
| cord_fly | 0.20 | 0.20 | 0.00 | 145 | 1 | 0.052 |
| cord_shuf | 0.25 | 0.25 | 0.00 | 145 | 1 | 0.052 |
| cord_rand | 0.25 | 0.25 | 0.00 | 145 | 1 | 0.052 |

Paired tests on flakes lost (Holm over the 7 comparisons):

- local_ridge_bands - open_loop: -8.45 flakes, p = 8.1e-05, Holm 0.00056
- local_ridge_bands - planner_D2: +0.00 flakes, p = 1, Holm 1
- local_ridge_bands - planner_D10: -0.30 flakes, p = 0.014, Holm 0.072
- local_ridge_bands - planner_D60: -10.50 flakes, p = 8.6e-05, Holm 0.00056
- cord_fly - local_ridge_bands: +0.05 flakes, p = 0.32, Holm 1
- cord_fly - cord_shuf: -0.05 flakes, p = 0.32, Holm 1
- cord_fly - cord_rand: -0.05 flakes, p = 0.32, Holm 1

Nerve-cord features on the DGX: 0.97 ms per reading (one thread).

## S18b: status bit after the laser drop, Janus line (20 seeds)

| Detector | AUROC | false alarms, readings 100-400 after the drop | recall of new events there |
|---|---|---|---|
| fly | 0.951 | 0.092 | 0.75 |
| fly_gate | 0.951 | 0.016 | 0.47 |
| fly_ctx_gate | 0.946 | 0.000 | 0.00 |
| fly_shuf_ctx_gate | 0.945 | 0.000 | 0.01 |
| zscore | 0.977 | 0.018 | 0.77 |
| zscore_gate | 0.977 | 0.019 | 0.77 |
| zscore_reset_gate | 0.977 | 0.004 | 0.56 |
| knn | 0.920 | 0.013 | 0.45 |
| knn_gate | 0.920 | 0.030 | 0.60 |
| knn_reset_gate | 0.915 | 0.008 | 0.21 |

Paired tests on false alarm (Holm over 5):

- fly_gate - fly: -0.076, p = 1.9e-06, Holm 9.5e-06
- fly_ctx_gate - fly_gate: -0.016, p = 0.0002, Holm 0.00078
- fly_gate - zscore_reset_gate: +0.012, p = 0.0018, Holm 0.0055
- fly_gate - knn_reset_gate: +0.007, p = 0.04, Holm 0.08
- fly_ctx_gate - fly_shuf_ctx_gate: +0.000, p = 1, Holm 1

Paired tests on recall (Holm over 5):

- fly_gate - fly: -0.273, p = 0.00013, Holm 0.00039
- fly_ctx_gate - fly_gate: -0.472, p = 1.9e-06, Holm 9.5e-06
- fly_gate - zscore_reset_gate: -0.090, p = 0.022, Holm 0.043
- fly_gate - knn_reset_gate: +0.265, p = 1.9e-06, Holm 9.5e-06
- fly_ctx_gate - fly_shuf_ctx_gate: -0.012, p = 0.11, Holm 0.11

## S18b: status bit after the laser drop, four-family facility (20 seeds)

| Detector | AUROC | false alarms, readings 100-400 after the drop | recall of new events there |
|---|---|---|---|
| fly | 0.860 | 0.098 | 0.56 |
| fly_gate | 0.860 | 0.026 | 0.39 |
| fly_ctx_gate | 0.962 | 0.000 | 0.00 |
| fly_shuf_ctx_gate | 0.962 | 0.000 | 0.00 |
| zscore | 0.847 | 0.020 | 0.47 |
| zscore_gate | 0.847 | 0.035 | 0.56 |
| zscore_reset_gate | 0.843 | 0.003 | 0.24 |
| knn | 0.919 | 0.021 | 0.35 |
| knn_gate | 0.919 | 0.037 | 0.54 |
| knn_reset_gate | 0.912 | 0.008 | 0.09 |

Paired tests on false alarm (Holm over 5):

- fly_gate - fly: -0.072, p = 1.9e-06, Holm 9.5e-06
- fly_ctx_gate - fly_gate: -0.026, p = 0.00013, Holm 0.00039
- fly_gate - zscore_reset_gate: +0.023, p = 1.3e-05, Holm 5.3e-05
- fly_gate - knn_reset_gate: +0.018, p = 0.00039, Holm 0.00078
- fly_ctx_gate - fly_shuf_ctx_gate: +0.000, p = 1, Holm 1

Paired tests on recall (Holm over 5):

- fly_gate - fly: -0.169, p = 0.0002, Holm 0.00079
- fly_ctx_gate - fly_gate: -0.388, p = 8.8e-05, Holm 0.00044
- fly_gate - zscore_reset_gate: +0.150, p = 0.00063, Holm 0.0013
- fly_gate - knn_reset_gate: +0.300, p = 0.0002, Holm 0.00079
- fly_ctx_gate - fly_shuf_ctx_gate: +0.000, p = 1, Holm 1

## S18c: the local stop on the Raspberry Pi 5 (monsterpi, nice 19) and a DGX Grace core

| Machine | nerve-cord stop (ms per reading) | ridge on bands (ms) | synapse entries | list at 8-bit weights, 32-bit indices | peak memory |
|---|---|---|---|---|---|
| monsterpi (aarch64) | 2.33 | 0.0019 | 681,580 | 6.1 MB | 42 MB |
| DGX Spark (aarch64) | 1.29 | 0.0005 | 681,580 | 6.1 MB | 45 MB |
