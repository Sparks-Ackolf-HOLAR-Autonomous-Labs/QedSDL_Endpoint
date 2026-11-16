# S26: martingale posterior networks and an early-exit network

## S26a: recipe planner on the Janus map (20 seeds, 30 experiments, F* = 0.907)

| Arm | gap after 10 | after 20 | after 30 | seeds within 0.02 | mean gap over 30 (area) | proposal ms | update ms |
|---|---|---|---|---|---|---|---|
| FLoRES | 0.096 [0.059, 0.202] | 0.044 [0.011, 0.087] | 0.006 [0.003, 0.084] | 0.60 | 0.138 | 4.6 | 12.6 |
| LRKF | 0.065 [0.038, 0.113] | 0.048 [0.009, 0.067] | 0.041 [0.005, 0.066] | 0.35 | 0.112 | 5.8 | 13.3 |
| FLoRES@0.01 | 0.112 [0.059, 0.162] | 0.068 [0.022, 0.087] | 0.006 [0.004, 0.069] | 0.65 | 0.148 | 4.2 | 13.7 |
| LRKF@0.01 | 0.062 [0.038, 0.250] | 0.041 [0.015, 0.083] | 0.033 [0.006, 0.056] | 0.35 | 0.105 | 5.7 | 14.5 |
| gp_rerun | 0.037 [0.011, 0.118] | 0.004 [0.003, 0.033] | 0.003 [0.003, 0.005] | 0.85 | 0.079 | 24.3 | nan |
| fly_rerun | 0.090 [0.051, 0.192] | 0.060 [0.034, 0.092] | 0.047 [0.016, 0.071] | 0.30 | 0.103 | nan | nan |
| fly_shuf_rerun | 0.082 [0.040, 0.180] | 0.057 [0.034, 0.100] | 0.018 [0.009, 0.053] | 0.55 | 0.116 | nan | nan |
| random_rerun | 0.066 [0.034, 0.128] | 0.040 [0.010, 0.074] | 0.014 [0.006, 0.046] | 0.60 | 0.094 | nan | nan |

Paired Wilcoxon on the mean gap over the 30 experiments (lower is better), Holm over these comparisons:

| A | B | median A | median B | seeds A better | p | p (Holm) |
|---|---|---|---|---|---|---|
| FLoRES | gp_rerun | 0.138 | 0.079 | 3 | 0.0037 | 0.033 |
| LRKF | gp_rerun | 0.112 | 0.079 | 3 | 0.0073 | 0.058 |
| FLoRES | fly_rerun | 0.138 | 0.103 | 14 | 0.2 | 0.99 |
| LRKF | fly_rerun | 0.112 | 0.103 | 10 | 0.36 | 1 |
| FLoRES | random_rerun | 0.138 | 0.094 | 11 | 0.7 | 1 |
| LRKF | random_rerun | 0.112 | 0.094 | 8 | 0.94 | 1 |
| FLoRES | LRKF | 0.138 | 0.112 | 8 | 0.24 | 0.99 |
| FLoRES@0.01 | gp_rerun | 0.148 | 0.079 | 5 | 0.04 | 0.24 |
| LRKF@0.01 | gp_rerun | 0.105 | 0.079 | 4 | 0.0094 | 0.066 |

Reproduction check: the re-run arms against results/janus/summary.md (gap after 30, median): gp 0.003, fly 0.047, fly_shuf 0.018, random 0.014.

## S26b: furnace stop with the planner's close delayed by D (20 seeds, 30 flakes each)

| Arm | flakes lost (mean) | under | over |
|---|---|---|---|
| local_ridge_bands | 0.10 | 0.10 | 0.00 |
| planner_D0.02 | 0.10 | 0.10 | 0.00 |
| planner_D0.2 | 0.10 | 0.10 | 0.00 |
| planner_D2 | 0.10 | 0.10 | 0.00 |
| planner_D10 | 0.25 | 0.05 | 0.20 |
| planner_D60 | 11.00 | 0.05 | 10.95 |

Paired Wilcoxon against the local stop (Holm): planner_D0.02: p = 1 (Holm 1); planner_D0.2: p = 1 (Holm 1); planner_D2: p = 1 (Holm 1); planner_D10: p = 0.18 (Holm 0.72); planner_D60: p = 8.7e-05 (Holm 0.00043)

## S26c: early-exit network (BranchyNet) on the failing-run stop (20 seeds; S15 task-a splits)

Held-out AUROC, median [IQR]; exit-1 share at T = 0.2 bits (fixed in advance).

| Training probes | BranchyNet, T = 0.2 | share leaving at exit 1 | BranchyNet, final exit only | two-layer perceptron (re-run / S15) | trained connectome model (S15) | logistic regression (S15) |
|---|---|---|---|---|---|---|
| 25 | 0.972 [0.952, 0.984] | 0.82 | 0.973 [0.956, 0.984] | 0.976 / 0.976 | 0.957 [0.935, 0.975] | 0.974 [0.964, 0.980] |
| 100 | 0.988 [0.984, 0.992] | 0.80 | 0.988 [0.985, 0.992] | 0.989 / 0.988 | 0.984 [0.975, 0.989] | 0.991 [0.988, 0.992] |
| 400 | 0.994 [0.993, 0.995] | 0.79 | 0.995 [0.994, 0.996] | 0.995 / 0.995 | 0.988 [0.985, 0.992] | 0.994 [0.993, 0.995] |

Paired Wilcoxon, BranchyNet at T = 0.2 against each (Holm over these 9):

| probes | against | median difference | seeds BranchyNet higher | p | p (Holm) |
|---|---|---|---|---|---|
| 25 | trained connectome model | +0.010 | 13 | 0.021 | 0.092 |
| 25 | logistic regression | +0.004 | 12 | 0.78 | 0.78 |
| 25 | BranchyNet final exit | -0.000 | 8 | 0.083 | 0.25 |
| 100 | trained connectome model | +0.007 | 14 | 0.0032 | 0.022 |
| 100 | logistic regression | -0.002 | 6 | 0.017 | 0.092 |
| 100 | BranchyNet final exit | -0.000 | 4 | 0.015 | 0.092 |
| 400 | trained connectome model | +0.004 | 19 | 2.7e-05 | 0.00021 |
| 400 | logistic regression | +0.000 | 13 | 0.28 | 0.55 |
| 400 | BranchyNet final exit | -0.001 | 1 | 5.7e-06 | 5.1e-05 |

Threshold sweep at 100 probes (AUROC / exit-1 share): T=0.05: 0.987 / 0.73; T=0.1: 0.988 / 0.77; T=0.2: 0.988 / 0.80; T=0.3: 0.988 / 0.81; T=0.5: 0.988 / 0.84; T=0.7: 0.989 / 0.86

Time per probe, NumPy, one Grace core, batch 1: early path 4.7 us, full path 5.6 us.

