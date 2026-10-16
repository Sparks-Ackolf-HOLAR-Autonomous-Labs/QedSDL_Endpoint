# S15b: small networks on the Ar+ irradiation-dose stop (20 seeds x 30 test flakes)

Same flakes as S15 task b: largest difference in the recomputed ratio/ridge/knn errors = 0.0e+00.

## 1 calibration flake(s)

| arm | success (within 15 %) | lost (overshoot) | mean abs. error |
|---|---|---|---|
| ratio | 0.297 | 0.052 | 23.9 % |
| ridge | 0.855 | 0.052 | 8.5 % |
| knn | 0.818 | 0.025 | 9.2 % |
| fly | 0.967 | 0.023 | 4.9 % |
| shuf | 0.965 | 0.017 | 4.9 % |
| rand | 0.965 | 0.025 | 5.3 % |
| dense | 0.947 | 0.015 | 5.9 % |
| mlp32 | 0.922 | 0.073 | 6.2 % |
| mlp64x2 | 0.987 | 0.013 | 4.1 % |

Paired error tests (positive = network has the smaller error), Holm over this table:

| network | reference | error difference | seeds network better | p (Holm) |
|---|---|---|---|---|
| mlp32 | fly | -0.0134 | 5/20 | 0.109 |
| mlp32 | ridge | +0.0231 | 15/20 | 0.127 |
| mlp32 | knn | +0.0302 | 16/20 | 0.00193 |
| mlp64x2 | fly | +0.0076 | 14/20 | 0.127 |
| mlp64x2 | ridge | +0.0441 | 20/20 | 1.14e-05 |
| mlp64x2 | knn | +0.0512 | 20/20 | 1.14e-05 |

## 3 calibration flake(s)

| arm | success (within 15 %) | lost (overshoot) | mean abs. error |
|---|---|---|---|
| ratio | 0.275 | 0.067 | 23.3 % |
| ridge | 0.937 | 0.035 | 6.0 % |
| knn | 0.923 | 0.025 | 6.3 % |
| fly | 0.992 | 0.008 | 4.0 % |
| shuf | 0.988 | 0.012 | 4.3 % |
| rand | 0.988 | 0.008 | 4.2 % |
| dense | 0.990 | 0.008 | 4.2 % |
| mlp32 | 0.958 | 0.035 | 5.4 % |
| mlp64x2 | 0.988 | 0.012 | 4.4 % |

Paired error tests (positive = network has the smaller error), Holm over this table:

| network | reference | error difference | seeds network better | p (Holm) |
|---|---|---|---|---|
| mlp32 | fly | -0.0137 | 5/20 | 0.0146 |
| mlp32 | ridge | +0.0064 | 14/20 | 0.0983 |
| mlp32 | knn | +0.0092 | 14/20 | 0.0983 |
| mlp64x2 | fly | -0.0039 | 8/20 | 0.189 |
| mlp64x2 | ridge | +0.0161 | 19/20 | 0.00161 |
| mlp64x2 | knn | +0.0190 | 17/20 | 0.000492 |

# S19: online networks on the familiarity jobs

## New-phase flag, S1b shared-facility stream (20 seeds, 4,000 readings)

| detector | stored numbers | AUROC | recall 1T'_MoS2 | recall MoTe2 | recall spike | recall MoS2/WSe2_stack | false alarms after laser drop | ms per reading |
|---|---|---|---|---|---|---|---|---|
| fly | 5,177 | 0.860 | 0.59 | 0.59 | 0.93 | 0.10 | 0.190 | 1.05 |
| fly_shuf | 5,177 | 0.867 | 0.62 | 0.57 | 0.94 | 0.10 | 0.182 | 1.09 |
| knn_budget | 5,152 | 0.919 | 0.59 | 0.56 | 0.88 | 0.40 | 0.031 | 0.03 |
| zscore | 112 | 0.847 | 0.94 | 0.75 | 0.80 | 0.03 | 0.040 | 0.01 |
| ae16 | 1,864 | 0.831 | 0.29 | 0.22 | 0.56 | 0.09 | 0.030 | 0.16 |
| ae46 | 5,254 | 0.818 | 0.50 | 0.53 | 0.98 | 0.09 | 0.306 | 0.17 |
| lstm16 | 5,688 | 0.513 | 0.00 | 0.01 | 0.09 | 0.05 | 0.027 | 0.34 |
| lstmae16 | 5,688 | 0.730 | 0.01 | 0.03 | 0.15 | 0.07 | 0.025 | 0.34 |

Paired AUROC tests (network minus reference), Holm over this table:

| network | reference | AUROC difference | seeds network higher | p (Holm) |
|---|---|---|---|---|
| ae16 | fly | -0.029 | 1/20 | 1.53e-05 |
| ae16 | knn_budget | -0.088 | 0/20 | 1.53e-05 |
| ae46 | fly | -0.042 | 2/20 | 1.53e-05 |
| ae46 | knn_budget | -0.101 | 0/20 | 1.53e-05 |
| lstm16 | fly | -0.347 | 0/20 | 1.53e-05 |
| lstm16 | knn_budget | -0.406 | 0/20 | 1.53e-05 |
| lstmae16 | fly | -0.130 | 0/20 | 1.53e-05 |
| lstmae16 | knn_budget | -0.189 | 0/20 | 1.53e-05 |

## Memory of rare phases, S5b week-long stream (30 seeds, 10^5 readings)

Return ratio = mean score on return / mean score at first sighting: below 1 = remembered, about 1 = treated as new. Gaps before return: alpha-MoO3 1,000, GaSe 10,000, SnS2 60,000 readings.

| detector | stored numbers | AUROC first / last block | false alarms last block | alpha-MoO3 recall first / return, ratio | GaSe recall first / return, ratio | SnS2 recall first / return, ratio | ms per reading |
|---|---|---|---|---|---|---|---|
| fly_r0 | 5,177 | 0.875 / 0.837 | 0.000 | 0.80 / 0.24, 0.13 | 0.68 / 0.19, 0.15 | 0.79 / 0.17, 0.08 | 3.10 |
| fly_r1e-4 | 5,177 | 0.901 / 0.872 | 0.044 | 0.77 / 0.47, 0.36 | 0.74 / 0.73, 0.88 | 0.86 / 0.81, 0.84 | 3.28 |
| fly_r1e-5 | 5,177 | 0.890 / 0.871 | 0.002 | 0.80 / 0.32, 0.18 | 0.71 / 0.52, 0.37 | 0.84 / 0.68, 0.43 | 3.42 |
| knn_budget | 5,152 | 0.913 / 0.908 | 0.034 | 0.89 / 0.89, 0.98 | 0.82 / 0.80, 0.97 | 0.92 / 0.83, 0.87 | 0.03 |
| zscore | 112 | 0.850 / 0.841 | 0.037 | 0.95 / 0.91, 0.94 | 0.96 / 0.96, 0.94 | 1.00 / 1.00, 0.81 | 0.01 |
| ae16 | 1,864 | 0.840 / 0.630 | 0.000 | 0.39 / 0.32, 0.85 | 0.45 / 0.39, 0.86 | 0.63 / 0.56, 0.73 | 0.19 |
| ae46 | 5,254 | 0.725 / 0.635 | 0.005 | 0.43 / 0.13, 0.32 | 0.45 / 0.17, 0.29 | 0.49 / 0.21, 0.39 | 0.21 |
| lstm16 | 5,688 | 0.508 / 0.514 | 0.024 | 0.00 / 0.00, 0.96 | 0.01 / 0.01, 0.95 | 0.03 / 0.01, 0.70 | 0.38 |
| lstmae16 | 5,688 | 0.801 / 0.676 | 0.000 | 0.04 / 0.02, 0.83 | 0.11 / 0.08, 0.88 | 0.18 / 0.07, 0.63 | 0.37 |

Paired tests (network minus reference) on the drift-corrected ratio (negative = network remembers better) and on the last-block AUROC (negative = network detects less); 30 seeds, Holm over 28 comparisons.

| network | reference | measure | difference | seeds network lower | p (Holm) |
|---|---|---|---|---|---|
| ae16 | fly_r1e-4 | alpha-MoO3 | +0.500 | 0/30 | 5.22e-08 |
| ae16 | fly_r1e-4 | GaSe | +0.028 | 14/30 | 0.792 |
| ae16 | fly_r1e-4 | SnS2 | -0.468 | 30/30 | 5.22e-08 |
| ae16 | knn_budget | alpha-MoO3 | -0.111 | 24/30 | 0.0113 |
| ae16 | knn_budget | GaSe | -0.069 | 20/30 | 0.513 |
| ae16 | knn_budget | SnS2 | -0.470 | 30/30 | 5.22e-08 |
| ae16 | fly_r1e-4 | AUROC last block | -0.242 | 30/30 | 5.22e-08 |
| ae46 | fly_r1e-4 | alpha-MoO3 | -0.020 | 19/30 | 0.709 |
| ae46 | fly_r1e-4 | GaSe | -0.529 | 30/30 | 5.22e-08 |
| ae46 | fly_r1e-4 | SnS2 | -0.501 | 29/30 | 5.22e-08 |
| ae46 | knn_budget | alpha-MoO3 | -0.630 | 30/30 | 5.22e-08 |
| ae46 | knn_budget | GaSe | -0.626 | 30/30 | 5.22e-08 |
| ae46 | knn_budget | SnS2 | -0.504 | 30/30 | 5.22e-08 |
| ae46 | fly_r1e-4 | AUROC last block | -0.237 | 30/30 | 5.22e-08 |
| lstm16 | fly_r1e-4 | alpha-MoO3 | +0.580 | 0/30 | 5.22e-08 |
| lstm16 | fly_r1e-4 | GaSe | +0.064 | 13/30 | 0.513 |
| lstm16 | fly_r1e-4 | SnS2 | +0.166 | 8/30 | 0.00669 |
| lstm16 | knn_budget | alpha-MoO3 | -0.030 | 17/30 | 0.709 |
| lstm16 | knn_budget | GaSe | -0.033 | 19/30 | 0.709 |
| lstm16 | knn_budget | SnS2 | +0.163 | 5/30 | 0.000596 |
| lstm16 | fly_r1e-4 | AUROC last block | -0.357 | 30/30 | 5.22e-08 |
| lstmae16 | fly_r1e-4 | alpha-MoO3 | +0.528 | 0/30 | 5.22e-08 |
| lstmae16 | fly_r1e-4 | GaSe | +0.642 | 0/30 | 5.22e-08 |
| lstmae16 | fly_r1e-4 | SnS2 | -0.025 | 21/30 | 0.418 |
| lstmae16 | knn_budget | alpha-MoO3 | -0.082 | 21/30 | 0.294 |
| lstmae16 | knn_budget | GaSe | +0.545 | 1/30 | 7.26e-08 |
| lstmae16 | knn_budget | SnS2 | -0.028 | 19/30 | 0.513 |
| lstmae16 | fly_r1e-4 | AUROC last block | -0.196 | 30/30 | 5.22e-08 |

Drift-corrected memory of the networks (contrast = episode score / score of the routine readings in the 500 readings before; contrast ratio = return contrast / first contrast; below 1 = remembered):

| network | alpha-MoO3 contrast first / return, ratio | GaSe contrast first / return, ratio | SnS2 contrast first / return, ratio |
|---|---|---|---|
| ae16 | 17.71 / 15.21, 0.87 | 31.13 / 27.48, 0.91 | 53.04 / 14.29, 0.30 |
| ae46 | 7.88 / 2.70, 0.35 | 15.20 / 5.12, 0.36 | 18.47 / 3.78, 0.26 |
| lstm16 | 0.73 / 0.67, 0.95 | 0.73 / 0.67, 0.95 | 0.83 / 0.74, 0.93 |
| lstmae16 | 5.46 / 4.69, 0.89 | 11.10 / 16.14, 1.53 | 21.12 / 13.76, 0.74 |
