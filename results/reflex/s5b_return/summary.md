## S5b: recovery-rate sweep with returning rare phases (30-30 seeds)

| Detector | seeds | AUROC first / last block | false alarms last block | alpha-MoO3 recall first / return, ratio | GaSe recall first / return, ratio | SnS2 recall first / return, ratio | ms per reading |
|---|---|---|---|---|---|---|---|
| fly_r0 | 30 | 0.875 / 0.837 | 0.000 | 0.80 / 0.24, 0.13 (0.23) | 0.68 / 0.19, 0.15 (2.52) | 0.79 / 0.17, 0.08 (22.93) | 3.10 |
| fly_r1e-5 | 30 | 0.890 / 0.871 | 0.002 | 0.80 / 0.32, 0.18 (0.18) | 0.71 / 0.52, 0.37 (0.38) | 0.84 / 0.69, 0.43 (0.39) | 3.42 |
| fly_r3e-5 | 30 | 0.896 / 0.872 | 0.016 | 0.79 / 0.43, 0.24 (0.24) | 0.74 / 0.68, 0.61 (0.61) | 0.87 / 0.79, 0.71 (0.64) | 3.92 |
| fly_r1e-4 | 30 | 0.901 / 0.872 | 0.044 | 0.77 / 0.47, 0.36 (0.37) | 0.74 / 0.73, 0.88 (0.88) | 0.86 / 0.81, 0.84 (0.76) | 3.28 |
| fly_r3e-4 | 30 | 0.900 / 0.871 | 0.059 | 0.66 / 0.45, 0.53 (0.54) | 0.68 / 0.68, 0.99 (0.99) | 0.80 / 0.75, 0.84 (0.78) | 3.82 |
| fly_r1e-3 | 30 | 0.894 / 0.866 | 0.059 | 0.43 / 0.38, 0.80 (0.80) | 0.48 / 0.48, 0.99 (0.99) | 0.67 / 0.59, 0.87 (0.82) | 3.78 |
| knn_budget | 30 | 0.913 / 0.908 | 0.034 | 0.89 / 0.89, 0.98 (0.98) | 0.82 / 0.80, 0.97 (0.98) | 0.92 / 0.83, 0.87 (0.77) | 0.03 |
| zscore | 30 | 0.850 / 0.841 | 0.037 | 0.95 / 0.91, 0.94 (0.94) | 0.96 / 0.96, 0.94 (0.94) | 1.00 / 1.00, 0.81 (0.79) | 0.01 |

Return ratio = mean score of the return episode / mean score of the first episode (below 1: remembered; about 1: treated as new). In brackets: contrast ratio, the same after dividing each episode's score by the mean routine score in the 500 readings before it (removes general drift).

Paired Wilcoxon against the equal-memory kNN (seeds present in both), Holm over 28 comparisons:

| detector | measure | difference (detector minus kNN) | seeds detector higher | p | p (Holm) |
|---|---|---|---|---|---|
| fly_r0 | last-block AUROC | -0.071 | 2 / 30 | 6.1e-08 | 1.2e-06 |
| fly_r0 | alpha-MoO3 contrast ratio | -0.747 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r0 | GaSe contrast ratio | +1.542 | 18 / 30 | 0.045 | 0.27 |
| fly_r0 | SnS2 contrast ratio | +22.160 | 21 / 30 | 0.0011 | 0.0091 |
| fly_r1e-5 | last-block AUROC | -0.036 | 5 / 30 | 0.00028 | 0.0034 |
| fly_r1e-5 | alpha-MoO3 contrast ratio | -0.794 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r1e-5 | GaSe contrast ratio | -0.604 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r1e-5 | SnS2 contrast ratio | -0.372 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r3e-5 | last-block AUROC | -0.036 | 5 / 30 | 0.00031 | 0.0034 |
| fly_r3e-5 | alpha-MoO3 contrast ratio | -0.733 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r3e-5 | GaSe contrast ratio | -0.369 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r3e-5 | SnS2 contrast ratio | -0.126 | 5 / 30 | 7.1e-05 | 0.00099 |
| fly_r1e-4 | last-block AUROC | -0.036 | 5 / 30 | 0.00034 | 0.0034 |
| fly_r1e-4 | alpha-MoO3 contrast ratio | -0.610 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r1e-4 | GaSe contrast ratio | -0.097 | 6 / 30 | 3.5e-05 | 0.00055 |
| fly_r1e-4 | SnS2 contrast ratio | -0.003 | 16 / 30 | 0.92 | 1 |
| fly_r3e-4 | last-block AUROC | -0.037 | 5 / 30 | 0.00023 | 0.003 |
| fly_r3e-4 | alpha-MoO3 contrast ratio | -0.441 | 0 / 30 | 1.9e-09 | 5.2e-08 |
| fly_r3e-4 | GaSe contrast ratio | +0.008 | 17 / 30 | 0.6 | 1 |
| fly_r3e-4 | SnS2 contrast ratio | +0.013 | 16 / 30 | 0.37 | 1 |
| fly_r1e-3 | last-block AUROC | -0.042 | 5 / 30 | 5.6e-05 | 0.00084 |
| fly_r1e-3 | alpha-MoO3 contrast ratio | -0.173 | 1 / 30 | 3.7e-09 | 7.5e-08 |
| fly_r1e-3 | GaSe contrast ratio | +0.013 | 18 / 30 | 0.42 | 1 |
| fly_r1e-3 | SnS2 contrast ratio | +0.057 | 23 / 30 | 0.003 | 0.021 |
| zscore | last-block AUROC | -0.067 | 3 / 30 | 1.6e-07 | 3e-06 |
| zscore | alpha-MoO3 contrast ratio | -0.037 | 6 / 30 | 0.00042 | 0.0038 |
| zscore | GaSe contrast ratio | -0.041 | 6 / 30 | 8e-06 | 0.00014 |
| zscore | SnS2 contrast ratio | +0.022 | 19 / 30 | 0.092 | 0.46 |
