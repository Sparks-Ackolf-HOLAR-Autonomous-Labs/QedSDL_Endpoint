# S4L: S4 with a literature-anchored in-situ Raman probe (20 seeds, paired with S4)

GP alone loses 7.75 flakes and constrained BO 4.35 (no probe; identical to S4).

| Arm | flakes lost S4 / S4L / S4L x2 shift | false stops S4 / S4L / x2 | gap S4 / S4L / x2 | p lost S4 vs S4L |
|---|---|---|---|---|
| gp+fly | 3.55 / 3.15 / 3.35 | 4.05 / 3.45 / 2.45 | 0.122 / 0.116 / 0.048 | 0.07 |
| gp+fly_best | 3.30 / 3.15 / 3.20 | 1.80 / 2.80 / 1.80 | 0.075 / 0.072 / 0.034 | 0.41 |
| gp+fly_best_shuf | 3.15 / 3.05 / 3.35 | 2.10 / 2.75 / 2.45 | 0.068 / 0.042 / 0.036 | 0.59 |
| gp+fly_best_rand | 3.50 / 3.60 / 3.50 | 2.90 / 3.60 / 3.10 | 0.089 / 0.101 / 0.076 | 0.78 |
| gp+knn | 3.90 / 3.95 / 4.05 | 3.25 / 2.45 / 2.40 | 0.026 / 0.053 / 0.054 | 0.8 |
| gp+logreg | 3.90 / 3.65 / 3.55 | 2.50 / 2.00 / 1.65 | 0.023 / 0.032 / 0.022 | 0.35 |

S4L, paired on flakes lost, gp+fly_best vs gp+knn p = 0.0099 (-0.80); vs gp+logreg p = 0.039 (-0.50); vs gp+fly_best_shuf p = 0.41 (+0.10); vs gp+fly_best_rand p = 0.0027 (-0.45); vs gp p = 8.1e-05; vs cbo p = 0.011

S4L x2, paired on flakes lost, gp+fly_best vs gp+knn p = 0.0051 (-0.85); vs gp+logreg p = 0.16 (-0.35); vs gp+fly_best_shuf p = 0.083 (-0.15); vs gp+fly_best_rand p = 0.034 (-0.30); vs gp p = 8.2e-05; vs cbo p = 0.012

