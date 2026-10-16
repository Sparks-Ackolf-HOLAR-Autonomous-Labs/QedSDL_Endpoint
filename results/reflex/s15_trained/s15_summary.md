# S15: connectome model trained on fixed wiring (20 seeds)

## Task a, selenization stop: held-out Mo AUROC by number of training probes (mean)

| arm | n=25 | n=50 | n=100 | n=200 | n=400 | MoS2->WS2 zero-shot | fine-tuned on 25 W | 25 W from scratch |
|---|---|---|---|---|---|---|---|---|
| fly | 0.946 | 0.969 | 0.980 | 0.986 | 0.988 | 0.765 | 0.886 | 0.884 |
| fly_fixmap | 0.943 | 0.973 | 0.986 | 0.992 | 0.994 | 0.732 | 0.915 | 0.896 |
| shuf | 0.943 | 0.965 | 0.981 | 0.986 | 0.989 | 0.694 | 0.889 | 0.890 |
| rand | 0.942 | 0.966 | 0.981 | 0.986 | 0.990 | 0.705 | 0.894 | 0.900 |
| dense | 0.884 | 0.921 | 0.972 | 0.955 | 0.966 | 0.738 | 0.860 | 0.819 |
| logreg | 0.966 | 0.977 | 0.990 | 0.992 | 0.994 | 0.742 | 0.941 | 0.914 |
| knn | 0.839 | 0.914 | 0.958 | 0.970 | 0.978 | 0.684 | 0.846 | 0.794 |

## Training speed: held-out AUROC after k optimiser steps, n = 100 (mean)

| arm | 25 steps | 50 steps | 100 steps | 200 steps | 400 steps |
|---|---|---|---|---|---|
| fly | 0.988 | 0.986 | 0.982 | 0.981 | 0.980 |
| fly_fixmap | 0.987 | 0.988 | 0.988 | 0.987 | 0.986 |
| shuf | 0.989 | 0.985 | 0.981 | 0.979 | 0.981 |
| rand | 0.988 | 0.985 | 0.982 | 0.981 | 0.981 |
| dense | 0.969 | 0.970 | 0.965 | 0.904 | 0.972 |

## Paired tests, task a (Wilcoxon signed-rank, Holm over the table)

| comparison | mean difference | seeds a > b | p (Holm) |
|---|---|---|---|
| n=25: fly vs shuf | +0.0029 | 11/20 | 1 |
| n=25: fly vs rand | +0.0047 | 11/20 | 1 |
| n=25: fly vs dense | +0.0623 | 18/20 | 0.0042 |
| n=25: fly vs fly_fixmap | +0.0037 | 10/20 | 1 |
| n=25: fly vs logreg | -0.0193 | 5/20 | 0.37 |
| n=25: fly vs knn | +0.1074 | 20/20 | 5.5e-05 |
| n=50: fly vs shuf | +0.0041 | 15/20 | 0.64 |
| n=50: fly vs rand | +0.0028 | 12/20 | 1 |
| n=50: fly vs dense | +0.0486 | 17/20 | 0.0016 |
| n=50: fly vs fly_fixmap | -0.0034 | 7/20 | 1 |
| n=50: fly vs logreg | -0.0080 | 4/20 | 0.14 |
| n=50: fly vs knn | +0.0551 | 20/20 | 5.5e-05 |
| n=100: fly vs shuf | -0.0005 | 10/20 | 1 |
| n=100: fly vs rand | -0.0004 | 12/20 | 1 |
| n=100: fly vs dense | +0.0079 | 13/20 | 1 |
| n=100: fly vs fly_fixmap | -0.0054 | 5/20 | 0.15 |
| n=100: fly vs logreg | -0.0093 | 4/20 | 0.02 |
| n=100: fly vs knn | +0.0221 | 19/20 | 0.00015 |
| w_zero_shot: fly vs shuf | +0.0716 | 14/20 | 0.5 |
| w_zero_shot: fly vs rand | +0.0598 | 14/20 | 0.21 |
| w_zero_shot: fly vs dense | +0.0276 | 13/20 | 1 |
| w_zero_shot: fly vs logreg | +0.0235 | 12/20 | 1 |
| w_fine_tuned: fly vs shuf | -0.0029 | 10/20 | 1 |
| w_fine_tuned: fly vs rand | -0.0074 | 9/20 | 1 |
| w_fine_tuned: fly vs dense | +0.0262 | 12/20 | 1 |
| w_fine_tuned: fly vs logreg | -0.0551 | 4/20 | 0.023 |
| fly: fine-tuned vs from scratch | +0.0028 | 11/20 | 1 |
| shuf: fine-tuned vs from scratch | -0.0004 | 9/20 | 1 |
| fly: step 25 vs shuf step 25 | -0.0007 | 5/20 | 0.39 |

## Task b, Ar+ dose stop, 1 calibration flake(s): success (within 15 %), lost (overshoot), mean absolute error

| arm | success | lost | error |
|---|---|---|---|
| ratio | 0.297 | 0.052 | 23.9 % |
| ridge | 0.855 | 0.052 | 8.5 % |
| knn | 0.818 | 0.025 | 9.2 % |
| fly | 0.967 | 0.023 | 4.9 % |
| fly_fixmap | 0.977 | 0.018 | 5.0 % |
| shuf | 0.965 | 0.017 | 4.9 % |
| rand | 0.965 | 0.025 | 5.3 % |
| dense | 0.947 | 0.015 | 5.9 % |

Paired tests (positive = fly has the smaller error):

| comparison | mean difference | seeds a > b | p (Holm) |
|---|---|---|---|
| error: shuf vs fly | -0.0002 | 8/20 | 1 |
| error: rand vs fly | +0.0037 | 12/20 | 1 |
| error: dense vs fly | +0.0102 | 11/20 | 1 |
| error: fly_fixmap vs fly | +0.0010 | 9/20 | 1 |
| error: ridge vs fly | +0.0365 | 17/20 | 0.0029 |
| error: knn vs fly | +0.0436 | 19/20 | 2.3e-05 |
| error: ratio vs fly | +0.1901 | 20/20 | 1.3e-05 |

## Task b, Ar+ dose stop, 3 calibration flake(s): success (within 15 %), lost (overshoot), mean absolute error

| arm | success | lost | error |
|---|---|---|---|
| ratio | 0.275 | 0.067 | 23.3 % |
| ridge | 0.937 | 0.035 | 6.0 % |
| knn | 0.923 | 0.025 | 6.3 % |
| fly | 0.992 | 0.008 | 4.0 % |
| fly_fixmap | 0.988 | 0.008 | 4.5 % |
| shuf | 0.988 | 0.012 | 4.3 % |
| rand | 0.988 | 0.008 | 4.2 % |
| dense | 0.990 | 0.008 | 4.2 % |

Paired tests (positive = fly has the smaller error):

| comparison | mean difference | seeds a > b | p (Holm) |
|---|---|---|---|
| error: shuf vs fly | +0.0029 | 14/20 | 0.61 |
| error: rand vs fly | +0.0018 | 14/20 | 0.7 |
| error: dense vs fly | +0.0013 | 12/20 | 0.7 |
| error: fly_fixmap vs fly | +0.0045 | 15/20 | 0.12 |
| error: ridge vs fly | +0.0200 | 19/20 | 0.00024 |
| error: knn vs fly | +0.0229 | 19/20 | 3.4e-05 |
| error: ratio vs fly | +0.1929 | 20/20 | 1.3e-05 |
