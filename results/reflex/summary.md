## T1 fault flagging (20 seeds, mean 10.8 effective faults per 160 scored runs)

| Detector | AUROC mean | IQR | recall at 5 % false alarms | ms per reading | p (AUROC vs fly) |
|---|---|---|---|---|---|
| fly | 0.740 | 0.695 to 0.824 | 0.388 | 2.2 |  |
| fly_shuf | 0.740 | 0.684 to 0.818 | 0.351 | 2.2 | 0.956 |
| fly_rand | 0.731 | 0.681 to 0.807 | 0.369 | 2.2 | 0.294 |
| gp | 0.798 | 0.735 to 0.853 | 0.316 | 32.1 | 0.0192 |
| iforest | 0.732 | 0.691 to 0.788 | 0.167 | 34.6 | 0.729 |
| knn | 0.753 | 0.734 to 0.790 | 0.291 | 0.1 | 0.43 |

Recall at 5 % false alarms by fault kind (mean over seeds where the kind occurred):

| Detector | furnace_offset | se_empty | short_anneal | readout_glitch |
|---|---|---|---|---|
| fly | 0.22 | 0.03 | 0.13 | 0.79 |
| fly_shuf | 0.16 | 0.08 | 0.13 | 0.67 |
| fly_rand | 0.22 | 0.02 | 0.18 | 0.79 |
| gp | 0.42 | 0.07 | 0.01 | 0.52 |
| iforest | 0.25 | 0.02 | 0.11 | 0.25 |
| knn | 0.27 | 0.03 | 0.05 | 0.51 |

## T2 epochs, MoSSe then WSSe (20 seeds; RMSE of predicted fidelity over 1400 recipes)

| Model | MoSSe after MoSSe | MoSSe after WSSe | forgetting | WSSe after WSSe |
|---|---|---|---|---|
| fly | 0.212 | 0.222 | +0.010 (IQR -0.001 to +0.019) | 0.213 |
| fly_shuf | 0.214 | 0.222 | +0.008 (IQR -0.001 to +0.018) | 0.209 |
| fly_rand | 0.215 | 0.215 | +0.001 (IQR -0.007 to +0.008) | 0.209 |
| mlp_seq | 0.260 | 0.259 | -0.002 (IQR -0.033 to +0.042) | 0.232 |
| gp_ctx | 0.096 | 0.087 | -0.009 (IQR -0.014 to -0.001) | 0.079 |
| gp_pool | 0.096 | 0.108 | +0.012 (IQR -0.001 to +0.023) | 0.104 |
| gp_new | 0.096 | 0.158 | +0.062 (IQR +0.049 to +0.070) | 0.091 |

Reference: predicting the MoSSe mean everywhere gives RMSE 0.276; using the WSSe truth map for MoSSe gives 0.150. KC code overlap between the two epochs of one recipe: 0.26.

## T3 failure avoidance (20 seeds, 30 runs, 29 % of grid recipes destroy the flake)

| Arm | failed runs mean | IQR | gap mean | p fails vs gp | p gap vs gp |
|---|---|---|---|---|---|
| random | 10.20 | 9 to 12 | 0.0518 | 0.000933 | 0.0152 |
| gp | 7.75 | 7 to 8 | 0.0233 |  |  |
| cbo | 4.35 | 3 to 5 | 0.0117 | 0.000159 | 0.139 |
| fly | 11.40 | 10 to 12 | 0.0502 | 0.00018 | 0.0441 |
| fly_avoid | 11.10 | 10 to 12 | 0.0532 | 0.000292 | 0.0206 |
| gp+fly_avoid | 7.80 | 6 to 10 | 0.0173 | 1 | 0.401 |
| gp+fly_avoid_shuf | 8.00 | 7 to 8 | 0.0146 | 0.607 | 0.861 |
| gp+fly_avoid_rand | 8.75 | 7 to 11 | 0.0216 | 0.14 | 0.409 |
| fly_avoid_shuf | 11.80 | 10 to 14 | 0.0432 | 0.000121 | 0.0569 |
| fly_avoid_rand | 11.65 | 10 to 13 | 0.0444 | 0.000161 | 0.04 |

## T4 reflex layer in a three-party network (20 seeds; furnace 2 faulty from round 4; 36 runs)

| Arm | faulty readings shared (furnace offset / Se empty) | repeats | furnace taken out (seeds) | gap mean |
|---|---|---|---|---|
| none | 9.0 / 9.0 | 0.0 | 0 of 20 | 0.0155 |
| fly | 8.0 / 8.8 | 0.6 | 1 of 20 | 0.0164 |
| fly_shuf | 8.6 / 8.9 | 0.3 | 0 of 20 | 0.0157 |
| fly_rand | 8.6 / 8.9 | 0.2 | 0 of 20 | 0.0157 |
| gp_resid | 5.2 / 7.2 | 3.5 | 9 of 20 | 0.0092 |
| oracle | 0.0 / 0.0 | 0.0 | 20 of 20 | 0.0034 |

Paired Wilcoxon on faulty readings shared: fly vs none p = 0.00666; gp_resid vs none p = 0.000591; fly vs gp_resid p = 0.00353.

## T5 literature values (20 seeds; mean 19.9 injected errors in 101 scored values)

| Detector | AUROC mean | IQR | recall at 5 % false alarms | digit | unit | material | p (AUROC vs fly) |
|---|---|---|---|---|---|---|---|
| fly | 0.722 | 0.661 to 0.837 | 0.228 | 0.24 | 0.29 | 0.18 |  |
| fly_shuf | 0.715 | 0.647 to 0.801 | 0.237 | 0.25 | 0.29 | 0.20 | 0.114 |
| fly_rand | 0.711 | 0.639 to 0.793 | 0.243 | 0.25 | 0.29 | 0.21 | 0.0766 |
| robust_z | 0.821 | 0.802 to 0.873 | 0.597 | 0.58 | 0.89 | 0.48 | 0.00233 |
| knn | 0.828 | 0.800 to 0.883 | 0.519 | 0.60 | 0.80 | 0.33 | 0.000395 |
