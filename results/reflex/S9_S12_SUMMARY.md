## Janus MoSSe selenization line (S1 stream), 20 seeds

| Detector | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |
|---|---|---|---|---|---|---|
| fly_arb | 0.951 | 0.943 to 0.956 | 0.220 |  | 1.80 | 5177 |
| fly_var | 0.953 | 0.948 to 0.957 | 0.203 |  | 1.78 | 5177 |
| fly_coact | 0.940 | 0.931 to 0.943 | 0.224 |  | 1.75 | 5177 |
| fly_shuf_arb | 0.950 | 0.945 to 0.958 | 0.229 |  | 1.80 | 5177 |
| fly_shuf_var | 0.955 | 0.947 to 0.960 | 0.211 |  | 1.77 | 5177 |
| fly_shuf_coact | 0.939 | 0.933 to 0.945 | 0.236 |  | 1.75 | 5177 |
| fly_rand_arb | 0.939 | 0.935 to 0.946 | 0.186 |  | 1.81 | 5177 |
| fly_rand_var | 0.946 | 0.940 to 0.950 | 0.179 |  | 1.77 | 5177 |
| fly_rand_coact | 0.941 | 0.932 to 0.950 | 0.173 |  | 1.75 | 5177 |
| fly_2comp_min | 0.951 | 0.943 to 0.956 | 0.220 |  | 1.83 | 10354 |
| fly_2comp | 0.922 | 0.912 to 0.932 | 0.190 |  | 1.78 | 5178 |
| fly_2comp_shuf | 0.918 | 0.910 to 0.930 | 0.180 |  | 1.80 | 5178 |
| zscore | 0.977 | 0.973 to 0.980 | 0.112 |  | 0.03 | 112 |
| zscore_2t | 0.975 | 0.972 to 0.979 | 0.081 |  | 0.01 | 113 |
| knn_budget | 0.920 | 0.887 to 0.947 | 0.034 |  | 0.07 | 5152 |
| fly_left | 0.955 | 0.952 to 0.958 | 0.225 |  | 0.92 | 2580 |
| fly_right | 0.950 | 0.945 to 0.953 | 0.211 |  | 0.92 | 2597 |
| fly_lr_vote | 0.954 | 0.950 to 0.957 | 0.231 |  | 1.82 | 5177 |

Co-activation match (sum over band pairs of warm-up correlation x Kenyon-cell co-sampling of their glomeruli), mean:

| Wiring | name order | variance-matched | co-activation-matched |
|---|---|---|---|
| fly | 2.0 | 1.8 | 21.4 |
| fly_shuf | 0.3 | 0.3 | 7.0 |
| fly_rand | 4.4 | 4.1 | 13.4 |

Paired Wilcoxon tests (AUROC; false alarms after laser drop):

| A vs B | mean AUROC A - B | p | mean false-alarm A - B | p |
|---|---|---|---|---|
| fly_arb vs fly_shuf_arb | +0.0001 | 0.76 | -0.010 | 0.19 |
| fly_var vs fly_shuf_var | -0.0020 | 0.26 | -0.008 | 0.16 |
| fly_coact vs fly_shuf_coact | +0.0006 | 0.96 | -0.011 | 0.23 |
| fly_coact vs fly_rand_coact | -0.0014 | 0.67 | +0.052 | 0.00021 |
| fly_var vs fly_arb | +0.0021 | 0.33 | -0.017 | 0.16 |
| fly_coact vs fly_arb | -0.0109 | 6.3e-05 | +0.005 | 0.66 |
| fly_2comp vs fly_arb | -0.0284 | 1.9e-06 | -0.030 | 0.04 |
| fly_2comp vs zscore_2t | -0.0528 | 1.9e-06 | +0.109 | 3.8e-06 |
| zscore_2t vs zscore | -0.0018 | 0.00013 | -0.031 | 0.00045 |
| fly_left vs fly_right | +0.0052 | 5.7e-06 | +0.014 | 0.04 |
| fly_lr_vote vs fly_arb | +0.0030 | 0.01 | +0.011 | 0.10 |
| fly_left vs fly_arb | +0.0043 | 0.0023 | +0.005 | 0.60 |
| knn_budget vs fly_arb | -0.0301 | 0.0017 | -0.186 | 8.8e-05 |
| zscore vs fly_arb | +0.0262 | 1.9e-06 | -0.108 | 1.9e-06 |

## shared facility, four material families (S1b stream), 20 seeds

| Detector | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |
|---|---|---|---|---|---|---|
| fly_arb | 0.860 | 0.848 to 0.876 | 0.190 | 0.63 | 1.76 | 5177 |
| fly_var | 0.898 | 0.888 to 0.908 | 0.166 | 0.79 | 1.74 | 5177 |
| fly_coact | 0.851 | 0.838 to 0.862 | 0.166 | 0.62 | 1.71 | 5177 |
| fly_shuf_arb | 0.867 | 0.856 to 0.877 | 0.182 | 0.65 | 1.76 | 5177 |
| fly_shuf_var | 0.892 | 0.880 to 0.904 | 0.183 | 0.76 | 1.73 | 5177 |
| fly_shuf_coact | 0.854 | 0.851 to 0.873 | 0.183 | 0.64 | 1.70 | 5177 |
| fly_rand_arb | 0.841 | 0.825 to 0.849 | 0.109 | 0.60 | 1.76 | 5177 |
| fly_rand_var | 0.867 | 0.853 to 0.878 | 0.115 | 0.70 | 1.73 | 5177 |
| fly_rand_coact | 0.845 | 0.831 to 0.862 | 0.113 | 0.62 | 1.70 | 5177 |
| fly_2comp_min | 0.860 | 0.848 to 0.876 | 0.190 | 0.63 | 1.79 | 10354 |
| fly_2comp | 0.834 | 0.820 to 0.845 | 0.170 | 0.56 | 1.73 | 5178 |
| fly_2comp_shuf | 0.833 | 0.812 to 0.853 | 0.172 | 0.54 | 1.76 | 5178 |
| zscore | 0.847 | 0.838 to 0.862 | 0.040 | 0.51 | 0.03 | 112 |
| zscore_2t | 0.846 | 0.836 to 0.861 | 0.037 | 0.50 | 0.01 | 113 |
| knn_budget | 0.919 | 0.905 to 0.937 | 0.031 | 0.85 | 0.07 | 5152 |
| fly_left | 0.877 | 0.866 to 0.890 | 0.168 | 0.70 | 0.90 | 2580 |
| fly_right | 0.843 | 0.833 to 0.855 | 0.168 | 0.57 | 0.90 | 2597 |
| fly_lr_vote | 0.870 | 0.861 to 0.882 | 0.188 | 0.67 | 1.78 | 5177 |

Co-activation match (sum over band pairs of warm-up correlation x Kenyon-cell co-sampling of their glomeruli), mean:

| Wiring | name order | variance-matched | co-activation-matched |
|---|---|---|---|
| fly | 2.3 | 2.0 | 19.8 |
| fly_shuf | -0.0 | 0.1 | 6.4 |
| fly_rand | 1.7 | 1.8 | 10.6 |

Paired Wilcoxon tests (AUROC; false alarms after laser drop):

| A vs B | mean AUROC A - B | p | mean false-alarm A - B | p |
|---|---|---|---|---|
| fly_arb vs fly_shuf_arb | -0.0070 | 0.08 | +0.008 | 0.31 |
| fly_var vs fly_shuf_var | +0.0062 | 0.07 | -0.016 | 0.10 |
| fly_coact vs fly_shuf_coact | -0.0036 | 0.43 | -0.017 | 0.05 |
| fly_coact vs fly_rand_coact | +0.0061 | 0.20 | +0.053 | 0.00025 |
| fly_var vs fly_arb | +0.0385 | 1.9e-06 | -0.024 | 0.02 |
| fly_coact vs fly_arb | -0.0090 | 0.16 | -0.024 | 0.04 |
| fly_2comp vs fly_arb | -0.0258 | 1.3e-05 | -0.020 | 0.02 |
| fly_2comp vs zscore_2t | -0.0118 | 0.02 | +0.132 | 1.9e-06 |
| zscore_2t vs zscore | -0.0014 | 0.0094 | -0.003 | 0.20 |
| fly_left vs fly_right | +0.0339 | 1.9e-06 | -0.000 | 1.00 |
| fly_lr_vote vs fly_arb | +0.0105 | 1.9e-06 | -0.002 | 0.91 |
| fly_left vs fly_arb | +0.0169 | 1.9e-06 | -0.022 | 0.02 |
| knn_budget vs fly_arb | +0.0590 | 1.9e-06 | -0.159 | 1.9e-06 |
| zscore vs fly_arb | -0.0125 | 0.0017 | -0.150 | 8.8e-05 |

## S11 stop Ar+ irradiation at a target vacancy density, 20 seeds x 30 flakes

| Controller | on target (+-15 %) | overshoot (flake lost) | stopped short | mean abs. error | p (on target vs fly) |
|---|---|---|---|---|---|
| dose | 0.50 | 0.30 | 0.21 | 0.19 | 0.54 |
| ratio | 0.29 | 0.09 | 0.62 | 0.24 | 0.0048 |
| ratio_trend | 0.39 | 0.16 | 0.44 | 0.20 | 0.16 |
| ridge | 0.88 | 0.03 | 0.09 | 0.08 | 8.6e-05 |
| knn | 0.93 | 0.03 | 0.05 | 0.06 | 8.7e-05 |
| centroid | 0.36 | 0.07 | 0.57 | 0.20 | 0.0048 |
| fly | 0.47 | 0.38 | 0.16 | 0.22 |  |
| fly_shuf | 0.46 | 0.30 | 0.24 | 0.19 | 0.98 |
| fly_rand | 0.44 | 0.22 | 0.34 | 0.19 | 0.17 |
| fly_near | 0.03 | 0.00 | 0.97 | 0.26 | 8.6e-05 |
| fly_lookup | 0.95 | 0.04 | 0.01 | 0.05 | 8.6e-05 |
| fly_shuf_lookup | 0.96 | 0.04 | 0.01 | 0.06 | 8.6e-05 |
| fly_rand_lookup | 0.93 | 0.04 | 0.03 | 0.06 | 8.5e-05 |
| fly_trend | 0.49 | 0.46 | 0.05 | 0.26 | 0.11 |

Model fit to the digitised Fig. 3(a) spectra (relative rms per trace): 0.0: 0.020, 1.0: 0.031, 5.0: 0.079, 10.0: 0.125
