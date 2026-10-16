## S2/S3 on the Janus MoSSe line (S1 stream) (20 seeds)

| Detector | AUROC mean | IQR | false alarms after laser drop | stored numbers | p (AUROC vs fly) |
|---|---|---|---|---|---|
| fly | 0.951 | 0.943 to 0.956 | 0.220 | 5177 |  |
| fly_gain | 0.938 | 0.933 to 0.943 | 0.225 | 5177 | 1.9e-06 |
| fly_adapt | 0.857 | 0.847 to 0.867 | 0.139 | 5177 | 1.9e-06 |
| fly_coinc | 0.835 | 0.829 to 0.845 | 0.235 | 5177 | 1.9e-06 |
| fly_gain_coinc | 0.834 | 0.828 to 0.844 | 0.236 | 5177 | 1.9e-06 |
| fly_all | 0.698 | 0.685 to 0.708 | 0.089 | 5177 | 1.9e-06 |
| fly_best | 0.938 | 0.933 to 0.943 | 0.225 | 5177 | 1.9e-06 |
| fly_best_shuf | 0.943 | 0.936 to 0.951 | 0.228 | 5177 | 3.6e-05 |
| fly_best_rand | 0.952 | 0.948 to 0.956 | 0.245 | 5177 | 0.43 |
| zscore | 0.977 | 0.973 to 0.980 | 0.112 | 112 | 1.9e-06 |
| knn_budget | 0.920 | 0.887 to 0.947 | 0.034 | 5152 | 0.0017 |

Paired Wilcoxon, false alarms after the drop: fly vs zscore p = 1.9e-06 (mean diff +0.108); fly_gain vs zscore p = 8.8e-05 (mean diff +0.113); fly_adapt vs zscore p = 0.033 (mean diff +0.027); fly_coinc vs zscore p = 1.9e-06 (mean diff +0.124); fly_gain_coinc vs zscore p = 8.8e-05 (mean diff +0.124); fly_all vs zscore p = 0.04 (mean diff -0.023); fly_best vs zscore p = 8.8e-05 (mean diff +0.113); fly_best_shuf vs zscore p = 1.9e-06 (mean diff +0.116); fly_best_rand vs zscore p = 1.9e-06 (mean diff +0.133)

## S2/S3 at the four-family facility (S1b stream) (20 seeds)

| Detector | AUROC mean | IQR | false alarms after laser drop | stack AUROC | stored numbers | p (AUROC vs fly) |
|---|---|---|---|---|---|---|
| fly | 0.860 | 0.848 to 0.876 | 0.190 | 0.63 | 5177 |  |
| fly_gain | 0.901 | 0.890 to 0.909 | 0.216 | 0.78 | 5177 | 1.9e-06 |
| fly_adapt | 0.803 | 0.790 to 0.811 | 0.099 | 0.55 | 5177 | 3.8e-06 |
| fly_coinc | 0.816 | 0.808 to 0.826 | 0.206 | 0.72 | 5177 | 3.8e-06 |
| fly_gain_coinc | 0.816 | 0.807 to 0.827 | 0.209 | 0.73 | 5177 | 3.8e-06 |
| fly_all | 0.738 | 0.730 to 0.745 | 0.140 | 0.60 | 5177 | 1.9e-06 |
| fly_best | 0.901 | 0.890 to 0.909 | 0.216 | 0.78 | 5177 | 1.9e-06 |
| fly_best_shuf | 0.895 | 0.892 to 0.905 | 0.210 | 0.76 | 5177 | 1.9e-06 |
| fly_best_rand | 0.904 | 0.891 to 0.907 | 0.212 | 0.76 | 5177 | 1.9e-06 |
| zscore | 0.847 | 0.838 to 0.862 | 0.040 | 0.51 | 112 | 0.0017 |
| knn_budget | 0.919 | 0.905 to 0.937 | 0.031 | 0.85 | 5152 | 1.9e-06 |

Paired Wilcoxon, false alarms after the drop: fly vs zscore p = 8.8e-05 (mean diff +0.150); fly_gain vs zscore p = 8.8e-05 (mean diff +0.176); fly_adapt vs zscore p = 0.00013 (mean diff +0.059); fly_coinc vs zscore p = 1.9e-06 (mean diff +0.166); fly_gain_coinc vs zscore p = 1.9e-06 (mean diff +0.169); fly_all vs zscore p = 1.9e-06 (mean diff +0.101); fly_best vs zscore p = 8.8e-05 (mean diff +0.176); fly_best_shuf vs zscore p = 1.9e-06 (mean diff +0.170); fly_best_rand vs zscore p = 8.8e-05 (mean diff +0.172)

## S8 context input vs routed team (S1b stream) (20 seeds)

| Detector | AUROC mean | IQR | false alarms after laser drop | stack AUROC | stored numbers | p (AUROC vs fly) |
|---|---|---|---|---|---|---|
| fly | 0.860 | 0.848 to 0.876 | 0.190 | 0.63 | 5177 |  |
| fly_ctx1 | 0.963 | 0.960 to 0.968 | 0.199 | 0.94 | 5177 | 1.9e-06 |
| fly_ctx2 | 0.975 | 0.972 to 0.979 | 0.227 | 0.98 | 5177 | 1.9e-06 |
| fly_ctx4 | 0.973 | 0.969 to 0.977 | 0.209 | 0.98 | 5177 | 1.9e-06 |
| fly_ctx4_shuf | 0.975 | 0.973 to 0.978 | 0.214 | 0.98 | 5177 | 1.9e-06 |
| fly_ctx4_rand | 0.975 | 0.974 to 0.977 | 0.165 | 0.98 | 5177 | 1.9e-06 |
| fly_comp | 0.988 | 0.987 to 0.992 | 0.246 | 0.99 | 20708 | 1.9e-06 |
| fly_ctx2_comp | 0.982 | 0.979 to 0.985 | 0.224 | 0.99 | 20708 | 1.9e-06 |
| fam4_quarter | 0.982 | 0.978 to 0.986 | 0.199 | 0.99 | 5177 | 1.9e-06 |
| fam4_full | 0.988 | 0.987 to 0.992 | 0.246 | 0.99 | 20708 | 1.9e-06 |
| zscore_fam | 0.989 | 0.987 to 0.992 | 0.211 | 1.00 | 448 | 1.9e-06 |
| knn_budget_fam | 0.953 | 0.944 to 0.963 | 0.033 | 0.95 | 5152 | 1.9e-06 |

Paired Wilcoxon, false alarms after the drop: fly vs zscore_fam p = 0.03 (mean diff -0.021); fly_ctx1 vs zscore_fam p = 0.57 (mean diff -0.012); fly_ctx2 vs zscore_fam p = 0.083 (mean diff +0.017); fly_ctx4 vs zscore_fam p = 0.98 (mean diff -0.001); fly_ctx4_shuf vs zscore_fam p = 0.96 (mean diff +0.003); fly_ctx4_rand vs zscore_fam p = 0.00059 (mean diff -0.046); fly_comp vs zscore_fam p = 0.0019 (mean diff +0.035); fly_ctx2_comp vs zscore_fam p = 0.11 (mean diff +0.014)

Paired: fly_ctx4 vs fam4_quarter p = 1.9e-06 (-0.009); fly_ctx4 vs zscore_fam p = 1.9e-06 (-0.016); fly_comp vs fam4_full p = n/a (+0.000); fly_ctx4 vs fly_ctx4_shuf p = 0.001 (-0.002); fly_ctx4 vs fly_ctx4_rand p = 0.015 (-0.002)

## S4 in-situ Raman stop, Janus MoSSe selenization (20 seeds, 30 recipes each)

| Arm | flakes lost | IQR | runs stopped that would have failed | false stops | furnace time (runs) | gap | p (lost vs gp) |
|---|---|---|---|---|---|---|---|
| gp | 7.75 | 7 to 8 | 0.00 | 0.00 | 30.0 | 0.0233 |  |
| cbo | 4.35 | 3 to 5 | 0.00 | 0.00 | 30.0 | 0.0117 | 0.00016 |
| gp+fly | 3.55 | 3 to 4 | 4.50 | 4.05 | 23.6 | 0.1216 | 7.9e-05 |
| gp+fly_best | 3.30 | 3 to 4 | 5.00 | 1.80 | 24.9 | 0.0750 | 8.3e-05 |
| gp+fly_best_shuf | 3.15 | 3 to 4 | 4.45 | 2.10 | 25.1 | 0.0682 | 7.8e-05 |
| gp+fly_best_rand | 3.50 | 3 to 4 | 4.45 | 2.90 | 24.5 | 0.0885 | 8.2e-05 |
| gp+knn | 3.90 | 3 to 5 | 4.70 | 3.25 | 24.0 | 0.0265 | 8.2e-05 |
| gp+logreg | 3.90 | 3 to 4 | 4.70 | 2.50 | 24.6 | 0.0231 | 8.1e-05 |

Paired on flakes lost: gp+fly_best vs gp+knn p = 0.013 (-0.60); gp+fly_best vs gp+logreg p = 0.065 (-0.60); gp+fly_best vs cbo p = 0.033 (-1.05); gp+fly vs gp+knn p = 0.083 (-0.35); gp+fly_best vs gp+fly_best_shuf p = 0.26 (+0.15); gp+fly_best vs gp+fly_best_rand p = 0.37 (-0.20)

## S7b three Raman stations, private vs shared memory (20 seeds)

| Arm | AUROC new events | false alarms, network-known family (all / first 50) | false alarms, own families | MoTe2 recall at A | MoTe2 recall at B (all / first 5) | stored numbers |
|---|---|---|---|---|---|---|
| fly_private | 0.884 | 0.216 / 0.671 | 0.050 | 0.34 | 0.71 / 1.00 | 15531 |
| fly_shared | 0.837 | 0.007 / 0.013 | 0.050 | 0.41 | 0.16 / 0.30 | 5177 |
| fly_weighted | 0.860 | 0.019 / 0.034 | 0.050 | 0.38 | 0.42 / 0.67 | 15531 |
| fly_best_private | 0.916 | 0.221 / 0.654 | 0.050 | 0.29 | 0.72 / 1.00 | 15531 |
| fly_best_shared | 0.894 | 0.014 / 0.020 | 0.050 | 0.43 | 0.16 / 0.32 | 5177 |
| fly_best_weighted | 0.913 | 0.027 / 0.041 | 0.050 | 0.38 | 0.44 / 0.91 | 15531 |
| zscore_private | 0.872 | 0.492 / 0.799 | 0.050 | 0.55 | 0.96 / 1.00 | 336 |
| zscore_shared | 0.756 | 0.049 / 0.055 | 0.050 | 0.72 | 0.63 / 0.62 | 112 |
| knn_budget_private | 0.846 | 0.444 / 0.732 | 0.050 | 0.19 | 0.58 / 0.57 | 5040 |
| knn_budget_shared | 0.891 | 0.060 / 0.060 | 0.050 | 0.30 | 0.49 / 0.45 | 5152 |

Paired (shared vs private): fly: AUROC p = 1.9e-06, first-sighting false alarms p = 1.9e-06, MoTe2 at B p = 8.8e-05; fly_best: AUROC p = 1.9e-06, first-sighting false alarms p = 1.9e-06, MoTe2 at B p = 8.8e-05; zscore: AUROC p = 1.9e-06, first-sighting false alarms p = 1.9e-06, MoTe2 at B p = 8.8e-05; knn_budget: AUROC p = 1.3e-05, first-sighting false alarms p = 1.9e-06, MoTe2 at B p = 0.23

## S6 one module, several jobs (20 seeds, context gain 2.0)

| Detector | J1 new phase AUROC | J2 wrong sample AUROC | stack AUROC | stored numbers |
|---|---|---|---|---|
| module_ctx | 0.966 | 0.991 | 0.95 | 5177 |
| module_best_ctx | 0.964 | 0.989 | 0.96 | 5177 |
| module_ctx_shuf | 0.969 | 0.990 | 0.95 | 5177 |
| module_ctx_rand | 0.964 | 0.986 | 0.95 | 5177 |
| fam4_quarter | 0.975 | 0.996 | 0.97 | 5177 |
| zscore_fam | 0.986 | 0.997 | 0.98 | 448 |
| centroid | 0.905 | 1.000 | 0.96 | 224 |

Paired: module_ctx vs zscore_fam j1_new_phase p = 1.9e-06 (-0.020); module_ctx vs zscore_fam j2_wrong_sample p = 1.9e-06 (-0.007); module_ctx vs centroid j2_wrong_sample p = 1.9e-06 (-0.009); module_ctx vs module_ctx_shuf j1_new_phase p = 0.00048 (-0.003); module_ctx vs module_ctx_rand j1_new_phase p = 0.097 (+0.002)

## S5 long streams (10^4-reading blocks)

### s5_dgx_s1.json: DGX Spark (aarch64), seed 1, 100000 readings, peak memory 97 MB

| Detector | AUROC first / last block | false alarms first / last block | ms per reading first / last block | stored numbers |
|---|---|---|---|---|
| fly | 0.883 / 0.836 | 0.050 / 0.000 | 4.78 / 4.61 | 5177 |
| fly_best | 0.920 / 0.866 | 0.050 / 0.001 | 4.80 / 4.63 | 5177 |
| fly_recover | 0.903 / 0.882 | 0.050 / 0.045 | 4.77 / 4.60 | 5177 |
| knn_budget | 0.906 / 0.923 | 0.050 / 0.031 | 0.06 / 0.07 | 5152 |
| zscore | 0.842 / 0.856 | 0.050 / 0.038 | 0.02 / 0.02 | 112 |

### s5_dgx_s2.json: DGX Spark (aarch64), seed 2, 100000 readings, peak memory 97 MB

| Detector | AUROC first / last block | false alarms first / last block | ms per reading first / last block | stored numbers |
|---|---|---|---|---|
| fly | 0.869 / 0.851 | 0.050 / 0.000 | 2.53 / 1.27 | 5177 |
| fly_best | 0.919 / 0.859 | 0.050 / 0.001 | 2.55 / 1.29 | 5177 |
| fly_recover | 0.887 / 0.890 | 0.050 / 0.041 | 2.52 / 1.27 | 5177 |
| knn_budget | 0.950 / 0.858 | 0.050 / 0.067 | 0.05 / 0.03 | 5152 |
| zscore | 0.832 / 0.848 | 0.050 / 0.039 | 0.01 / 0.01 | 112 |

### s5_dgx_s3.json: DGX Spark (aarch64), seed 3, 100000 readings, peak memory 97 MB

| Detector | AUROC first / last block | false alarms first / last block | ms per reading first / last block | stored numbers |
|---|---|---|---|---|
| fly | 0.876 / 0.838 | 0.050 / 0.000 | 4.82 / 3.79 | 5177 |
| fly_best | 0.918 / 0.852 | 0.050 / 0.001 | 4.84 / 3.81 | 5177 |
| fly_recover | 0.913 / 0.875 | 0.050 / 0.043 | 4.81 / 3.78 | 5177 |
| knn_budget | 0.902 / 0.954 | 0.050 / 0.034 | 0.06 / 0.06 | 5152 |
| zscore | 0.859 / 0.837 | 0.050 / 0.040 | 0.02 / 0.02 | 112 |

### s5_pi5_s1.json: monsterpi (aarch64), seed 1, 100000 readings, peak memory 94 MB

| Detector | AUROC first / last block | false alarms first / last block | ms per reading first / last block | stored numbers |
|---|---|---|---|---|
| fly | 0.883 / 0.836 | 0.050 / 0.000 | 9.67 / 9.65 | 5177 |
| fly_best | 0.920 / 0.866 | 0.050 / 0.001 | 9.77 / 9.72 | 5177 |
| fly_recover | 0.903 / 0.882 | 0.050 / 0.045 | 9.71 / 9.65 | 5177 |
| knn_budget | 0.906 / 0.923 | 0.050 / 0.031 | 0.13 / 0.13 | 5152 |
| zscore | 0.842 / 0.856 | 0.050 / 0.038 | 0.05 / 0.05 | 112 |

