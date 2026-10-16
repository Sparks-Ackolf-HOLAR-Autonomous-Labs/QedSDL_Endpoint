## S7 one connectome model vs several, four-family Raman stream (S1b), 20 seeds

| Arm | memory (x one model) | AUROC mean | IQR | recall at 5 % false alarms | false alarms after laser change | stack AUROC | p (AUROC vs fly) |
|---|---|---|---|---|---|---|---|
| fly | 1.00 | 0.860 | 0.848 to 0.876 | 0.55 | 0.190 | 0.63 |  |
| ens4_wiring | 4.00 | 0.872 | 0.856 to 0.886 | 0.57 | 0.214 | 0.66 | 3.6e-05 |
| ens4_quarter | 1.00 | 0.861 | 0.852 to 0.878 | 0.55 | 0.193 | 0.63 | 0.55 |
| fam4_full | 4.00 | 0.988 | 0.987 to 0.992 | 0.94 | 0.246 | 0.99 | 1.9e-06 |
| fam4_quarter | 1.00 | 0.984 | 0.983 to 0.988 | 0.93 | 0.212 | 0.99 | 1.9e-06 |
| zscore | 0.02 | 0.847 | 0.838 to 0.862 | 0.64 | 0.040 | 0.51 | 0.0017 |
| zscore_fam | 0.09 | 0.989 | 0.987 to 0.992 | 0.96 | 0.211 | 1.00 | 1.9e-06 |
| knn_budget | 1.00 | 0.919 | 0.905 to 0.937 | 0.61 | 0.031 | 0.85 | 1.9e-06 |
| knn_budget_fam | 1.00 | 0.953 | 0.944 to 0.963 | 0.77 | 0.033 | 0.95 | 1.9e-06 |

AUROC by kind of new event:

| Arm | 1T'_MoS2 | MoTe2 | spike | MoS2/WSe2_stack |
|---|---|---|---|---|
| fly | 0.914 | 0.918 | 0.978 | 0.629 |
| ens4_wiring | 0.920 | 0.918 | 0.985 | 0.665 |
| ens4_quarter | 0.916 | 0.918 | 0.979 | 0.630 |
| fam4_full | 0.987 | 0.987 | 0.987 | 0.992 |
| fam4_quarter | 0.985 | 0.985 | 0.978 | 0.990 |
| zscore | 0.986 | 0.956 | 0.931 | 0.507 |
| zscore_fam | 0.997 | 0.987 | 0.976 | 0.996 |
| knn_budget | 0.918 | 0.932 | 0.974 | 0.852 |
| knn_budget_fam | 0.945 | 0.942 | 0.976 | 0.950 |

Paired Wilcoxon on AUROC: fam4_quarter vs zscore_fam p = 0.001 (mean diff -0.005); fam4_quarter vs knn_budget_fam p = 1.9e-06 (mean diff +0.032); fam4_full vs fam4_quarter p = 1.9e-06 (mean diff +0.004); ens4_wiring vs fly p = 3.6e-05 (mean diff +0.012)
