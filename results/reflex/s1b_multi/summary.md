## S1b four material families in one stream (20 seeds, 56 glomeruli = spectral bands, 184 new events in 3700 scored readings)

| Detector | AUROC mean | IQR | recall at 5 % false alarms | false alarms after laser change | ms per reading | stored numbers | p (AUROC vs fly) |
|---|---|---|---|---|---|---|---|
| fly | 0.860 | 0.848 to 0.876 | 0.55 | 0.190 | 1.05 | 5177 |  |
| fly_shuf | 0.867 | 0.856 to 0.877 | 0.56 | 0.182 | 1.09 | 5177 | 0.083 |
| fly_rand | 0.841 | 0.825 to 0.849 | 0.51 | 0.109 | 1.10 | 5177 | 0.00013 |
| fly_dense | 0.885 | 0.874 to 0.892 | 0.52 | 0.273 | 1.08 | 5177 | 1.9e-06 |
| knn_raw | 0.735 | 0.714 to 0.748 | 0.24 | 0.000 | 0.40 | 224000 | 1.9e-06 |
| knn_norm | 0.859 | 0.846 to 0.875 | 0.56 | 0.200 | 0.36 | 224000 | 0.96 |
| knn_budget | 0.919 | 0.905 to 0.937 | 0.61 | 0.031 | 0.03 | 5152 | 1.9e-06 |
| zscore | 0.847 | 0.838 to 0.862 | 0.64 | 0.040 | 0.01 | 112 | 0.0017 |

AUROC by kind of new event:

| Detector | 1T'_MoS2 | MoTe2 | spike | MoS2/WSe2_stack |
|---|---|---|---|---|
| fly | 0.914 | 0.918 | 0.978 | 0.629 |
| fly_shuf | 0.925 | 0.911 | 0.986 | 0.646 |
| fly_rand | 0.902 | 0.877 | 0.985 | 0.598 |
| fly_dense | 0.915 | 0.926 | 0.974 | 0.725 |
| knn_raw | 0.761 | 0.706 | 0.835 | 0.640 |
| knn_norm | 0.847 | 0.913 | 0.997 | 0.684 |
| knn_budget | 0.918 | 0.932 | 0.974 | 0.852 |
| zscore | 0.986 | 0.956 | 0.931 | 0.507 |

Paired Wilcoxon on false alarms after the laser change: fly_shuf vs fly p = 0.31; fly_rand vs fly p = 1.9e-06; fly_dense vs fly p = 1.9e-06; knn_raw vs fly p = 8.8e-05; knn_norm vs fly p = 0.22; knn_budget vs fly p = 1.9e-06; zscore vs fly p = 8.8e-05
