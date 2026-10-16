## S1 Raman spectra as odours (20 seeds, 56 glomeruli = spectral bands, 186 new events in 3700 scored readings)

| Detector | AUROC mean | IQR | recall at 5 % false alarms | false alarms after laser change | ms per reading | stored numbers | p (AUROC vs fly) |
|---|---|---|---|---|---|---|---|
| fly | 0.951 | 0.943 to 0.956 | 0.74 | 0.220 | 1.01 | 5177 |  |
| fly_shuf | 0.950 | 0.945 to 0.958 | 0.73 | 0.229 | 1.06 | 5177 | 0.76 |
| fly_rand | 0.939 | 0.935 to 0.946 | 0.69 | 0.186 | 1.07 | 5177 | 5.7e-06 |
| fly_dense | 0.938 | 0.932 to 0.944 | 0.68 | 0.268 | 1.04 | 5177 | 3.8e-06 |
| knn_raw | 0.730 | 0.722 to 0.738 | 0.24 | 0.000 | 0.39 | 224000 | 1.9e-06 |
| knn_norm | 0.856 | 0.843 to 0.866 | 0.56 | 0.239 | 0.35 | 224000 | 1.9e-06 |
| knn_budget | 0.920 | 0.887 to 0.947 | 0.70 | 0.034 | 0.03 | 5152 | 0.0017 |
| zscore | 0.977 | 0.973 to 0.980 | 0.90 | 0.112 | 0.01 | 112 | 1.9e-06 |

AUROC by kind of new event:

| Detector | 1T'_MoS2 | WS2 | spike |
|---|---|---|---|
| fly | 0.947 | 0.917 | 0.989 |
| fly_shuf | 0.945 | 0.917 | 0.991 |
| fly_rand | 0.925 | 0.898 | 0.995 |
| fly_dense | 0.935 | 0.905 | 0.977 |
| knn_raw | 0.708 | 0.663 | 0.816 |
| knn_norm | 0.726 | 0.838 | 0.997 |
| knn_budget | 0.875 | 0.901 | 0.985 |
| zscore | 0.994 | 0.967 | 0.971 |

Paired Wilcoxon on false alarms after the laser change: fly_shuf vs fly p = 0.19; fly_rand vs fly p = 0.014; fly_dense vs fly p = 0.00039; knn_raw vs fly p = 8.8e-05; knn_norm vs fly p = 0.064; knn_budget vs fly p = 8.8e-05; zscore vs fly p = 1.9e-06
