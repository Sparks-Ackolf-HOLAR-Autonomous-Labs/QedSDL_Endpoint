# S13: the connectome model built from three animals

Shared glomerulus list: G = 55 glomeruli present as uniglomerular projection neurons in all three animals; PN-KC pairs with at least 3 synapses.

| Wiring | KCs (left / right) | PNs | PN-KC pairs | synapses | cross-side pairs (synapses) | mean PNs per KC |
|---|---|---|---|---|---|---|
| fafb | 5177 (2580 / 2597) | 685 | 23516 | 323907 | 487 (5943) | 4.54 |
| banc | 4552 (2260 / 2245) | 699 | 16720 | 195910 | 468 (4941) | 3.67 |
| malecns | 4064 (2019 / 2045) | 686 | 20777 | 388674 | 407 (7339) | 5.11 |
| fafb_raw | 5177 (2580 / 2597) | 685 | 27848 | 329394 | 589 (6074) | 5.38 |

## Janus MoSSe selenization line (S1 stream), 20 seeds

| Detector | KCs | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |
|---|---|---|---|---|---|---|---|
| fafb_left | 2580 | 0.952 | 0.949 to 0.956 | 0.217 |  | 0.52 | 2580 |
| fafb_right | 2597 | 0.949 | 0.944 to 0.952 | 0.204 |  | 0.52 | 2597 |
| fafb_both | 5177 | 0.948 | 0.943 to 0.952 | 0.217 |  | 1.96 | 5177 |
| fafb_both_nox | 5177 | 0.948 | 0.941 to 0.953 | 0.219 |  | 1.95 | 5177 |
| fafb_shuf | 5177 | 0.949 | 0.944 to 0.952 | 0.232 |  | 1.95 | 5177 |
| fafb_knn |  | 0.914 | 0.885 to 0.947 | 0.031 |  | 0.07 | 5170 |
| banc_left | 2260 | 0.935 | 0.931 to 0.940 | 0.161 |  | 0.46 | 2260 |
| banc_right | 2245 | 0.936 | 0.933 to 0.941 | 0.156 |  | 0.44 | 2245 |
| banc_both | 4552 | 0.937 | 0.930 to 0.944 | 0.175 |  | 1.76 | 4552 |
| banc_both_nox | 4552 | 0.936 | 0.929 to 0.944 | 0.181 |  | 1.76 | 4552 |
| banc_shuf | 4552 | 0.937 | 0.932 to 0.943 | 0.200 |  | 1.76 | 4552 |
| banc_knn |  | 0.918 | 0.897 to 0.945 | 0.028 |  | 0.06 | 4510 |
| malecns_left | 2019 | 0.949 | 0.945 to 0.953 | 0.219 |  | 0.41 | 2019 |
| malecns_right | 2045 | 0.944 | 0.938 to 0.951 | 0.206 |  | 0.41 | 2045 |
| malecns_both | 4064 | 0.947 | 0.943 to 0.951 | 0.215 |  | 1.53 | 4064 |
| malecns_both_nox | 4064 | 0.947 | 0.944 to 0.952 | 0.221 |  | 1.53 | 4064 |
| malecns_shuf | 4064 | 0.949 | 0.945 to 0.953 | 0.208 |  | 1.53 | 4064 |
| malecns_knn |  | 0.917 | 0.893 to 0.943 | 0.031 |  | 0.05 | 4015 |
| fafb_raw_both | 5177 | 0.949 | 0.945 to 0.954 | 0.214 |  | 1.95 | 5177 |
| zscore |  | 0.975 | 0.970 to 0.980 | 0.116 |  | 0.02 | 110 |

Paired Wilcoxon signed-rank tests over seeds, Holm-corrected within this stream:

| A vs B | mean AUROC A - B | seeds A > B | p (Holm) | mean false-alarm A - B | p (Holm) |
|---|---|---|---|---|---|
| fafb_both vs banc_both | +0.0110 | 20/20 | 4.8e-05 | +0.043 | 0.012 |
| fafb_both vs malecns_both | +0.0014 | 12/20 | 1 | +0.002 | 1 |
| banc_both vs malecns_both | -0.0096 | 0/20 | 4.8e-05 | -0.040 | 0.022 |
| fafb_left vs fafb_right | +0.0024 | 14/20 | 0.35 | +0.013 | 0.88 |
| fafb_both vs fafb_both_nox | +0.0004 | 10/20 | 1 | -0.002 | 1 |
| fafb_both vs fafb_shuf | -0.0004 | 8/20 | 1 | -0.015 | 1 |
| fafb_both vs fafb_left | -0.0034 | 6/20 | 0.055 | +0.000 | 1 |
| fafb_both vs fafb_right | -0.0010 | 7/20 | 1 | +0.013 | 1 |
| fafb_both vs fafb_knn | +0.0338 | 15/20 | 0.0079 | +0.186 | 4.8e-05 |
| zscore vs fafb_both | +0.0268 | 20/20 | 4.8e-05 | -0.102 | 4.8e-05 |
| banc_left vs banc_right | -0.0001 | 10/20 | 1 | +0.005 | 1 |
| banc_both vs banc_both_nox | +0.0010 | 14/20 | 0.032 | -0.007 | 0.99 |
| banc_both vs banc_shuf | +0.0004 | 12/20 | 1 | -0.025 | 0.17 |
| banc_both vs banc_left | +0.0018 | 12/20 | 1 | +0.013 | 1 |
| banc_both vs banc_right | +0.0016 | 11/20 | 1 | +0.019 | 0.7 |
| banc_both vs banc_knn | +0.0190 | 12/20 | 0.83 | +0.146 | 4.8e-05 |
| zscore vs banc_both | +0.0378 | 20/20 | 4.8e-05 | -0.059 | 0.0036 |
| malecns_left vs malecns_right | +0.0055 | 18/20 | 0.015 | +0.013 | 1 |
| malecns_both vs malecns_both_nox | -0.0006 | 8/20 | 1 | -0.006 | 1 |
| malecns_both vs malecns_shuf | -0.0022 | 5/20 | 0.83 | +0.007 | 1 |
| malecns_both vs malecns_left | -0.0022 | 6/20 | 0.39 | -0.005 | 1 |
| malecns_both vs malecns_right | +0.0033 | 14/20 | 0.34 | +0.008 | 1 |
| malecns_both vs malecns_knn | +0.0294 | 15/20 | 0.029 | +0.184 | 4.8e-05 |
| zscore vs malecns_both | +0.0282 | 20/20 | 4.8e-05 | -0.099 | 8e-05 |
| fafb_raw_both vs fafb_both | +0.0011 | 15/20 | 0.011 | -0.004 | 1 |

## shared facility, four material families (S1b stream), 20 seeds

| Detector | KCs | AUROC mean | IQR | false alarms after laser drop | stack AUROC | ms per reading | stored numbers |
|---|---|---|---|---|---|---|---|
| fafb_left | 2580 | 0.886 | 0.878 to 0.894 | 0.164 | 0.72 | 0.43 | 2580 |
| fafb_right | 2597 | 0.844 | 0.836 to 0.855 | 0.172 | 0.59 | 0.43 | 2597 |
| fafb_both | 5177 | 0.866 | 0.856 to 0.879 | 0.191 | 0.65 | 1.62 | 5177 |
| fafb_both_nox | 5177 | 0.868 | 0.856 to 0.882 | 0.183 | 0.65 | 1.60 | 5177 |
| fafb_shuf | 5177 | 0.872 | 0.858 to 0.884 | 0.186 | 0.68 | 1.60 | 5177 |
| fafb_knn |  | 0.916 | 0.908 to 0.928 | 0.040 | 0.86 | 0.06 | 5170 |
| banc_left | 2260 | 0.820 | 0.807 to 0.833 | 0.115 | 0.57 | 0.38 | 2260 |
| banc_right | 2245 | 0.810 | 0.796 to 0.824 | 0.109 | 0.53 | 0.37 | 2245 |
| banc_both | 4552 | 0.809 | 0.797 to 0.818 | 0.116 | 0.52 | 1.45 | 4552 |
| banc_both_nox | 4552 | 0.805 | 0.793 to 0.815 | 0.123 | 0.52 | 1.44 | 4552 |
| banc_shuf | 4552 | 0.830 | 0.817 to 0.847 | 0.127 | 0.54 | 1.44 | 4552 |
| banc_knn |  | 0.925 | 0.913 to 0.950 | 0.036 | 0.87 | 0.05 | 4510 |
| malecns_left | 2019 | 0.888 | 0.879 to 0.895 | 0.150 | 0.75 | 0.34 | 2019 |
| malecns_right | 2045 | 0.871 | 0.861 to 0.879 | 0.149 | 0.68 | 0.34 | 2045 |
| malecns_both | 4064 | 0.877 | 0.867 to 0.884 | 0.173 | 0.70 | 1.25 | 4064 |
| malecns_both_nox | 4064 | 0.877 | 0.867 to 0.883 | 0.174 | 0.70 | 1.25 | 4064 |
| malecns_shuf | 4064 | 0.879 | 0.868 to 0.888 | 0.163 | 0.69 | 1.25 | 4064 |
| malecns_knn |  | 0.915 | 0.900 to 0.932 | 0.030 | 0.86 | 0.04 | 4015 |
| fafb_raw_both | 5177 | 0.863 | 0.851 to 0.879 | 0.184 | 0.65 | 1.60 | 5177 |
| zscore |  | 0.847 | 0.839 to 0.862 | 0.040 | 0.51 | 0.02 | 110 |

Paired Wilcoxon signed-rank tests over seeds, Holm-corrected within this stream:

| A vs B | mean AUROC A - B | seeds A > B | p (Holm) | mean false-alarm A - B | p (Holm) |
|---|---|---|---|---|---|
| fafb_both vs banc_both | +0.0572 | 20/20 | 4.8e-05 | +0.075 | 4.8e-05 |
| fafb_both vs malecns_both | -0.0110 | 2/20 | 0.0009 | +0.018 | 0.22 |
| banc_both vs malecns_both | -0.0682 | 0/20 | 4.8e-05 | -0.057 | 4.8e-05 |
| fafb_left vs fafb_right | +0.0423 | 20/20 | 4.8e-05 | -0.008 | 1 |
| fafb_both vs fafb_both_nox | -0.0023 | 7/20 | 0.16 | +0.008 | 1 |
| fafb_both vs fafb_shuf | -0.0060 | 5/20 | 0.25 | +0.005 | 1 |
| fafb_both vs fafb_left | -0.0204 | 0/20 | 4.8e-05 | +0.027 | 0.3 |
| fafb_both vs fafb_right | +0.0219 | 20/20 | 4.8e-05 | +0.019 | 0.35 |
| fafb_both vs fafb_knn | -0.0504 | 0/20 | 4.8e-05 | +0.151 | 4.8e-05 |
| zscore vs fafb_both | -0.0192 | 2/20 | 0.00043 | -0.151 | 0.0017 |
| banc_left vs banc_right | +0.0094 | 16/20 | 0.1 | +0.005 | 1 |
| banc_both vs banc_both_nox | +0.0035 | 19/20 | 9.2e-05 | -0.007 | 0.66 |
| banc_both vs banc_shuf | -0.0209 | 2/20 | 0.0019 | -0.011 | 0.74 |
| banc_both vs banc_left | -0.0111 | 2/20 | 0.00019 | +0.002 | 1 |
| banc_both vs banc_right | -0.0017 | 11/20 | 1 | +0.007 | 1 |
| banc_both vs banc_knn | -0.1164 | 0/20 | 4.8e-05 | +0.080 | 7.6e-05 |
| zscore vs banc_both | +0.0380 | 20/20 | 4.8e-05 | -0.076 | 4.8e-05 |
| malecns_left vs malecns_right | +0.0178 | 18/20 | 0.00014 | +0.001 | 1 |
| malecns_both vs malecns_both_nox | -0.0003 | 10/20 | 1 | -0.001 | 1 |
| malecns_both vs malecns_shuf | -0.0016 | 8/20 | 1 | +0.010 | 1 |
| malecns_both vs malecns_left | -0.0115 | 2/20 | 0.00025 | +0.023 | 0.13 |
| malecns_both vs malecns_right | +0.0064 | 15/20 | 0.014 | +0.024 | 0.027 |
| malecns_both vs malecns_knn | -0.0377 | 1/20 | 0.001 | +0.143 | 0.0017 |
| zscore vs malecns_both | -0.0302 | 1/20 | 6.5e-05 | -0.133 | 4.8e-05 |
| fafb_raw_both vs fafb_both | -0.0026 | 4/20 | 0.022 | -0.007 | 1 |
