# S14: left versus right mushroom body, split into Kenyon-cell wiring and projection-neuron input

## Janus MoSSe selenization line (S1 stream), 20 seeds

| Arm | fafb | banc | malecns |
|---|---|---|---|
| left | 0.952 | 0.935 | 0.949 |
| right | 0.949 | 0.936 | 0.944 |
| left_nm | 0.951 | 0.935 | 0.947 |
| right_nm | 0.949 | 0.934 | 0.942 |
| left_gp | 0.938 | 0.922 | 0.940 |
| right_gp | 0.934 | 0.923 | 0.938 |
| LL | 0.948 | 0.938 | 0.948 |
| LR | 0.946 | 0.934 | 0.944 |
| RL | 0.948 | 0.938 | 0.947 |
| RR | 0.947 | 0.934 | 0.946 |

Per-band z-score: 0.975

| Animal | Contrast | mean | seeds > 0 | p (Holm) |
|---|---|---|---|---|
| fafb | left - right (measured) | +0.0024 | 14/20 | 0.61 |
| fafb | left_nm - right_nm (multiglomerular PNs silenced) | +0.0018 | 14/20 | 1 |
| fafb | LL - left_nm (exact left wiring vs its side shuffle) | -0.0030 | 6/20 | 0.72 |
| fafb | RR - right_nm (exact right wiring vs its side shuffle) | -0.0021 | 7/20 | 1 |
| fafb | LL - RR (side shuffles) | +0.0008 | 11/20 | 1 |
| fafb | left_gp - right_gp (random band map, same for both sides) | +0.0037 | 12/20 | 1 |
| fafb | KC effect: (LL+LR)/2 - (RL+RR)/2 | -0.0003 | 11/20 | 1 |
| fafb | PN effect: (LL+RL)/2 - (LR+RR)/2 | +0.0012 | 12/20 | 1 |
| fafb | interaction: (LL-LR) - (RL-RR) | +0.0008 | 13/20 | 1 |
| banc | left - right (measured) | -0.0001 | 10/20 | 1 |
| banc | left_nm - right_nm (multiglomerular PNs silenced) | +0.0005 | 11/20 | 1 |
| banc | LL - left_nm (exact left wiring vs its side shuffle) | +0.0035 | 16/20 | 0.72 |
| banc | RR - right_nm (exact right wiring vs its side shuffle) | +0.0001 | 12/20 | 1 |
| banc | LL - RR (side shuffles) | +0.0038 | 13/20 | 0.72 |
| banc | left_gp - right_gp (random band map, same for both sides) | -0.0012 | 9/20 | 1 |
| banc | KC effect: (LL+LR)/2 - (RL+RR)/2 | -0.0001 | 12/20 | 1 |
| banc | PN effect: (LL+RL)/2 - (LR+RR)/2 | +0.0039 | 14/20 | 0.18 |
| banc | interaction: (LL-LR) - (RL-RR) | +0.0005 | 10/20 | 1 |
| malecns | left - right (measured) | +0.0055 | 18/20 | 0.023 |
| malecns | left_nm - right_nm (multiglomerular PNs silenced) | +0.0050 | 18/20 | 0.023 |
| malecns | LL - left_nm (exact left wiring vs its side shuffle) | +0.0009 | 8/20 | 1 |
| malecns | RR - right_nm (exact right wiring vs its side shuffle) | +0.0044 | 15/20 | 0.33 |
| malecns | LL - RR (side shuffles) | +0.0016 | 12/20 | 1 |
| malecns | left_gp - right_gp (random band map, same for both sides) | +0.0018 | 12/20 | 1 |
| malecns | KC effect: (LL+LR)/2 - (RL+RR)/2 | -0.0006 | 9/20 | 1 |
| malecns | PN effect: (LL+RL)/2 - (LR+RR)/2 | +0.0022 | 11/20 | 1 |
| malecns | interaction: (LL-LR) - (RL-RR) | +0.0024 | 13/20 | 1 |

## shared facility, four material families (S1b stream), 20 seeds

| Arm | fafb | banc | malecns |
|---|---|---|---|
| left | 0.886 | 0.820 | 0.888 |
| right | 0.844 | 0.810 | 0.871 |
| left_nm | 0.878 | 0.818 | 0.884 |
| right_nm | 0.839 | 0.808 | 0.867 |
| left_gp | 0.837 | 0.812 | 0.846 |
| right_gp | 0.841 | 0.822 | 0.850 |
| LL | 0.869 | 0.817 | 0.868 |
| LR | 0.878 | 0.814 | 0.866 |
| RL | 0.865 | 0.824 | 0.868 |
| RR | 0.875 | 0.823 | 0.870 |

Per-band z-score: 0.847

| Animal | Contrast | mean | seeds > 0 | p (Holm) |
|---|---|---|---|---|
| fafb | left - right (measured) | +0.0423 | 20/20 | 5.1e-05 |
| fafb | left_nm - right_nm (multiglomerular PNs silenced) | +0.0393 | 20/20 | 5.1e-05 |
| fafb | LL - left_nm (exact left wiring vs its side shuffle) | -0.0089 | 6/20 | 1 |
| fafb | RR - right_nm (exact right wiring vs its side shuffle) | +0.0365 | 20/20 | 5.1e-05 |
| fafb | LL - RR (side shuffles) | -0.0061 | 7/20 | 1 |
| fafb | left_gp - right_gp (random band map, same for both sides) | -0.0038 | 7/20 | 1 |
| fafb | KC effect: (LL+LR)/2 - (RL+RR)/2 | +0.0037 | 11/20 | 1 |
| fafb | PN effect: (LL+RL)/2 - (LR+RR)/2 | -0.0098 | 6/20 | 0.53 |
| fafb | interaction: (LL-LR) - (RL-RR) | +0.0015 | 9/20 | 1 |
| banc | left - right (measured) | +0.0094 | 16/20 | 0.34 |
| banc | left_nm - right_nm (multiglomerular PNs silenced) | +0.0098 | 16/20 | 0.2 |
| banc | LL - left_nm (exact left wiring vs its side shuffle) | -0.0011 | 12/20 | 1 |
| banc | RR - right_nm (exact right wiring vs its side shuffle) | +0.0151 | 15/20 | 0.051 |
| banc | LL - RR (side shuffles) | -0.0064 | 11/20 | 1 |
| banc | left_gp - right_gp (random band map, same for both sides) | -0.0094 | 5/20 | 0.62 |
| banc | KC effect: (LL+LR)/2 - (RL+RR)/2 | -0.0084 | 5/20 | 1 |
| banc | PN effect: (LL+RL)/2 - (LR+RR)/2 | +0.0021 | 14/20 | 1 |
| banc | interaction: (LL-LR) - (RL-RR) | +0.0012 | 8/20 | 1 |
| malecns | left - right (measured) | +0.0178 | 18/20 | 0.00022 |
| malecns | left_nm - right_nm (multiglomerular PNs silenced) | +0.0175 | 20/20 | 5.1e-05 |
| malecns | LL - left_nm (exact left wiring vs its side shuffle) | -0.0163 | 6/20 | 0.46 |
| malecns | RR - right_nm (exact right wiring vs its side shuffle) | +0.0034 | 12/20 | 1 |
| malecns | LL - RR (side shuffles) | -0.0023 | 12/20 | 1 |
| malecns | left_gp - right_gp (random band map, same for both sides) | -0.0040 | 7/20 | 1 |
| malecns | KC effect: (LL+LR)/2 - (RL+RR)/2 | -0.0025 | 9/20 | 1 |
| malecns | PN effect: (LL+RL)/2 - (LR+RR)/2 | +0.0002 | 12/20 | 1 |
| malecns | interaction: (LL-LR) - (RL-RR) | +0.0043 | 11/20 | 1 |
