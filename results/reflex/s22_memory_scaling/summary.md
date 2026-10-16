# S22 memory against data and model size

## S22a: memory against readings seen (10 seeds, facility stream without laser drop)

Bytes held / median us per reading (last 200) / AUROC in the window ending at N (median over seeds).

| detector | N=300 | N=1,000 | N=3,000 | N=10,000 | N=30,000 | N=100,000 |
|---|---|---|---|---|---|---|
| fly | 373 kB / 274 / - | 373 kB / 268 / 0.914 | 373 kB / 272 / 0.908 | 373 kB / 167 / 0.898 | 373 kB / 179 / 0.889 | 373 kB / 175 / 0.858 |
| zscore | 448 B / 26 / - | 448 B / 24 / 0.836 | 448 B / 25 / 0.849 | 448 B / 12 / 0.853 | 448 B / 14 / 0.848 | 448 B / 14 / 0.844 |
| knn_budget | 20.6 kB / 31 / - | 20.6 kB / 28 / 0.946 | 20.6 kB / 29 / 0.908 | 20.6 kB / 15 / 0.907 | 20.6 kB / 17 / 0.911 | 20.6 kB / 17 / 0.912 |
| knn_full | 67.2 kB / 25 / - | 224 kB / 62 / 0.933 | 672 kB / 169 / 0.876 | 2.24 MB / 364 / 0.833 | 6.72 MB / 1310 / 0.787 | 22.4 MB / 4033 / 0.745 |
| ae_res | 108 kB / 1679 / - | 108 kB / 1570 / 0.881 | 108 kB / 1625 / 0.745 | 108 kB / 568 / 0.742 | 108 kB / 757 / 0.747 | 108 kB / 731 / 0.784 |
| ae_full | 155 kB / 1630 / - | 311 kB / 1531 / 0.876 | 759 kB / 1565 / 0.730 | 2.33 MB / 396 / 0.683 | 6.81 MB / 452 / 0.696 | 22.5 MB / 427 / 0.726 |
| fly_rec | 373 kB / 255 / - | 373 kB / 250 / 0.909 | 373 kB / 252 / 0.901 | 373 kB / 198 / 0.906 | 373 kB / 250 / 0.901 | 373 kB / 149 / 0.899 |

AUROC over readings 3,000-100,000 (window-size-weighted), paired vs connectome model:

| detector | median | vs model (median diff) | seeds above model | Holm p |
|---|---|---|---|---|
| fly | 0.866 | | | |
| zscore | 0.845 | -0.022 | 0/10 | 0.0117 |
| knn_budget | 0.909 | +0.045 | 10/10 | 0.0117 |
| knn_full | 0.761 | -0.106 | 0/10 | 0.0117 |
| ae_res | 0.769 | -0.096 | 0/10 | 0.0117 |
| ae_full | 0.719 | -0.148 | 0/10 | 0.0117 |
| fly_rec | 0.900 | +0.033 | 10/10 | 0.0117 |

Growing against fixed memory (same method family):

| comparison | median diff | seeds above | Holm p |
|---|---|---|---|
| knn_full - knn_budget | -0.149 | 0/10 | 0.00781 |
| ae_full - ae_res | -0.054 | 0/10 | 0.00781 |
| knn_budget - fly_rec | +0.010 | 6/10 | 0.77 |
| fly_rec - fly | +0.033 | 10/10 | 0.00781 |

With 10 seeds the smallest two-sided Wilcoxon p is 0.00195.

## S22b: detection against model size (10 seeds, facility stream S1b)

| arm | bytes held | of which plastic | us per reading | AUROC median [IQR] |
|---|---|---|---|---|
| fly_measured | 373 kB | 20.7 kB | 311 | 0.861 [0.840, 0.875] |
| kc_x0.25 | 107 kB | 5.18 kB | 95 | 0.855 [0.831, 0.866] |
| knn_eq0.25 | 5.15 kB | 5.15 kB | 37 | 0.853 [0.842, 0.873] |
| kc_x0.5 | 197 kB | 10.4 kB | 151 | 0.843 [0.824, 0.869] |
| knn_eq0.5 | 10.3 kB | 10.3 kB | 24 | 0.905 [0.902, 0.919] |
| kc_x1 | 374 kB | 20.7 kB | 262 | 0.870 [0.856, 0.880] |
| knn_eq1 | 20.6 kB | 20.6 kB | 27 | 0.929 [0.914, 0.938] |
| kc_x2 | 727 kB | 41.4 kB | 485 | 0.866 [0.848, 0.879] |
| knn_eq2 | 41.2 kB | 41.2 kB | 34 | 0.921 [0.910, 0.936] |
| kc_x4 | 1.44 MB | 82.8 kB | 927 | 0.877 [0.862, 0.886] |
| knn_eq4 | 82.7 kB | 82.7 kB | 57 | 0.913 [0.890, 0.924] |
| kc_x8 | 2.85 MB | 166 kB | 1808 | 0.879 [0.866, 0.883] |
| knn_eq8 | 166 kB | 166 kB | 84 | 0.903 [0.885, 0.908] |
| knn_full | 896 kB | 896 kB | 145 | 0.864 [0.847, 0.874] |
| ae16 | 22.4 kB | 22.4 kB | 1398 | 0.854 [0.834, 0.878] |
| ae64 | 87.5 kB | 87.5 kB | 1246 | 0.784 [0.749, 0.789] |
| ae256 | 348 kB | 348 kB | 1356 | 0.774 [0.745, 0.784] |
| ae1024 | 1.39 MB | 1.39 MB | 1776 | 0.771 [0.753, 0.786] |

| comparison | median AUROC diff | Holm p |
|---|---|---|
| kc_x8 - kc_x1 | +0.012 | 0.336 |
| kc_x0.25 - kc_x1 | -0.031 | 0.336 |
| kc_x1 - fly_measured | +0.008 | 0.336 |
| knn_eq1 - kc_x1 | +0.063 | 0.0234 |
| knn_eq8 - kc_x8 | +0.023 | 0.0488 |
| knn_full - knn_eq1 | -0.065 | 0.0156 |
| ae1024 - ae16 | -0.088 | 0.0156 |
| ae16 - kc_x1 | -0.003 | 0.336 |

## Gaussian-process planner (DGX Grace core)

| stored readings | s per refit | bytes |
|---|---|---|
| 100 | 0.00145 | 205 kB |
| 300 | 0.0198 | 1.57 MB |
| 1000 | 0.241 | 16.4 MB |
| 3000 | 3.15 | 145 MB |
- fills Raspberry Pi Pico (RP2040) RAM at n = 128 readings (16 n^2 bytes)
- fills Raspberry Pi 5 RAM at n = 22,360 readings (16 n^2 bytes)
- fills DGX Spark unified memory at n = 89,442 readings (16 n^2 bytes)

## Raspberry Pi 5 (monsterpi), single thread

| method | stored | ms per decision | bytes |
|---|---|---|---|
| kNN, every reading | 1000 | 0.0817 | 224 kB |
| kNN, every reading | 10000 | 1.49 | 2.24 MB |
| kNN, every reading | 100000 | 24.3 | 22.4 MB |
| kNN, every reading | 1000000 | 367 | 224 MB |
| GP refit | 100 | 5.3 | 205 kB |
| GP refit | 300 | 89.9 | 1.57 MB |
| GP refit | 1000 | 1.71e+03 | 16.4 MB |
| GP refit | 2000 | 1.12e+04 | 64.9 MB |
- kNN keeping every reading fills Raspberry Pi Pico (RP2040) RAM after 1,178 readings
- kNN keeping every reading fills Raspberry Pi 5 RAM after 35,714,285 readings
- kNN keeping every reading fills DGX Spark unified memory after 571,428,571 readings

## MACE-MP-0 medium on the DGX GPU (2H MoS2 supercells)

- float32: 4,688,656 parameters (18.8 MB); peak memory per call 48 atoms 203 MB (0.47 s), 192 atoms 532 MB (0.82 s), 432 atoms 1.19 GB (1.1 s), 768 atoms 2.11 GB (0.26 s), 1728 atoms 4.74 GB (0.53 s), 3072 atoms 8.42 GB (0.95 s), 6912 atoms 18.9 GB (2.1 s); slope 2.74 MB per atom; Pi 5 RAM holds about 2,922 atoms
- float64: 4,688,656 parameters (37.5 MB); peak memory per call 48 atoms 273 MB (0.96 s), 192 atoms 1.06 GB (0.35 s), 432 atoms 2.37 GB (0.79 s), 768 atoms 4.21 GB (1.4 s), 1728 atoms 9.47 GB (3.2 s), 3072 atoms 16.8 GB (5.6 s), 6912 atoms 37.8 GB (12 s); slope 5.47 MB per atom; Pi 5 RAM holds about 1,461 atoms

## Language models (weights and per-sequence context memory)

| model | weights | KV cache per token | fixed recurrent state | tokens per reading | readings in max context | context memory at max |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 19.3 GB | 32.8 kB | 51.5 MB | 299 | 876 | 8.64 GB |
| Qwen3.6-35B-A3B-FP8 | 37.5 GB | 20.5 kB | 64.4 MB | 299 | 876 | 5.43 GB |

