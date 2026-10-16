# S15b: small networks on the Ar+ irradiation-dose stop (20 seeds x 30 test flakes)

Same flakes as S15 task b: largest difference in the recomputed ratio/ridge/knn errors = 0.0e+00.

## 1 calibration flake(s)

| arm | success (within 15 %) | lost (overshoot) | mean abs. error |
|---|---|---|---|
| ratio | 0.297 | 0.052 | 23.9 % |
| ridge | 0.855 | 0.052 | 8.5 % |
| knn | 0.818 | 0.025 | 9.2 % |
| fly | 0.967 | 0.023 | 4.9 % |
| shuf | 0.965 | 0.017 | 4.9 % |
| rand | 0.965 | 0.025 | 5.3 % |
| dense | 0.947 | 0.015 | 5.9 % |
| mlp32 | 0.922 | 0.073 | 6.2 % |
| mlp64x2 | 0.987 | 0.013 | 4.1 % |

Paired error tests (positive = network has the smaller error), Holm over this table:

| network | reference | error difference | seeds network better | p (Holm) |
|---|---|---|---|---|
| mlp32 | fly | -0.0134 | 5/20 | 0.109 |
| mlp32 | ridge | +0.0231 | 15/20 | 0.127 |
| mlp32 | knn | +0.0302 | 16/20 | 0.00193 |
| mlp64x2 | fly | +0.0076 | 14/20 | 0.127 |
| mlp64x2 | ridge | +0.0441 | 20/20 | 1.14e-05 |
| mlp64x2 | knn | +0.0512 | 20/20 | 1.14e-05 |

## 3 calibration flake(s)

| arm | success (within 15 %) | lost (overshoot) | mean abs. error |
|---|---|---|---|
| ratio | 0.275 | 0.067 | 23.3 % |
| ridge | 0.937 | 0.035 | 6.0 % |
| knn | 0.923 | 0.025 | 6.3 % |
| fly | 0.992 | 0.008 | 4.0 % |
| shuf | 0.988 | 0.012 | 4.3 % |
| rand | 0.988 | 0.008 | 4.2 % |
| dense | 0.990 | 0.008 | 4.2 % |
| mlp32 | 0.958 | 0.035 | 5.4 % |
| mlp64x2 | 0.988 | 0.012 | 4.4 % |

Paired error tests (positive = network has the smaller error), Holm over this table:

| network | reference | error difference | seeds network better | p (Holm) |
|---|---|---|---|---|
| mlp32 | fly | -0.0137 | 5/20 | 0.0146 |
| mlp32 | ridge | +0.0064 | 14/20 | 0.0983 |
| mlp32 | knn | +0.0092 | 14/20 | 0.0983 |
| mlp64x2 | fly | -0.0039 | 8/20 | 0.189 |
| mlp64x2 | ridge | +0.0161 | 19/20 | 0.00161 |
| mlp64x2 | knn | +0.0190 | 17/20 | 0.000492 |

