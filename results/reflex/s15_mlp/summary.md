# S15-MLP: small trained networks vs the connectome model (20 seeds, held-out AUROC, mean)

| arm | n=25 | n=50 | n=100 | n=200 | n=400 | w_zero_shot | w_fine_tuned | w_scratch |
|---|---|---|---|---|---|---|---|---|
| fly | 0.946 | 0.969 | 0.980 | 0.986 | 0.988 | 0.765 | 0.886 | 0.884 |
| shuf | 0.943 | 0.965 | 0.981 | 0.986 | 0.989 | 0.694 | 0.889 | 0.890 |
| rand | 0.942 | 0.966 | 0.981 | 0.986 | 0.990 | 0.705 | 0.894 | 0.900 |
| dense | 0.884 | 0.921 | 0.972 | 0.955 | 0.966 | 0.738 | 0.860 | 0.819 |
| logreg | 0.966 | 0.977 | 0.990 | 0.992 | 0.994 | 0.742 | 0.941 | 0.914 |
| knn | 0.839 | 0.914 | 0.958 | 0.970 | 0.978 | 0.684 | 0.846 | 0.794 |
| mlp32 | 0.961 | 0.978 | 0.989 | 0.993 | 0.995 | 0.680 | 0.922 | 0.918 |
| mlp64x2 | 0.962 | 0.977 | 0.988 | 0.992 | 0.995 | 0.684 | 0.917 | 0.917 |

Paired tests (MLP minus reference; Wilcoxon signed-rank, Holm over these 20 tests)

| MLP | reference | measure | mean difference | seeds MLP higher | p (Holm) |
|---|---|---|---|---|---|
| mlp32 | fly | 25 | +0.015 | 13/20 | 0.487 |
| mlp32 | fly | 100 | +0.009 | 16/20 | 0.00761 |
| mlp32 | fly | 400 | +0.007 | 20/20 | 3.81e-05 |
| mlp32 | fly | w_zero_shot | -0.085 | 3/20 | 0.0209 |
| mlp32 | fly | w_fine_tuned | +0.036 | 18/20 | 0.00294 |
| mlp32 | logreg | 25 | -0.004 | 11/20 | 1 |
| mlp32 | logreg | 100 | -0.000 | 11/20 | 1 |
| mlp32 | logreg | 400 | +0.002 | 20/20 | 3.81e-05 |
| mlp32 | logreg | w_zero_shot | -0.062 | 1/20 | 0.000162 |
| mlp32 | logreg | w_fine_tuned | -0.019 | 8/20 | 1 |
| mlp64x2 | fly | 25 | +0.016 | 13/20 | 0.237 |
| mlp64x2 | fly | 100 | +0.008 | 15/20 | 0.0102 |
| mlp64x2 | fly | 400 | +0.007 | 20/20 | 3.81e-05 |
| mlp64x2 | fly | w_zero_shot | -0.081 | 4/20 | 0.0186 |
| mlp64x2 | fly | w_fine_tuned | +0.031 | 18/20 | 0.0186 |
| mlp64x2 | logreg | 25 | -0.004 | 12/20 | 1 |
| mlp64x2 | logreg | 100 | -0.001 | 7/20 | 0.487 |
| mlp64x2 | logreg | 400 | +0.001 | 19/20 | 0.00058 |
| mlp64x2 | logreg | w_zero_shot | -0.058 | 2/20 | 0.000715 |
| mlp64x2 | logreg | w_fine_tuned | -0.024 | 8/20 | 0.487 |
