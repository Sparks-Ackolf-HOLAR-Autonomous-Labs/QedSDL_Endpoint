# S19b: tuned online networks and replay

## Tuning sweep (S1b, seeds 101-105; mean AUROC)

| network | learning rate | window | AUROC |
|---|---|---|---|
| ae16 | 1e-2 | - | 0.853 |
| ae16 | 1e-3 | - | 0.843 |
| ae16 | 3e-2 | - | 0.552 |
| ae16 | 3e-3 | - | 0.865 |
| ae16 | 3e-4 | - | 0.715 |
| ae46 | 1e-2 | - | 0.850 |
| ae46 | 1e-3 | - | 0.832 |
| ae46 | 3e-2 | - | 0.543 |
| ae46 | 3e-3 | - | 0.777 |
| ae46 | 3e-4 | - | 0.807 |
| lstm16 | 1e-2 | 16 | 0.486 |
| lstm16 | 1e-2 | 4 | 0.486 |
| lstm16 | 1e-2 | 8 | 0.486 |
| lstm16 | 1e-3 | 16 | 0.486 |
| lstm16 | 1e-3 | 4 | 0.486 |
| lstm16 | 1e-3 | 8 | 0.486 |
| lstm16 | 3e-3 | 16 | 0.486 |
| lstm16 | 3e-3 | 4 | 0.486 |
| lstm16 | 3e-3 | 8 | 0.486 |
| lstm16 | 3e-4 | 16 | 0.484 |
| lstm16 | 3e-4 | 4 | 0.484 |
| lstm16 | 3e-4 | 8 | 0.484 |
| lstmae16 | 1e-2 | 16 | 0.864 |
| lstmae16 | 1e-2 | 4 | 0.871 |
| lstmae16 | 1e-2 | 8 | 0.868 |
| lstmae16 | 1e-3 | 16 | 0.717 |
| lstmae16 | 1e-3 | 4 | 0.715 |
| lstmae16 | 1e-3 | 8 | 0.716 |
| lstmae16 | 3e-2 | 4 | 0.768 |
| lstmae16 | 3e-2 | 8 | 0.789 |
| lstmae16 | 3e-3 | 16 | 0.817 |
| lstmae16 | 3e-3 | 4 | 0.821 |
| lstmae16 | 3e-3 | 8 | 0.821 |
| lstmae16 | 3e-4 | 16 | 0.570 |
| lstmae16 | 3e-4 | 4 | 0.576 |
| lstmae16 | 3e-4 | 8 | 0.574 |

## New-phase flag, S1b shared facility (20 test seeds)

| detector | stored numbers | AUROC | AUROC before tuning | false alarms after laser drop |
|---|---|---|---|---|
| connectome model | 5,177 | 0.860 | | 0.190 |
| autoencoder 16, tuned (lr 3e-3) | 1,864 | 0.856 | 0.831 | 0.169 |
| autoencoder 46, tuned (lr 1e-2) | 5,254 | 0.860 | 0.818 | 0.036 |
| LSTM reconstructing, tuned (lr 1e-2, window 4) | 5,688 | 0.866 | 0.730 | 0.032 |
| autoencoder 16 + replay of 59 readings (equal memory) | 5,168 | 0.854 |  | 0.266 |
| autoencoder 16 + replay of 500 readings | 29,864 | 0.842 |  | 0.298 |

Paired against the connectome model (positive = network higher AUROC), Holm over this table:

| detector | AUROC difference | seeds network higher | p (Holm) |
|---|---|---|---|
| autoencoder 16, tuned (lr 3e-3) | -0.004 | 10/20 | 1 |
| autoencoder 46, tuned (lr 1e-2) | -0.000 | 9/20 | 1 |
| LSTM reconstructing, tuned (lr 1e-2, window 4) | +0.006 | 13/20 | 0.389 |
| autoencoder 16 + replay of 59 readings (equal memory) | -0.006 | 9/20 | 1 |
| autoencoder 16 + replay of 500 readings | -0.018 | 6/20 | 0.0681 |

## Rare phases that return, S5b (seeds per row in brackets, 10^5 readings)

Return ratio = score on return / score at first sighting; contrast ratio = the same after dividing each episode's score by the mean score of the routine readings in the 500 readings before it (removes general drift). Below 1 = remembered.

| detector | AUROC first / last block | alpha-MoO3 return / contrast | GaSe return / contrast | SnS2 return / contrast | recall at first sighting |
|---|---|---|---|---|---|
| connectome model, recovery 0 (30) | 0.875 / 0.837 | 0.13 / 0.23 | 0.15 / 2.52 | 0.08 / 22.93 | 0.76 |
| connectome model, recovery 1e-5 (30) | 0.890 / 0.871 | 0.18 / 0.18 | 0.37 / 0.38 | 0.43 / 0.39 | 0.78 |
| connectome model, recovery 3e-5 (30) | 0.896 / 0.872 | 0.24 / 0.24 | 0.61 / 0.61 | 0.71 / 0.64 | 0.80 |
| connectome model, recovery 1e-4 (30) | 0.901 / 0.872 | 0.36 / 0.37 | 0.88 / 0.88 | 0.84 / 0.76 | 0.79 |
| connectome model, recovery 1e-3 (30) | 0.894 / 0.866 | 0.80 / 0.80 | 0.99 / 0.99 | 0.87 / 0.82 | 0.53 |
| autoencoder 16, tuned (lr 3e-3) (30) | 0.813 / 0.755 | 0.84 / 0.85 | 0.87 / 0.91 | 0.74 / 0.42 | 0.69 |
| autoencoder 46, tuned (lr 1e-2) (30) | 0.881 / 0.883 | 0.99 / 1.02 | 0.98 / 0.98 | 0.93 / 0.73 | 0.47 |
| LSTM reconstructing, tuned (lr 1e-2, window 4) (30) | 0.898 / 0.908 | 0.91 / 0.90 | 0.97 / 0.96 | 0.73 / 0.74 | 0.40 |
| autoencoder 16 + replay of 59 readings (equal memory) (30) | 0.813 / 0.727 | 0.83 / 0.84 | 0.89 / 0.92 | 0.68 / 0.30 | 0.90 |
| autoencoder 16 + replay of 500 readings (30) | 0.783 / 0.686 | 0.85 / 0.86 | 0.92 / 0.94 | 0.59 / 0.26 | 0.91 |
