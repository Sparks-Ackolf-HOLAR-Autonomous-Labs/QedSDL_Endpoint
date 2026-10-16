# S21: networks with memory and four replay designs against the connectome model

## New-phase flag, S1b shared facility (20 seeds)

Connectome model: AUROC 0.860, false alarms after the laser drop 0.190, 5,177 stored numbers. Equal-memory nearest neighbour: AUROC 0.919.

| network | AUROC, no replay | AUROC, reservoir | AUROC, recent (FIFO) | AUROC, surprise-prioritised | stored numbers (no replay / replay) | ms per reading (no replay) |
|---|---|---|---|---|---|---|
| autoencoder | 0.856 | 0.853 | 0.837 | 0.789 | 1,864 / 5,448 | 0.21 |
| LSTM | 0.864 | 0.885 | 0.873 | 0.826 | 5,688 / 20,024 | 0.37 |
| GRU | 0.844 | 0.868 | 0.861 | 0.810 | 4,504 / 18,840 | 0.66 |
| peephole LSTM | 0.863 | 0.888 | 0.878 | 0.822 | 5,672 / 20,008 | 0.60 |
| Mogrifier LSTM | 0.853 | 0.856 | 0.843 | 0.766 | 8,072 / 22,408 | 1.58 |
| transformer | 0.852 | 0.858 | 0.852 | 0.784 | 4,216 / 18,552 | 0.66 |
| memory slots learned by gradient (MemAE) | 0.738 | 0.855 | 0.856 | 0.788 | 5,064 / 8,648 | 0.63 |
| fast slot memory written in one shot | 0.845 | 0.858 | 0.863 | 0.846 | 5,064 / 8,648 | 0.36 |
| fast weights (delta rule) | 0.856 | 0.838 | 0.827 | 0.775 | 2,760 / 6,344 | 0.25 |

False alarms after the laser drop (share of routine readings over the pre-drop threshold):

| network | no replay | reservoir | recent (FIFO) | surprise-prioritised |
|---|---|---|---|---|
| autoencoder | 0.169 | 0.278 | 0.289 | 0.207 |
| LSTM | 0.032 | 0.027 | 0.030 | 0.042 |
| GRU | 0.031 | 0.027 | 0.032 | 0.030 |
| peephole LSTM | 0.031 | 0.032 | 0.030 | 0.038 |
| Mogrifier LSTM | 0.030 | 0.043 | 0.050 | 0.103 |
| transformer | 0.060 | 0.252 | 0.290 | 0.264 |
| memory slots learned by gradient (MemAE) | 0.029 | 0.028 | 0.029 | 0.038 |
| fast slot memory written in one shot | 0.029 | 0.034 | 0.031 | 0.043 |
| fast weights (delta rule) | 0.066 | 0.179 | 0.148 | 0.161 |

Paired against the connectome model (positive = network higher AUROC), Holm over 36 comparisons:

| network, replay | AUROC difference | seeds network higher | p (Holm) |
|---|---|---|---|
| autoencoder, no replay | -0.004 | 10/20 | 1 |
| autoencoder, reservoir | -0.007 | 9/20 | 1 |
| autoencoder, recent (FIFO) | -0.022 | 5/20 | 0.117 |
| autoencoder, surprise-prioritised | -0.071 | 0/20 | 6.87e-05 |
| LSTM, no replay | +0.004 | 14/20 | 1 |
| LSTM, reservoir | +0.025 | 16/20 | 0.0264 |
| LSTM, recent (FIFO) | +0.013 | 11/20 | 1 |
| LSTM, surprise-prioritised | -0.034 | 2/20 | 0.0047 |
| GRU, no replay | -0.016 | 5/20 | 0.112 |
| GRU, reservoir | +0.008 | 14/20 | 1 |
| GRU, recent (FIFO) | +0.001 | 13/20 | 1 |
| GRU, surprise-prioritised | -0.050 | 1/20 | 0.000118 |
| peephole LSTM, no replay | +0.003 | 9/20 | 1 |
| peephole LSTM, reservoir | +0.028 | 17/20 | 0.0087 |
| peephole LSTM, recent (FIFO) | +0.018 | 16/20 | 0.117 |
| peephole LSTM, surprise-prioritised | -0.038 | 2/20 | 0.00183 |
| Mogrifier LSTM, no replay | -0.007 | 8/20 | 1 |
| Mogrifier LSTM, reservoir | -0.004 | 9/20 | 1 |
| Mogrifier LSTM, recent (FIFO) | -0.016 | 5/20 | 0.112 |
| Mogrifier LSTM, surprise-prioritised | -0.094 | 0/20 | 6.87e-05 |
| transformer, no replay | -0.008 | 7/20 | 0.749 |
| transformer, reservoir | -0.001 | 9/20 | 1 |
| transformer, recent (FIFO) | -0.008 | 7/20 | 1 |
| transformer, surprise-prioritised | -0.075 | 0/20 | 6.87e-05 |
| memory slots learned by gradient (MemAE), no replay | -0.122 | 0/20 | 6.87e-05 |
| memory slots learned by gradient (MemAE), reservoir | -0.005 | 14/20 | 1 |
| memory slots learned by gradient (MemAE), recent (FIFO) | -0.003 | 12/20 | 1 |
| memory slots learned by gradient (MemAE), surprise-prioritised | -0.072 | 1/20 | 0.000172 |
| fast slot memory written in one shot, no replay | -0.015 | 7/20 | 0.365 |
| fast slot memory written in one shot, reservoir | -0.002 | 10/20 | 1 |
| fast slot memory written in one shot, recent (FIFO) | +0.003 | 12/20 | 1 |
| fast slot memory written in one shot, surprise-prioritised | -0.014 | 8/20 | 0.72 |
| fast weights (delta rule), no replay | -0.004 | 11/20 | 1 |
| fast weights (delta rule), reservoir | -0.022 | 3/20 | 0.0264 |
| fast weights (delta rule), recent (FIFO) | -0.032 | 5/20 | 0.0757 |
| fast weights (delta rule), surprise-prioritised | -0.085 | 0/20 | 6.87e-05 |

Replay design, mean over the nine networks (AUROC / false alarms after the drop):

| replay | AUROC | false alarms |
|---|---|---|
| no replay | 0.841 | 0.053 |
| reservoir | 0.862 | 0.100 |
| recent (FIFO) | 0.855 | 0.103 |
| surprise-prioritised | 0.801 | 0.103 |

## Rare phases that return, S5b (10^5 readings; seeds per row in brackets)

Contrast = episode mean score / mean score of the routine readings in the 500 readings before it; contrast ratio = return contrast / first contrast (below 1 = remembered, after removing general drift). Recall at first sighting at the threshold of the first block.

| detector | AUROC first / last block | alpha-MoO3 contrast ratio | GaSe contrast ratio | SnS2 contrast ratio | recall at first sighting |
|---|---|---|---|---|---|
| connectome model, recovery 0 (30) | 0.875 / 0.837 | 0.23 | 2.52 | 22.93 | 0.76 |
| connectome model, recovery 1e-5 (30) | 0.890 / 0.871 | 0.18 | 0.38 | 0.39 | 0.78 |
| connectome model, recovery 1e-4 (30) | 0.901 / 0.872 | 0.37 | 0.88 | 0.76 | 0.79 |
| autoencoder, no replay (30) | 0.813 / 0.755 | 0.85 | 0.91 | 0.42 | 0.69 |
| autoencoder, reservoir (30) | 0.812 / 0.741 | 0.87 | 0.91 | 0.31 | 0.89 |
| autoencoder, recent (FIFO) (30) | 0.790 / 0.697 | 0.80 | 0.88 | 0.37 | 0.75 |
| autoencoder, surprise-prioritised (30) | 0.774 / 0.732 | 0.34 | 1.16 | 0.91 | 0.66 |
| LSTM, no replay (30) | 0.892 / 0.908 | 0.89 | 1.02 | 0.75 | 0.39 |
| LSTM, reservoir (30) | 0.904 / 0.876 | 0.87 | 0.97 | 0.61 | 0.69 |
| LSTM, recent (FIFO) (30) | 0.891 / 0.799 | 0.83 | 0.97 | 0.60 | 0.57 |
| LSTM, surprise-prioritised (30) | 0.811 / 0.765 | 0.37 | 1.03 | 0.81 | 0.44 |
| GRU, no replay (30) | 0.865 / 0.894 | 0.92 | 1.03 | 0.95 | 0.20 |
| GRU, reservoir (30) | 0.884 / 0.893 | 0.94 | 0.93 | 0.75 | 0.42 |
| GRU, recent (FIFO) (30) | 0.878 / 0.885 | 0.92 | 0.98 | 0.73 | 0.36 |
| GRU, surprise-prioritised (30) | 0.814 / 0.816 | 0.61 | 0.98 | 0.86 | 0.32 |
| peephole LSTM, no replay (30) | 0.890 / 0.905 | 0.86 | 1.04 | 0.77 | 0.38 |
| peephole LSTM, reservoir (30) | 0.901 / 0.857 | 0.87 | 0.97 | 0.60 | 0.67 |
| peephole LSTM, recent (FIFO) (30) | 0.887 / 0.808 | 0.86 | 0.96 | 0.58 | 0.57 |
| peephole LSTM, surprise-prioritised (30) | 0.809 / 0.766 | 0.38 | 1.03 | 0.80 | 0.45 |
| Mogrifier LSTM, no replay (30) | 0.866 / 0.746 | 0.84 | 1.12 | 0.59 | 0.41 |
| Mogrifier LSTM, reservoir (30) | 0.826 / 0.683 | 0.80 | 0.92 | 0.37 | 0.73 |
| Mogrifier LSTM, recent (FIFO) (30) | 0.782 / 0.641 | 0.67 | 0.68 | 0.20 | 0.64 |
| Mogrifier LSTM, surprise-prioritised (30) | 0.762 / 0.738 | 0.21 | 1.06 | 0.85 | 0.50 |
| transformer, no replay (30) | 0.784 / 0.630 | 0.81 | 0.51 | 0.21 | 0.50 |
| transformer, reservoir (30) | 0.773 / 0.654 | 0.68 | 0.65 | 0.26 | 0.75 |
| transformer, recent (FIFO) (30) | 0.739 / 0.640 | 0.58 | 0.47 | 0.30 | 0.57 |
| transformer, surprise-prioritised (30) | 0.770 / 0.730 | 0.23 | 1.13 | 1.34 | 0.53 |
| memory slots learned by gradient (MemAE), no replay (30) | 0.774 / 0.739 | 0.95 | 0.90 | 0.86 | 0.12 |
| memory slots learned by gradient (MemAE), reservoir (30) | 0.886 / 0.896 | 0.86 | 0.95 | 0.77 | 0.39 |
| memory slots learned by gradient (MemAE), recent (FIFO) (30) | 0.882 / 0.888 | 0.77 | 0.97 | 0.69 | 0.35 |
| memory slots learned by gradient (MemAE), surprise-prioritised (30) | 0.810 / 0.828 | 0.50 | 1.06 | 0.73 | 0.34 |
| fast slot memory written in one shot, no replay (30) | 0.878 / 0.872 | 0.94 | 1.07 | 0.52 | 0.36 |
| fast slot memory written in one shot, reservoir (30) | 0.888 / 0.859 | 0.91 | 1.02 | 0.57 | 0.54 |
| fast slot memory written in one shot, recent (FIFO) (30) | 0.889 / 0.853 | 0.91 | 1.10 | 0.56 | 0.51 |
| fast slot memory written in one shot, surprise-prioritised (30) | 0.830 / 0.763 | 0.57 | 0.98 | 0.71 | 0.55 |
| fast weights (delta rule), no replay (30) | 0.829 / 0.718 | 0.90 | 1.00 | 0.46 | 0.46 |
| fast weights (delta rule), reservoir (30) | 0.806 / 0.781 | 1.03 | 1.12 | 0.51 | 0.76 |
| fast weights (delta rule), recent (FIFO) (30) | 0.786 / 0.692 | 0.77 | 1.08 | 0.44 | 0.50 |
| fast weights (delta rule), surprise-prioritised (30) | 0.781 / 0.747 | 0.50 | 0.98 | 0.83 | 0.54 |

Paired against the connectome model at recovery 1e-4 (difference = network minus model), Holm over 108 comparisons over 30 seeds:

| network, replay | measure | difference | seeds network lower / higher | p | p (Holm) |
|---|---|---|---|---|---|
| autoencoder, no replay | alpha-MoO3 contrast ratio | +0.482 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| autoencoder, no replay | recall at first sighting | -0.098 | 26 / 1 | 6.22e-06 | 0.000174 |
| autoencoder, no replay | AUROC last block | -0.116 | 29 / 1 | 3.73e-09 | 2.68e-07 |
| autoencoder, reservoir | alpha-MoO3 contrast ratio | +0.505 | 1 / 29 | 3.73e-09 | 2.68e-07 |
| autoencoder, reservoir | recall at first sighting | +0.104 | 1 / 28 | 9.72e-06 | 0.000263 |
| autoencoder, reservoir | AUROC last block | -0.131 | 26 / 4 | 8.01e-08 | 4.89e-06 |
| autoencoder, recent (FIFO) | alpha-MoO3 contrast ratio | +0.432 | 3 / 27 | 2.05e-07 | 1.19e-05 |
| autoencoder, recent (FIFO) | recall at first sighting | -0.039 | 18 / 12 | 0.0386 | 0.425 |
| autoencoder, recent (FIFO) | AUROC last block | -0.175 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| autoencoder, surprise-prioritised | alpha-MoO3 contrast ratio | -0.028 | 18 / 12 | 0.229 | 1 |
| autoencoder, surprise-prioritised | recall at first sighting | -0.127 | 26 / 3 | 4.77e-06 | 0.000138 |
| autoencoder, surprise-prioritised | AUROC last block | -0.140 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| LSTM, no replay | alpha-MoO3 contrast ratio | +0.528 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| LSTM, no replay | recall at first sighting | -0.396 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| LSTM, no replay | AUROC last block | +0.036 | 1 / 29 | 1.3e-08 | 8.48e-07 |
| LSTM, reservoir | alpha-MoO3 contrast ratio | +0.503 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| LSTM, reservoir | recall at first sighting | -0.103 | 26 / 4 | 5.03e-05 | 0.000957 |
| LSTM, reservoir | AUROC last block | +0.005 | 12 / 18 | 0.49 | 1 |
| LSTM, recent (FIFO) | alpha-MoO3 contrast ratio | +0.461 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| LSTM, recent (FIFO) | recall at first sighting | -0.218 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| LSTM, recent (FIFO) | AUROC last block | -0.073 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| LSTM, surprise-prioritised | alpha-MoO3 contrast ratio | +0.001 | 18 / 12 | 0.584 | 1 |
| LSTM, surprise-prioritised | recall at first sighting | -0.346 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| LSTM, surprise-prioritised | AUROC last block | -0.107 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| GRU, no replay | alpha-MoO3 contrast ratio | +0.551 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| GRU, no replay | recall at first sighting | -0.587 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| GRU, no replay | AUROC last block | +0.023 | 5 / 25 | 7.06e-05 | 0.00127 |
| GRU, reservoir | alpha-MoO3 contrast ratio | +0.572 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| GRU, reservoir | recall at first sighting | -0.373 | 30 / 0 | 1.71e-06 | 9.4e-05 |
| GRU, reservoir | AUROC last block | +0.021 | 7 / 23 | 0.00134 | 0.0201 |
| GRU, recent (FIFO) | alpha-MoO3 contrast ratio | +0.551 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| GRU, recent (FIFO) | recall at first sighting | -0.425 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| GRU, recent (FIFO) | AUROC last block | +0.013 | 8 / 22 | 0.0106 | 0.138 |
| GRU, surprise-prioritised | alpha-MoO3 contrast ratio | +0.243 | 1 / 29 | 1.86e-08 | 1.17e-06 |
| GRU, surprise-prioritised | recall at first sighting | -0.473 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| GRU, surprise-prioritised | AUROC last block | -0.056 | 29 / 1 | 3.73e-09 | 2.68e-07 |
| peephole LSTM, no replay | alpha-MoO3 contrast ratio | +0.495 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| peephole LSTM, no replay | recall at first sighting | -0.407 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| peephole LSTM, no replay | AUROC last block | +0.033 | 3 / 27 | 2.55e-07 | 1.45e-05 |
| peephole LSTM, reservoir | alpha-MoO3 contrast ratio | +0.508 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| peephole LSTM, reservoir | recall at first sighting | -0.123 | 28 / 2 | 1.42e-05 | 0.000355 |
| peephole LSTM, reservoir | AUROC last block | -0.015 | 18 / 12 | 0.0803 | 0.723 |
| peephole LSTM, recent (FIFO) | alpha-MoO3 contrast ratio | +0.491 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| peephole LSTM, recent (FIFO) | recall at first sighting | -0.220 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| peephole LSTM, recent (FIFO) | AUROC last block | -0.064 | 25 / 5 | 1.42e-06 | 7.95e-05 |
| peephole LSTM, surprise-prioritised | alpha-MoO3 contrast ratio | +0.010 | 12 / 18 | 0.428 | 1 |
| peephole LSTM, surprise-prioritised | recall at first sighting | -0.341 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| peephole LSTM, surprise-prioritised | AUROC last block | -0.105 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| Mogrifier LSTM, no replay | alpha-MoO3 contrast ratio | +0.478 | 1 / 29 | 3.73e-09 | 2.68e-07 |
| Mogrifier LSTM, no replay | recall at first sighting | -0.382 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| Mogrifier LSTM, no replay | AUROC last block | -0.126 | 21 / 9 | 3.9e-05 | 0.000781 |
| Mogrifier LSTM, reservoir | alpha-MoO3 contrast ratio | +0.435 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| Mogrifier LSTM, reservoir | recall at first sighting | -0.054 | 16 / 12 | 0.0602 | 0.602 |
| Mogrifier LSTM, reservoir | AUROC last block | -0.189 | 29 / 1 | 3.73e-09 | 2.68e-07 |
| Mogrifier LSTM, recent (FIFO) | alpha-MoO3 contrast ratio | +0.307 | 1 / 29 | 5.59e-09 | 3.74e-07 |
| Mogrifier LSTM, recent (FIFO) | recall at first sighting | -0.144 | 29 / 1 | 2.21e-06 | 9.4e-05 |
| Mogrifier LSTM, recent (FIFO) | AUROC last block | -0.230 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| Mogrifier LSTM, surprise-prioritised | alpha-MoO3 contrast ratio | -0.159 | 28 / 2 | 9.31e-09 | 6.15e-07 |
| Mogrifier LSTM, surprise-prioritised | recall at first sighting | -0.291 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| Mogrifier LSTM, surprise-prioritised | AUROC last block | -0.133 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| transformer, no replay | alpha-MoO3 contrast ratio | +0.444 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| transformer, no replay | recall at first sighting | -0.284 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| transformer, no replay | AUROC last block | -0.242 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| transformer, reservoir | alpha-MoO3 contrast ratio | +0.317 | 1 / 29 | 1.3e-08 | 8.48e-07 |
| transformer, reservoir | recall at first sighting | -0.036 | 23 / 6 | 0.000371 | 0.00631 |
| transformer, reservoir | AUROC last block | -0.218 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| transformer, recent (FIFO) | alpha-MoO3 contrast ratio | +0.212 | 3 / 27 | 1.64e-07 | 9.67e-06 |
| transformer, recent (FIFO) | recall at first sighting | -0.220 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| transformer, recent (FIFO) | AUROC last block | -0.232 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| transformer, surprise-prioritised | alpha-MoO3 contrast ratio | -0.132 | 28 / 2 | 2.37e-05 | 0.000521 |
| transformer, surprise-prioritised | recall at first sighting | -0.261 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| transformer, surprise-prioritised | AUROC last block | -0.142 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| memory slots learned by gradient (MemAE), no replay | alpha-MoO3 contrast ratio | +0.584 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| memory slots learned by gradient (MemAE), no replay | recall at first sighting | -0.664 | 30 / 0 | 1.71e-06 | 9.4e-05 |
| memory slots learned by gradient (MemAE), no replay | AUROC last block | -0.133 | 24 / 6 | 1.6e-05 | 0.000383 |
| memory slots learned by gradient (MemAE), reservoir | alpha-MoO3 contrast ratio | +0.491 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| memory slots learned by gradient (MemAE), reservoir | recall at first sighting | -0.401 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| memory slots learned by gradient (MemAE), reservoir | AUROC last block | +0.024 | 8 / 22 | 0.000798 | 0.0128 |
| memory slots learned by gradient (MemAE), recent (FIFO) | alpha-MoO3 contrast ratio | +0.408 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| memory slots learned by gradient (MemAE), recent (FIFO) | recall at first sighting | -0.441 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| memory slots learned by gradient (MemAE), recent (FIFO) | AUROC last block | +0.017 | 4 / 26 | 1.82e-05 | 0.00042 |
| memory slots learned by gradient (MemAE), surprise-prioritised | alpha-MoO3 contrast ratio | +0.133 | 7 / 23 | 0.00277 | 0.0387 |
| memory slots learned by gradient (MemAE), surprise-prioritised | recall at first sighting | -0.450 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| memory slots learned by gradient (MemAE), surprise-prioritised | AUROC last block | -0.044 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| fast slot memory written in one shot, no replay | alpha-MoO3 contrast ratio | +0.574 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast slot memory written in one shot, no replay | recall at first sighting | -0.427 | 30 / 0 | 1.73e-06 | 9.4e-05 |
| fast slot memory written in one shot, no replay | AUROC last block | +0.000 | 13 / 17 | 0.529 | 1 |
| fast slot memory written in one shot, reservoir | alpha-MoO3 contrast ratio | +0.540 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast slot memory written in one shot, reservoir | recall at first sighting | -0.244 | 24 / 5 | 1.32e-05 | 0.000342 |
| fast slot memory written in one shot, reservoir | AUROC last block | -0.013 | 14 / 16 | 0.598 | 1 |
| fast slot memory written in one shot, recent (FIFO) | alpha-MoO3 contrast ratio | +0.543 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast slot memory written in one shot, recent (FIFO) | recall at first sighting | -0.277 | 29 / 1 | 2.11e-06 | 9.4e-05 |
| fast slot memory written in one shot, recent (FIFO) | AUROC last block | -0.019 | 13 / 17 | 0.543 | 1 |
| fast slot memory written in one shot, surprise-prioritised | alpha-MoO3 contrast ratio | +0.202 | 6 / 24 | 2.37e-05 | 0.000521 |
| fast slot memory written in one shot, surprise-prioritised | recall at first sighting | -0.234 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| fast slot memory written in one shot, surprise-prioritised | AUROC last block | -0.109 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| fast weights (delta rule), no replay | alpha-MoO3 contrast ratio | +0.532 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast weights (delta rule), no replay | recall at first sighting | -0.325 | 30 / 0 | 1.71e-06 | 9.4e-05 |
| fast weights (delta rule), no replay | AUROC last block | -0.154 | 28 / 2 | 1.86e-08 | 1.17e-06 |
| fast weights (delta rule), reservoir | alpha-MoO3 contrast ratio | +0.666 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast weights (delta rule), reservoir | recall at first sighting | -0.024 | 18 / 11 | 0.256 | 1 |
| fast weights (delta rule), reservoir | AUROC last block | -0.091 | 27 / 3 | 8.01e-08 | 4.89e-06 |
| fast weights (delta rule), recent (FIFO) | alpha-MoO3 contrast ratio | +0.400 | 0 / 30 | 1.86e-09 | 2.01e-07 |
| fast weights (delta rule), recent (FIFO) | recall at first sighting | -0.288 | 30 / 0 | 1.72e-06 | 9.4e-05 |
| fast weights (delta rule), recent (FIFO) | AUROC last block | -0.180 | 30 / 0 | 1.86e-09 | 2.01e-07 |
| fast weights (delta rule), surprise-prioritised | alpha-MoO3 contrast ratio | +0.132 | 8 / 22 | 0.012 | 0.145 |
| fast weights (delta rule), surprise-prioritised | recall at first sighting | -0.250 | 30 / 0 | 1.71e-06 | 9.4e-05 |
| fast weights (delta rule), surprise-prioritised | AUROC last block | -0.124 | 30 / 0 | 1.86e-09 | 2.01e-07 |

Replay design, mean over the nine networks (alpha-MoO3 contrast ratio / recall at first sighting / AUROC last block):

| replay | alpha-MoO3 contrast ratio | recall at first sighting | AUROC last block |
|---|---|---|---|
| no replay | 0.88 | 0.39 | 0.796 |
| reservoir | 0.87 | 0.65 | 0.804 |
| recent (FIFO) | 0.79 | 0.54 | 0.767 |
| surprise-prioritised | 0.41 | 0.48 | 0.765 |

## Time-course stop, S20 (flakes lost of 30)

Learning rate per arm chosen on seeds 101-105 (10 calibration flakes, 600 epochs, 16 units; mean flakes lost):

| arm | lr 3e-3 | lr 1e-2 |
|---|---|---|
| GRU | 4.00 | 3.60 |
| peephole LSTM | 1.60 | 2.20 |
| Mogrifier LSTM | 2.20 | 2.40 |
| transformer | 1.40 | 1.00 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | 3.20 | 4.20 |

### 3 calibration flakes (20 seeds)

| arm | lost of 30 | under-converted | over-selenized | final x_t error (held out) | trained or stored numbers |
|---|---|---|---|---|---|
| Kenyon-cell code, one spectrum | 2.45 | 2.20 | 0.25 | 0.033 |  |
| Kenyon-cell codes now and 14 s earlier (connectome model) | 2.35 | 2.25 | 0.10 | 0.027 |  |
| perceptron, one spectrum | 3.00 | 2.70 | 0.30 | 0.049 |  |
| LSTM (S20, tuned) | 4.15 | 3.65 | 0.50 | 0.043 |  |
| GRU | 6.50 | 5.95 | 0.55 | 0.047 | 3,586 |
| peephole LSTM | 4.00 | 3.65 | 0.35 | 0.039 | 4,754 |
| Mogrifier LSTM | 4.15 | 3.65 | 0.50 | 0.036 | 7,154 |
| transformer | 3.60 | 3.05 | 0.55 | 0.038 | 3,298 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | 4.75 | 4.20 | 0.55 | 0.028 | 13,273 |

Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:

| arm | reference | difference | seeds better / worse | p (Holm) |
|---|---|---|---|---|
| GRU | Kenyon-cell codes now and 14 s earlier (connectome model) | +4.15 | 0 / 19 | 0.00138 |
| peephole LSTM | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.65 | 2 / 12 | 0.0175 |
| Mogrifier LSTM | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.80 | 3 / 16 | 0.0148 |
| transformer | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.25 | 3 / 11 | 0.088 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | Kenyon-cell codes now and 14 s earlier (connectome model) | +2.40 | 2 / 17 | 0.00337 |
| LSTM (S20, tuned) | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.80 | 2 / 14 | 0.0115 |
| GRU | LSTM (S20, tuned) | +2.35 | 3 / 15 | 0.00448 |
| peephole LSTM | LSTM (S20, tuned) | -0.15 | 9 / 7 | 1 |
| Mogrifier LSTM | LSTM (S20, tuned) | +0.00 | 5 / 7 | 1 |
| transformer | LSTM (S20, tuned) | -0.55 | 8 / 4 | 0.318 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | LSTM (S20, tuned) | +0.60 | 5 / 12 | 0.563 |

### 10 calibration flakes (20 seeds)

| arm | lost of 30 | under-converted | over-selenized | final x_t error (held out) | trained or stored numbers |
|---|---|---|---|---|---|
| Kenyon-cell code, one spectrum | 1.90 | 1.65 | 0.25 | 0.034 |  |
| Kenyon-cell codes now and 14 s earlier (connectome model) | 1.00 | 1.00 | 0.00 | 0.025 |  |
| perceptron, one spectrum | 2.10 | 1.80 | 0.30 | 0.038 |  |
| LSTM (S20, tuned) | 2.90 | 2.65 | 0.25 | 0.038 |  |
| GRU | 4.40 | 4.10 | 0.30 | 0.039 | 3,586 |
| peephole LSTM | 2.70 | 2.35 | 0.35 | 0.036 | 4,754 |
| Mogrifier LSTM | 2.70 | 2.50 | 0.20 | 0.033 | 7,154 |
| transformer | 1.50 | 1.40 | 0.10 | 0.032 | 3,298 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | 4.35 | 4.05 | 0.30 | 0.031 | 35,953 |

Paired tests on flakes lost (negative = first arm loses fewer), Holm over this table:

| arm | reference | difference | seeds better / worse | p (Holm) |
|---|---|---|---|---|
| GRU | Kenyon-cell codes now and 14 s earlier (connectome model) | +3.40 | 0 / 19 | 0.00139 |
| peephole LSTM | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.70 | 0 / 15 | 0.00447 |
| Mogrifier LSTM | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.70 | 0 / 15 | 0.00447 |
| transformer | Kenyon-cell codes now and 14 s earlier (connectome model) | +0.50 | 1 / 8 | 0.0875 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | Kenyon-cell codes now and 14 s earlier (connectome model) | +3.35 | 0 / 18 | 0.00184 |
| LSTM (S20, tuned) | Kenyon-cell codes now and 14 s earlier (connectome model) | +1.90 | 0 / 16 | 0.00345 |
| GRU | LSTM (S20, tuned) | +1.50 | 3 / 14 | 0.0875 |
| peephole LSTM | LSTM (S20, tuned) | -0.20 | 6 / 5 | 1 |
| Mogrifier LSTM | LSTM (S20, tuned) | -0.20 | 8 / 5 | 1 |
| transformer | LSTM (S20, tuned) | -1.40 | 13 / 2 | 0.0262 |
| memory-augmented regressor (GRU keys, memory of calibration windows) | LSTM (S20, tuned) | +1.45 | 4 / 14 | 0.0875 |
