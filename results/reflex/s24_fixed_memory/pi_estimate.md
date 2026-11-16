
## Raspberry Pi 5: estimated, not measured

The Pi could not be reached from the compute server during S24. These times scale the Grace-core
times above by Pi/Grace ratios of the same operations measured earlier on both machines with the same scripts: sparse code
2.30 (edge_compare; nerve-cord read-out 1.81), small vector ops 5.3 (z-score, logistic, ridge 5.1, 5.4, 5.4; band ridge stop 3.8),
nearest-neighbour scan 4.4 at 92 spectra and 5.6 at 10,000 (log-linear in between). The Pi ran NumPy 1.24.2 and the
DGX 2.5.3, so the ratios include the NumPy version. Autoencoders (PyTorch) have no Pi analogue and are not estimated. `s24_pi_estimate.py`, `pi_estimate.json`.

| arm | Grace ms (measured) | Pi 5 ms (estimate) | range |
|---|---|---|---|
| fly | 0.132 | 0.303 | 0.238–0.303 |
| fly_rec | 0.133 | 0.306 | 0.240–0.306 |
| fly_ctx2 | 0.142 | 0.326 | 0.256–0.326 |
| zscore | 0.004 | 0.022 | 0.016–0.023 |
| zscore_fam | 0.004 | 0.022 | 0.016–0.023 |
| knn_n92 | 0.006 | 0.026 | 0.026–0.026 |
| knn_n184 | 0.008 | 0.038 | 0.036–0.038 |
| knn_n678 | 0.020 | 0.099 | 0.089–0.099 |
| knn_n1356 | 0.039 | 0.198 | 0.172–0.198 |
| knn_n1664 | 0.047 | 0.239 | 0.206–0.239 |
| knnfam_n184 | 0.005 | 0.021 | 0.021–0.021 |
| knnfam_n678 | 0.008 | 0.036 | 0.035–0.036 |
| knnfam_n1664 | 0.014 | 0.067 | 0.062–0.067 |

| serving all four jobs | Grace µs (measured) | Pi 5 µs (estimate) | range |
|---|---|---|---|
| connectome model | 233.5 | 811 | 422–821 |
| engineered stack | 9.2 | 49 | 35–50 |
| stack with nearest neighbours | 19.0 | 84 | 73–106 |

The ranking does not change on the Pi. The engineered stack serves all four jobs about 17× faster than the model (range 8–23×), against 25× measured on a Grace core; the model's code slows less on the Pi (×1.8–2.3) than small vector operations (×3.8–5.4).
