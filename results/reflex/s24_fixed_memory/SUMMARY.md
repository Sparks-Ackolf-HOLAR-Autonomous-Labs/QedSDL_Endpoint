# S24: does the connectome model beat fixed-memory baselines? (20 Oct 2026)

Two experiments. (1) Flagging, with every fixed-memory baseline given the model's whole footprint instead of
only its 5,177 plastic weights. (2) One connectome model against the cheapest engineered stack that covers
the same four jobs. Full tables are in `tables.md` and every number is in `summary.json`. Per-seed files:
`a_s1b/`, `a_long/`, `b_stack/`; timing and bytes: `timing.json`. Paired two-sided Wilcoxon tests, not
corrected for multiplicity (with 10 seeds the smallest possible p is 0.002; with 20 it is 1.9e-6).

**Budgets.** Plastic weights only (as in S22): 20.7 kB. Packed for a microcontroller (uint16 KC and PN indices,
uint8 synapse counts and KC weights; `edge_compare.FlySparse.mcu_bytes`): 151.9 kB. As run in NumPy: 372.9 kB.
Baselines store float32 values (S22 accounting). The 1,356-spectrum reservoir is the 152 kB budget at float16.
"Tuned ≤ budget" means the reservoir size (or autoencoder width and buffer) chosen on seeds 101–105 among
those whose bytes fit the budget.

## Experiment 1: memory-matched flagging (AUROC, new vs normal)

Reference: the connectome model with slow recovery (fly_rec, r = 1e-4). Routed arms see the sample log and are
compared with the model that is told the family (fly_ctx2: context input, gain 2, 5 % active; tuned: 2.5 %).
Time is ms per decision on one DGX Grace core; Pi 5 estimates are in the section below.

| arm | bytes | S1b, 10 seeds (S22b) median [IQR], p vs ref | S1b, 20 seeds (s1b_multi) median [IQR], p vs ref | long stream, 10 seeds median [IQR], p vs ref | ms |
|---|---|---|---|---|---|
| model, no recovery | 373 kB | 0.861 [0.840, 0.875], 0.002 | 0.859 [0.848, 0.876], 1.9e-6 | 0.866 [0.865, 0.868], 0.002 | 0.132 |
| **model, recovery 1e-4 (ref)** | 373 kB | **0.876** [0.865, 0.886] | **0.881** [0.868, 0.886] | **0.900** [0.898, 0.902] | 0.133 |
| nearest neighbour, 92 spectra (S22 "equal memory") | 20.6 kB | 0.929 [0.914, 0.938], 0.002 | 0.923 [0.908, 0.934], 1.9e-6 | 0.915 [0.899, 0.939], 0.037 | 0.006 |
| nearest neighbour, tuned ≤ budget (184 S1b; 92 long) | 41.2 / 20.6 kB | 0.921 [0.910, 0.936], 0.002 | 0.927 [0.911, 0.940], 1.9e-6 | 0.915 (as above) | 0.008 |
| nearest neighbour, fills 152 kB (678) | 152 kB | 0.891 [0.883, 0.916], 0.006 | 0.895 [0.882, 0.916], 0.008 | 0.902 [0.897, 0.905], 0.62 | 0.020 |
| nearest neighbour, 152 kB at float16 (1,356) | 304 kB f32 | 0.873 [0.862, 0.891], 0.85 | 0.872 [0.860, 0.894], 0.39 | 0.877 [0.873, 0.881], 0.002 (model ahead) | 0.039 |
| nearest neighbour, fills 0.37 MB (1,664) | 373 kB | 0.878 [0.853, 0.889], 0.49 | 0.866 [0.857, 0.890], 0.064 | 0.870 [0.860, 0.876], 0.002 (model ahead) | 0.047 |
| autoencoder, tuned ≤ budget (16 hidden; buffer 578 S1b; none long) | 152 / 22.4 kB | 0.849 [0.829, 0.878], 0.020 | 0.856 [0.823, 0.879], 1.3e-4 | 0.787 [0.769, 0.856], 0.002 | 0.253 / 0.153 |
| autoencoder 64 hidden, fills 152 kB (buffer 287) | 152 kB | 0.739 [0.727, 0.750], 0.002 | 0.743 [0.726, 0.755], 1.9e-6 | 0.740 [0.727, 0.746], 0.002 | 0.272 |
| autoencoder 16 hidden, fills 0.37 MB (buffer 1,564) | 373 kB | 0.835 [0.821, 0.875], 0.010 | 0.844 [0.821, 0.867], 3.6e-5 | 0.693 [0.684, 0.715], 0.002 | 0.253 |
| per-band z-score | 448 B | 0.840 [0.831, 0.861], 0.002 | 0.844 [0.838, 0.862], 1.9e-6 | 0.845 [0.843, 0.846], 0.002 | 0.004 |
| *with the sample log:* model told the family (ref for routed rows) | 374 kB | 0.978 [0.973, 0.979] | 0.977 [0.972, 0.979] | 0.977 [0.976, 0.978] | 0.142 |
| model told the family, tuned 2.5 % active | 374 kB | 0.983 [0.980, 0.983], 0.002 | 0.982 [0.977, 0.983], 1.9e-6 | 0.981 [0.980, 0.982], 0.002 | 0.137 |
| per-band z-score routed by family | 1.8 kB | **0.990** [0.989, 0.993], 0.002 | **0.990** [0.987, 0.992], 1.9e-6 | **0.995** [0.995, 0.995], 0.002 | 0.004 |
| nearest neighbour routed by family, best size (184) | 41.2 kB | 0.957 [0.945, 0.965], 0.002 | 0.959 [0.946, 0.965], 1.9e-6 | 0.954 [0.952, 0.958], 0.002 | 0.005 |
| nearest neighbour routed by family, fills 152 kB / 0.37 MB | 151 / 373 kB | 0.955 / 0.919, both 0.002 | 0.948 / 0.917, both 1.9e-6 | 0.938 / 0.918, both 0.002 | 0.008 / 0.014 |

Routed z-score against the tuned model told the family: +0.008, 18 of 20 seeds, p = 1.3e-5 (S1b); +0.014,
10 of 10, p = 0.002 (long). Reproduction: model 0.861 / 0.866 / 0.900, the S1b 92-spectrum nearest neighbour 0.929, the fixed-buffer
autoencoder 0.769 and the z-score 0.845 equal S22a and S22b to three decimals. On the long stream the 92-spectrum
nearest neighbour gives 0.915 against S22a's 0.909, because the reservoir uses a per-seed random draw (seed 3 + seed, as in S22b).

## Experiment 2: one structure against an engineered stack

Paper harnesses imported unchanged, test seeds 1–20 (as in the paper). The connectome model is one measured-wiring
model at its defaults (5 % active, no gain control): one PN→KC drive per spectrum, with a plain code for jobs ii–iv
and the context code for job i. The furnace read-out from the Kenyon-cell code is new here; the paper's nerve-cord
estimator uses a different wiring (BANC), so it is not part of one structure. Its penalty (0.1) was chosen on seeds 101–105.

| job (score) | connectome model | engineered stack | p | other arms |
|---|---|---|---|---|
| i flag new phase / wrong sample (AUROC) | 0.966 / 0.991 (context, gain 2) | z-score routed by family **0.986 / 0.997** | 1.9e-6 / 1.9e-6 | routed nearest neighbour 92: 0.949 / 0.968; 184: 0.956 / 0.963 |
| ii stop a failing run (flakes lost of 30) | 3.55 (plain); 3.30 with the paper's gain-control setting | logistic regression C = 1: 3.90; C = 10 (S23 tuned): 3.40 | plain vs C 10: 0.45; paper setting vs C 1: 0.065, vs C 10: 0.75 | nearest neighbour 3.90 (vs paper setting p = 0.013) |
| iii end irradiation at target (share on target; mean abs error) | **0.952; 5.5 %** (code lookup) | ridge on bands 0.883; 8.2 % | 0.0085; 0.0042 | nearest calibration spectrum 0.928 (p = 0.14); 6.4 % (p = 1.7e-4) |
| iv end point at the furnace (flakes lost of 30) | 0.00 (ridge on the Kenyon-cell code) | ridge on bands, default penalty: 0.10 | 0.32 | nerve-cord read-out (S23): 0.10 default, 0.0 tuned |

| system | packed bytes | of which | µs per spectrum, all four jobs (one Grace core) |
|---|---|---|---|
| connectome model, one structure | **259.9 kB** | wiring 146.7 kB; job i novelty weights 5.2 kB + context mask 1.3 kB; job ii compartments 1.3 kB (bits); job iii 123 calibration codes 64.0 kB; job iv read-out 41.4 kB | **234** |
| engineered stack (routed z-score, logistic regression, ridge, ridge) | **9.4 kB** | z-score 1.8 kB; logistic regression 228 B + 30 labelled probes 6.7 kB (refit); ridge 228 B; ridge 456 B | **9.2** |
| stack with routed nearest neighbour (184) for i and nearest calibration spectrum for iii | 42.3 kB | 20.6 kB + 7.0 kB + 14.3 kB + 456 B | 19.0 |

Learning: the logistic-regression refit takes 1.47 ms once per completed run (the harness refits it at every probe:
4.1 ms per check); the model's one-shot compartment write takes 1.5 µs. Stored spectra are float16 and regression
weights float32 (edge_compare accounting).

## What the paper can and cannot claim

The model does not beat fixed-memory baselines once they get its memory. The flagging gaps come from method, not budget.
A reservoir nearest neighbour that fills the model's 152 kB still beats the slow-recovery model on the facility
stream (0.895 against 0.881, p = 0.008) and ties it on the long stream (p = 0.62). Only when it is forced to fill the
whole 0.37 MB does it fall level (S1b, p = 0.064) or below (long, 0.870 against 0.900, p = 0.002). Even then the cause
is over-filling: a larger reservoir scores lower, and a nearest neighbour allowed to choose its size within either
budget beats the model in all three settings (S1b 0.927, p = 1.9e-6; long 0.915, p = 0.037).

Fixed-buffer autoencoders lose at every budget, and the pooled z-score loses at 448 B. With the sample log, the model
told the family beats routed nearest neighbours and routed autoencoders at any budget. It loses to the per-family
z-score, which reaches 0.990–0.995 against the tuned model's 0.981–0.982 at 1/200 of the memory (p ≤ 1.3e-5). The
paper's sentence that the routed z-score "matches the model once tuned" should become "exceeds".

The Discussion's "one structure does all jobs more cheaply than an engineered stack" is false as measured. The cheapest
stack covering the four jobs takes 9.4 kB and 9.2 µs per spectrum, against the model's 260 kB packed and 234 µs: 28 times
less memory and 25 times less time. The stack is better at flagging, level at stopping a failing run (tuned logistic
regression 3.40 against 3.55, p = 0.45) and level at the furnace.

What the paper can claim: one fixed representation at one default setting serves all four jobs, including the furnace
end point read from the same Kenyon-cell code (0 flakes lost, new here), and is never far behind. Its one clear
advantage is dose: 0.952 on target and 5.5 % error against ridge (0.883, 8.2 %; p = 0.0085 and 0.004) and against the
nearest calibration spectrum on error (6.4 %, p = 1.7e-4, but not on share on target, p = 0.14). It also keeps a fixed,
bounded footprint that fits a microcontroller. It cannot claim lower cost than an engineered stack, or a flagging
advantage over fixed-memory baselines at matched memory. The 3.30 failing-run result uses a second (gain-control) code,
so it is not the same single structure as the other jobs.

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

## Notes

- Python: `~/venvs/mat_theory` has no torch, scikit-learn or pandas. All runs used `~/venv`, which produced
  S22. `~/flywire_v783` no longer exists, so the scripts ran with `HOME=<workspace>/s24_home`, which links
  `flywire_v783 → ~/flywire_v783`. The PN→KC wiring and PN map are served from one cached npz (`s24_wcache.py`;
  output byte-identical to the uncached run).
- Raspberry Pi 5: the Pi could not be reached during S24, so every timing is one pinned Grace core
  (taskset, NumPy 2.5.3). Pi 5 times are estimated from matched Pi/Grace pairs (section above), not measured.
- The first launch (20 workers at about 2.1 GB each, all reading the FlyWire tables with pandas) helped push the shared DGX
  out of memory at about 20:00 MDT alongside two vLLM engines. The machine rebooted at 20:34. The rerun used 14 workers
  at 0.8 GB each, under `watchdog.sh`. S1b seeds 1–5 were finished before the crash and kept.
- Scripts: `src/reflex/s24_fixed_memory.py`, `s24_stack.py`, `s24_timing.py`, `s24_wcache.py`, `analyze_s24.py`.
  Run files: `run_a.sh`, `run_b.sh`, `a_jobs.txt`.
