# S23: one tuning protocol for every method (fly model, small neural networks, classical baselines)

Added 18 Oct 2026. Question: are the paper's comparisons fair when the small neural networks and the
baselines get the same tuning as the connectome model, and do the paper's own numbers reproduce?

**Protocol** (the paper's own rule for its settings): every method gets a small grid of its own settings; the
setting is chosen on separate seeds **101–105**, frozen, and tested on seeds **1–20** inside the paper's
harnesses (`src/reflex`, imported unchanged). Each job also re-runs published arms as a reproduction check.

| job | harness | methods and grids | score |
|---|---|---|---|
| 1 stop a failing run | `s4_insitu.py` | fly stop rule: sparsity {2.5, 5, 10 %} × gain control {off, β 0.5, β 1.0}; logistic regression C {0.01, 0.1, 1, 10, 100}; small NN (MLP classifier on the probe, trained on completed runs) hidden {16, 32, 64×64} × L2 α {1e-4, 1e-2, 1, 10} | flakes lost of 30 (lower) |
| 2 flag a surprise | `s8_context.py` streams, `s2_s3.run` | fly with sample-log context: sparsity × context gain {1, 2, 4}; small NN = `s19_online_nets.Online("ae46")` (equal memory), unrouted and routed by the sample log (one per family, as the paper's routed baselines), Adam lr {1e-3, 3e-3, 1e-2, 3e-2} | AUROC (higher) |
| 3 hit a dose | `s11_irradiation.py` | fly code lookup: sparsity × gain; ridge λ {1e-6, 1e-4, 1e-2, 1, 100}; small NN (MLP regressor bands → fluence on the 3 calibration flakes) same grid as job 1 | share of flakes on target (higher) |
| furnace local stop | `s18a_furnace.py` | ridge readout λ {1e-4 … 1} on the 56 bands and on the BANC nerve-cord wiring; small NN (MLP regressor on the bands) same grid as job 1 | flakes lost of 30 (lower) |

## Reproduction of the paper (default settings, seeds 1–20)

| arm | this run | paper |
|---|---|---|
| job 1 fly, plain default (`gp+fly`) | 3.55 | 3.55 |
| job 1 logistic regression | 3.90 | 3.90 |
| job 2 small NN unrouted, lr 1e-2 | 0.86 | 0.86 |
| job 2 fly with context (5 %, gain 2) | 0.975 | 0.973 |
| job 3 fly code lookup | 0.952 | 0.952 |
| job 3 ridge | 0.883 | 0.883 |
| furnace, ridge on bands | 0.10 | 0.15 |
| furnace, nerve-cord wiring | 0.10 | 0.20 |

All but the two furnace arms match to the reported precision. The furnace arms differ by one to two flakes in
600 (estimator error matches the paper, 0.051); treated as numerical noise between library versions.

## Results with every method tuned the same way (test seeds 1–20)

| job | fly model | small NN | classical baseline |
|---|---|---|---|
| 1 flakes lost | 3.55 (chosen: 5 %, gain off = default) | 3.60 (64×64, α 1e-4 = default) | logistic regression **3.40** (C 10) |
| 2 AUROC with sample log, all / stack | **0.980 / 0.983** (2.5 %, gain 2) | 0.970 / 0.973 (routed, lr 1e-2; 4× memory: 21,016 numbers vs 5,177) | — |
| 2 AUROC without sample log | 0.86 (paper) | 0.86 (lr 1e-2) | — |
| 3 share on target | **0.972** (10 %, gain off) | 0.938 (64×64, α 1) | ridge 0.883 (λ 1e-4 = default) |
| furnace flakes lost | **0.0** (nerve-cord readout λ 0.1) | **0.0** (64×64, α 1) | ridge 0.10 (λ 1e-2 = default) |

Job 1 differences are within noise (fly vs small NN at defaults: fewer flakes lost in 6 seeds, equal in 11, more in 3,
Wilcoxon p = 0.25; tuned fly vs tuned logistic regression: fewer in 2, equal in 14, more in 4, p = 0.45). The paper's 3.30 for job 1 used `fly_best` (gain β 1.0), chosen by the paper's tuning rule
on the S2 novelty streams; tuned on job 1's own seeds, the plain default is chosen instead.

## Out of the box (defaults) against tuned

| job | fly default → tuned | small NN default → tuned | spread of the small NN over its grid (tuning seeds) |
|---|---|---|---|
| 1 flakes lost | 3.55 → 3.55 | 3.60 → 3.60 | 3.0 – 5.8 |
| 2 AUROC with log | 0.975 → 0.980 | 0.945 (lr 1e-3) → 0.970 | 0.94 – 0.98 routed; 0.54 – 0.85 unrouted (unrouted default 0.818) |
| 3 share on target | 0.952 → 0.972 | 0.788 → 0.938 | 0.65 – 0.97 (on test seeds 0.67 – 0.97) |
| furnace flakes lost | 0.10 → 0.0 | 15.8 → 0.0 | 0.2 – 30 |

Reading: tuned on equal terms, the fly model is best or tied in every job (job 1 tied within noise, a small
network level at 4× memory in job 2, the fly model ahead in job 3, all near zero at the furnace). The larger
difference is robustness of the defaults: the fly model's defaults are within 0.02 of its best in every job,
while the small networks range widely over their grids and need the search to be competitive.

Per-decision cost (job 1 stop check, one Grace core): small NN 21–27 ms (refit at every probe), logistic
regression ~21 ms in this harness, fly model a fraction of a millisecond (paper, `edge_timing.py`).

## Files

- `src/reflex/s23_fair_nn.py` — small networks, jobs j1 (stop), j2 (station), j3 (dose); `python s23_fair_nn.py j1|j2|j3`
- `src/reflex/s23_fair_fly.py` — fly model and classical baselines, t1 (stop), t2 (station), t3 (dose), t4 (furnace);
  `python s23_fair_fly.py t1|t2|t3|t4` (t4 needs the BANC cord cache; set `S23_BANC_CACHE` if it is not at
  `work/p4_s18/banc_cord.npz`)
- `src/reflex/s23_log_mlp.py` — seed 20 of job 1 with the small-NN stop rule, run by run (used in the talk slides)
- `nn_fair_j1.json`, `nn_fair_j2.json`, `nn_fair_j3.json` — small networks: tuning grid means, chosen setting, per-seed test results
  (j2, j3 also `test_every_setting`: the test-seed mean of every setting, for the out-of-the-box column)
- `fly_fair_t1.json` … `fly_fair_t4.json` — fly model and baselines: tuning and test means for every setting, per-seed values
- `nn_e1.json`, `nn_e2.json` — first look (job 1 small NN at defaults; furnace small NN with α 1), consistent with the above
- `s4_seed20_mlp.json` — run-by-run outcomes for seed 20 (small NN stop rule)

Run with the repository venv (torch on CPU: `CUDA_VISIBLE_DEVICES=""`), 20 worker processes; each job takes
1–15 minutes on the DGX Spark.

Out-of-the-box values are test-seed means at each method's default setting (scikit-learn / PyTorch defaults for
the small networks: α 1e-4, Adam lr 1e-3; the paper's defaults for the fly model and the baselines).

## Wrong-sample flag with the small networks (added 18 Oct 2026)

`src/reflex/s23_wrong_nn.py`, streams and mislabel draw of `s6_one_module` (3 % of normal flakes logged under the
wrong family), seeds 1–20, small NN at the job 2 learning rate (1e-2). `nn_wrong_sample.json`.

| arm | any new phase | wrong sample | stack | stored numbers |
|---|---|---|---|---|
| fly module with context, gain 2 (reproduction of `s6_one_module`) | 0.966 | 0.991 | 0.954 | 5,177 |
| small NN unrouted (never sees the log) | 0.860 | 0.509 | 0.650 | 5,254 |
| small NN routed by the log (one per family) | 0.966 | 0.987 | 0.958 | 21,016 |
| per-family z-score | 0.986 | 0.997 | 0.982 | 448 |
| fly model without the log (context off; `S23_ONLY=fly_plain`, `fly_plain_wrong_sample.json`) | 0.860 | 0.511 | 0.629 | 5,177 |

Without the log a wrong sample is invisible by construction (chance) for the fly model and the small network alike. With it the routed small networks match the
fly module at 4× the memory; the per-family z-score is best, as in the paper.

## When to look: one fixed probe, or a stop at any time (added 18 Oct 2026)

`s4_insitu` takes one in-situ spectrum at `PROBE_FRAC = 0.25` of each anneal and may stop there (a stop costs 0.25 anneal).

**Fixed probe at other times** (`src/reflex/s23_probe_time.py`, `probe_time.json`; `s4_insitu.run_arm` unchanged, seeds 1–20):

| probe at | fly (plain): lost / false stops / furnace | logistic regression | nearest neighbour |
|---|---|---|---|
| 0.10 | 3.55 / 4.05 / 22.5 | 3.90 / 2.65 / 23.5 | 4.20 / 3.70 / 22.4 |
| 0.25 (paper) | 3.55 / 4.05 / 23.6 | 3.90 / 2.50 / 24.6 | 3.90 / 3.25 / 24.0 |
| 0.50 | 3.50 / 4.15 / 25.7 | 3.75 / 2.25 / 26.4 | 3.95 / 2.70 / 26.3 |
| 0.75 | 3.45 / 3.45 / 28.0 | 3.50 / 1.50 / 28.5 | 3.75 / 2.60 / 28.2 |

BO planner alone: 7.75 lost, 30 anneals. Flakes lost barely depend on the probe time; furnace time does.

**Stop at any time** (`src/reflex/s23_stop_anytime.py`, `stop_anytime.json`): the same runs (identical recipe choice,
trajectory, outcome and Raman model) probed at 0.1, 0.2, … 0.9; stop at the first checkpoint that says fail; one memory per
checkpoint, written only from completed runs. Small NN = MLP 64×64, α 1e-4 (the j1 tuned setting).

| arm | lost (one look at 0.25 → any time) | false stops | furnace time | gap |
|---|---|---|---|---|
| fly (plain) | 3.55 → 2.65 | 4.05 → 5.95 | 23.6 → 20.7 | 0.12 → 0.17 |
| logistic regression | 3.90 → 2.35 | 2.50 → 5.60 | 24.6 → 21.2 | 0.02 → 0.10 |
| nearest neighbour | 3.90 → 2.30 | 3.25 → 7.70 | 24.0 → 19.4 | 0.03 → 0.15 |
| small NN | 3.60 → 2.35 | 2.10 → 7.05 | 24.9 → 20.1 | 0.03 → 0.12 |

Most stops come at the first check (0.1). Every method loses fewer flakes and uses less furnace time, at about twice the
false stops and a larger gap to the best recipe (stopped runs teach the planner less). The fly model has no edge here.

Stop decisions as a classifier (positive = run would fail; pooled over 20 seeds, all 30 runs, including the first 3 that no
rule may stop; the run mix differs by method because stops change the planner's next recipes):

| arm, look | sensitivity | specificity | precision | NPV | accuracy | failure prevalence |
|---|---|---|---|---|---|---|
| fly, 0.25 | 0.56 | 0.82 | 0.53 | 0.83 | 0.75 | 0.27 |
| logistic regression, 0.25 | 0.55 | 0.88 | 0.65 | 0.83 | 0.79 | 0.29 |
| nearest neighbour, 0.25 | 0.55 | 0.85 | 0.59 | 0.82 | 0.76 | 0.29 |
| small NN, 0.25 | 0.57 | 0.90 | 0.69 | 0.84 | 0.81 | 0.28 |
| fly, any time | 0.67 | 0.73 | 0.48 | 0.86 | 0.71 | 0.27 |
| logistic regression, any time | 0.70 | 0.75 | 0.49 | 0.88 | 0.73 | 0.26 |
| nearest neighbour, any time | 0.73 | 0.64 | 0.45 | 0.86 | 0.67 | 0.29 |
| small NN, any time | 0.72 | 0.67 | 0.46 | 0.86 | 0.69 | 0.28 |

## Under-selenized recipes (added 18 Oct 2026)

`src/reflex/s23_under.py`, `recipe_outcomes.npz`: 8 replicates of every recipe with `s4_insitu.simulate_probe`. Destroyed
(the s4 loss rule) 29.4 %; kept with the top layer < 50 % converted 43.8 %; kept with < 90 % converted 63.7 %; kept and
≥ 90 % converted (good Janus) 6.8 %. Job 1 counts only destroyed flakes as lost; an under-selenized flake can be re-annealed.

### The paper's fly setting in the same tests (added 18 Oct 2026)

`S23_ARMS=gp+fly_best` (`H.default_full_cfg`, the fly setting of the paper's job 1 result), `probe_time_flybest.json`,
`stop_anytime_flybest.json`. One look at 0.25 reproduces the paper (3.30 lost).

| fly, paper's setting | lost | false stops | furnace | gap |
|---|---|---|---|---|
| one look at 0.10 | 3.65 | 2.70 | 23.4 | 0.05 |
| one look at 0.25 (paper) | 3.30 | 1.80 | 24.9 | 0.07 |
| one look at 0.50 | 3.40 | 2.05 | 26.7 | 0.08 |
| one look at 0.75 | 3.40 | 2.25 | 28.3 | 0.08 |
| any time (0.1 … 0.9) | 2.60 | 4.65 | 22.1 | 0.14 |

Classifier view (as above): one look at 0.25 — sensitivity 0.60, specificity 0.92, precision 0.74, NPV 0.86, accuracy 0.83;
any time — 0.66, 0.79, 0.52, 0.87, 0.76. At one look the paper's setting has the highest precision of all rules (logistic
regression 0.65, small NN 0.69); with stops at any time it makes the fewest false stops but loses more flakes than the others (2.60
against 2.30–2.35). The talk slides (6.9, 6.10, 6.11, 8.2) use this setting for job 1 and say so; 8.4 reports defaults (plain,
3.55) and job-1 tuning (plain chosen on seeds 101–105).

## Shifted Raman axis and unequal band gains (added 19 Oct 2026)

`src/reflex/s23_raman_axis.py`, `raman_axis.json`. Nine phases of the multi-family streams coded by the plain connectome
model (measured wiring); a reference code at the true axis against readings drifted by d cm^-1; self overlap = share of
active cells shared with the own reference; identified = own reference is the closest (of 9).

| drift (cm^-1) | 56 bands | broad bands, s.d. 7 | broad bands, s.d. 12 | 56 bands + offset search (41 corrections) |
|---|---|---|---|---|
| 2 | 0.84, 9 | 0.88, 9 | 0.93, 9 | 1.00, 9 |
| 4 | 0.60, 8 | 0.76, 9 | 0.85, 9 | 1.00, 9 |
| 7 (one band) | 0.23, 4 | 0.59, 7 | 0.75, 8 | 1.00, 9 |
| 10 | 0.15, 4 | 0.48, 6 | 0.66, 8 | 1.00, 9 |
| 15 | 0.10, 1 | 0.32, 4 | 0.51, 4 | 1.00, 9 |

Overlap of the closest other phase: 0.26 (56 bands), 0.39 and 0.47 (broad bands). A fixed per-band gain error (log-normal,
20 draws): 5 %: 0.97, 180/180; 10 %: 0.93, 180/180; 20 %: 0.87, 180/180; 30 %: 0.83, 179/180.

## Where these results are in the paper (19 Oct 2026)

`paper4_connectome_agent/draft/draft.tex`: methods paragraphs "Equal tuning of every method" and "Shifted Raman axis and
unequal band gains"; stop timing and recipe outcome shares in "In situ stop of selenization"; results in the stop, dose,
facility and furnace subsections and in the new subsection on the Raman axis; annex tables A14 (out of the box and tuned)
and A15 (stop timing and calls). Paired tests computed from the per-seed files here: tuned dose lookup against the tuned
small network p = 0.006 (ahead in 13, level in 4 of 20) and against ridge p = 0.001; stop at any time against one probe,
connectome model 2.60 against 3.30 flakes lost p = 0.005, false stops 4.65 against 1.80 p < 0.001; facility, tuned model
(context gain 2, 2.5 %) 0.980 against routed small networks 0.970, ahead in 20 of 20 seeds, p = 2e-6.
