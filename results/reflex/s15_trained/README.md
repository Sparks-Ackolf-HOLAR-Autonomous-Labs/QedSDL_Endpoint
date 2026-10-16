# S15: the connectome model trained on its fixed wiring (12 Oct 2026)

Question: once the band-to-glomerulus
input map and the read-out are learned from data, does the measured FlyWire PN-to-KC wiring train faster,
need fewer examples, or transfer better to a new material than shuffled, random sparse or dense wiring?

## Set-up
- Fixed: PN-to-KC synapse counts, top-5 % KC threshold. Trained (Adam, 400 steps): 56 x 56
  band-to-glomerulus map (initialised to the name-order map of S1-S14) and one read-out weight per KC.
- Data (`src/reflex/s15_pool.py`): 4,200 in-situ Raman probes of MoS2 -> MoSSe and 4,200 of WS2 -> WSSe
  selenization (lattice kMC, Mo and W parameter sets, 3 draws per recipe, S4L temperature model; W lines from
  Harris et al. 2023, WSSe E mode at 330 cm-1 and WS2 LA(M) at 176 cm-1 assumed). Fail rate 29 % (Mo), 22 % (W).
- Task a: stop prediction. Train on 25-400 Mo probes, score on 1,200 probes of held-out recipes; transfer to
  WS2 (zero-shot, fine-tuned on 25 W probes, or 25 W probes from scratch).
- Task b: S11 Ar+ irradiation, read-out regresses fluence from 1 or 3 calibration flakes, 30 test flakes.
- Arms: fly (measured), shuf, rand, dense (Gaussian all-to-all), fly_fixmap (map held at name order),
  logistic regression, 5-nearest-neighbour; ridge, kNN and the LA(M)/A1g ratio for task b.
- 20 seeds; paired Wilcoxon, Holm-corrected. Code `src/reflex/s15_trained.py`, `analyze_s15_s16.py`;
  ran on the DGX Spark GPU. Tables: `s15_summary.md`.

## Findings
1. **Trained, the measured wiring is no better than shuffled or random sparse wiring.** Held-out AUROC at
   25 / 100 / 400 probes: measured 0.946 / 0.980 / 0.988, shuffled 0.943 / 0.981 / 0.989, random 0.942 / 0.981
   / 0.990 (all p = 1). Training speed is the same (all peak by 25 steps).
2. **Sparse beats dense with few examples.** Dense random expansion 0.884 at 25 probes (measured +0.062, 18/20,
   p = 0.004) and 0.921 at 50 (p = 0.002). The sparse, low-degree expansion is the useful prior, not the
   partner map.
3. **Learning the input map adds nothing.** The model with the name-order map held fixed is as good or slightly
   better (n = 100: 0.986 against 0.980, p = 0.15).
4. **Logistic regression on the 56 bands is the best stop classifier** (0.966 at 25 probes; ahead of the
   trained model at 100 probes by 0.009, p = 0.02); 5-nearest-neighbour is the worst (0.839 at 25).
5. **Transfer MoS2 -> WS2.** Zero-shot the measured wiring scored 0.765 against 0.694 shuffled and 0.705 random
   (14/20 seeds, p = 0.5 and 0.21 after Holm): a hint, not a result. Fine-tuning on 25 WS2 probes gave 0.886,
   the same as training on them from scratch (0.884), so nothing learned on MoS2 carried over; logistic
   regression fine-tuned reached 0.941 (p = 0.023).
6. **Irradiation stop: a trained read-out is the best dose reader so far.** With 3 calibration flakes the
   trained model hit the target on 99 % of flakes with 4.0 % mean error, against 6.0 % (ridge) and 6.3 %
   (kNN), p <= 2e-4; with 1 flake 4.9 % against 8.5 % and 9.2 % (p <= 0.003). S11's code lookup reached 5.5 %.
   Wiring made no difference (shuffled 4.3 %, random 4.2 %, dense 4.2 %; p >= 0.61).

## Meaning
Training on the fixed wiring, the proper test of whether the measured wiring is a useful prior (Lappalainen
et al. 2024 for vision), gives the same answer as the untrained tests: the benefit is the sparse expansion
with its degree statistics; which projection neuron connects to which Kenyon cell does not help on these
materials tasks. The expansion with a trained read-out is a strong low-data regressor for the dose task.
