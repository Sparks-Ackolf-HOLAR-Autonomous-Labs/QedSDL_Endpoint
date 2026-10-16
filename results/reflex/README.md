# Reflex-layer tests T1-T5 (10 Oct 2026)

All tests use Janus MoSSe (MoS2 -> MoSSe selenization, lattice kinetic Monte Carlo with MACE-MP-0
energetics, `src/janus/kmc.py`) or literature and public-database values. No CLSLab unit or CLSLab
reading is used. Tuned settings (T2 readout step, T3 avoid threshold) were chosen on separate
tuning seeds (0 and 100-103) before the 20 test seeds (1-20) were run. Every test runs the
mushroom body on the measured FlyWire 783 wiring (`fly`), a degree-preserving shuffle (`fly_shuf`)
and random sparse wiring with the same synapse count (`fly_rand`).

Tables with interquartile ranges and paired Wilcoxon tests: `summary.md` (`src/reflex/analyze.py`).
Run scripts: `src/reflex/t1_janus_faults.py`, `t2_epochs.py`, `t3_avoid.py`, `t4_network.py`,
`edge_timing.py`; all ran on the DGX Spark.

## Findings

1. **T1 fault flagging (fail).** Familiarity of a readout given its recipe separates faulty furnace
   runs from clean ones about as well as a nearest-neighbour distance or an isolation forest
   (AUROC 0.74 against 0.75 and 0.73) and worse than a Gaussian-process residual (0.80, p = 0.02).
   At 5 % false alarms the fly recalls the most faults (0.39 against 0.32 for the GP), mostly
   readout glitches (0.79); it misses physical faults that move the flake to another valid
   state (furnace offset 0.22, empty Se source 0.03). It scores a reading in 2.2 ms against 32 ms
   for the refitted GP.
2. **T2 epochs (fail on accuracy; no forgetting in any model worth the name).** Learning WSSe after
   MoSSe barely changes the fly's MoSSe error (+0.010 RMSE), but its error is about twice the GP's
   (0.21 against 0.09-0.10) and only a quarter better than predicting the mean (0.28). The GP told
   the epoch keeps and improves its MoSSe map (-0.009); the GP refitted on the new system alone
   forgets (+0.062). The two Janus systems are close (their truth maps differ by 0.15 RMSE), so
   this test is mild for every model.
3. **T3 failure avoidance (fail).** A one-shot avoid compartment written by failures does not reduce
   lost furnace runs: GP alone 7.75 of 30, GP with the fly veto 7.80 (p = 1). Constrained Bayesian
   optimisation halves them (4.35, p = 2e-4) and also finds better recipes.
4. **T4 reflex layer in a network (fail).** With a hidden furnace fault from round 4, the fly check
   kept back about one faulty reading per campaign (8.4 of 9 shared, p = 0.007 against no check)
   and took the furnace out of service in 1 of 20 seeds. The GP-residual check with the same
   repeat-and-strike policy kept back 2.8 (6.2 shared, p = 0.004 against the fly) and took the
   furnace out in 9 of 20, mostly for the temperature offset; an exhausted Se source is close to
   invisible to every detector.
5. **T5 literature values (fail).** 101 Janus MoSSe values reported in 30 papers and C2DB
   (Raman A1 and E modes, PL energy, computed gap, lattice constant; `t5_literature/`), read in
   random order with 20 % injected reading errors (changed digit or decimal shift, nm for eV or
   Angstrom, an MoS2 or MoSe2 value given as MoSSe). The fly familiarity check reached AUROC 0.72
   and caught 23 % of errors at 5 % false alarms; a robust z-score on the values read so far
   reached 0.82 and 60 % (p = 0.002), a nearest-value distance 0.83 and 52 %. The vision-language
   reading step was not run: values were taken from text snippets, C2DB pages and tables.
6. **Wiring.** Measured, shuffled and random wiring give the same results in every test
   (T1 AUROC 0.740 / 0.740 / 0.731; T2-T4 within seed scatter). Whatever the model does comes
   from sparse expansion, not from this fly's PN->KC partners.
7. **Edge compute (pass).** On monsterpi (Raspberry Pi 5, 8 GB, NumPy 1.24, one thread, nice 19,
   USB bridge running) one fault score takes 48-69 ms with 50-1000 stored readings, the avoid
   values of all 1400 recipes take 18 ms, the one-off code table 4.3 s, peak memory 138 MB. The
   DGX takes 9-13 ms. A furnace run lasts minutes, so the Pi is fast enough by three orders of
   magnitude. A GP refit per reading was not timed on the Pi.

## Interpretation of T1-T5

Across T1-T5 the connectome model is fast and small, but no better than standard tools at any of the
five jobs, and its measured wiring adds nothing over shuffled wiring. These results do not support
the claim that the connectome model does a job the GP or LLM cannot. They support a narrow role: a 138 MB, 50 ms novelty check that runs on the controller next
to the instruments and is best at catching corrupted readouts (glitches), as a low-cost first filter
before a GP-residual check.

## S1 and S1b: spectra fed one band per glomerulus (added 10 Oct 2026, evening)

Simulated Raman spectra of selenized MoS2 flakes (S1) and of four material families in one stream
(S1b), 4,000 readings, bounded familiarity memory, 20 seeds each (`src/reflex/s1_spectra.py`,
tables in `s1_spectra/summary.md` and `s1b_multi/summary.md`). The connectome model reached AUROC 0.95
(S1) and 0.86 (S1b), against 0.74 in T1; a per-band z-score was best in S1 (0.98), an equal-memory
nearest neighbour in S1b (0.92). Measured and shuffled wiring beat random sparse wiring in both
(p = 6e-6, 1e-4). The revised tests are S2-S8 below.

## S7 one model vs several (10 Oct 2026)
`s7_swarm/summary.md`: copies voting +0.012 AUROC at 4x memory; one model per material family
0.98 at equal memory, but a per-family z-score reaches 0.989, so the gain is the routing context.
