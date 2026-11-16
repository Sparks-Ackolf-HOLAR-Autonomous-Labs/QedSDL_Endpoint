# S25: which jobs need memory, and one generic model or specialists at the edge (21 Oct 2026)

Harnesses, seeds and environment as in S24 (`src/reflex`, test seeds 1–20, 10 for the 100k streams; `~/venv`; wiring
from the S24 cache). Full tables are in `tables.md` and every number is in `summary.json`. Per-seed files: `a/` (part A), `b/` (part B);
timing and bytes: `timing.json`. Paired two-sided Wilcoxon tests; in part A, Holm-corrected within each job.
Checks: online arms reproduce S24 and the paper exactly. Seed-1 dose and furnace curves at the paper's calibration size equal
the S24 per-seed values.

## A. Does each job need memory?

Conditions: **frozen** learns during calibration only (flagging: the first 300 readings; stop: the first C completed runs);
**reset k** clears memory every k readings; **online** is the paper's setting. Dose and furnace end point learn only from their
calibration flakes and receive no labels afterwards, so they are frozen by construction (frozen = online).

**Flagging, AUROC median.** Holm p, frozen against online, over 27 tests.

| method | S1b, 20 seeds: frozen / reset 1,000 / online (p) | long 100k, 10 seeds: frozen / online (p) | shifting stream, 10 seeds: frozen / online (p) |
|---|---|---|---|
| connectome model | 0.902 / 0.836 / 0.859 (5e-5) | 0.928 / 0.866 (0.037) | **0.703 / 0.820** (0.037) |
| model, recovery 1e-4 | 0.904 / 0.838 / 0.881 (2e-4) | 0.931 / 0.900 (0.037) | **0.706 / 0.849** (0.037) |
| model told the family | 0.991 / 0.940 / 0.977 (5e-5) | 0.996 / 0.977 (0.037) | 0.955 / 0.965 (0.037) |
| z-score | 0.864 / 0.841 / 0.844 (0.010) | 0.857 / 0.845 (0.22) | 0.669 / 0.698 (0.037) |
| z-score routed by family | 0.993 / 0.975 / 0.990 (2e-4) | 0.996 / 0.995 (0.037) | 0.953 / **0.979** (0.037) |
| nearest neighbour, 184 | 0.985 / 0.900 / 0.927 (5e-5) | 0.985 / 0.903 (0.037) | 0.944 / 0.919 (0.037) |
| nearest neighbour routed, 184 | 0.987 / 0.939 / 0.959 (5e-5) | 0.987 / 0.954 (0.037) | 0.964 / 0.958 (0.22) |
| small NN (autoencoder) | 0.850 / 0.772 / 0.854 (0.7) | 0.839 / 0.787 (0.7) | 0.827 / 0.802 (0.7) |

A reset every 100 readings is the worst condition for every method (S1b 0.51–0.86).

**Shifting stream, alarm shares, median.** Threshold: 95th percentile of routine scores in readings 300–5,000.

| method | accepted MoTe2 (should be low): frozen / online | routine after 3 cm-1 drift: frozen / online | rare phase back after 6x10^4: frozen / online | genuinely new, late recall: frozen / online |
|---|---|---|---|---|
| connectome model | 1.00 / 0.00 | 0.83 / 0.00 | 1.00 / 0.25 | 0.88 / 0.25 |
| model, recovery 1e-4 | 1.00 / 0.00 | 0.82 / 0.02 | 1.00 / 1.00 | 0.87 / 0.47 |
| model told the family | 1.00 / 0.00 | 0.78 / 0.00 | 1.00 / 0.45 | 0.99 / 0.32 |
| z-score routed | 0.00 / 0.10 | 0.71 / 0.15 | 1.00 / 1.00 | 0.99 / 0.98 |
| nearest neighbour routed, 184 | 1.00 / 0.00 | 0.41 / 0.10 | 1.00 / 1.00 | 0.99 / 0.96 |

**Laser drop (S1b), readings until false alarms return to the pre-drop level (median; seeds that never recover within 2,000).**

| method | frozen | online |
|---|---|---|
| connectome model | 1,151 (17 of 20 never) | 953 (0 never) |
| model told the family | 1,036 (19 never) | 958 (1 never) |
| z-score / nearest neighbour, pooled | 0 | 0 (never above the level) |
| z-score routed | never (20) | 1,166 (14 never) |

**Failing-run stop, flakes lost of 30 (20 seeds).** The GP planner alone loses 7.75. Holm p over 12 tests.

| method | frozen after 3 runs | after 6 | after 10 | online | reset every 10 runs | p, frozen 3 / frozen 10 vs online | runs to 95 % of benefit |
|---|---|---|---|---|---|---|---|
| connectome model | 5.95 | 3.95 | 3.50 | 3.55 | 4.80 | 0.011 / 1 | 10 |
| model, paper's gain setting | 5.90 | 4.30 | 3.35 | 3.30 | 4.85 | 0.019 / 1 | 10 |
| logistic regression C 10 | 5.30 | 3.75 | 3.60 | 3.40 | 4.80 | 0.055 / 1 | 10 |
| nearest neighbour | 5.50 | 3.95 | 4.00 | 3.90 | 4.90 | 0.15 / 1 | 6 |
| small NN (MLP) | 5.20 | 4.10 | 3.70 | 3.60 | 4.95 | 0.049 / 1 | 10 |

### Which jobs need memory

- **Detection itself does not need memory.** On the stationary streams, a flagger that stops learning after 300 readings
  is better than the same flagger learning online. Frozen is higher in 15 of the 18 stationary-stream tests (Holm p ≤ 0.037),
  and the other three are ties (the small autoencoder on both streams, the pooled z-score on the long stream). Online memory habituates to new events.
- **Memory is needed when "normal" changes.** On the shifting stream, a frozen flagger alarms on every reading of a phase the
  facility has accepted (1.00 for all but the routed z-score and the autoencoder) and on up to 83 % of routine readings after a
  3 cm-1 axis drift (7–83 % across methods). Online memory brings both to 0–18 %.
  For the connectome model and the z-scores, frozen then loses on AUROC (0.70 against 0.82–0.85 for the model, p = 0.037).
- **Memory has a price on a changing stream.** The plain online model also habituates to genuinely new events (late recall 0.25)
  and treats a rare phase returning after 6x10^4 readings as familiar (0.25). Recovery r = 1e-4 keeps the returns (1.00) and half
  the recall (0.47). The routed z-score keeps both (late recall 0.98) at 0.10–0.15 alarms on accepted and drifted routine.
- **Instrument changes:** area-normalised pooled methods need no memory to absorb a laser drop (no excess alarms). The connectome
  model needs online memory for it: about 950 readings to settle, and frozen it mostly never recovers.
- **Failing-run stop** needs a handful of labelled outcomes, not ongoing memory. Ten completed runs reach 95 % of the benefit
  for most methods, and frozen after 10 runs equals online (Holm p = 1). Fewer labels cost flakes: frozen after 3 runs, 5.2–6.0
  lost against 3.3–3.9 online. Clearing the memory every 10 runs costs about one flake.
- **Dose and furnace end point** need only their calibration set.

In short, memory is for tracking what counts as normal (accepted phases, drift, instrument changes) and for collecting the
few labels of the stop job. It does not make detection itself better, and an unbounded familiarity memory slowly costs recall.

## B. One generic model or specialists on the edge

G = one generic connectome model at defaults (one drive, read out per job). P = one tuned connectome model per job, sharing the
wiring (S23 settings). E = engineered specialists (routed z-score, logistic regression, ridge, ridge).

| | G generic | P per-job connectome | E engineered |
|---|---|---|---|
| flag new phase / wrong sample (AUROC) | 0.966 / 0.991 | 0.973 / 0.992 | **0.986 / 0.997** |
| failing-run stop (flakes lost) | 3.55 | **3.30** | 3.40 (C 10) |
| dose (share on target; error) | 0.952; 5.5 % | **0.972; 5.1 %** | 0.883; 8.2 % |
| furnace end point (flakes lost) | 0.00 | 0.00 | 0.10 |
| packed bytes | 260 kB | 324 kB (764 kB with four separate wirings) | **9.4 kB** |
| µs per spectrum, all four jobs (one Grace core) | 222 | 395 | **9.2** |
| settings tuned (candidates) | **1 (5)** | 9 (41) | 3 (15) |
| labels to add the dose job (n to 95 %) | 82 spectra | **41** | 82 (and a lower ceiling, 0.88) |
| labels to add the furnace end point (n to 95 %) | **45 readings** | **45** | 360 |
| readings of a new family to reach 95 % (flagging) | **10** | **10** | 20 (z-score routed) |
| laser drop: deployed models degraded | 2 of 4: flag (21 % false alarms at first, back to the pre-drop level after a median 953 readings, all seeds) and dose (0.952 → 0.918, p = 0.005) | 2 of 4: flag (as G) and dose (0.972 → 0.948, p = 0.012) | 3 of 4: flag (routed z-score: 21 % at first, not back within 2,000 readings in 14 of 20 seeds), dose (0.883 → 0.745, p = 1e-4), furnace (0.10 → 1.55 lost, p = 0.003) |
| laser drop: dose / furnace recovered by adding post-drop spectra? | dose no (up to 41 added); furnace not affected | as G | no (up to 41 dose, 180 furnace added) |
| failing-run stop after the drop (memory kept) | 3.65 (p = 0.16) | 3.30 (p = 1) | 3.50 (p = 0.16) |
| Pico (264 kB SRAM), all in RAM | no (283 kB; fits any job set without both dose and furnace) | no | **yes** |
| Pico, wiring and stored codes in flash | yes (30 kB RAM) | yes (53 kB) | yes |
| Pico 2 (520 kB SRAM), all in RAM | yes | yes (shared wiring) / no (four wirings) | yes |

For the dose and furnace recalibration test, the spectra of one new calibration flake were added to the original calibration set.
A calibration made only at the new intensity was not tested.

### One generic model or several specialists on the edge, and when each wins

On a station whose jobs are fixed and well calibrated, the engineered specialists win on cost by a wide margin:
9.4 kB and 9.2 µs per spectrum for all four jobs, against 260 kB and 222 µs for the generic model. They fit a Pico entirely in
RAM, they are the best flaggers (0.986 / 0.997), and they are level on the failing-run stop and the furnace. They lose only on
dose (0.883 against 0.952).

The generic model wins on what happens after deployment:
- **Adding a job:** a new read-out on the existing Kenyon-cell code needs 45 labelled furnace readings to reach 95 % of its
  result, against 360 for a new ridge on the bands. A new material family is flagged well after 10 readings, against 20.
- **Tuning:** it needs one setting tuned across all four jobs (5 candidates, against 15 for the specialists).
- **Instrument changes:** after a laser drop, its flag recovers on its own (median 953 readings, every seed) and its dose
  read-out loses 0.03 on target; the stop and furnace read-outs are unaffected. Among the engineered models, the routed z-score
  stays above its pre-drop alarm level for 2,000 readings in 14 of 20 seeds. The band ridges degrade for dose (−0.14 on target)
  and the furnace (+1.45 flakes lost), and neither is repaired by adding up to 41 or 180 spectra from the new intensity to the
  old calibration set. A pooled z-score would not be affected by the drop, but it flags worse (0.84).
- **Fit:** it still fits a Pico if the wiring and stored codes live in flash (30 kB RAM), and a Pico 2 entirely in RAM.

Tuning one connectome model per job (P) buys little: +0.007 on flagging, 0.25 fewer flakes lost on the stop and +0.02 on dose.
For that it pays 1.8× the time, 25 % more memory (3× with separate wirings) and 41 tuning candidates instead of 5.

The choice therefore turns on how much the station will change. Use engineered specialists for a stable instrument, fixed jobs,
plentiful labels or the tightest memory. Use one generic connectome model where jobs and material families are added over the
station's life with few labels, where the instrument drifts or is serviced, or where dose-type read-outs matter. Per-job tuned
connectome models are not worth their cost.

## Notes

- Memory safety: at most 12 workers (8 for part A, 4 for part B), about 0.8 GB each, 6–10 GB in all. `watchdog.sh` would have
  stopped the S25 workers below 30 GB available; it never fired (about 78 GB available at every check). The Qwen3-4B vLLM job
  (Desktop/Agents) was left running.
- The shifting stream is new (generator in `s25_memory.stream_shift`). Its MoTe2 lines are the S1b "new" MoTe2 lines, now a
  routine fifth material.
- Timing: one pinned Grace core (taskset). G is re-timed here (222 µs; 234 in S24). E is S24's measurement.
- Scripts: `src/reflex/s25_memory.py` (part A), `s25_generic.py` (part B), `s25_timing.py`, `analyze_s25.py`.
  Run files: `run_a.sh`, `run_b.sh`, `a_jobs.txt`.
