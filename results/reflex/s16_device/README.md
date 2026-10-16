# S16: what a synapse device must provide to run the connectome model (12 Oct 2026)

Question: turn the connectome model into a
device specification for a 2D-semiconductor memristor or ferroelectric-transistor build. How many
conductance levels, how much device spread and read noise, how long a written state must last, and how
many writes per device?

## Set-up
- Two arrays: the fixed projection-neuron (PN) to Kenyon-cell (KC) crossbar (occupied cross points =
  FlyWire v783 PN-KC pairs) and the 5,177 plastic KC-to-output weights (the familiarity memory).
- Non-idealities, one at a time from the floating-point model, then two combinations:
  crossbar levels (1 = connected or not), output levels (depression with stochastic rounding),
  device-to-device spread (log-normal gain per device), read noise (Gaussian, every reading),
  retention loss (written weight relaxes toward 1 at rate rho per reading).
- Streams: S1 Janus line and S1b four-family facility, 4,000 readings, 20 seeds each, measured wiring.
- Trained model: the S15 selenization model (400 Mo probes) quantised after training.
- Code: `src/reflex/s16_device.py`, `analyze_s15_s16.py`; ran on the DGX Spark.
- Full tables with Holm-corrected paired Wilcoxon tests: `s16_summary.md`, `s16_summary.json`.

## Findings
1. **The fixed crossbar can be binary.** Storing each PN-KC pair as connected or not leaves the Janus line
   unchanged (+0.001, p = 1) and raises the facility AUROC (+0.016, p = 1e-4): which cells connect, not the
   synapse count, carries the benefit (the same message as measured = shuffle). A trained read-out fitted to
   the exact counts needs 8 crossbar levels (-0.002, p = 0.12; 4 levels -0.025).
2. **The plastic output synapse sets the specification.** 2 / 4 / 8 / 16 / 32 / 64 levels lose 0.082 / 0.045 /
   0.027 / 0.018 / 0.014 / 0.009 AUROC (Janus; facility the same within 0.001), all p = 5e-5. About 64 levels
   (analogue-like) are needed to stay within 0.01. The trained read-out (signed weights) needs 17 levels for
   -0.005 and 33 for no loss.
3. **Device spread up to 20 % is tolerated** (Janus -0.003, facility -0.005, not significant); 50 % costs
   0.013 and 0.037.
4. **Read noise matters only when several families share the code**: facility 2 % -0.003, 5 % -0.019,
   10 % -0.034; Janus line at most -0.004.
5. **Retention is not a constraint, and a volatile device helps.** Relaxation at 1e-5 to 1e-3 per reading
   raises AUROC (Janus +0.013 to +0.041, facility +0.008 to +0.019), because it acts as the S5 recovery;
   only 1e-2 per reading hurts the facility (-0.027).
6. **Endurance.** With analogue updates each output device is written on 5 % of readings (about 5,000 pulses
   per 1e5 spectra); with 16-64 levels and stochastic rounding 0.07-0.10 % (about 70-100 per 1e5).
7. **A typical 2D memristor** (8 crossbar levels, 16 output levels, 20 % spread, 5 % read noise, retention
   1e-5): Janus line 0.951 (unchanged), facility -0.031 (mainly read noise), trained model -0.010.

## Limits
- Device effects are applied to the model in software; no device was fabricated or measured.
- Stochastic rounding of the depression is assumed to be available (for example by a probabilistic
  programming pulse); deterministic rounding to coarse levels would stop depression at the first level.
- Energy per write and per read is not modelled.
