# S18a-c: the nerve-cord arrangement as an instrument-side controller

Builds on the S18 structure check (`../s18_nerve_cord_roles/`, now BANC v888 and MaleCNS v1.0).
Simulated Janus MoS2 selenization and the S1/S7 Raman streams only; no CLSLab units or readings.
Full numbers and paired tests: `summary.md` (from `src/reflex/analyze_s18.py`).

| Test | Code | Question |
|---|---|---|
| S18a | `src/reflex/s18a_furnace.py` | Does a local stop at the furnace (command down, reflex at the instrument, one status message up) beat a planner in the loop, and does the BANC nerve-cord wiring help as that reflex? |
| S18b | `src/reflex/s18b_status.py` | Does a one-bit status report ("laser power changed") repair the false alarms after a laser drop that the S2 mechanisms could not? |
| S18c | `src/reflex/s18c_pi_timing.py` | What does the nerve-cord reflex cost on the Raspberry Pi 5 next to the instruments? |

## S18a (20 seeds x 30 flakes)
Reduced furnace kinetics with assumed rates: top-layer exchange, a slower bottom-layer exchange that starts
as the top completes, a 10 s chamber purge after the valve closes, flake-to-flake rate spread (log-normal,
s.d. 0.4). The window between "top layer done" (x_t >= 0.90) and "bottom layer spoiled" (x_b > 0.10) is about
100 s at the median rate. Stop rule: three readings in a row past x_t >= 0.95 or x_b >= 0.06, estimated
from the literature-anchored in situ probe of S4L by a ridge fitted on three calibration flakes.

| Arm | flakes lost of 30 |
|---|---|
| open loop (valve time from the median rate) | 8.60 |
| planner in the loop, 2 s / 10 s / 60 s per turn | 0.15 / 0.45 / 10.65 |
| same rule at the instrument | 0.15 |
| nerve-cord reflex, measured / shuffled / random wiring | 0.20 / 0.25 / 0.25 |

- The local stop removed the losses of the open-loop recipe (-8.45, Holm p = 0.0006) and of a planner that
  needs 60 s per turn (-10.5, Holm p = 0.0006). Against a 2-10 s planner it made no significant difference.
  The local arrangement matters when the planner's turn is comparable to the process window; it also sends
  one message per run instead of 73.
- The nerve-cord wiring (VNC sensory -> intrinsic -> motor, 681,580 synapse entries, BANC v888) gave the same
  conversion estimate as a ridge on the 56 bands (error 0.052 against 0.051) and the same losses; measured =
  shuffled = random (p = 0.32, Holm 1).

## S18b (20 seeds per stream)
| Detector | Janus line: false alarms / recall | facility: false alarms / recall |
|---|---|---|
| connectome model, no status | 0.092 / 0.75 | 0.098 / 0.56 |
| connectome model, status gate (re-set threshold) | 0.016 / 0.47 | 0.026 / 0.39 |
| connectome model, status as context + gate | 0.000 / 0.00 | 0.000 / 0.00 |
| z-score, no status | 0.018 / 0.77 | 0.020 / 0.47 |
| z-score, reset + gate | 0.004 / 0.56 | 0.003 / 0.24 |
| nearest neighbour, reset + gate | 0.008 / 0.21 | 0.008 / 0.09 |

- The status gate cut the model's false alarms after the drop by about 0.07 (Holm p < 1e-5) but cost
  recall (0.27 and 0.17, Holm p < 0.001): it moves along the same trade-off, it does not remove it. The
  per-band z-score without any status already had fewer false alarms and higher recall on the Janus line.
- Used as a context input, the status bit gives the readings after the change a fresh familiarity memory;
  within 400 readings that memory had not settled, so the re-set threshold was above every later score
  (no alarms at all). The facility AUROC of this arm (0.962) comes from the family context it also
  carries (S8), not from the status bit.
- The gate is an engineered mechanism and works the same for any detector; nothing here needs the connectome.

## S18c
| Machine | nerve-cord stop | ridge on bands | memory |
|---|---|---|---|
| Raspberry Pi 5 (monsterpi) | 2.33 ms | 0.002 ms | 6.1 MB as a synapse list; does not fit a microcontroller |
| DGX Grace core | 1.29 ms | 0.0005 ms | |

## Reading
The nerve cord's arrangement (one command down, a stop decided at the instrument, one status message up) is
worth copying when the planner is slow relative to the process, as a 60 s language-model turn is for a
100 s selenization window. The nerve cord's wiring is not: an engineered estimator does the same job a
thousand times faster in a thousandth of the memory. This matches the mushroom-body result (S13-S15): what
transfers from the connectome is the architecture and its statistics, not the partner map.
