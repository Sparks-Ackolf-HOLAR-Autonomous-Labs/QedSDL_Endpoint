# S14: why the left mushroom body beat the right (11 Oct 2026)

Question: S12 and S13 found the left side's connectome model beats the right on the facility
stream in all three animals (FAFB +0.042, MaleCNS +0.018, BANC +0.009). Is that the Kenyon-cell (KC) wiring, the
projection-neuron (PN) input, or something else? Second question: what does the nerve cord do here?

## Arms (per animal; same wirings, streams, KC code and memory as S13; 20 seeds x 2 streams)
| Arm | What it changes |
|---|---|
| left, right | measured wiring of one side (= S13) |
| left_nm, right_nm | multiglomerular PNs silenced (in S12-S13 they read 3 random bands, drawn separately per side) |
| left_gp, right_gp | measured wiring, glomerulus-to-band map permuted at random per seed, the same permutation for both sides (S1-S13 used alphabetical glomerulus order) |
| LL, RR | per-side configuration model: each KC keeps its partner count and synapse counts, partners redrawn from its own side's PNs in proportion to PN strength |
| LR, RL | transplant: left KC wiring on right PN input, and the reverse |

Code: `src/reflex/s14_left_right.py`, `analyze_s14.py`, `run_s14.sh`; ran on the DGX Spark.
Full tables and 27 tests per stream (Holm-corrected): `summary.md`.

## Result (facility stream, AUROC, mean of 20 seeds)
| Arm | FAFB | BANC | MaleCNS |
|---|---|---|---|
| left (measured) | 0.886 | 0.820 | 0.888 |
| right (measured) | 0.844 | 0.810 | 0.871 |
| left, multiglomerular PNs silenced | 0.878 | 0.818 | 0.884 |
| right, multiglomerular PNs silenced | 0.839 | 0.808 | 0.867 |
| **left, random band map** | **0.837** | **0.812** | **0.846** |
| **right, random band map** | **0.841** | **0.822** | **0.850** |
| LL (left KCs, left PNs, redrawn) | 0.869 | 0.817 | 0.868 |
| LR (left KCs, right PNs) | 0.878 | 0.814 | 0.866 |
| RL (right KCs, left PNs) | 0.865 | 0.824 | 0.868 |
| RR (right KCs, right PNs, redrawn) | 0.875 | 0.823 | 0.870 |

## Findings
1. **The left advantage comes from the band order, not from the animal.** With a random glomerulus-to-band map
   (same for both sides), left and right are equal in all three animals (left - right: FAFB -0.004, BANC -0.009,
   MaleCNS -0.004; p >= 0.62). With the alphabetical map of S1-S13, FAFB left led by +0.042 (20/20 seeds).
2. **Neither side's KC wiring nor its PN input is better.** After redrawing partners within each side, LL = RR in
   every animal (p = 1), and the two-way split gives no KC effect and no PN effect (all p >= 0.53).
3. **Multiglomerular PNs are not the cause.** Silencing them changes the side gap by at most 0.003; they carry only
   2-4 % of PN-KC synapses.
4. **What the measured map does:** the exact pairing of glomeruli on each KC (which glomeruli converge) lined up well
   with the alphabetical band order on the left side and badly on the right (FAFB right measured 0.839 against its
   side shuffle 0.875, p = 5e-5). Because the band order is arbitrary, this is a coincidence of the test design.
5. **The alphabetical order was a lucky draw for the left side.** A random band order lowered the left side's
   facility AUROC by 0.04-0.05 in FAFB and MaleCNS. Results that compare measured wiring with a shuffle under one
   fixed band order (S1-S13) should be read with this in mind; the ranking against nearest neighbour and z-score is
   unchanged, because those do not use the map.
6. Janus line: all side differences stay within 0.006, as before.

## What the nerve cord does (BANC v888 and MaleCNS v1.0, pairs with >= 3 synapses; `nerve_cord.md`, `s14_nerve_cord.py`)
| Dataset | Cell group | input from olfactory PNs | input from ascending (nerve-cord) neurons | output onto descending neurons |
|---|---|---|---|---|
| BANC | Kenyon cells (4,552) | 60.7 % | 0.03 % (13 KCs) | 0.02 % |
| BANC | dopaminergic neurons (301) | 3.0 % | 0.30 % | 0.29 % |
| BANC | output neurons (104) | 0.9 % | 0.25 % | 1.4 % (27 MBONs) |
| MaleCNS | Kenyon cells (4,064) | 30.9 % | 0.02 % (25 KCs) | 0.00 % |
| MaleCNS | dopaminergic neurons (340) | 1.1 % | 0.25 % | 0.15 % |
| MaleCNS | output neurons (97) | 0.2 % | 0.08 % | 1.7 % (34 MBONs) |

- The connectome model uses only PN-to-KC wiring inside the brain, so the nerve cord plays no part in it. In BANC
  and MaleCNS the nerve-cord files were present, but no nerve-cord neuron enters the model.
- In the animal, the nerve cord reaches the mushroom body mainly through **ascending neurons** that feed the
  dopaminergic teaching neurons and the output neurons (body state, gustatory and mechanosensory signals; the
  largest single source onto DANs in both animals is type AN19B019). The mushroom body talks back to the nerve cord
  through **descending neurons**, reached directly by about a third of MBONs, but with under 2 % of their output.
- In materials terms: the mushroom body is the memory that says "this reading is new"; the nerve cord is the body
  that carries out the action and reports its own state. The closest analogue at an instrument is a few signals from
  the stage itself (furnace temperature, laser power, a fault line) gating whether a new reading should teach the
  memory, which is the DAN role tested as context input in S8. Direct nerve-cord input is a few tenths of a percent
  of the mushroom body's synapses, so building the model from a whole-CNS connectome does not change its wiring.
