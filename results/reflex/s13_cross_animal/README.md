# S13: the connectome model built from three animals (11 Oct 2026)

Question: does the connectome model behave the same when its projection-neuron
(PN) to Kenyon-cell (KC) wiring comes from a different fly, of a different sex, and from one side of the
brain or both?

## Data (all public, no login)
| Animal | Release | Source | Files used |
|---|---|---|---|
| female, brain | FlyWire FAFB v783 | Zenodo (Shiu et al. packaging, Schlegel et al. annotations), as in S1-S12 | `Connectivity_783.parquet`, `neuron_annotations.tsv` |
| female, brain and nerve cord | BANC v888 | `gs://lee-lab_brain-and-nerve-cord-fly-connectome`, Harvard Dataverse doi:10.7910/DVN/7WTH1N | `compiled_data/banc_888/banc_888_meta.feather`, `neuron_connectivity/v888/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet` |
| male, brain and nerve cord | MaleCNS v1.0 | `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome` | `body-annotations-male-cns-v1.0-minconf-0.5.feather`, `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` |

Only derived statistics and results are included; the raw per-neuron files are not.

## Method
- One glomerulus list for all three animals: the 55 glomeruli with a uniglomerular PN in every animal
  (so 55 spectral bands, against 56 in S1-S12). PNs of other glomeruli (46 in MaleCNS, 0 in the others)
  get no input; multiglomerular PNs read 3 random bands, as before.
- A PN-KC pair counts if it has at least 3 synapses (the BANC release threshold). FAFB without the
  threshold (`fafb_raw_both`, the S1-S12 wiring) is kept as a link to earlier results.
- Arms per animal: left KCs with left PNs; right with right; both sides with every connection; both
  sides with cross-side PN-KC pairs removed (`both_nox`); degree-preserving shuffle of `both`; an
  equal-memory nearest-neighbour baseline (`knn`); and one per-band z-score.
- Streams, KC code, memory and measures unchanged from S1/S12. 20 seeds per stream; every arm of a seed
  sees the same stream. Paired Wilcoxon tests, Holm-corrected within each stream.
- Code: `src/reflex/s13_prep.py`, `s13_cross_animal.py`, `analyze_s13.py`, `run_s13.sh`.

## Wiring of the three animals
| Animal | KCs (left / right) | PNs | PN-KC pairs | mean PNs per KC | cross-side pairs |
|---|---|---|---|---|---|
| FAFB v783 | 5,177 (2,580 / 2,597) | 685 | 23,516 | 4.5 | 487 (2.1 %) |
| BANC v888 | 4,552 (2,260 / 2,245) | 699 | 16,720 | 3.7 | 468 (2.8 %) |
| MaleCNS v1.0 | 4,064 (2,019 / 2,045) | 686 | 20,777 | 5.1 | 407 (2.0 %) |

## Result (AUROC, mean of 20 seeds)
| Arm | FAFB: Janus line | FAFB: facility | BANC: Janus line | BANC: facility | MaleCNS: Janus line | MaleCNS: facility |
|---|---|---|---|---|---|---|
| left only | 0.952 | **0.886** | 0.935 | **0.820** | 0.949 | **0.888** |
| right only | 0.949 | 0.844 | 0.936 | 0.810 | 0.944 | 0.871 |
| both, with cross-side pairs | 0.948 | 0.866 | 0.937 | 0.809 | 0.947 | 0.877 |
| both, without cross-side pairs | 0.948 | 0.868 | 0.936 | 0.805 | 0.947 | 0.877 |
| both, shuffled | 0.949 | 0.872 | 0.937 | 0.830 | 0.949 | 0.879 |
| equal-memory nearest neighbour | 0.914 | 0.916 | 0.918 | 0.925 | 0.917 | 0.915 |

Per-band z-score (same for all): 0.975 (Janus line), 0.847 (facility). Full tables, false alarms,
stack AUROC and all 25 tests per stream: `summary.md`.

## Findings
1. **Female FAFB and male MaleCNS give the same model.** Janus line: no difference (+0.001, p = 1).
   Facility: MaleCNS slightly higher (+0.011, 18 of 20 seeds, p = 0.0009). The sex of the animal is not
   what sets the result.
2. **BANC is lower on both streams** (Janus line -0.011, facility -0.057 against FAFB; 20 of 20 seeds,
   p = 5e-5), and its measured wiring is worse than its own shuffle on the facility stream (-0.021,
   p = 0.002). BANC has the sparsest PN-KC wiring (3.7 PNs per KC, against 4.5 and 5.1) and 11 % of its
   KCs have no PN input at this threshold. The likely reason is how completely the PN-KC synapses are
   reconstructed in this release, not a property of the animal; with one fly per dataset the two cannot
   be separated.
3. **The left side beats the right in all three animals on the facility stream**: FAFB +0.042 (20/20
   seeds, p = 5e-5), MaleCNS +0.018 (18/20, p = 1e-4), BANC +0.009 (16/20, p = 0.10). On the Janus line
   the sides are within 0.006. S12 found left > right in FAFB; it now holds in three animals. Why is not
   known: the run does not separate KC wiring from the PN set and glomerulus labels of each side.
4. **Cross-side connections do not matter.** Removing them (about 2 % of pairs) changed AUROC by at most
   0.004 (largest: BANC facility +0.0035, p = 1e-4; elsewhere p >= 0.16).
5. **Both sides together are not better than the better side alone.** On the facility stream the left
   side alone (half the cells and memory) beat both sides in all three animals (p <= 3e-4).
6. **Measured wiring = degree-preserving shuffle** in FAFB and MaleCNS (p >= 0.25), as in S1-S12. The
   finding that degree structure, not the exact map, sets performance holds across animals.
7. **The ranking against other methods is unchanged in every animal**: per-band z-score is best on the
   Janus line (by 0.027-0.038), equal-memory nearest neighbour best on the facility (by 0.038-0.116).
8. **Synapse threshold:** FAFB at >= 3 synapses is within 0.003 of the unthresholded S1-S12 wiring.

## Limits
- One animal per dataset; individual and sex differences cannot be separated from reconstruction
  differences between the three datasets.
- Side labels come from each dataset's annotations; a mirror convention difference between datasets would
  swap left and right for that animal. All three agree in direction, which argues against an error.
- Simulated Raman streams, as in S1-S12.
