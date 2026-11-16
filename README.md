# How the Fruit Fly Mushroom Body Can Help Autonomous Materials Discovery

Code and result files for the preprint "How the Fruit Fly Mushroom Body Can Help Autonomous Materials Discovery: Learning to Stop, Dose and End Runs from a Few Examples at One Default Setting, in Simulation" (Ackölf and Sparks). This repository holds code and data only. The paper text and its figures are not included.

Every analysis script runs from the repository root with the packages in `requirements.txt`. The test scripts also need the FlyWire v783 release (S13, S14 and S18 also need BANC v888 and MaleCNS v1.0, and S23 needs the BANC read-out from S18). S15, S16, S19, S20, S24, S25 and S26c need PyTorch.

| Folder | Contents |
|---|---|
| `src/reflex/` | connectome model (`s1_spectra.py`, `fly2.py`), Raman stream generators, every test (S1 to S26, S5b, T1 to T5), edge timing, memory scaling and the analysis scripts |
| `src/janus/` | Janus MoSSe selenization: MACE-MP-0 energetics, the PBE (Quantum ESPRESSO) check of its two hop barriers (`dft_neb/`), lattice kinetic Monte Carlo, truth maps and the recipe-choice campaign |
| `src/digitise/` | digitisation of the published in situ Raman spectra of irradiated monolayer MoS2 |
| `src/econ_L.py`, `src/stats_ci.py`, `src/at351_defaults_p.py` | cost re-analysis of the stop call (writes `results/s23_econ/`), bootstrap intervals (writes `results/stats_ci/`) and p values for the default-setting arms |
| `src/shift1x/` | shifted-input reruns of S18 and S21, with their pre-registered picks |
| `results/` | per-seed result files and summary tables of every test |
| `paper_figures/regen_figs/paper_fig_style.py` | plot style module, used by `src/econ_L.py` for its own diagnostic plot |

The tests read the FlyWire connectome (release v783, Zenodo 10.5281/zenodo.10676866: `Connectivity_783.parquet` and `neuron_annotations.tsv`) from `~/flywire_v783`. It is not redistributed here. Spectra digitised from Daniel et al., 2D Mater. 13, 025033 (2026), CC BY 4.0, are included with their source.

## Results by test

| Test | Result files | Analysis script |
|---|---|---|
| S1 to S12 | `results/reflex/summary.md`, `S2_S8_SUMMARY.md`, `S9_S12_SUMMARY.md`, `results/janus/summary.md` | `src/reflex/analyze*.py`, `src/janus/analyze.py` |
| S5b | `results/reflex/s5b_return/` | `src/reflex/analyze_s5b.py` |
| S13, S14 | `results/reflex/s13_cross_animal/summary.md`, `s14_left_right/summary.md` | `src/reflex/analyze_s13.py`, `analyze_s14.py` |
| S15, S16 | `results/reflex/s15_trained/s15_summary.md`, `s16_device/s16_summary.md` | `src/reflex/analyze_s15_s16.py` |
| S18 | `results/reflex/s18_control/summary.md` | `src/reflex/analyze_s18.py` |
| S19 | `results/reflex/s19_online_nets/summary.md`, `summary_s19b.md` | `src/reflex/analyze_s19.py`, `analyze_s19b.py` |
| S20, S21 | `results/reflex/s20_timecourse/summary.md`, `s21_memory_nets/summary.md` | `src/reflex/analyze_s20.py`, `analyze_s21.py` |
| S22 | `results/reflex/s22_memory_scaling/summary.md` | `src/reflex/analyze_s22.py` |
| S23 | `results/reflex/s23_fair_tuning/README.md`, `results/s23_econ/` | `src/reflex/s23_*.py`, `src/econ_L.py` |
| S24 | `results/reflex/s24_fixed_memory/tables.md` | `src/reflex/analyze_s24.py` |
| S25 | `results/reflex/s25_memory_need/tables.md`, `recal_tables.md` | `src/reflex/analyze_s25.py`, `analyze_recal.py` |
| S26 | `results/reflex/s26_mpnn_branchy/SUMMARY.md` | `src/reflex/analyze_s26.py` |
| Edge timing | `results/reflex/edge_compare/` | `src/reflex/edge_compare.py` |
| Shifted inputs | `results/shift1x/` | `src/shift1x/` |
| Intervals | `results/stats_ci/` | `src/stats_ci.py` |

`CONTENTS.tsv` lists every file with its role and licence, and `FILES.sha256` gives its checksum.

## Licence

Code: MIT License (`LICENSE`). Data and result files: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).

## Acknowledgement

Taylor D. Sparks and Kinston Ackölf acknowledge support from the Army Research Office Materials Design, under Contract No. W911NF-23-1-0333. Taylor D. Sparks acknowledges support from the National Science Foundation through the AURORA Programmable Cloud Lab (NSF Award No. 2607535).
