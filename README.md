# How the fruit fly connectome can help autonomous materials discovery

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23413469.svg)](https://doi.org/10.5281/zenodo.23413469)

Pre-release code and data for a paper in preparation (Neuromorphic Computing and Engineering, IOP). This folder runs on its own: copy it out, run `git init` inside it, and every figure script runs from the folder root with numpy and matplotlib. The test scripts additionally need the FlyWire v783 release (S13-S14 and S18 also BANC v888 and MaleCNS v1.0; S15-S16, S19 and S20 PyTorch).

| Folder | Contents |
|---|---|
| `src/reflex/` | connectome model (`s1_spectra.py`, `fly2.py`), Raman stream generators, every test (S1-S22, S5b, T1-T5), edge timing and memory scaling and analysis scripts |
| `src/janus/` | Janus MoSSe selenization: MACE-MP-0 energetics, the PBE (Quantum ESPRESSO) check of its two hop barriers (`dft_neb/`), lattice kinetic Monte Carlo, truth maps and the recipe-choice campaign |
| `src/digitise/` | digitisation of the published in situ Raman spectra of irradiated monolayer MoS2 |
| `results/` | per-seed result files and summary tables of every test |
| `paper_figures/regen_figs/` | one script per figure, drawn at print size from committed result files |
| `paper_figures/figures/` | the figures as they appear in the paper |

The tests read the FlyWire connectome (release v783, Zenodo 10.5281/zenodo.10676866: Connectivity_783.parquet and neuron_annotations.tsv) from `~/flywire_v783`; it is not redistributed here. Spectra digitised from Daniel et al., 2D Mater. 13, 025033 (2026), CC BY 4.0, are included with their source.

## Figures

| Figure | File | Script | Data |
|---|---|---|---|
| Fig. 1 | `paper_figures/figures/fig_concept.png` | `paper_figures/regen_figs/fig_concept.py` | none (schematic) |
| Fig. 2 | `paper_figures/figures/fig_spectra.png` | `paper_figures/regen_figs/fig_spectra.py` | line set in src/reflex/s1_spectra.py; results/digitise_daniel2026/fig3a_spectra.csv |
| Fig. 3 | `paper_figures/figures/fig_one_shot.png` | `paper_figures/regen_figs/fig_one_shot.py` | results/reflex/s4_insitu, s4_lit, s11_irradiation (20 seeds each) |
| Fig. 4 | `paper_figures/figures/fig_facility.png` | `paper_figures/regen_figs/fig_facility.py` | results/reflex/s8_context (20 seeds) |
| Fig. 5 | `paper_figures/figures/fig_s5b.png` | `paper_figures/regen_figs/fig_s5b.py` | results/reflex/s5b_return (30 seeds) |
| Fig. 6 | `paper_figures/figures/fig_memory_scaling.png` | `paper_figures/regen_figs/fig_memory_scaling.py` | results/reflex/s22_memory_scaling (10 seeds; llm.json) |
| Fig. 7 | `paper_figures/figures/fig_timecourse.png` | `paper_figures/regen_figs/fig_timecourse.py` | results/reflex/s20_timecourse (20 seeds), s1b_multi, s19_online_nets (20 seeds) |
| Fig. 8 | `paper_figures/figures/fig_wiring.png` | `paper_figures/regen_figs/fig_wiring.py` | results/reflex/s1_spectra, s1b_multi, s4_lit, s11_irradiation (20 seeds each) |
| Fig. 9 | `paper_figures/figures/fig_trained_device.png` | `paper_figures/regen_figs/fig_trained_device.py` | results/reflex/s15_trained, s16_device (20 seeds each) |

## Tables

| Table | Source |
|---|---|
| Table 1 | results/reflex/S2_S8_SUMMARY.md (S6, S8; src/reflex/analyze_s2s8.py) |
| Table 2 | results/reflex/edge_compare/edge_pi5.json, edge_dgx.json (src/reflex/edge_compare.py) |
| Tables A1-A5 | results/janus/summary.md, results/reflex/summary.md, S2_S8_SUMMARY.md, S9_S12_SUMMARY.md |
| Table A6 | results/reflex/s13_cross_animal/summary.md, s14_left_right/summary.md |
| Table A7 | results/reflex/s15_trained/s15_summary.md (src/reflex/analyze_s15_s16.py) |
| Table A8 | results/reflex/s19_online_nets/summary.md, summary_s19b.md (src/reflex/analyze_s19.py, analyze_s19b.py) |
| Tables A9-A10 | results/reflex/s21_memory_nets/summary.md (src/reflex/analyze_s21.py) |
| Table A11 | results/reflex/s20_timecourse/summary.md, s21_memory_nets/summary.md (src/reflex/analyze_s20.py, analyze_s21.py) |
| Table A12 | results/reflex/s16_device/s16_summary.md (src/reflex/analyze_s15_s16.py) |
| Table A13 | results/reflex/s22_memory_scaling/summary.md (src/reflex/analyze_s22.py) |

## Licence

Code: MIT License (`LICENSE`). Data, result files and figures: CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`).

## Acknowledgement

Taylor D. Sparks acknowledges support from the Army Research Office Materials Design, under Contract No. W911NF-23-1-0333.
