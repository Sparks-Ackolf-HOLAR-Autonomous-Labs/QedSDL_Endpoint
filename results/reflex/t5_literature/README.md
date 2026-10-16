# Reported values for Janus MoSSe, MoS2 and MoSe2 monolayers

`janus_lit_values.csv` lists values for monolayer Janus MoSSe as reported in published papers, preprints and the C2DB database. Monolayer MoS2 and MoSe2 rows are included as out-of-family controls. Each row gives one value from one source. The table is input for a consistency check on extracted literature values, so it holds the values as reported, without any curation or averaging.

## Columns

`material, quantity, value, unit, method, source_doi_or_url, first_author, year, where_found`

- `quantity`: `raman_A1_cm-1` (out-of-plane A1 or A1g mode), `raman_E_cm-1` (in-plane E or E2g mode), `pl_eV` (A-exciton photoluminescence energy), `gap_eV` (computed band gap), `a_A` (in-plane lattice constant), `dipole_D` (out-of-plane dipole).
- `method`: `exp` marks a measurement. Measurement temperature is given when it is not room temperature. A computed value is labelled with its functional (PBE, HSE06, G0W0, GW+BSE, BEEF-vdW) or, if the snippet does not name the functional, with `DFT`. Computed Raman frequencies appear under the same Raman quantities and are labelled `DFT`.
- `where_found`: the text the value was taken from, shortened to 150 characters at most. Table entries are paraphrased in the form "Table S1: ...".

## Counts

| material | rows | distinct sources |
|---|---|---|
| Janus MoSSe | 62 | 30 |
| MoS2 | 26 | 12 |
| MoSe2 | 21 | 11 |
| all | 109 | 43 |

Janus MoSSe rows by quantity: gap 14, Raman A1 12, lattice constant 12, Raman E 11, PL 9, dipole 4.

## How values were found

1. **Full-text snippet search** (Semantic Scholar through the Asta MCP helper `shared/asta_mcp.py`, `snippet_search`). Queries targeted Janus MoSSe Raman, PL, band gap, lattice constant and dipole values, and monolayer MoS2 and MoSe2 Raman, PL and DFT values. A value was kept only when the number and its assignment were both readable in the snippet. DOIs were resolved with `get_paper` on the Semantic Scholar corpus id. Five sources have no DOI in Semantic Scholar and are cited by arXiv URL: arXiv:2112.11211, 2304.00750, 2607.23103, 2502.20107 and 2006.12104. The last of these, Petric et al., gave no usable numbers and is not in the table.
2. **Primary papers read in full.** Lu et al., Nat. Nanotechnol. 12, 744 (2017), doi:10.1038/nnano.2017.100, PDF hosted by HKU. Zhang et al., ACS Nano 11, 8192 (2017), doi:10.1021/acsnano.7b03186, arXiv:1704.06389. Text was extracted with `pdftotext`, and the Raman, PL and lattice values were read from the main text and from Table S1 of Zhang et al.
3. **C2DB** (Haastrup et al., 2D Mater. 5, 042002 (2018)). Material pages 1MoSSe-1, 1MoS2-1 and 1MoSe2-1 at https://c2db.fysik.dtu.dk/material/ give the lattice constant, PBE, HSE06 and G0W0 gaps, and the out-of-plane dipole.

## Conversions and caveats

- Three values were converted, and each conversion is written out in `where_found`. Wavelengths became energies by E = 1239.84 / lambda (Jang et al., 726 nm and 809 nm). The C2DB dipole of -0.038 e A per cell and the 37.4 me A of Yang et al. were converted to Debye with 1 e A = 4.803 D, keeping the magnitude only.
- The dipole of Zhao et al. (6.16 D) refers to a 5x5x1 supercell and is left in that unit. The 2.99 D of Yeh et al. is given as reported; the snippet does not state the reference cell.
- Some experimental values come from samples that are not freestanding monolayers on SiO2: Xu et al. measured MoSSe on MoS2, and Zhang et al. (2024) report a Raman value from a bilayer region. Coskun et al. (2025) is a review, and its two PL values are secondary citations of a pulsed-laser-deposition study.
- Barakat et al. report a 1.85 eV gap for a 3x3 supercell, located at Gamma. This value lies outside the usual PBE range and is kept as reported. The same applies to the 3.19 A lattice constant of Zhang et al. (2023).
- The C2DB gaps for the three materials are 1.453 eV (MoSSe), 1.580 eV (MoS2) and 1.321 eV (MoSe2). These are below the plain-PBE values given in many papers. The difference probably comes from C2DB including spin-orbit coupling, but the C2DB pages were not checked to confirm this.
- No values were estimated or read from figures. A number was left out if the snippet did not say which mode or material it belonged to.
