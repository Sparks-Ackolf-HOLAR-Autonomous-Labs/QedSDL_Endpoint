# Published in-situ Raman of MoS2 selenization: what exists (checked 11 Oct 2026)

Question: is there a published in-situ Raman of MoS2 selenization to set against the
simulated probe of S4? Search: about 15 web searches, 4 Asta snippet searches, DOIs checked on Crossref.

## Answer

No published study records Raman spectra **at temperature, in Se vapour or H2Se, while MoS2 is thermally
selenized**. Every MoS2 selenization Raman series found is ex situ (sample cooled or stopped, then measured).
In-situ series exist for the tungsten analogue and for plasma-driven Janus conversion. None of these
papers deposits its spectra in a public repository; the data are in figures only (Qin 2022: on request).

## In situ (spectra during the reaction)

| Study | System and conditions | What the in-situ spectra show | Figures with the series |
|---|---|---|---|
| Harris et al., ACS Nano 2023, [10.1021/acsnano.2c09952](https://doi.org/10.1021/acsnano.2c09952) (green OA copy at OSTI) | WS2 monolayer selenized by a pulsed-laser Se plasma (3.2-9.6 eV per Se atom) at 450 C; 532 nm Raman, one spectrum per pulse | layer by layer: WS2 to Janus WSSe to WSe2. Janus A1 appears near 276 cm^-1 and moves up to about 285 cm^-1 as the top layer completes; WSe2 A1 appears after about 200 pulses. At 3.2 eV the conversion stops at the Janus structure (17,500 pulses). At 450 C the WS2 E' line sits at 350.7 cm^-1 (352.3 at room temperature) and the WSe2 A1 at 245.1 (249.5) | Fig. 1b-e, 2a, 3b-e (frequency against Se fraction), 4a-e |
| Qin et al., Adv. Mater. 2022, [10.1002/adma.202106222](https://doi.org/10.1002/adma.202106222) | WSe2 to Janus SWSe in a room-temperature H2 plasma with S radicals, inside a Raman/PL microscope (reverse direction: S replaces Se) | about 5 min with no change, then the WSe2 A1 line shifts, weakens and broadens as Se vacancies build up (up to about 60 %), before the Janus A1 line (284 cm^-1) appears. MoSe2 to SMoSe shown ex situ only | Fig. 3a-e |
| Sayyad et al., Adv. Mater. 2024, [10.1002/adma.202403583](https://doi.org/10.1002/adma.202403583) | same plasma reactor, WSe2 to WSSe | time-resolved map: Janus line near 290 cm^-1 from about 90 s, complete by about 240 s | Fig. 1a |
| Afaneh et al., Adv. Funct. Mater. 2018, [10.1002/adfm.201802949](https://doi.org/10.1002/adfm.201802949) | WSe2 to WS2 driven locally by the 532 nm laser in H2S | WSe2 A1g decay with time constants 19 s and 321 s, WS2 A1g growth 179 s | Fig. 3a, 4a; Table 1 |

## Ex situ MoS2 selenization series (closest Mo data)

| Study | Series | Observation |
|---|---|---|
| Guo et al., PNAS 2021, [10.1073/pnas.2106124118](https://doi.org/10.1073/pnas.2106124118) | MoS2 to Janus MoSSe, room-temperature H2 plasma with Se, 10 s to 15 min | Janus A1 present after 10 s, about 10 cm^-1 below its final position; MoS2 A1g gone by 15 min |
| Su et al., Front. Energy Res. 2014, [10.3389/fenrg.2014.00027](https://doi.org/10.3389/fenrg.2014.00027) | MoS2 in Se vapour, 600-900 C, 2 h each | no change at 600 C, alloy modes from 700 C, near-complete MoSe2 at 900 C |
| Lu et al., Nat. Nanotechnol. 2017, [10.1038/nnano.2017.100](https://doi.org/10.1038/nnano.2017.100) | H2-plasma stripping then selenization at 350 C | spectra after each step (Janus A1 288, E 355 cm^-1) |

Lin et al., ACS Nano 2020 ([10.1021/acsnano.9b10196](https://doi.org/10.1021/acsnano.9b10196)), the predecessor
of Harris 2023, is WS2 to WSSe with in-situ plasma diagnostics, not MoS2 and not in-situ Raman.

## What this means for S4

The S4 probe assumed room-temperature line positions, a Janus line fixed at its final position and
temperature-independent widths. The in-situ studies show three things that this leaves out: lines measured
at the anneal temperature are red-shifted (by an amount that varies with the recipe temperature), the Janus
line moves with the top-layer Se fraction, and a vacancy-rich intermediate broadens the parent lines before
the Janus line appears (the last is already in S4 through the lattice-defect broadening). S4L
(`src/reflex/s4_lit.py`, `results/reflex/s4_lit/summary.md`) re-runs S4 with the first two, plus thermal
broadening and emission background; the values are taken from the WS2 study and are assumptions for MoS2.
A test on measured MoS2 spectra is not possible from the literature; it needs our own in-situ run or the
spectra from a group with an in-situ furnace cell.
