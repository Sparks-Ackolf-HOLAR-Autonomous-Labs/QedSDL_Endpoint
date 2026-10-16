# Reading published Raman spectra from a figure: Daniel et al. 2026

Source: L. Daniel, Y. Liebsch, C. Lintz, U. Javed, O. Kharsah, L. Breuer, J. Kotakoski and M. Schleberger,
"In situ Raman study on sulfur vacancies in monolayer MoS2", *2D Mater.* **13**, 025033 (2026),
doi:10.1088/2053-1583/ae6b2b, CC BY 4.0. The digitised values below are derived from that figure and
keep the same licence and attribution.

## What the paper holds

- Monolayer CVD MoS2 irradiated with 600 eV Ar+ in ultrahigh vacuum; in situ Raman (532 nm) after
  fluences of 0, 1e13, 5e13 and 1e14 cm-2, then ambient exposure and return to vacuum. STEM counts
  sulfur vacancies (randomly distributed).
- Numbers printed in the paper: Table 1 (defect-mode positions), Tables 2 and 3 (A1g and E12g position,
  shift and FWHM for every step, with standard errors), the linear fit in Fig. 3(d)
  (R_DM/OM = 2.2e-14 cm2 x fluence + 0.21).
- Raw spectra: Zenodo record 10.5281/zenodo.17987589 ("STEM (TIFF), Raman and PL spectra (TXT)").
  The paper says the embargo ends 31 July 2026, but on 11 Oct 2026 the Zenodo API lists the record as
  embargoed until **30 July 2032** and refuses the files (HTTP 403). This looks like a date entry error;
  the corresponding author (marika.schleberger@uni-due.de) can lift it.

## Can the figure be read by computer vision?

Yes, for the in situ spectra in Fig. 3(a). The figures are raster images (235 ppi), not vector paths.
`src/digitise/daniel2026_fig3a.py`:

1. extracts the panel image (`pdfimages -png -f 4 -l 6`, image 6);
2. calibrates the shift axis from the tick marks (93 px per 100 cm-1, so 1.08 cm-1 per pixel);
3. follows each trace by its colour, column by column from the right edge, so the same-coloured
   labels on the left are not picked up;
4. fits A1g, E12g and one broad LO/TO(M) band with Lorentzians and compares with Tables 2 and 3.

| Fluence (cm-2) | A1g position, figure / table | A1g FWHM, figure / table | E12g position, figure / table | E12g FWHM, figure / table |
|---|---|---|---|---|
| 0 | 403.8 / 404.0 | 5.5 / 5.2 | 383.3 / 383.2 | 4.4 / 3.8 |
| 1e13 | 404.1 / 404.3 | 6.1 / 6.2 | 382.1 / 382.3 | 4.4 / 4.1 |
| 5e13 | 404.8 / 405.2 | 10.3 / 10.5 | 380.1 / 380.6 | 5.8 / 6.3 |
| 1e14 | 405.2 / 405.5 | 12.3 / 13.5 | 374.2 / 376.7 | 12.3 / 9.7 |

All values in cm-1. Up to 5e13 cm-2 the positions agree within 0.1-0.5 cm-1 (half a pixel) and the
widths within 0.6 cm-1. The A1g readings sit 0.2-0.4 cm-1 low in every row, which points to a
sub-pixel offset in the tick calibration. At 1e14 cm-2 the E12g mode is a shoulder on the LO(M) band,
and the three-peak fit puts it 2.5 cm-1 too low; that is a fitting ambiguity, not a reading error, and
the authors' own value comes from fitting the raw spectra.

What is lost from the figure: absolute intensities (each trace is normalised and offset), the
repeat spectra behind the standard errors, and any detail finer than about 1 cm-1.

## Where a vision-language model helps

The digitiser needed four facts that were read by eye from the figure: the axis range and tick
values, which colour is which fluence, where the labels sit, and that panel (a) is in situ. A VLM
reads exactly these (text, legends, layout) reliably, and is weak at the part the pixel code does
well (sub-pixel curve positions). A workable pipeline is therefore: VLM reads the layout and labels,
pixel code extracts the curves, and the result is checked against any numbers the paper prints in
tables. This reading was done by Claude (a VLM) while the digitisation code was written, so it is not a blind test; a blind
test would give a local VLM only the figure and score its layout reading.

Files: `fig3a_spectra.csv` (shift, intensity in pixels above each trace's 460-500 cm-1 baseline),
`fig3a_peak_check.json`, `fig_digitised.png`.
