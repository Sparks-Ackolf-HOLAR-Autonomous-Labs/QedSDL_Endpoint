"""Digitise Fig. 3(a) of Daniel et al., 2D Mater. 13 (2026) 025033 (in situ Raman of Ar+-irradiated
monolayer MoS2, CC BY 4.0) from the raster image embedded in the PDF, and check the result against
the paper's own Tables 2 and 3 (A1g and E12g peak position and FWHM).

Usage: python daniel2026_fig3a.py <fig3.png> <out_dir>
fig3.png is the page-5 raster extracted with `pdfimages -png -f 4 -l 6 <pdf> img` (img-006.png).
Pixel geometry (axis box, ticks) was read from that image; see the README next to the outputs.
"""
import sys, json, csv
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit

# axis calibration of panel (a): ticks at x px 110.5, 203.5, 296.5 = 200, 300, 400 cm-1
X0_PX, X0_CM, PX_PER_CM = 110.5, 200.0, 0.93
BOX = dict(x0=38, x1=388, y0=42, y1=636)
TRACES = {  # label: (reference RGB, fluence 1e13 cm-2, start y px at right edge)
    "pristine": ((74, 30, 0), 0.0),
    "1e13": ((255, 141, 8), 1.0),
    "5e13": ((255, 218, 78), 5.0),
    "1e14": ((126, 176, 40), 10.0),
}
# Tables 2 and 3 of the paper, in situ rows (position, FWHM), cm-1
TABLE = {0.0: dict(A1g=(404.0, 5.2), E2g=(383.2, 3.8)), 1.0: dict(A1g=(404.3, 6.2), E2g=(382.3, 4.1)),
         5.0: dict(A1g=(405.2, 10.5), E2g=(380.6, 6.3)), 10.0: dict(A1g=(405.5, 13.5), E2g=(376.7, 9.7))}


def px_to_cm(x):
    return X0_CM + (x - X0_PX) / PX_PER_CM


def trace(im, ref, tol=70.0):
    """Per column, centroid of pixels within `tol` of the trace colour, tracked from the right edge
    so text labels of the same colour (left side) are not picked up."""
    dist = np.sqrt(((im - np.array(ref)) ** 2).sum(2))
    m = dist < tol
    xs, ys, prev = [], [], None
    for x in range(BOX["x1"] - 2, BOX["x0"] + 2, -1):
        yy = np.where(m[BOX["y0"]:BOX["y1"], x])[0] + BOX["y0"]
        if len(yy) == 0:
            continue
        # split into vertical runs; keep the run nearest the previous column
        runs = np.split(yy, np.where(np.diff(yy) > 3)[0] + 1)
        if prev is None:
            run = max(runs, key=len)
        else:
            run = min(runs, key=lambda r: abs(r.mean() - prev))
            if abs(run.mean() - prev) > 60:
                continue
        w = 1.0 / (1.0 + dist[run, x])
        prev = float((run * w).sum() / w.sum())
        xs.append(x); ys.append(prev)
    xs, ys = np.array(xs[::-1]), np.array(ys[::-1])
    return px_to_cm(xs), ys


def lor(x, a, x0, g):
    return a * (g / 2) ** 2 / ((x - x0) ** 2 + (g / 2) ** 2)


def two_lor(x, a1, c1, g1, a2, c2, g2, b0, b1, a3, c3, g3):
    return lor(x, a1, c1, g1) + lor(x, a2, c2, g2) + lor(x, a3, c3, g3) + b0 + b1 * (x - 390)


def main(png, out):
    im = np.array(Image.open(png).convert("RGB")).astype(float)
    rows, spectra = [], {}
    for lab, (ref, flu) in TRACES.items():
        cm, ypx = trace(im, ref)
        base = np.median(ypx[(cm > 460) & (cm < 500)])
        inten = base - ypx  # pixels above the trace's own high-shift baseline
        spectra[lab] = (cm, inten)
        sel = (cm > 340) & (cm < 425)
        x, y = cm[sel], inten[sel]
        tE, tA = TABLE[flu]["E2g"][0], TABLE[flu]["A1g"][0]
        # A1g, E12g, plus one broad LO/TO(M) defect band below 375 cm-1
        p0 = [y.max(), 405, 6, y.max() / 2, 382, 5, 0, 0, y.max() / 5, 360, 25]
        lo = [0, 398, 1, 0, 370, 1, -50, -5, 0, 340, 8]
        hi = [1e4, 412, 30, 1e4, 390, 30, 50, 5, 1e4, 375, 60]
        p, cov = curve_fit(two_lor, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        err = np.sqrt(np.diag(cov))
        rows.append(dict(trace=lab, fluence_1e13=flu,
                         A1g_pos=round(p[1], 2), A1g_pos_se=round(err[1], 2), A1g_pos_table=tA,
                         A1g_fwhm=round(p[2], 2), A1g_fwhm_table=TABLE[flu]["A1g"][1],
                         E2g_pos=round(p[4], 2), E2g_pos_se=round(err[4], 2), E2g_pos_table=tE,
                         E2g_fwhm=round(p[5], 2), E2g_fwhm_table=TABLE[flu]["E2g"][1],
                         A1g_minus_E2g=round(p[1] - p[4], 2), A1g_minus_E2g_table=round(tA - tE, 2)))
    with open(f"{out}/fig3a_spectra.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trace", "raman_shift_cm-1", "intensity_px"])
        for lab, (cm, inten) in spectra.items():
            for a, b in zip(cm, inten):
                w.writerow([lab, round(a, 2), round(b, 2)])
    json.dump(rows, open(f"{out}/fig3a_peak_check.json", "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
