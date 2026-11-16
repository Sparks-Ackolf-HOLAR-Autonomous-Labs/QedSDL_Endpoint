# S25b-recal: the laser drop with recalibration from scratch at the new intensity (21 Oct 2026)

In S25 the engineered specialists (E) were "repaired" only by adding new-intensity spectra to their old calibration. Here every
method discards its old calibration or memory at the drop (laser power ×1/3) and refits or relearns from post-drop data only.
The generic connectome model (G) gets the same option, next to its no-recalibration result. Seeds 1–20, the S25 harnesses, run on
the DGX. Full tables are in `recal_tables.md` and every number is in `recal_summary.json`; per-seed files are in `recal/`; the script is
`src/reflex/s25_recal.py`. The pre-drop and no-recalibration references reproduce S25.
**Recovered** means back within 95 % of the method's own pre-drop result, measured as 95 % of its pre-drop gain over a no-learning
baseline (share on target over 0; flakes lost under the open-loop valve, 8.6, or the GP planner alone, 7.75). For flagging it means
false alarms ≤ 7.5 % and AUROC gain at least 95 % of the pre-drop gain. Paired Wilcoxon against the method's own pre-drop result.

| job | method | before drop | no recalibration | recalibrated from scratch on n post-drop examples | examples to recover |
|---|---|---|---|---|---|
| dose (share on target) | **E** ridge | 0.883 | 0.745 | 5: 0.475 · 10: 0.688 · 20: 0.840 · 41: **0.860** · 82: 0.825 · 123: 0.770 | **20 spectra** (best 0.860 at 41; 123: p = 0.009 vs before) |
| | **G** code lookup | 0.952 | **0.918** | 5: 0.475 · 20: 0.840 · 41: 0.873 · 123: 0.903 | not within 123 (0.903 against 0.904 needed); recalibrating is worse than keeping the old codes |
| | P lookup (10 %) | 0.972 | 0.948 | 41: 0.913 · 123: 0.922 | not within 123 |
| | nearest calibration spectrum | 0.928 | 0.898 | 41: 0.793 · 123: 0.915 | 82 |
| furnace end point (flakes lost) | **E** ridge on bands | 0.10 | 1.55 | 10: 15.2 · 45: 9.5 · 180: 5.55 · 360: 2.15 · 540: **1.60** | **not within 540 readings** (3 flakes; 540: p = 0.005 vs before) |
| | **G** ridge on the code | 0.00 | **0.05** | 10: 13.95 · 45: 3.50 · 90: 1.00 · 360: 0.05 · 540: 0.35 | 360 readings; not needed (no recalibration: 0.05) |
| failing-run stop (flakes lost; drop at run 15 of 30) | **E** logistic regression C 10 | 3.40 | **3.50** (memory kept; p = 0.16) | cleared, then learns from the first C post-drop runs: 3: 4.75 · 6: 4.70 · 10: 4.60 · all 15: 4.55 | not within 15 runs; not needed |
| | **G** model | 3.55 | **3.65** (p = 0.16) | 3: 4.65 · 10: 4.65 · all 15: 4.65 | not within 15 runs; not needed |
| flagging, S1b (false alarms / recall / AUROC in a 400-reading window starting n readings after the drop; pre-drop threshold) | **E** z-score routed | 0.05 / 0.98 / 0.990 | n 0: 0.21 / 0.99 · n 1,000: 0.16 / 0.99 | reset: n 0: 0.24 / 0.96 · n 400: 0.14 / 0.98 · n 1,000: 0.14 / 0.99 | not within 1,000 readings (false alarms stay at 0.14–0.16) |
| | E nearest neighbour routed (184) | 0.05 / 0.75 / 0.949 | **0.04 / 0.81** at every n (no excess alarms) | reset: n 41: 0.07 / 0.85 · n 1,000: 0.05 / 0.78 | none needed (41 after a reset) |
| | **G** model told the family | 0.06 / 0.98 / 0.991 | n 0: 0.22 / 0.98 · n 400: 0.14 · n 1,000: **0.09 / 0.91** | reset: n 0: 0.81 / 1.00 · n 400: 0.37 · n 1,000: 0.17 / 0.99 | not within 1,000 readings by the 7.5 % rule; recalibrating is worse than not |

**Time to recover from scratch.** It is dominated by collecting the examples, not by computing:
- Dose: E needs 20 spectra, about half of one calibration flake (41 dose steps). G's best from-scratch recalibration still does not
  reach its no-recalibration result.
- Furnace: E does not recover within 540 readings, three calibration flakes of 180 readings each, one every 2 s, so about 18
  minutes of furnace time. G needs 360 readings (12 minutes) from scratch, or none if it keeps its old read-out.
- Refitting takes 1 ms for each engineered ridge. G's lookup needs 0.13 s to code 123 spectra, and its furnace ridge on 5,177
  code cells takes 2.0 s on one busy Grace core.
- Flagging: the routed z-score and the generic model both need more than 1,000 readings to be back under 7.5 % false alarms at the
  old threshold. Resetting either at the drop does not shorten that.

## Does the S25 verdict ("G wins where the instrument changes") hold?

- **Dose: holds.** A clean recalibration lets the engineered ridge recover relative to its own pre-drop score with only
  20 new-intensity spectra (0.840; best 0.860 at 41). But its best is still below the generic model with no recalibration at all
  (0.918), and the gap to the generic model before the drop (0.952) remains. For the generic model, recalibrating from scratch is
  worse than keeping its old codes (0.903 at 123 against 0.918), so it should not be recalibrated.
- **Furnace end point: holds, and strengthens.** The band ridge does not come back even when refitted on three new-intensity
  flakes (1.60 lost against 0.10 before; 540 readings, p = 0.005). The drop removes information the band ridge relied on, and
  more calibration does not restore it. The read-out from the Kenyon-cell code loses 0.05 flakes without any recalibration.
- **Failing-run stop: unchanged (a tie).** The drop does not hurt any method that keeps its memory (p ≥ 0.16). Clearing memory
  costs about one flake for every method, G and E alike, and 15 post-drop runs are not enough to win it back.
- **Flagging: softens.** The routed z-score (S25's E flagger) does not recover within 1,000 readings, with or without a reset
  (false alarms 0.14–0.16 at the old threshold). The generic model falls to 0.09 by then and resetting it hurts. But an engineered
  alternative, the routed nearest neighbour, shows no excess false alarms after the drop at all (0.04). It pays for that with lower
  recall (0.78–0.85 against 0.98) and lower AUROC (0.95–0.96 against 0.97–0.99). So E can buy instrument robustness for flagging by
  switching method, at a cost in sensitivity.

**Net.** The verdict holds for the jobs that read a quantity from the spectrum (dose, furnace end point). After a clean
recalibration, the engineered band models are still worse than the generic model with none, and for the furnace no amount of
new-intensity calibration up to three flakes restores them. It is a tie for the failing-run stop. It softens for flagging,
where a robust but less sensitive engineered flagger exists. The S25 sentence should become: "after an instrument change, the
generic model's read-outs need no recalibration and stay ahead on dose and the furnace end point; engineered specialists
recalibrated from scratch at the new intensity recover on dose relative to their own lower baseline (20 spectra) but not on
the furnace end point, and for flagging the choice is between a robust, less sensitive nearest neighbour and a z-score or
connectome model that needs about 1,000 readings to settle."

Notes: every simulation ran on the DGX (6 workers; about 116 GB available throughout; watchdog at 30 GB never fired). The alarm
threshold for flagging is the deployed pre-drop one. A threshold re-set from post-drop data was not tested, and would shorten every
flagger's recovery.
