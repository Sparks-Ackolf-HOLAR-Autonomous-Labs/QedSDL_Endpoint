# G1 sme-stats re-analysis (2026-10-22)

The script is `<workspace>/g1_visuals/src/stats_ci.py`. It reads the release worktree `results/reflex` (read only). The full numbers are in `ci_reanalysis.json`.

Every comparison is paired over seeds. For each one the script reports:
- the Hodges-Lehmann (HL) difference, with a 95% CI from inverting the exact signed-rank distribution (approximate when there are ties);
- a 95% CI for the mean from a paired bootstrap (10,000 resamples);
- the Wilcoxon p, with Holm correction within each family;
- TOST, as two one-sided Wilcoxon tests, plus a 90% HL CI.

Equivalence margins (δ):
- AUROC: 0.01.
- Flakes: 0.3.
- Dose share on target: 0.05. This is an assumption, because review item 1 gives δ in dose points. The mae margin (0.5) is not used here.
- Janus final gap: 0.01. This is an assumption.

## Stop-and-tell items (P85: "If the analysis moves a headline")

1. **Precision.** The claim is "paper setting has the highest precision" (0.74 vs 0.65 and 0.69).
   - Pooled seed-cluster bootstrap, fly_best minus each rival:
     - logreg: +0.08, 95% CI [−0.02, +0.19], p = 0.13.
     - small NN: +0.04, CI [−0.02, +0.11], p = 0.17.
     - knn: +0.14, CI [+0.06, +0.24].
     - plain fly: +0.21, CI [+0.11, +0.31].
   - Holm over the 4 tests: logreg and NN are 0.26.
   - Under equal tuning, the plain fly setting has precision 0.53, the lowest of all rules. The 0.74 is from gp+fly_best, which was tuned on the novelty streams.
   - **Supported:** precision is not distinguishable from logreg or the NN.
   - **Not supported:** "highest precision".
2. **Job ii, flakes lost with fly_best against the baselines.** The paper quotes unadjusted p-values.
   - Against gp: −4.5 [−5.5, −3.5], Holm 0.0004.
   - Against knn: −0.5 HL [−1.0, 0.0], bootstrap mean [−1.10, −0.20]. Raw p is 0.013, Holm 0.052.
   - Against cbo: −1.0 [−2.0, 0.0]. Raw p is 0.033, Holm 0.10.
   - Against logreg: Holm 0.13.
   - Against plain fly: Holm 0.17.
   - If the family is these 5 comparisons, only the win over gp survives Holm. If the paper's own family is different, the SI table for item 6 must say which family was used.
3. **Measured against shuffled or random wiring (TOST at 0.01 AUROC).** Most pairs are equivalent. Three are not:
   - **Janus flagging (s2):** random wiring is better than measured by 0.013 [0.011, 0.016], Holm < 0.001.
   - **Cross-animal multi stream (s13, BANC):** shuffled wiring is better by 0.021 [0.012, 0.030], Holm 0.004.
   - **Facility (s8):** shuffled is better by 0.002 (Holm 0.016). That is inside the margin, so the pair is still equivalent.
   - Any sentence that says the wiring makes no difference needs this caveat: in two settings, measured wiring is measurably worse than control wiring.
4. **Planner, line 53.**
   - Fly minus gp: +0.035 [+0.014, +0.064], p = 0.001. Fly is worse than gp.
   - Fly against random: +0.025 [+0.007, +0.044], raw p 0.010, Holm 0.075.
   - "Shuffled did better" (fly minus fly_shuf): +0.012 [−0.010, +0.043], p = 0.29. This is not supported; the wording needs to change.

## Staged text for MINOR items 3 to 8 (approved in P85, apply in the next pass)

- **Item 3 (SI tables).** Add an "HL difference [95% CI]" column to each paired table, using the rows of `ci_reanalysis.json` keyed by family.
- **Item 4 (scaling).**
  - Current text: "+0.012, p = 0.34".
  - Proposed: "Eight times more Kenyon cells changed AUROC by +0.011 (95% CI −0.003 to +0.023; Holm p = 0.34). Any gain from scaling is no larger than 0.023."
- **Item 5.** Add: "Rules were tuned on seeds 100–105 and tested on seeds 1–20, which do not overlap."
- **Item 6.** Add an SI table that lists each Holm family and its members (the family keys in the JSON). The table must say which family the job ii p-values belong to; see stop item 2.
- **Item 7 (planner).** Medians and IQRs of the final gap:
  - random 0.014 (0.006–0.046);
  - gp 0.003 (0.003–0.005);
  - fly 0.047 (0.016–0.071);
  - fly_shuf 0.018 (0.009–0.053);
  - llm 0.007 (0.006–0.020).
  Paired tests against random (Holm over 12): gp 0.40; fly 0.075; fly_shuf 1; llm 1. With the prior: gp_prior 0.007 and llm_prior 0.003.
- **Item 8 (limitations).** Add: "All p-values and intervals are over simulator seeds. They do not include the variation between instruments, so they understate the uncertainty of a deployment."

## MAJOR items 1 and 2 (numbers ready, waiting for approval)

- **Item 1 (equivalence).** These pairs pass TOST:
  - s4 fly_best against shuffled wiring (0.3 flakes): p = 0.023.
  - s11 dose against shuffled and random: p < 0.01.
  - s13 Janus, all three animals: p < 0.001.
  - s18a cord_fly against shuffled and random: p < 0.001.
  - Equal-tuned dose, fly against the NN: p = 0.001.
  - s11 fly against knn: p = 0.008.

  These pairs do **not** pass:
  - Job ii, equal-tuning fly against logreg: TOST p = 0.075.
  - Fly against the NN: TOST p = 0.18.
  - s4 fly_best against random wiring: TOST p = 0.29.
  - The planner pair fly against fly_shuf: TOST p = 0.55.

  The phrase "all within noise" should become "no difference detected (90% CI 0 to +0.5 flakes); equivalence within 0.3 flakes not shown".
- **Item 2 (Highlight 1).** The equal-tuning result: fly 3.55, logreg 3.40, NN 3.60. HL differences are 0.0 [0.0, +0.5] and 0.0 [−0.5, +0.5]. The precision lead is not significant (stop item 1). This item cannot be staged until the PM rules on the headline.

## Unchanged findings (now with CIs)

- **Furnace (s18a).** Ridge at the furnace minus the planner:
  - at 60 s: −10.5 [−12.0, −9.0];
  - at 10 s: −0.25 [−0.5, 0.0], Holm 0.043.

  Ridge at the furnace is equivalent to the planner at 2 s, and cord_fly is equivalent to ridge.
- **Time course (s20).** seq_kc beats every other arm, Holm ≤ 0.015.
- **Memory (s25).** Frozen minus online:
  - s1b +0.041 [0.034, 0.048];
  - long +0.061 [0.052, 0.071];
  - shift −0.115 [−0.121, −0.109].
- **Dose (s11).** fly_lookup beats ridge by +0.058 [0.033, 0.100] and dose by +0.45. It is equivalent to knn and to the tuned NN.
- **Facility (s8).** fly_ctx4 is 0.009 to 0.016 below fam4 and zscore_fam, and 0.019 above knn_budget_fam.

## PM ruling P119 (2026-10-23)

- **Approved as MINOR, apply in the next pass:**
  - review item 1 (TOST equivalence wording, as staged above);
  - stop item 4, line 53: delete "shuffled did better". Replace it with: "The fly planner ended further from the target than gp (median gap 0.047 vs 0.003; paired p = 0.001). Against random choice the difference was not significant after Holm correction (p = 0.075). Measured and shuffled wiring did not differ (p = 0.29)."
  - items 3 to 8.
- **On hold until the user decides on card G1-hl:** stop items 1 to 3 (precision, Holm for job ii, the s2/s13 wiring caveat) and review item 2 (Highlight 1).
