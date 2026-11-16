# RUN CARD: G1 QedSDL_Endpoint, S18a and S20 reruns at an explicit SHIFT_SCALE = 1x

Owner: g1-fly. Prepared by the analyst, 2026-10-22.
- **Status:**
  - **D-031 (blocks A+B)** was launched by computemanager2 at 2026-10-22T21:23 (pid 1511836), with `SHIFT_SCALE=1 BLOCKS="s18a s20"`.
  - **The remaining four blocks are not launched.** g1-fly books them with computemanager2 (proposed D-033), and cm launches them after pid 1511836 exits.
  - Full runs are launched by cm only; the analyst runs smoke tests only.

## Ruling and conditions (P263)
The PM ruled this a MINOR error fix, on these conditions:
- at most 50 node-h in total;
- DGX only, no CHPC;
- registered with computemanager2 before anything launches;
- nice -n 10 for anything longer than about 30 min;
- existing inputs only.

This card fits all of them:
- **about 0.3 node-h expected, at most 0.6 node-h** for all six blocks (see Cost);
- DGX only;
- the launcher always runs `nice -n 10 ionice -c3`;
- it uses only the existing cord caches and P4 code.

## The bug and the fix
- `s4_lit.py:33` reads `SHIFT_SCALE = float(sys.argv[4])` at import. The callers put WORKERS in argv[4], so the published runs ran at whatever worker count they used (see the next section).
- The fix is `<repo>/results/shift1x/pin_shift.py`. **SHIFT_SCALE is an explicit, required argument:**
  ```
  pin_shift.py --shift-scale X {s18a|s20|signed} ARGS...
  ```
  - Without `--shift-scale`, the run stops with a usage error. The worker count no longer reaches SHIFT_SCALE.
- **P4 is untouched** (read-only, HEAD 975ff83f8). The wrapper:
  - imports `s4_lit` first and sets `L.SHIFT_SCALE = X`;
  - wraps `L.raman_lit` so it asserts the value has not drifted;
  - forces the `fork` start method, so pool workers inherit it;
  - runs the target script unchanged as `__main__`;
  - refuses more than 4 workers (D-018);
  - writes a sidecar beside the outputs: `pin_s18a.json`, `pin_s20_cal{3,10}[_lstmtuned].json` or `pin_signed.json`. Each records shift_scale, argv, workers, P4 head, python, start method, S20_ARMS, S20_TAG and S20_LSTM.
- The launcher is `shift1x/run_shift1x.sh`. It takes the scale from the `SHIFT_SCALE` environment variable (default 1) and logs it, with the S20 settings, on each block's START line.
- No other module in the import chain reads argv at import. Checked: s4_insitu, common, janus/campaign, s18a_furnace, s20_timecourse, signed_cord.

## Scale of each published run
No command or nohup log survives for the published S20 cal3 or LSTM-tuned runs. Searches covered `<home>/monster_blood/*`, `g1_visuals` and the P4 git history: the only S20 launch scripts there are the S21 `net_*` runs, and commits 29a8c26f1 and c0f543655 contain no launch line.

So the scale was recovered by rerunning seed 1 at argv[4] = 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20 and comparing with the published JSON. The probe is `shift1x/probe/s20_scale_probe.py` and its logs are `p_*.log`. A match means identical lost, under, over and lost_fast per arm, held-out x_t error and flake rates.

| Published set | Ran at | Evidence | On this card |
|---|---|---|---|
| S18a `s18a_s*` | 6x | `--shift-scale 6` reproduces `s18a_s1.json` (smoke 2) | block A |
| S20 `s20_cal10_s*` | 5x | argv-5 diagnostic `neuro_reruns/s20_argv/s20_argv5.log` reproduces cal10 s1 | block B |
| S20 `s20_cal3_s*` | **5x** | exact match at 5 only; 1, 2, 3, 4 and 6 all differ | **added: s20_cal3** |
| S20 `s20_cal10_s*_lstmtuned` | **10x** | exact match at 10 only; 1-6, 8 and 12 differ | **added: s20_cal10_lt** |
| S20 `s20_cal3_s*_lstmtuned` | **10x** | exact match at 10 only; 1-6, 8, 12, 16 and 20 differ | **added: s20_cal3_lt** |

**None ran at 1x, so nothing is skipped.** That conclusion is recorded in `shift1x/probe/at1x_none.txt`.
- The tuned setting is lr 1e-2, 600 epochs, 16 units, with `S20_ARMS=lstm` and `S20_TAG=_lstmtuned`. It comes from the `analyze_s20.py` comment, and it is also the best of the 7 configurations in `lstm_tuning_seeds101-105` by flakes lost.
- The 10x match also confirms that setting.
- Not on the card, for the owner to decide: whether to redo the tuning sweep itself (seeds 101-105, 7 configurations) at 1x. That is about 35 LSTM-only cal10 runs, roughly 0.05 node-h. As it stands, the setting chosen at the old scale is kept.

## Blocks (20 seeds each, 1-20, as published)
| Block | Target and settings | Published (P4 `results/reflex/`) | Output |
|---|---|---|---|
| A `s18a`: S18a job iv (tab:control) | `s18a_furnace.py` | `s18_control/s18a/s18a_s*.json` (6x) | `shift1x/s18a/` |
| B `s20`: S20 cal10, all arms | `s20_timecourse.py`, N_CAL 10, S20_LSTM default 3e-3,600,16 | `s20_timecourse/s20_cal10_s*.json` (5x) | `shift1x/s20/` |
| B2 `s20_cal3`: S20 cal3, all arms | same, N_CAL 3 | `s20_timecourse/s20_cal3_s*.json` (5x) | `shift1x/s20/` |
| B3 `s20_cal10_lt`: tuned LSTM, cal10 | S20_ARMS=lstm, S20_TAG=_lstmtuned, S20_LSTM=1e-2,600,16 | `s20_timecourse/s20_cal10_s*_lstmtuned.json` (10x) | `shift1x/s20/` |
| B4 `s20_cal3_lt`: tuned LSTM, cal3 | same, N_CAL 3 | `s20_timecourse/s20_cal3_s*_lstmtuned.json` (10x) | `shift1x/s20/` |
| C `signed`: signed cord | neuro_reruns `src/signed_cord.py` | none published | `shift1x/signed/` |

- All S20 blocks share `shift1x/s20/`, as in P4, so the analyzer copy reads them the same way.
- BANC_DIR is passed as `NA`. Both cord caches exist and are checked: `<home>/monster_blood/p4_s18/banc_cord.npz` and `neuro_reruns/cache/banc_cord_signed.npz`.

## Launch (by computemanager2 only; DGX; D-018: 4 workers, nice -n 10 + ionice idle, nohup, every log ends rc=)
D-031 (running since 21:23, pid 1511836):
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s18a s20" nohup bash shift1x/run_shift1x.sh > shift1x/launch.nohup 2>&1 < /dev/null &
```
Remaining blocks (proposed D-033, launch after pid 1511836 exits; writes to a separate nohup file):
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s20_cal3 s20_cal10_lt s20_cal3_lt signed" nohup bash shift1x/run_shift1x.sh > shift1x/launch2.nohup 2>&1 < /dev/null &
```
- Python: `<workspace>/venvs/venv_shared/bin/python` (3.13.15, torch 2.14.0, numpy 2.5.3). Threads are pinned to 1.
- Logs: `shift1x/logs/<block>.log`. Each has START with shift_scale and the S20 settings, END with wall_s, and ends `rc=N`.
- `launch.nohup` ends `SHIFT1X_DONE`. A failed block stops the chain.
- It does not touch vLLM on 8020 or the slide server on 8131 (CPU only, no ports).
- `OUT_ROOT=` redirects outputs for smoke tests only. Leave it unset for the real run.

## After the runs: analysis (copies; P4 untouched)
The analysis lives in `shift1x/analysis/`:
- `analyze_s18.py` and `analyze_s20.py` are verbatim copies of P4 (`cp -p`).
- `compare_shift.py` is new.
```bash
cd <repo>/results/shift1x && PY=<workspace>/venvs/venv_shared/bin/python && $PY analysis/analyze_s18.py . && $PY analysis/analyze_s20.py s20 && $PY analysis/compare_shift.py
```
- These write `shift1x/summary.md` (S18a), `shift1x/s20/summary.{md,json}` (S20, cal3 and cal10 with lstm_tuned), and **`shift1x/compare/compare_shift.{md,json}`**.
- **The comparison tables** put published (6x/5x/10x) next to 1x, for each table:
  - per arm: lost of 30 and x_t error, with the change;
  - per claim: each paired Wilcoxon (Holm) test from the analyzers, with diff, seeds better/worse and Holm p on both sides, plus a "holds at 1x?" column: yes / direction kept but significance changed / **direction flips**.
  - Covered: S18a (7 claims), S20 cal3 (9), S20 cal10 (9).
- Self-test: with the published dirs on both sides, all 25 claims read "yes" and the Holm p values equal the published README (e.g. local_ridge_bands vs open_loop -8.45, Holm 0.000565).
- Block C has no published counterpart. Its 1x numbers go into a separate short table.

## Smoke tests (2026-10-22, nice 10 + ionice idle)
**Smoke 1** (`shift1x/smoke/`, earlier hard-coded 1x wrapper, kept as `smoke1_pin1.sh.old`):
- S18a s1: WORKERS 1 and 4 give identical results.
- S18a s1 at 1x equals the unpinned 1x diagnostic.
- S20 cal10 s1 at 1x equals `s20_argv1.log`.
- Signed s1 at 1x equals `neuro_reruns/test/signed_s1.log`.

**Smoke 2** (`shift1x/smoke2/`, explicit argument):
- With no `--shift-scale`: usage error, and nothing is written.
- `--shift-scale 1`, WORKERS 1 vs 4: identical, and identical to smoke 1. Workers no longer change the drift.
- `--shift-scale 6` is identical to the published `s18a_s1.json`.

**Smoke 3** (`shift1x/smoke3/`, the new blocks through the launcher, seed 1, WORKERS 1):
- At 1x: all three blocks give rc=0, and the results equal the 1x probes.
  - cal3 lost: now_ridge 8, snap_ridge 1, snap_mlp 2, seq_ridge 3, lstm 1, trend_rule 9, others 0.
  - lstmtuned: cal3 lost 0, cal10 lost 3.
- At 10x: both LSTM-tuned blocks reproduce the published `_lstmtuned` s1 exactly in arms, errors and flake rates. The only differences are `fit_s` (timing) and `params.mlp`. The current P4 code skips fitting the MLP when `S20_ARMS=lstm`, so this field is null in the new files and 7938 in the published ones. The `lstm` count (4770) is unchanged.
- Sidecars record the right shift_scale and S20_LSTM.

## Cost (measured per seed under load about 19 on 20 cores)
| Block | Per seed (1 core) | 20 seeds | Wall at 4 workers |
|---|---|---|---|
| A S18a | 14-16 s | about 5 core-min | about 1.5 min |
| B S20 cal10 | 51 s | about 17 core-min | about 5 min |
| B2 S20 cal3 | 38 s | about 13 core-min | about 4 min |
| B3 lstmtuned cal10 | 43-49 s | about 16 core-min | about 4 min |
| B4 lstmtuned cal3 | 16-18 s | about 6 core-min | about 2 min |
| C signed | 32 s | about 11 core-min | about 3 min |
| **All six** | | **about 68 core-min (1.1 core-h)** | **about 20 min (allow 35)** |

**Node-hours: about 0.3 expected and at most 0.6, counting the full wall allowance as node time.** That is far under the 50 node-h cap. CPU only, no GPU. Disk: a few MB. The probes and smoke tests used about 0.5 core-h in total.

## Outputs for the owner
- Run outputs: `<repo>/results/shift1x/{s18a,s20,signed}/`, with the same file names and schema as P4, plus the pin sidecars.
- Comparison tables: `shift1x/compare/compare_shift.md`.
- All old outputs are kept: the P4 results (read only), `shift1x/smoke/`, `smoke2/`, `smoke3/` and `probe/`.
