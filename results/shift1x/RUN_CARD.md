# RUN CARD: G1 QedSDL_Endpoint, S18a and S20 reruns at an explicit SHIFT_SCALE = 1x

Owner: g1-fly. Prepared by the analyst, 2026-10-22.
- **Status:**
  - **D-031 (blocks A+B)** was launched by computemanager2 at 2026-10-22T21:23 (pid 1511836), with `SHIFT_SCALE=1 BLOCKS="s18a s20"`.
  - D-031 finished at 21:28: `s18a rc=0`, `s20_cal10 rc=0`, `SHIFT1X_DONE`.
  - **D-033** (booked by g1-fly with cm, in three steps; see Launch):
    - **step 1** `s20_cal3 signed` (cm, pid 1523323, `launch2.nohup`): done 21:34, both rc=0, `SHIFT1X_DONE`.
    - **step 2** the LSTM tuning sweep at 1x (`s20_tune`): launched by cm at 21:34 (pid 1535197, `launch3.nohup`). The pick rule was frozen at 21:35 in `shift1x/PREREG_lstm_pick.md`, with 0 of 35 sweep outputs written.
    - step 2 done 21:49: 7/7 rc=0, pick **1e-2,600,16** (same as the published pick; 10 lost vs 12 for the runners-up).
    - **step 3** `s20_cal10_lt s20_cal3_lt` (cm watcher, `launch4.nohup`): both rc=0, `SHIFT1X_DONE`.
  - **Analysis done:** `shift1x/compare/compare_shift.md` (22/25 claims "yes"; cal10 lstm_tuned vs snap_ridge loses significance; cal3 lstm_tuned vs lstm flips, n.s. both). Sent to g1-fly.
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
| LSTM tuning sweep `lstm_tuning_seeds101-105/` | **5x** | seed 101, setting 1e-2,600,16: exact match at 5 only; 1-4, 6 and 8 differ (`probe/tune_probe.py`, `t_x*.log`). 10x also differs on all 5 seeds (smoke 4) | **added: s20_tune** |

**None ran at 1x, so nothing is skipped.** That conclusion is recorded in `shift1x/probe/at1x_none.txt`.
- So the published sweep (5x) and the published tuned runs (10x) were not even at the same scale.
- The published tuned setting is lr 1e-2, 600 epochs, 16 units (`analyze_s20.py` comment; the 10x match confirms it).
- **At 1x the setting is not fixed.** g1-fly asked (D-033) for the sweep to be redone at 1x, because the paper says the LSTM was tuned on separate seeds. Blocks B3/B4 take whatever the 1x sweep picks.
- **Pick rule** (`shift1x/analysis/pick_lstm.py`, now read-only; **preregistered in `shift1x/PREREG_lstm_pick.md`** with the grid, seeds, sha256 and timestamp; accepted by g1-fly; made once, never revised): fewest flakes lost in total over seeds 101-105, with ties broken by the lower mean held-out x_t error. On the published sweep this rule gives 1e-2,600,16, the published pick (`probe/pub_pick/lstm_pick.json`). The draft gives no rule beyond "chosen on seeds 101 to 105", so this one is ours and is written into `lstm_pick.json`.

## Blocks (20 seeds each, 1-20, as published)
| Block | Target and settings | Published (P4 `results/reflex/`) | Output |
|---|---|---|---|
| A `s18a`: S18a job iv (tab:control) | `s18a_furnace.py` | `s18_control/s18a/s18a_s*.json` (6x) | `shift1x/s18a/` |
| B `s20`: S20 cal10, all arms | `s20_timecourse.py`, N_CAL 10, S20_LSTM default 3e-3,600,16 | `s20_timecourse/s20_cal10_s*.json` (5x) | `shift1x/s20/` |
| B2 `s20_cal3`: S20 cal3, all arms | same, N_CAL 3 | `s20_timecourse/s20_cal3_s*.json` (5x) | `shift1x/s20/` |
| T `s20_tune`: LSTM tuning sweep | S20_ARMS=lstm, cal10, seeds **101-105**, 7 settings: 1e-2,2000,16; 1e-2,2000,32; 1e-2,600,16; 1e-3,600,16; 3e-3,2000,16; 3e-3,600,16; 3e-3,600,32; S20_TAG=_{setting} | `s20_timecourse/lstm_tuning_seeds101-105/s20_cal10_s{seed}_{setting}.json` (5x) | `shift1x/s20_tune/`; pick in **`shift1x/lstm_pick.txt`** |
| B3 `s20_cal10_lt`: tuned LSTM, cal10 | S20_ARMS=lstm, S20_TAG=_lstmtuned, **S20_LSTM = first line of `shift1x/lstm_pick.txt`** | `s20_timecourse/s20_cal10_s*_lstmtuned.json` (10x) | `shift1x/s20/` |
| B4 `s20_cal3_lt`: tuned LSTM, cal3 | same, N_CAL 3 | `s20_timecourse/s20_cal3_s*_lstmtuned.json` (10x) | `shift1x/s20/` |
| C `signed`: signed cord | neuro_reruns `src/signed_cord.py` | none published | `shift1x/signed/` |

- All S20 blocks share `shift1x/s20/`, as in P4, so the analyzer copy reads them the same way.
- BANC_DIR is passed as `NA`. Both cord caches exist and are checked: `<home>/monster_blood/p4_s18/banc_cord.npz` and `neuro_reruns/cache/banc_cord_signed.npz`.

## Launch (by computemanager2 only; DGX; D-018: 4 workers, nice -n 10 + ionice idle, nohup, every log ends rc=)
D-031 (done 21:28, all rc=0):
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s18a s20" nohup bash shift1x/run_shift1x.sh > shift1x/launch.nohup 2>&1 < /dev/null &
```
**D-033, three steps, run one after the other** (each step after the previous nohup ends `SHIFT1X_DONE`):

Step 1 (launched by cm, pid 1523323; done):
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s20_cal3 signed" nohup bash shift1x/run_shift1x.sh > shift1x/launch2.nohup 2>&1 < /dev/null &
```
Step 2, the LSTM tuning sweep at 1x (launched by cm, pid 1535197):
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS=s20_tune nohup bash shift1x/run_shift1x.sh > shift1x/launch3.nohup 2>&1 < /dev/null &
```
- It runs the 7 settings in turn, each over seeds 101-105 at 4 workers. Outputs go to `shift1x/s20_tune/s20_cal10_s{seed}_{setting}.json`, logs to `shift1x/logs/s20_tune_{setting}.log`, and sidecars `pin_s20_cal10_{setting}.json`.
- Then `analysis/pick_lstm.py shift1x/s20_tune shift1x` writes **`shift1x/lstm_pick.txt`** (one line, e.g. `1e-2,600,16`) and `shift1x/lstm_pick.json` (pick, rule, full ranking). Its log is `shift1x/logs/lstm_pick.log`, ending `PICK ...` and `rc=0`.
- pick_lstm refuses an incomplete sweep (all 35 files must exist), and the step then fails with rc=3.

Step 3, the tuned-LSTM blocks with the 1x pick:
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s20_cal10_lt s20_cal3_lt" nohup bash shift1x/run_shift1x.sh > shift1x/launch4.nohup 2>&1 < /dev/null &
```
- Both blocks read `shift1x/lstm_pick.txt`. Without it they stop with rc=2 and `no shift1x/lstm_pick.txt: run BLOCKS=s20_tune first`, so step 3 cannot run on a stale or missing pick.
- The setting used is logged on each START line (`S20_LSTM=`) and in the sidecars.
- cm has chained step 3 in watcher pid 1587585 (`<workspace>/COMPUTE/watch_d033s3.log`). It starts after launch3.nohup ends SHIFT1X_DONE, `lstm_pick.log` ends rc=0 and `lstm_pick.txt` is non-empty. the analyst also sends the pick to cm (copy to g1-fly).
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
- compare_shift.md opens with **the LSTM tuning pick, published (5x) vs 1x**: the full ranking of the 7 settings on both sides (rank, flakes lost, mean x_t error) and whether the pick is the same or changed.
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

**Smoke 4** (`shift1x/smoke4/`, the sweep and the pick file):
- `nopick/`: s20_cal3_lt with no `lstm_pick.txt` stops with rc=2 and the message above. Nothing is written.
- `x10/`: `TUNE_GRID=1e-2,600,16`, seeds 101-105, 4 workers: the tune block gives rc=0 in 87 s wall. pick_lstm then refuses the one-setting sweep as incomplete (rc=1), and the launcher stops with rc=3, as intended.
- The 10x tuning outputs do **not** match the published sweep (lost new/published, seeds 101-105: 4/3, 1/0, 0/1, 2/1, 5/3). The follow-up probe found the published sweep at 5x (table above).
- pick_lstm on the published sweep gives 1e-2,600,16. compare_shift's pick section, self-tested with the published sweep on both sides, shows identical rankings.

## Cost (measured per seed under load about 19 on 20 cores)
| Block | Per seed (1 core) | 20 seeds | Wall at 4 workers |
|---|---|---|---|
| A S18a | 14-16 s | about 5 core-min | about 1.5 min |
| B S20 cal10 | 51 s | about 17 core-min | about 5 min |
| B2 S20 cal3 | 38 s | about 13 core-min | about 4 min |
| B3 lstmtuned cal10 | 43-49 s | about 16 core-min | about 4 min |
| B4 lstmtuned cal3 | 16-18 s | about 6 core-min | about 2 min |
| C signed | 32 s | about 11 core-min | about 3 min |
| T LSTM sweep, 7 x 5 seeds | about 70 s per seed at 600 epochs, 3-4 min at 2000 | about 70 core-min | about 25 min (allow 45) |
| **All seven** | | **about 2.3 core-h** | **about 45 min (allow 80)** |

Measured so far: D-031 took 74 s (s18a) and 268 s (s20 cal10); step 1 s20_cal3 took 143 s.

**Node-hours: about 0.6 expected and at most 1.3, counting the full wall allowance as node time.** That is far under the 50 node-h cap. CPU only, no GPU. Disk: a few MB. The probes and smoke tests used about 0.5 core-h in total.

## Outputs for the owner
- Run outputs: `<repo>/results/shift1x/{s18a,s20,s20_tune,signed}/`, with the same file names and schema as P4, plus the pin sidecars.
- The 1x LSTM pick: `shift1x/lstm_pick.txt` and `lstm_pick.json`.
- Comparison tables: `shift1x/compare/compare_shift.md`.
- All old outputs are kept: the P4 results (read only), `shift1x/smoke/`, `smoke2/`, `smoke3/`, `smoke4/` and `probe/`.

## S21 extra networks on the S20 stop (Table S11, Note S3): 1x rerun (cm D-035, P263a; launched by cm 22:12, pid 1673379)
**Status 2026-10-22:** D-035 done, S21_DONE: 2 tune blocks + pick (rc=0) + 4 test blocks, all rc=0. 1x picks: lr 1e-2 for all five nets (published: peep, mog and memnet at 3e-3). Results in the S21 section of `shift1x/compare/compare_shift.md` (copy sent: `shift1x/compare_d035_sent/`). Note S3: codes vs GRU / peephole / Mogrifier hold at p<=0.005 (Holm 0.0009 / 0.0014 / 0.0047); codes vs transformer p=0.09 -> 0.012; transformer vs tuned LSTM p=0.03 -> 0.054 (direction kept, not significant). Finding (on the card): published Table S11 pairs three scales (nets 4x, KC/perceptron 5x, tuned LSTM 10x).
Asked by g1-fly at 2026-10-22T21:57. Same P263 scope: DGX only, at most 50 node-h, nice 10, existing inputs, P4 untouched.

**The bug here.** P4 `results/reflex/s21_memory_nets/run_s20.sh` calls `s20_timecourse.py <out> <seeds> <ncal> 4`, so `s4_lit` read WORKERS=4 as SHIFT_SCALE. Tuning and test both ran at 4x.
- Check: seed 1, cal10, lr 1e-2, all 5 nets, through `pin_shift.py`: **exact match to the published `s21_memory_nets/s20/s20_cal10_s1_s21_lr1e-2.json` at 4x only** (arms, x_t error, parameter counts; only `fit_s` differs). 1x and 5x differ (`shift1x/smoke5/x{1,4,5}`).
- The published runs used the code copy `<home>/monster_blood/p4_s21/src`. Its s20_timecourse, s4_insitu, common and memnets differ from P4 only in docstrings, and s4_lit is identical. So P4 code is used here.
- **Mixed scales in Table S11:** published `analyze_s21.py` pairs the nets (4x) with the reference arms from `s20_timecourse/`: Kenyon-cell and perceptron arms at 5x, tuned LSTM at 10x. At 1x all arms share one scale (nets from `shift1x/s21/`, reference arms from the D-031/D-033 runs in `shift1x/s20/`).

**Launcher:** `shift1x/run_s21.sh` (read-only). It is separate from run_shift1x.sh, whose sha256 is frozen in PREREG_lstm_pick.md.
- `s21_tune`: lr 3e-3 and 1e-2, 600 epochs, 16 units, seeds 101-105, cal10, the 5 nets (S20_ARMS=net_gru,net_peep,net_mog,net_tfm,net_memnet), then `analysis/pick_s21.py` -> `shift1x/s21_pick.{txt,json}`.
- `s21_test`: lr 1e-2 and 3e-3 x cal10 and cal3, seeds 1-20. Both learning rates are run, as published, so the test does not depend on the pick; the analysis takes the pick.
- **Pick rule: preregistered in `shift1x/PREREG_s21_pick.md`** (frozen 2026-10-22T22:10:25-06:00, 0 of 10 sweep outputs, sha256 of pick_s21.py, run_s21.sh and pin_shift.py). Same rule as PREREG_lstm_pick, per network. On the published sweep it reproduces the published picks: GRU 1e-2, peephole 3e-3, Mogrifier 3e-3, transformer 1e-2, memory-augmented 3e-3.

**Command (one launch; tune, then test):**
```bash
cd <run-dir> && SHIFT_SCALE=1 BLOCKS="s21_tune s21_test" nohup bash shift1x/run_s21.sh > shift1x/launch_s21.nohup 2>&1 < /dev/null &
```
- Ends `S21_DONE`. Block logs: `shift1x/logs/s21_*.log` (START with shift_scale and S20 settings, END wall_s, `rc=`). Pick log: `logs/s21_pick.log`.
- If the pick fails (incomplete sweep), the launcher stops with rc=3 before the test.

**Smoke 5** (`shift1x/smoke5/`, 2026-10-22):
- `x1/x4/x5`: the 4x match above.
- `launch/`: `TUNE_SEEDS=101 BLOCKS=s21_tune`: both lrs rc=0, then pick_s21 refuses the one-seed sweep (rc=1) and the launcher stops with rc=3, as intended. `SEEDS=1 BLOCKS=s21_test`: 4 blocks rc=0, `S21_DONE`; the launcher's cal10 lr 1e-2 output equals the direct 1x run.
- `compare_shift.py` self-test with published data on both sides: all 22 S21 claims "yes" (47/47 overall); the published side reproduces all 18 entries of the published `s21_memory_nets/summary.json` S20 section exactly, and its picks.

**Cost** (one core per seed, load about 12-15, all 5 nets in one run):
| Block | Per seed | Runs | Core-min | Wall at 4 workers |
|---|---|---|---|---|
| s21_tune, cal10 | 86-98 s | 2 lr x 5 seeds | about 16 | about 6 min |
| s21_test, cal10 | 84-89 s | 2 lr x 20 seeds | about 60 | about 16 min |
| s21_test, cal3 | 30-33 s | 2 lr x 20 seeds | about 21 | about 6 min |
| **Total** | | 90 runs | **about 1.6 core-h** | **about 28 min (allow 50)** |

**Node-hours: about 0.5 expected and at most 0.9, counting the full wall allowance as node time.** Far under 50, so it does not need the PM. CPU only.

**Analysis:** the S21 section of `shift1x/analysis/compare_shift.py`, the same pairing and tests as P4 `analyze_s21.py` on both sides:
- per-network pick, published (4x) vs 1x;
- Table S11 rows for cal10 and cal3: lost, under / over, x_t error, trained numbers;
- the 11 paired tests per table (Holm), with the Note S3 claims marked: two codes vs GRU and the gated LSTMs (p <= 0.005), vs transformer (p = 0.09), transformer vs LSTM (p = 0.03).
- Outputs: `shift1x/s21/{s20_tune,s20}/`, `shift1x/s21_pick.{txt,json}`, and the S21 section of `shift1x/compare/compare_shift.md`.
