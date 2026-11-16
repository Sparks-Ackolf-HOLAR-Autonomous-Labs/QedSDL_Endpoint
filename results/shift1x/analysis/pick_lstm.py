#!/usr/bin/env python3
"""Pick the S20 LSTM setting from a tuning sweep (cal10, seeds 101-105, files s20_cal10_s{seed}_{lr,epochs,hidden}.json).
Rule: fewest flakes lost summed over the tuning seeds; ties broken by the lower mean held-out x_t error.
(The rule reproduces the published pick, 1e-2,600,16, from P4 results/reflex/s20_timecourse/lstm_tuning_seeds101-105.)
  python pick_lstm.py SWEEP_DIR [OUT_DIR]   -> OUT_DIR/lstm_pick.txt (the setting, one line) and lstm_pick.json
"""
import glob, json, os, sys

SEEDS = (101, 102, 103, 104, 105)
GRID = ["1e-2,2000,16", "1e-2,2000,32", "1e-2,600,16", "1e-3,600,16", "3e-3,2000,16", "3e-3,600,16", "3e-3,600,32"]


def rank(d):
    rows = []
    for cfg in GRID:
        fs = [f"{d}/s20_cal10_s{s}_{cfg}.json" for s in SEEDS]
        miss = [f for f in fs if not os.path.exists(f)]
        if miss:
            raise SystemExit(f"incomplete sweep in {d}: {len(miss)} missing for {cfg}")
        R = [json.load(open(f)) for f in fs]
        rows.append(dict(cfg=cfg, lost=sum(r["arms"]["lstm"]["lost"] for r in R),
                         mae=sum(r["final_xt_mae"]["lstm"] for r in R) / len(R), per_seed=[r["arms"]["lstm"]["lost"] for r in R]))
    return sorted(rows, key=lambda r: (r["lost"], r["mae"]))


if __name__ == "__main__":
    d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(d))
    rows = rank(d)
    open(f"{out}/lstm_pick.txt", "w").write(rows[0]["cfg"] + "\n")
    json.dump(dict(pick=rows[0]["cfg"], rule="min total lost over seeds 101-105 (cal10), tie -> min mean held-out x_t error",
                   sweep_dir=d, ranking=rows), open(f"{out}/lstm_pick.json", "w"), indent=1)
    for r in rows: print(r["cfg"], "lost", r["lost"], "mae", round(r["mae"], 4))
    print("PICK", rows[0]["cfg"])
