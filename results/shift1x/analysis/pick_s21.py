#!/usr/bin/env python3
"""Pick the learning rate of each S21 network on the S20 tuning seeds (cal10, seeds 101-105, 600 epochs, 16 units;
files s20_cal10_s{seed}_s21_lr{lr}.json, each holding all five nets).
Rule (as PREREG_lstm_pick.md, per network): fewest flakes lost summed over the tuning seeds; ties broken by the lower
mean held-out x_t error. (Published analyze_s21.py used mean flakes lost only, tie -> first lr listed, 3e-3; on the
published sweep the two rules agree, no ties.)
  python pick_s21.py SWEEP_DIR [OUT_DIR]   -> OUT_DIR/s21_pick.txt ("net lr" per line) and s21_pick.json
"""
import json, os, sys

SEEDS = (101, 102, 103, 104, 105)
LRS = ("3e-3", "1e-2")
NETS = ("net_gru", "net_peep", "net_mog", "net_tfm", "net_memnet")


def rank(d):
    R = {}
    for lr in LRS:
        fs = [f"{d}/s20_cal10_s{s}_s21_lr{lr}.json" for s in SEEDS]
        miss = [f for f in fs if not os.path.exists(f)]
        if miss:
            raise SystemExit(f"incomplete sweep in {d}: {len(miss)} missing for lr {lr}")
        R[lr] = [json.load(open(f)) for f in fs]
    out = {}
    for a in NETS:
        rows = [dict(lr=lr, lost=sum(r["arms"][a]["lost"] for r in R[lr]), mae=sum(r["final_xt_mae"][a] for r in R[lr]) / len(SEEDS),
                     per_seed=[r["arms"][a]["lost"] for r in R[lr]]) for lr in LRS]
        out[a] = sorted(rows, key=lambda r: (r["lost"], r["mae"]))
    return out


if __name__ == "__main__":
    d = sys.argv[1]; o = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(d))
    rk = rank(d); pick = {a: rk[a][0]["lr"] for a in NETS}
    open(f"{o}/s21_pick.txt", "w").write("".join(f"{a} {pick[a]}\n" for a in NETS))
    json.dump(dict(pick=pick, rule="per net: min total lost over seeds 101-105 (cal10), tie -> min mean held-out x_t error",
                   sweep_dir=d, ranking=rk), open(f"{o}/s21_pick.json", "w"), indent=1)
    for a in NETS:
        print(a, " ".join(f"lr{r['lr']} lost {r['lost']:.0f} mae {r['mae']:.4f}" for r in rk[a]), "->", pick[a])
    print("PICK", " ".join(f"{a}={pick[a]}" for a in NETS))
