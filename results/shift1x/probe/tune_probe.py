"""Scale of the published LSTM tuning sweep: seed 101, cal10, lstm only, setting 1e-2,600,16, at argv4 = SCALE."""
import json, os, sys
P4 = os.path.expanduser("<repo>/src/reflex")
scale = sys.argv[1]; sys.path.insert(0, P4); sys.argv = ["s20", "x", "101", "10", scale]
import s4_lit as L, s20_timecourse as T
r = T.run((101, 10))
pub = json.load(open(f"{P4}/../../results/reflex/s20_timecourse/lstm_tuning_seeds101-105/s20_cal10_s101_1e-2,600,16.json"))
m = r["arms"] == pub["arms"] and r["final_xt_mae"] == pub["final_xt_mae"]
nl, pl = r["arms"]["lstm"]["lost"], pub["arms"]["lstm"]["lost"]
nm, pm = r["final_xt_mae"]["lstm"], pub["final_xt_mae"]["lstm"]
print(f"TPROBE scale={L.SHIFT_SCALE} match={m} new={nl} pub={pl} mae_new={nm:.6f} mae_pub={pm:.6f}")
