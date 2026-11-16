"""Which SHIFT_SCALE (= argv[4]) did a published S20 variant run at? No command log survives for
s20_cal3_* and *_lstmtuned, so rerun seed 1 at each candidate scale and compare with the published JSON.
  python s20_scale_probe.py best                       -> tuned LSTM setting picked from lstm_tuning_seeds101-105
  python s20_scale_probe.py VARIANT N_CAL SCALE        -> VARIANT is base or lstmtuned (env S20_LSTM for tuned)
"""
import glob, json, os, sys
P4 = os.path.expanduser("<repo>/src/reflex")
R = f"{P4}/../../results/reflex/s20_timecourse"
if sys.argv[1] == "best":
    tot = {}
    for f in glob.glob(f"{R}/lstm_tuning_seeds101-105/*.json"):
        cfg = os.path.basename(f).split("_", 3)[3][:-5]
        d = json.load(open(f)); tot.setdefault(cfg, []).append((d["arms"]["lstm"]["lost"], d["final_xt_mae"]["lstm"]))
    for cfg, v in sorted(tot.items(), key=lambda kv: (sum(x[0] for x in kv[1]), sum(x[1] for x in kv[1]))):
        print(cfg, len(v), "lost", sum(x[0] for x in v), "mae", round(sum(x[1] for x in v) / len(v), 4))
    sys.exit()
var, n_cal, scale = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, P4)
sys.argv = ["s20", "x", "1", str(n_cal), scale]  # what s4_lit sees at import
import s4_lit as L, s20_timecourse as T  # noqa: E402
r = T.run((1, n_cal))
pub = json.load(open(f"{R}/s20_cal{n_cal}_s1{'_lstmtuned' if var == 'lstmtuned' else ''}.json"))
lost = lambda d: {a: v["lost"] for a, v in d["arms"].items()}
mae = lambda d: {a: round(v, 6) for a, v in d["final_xt_mae"].items()}
same = lost(r) == lost(pub) and mae(r) == mae(pub)
print(f"PROBE {var} cal{n_cal} scale={L.SHIFT_SCALE} match={same} new={lost(r)} pub={lost(pub)} mae_new={mae(r)} mae_pub={mae(pub)}")
