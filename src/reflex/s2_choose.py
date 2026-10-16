"""Apply the S2/S3 choice rule (s2_tune.py docstring) to the two tuning logs; write s2_params.json.
Usage: python s2_choose.py TUNE_JANUS TUNE_MULTI OUT_JSON"""
import json, sys
J, M = (json.load(open(f)) for f in sys.argv[1:3]); g = J["grid"]
ok = lambda n: all(L["log"][n]["auroc"] >= L["log"]["fly"]["auroc"] - 0.01 for L in (J, M))
fa = lambda n: (J["log"][n]["fa"] + M["log"][n]["fa"]) / 2
pick = lambda pre, n: min([f"{pre}{i}" for i in range(n) if ok(f"{pre}{i}")] or [f"{pre}{i}" for i in range(n)], key=fa)
gi = pick("gain", len(g["gain"])); ai = pick("adapt", len(g["adapt"]))
ci = max([f"coinc{i}" for i in range(len(g["coinc"])) if J["log"][f"coinc{i}"]["auroc"] >= J["log"]["fly"]["auroc"] - 0.01] or
         [f"coinc{i}" for i in range(len(g["coinc"]))], key=lambda n: M["log"][n]["stack"])
P = dict(gain=g["gain"][int(gi[4:])], adapt=g["adapt"][int(ai[5:])], coinc=g["coinc"][int(ci[5:])], chosen=[gi, ai, ci])
json.dump(P, open(sys.argv[3], "w"), indent=1); print(P)
