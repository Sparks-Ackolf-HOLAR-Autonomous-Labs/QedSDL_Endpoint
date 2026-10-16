"""Summary of S15-MLP against the S15 arms (paired by seed, Wilcoxon signed-rank, Holm over the table)."""
import glob, json, os, re
import numpy as np
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "reflex")
mlp = {int(re.findall(r"s(\d+)\.json", f)[0]): json.load(open(f)) for f in glob.glob(f"{R}/s15_mlp/s15mlp_s[0-9]*.json")}
s15 = {int(re.findall(r"s(\d+)\.json", f)[0]): json.load(open(f)) for f in glob.glob(f"{R}/s15_trained/s15_s[0-9]*.json")}
seeds = sorted(set(mlp) & set(s15))
def get(arm, key):
    src = [mlp[s]["arms"].get(arm) or s15[s]["task_a"]["arms"][arm] for s in seeds]
    return np.array([d["auroc_by_n"][key] if key in ("25", "50", "100", "200", "400") else d[key] for d in src])
arms = ["fly", "shuf", "rand", "dense", "logreg", "knn", "mlp32", "mlp64x2"]
keys = ["25", "50", "100", "200", "400", "w_zero_shot", "w_fine_tuned", "w_scratch"]
L = [f"# S15-MLP: small trained networks vs the connectome model ({len(seeds)} seeds, held-out AUROC, mean)", "",
     "| arm | " + " | ".join(f"n={k}" if k.isdigit() else k for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
for a in arms:
    L.append(f"| {a} | " + " | ".join(f"{get(a, k).mean():.3f}" for k in keys) + " |")
tests = []
for m in ("mlp32", "mlp64x2"):
    for ref in ("fly", "logreg"):
        for k in ("25", "100", "400", "w_zero_shot", "w_fine_tuned"):
            x, y = get(m, k), get(ref, k); p = wilcoxon(x, y).pvalue
            tests.append([m, ref, k, float((x - y).mean()), int((x > y).sum()), p])
P = np.array([t[-1] for t in tests]); o = np.argsort(P); h = np.empty_like(P); run = 0
for r, i in enumerate(o): run = max(run, min(1, (len(P) - r) * P[i])); h[i] = run
L += ["", "Paired tests (MLP minus reference; Wilcoxon signed-rank, Holm over these 20 tests)", "",
      "| MLP | reference | measure | mean difference | seeds MLP higher | p (Holm) |", "|---|---|---|---|---|---|"]
for t, hp in zip(tests, h):
    L.append(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]:+.3f} | {t[4]}/{len(seeds)} | {hp:.3g} |")
open(f"{R}/s15_mlp/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
