"""Summary table for S7 (one connectome model vs several). Usage: python analyze_s7.py RESULT_DIR"""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon
d = sys.argv[1]; R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s7_s*.json"))]
names = list(R[0]["det"]); kinds = list(R[0]["det"]["fly"]["auroc_by_kind"])
g = lambda n, k: np.array([r["det"][n][k] for r in R])
base = R[0]["det"]["fly"]["stored_numbers"]
L = [f"## S7 one connectome model vs several, four-family Raman stream (S1b), {len(R)} seeds", "",
     "| Arm | memory (x one model) | AUROC mean | IQR | recall at 5 % false alarms | false alarms after laser change | stack AUROC | p (AUROC vs fly) |",
     "|---|---|---|---|---|---|---|---|"]
for n in names:
    a = g(n, "auroc"); st = np.array([r["det"][n]["auroc_by_kind"]["MoS2/WSe2_stack"] for r in R])
    p = "" if n == "fly" else f"{wilcoxon(a, g('fly', 'auroc')).pvalue:.2g}"
    L.append(f"| {n} | {g(n,'stored_numbers').mean()/base:.2f} | {a.mean():.3f} | {np.quantile(a,.25):.3f} to {np.quantile(a,.75):.3f} | "
             f"{g(n,'recall_at_5fpr').mean():.2f} | {g(n,'false_alarm_after_change').mean():.3f} | {st.mean():.2f} | {p} |")
L += ["", "AUROC by kind of new event:", "", "| Arm | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 1) + "|"]
for n in names:
    L.append(f"| {n} | " + " | ".join(f"{np.mean([r['det'][n]['auroc_by_kind'][k] for r in R]):.3f}" for k in kinds) + " |")
pairs = [("fam4_quarter", "zscore_fam"), ("fam4_quarter", "knn_budget_fam"), ("fam4_full", "fam4_quarter"), ("ens4_wiring", "fly")]
L += ["", "Paired Wilcoxon on AUROC: " + "; ".join(f"{a} vs {b} p = {wilcoxon(g(a,'auroc'), g(b,'auroc')).pvalue:.2g} "
      f"(mean diff {np.mean(g(a,'auroc')-g(b,'auroc')):+.3f})" for a, b in pairs)]
open(f"{d}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
