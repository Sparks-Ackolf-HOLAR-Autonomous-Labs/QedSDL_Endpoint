#!/usr/bin/env python3
"""S24: Raspberry Pi 5 times ESTIMATED (not measured) from the S24 Grace-core times and Pi/Grace ratios of the
same operations measured on both machines earlier (the Pi was unreachable for S24).

Matched pairs (same script, one thread, on monsterpi and on a DGX Grace core):
  sparse PN->KC code      edge_compare fly_sparse (ratio r_fly); s18c cord_stop (second sparse point, lower bound)
  small vector ops        edge_compare zscore, logreg, ridge; s18c bands_stop (lower bound)
  nearest-neighbour scan  edge_compare knn_budget (92 spectra); s22 probe, 10,000 spectra (Pi: s22 pi.json,
                          Grace: s22_probe_dgx.json, run here); log-linear in the number of stored spectra between them.
                          The probe's 1,000-spectrum pair (Grace slower than Pi) looks like a warm-up artefact; it is not used.
Estimate = Grace time x central ratio; range = Grace time x [lowest, highest] ratio of that kind.
Serving all four jobs: the connectome model's time is the code (r_fly) plus the read-outs (small-op ratio);
the stacks use the small-op ratio, and the nearest-neighbour parts use the scan ratio.
No Pi analogue for the PyTorch autoencoders (the Pi measurements are NumPy only); they are not estimated.
Usage: python s24_pi_estimate.py RESULTS_REFLEX_DIR
"""
import json, sys
import numpy as np

R = sys.argv[1]; S24 = f"{R}/s24_fixed_memory"
J = lambda p: json.load(open(f"{R}/{p}"))  # noqa: E731
ed, ep = J("edge_compare/edge_dgx.json")["methods"], J("edge_compare/edge_pi5.json")["methods"]
cd, cp = J("s18_control/s18c_dgx.json"), J("s18_control/s18c_pi.json")
pd_, pp = J("s24_fixed_memory/s22_probe_dgx.json")["knn_full"], J("s22_memory_scaling/pi.json")["knn_full"]
T = J("s24_fixed_memory/timing.json")

rat = lambda a, b: b / a  # noqa: E731
r_fly = rat(ed["fly_sparse"]["us_per_decision"], ep["fly_sparse"]["us_per_decision"])
r_fly_lo = rat(cd["cord_stop_ms"], cp["cord_stop_ms"])
small = {k: rat(ed[k]["us_per_decision"], ep[k]["us_per_decision"]) for k in ("zscore", "logreg", "ridge")}
small["bands_stop"] = rat(cd["bands_stop_ms"], cp["bands_stop_ms"])
r_small = float(np.mean([small[k] for k in ("zscore", "logreg", "ridge")])); r_small_lo, r_small_hi = min(small.values()), max(small.values())
k92 = rat(ed["knn_budget"]["us_per_decision"], ep["knn_budget"]["us_per_decision"])
k1e4 = rat(pd_["10000"]["ms"], pp["10000"]["ms"])


def r_knn(n):
    t = np.clip((np.log(n) - np.log(92)) / (np.log(1e4) - np.log(92)), 0, 1); return float(k92 + t * (k1e4 - k92))


def ratio_for(arm):
    if arm.startswith("fly"): return r_fly, r_fly_lo, r_fly
    if arm.startswith("zscore"): return r_small, r_small_lo, r_small_hi
    if arm.startswith("knnfam_n"): n = int(arm[8:]) // 4; r = r_knn(n); return r, k92, max(r, k92)
    if arm.startswith("knn_n"): n = int(arm[5:]); r = r_knn(n); return r, k92, max(r, k92)
    return None


det = {}
for a, v in T["detectors"].items():
    q = ratio_for(a)
    if q is None: det[a] = None; continue
    us = v["us_per_decision"]; det[a] = dict(grace_us=us, ratio=q[0], pi_est_us=us * q[0], pi_range_us=[us * q[1], us * q[2]])

# serving all four jobs
S = T["serve"]; fly_code = T["detectors"]["fly_ctx2"]["us_per_decision"]
c_us = S["connectome"]["us_per_spectrum"]; rest = max(c_us - fly_code, 0.0)
serve = dict(
    connectome=dict(grace_us=c_us, pi_est_us=fly_code * r_fly + rest * r_small,
                    pi_range_us=[c_us * r_fly_lo, fly_code * r_fly + rest * r_small_hi],
                    note="code part (context code timing) x sparse ratio, read-outs x small-op ratio"),
    stack=dict(grace_us=S["stack"]["us_per_spectrum"], pi_est_us=S["stack"]["us_per_spectrum"] * r_small,
               pi_range_us=[S["stack"]["us_per_spectrum"] * r_small_lo, S["stack"]["us_per_spectrum"] * r_small_hi]),
    stack_alt=dict(grace_us=S["stack_alt"]["us_per_spectrum"], pi_est_us=S["stack_alt"]["us_per_spectrum"] * r_knn(184 // 4),
                   pi_range_us=[S["stack_alt"]["us_per_spectrum"] * r_small_lo, S["stack_alt"]["us_per_spectrum"] * max(r_small_hi, k1e4)]))
out = dict(estimated_not_measured=True,
           ratios=dict(sparse_code=r_fly, sparse_code_low=r_fly_lo, small_ops=small, small_ops_mean=r_small,
                       knn_92=k92, knn_10000=k1e4,
                       knn_1000_excluded=rat(pd_["1000"]["ms"], pp["1000"]["ms"])),
           numpy=dict(pi="1.24.2 (edge_compare, s18c, s22)", grace=T["numpy"]), detectors=det, serve=serve)
json.dump(out, open(f"{S24}/pi_estimate.json", "w"), indent=1)

f = lambda us: f"{us / 1e3:.3f}"  # noqa: E731
L = ["", "## Raspberry Pi 5: estimated, not measured", "",
     "The Pi could not be reached from the compute server during S24. These times scale the Grace-core",
     "times above by Pi/Grace ratios of the same operations measured earlier on both machines with the same scripts: sparse code",
     f"{r_fly:.2f} (edge_compare; nerve-cord read-out {r_fly_lo:.2f}), small vector ops {r_small:.1f} (z-score, logistic, ridge "
     f"{', '.join(f'{small[k]:.1f}' for k in ('zscore', 'logreg', 'ridge'))}; band ridge stop {small['bands_stop']:.1f}),",
     f"nearest-neighbour scan {k92:.1f} at 92 spectra and {k1e4:.1f} at 10,000 (log-linear in between). The Pi ran NumPy 1.24.2 and the",
     "DGX 2.5.3, so the ratios include the NumPy version. Autoencoders (PyTorch) have no Pi analogue and are not estimated. `s24_pi_estimate.py`, `pi_estimate.json`.", "",
     "| arm | Grace ms (measured) | Pi 5 ms (estimate) | range |", "|---|---|---|---|"]
for a in ("fly", "fly_rec", "fly_ctx2", "zscore", "zscore_fam", "knn_n92", "knn_n184", "knn_n678", "knn_n1356", "knn_n1664",
          "knnfam_n184", "knnfam_n678", "knnfam_n1664"):
    d = det[a]; L.append(f"| {a} | {f(d['grace_us'])} | {f(d['pi_est_us'])} | {f(d['pi_range_us'][0])}–{f(d['pi_range_us'][1])} |")
L += ["", "| serving all four jobs | Grace µs (measured) | Pi 5 µs (estimate) | range |", "|---|---|---|---|"]
for k, lbl in (("connectome", "connectome model"), ("stack", "engineered stack"), ("stack_alt", "stack with nearest neighbours")):
    d = serve[k]; L.append(f"| {lbl} | {d['grace_us']:.1f} | {d['pi_est_us']:.0f} | {d['pi_range_us'][0]:.0f}–{d['pi_range_us'][1]:.0f} |")
c, s_ = serve["connectome"], serve["stack"]
L += ["", f"The ranking does not change on the Pi. The engineered stack serves all four jobs about {c['pi_est_us'] / s_['pi_est_us']:.0f}× faster than the model "
      f"(range {c['pi_range_us'][0] / s_['pi_range_us'][1]:.0f}–{c['pi_range_us'][1] / s_['pi_range_us'][0]:.0f}×), against "
      f"{c['grace_us'] / s_['grace_us']:.0f}× measured on a Grace core; the model's code slows less on the Pi (×{r_fly_lo:.1f}–{r_fly:.1f}) than small "
      f"vector operations (×{r_small_lo:.1f}–{r_small_hi:.1f}).", ""]
open(f"{S24}/pi_estimate.md", "w").write("\n".join(L))
print("\n".join(L))
