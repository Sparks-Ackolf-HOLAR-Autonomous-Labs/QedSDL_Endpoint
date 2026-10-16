"""Summary tables for S2-S8. Usage: python analyze_s2s8.py RESULTS_REFLEX_DIR"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
D = sys.argv[1]


def load(sub, pat):
    return [json.load(open(f)) for f in sorted(glob.glob(f"{D}/{sub}/{pat}"), key=lambda s: int(s.split("_s")[-1][:-5]))]


def p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return "n/a" if np.allclose(a, b) else f"{wilcoxon(a, b).pvalue:.2g}"


def stream_table(R, title, ref="fly", extra_kind="MoS2/WSe2_stack", zref="zscore"):
    names = list(R[0]["det"]); g = lambda n, k: np.array([r["det"][n][k] for r in R])
    has = extra_kind in R[0]["det"][ref]["auroc_by_kind"]
    L = [f"## {title} ({len(R)} seeds)", "", "| Detector | AUROC mean | IQR | false alarms after laser drop | "
         + ("stack AUROC | " if has else "") + "stored numbers | p (AUROC vs fly) |", "|---" * (6 + has) + "|"]
    for n in names:
        a = g(n, "auroc")
        st = f"{np.mean([r['det'][n]['auroc_by_kind'][extra_kind] for r in R]):.2f} | " if has else ""
        L.append(f"| {n} | {a.mean():.3f} | {np.quantile(a,.25):.3f} to {np.quantile(a,.75):.3f} | "
                 f"{g(n,'false_alarm_after_change').mean():.3f} | {st}{int(g(n,'stored_numbers').mean())} | {'' if n == ref else p(a, g(ref,'auroc'))} |")
    fa = lambda n: g(n, "false_alarm_after_change")
    L += ["", "Paired Wilcoxon, false alarms after the drop: " + "; ".join(
        f"{n} vs {zref} p = {p(fa(n), fa(zref))} (mean diff {np.mean(fa(n)-fa(zref)):+.3f})" for n in names if n.startswith("fly"))]
    return L


out = []
for mode, t in (("janus", "S2/S3 on the Janus MoSSe line (S1 stream)"), ("multi", "S2/S3 at the four-family facility (S1b stream)")):
    R = load("s2_s3", f"s2_{mode}_s*.json")
    if R: out += stream_table(R, t) + [""]
R = load("s8_context", "s8_s*.json")
if R:
    out += stream_table(R, "S8 context input vs routed team (S1b stream)", zref="zscore_fam") + [""]
    g = lambda n: np.array([r["det"][n]["auroc"] for r in R])
    out += ["Paired: " + "; ".join(f"{a} vs {b} p = {p(g(a), g(b))} ({np.mean(g(a)-g(b)):+.3f})" for a, b in
            [("fly_ctx4", "fam4_quarter"), ("fly_ctx4", "zscore_fam"), ("fly_comp", "fam4_full"), ("fly_ctx4", "fly_ctx4_shuf"), ("fly_ctx4", "fly_ctx4_rand")]
            if a in R[0]["det"] and b in R[0]["det"]), ""]
R = load("s4_insitu", "s4_s*.json")
if R:
    arms = list(R[0]["arms"]); g = lambda a, k: np.array([r["arms"][a][k] for r in R])
    out += [f"## S4 in-situ Raman stop, Janus MoSSe selenization ({len(R)} seeds, 30 recipes each)", "",
            "| Arm | flakes lost | IQR | runs stopped that would have failed | false stops | furnace time (runs) | gap | p (lost vs gp) |", "|---" * 8 + "|"]
    for a in arms:
        l = g(a, "lost")
        out.append(f"| {a} | {l.mean():.2f} | {np.quantile(l,.25):.0f} to {np.quantile(l,.75):.0f} | {g(a,'saved').mean():.2f} | {g(a,'false_stops').mean():.2f} | "
                   f"{g(a,'furnace_runs').mean():.1f} | {g(a,'gap').mean():.4f} | {'' if a == 'gp' else p(l, g('gp','lost'))} |")
    out += ["", "Paired on flakes lost: " + "; ".join(f"{a} vs {b} p = {p(g(a,'lost'), g(b,'lost'))} ({np.mean(g(a,'lost')-g(b,'lost')):+.2f})"
            for a, b in [("gp+fly_best", "gp+knn"), ("gp+fly_best", "gp+logreg"), ("gp+fly_best", "cbo"), ("gp+fly", "gp+knn"), ("gp+fly_best", "gp+fly_best_shuf"), ("gp+fly_best", "gp+fly_best_rand")]), ""]
R = load("s7b_stations", "s7b_s*.json")
if R:
    names = list(R[0]["det"]); g = lambda n, k: np.array([r["det"][n][k] for r in R])
    out += [f"## S7b three Raman stations, private vs shared memory ({len(R)} seeds)", "",
            "| Arm | AUROC new events | false alarms, network-known family (all / first 50) | false alarms, own families | MoTe2 recall at A | MoTe2 recall at B (all / first 5) | stored numbers |", "|---" * 7 + "|"]
    for n in names:
        out.append(f"| {n} | {g(n,'auroc').mean():.3f} | {g(n,'fa_network_known').mean():.3f} / {g(n,'fa_network_known_first50').mean():.3f} | "
                   f"{g(n,'fa_station_normal').mean():.3f} | {g(n,'mote2_recall_A').mean():.2f} | {g(n,'mote2_recall_B').mean():.2f} / {g(n,'mote2_recall_B_first5').mean():.2f} | {int(g(n,'stored_numbers').mean())} |")
    out += ["", "Paired (shared vs private): " + "; ".join(
        f"{b}: AUROC p = {p(g(b+'_shared','auroc'), g(b+'_private','auroc'))}, first-sighting false alarms p = {p(g(b+'_shared','fa_network_known'), g(b+'_private','fa_network_known'))}, "
        f"MoTe2 at B p = {p(g(b+'_shared','mote2_recall_B'), g(b+'_private','mote2_recall_B'))}" for b in ("fly", "fly_best", "zscore", "knn_budget")), ""]
R = load("s6_one_module", "s6_s*.json")
if R:
    names = list(R[0]["det"]); g = lambda n, k: np.array([r["det"][n][k] for r in R])
    out += [f"## S6 one module, several jobs ({len(R)} seeds, context gain {R[0]['ctx_gain']})", "",
            "| Detector | J1 new phase AUROC | J2 wrong sample AUROC | stack AUROC | stored numbers |", "|---" * 5 + "|"]
    for n in names:
        out.append(f"| {n} | {g(n,'j1_new_phase').mean():.3f} | {g(n,'j2_wrong_sample').mean():.3f} | {g(n,'stack').mean():.2f} | {int(g(n,'stored_numbers').mean())} |")
    out += ["", "Paired: " + "; ".join(f"{a} vs {b} {k} p = {p(g(a,k), g(b,k))} ({np.mean(g(a,k)-g(b,k)):+.3f})" for a, b, k in
            [("module_ctx", "zscore_fam", "j1_new_phase"), ("module_ctx", "zscore_fam", "j2_wrong_sample"), ("module_ctx", "centroid", "j2_wrong_sample"),
             ("module_ctx", "module_ctx_shuf", "j1_new_phase"), ("module_ctx", "module_ctx_rand", "j1_new_phase")]), ""]
S5 = sorted(glob.glob(f"{D}/s5_long/s5_*.json"))
if S5:
    out += ["## S5 long streams (10^4-reading blocks)", ""]
    for f in S5:
        r = json.load(open(f)); out += [f"### {os.path.basename(f)}: {r['host']} ({r['machine']}), seed {r['seed']}, {r['N']} readings, peak memory {r['peak_rss_MB']:.0f} MB", "",
                                        "| Detector | AUROC first / last block | false alarms first / last block | ms per reading first / last block | stored numbers |", "|---" * 5 + "|"]
        for n, d in r["det"].items():
            b = d["blocks"]
            out.append(f"| {n} | {b[0]['auroc']:.3f} / {b[-1]['auroc']:.3f} | {b[0]['fa']:.3f} / {b[-1]['fa']:.3f} | {b[0]['ms']:.2f} / {b[-1]['ms']:.2f} | {d['stored_numbers']} |")
        out.append("")
open(f"{D}/S2_S8_SUMMARY.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
