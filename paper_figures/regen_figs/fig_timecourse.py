"""Fig. 4: what a short memory of the time course adds, and how tuned networks compare.
(a) Slow-purge selenization furnace (S20): flakes lost of 30 per stop rule, 20 seeds, with 3 and 10
calibration flakes (median and interquartile range over seeds; diamond = mean).
(b) Four-family facility (S1b, S19, S19b): AUROC for new events of the online networks with the first
setting and with the setting chosen on separate tuning seeds, against the connectome model.
Reads results/reflex/s20_timecourse/*.json, results/reflex/s1b_multi, results/reflex/s19_online_nets."""
import glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_fig_style import apply_style, save, panel, RES, IOP_FULL, C, INK, GREY

apply_style()
NET = "#882255"
D = os.path.join(RES, "reflex", "s20_timecourse")
ARMS = [("now_ridge", "ridge, state now", C["lin"]), ("snap_ridge", "ridge, one spectrum", C["lin"]),
        ("snap_mlp", "perceptron, one spectrum", NET), ("snap_kc", "sparse code, one spectrum", C["conn"]),
        ("seq_ridge", "ridge, last 8 spectra", C["lin"]), ("lstm_tuned", "LSTM, last 8 spectra", NET),
        ("seq_kc", "sparse code, now + 14 s ago", C["conn"]), ("trend_rule", "rate rule from estimates", C["phys"]),
        ("oracle_k", "known rate (bound)", INK)]


def lost(nc):
    out = {a: [] for a, _, _ in ARMS}
    for f in sorted(glob.glob(f"{D}/s20_cal{nc}_s*[0-9].json")):
        r = json.load(open(f)); t = json.load(open(f[:-5] + "_lstmtuned.json"))
        r["arms"]["lstm_tuned"] = t["arms"]["lstm"]
        for a in out: out[a].append(r["arms"][a]["lost"])
    return out


fig = plt.figure(figsize=(IOP_FULL, 2.9)); a = fig.add_axes([0.29, 0.15, 0.27, 0.74]); b = fig.add_axes([0.765, 0.15, 0.225, 0.74])
y = np.arange(len(ARMS))[::-1]
for nc, dy, mk in ((3, 0.17, "o"), (10, -0.17, "s")):
    L = lost(nc)
    for yi, (k, _, col) in zip(y, ARMS):
        q = np.percentile(L[k], [25, 50, 75])
        a.plot([q[0], q[2]], [yi + dy] * 2, color=col, lw=1.6, alpha=0.45 if nc == 3 else 0.9, solid_capstyle="butt")
        a.plot(np.mean(L[k]), yi + dy, marker=mk, ms=3.2, color=col, mfc="white" if nc == 3 else col, mew=0.8, ls="")
a.set_yticks(y); a.set_yticklabels([lab for _, lab, _ in ARMS]); a.set_xlabel("flakes lost of 30")
a.plot([], [], "o", color=GREY, mfc="white", ms=3.2, label="3 calibration flakes")
a.plot([], [], "s", color=GREY, ms=3.2, label="10 calibration flakes")
a.legend(loc="upper right", bbox_to_anchor=(1.02, 0.78), handletextpad=0.2); a.set_xlim(-0.3, 16); panel(a, "a")

R = os.path.join(RES, "reflex")
s1 = {int(re.findall(r"_s(\d+)\.json", f)[0]): json.load(open(f))["det"] for f in glob.glob(f"{R}/s1b_multi/s1_s[0-9]*.json")}
old, new = {}, {}
for f in glob.glob(f"{R}/s19_online_nets/s19_s1b_s*_*.json"):
    j = json.load(open(f)); old.setdefault(list(j["det"])[0], []).append(list(j["det"].values())[0]["auroc"])
for f in glob.glob(f"{R}/s19_online_nets/tuned/s19_s1b_s*.json"):
    j = json.load(open(f)); det = list(j["det"])[0]; tag = os.path.basename(f).split(f"_{det}")[1][:-5]
    new.setdefault(det + tag, []).append(j["det"][det]["auroc"])
nets = [("ae16", "ae16_t_rp0", "autoencoder 16"), ("ae46", "ae46_t_rp0", "autoencoder 46"),
        ("lstmae16", "lstmae16_t_rp0", "LSTM, reconstructing"), (None, "ae16_t_rp59", "autoencoder 16\n+ replay")]
fly = np.mean([s1[s]["fly"]["auroc"] for s in s1 if s <= 20])
b.axvline(fly, color=C["conn"], lw=1.0, label="connectome model")
yy = np.arange(len(nets))[::-1]
for yi, (o, n, lab) in zip(yy, nets):
    if o:
        b.plot(np.mean(old[o]), yi, "o", mfc="white", mec=NET, ms=3.5, mew=0.8)
        b.annotate("", xy=(np.mean(new[n]), yi), xytext=(np.mean(old[o]), yi), arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
    q = np.percentile(new[n], [25, 75]); b.plot(q, [yi] * 2, color=NET, lw=1.6, alpha=0.6)
    b.plot(np.mean(new[n]), yi, "o", color=NET, ms=3.5)
b.plot([], [], "o", mfc="white", mec=NET, ms=3.5, label="first setting"); b.plot([], [], "o", color=NET, ms=3.5, label="tuned")
b.set_yticks(yy); b.set_yticklabels([lab for _, _, lab in nets]); b.set_xlabel("AUROC, new events")
b.set_xlim(0.5, 0.92); b.legend(loc="lower left", bbox_to_anchor=(-0.03, 0.09), handletextpad=0.3, handlelength=1.2); panel(b, "b")
save(fig, "fig_timecourse")
