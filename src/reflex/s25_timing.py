#!/usr/bin/env python3
"""S25b timing and bytes for the three edge-station options, one thread (taskset to one idle Grace core).

  G  one generic connectome model (S24 s24_timing serve_fly, re-timed here): one plain drive; 5 % code for stop,
     dose lookup and furnace read-out; context code (5 %) for flagging.
  P  one tuned connectome model per job, sharing the measured wiring: plain drive -> context code at 2.5 % (flag),
     10 % code (dose lookup over 123 calibration codes), 5 % code (furnace read-out); a second, gain-controlled
     drive (beta 1, alpha 1, n 1.5; fly2.Fly2 gain) -> 5 % code (failing-run stop).
  P4 as P but every job keeps its own copy of the wiring (four separate models): bytes only.
  E  engineered specialists: S24 timing.json (routed z-score, logistic regression, ridge, ridge).
Packed bytes as S24 (uint16 indices, uint8 counts and weights, binary memories as bits, float32 regression weights).
Edge fit: Pico (RP2040, 264 kB SRAM, 2 MB flash), Pico 2 (RP2350, 520 kB SRAM, 4 MB flash). "All in RAM" = packed
bytes + working arrays (drive float32 x n_KC, PN float32 x n_PN, per extra drive); "flash" = read-only parts
(wiring, stored codes, regression weights) in flash, RAM = writable memories + working arrays.
Usage: python s25_timing.py OUT_JSON
"""
import json, os, platform, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["S1_MODE"] = "multi"
import numpy as np  # noqa: E402
import s1_spectra as S  # noqa: E402
import s24_wcache  # noqa: E402,F401
import s7_swarm as S7  # noqa: E402
import s24_fixed_memory as A  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/reflex")
pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn).astype(np.float32)
fly = A.FlyCtx(W, Min, 2.0, 0.05); n = fly.n
k5, k25, k10 = int(0.05 * n), int(0.025 * n), int(0.10 * n)
rng = np.random.default_rng(0); NCAL = 123
av = (rng.random(n) < 0.1).astype(np.float32); ap = (rng.random(n) < 0.1).astype(np.float32)
avg = av.copy(); apg = ap.copy()
cal5 = np.zeros((NCAL, n), bool); cal10 = np.zeros((NCAL, n), bool)
for j in range(NCAL): cal5[j, rng.choice(n, k5, replace=False)] = 1; cal10[j, rng.choice(n, k10, replace=False)] = 1
calF = rng.random(NCAL); Bk = rng.normal(size=(n, 2)).astype(np.float32)


def drive(p): return np.bincount(fly.r, weights=fly.v * p[fly.c], minlength=n)
def pn_plain(x): return np.bincount(fly.pr, weights=fly.pv * x[fly.pc], minlength=fly.npn)


def pn_gain(x, beta=1.0, alpha=1.0, e=1.5):   # fly2.Fly2.pn with gain=True
    p = pn_plain(x); m = p.mean() + 1e-12; p = np.maximum(p - beta * m, 0); s = alpha * m
    return p ** e / (p ** e + s ** e)


def top(d, k): return d >= np.partition(d, -k)[-k]


def serve_G(x, f, Ft=0.5):
    d = drive(pn_plain(x)); c = top(d, k5); cc = top(d + 2.0 * d.std() * fly.cm[f], k5)
    s = float(fly.w[cc].sum() / k5); fly.w[cc] *= 0.7
    stop = av[c].mean() > ap[c].mean(); dose = calF[np.argmax(cal5[:, c].sum(1))] >= Ft; fz = Bk[c].sum(0)
    return s, stop, dose, fz


def serve_P(x, f, Ft=0.5):
    d = drive(pn_plain(x))
    cc = top(d + 2.0 * d.std() * fly.cm[f], k25); s = float(fly.w[cc].sum() / k25); fly.w[cc] *= 0.7   # flag, 2.5 %
    c10 = top(d, k10); dose = calF[np.argmax(cal10[:, c10].sum(1))] >= Ft                              # dose, 10 %
    c5 = top(d, k5); fz = Bk[c5].sum(0)                                                                # furnace, 5 %
    cg = top(drive(pn_gain(x)), k5); stop = avg[cg].mean() > apg[cg].mean()                           # stop, gain control
    return s, stop, dose, fz


st = S7.stream(1, G)
out = dict(machine=platform.machine(), node=platform.node(), numpy=np.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)))
for name, fn in (("G", serve_G), ("P", serve_P)):
    ts = []
    for i, (x, _, f) in enumerate(st):
        t = time.perf_counter(); fn(x, f); dt = time.perf_counter() - t
        if i >= 2000: ts.append(dt)
    out[f"{name}_us_per_spectrum"] = 1e6 * float(np.median(ts)); print(name, out[f"{name}_us_per_spectrum"], flush=True)
T24 = json.load(open(f"{R}/s24_fixed_memory/timing.json"))
out["E_us_per_spectrum_s24"] = T24["serve"]["stack"]["us_per_spectrum"]; out["G_us_per_spectrum_s24"] = T24["serve"]["connectome"]["us_per_spectrum"]

wiring = len(fly.r) * 5 + len(fly.pr) * 5
work = 4 * n + 4 * fly.npn          # one drive and one PN vector, float32
ro = lambda codes_k: NCAL * codes_k * 2 + NCAL * 4  # noqa: E731  stored calibration codes (uint16 indices) + fluences
Gb = dict(wiring=wiring, ctx_mask=n // 4, novelty=n, compartments=2 * n // 8, dose_codes=ro(k5), furnace=2 * (n + 1) * 4)
Pb = dict(wiring=wiring, ctx_mask=n // 4, novelty=n, compartments=2 * n // 8, dose_codes=ro(k10), furnace=2 * (n + 1) * 4)
P4b = dict(Pb, wiring=4 * wiring)
E24 = T24["serve"]["bytes_packed"]["stack"]
opts = {"G": (Gb, work, ("wiring", "ctx_mask", "dose_codes", "furnace")),
        "P": (Pb, 2 * work, ("wiring", "ctx_mask", "dose_codes", "furnace")),
        "P4": (P4b, 2 * work, ("wiring", "ctx_mask", "dose_codes", "furnace")),
        "E": ({k: v for k, v in E24.items() if k != "total"}, 4 * G, ("ii_logreg_weights", "iii_ridge", "iv_ridge"))}
fit = {}
for o, (b, wk, read_only) in opts.items():
    tot = sum(b.values()); ram_flash = sum(v for k, v in b.items() if k not in read_only) + wk
    fit[o] = dict(bytes=b, packed_total=tot, all_in_ram=tot + wk, ram_with_flash=ram_flash, flash=sum(b[k] for k in read_only if k in b),
                  pico_ram=tot + wk <= 264_000, pico2_ram=tot + wk <= 520_000,
                  pico_flash=ram_flash <= 264_000 and tot <= 2_000_000, pico2_flash=ram_flash <= 520_000 and tot <= 4_000_000)
# single-job subsets of the generic model (all in RAM)
jobs = dict(flag=("ctx_mask", "novelty"), stop=("compartments",), dose=("dose_codes",), furnace=("furnace",))
from itertools import combinations
sub = {}
for r in range(1, 5):
    for js in combinations(jobs, r):
        tot = Gb["wiring"] + sum(Gb[k] for j in js for k in jobs[j]) + work
        sub["+".join(js)] = dict(all_in_ram=tot, pico=tot <= 264_000, pico2=tot <= 520_000)
out.update(fit=fit, generic_job_subsets=sub)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(fit, indent=0)[:2000])
