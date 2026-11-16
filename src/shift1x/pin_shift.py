#!/usr/bin/env python3
"""Run a P4 reflex script (or the neuro_reruns signed_cord.py) with s4_lit.SHIFT_SCALE set explicitly.

s4_lit.py:33 reads SHIFT_SCALE from sys.argv[4] at import, and the callers pass WORKERS there, so the
published S18a ran at 6x and S20 at 5x. This wrapper imports s4_lit first, sets L.SHIFT_SCALE = SCALE,
guards L.raman_lit so any call with another value fails, forces the 'fork' start method so pool workers
inherit the patched module, then runs the target script unchanged as __main__ with its own argv.
SHIFT_SCALE is an explicit, required option (--shift-scale X); the worker count no longer reaches it.
P4 is read only; nothing in it is edited. A sidecar pin.json records the pin in OUT_DIR.

    python pin_shift.py --shift-scale 1 s18a   BANC_DIR OUT_DIR SEEDS [WORKERS]
    python pin_shift.py --shift-scale 1 s20    OUT_DIR SEEDS N_CAL [WORKERS]
    python pin_shift.py --shift-scale 1 signed BANC_DIR OUT_DIR SEEDS [WORKERS]
"""
import json, os, sys, runpy, subprocess, multiprocessing as mp

os.environ.setdefault("S1_MODE", "janus")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
P4 = os.path.expanduser("<repo>/src/reflex")
NR = os.path.expanduser("<workspace>/g1_visuals/neuro_reruns/src")
TARGET = {"s18a": (f"{P4}/s18a_furnace.py", 1, 3), "s20": (f"{P4}/s20_timecourse.py", 0, 3),
          "signed": (f"{NR}/signed_cord.py", 1, 3)}  # script, index of OUT_DIR, index of WORKERS in args

if len(sys.argv) < 4 or sys.argv[1] != "--shift-scale":
    raise SystemExit("usage: pin_shift.py --shift-scale X {s18a|s20|signed} ARGS...")
SCALE = float(sys.argv[2])
kind, args = sys.argv[3], sys.argv[4:]
script, i_out, i_w = TARGET[kind]
workers = int(args[i_w]) if len(args) > i_w else 4
if workers > 4:
    raise SystemExit("D-018: at most 4 workers")
mp.set_start_method("fork", force=True)
sys.path.insert(0, P4)
sys.argv = [os.path.basename(script)] + args  # what s4_lit sees at import (argv[4] is then overridden)
import s4_lit as L  # noqa: E402
L.SHIFT_SCALE = SCALE
_raman = L.raman_lit


def raman_pinned(*a, **k):
    assert L.SHIFT_SCALE == SCALE, f"SHIFT_SCALE drifted to {L.SHIFT_SCALE}"
    return _raman(*a, **k)


L.raman_lit = raman_pinned
outd = args[i_out]
os.makedirs(outd, exist_ok=True)
head = subprocess.run(["git", "-C", P4, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump(dict(kind=kind, script=script, argv=sys.argv, workers=workers, shift_scale=L.SHIFT_SCALE,
               p4_head=head, python=sys.version.split()[0], start_method=mp.get_start_method(),
               s20_arms=os.environ.get("S20_ARMS", ""), s20_tag=os.environ.get("S20_TAG", ""), s20_lstm=os.environ.get("S20_LSTM", "")),
          open(f"{outd}/pin_{kind}" + (f"_cal{args[2]}" + os.environ.get("S20_TAG", "") if kind == "s20" else "") + ".json", "w"), indent=1)
if kind == "signed":
    sys.path.insert(0, NR)
runpy.run_path(script, run_name="__main__")
