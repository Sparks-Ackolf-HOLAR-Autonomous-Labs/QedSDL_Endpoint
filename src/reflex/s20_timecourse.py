#!/usr/bin/env python3
"""S20: a stop that needs the time course of the spectra, on a selenization furnace with a slow purge.

Question (from S19): the familiarity jobs judge each reading
alone, which suits the connectome model. Where should an LSTM, whose hidden state carries the last
readings, have the advantage? A task whose answer is in the rate of change, not in one spectrum.

Furnace: the S18a reduced kinetics (top-layer exchange dx_t/dt = k se (1 - x_t), bottom-layer exchange
RB k se x_t^3 (1 - x_b)), with a slow chamber purge (TAU_SE = 40 s instead of 10 s) and a wider
flake-to-flake spread of the exchange rate k (log-normal s.d. 0.6). After the Se valve closes the
chamber keeps converting the flake for about TAU_SE, by an amount set by k. A rule that closes when
the current spectrum reaches the goal over-selenizes fast flakes; closing early by the right amount
needs k, which one spectrum does not show and the last few spectra do. All rates are assumptions
chosen to make the time course matter; they are not MACE-MP-0 or kMC outputs.
Success: x_t >= 0.90 and x_b <= 0.10 after the purge. Readings every 2 s while the valve is open.

Calibration flakes (N_CAL per seed, 3 or 10): valve open for 360 s, every reading labelled with the
state now and with the final state if the valve were closed at that reading (the purge tail run
forward), as a laboratory would learn from runs closed at different times.
Arms (all close after N_CONSEC = 3 consecutive votes; thresholds are cold-start defaults):
  now_ridge    ridge on the 56 bands estimates x_t, x_b now; close at x_t >= 0.95 or x_b >= 0.06 (S18a rule)
  snap_ridge   ridge on the bands of one reading predicts the final x_t, x_b if closed now;
               close when predicted final x_t >= XT_GOAL or x_b >= XB_GOAL
  snap_kc      same target, ridge on the Kenyon-cell code of one reading (FlyWire wiring, 5 % active)
  snap_mlp     same target, perceptron 2 x 64 on the bands of one reading
  seq_ridge    same target, ridge on the bands of the last W = 8 readings stacked (linear time course)
  seq_kc       same target, ridge on the Kenyon-cell codes of the current reading and of the reading
               W - 1 steps earlier (the connectome model given a short trace)
  lstm         same target, LSTM (16 hidden units) over the bands of the last W readings
  trend_rule   engineered: x_t now from now_ridge, rate from a line fit to the last W estimates,
               k = rate / (1 - x_t), predicted final x_t = 1 - (1 - x_t) exp(-k TAU_SE); close when
               >= XT_GOAL or estimated x_b >= 0.06
  oracle_k     knows k and the true state; closes when the computed final x_t >= XT_GOAL (upper bound)
  net_gru, net_peep, net_mog, net_tfm, net_memnet   (S21, 14 Oct 2026; only when named in S20_ARMS)
               same target as lstm, with a GRU, peephole LSTM, Mogrifier LSTM or one-layer transformer
               over the last W readings (memnets.py, same lr / epochs / width as lstm), or a memory-augmented
               regressor: a GRU encodes each window, the memory holds every calibration window with its
               label, and the estimate is the attention-weighted label (trained with each window's own
               entry masked, as in matching networks)
Options (environment): S20_LSTM="lr,epochs,hidden" (default 3e-3,600,16); S20_ARMS=comma list to run
only some arms (default all); S20_TAG output suffix.
Usage: python s20_timecourse.py OUT_DIR SEEDS N_CAL [WORKERS]
"""
import json, os, sys, time
os.environ.setdefault("S1_MODE", "janus")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_lit as L  # noqa: E402
import s1_spectra as S  # noqa: E402

DT, T_MAX, T_C, G = 2.0, 600.0, 450.0, 56
K0, SIG_K, RB, TAU_SE = 0.025, 0.6, 0.03, 40.0
XT_OK, XB_OK = 0.90, 0.10
XT_NOW, XB_NOW = 0.95, 0.06
XT_GOAL, XB_GOAL = 0.93, 0.07
N_RUN, N_CONSEC, W = 30, 3, 8
_l = os.environ.get("S20_LSTM", "3e-3,600,16").split(","); LSTM_LR, LSTM_EP, LSTM_H = float(_l[0]), int(_l[1]), int(_l[2])
NEW_SEQ = ["net_gru", "net_peep", "net_mog", "net_tfm", "net_memnet"]  # S21 (14 Oct 2026)
ONLY = [a for a in os.environ.get("S20_ARMS", "").split(",") if a]; TAG = os.environ.get("S20_TAG", "")


def spectrum(xt, xb, rng):
    st = dict(d_lat=0.0, x_top=xt, f_mos2=1 - xt, f_janus=xt * (1 - xb), f_mose2=xt * xb, theta=0.0)
    b = L.raman_lit(st, rng, G, T_C)
    return b / (b.sum() + 1e-12)


def purge(xt, xb, se, k):
    """Final state if the valve closes now."""
    while se >= 0.01:
        se += (0 - se) * (1 - np.exp(-DT / TAU_SE))
        xt += k * se * (1 - xt) * DT; xb += RB * k * se * xt ** 3 * (1 - xb) * DT
    return xt, xb


def calib(rng, ks):
    seqs = []
    for k in ks:
        xt = xb = se = 0.0; t = 0.0; X, Ynow, Yfin = [], [], []
        while t < 360:
            se += (1 - se) * (1 - np.exp(-DT / TAU_SE)); xt += k * se * (1 - xt) * DT
            xb += RB * k * se * xt ** 3 * (1 - xb) * DT; t += DT
            X.append(spectrum(xt, xb, rng)); Ynow.append((xt, xb)); Yfin.append(purge(xt, xb, se, k))
        seqs.append((np.array(X), np.array(Ynow), np.array(Yfin)))
    return seqs


def ridge(F, Y, lam=1e-2):
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    if Z.shape[1] > Z.shape[0]:  # dual form for wide features (Kenyon-cell codes)
        a = np.linalg.solve(Z @ Z.T + lam * len(Z) * np.eye(len(Z)), Y - Y.mean(0)); B = Z.T @ a
    else:
        B = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    return lambda f: ((f - mu) / sd) @ B + Y.mean(0)


def windows(X):
    """Stack of the last W readings for every reading (start padded with the first reading)."""
    P = np.concatenate([np.repeat(X[:1], W - 1, 0), X]); return np.stack([P[i:i + W] for i in range(len(X))])


def fit_torch(kind, Xs, Y, seed, epochs=600, lr=3e-3, hid=16):
    import torch
    import memnets as MN
    torch.manual_seed(seed); torch.set_num_threads(1)
    Xt = torch.tensor(Xs, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    mu, sd = Xt.reshape(-1, Xt.shape[-1]).mean(0), Xt.reshape(-1, Xt.shape[-1]).std(0) + 1e-6
    if kind == "mlp":
        net = torch.nn.Sequential(torch.nn.Linear(G, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        f = lambda x: net((x - mu) / sd)  # noqa: E731
        mods = [net]
    elif kind == "memnet":  # S21: window encoder (GRU) + memory of every calibration window with its label
        enc = MN.GRUc(G, hid); temp = torch.nn.Parameter(torch.tensor(0.0)); mods = [enc]
        Ym = Yt.mean(0)

        def att(q, keys, vals, mask_self=False):
            d = torch.cdist(q, keys) ** 2 * torch.exp(temp)
            if mask_self: d = d + torch.eye(len(q)) * 1e9
            return torch.softmax(-d, -1) @ (vals - Ym) + Ym
        f_train = lambda x: att(enc((x - mu) / sd), enc((x - mu) / sd), Yt, True)  # noqa: E731
        opt = torch.optim.Adam(list(enc.parameters()) + [temp], lr=lr)
        for _ in range(epochs):
            opt.zero_grad(); loss = ((f_train(Xt) - Yt) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            K = enc((Xt - mu) / sd)

        def pred(x):
            with torch.no_grad():
                return att(enc((torch.tensor(np.asarray(x)[None], dtype=torch.float32) - mu) / sd), K, Yt)[0].numpy()
        return pred, sum(p.numel() for p in enc.parameters()) + 1 + K.numel() + Yt.numel()
    else:  # sequence cells: lstm (torch), gru, peep, mog, tfm (memnets.py)
        if kind == "lstm":
            lstm = torch.nn.LSTM(G, hid, batch_first=True); cell = lambda z: lstm(z)[0][:, -1]; mods = [lstm]  # noqa: E731,E702
        else:
            c = MN.SEQ[kind](G, hid); cell = c; mods = [c]
        head = torch.nn.Linear(hid, 2); mods.append(head)
        f = lambda x: head(cell((x - mu) / sd))  # noqa: E731
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((f(Xt) - Yt) ** 2).mean(); loss.backward(); opt.step()
    nparams = sum(p.numel() for m in mods for p in m.parameters())
    with torch.no_grad():
        return (lambda x: f(torch.tensor(np.asarray(x)[None], dtype=torch.float32))[0].detach().numpy()), nparams


def run(args):
    seed, n_cal = args
    rng = np.random.default_rng(20000 + seed)
    ks_cal = K0 * np.exp(rng.normal(0, SIG_K, n_cal)); ks = K0 * np.exp(rng.normal(0, SIG_K, N_RUN))
    cal = calib(np.random.default_rng(20100 + seed), ks_cal)
    X = np.concatenate([c[0] for c in cal]); Ynow = np.concatenate([c[1] for c in cal]); Yfin = np.concatenate([c[2] for c in cal])
    Xw = np.concatenate([windows(c[0]) for c in cal])
    pn_ids, Min, _ = S.glom_map(np.random.default_rng(5)); fly = S.Fly(S.load_w(pn_ids), Min)
    code = lambda x: fly.code(x).astype(float)  # noqa: E731
    C = np.array([code(x) for x in X])
    Cw = np.concatenate([np.stack([np.array([code(x) for x in w[[0, -1]]]).ravel() for w in windows(c[0])]) for c in cal])
    t0 = time.perf_counter()
    m_now = ridge(X, Ynow); m_snap = ridge(X, Yfin); m_kc = ridge(C, Yfin)
    m_seq = ridge(Xw.reshape(len(Xw), -1), Yfin); m_seqkc = ridge(Cw, Yfin)
    want = lambda a: not ONLY or a in ONLY  # noqa: E731
    n_mlp = n_lstm = None; nets = {}
    if want("snap_mlp"): m_mlp, n_mlp = fit_torch("mlp", X, Yfin, seed)
    if want("lstm"): m_lstm, n_lstm = fit_torch("lstm", Xw, Yfin, seed, LSTM_EP, LSTM_LR, LSTM_H)
    for a in NEW_SEQ:  # S21 arms, run only when named in S20_ARMS
        if ONLY and a in ONLY: nets[a] = fit_torch(a[4:], Xw, Yfin, seed, LSTM_EP, LSTM_LR, LSTM_H)
    fit_s = time.perf_counter() - t0
    est = dict(
        now_ridge=lambda h: m_now(h[-1]), snap_ridge=lambda h: m_snap(h[-1]), snap_kc=lambda h: m_kc(code(h[-1])),
        snap_mlp=lambda h: m_mlp(h[-1]), seq_ridge=lambda h: m_seq(np.ravel(h)),
        seq_kc=lambda h: m_seqkc(np.r_[code(h[0]), code(h[-1])]), lstm=lambda h: m_lstm(np.array(h)))
    for a, (m, _) in nets.items(): est[a] = (lambda mm: lambda h: mm(np.array(h)))(m)
    goal = {a: ((XT_NOW, XB_NOW) if a == "now_ridge" else (XT_GOAL, XB_GOAL)) for a in est}

    def run_flake(k, prng, arm):
        xt = xb = se = 0.0; t = 0.0; u = 1.0; hist, votes, xs = [], [], []
        while t < T_MAX + 400:
            if u > 0 and t >= T_MAX: u = 0.0
            se += (u - se) * (1 - np.exp(-DT / TAU_SE))
            xt += k * se * (1 - xt) * DT; xb += RB * k * se * xt ** 3 * (1 - xb) * DT; t += DT
            if u == 0:
                if se < 0.01: break
                continue
            x = spectrum(xt, xb, prng); hist.append(x); hist = hist[-W:]
            h = [hist[0]] * (W - len(hist)) + hist
            if arm == "oracle_k":
                v = purge(xt, xb, se, k)[0] >= XT_GOAL
            elif arm == "trend_rule":
                p = m_now(x); xs.append(p[0]); xs = xs[-W:]
                if len(xs) >= 3:
                    rate = np.polyfit(np.arange(len(xs)) * DT, xs, 1)[0]; kk = max(rate, 0) / max(1 - p[0], 0.02)
                    v = (1 - (1 - p[0]) * np.exp(-kk * TAU_SE) >= XT_GOAL) or p[1] >= XB_NOW
                else:
                    v = False
            else:
                p = est[arm](h); g = goal[arm]; v = p[0] >= g[0] or p[1] >= g[1]
            votes.append(bool(v))
            if len(votes) >= N_CONSEC and all(votes[-N_CONSEC:]): u = 0.0
        return dict(ok=bool(xt >= XT_OK and xb <= XB_OK), under=bool(xt < XT_OK), over=bool(xb > XB_OK),
                    xt=float(xt), xb=float(xb))
    out = {}
    for arm in [a for a in list(est) + ["trend_rule", "oracle_k"] if not ONLY or a in ONLY]:
        prng = np.random.default_rng(20400 + seed)
        rs = [run_flake(k, prng, arm) for k in ks]
        out[arm] = dict(lost=float(sum(not r["ok"] for r in rs)), under=float(sum(r["under"] for r in rs)),
                        over=float(sum(r["over"] for r in rs)),
                        lost_fast=float(sum(not r["ok"] for r, k in zip(rs, ks) if k > K0 * np.exp(0.6))),
                        n_fast=int(sum(k > K0 * np.exp(0.6) for k in ks)))
    # estimator error on held-out flakes: final x_t if closed now
    hold = calib(np.random.default_rng(20500 + seed), K0 * np.exp(np.random.default_rng(20600 + seed).normal(0, SIG_K, 3)))
    err = {}
    for arm, f in est.items():
        if arm == "now_ridge" or (ONLY and arm not in ONLY): continue
        e = []
        for Xh, _, Yh in hold:
            Wh = windows(Xh); e += [abs(f(list(w))[0] - y[0]) for w, y in zip(Wh, Yh)]
        err[arm] = float(np.mean(e))
    return dict(seed=seed, n_cal=n_cal, arms=out, final_xt_mae=err, fit_s=fit_s,
                params=dict(mlp=n_mlp, lstm=n_lstm, **{a: n for a, (_, n) in nets.items()}),
                k_runs=ks.tolist())


if __name__ == "__main__":
    outd, seeds, n_cal = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], int(sys.argv[3])
    os.makedirs(outd, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex:
        for r in ex.map(run, [(s, n_cal) for s in seeds]):
            json.dump(r, open(f"{outd}/s20_cal{n_cal}_s{r['seed']}{TAG}.json", "w"), indent=1)
            print(n_cal, r["seed"], {a: d["lost"] for a, d in r["arms"].items()},
                  {a: round(v, 3) for a, v in r["final_xt_mae"].items()}, round(r["fit_s"], 1), flush=True)
