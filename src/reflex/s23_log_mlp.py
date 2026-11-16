"""Seed 20 of the in situ stop test, small-network stop rule (as in nn_tests.py E1), logged run by run."""
import json, os, sys
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC); os.chdir(SRC)
import numpy as np, warnings
warnings.filterwarnings("ignore")
import s4_insitu as S4
from sklearn.neural_network import MLPClassifier
K = S4.K


class MlpStop(S4.KnnStop):
    def stop(self, x):
        if len({f for _, f in self.mem}) < 2: return False
        X = np.array([m for m, _ in self.mem]); y = np.array([f for _, f in self.mem], int)
        c = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit(X * 50, y)
        return c.predict_proba((x / x.sum())[None] * 50)[0, 1] > 0.5


seed = 20
Wd, Min, G = S4.H.wirings(seed)
rng = np.random.default_rng(9000 + seed); prng = np.random.default_rng(7000 + seed)
start = list(np.random.default_rng(seed).choice(K.NC, S4.N0, replace=False))
det = MlpStop()
idx, y, fl, ev = [], [], [], []
for n in range(S4.BUDGET):
    if n < S4.N0: j = int(start[n])
    else:
        s = S4.gp_ucb(idx, y).astype(float); s[idx] = -np.inf; j = int(np.argmax(s))
    probe, m = S4.simulate_probe(K.R[j], K.PAR["Mo"], rng)
    m = {k: float(np.clip(v + rng.normal(0, 0.02), 0, 1)) for k, v in m.items()}
    fail = m["x_bot_Se"] > S4.TB or m["defects"] > S4.TD
    x = S4.raman(probe, prng, G)
    stop = det.stop(x) and n >= S4.N0
    idx.append(j)
    if stop:
        y.append(0.0); fl.append(True); ev.append("saved" if fail else "false_stop")
    else:
        y.append(m["F"]); fl.append(fail); ev.append("lost" if fail else "kept"); det.learn(x, fail)
print({e: ev.count(e) for e in set(ev)})
json.dump(ev, open(os.path.join(SRC, "../../results/reflex/s23_fair_tuning", "s4_seed20_mlp.json"), "w"))
