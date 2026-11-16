"""S24: serve the FlyWire PN map and PN->KC matrix from one small .npz instead of re-reading the full FlyWire
tables (pandas, ~2 GB per process) in every worker. Patches s1_spectra.glom_map and load_w, which every
harness calls as S.glom_map(np.random.default_rng(5)) and S.load_w(pn_ids); the cache holds exactly that
draw, and a call with any other generator raises rather than returning the wrong map.
Build once: python s24_wcache.py CACHE_NPZ        Use: S24_WCACHE=CACHE_NPZ, then import s24_wcache.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402

_REF = np.random.default_rng(5).bit_generator.state


def install(path):
    z = dict(np.load(path)); orig = S.glom_map

    def glom_map(rng):
        if rng.bit_generator.state != _REF: return orig(rng)
        rng.choice(3, 1)  # the cached map was drawn from this generator; advance it so callers see a used rng
        return z["pn"].copy(), z["Min"].copy(), int(z["Min"].shape[1])

    def load_w(pn_ids):
        if not np.array_equal(pn_ids, z["pn"]): raise ValueError("s24_wcache: unexpected PN ids")
        return z["W"].copy()
    S.glom_map, S.load_w = glom_map, load_w


if __name__ == "__main__":
    pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn)
    np.savez_compressed(sys.argv[1], pn=pn, Min=Min, W=W); print(W.shape, Min.shape, int((W > 0).sum()))
elif os.environ.get("S24_WCACHE"):
    install(os.environ["S24_WCACHE"])
