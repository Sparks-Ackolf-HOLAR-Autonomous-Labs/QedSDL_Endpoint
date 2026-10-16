#!/usr/bin/env python3
"""S13 preparation: projection-neuron (PN) to Kenyon-cell (KC) wiring of three animals on one glomerulus list.

Animals and public sources (no login needed):
  fafb    FlyWire FAFB v783, female brain. Connectivity_783.parquet (Shiu et al. packaging) and
          neuron_annotations.tsv (Schlegel et al.), as used in S1-S12.
  banc    BANC v888, female brain and nerve cord. gs://lee-lab_brain-and-nerve-cord-fly-connectome:
          compiled_data/banc_888/banc_888_meta.feather and
          neuron_connectivity/v888/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet
          (Harvard Dataverse doi:10.7910/DVN/7WTH1N).
  malecns MaleCNS v1.0, male brain and nerve cord. gs://flyem-male-cns/v1.0/connectome-data/flat-connectome:
          body-annotations-male-cns-v1.0-minconf-0.5.feather and
          connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather.

Common rules, so the three wirings differ only by animal:
  * Synapse threshold: a PN-KC pair counts if it has >= 3 synapses (the BANC table is released at this
    threshold). FAFB is also kept unthresholded ("fafb_raw") to link back to S1-S12.
  * Glomerulus list: the uniglomerular glomeruli (cell type prefix before "_", as in S1) that have at
    least one uniglomerular PN in every animal, sorted by name. G = its length = number of spectral bands.
  * A PN is uniglomerular if its type prefix is on that list and the type is not multiglomerular
    ("M_" prefix or "+" in the prefix); it reads its glomerulus band. Other uniglomerular PNs (glomerulus
    missing in some animal) get no input. Multiglomerular PNs read the mean of 3 random bands (fixed rng 5).
  * Side: soma side of each PN and KC (FAFB "side", BANC "side", MaleCNS "somaSide"). A cross-side pair
    is a PN on one side synapsing onto a KC on the other.
Output: OUT/wiring_<animal>.npz with W (KC x PN synapse counts), Min (PN x G), kc_side, pn_side, and
OUT/wiring_stats.json.
Usage: python s13_prep.py FAFB_DIR BANC_DIR MALECNS_DIR OUT_DIR
"""
import json, os, sys
import numpy as np
import pandas as pd

THR = 3


def prefix(t):
    return str(t).split("_")[0]


def is_multi(t):
    t = str(t)
    return t.startswith("M_") or "+" in prefix(t)


def fafb(d):
    a = pd.read_csv(f"{d}/neuron_annotations.tsv", sep="\t", low_memory=False)
    a["root_id"] = a.root_id.astype(np.int64)
    pn = a[a.cell_class == "ALPN"].drop_duplicates("root_id")
    kc = a[a.cell_class == "Kenyon_Cell"].drop_duplicates("root_id")
    uni = pn.cell_sub_class == "uniglomerular"
    con = pd.read_parquet(f"{d}/Connectivity_783.parquet")
    e = con[con.Presynaptic_ID.isin(set(pn.root_id)) & con.Postsynaptic_ID.isin(set(kc.root_id))]
    e = e.rename(columns={"Presynaptic_ID": "pre", "Postsynaptic_ID": "post", "Connectivity": "n"})
    return (pd.DataFrame(dict(id=pn.root_id.values, type=pn.cell_type.fillna("").values, side=pn.side.values,
                              uni=uni.values)),
            pd.DataFrame(dict(id=kc.root_id.values, side=kc.side.values)), e[["pre", "post", "n"]])


def banc(d):
    m = pd.read_feather(f"{d}/banc_888_meta.feather")
    m = m.dropna(subset=["banc_888_id"]); m["id"] = m.banc_888_id.astype(np.int64)
    pn = m[m.cell_class == "antennal_lobe_projection_neuron"].drop_duplicates("id")
    kc = m[m.cell_class == "kenyon_cell"].drop_duplicates("id")
    uni = pn.cell_sub_class == "uniglomerular_projection_neuron"
    s = pd.read_parquet(f"{d}/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet")
    e = s[s.pre_root_id.isin(set(pn.id)) & s.post_root_id.isin(set(kc.id))]
    e = e.rename(columns={"pre_root_id": "pre", "post_root_id": "post", "num_synapses": "n"})
    return (pd.DataFrame(dict(id=pn.id.values, type=pn.cell_type.fillna("").values, side=pn.side.values, uni=uni.values)),
            pd.DataFrame(dict(id=kc.id.values, side=kc.side.values)), e[["pre", "post", "n"]])


def malecns(d):
    m = pd.read_feather(f"{d}/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    sd = {"L": "left", "R": "right"}
    pn = m[m["class"] == "ALPN"].drop_duplicates("bodyId")
    kc = m[m["class"] == "Kenyon_Cell"].drop_duplicates("bodyId")
    w = pd.read_feather(f"{d}/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                        columns=["body_pre", "body_post", "weight"])
    e = w[w.body_pre.isin(set(pn.bodyId)) & w.body_post.isin(set(kc.bodyId))]
    e = e.rename(columns={"body_pre": "pre", "body_post": "post", "weight": "n"})
    # no sub-class column: uniglomerular decided below from the type name, the same rule for every animal
    return (pd.DataFrame(dict(id=pn.bodyId.values, type=pn.type.fillna("").values, side=pn.somaSide.map(sd).values,
                              uni=~pn.type.fillna("M_").map(is_multi).values)),
            pd.DataFrame(dict(id=kc.bodyId.values, side=kc.somaSide.map(sd).values)), e[["pre", "post", "n"]])


def build(pn, kc, e, gl, thr):
    pn = pn.sort_values("id").reset_index(drop=True); kc = kc.sort_values("id").reset_index(drop=True)
    G = len(gl); gi = {g: i for i, g in enumerate(gl)}; rng = np.random.default_rng(5)
    Min = np.zeros((len(pn), G)); kind = []
    for r, (t, u) in enumerate(zip(pn.type, pn.uni)):
        g = prefix(t)
        if u and not is_multi(t) and g in gi: Min[r, gi[g]] = 1; kind.append("uni")
        elif u and not is_multi(t): kind.append("uni_unshared")  # glomerulus missing in another animal: silent
        else: Min[r, rng.choice(G, 3, replace=False)] = 1 / 3; kind.append("multi")
    e = e[e.n >= thr]
    pi = {f: i for i, f in enumerate(pn.id)}; ki = {f: i for i, f in enumerate(kc.id)}
    W = np.zeros((len(kc), len(pn)))
    np.add.at(W, (e.post.map(ki).values, e.pre.map(pi).values), e.n.values)
    ks, ps = kc.side.fillna("").values.astype(str), pn.side.fillna("").values.astype(str)
    r, c = np.nonzero(W); cross = ks[r] != ps[c]
    stats = dict(n_pn=len(pn), n_pn_uni_shared=kind.count("uni"), n_pn_uni_unshared=kind.count("uni_unshared"),
                 n_pn_multi=kind.count("multi"), n_kc=len(kc),
                 n_kc_side={s: int((ks == s).sum()) for s in sorted(set(ks))},
                 n_pn_side={s: int((ps == s).sum()) for s in sorted(set(ps))},
                 pairs=int(len(r)), synapses=int(W.sum()), cross_side_pairs=int(cross.sum()),
                 cross_side_synapses=int(W[r[cross], c[cross]].sum()),
                 kc_with_input=int((W.sum(1) > 0).sum()), mean_pn_per_kc=float((W > 0).sum(1).mean()))
    return W, Min, ks, ps, stats


if __name__ == "__main__":
    fd, bd, md, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    raw = {"fafb": fafb(fd), "banc": banc(bd), "malecns": malecns(md)}
    sets = []
    for a, (pn, kc, e) in raw.items():
        u = pn[pn.uni & ~pn.type.map(is_multi)]
        sets.append(set(u.type.map(prefix)) - {""})
    gl = sorted(set.intersection(*sets))
    stats = dict(glomeruli=gl, G=len(gl), threshold=THR, per_animal_uni_glomeruli={a: len(s) for a, s in zip(raw, sets)})
    for a, (pn, kc, e) in raw.items():
        for name, thr in ((a, THR),) + ((("fafb_raw", 1),) if a == "fafb" else ()):
            W, Min, ks, ps, st = build(pn, kc, e, gl, thr)
            np.savez_compressed(f"{out}/wiring_{name}.npz", W=W, Min=Min, kc_side=np.asarray(ks, dtype="U8"), pn_side=np.asarray(ps, dtype="U8"))
            stats[name] = st; print(name, st, flush=True)
    json.dump(stats, open(f"{out}/wiring_stats.json", "w"), indent=1)
