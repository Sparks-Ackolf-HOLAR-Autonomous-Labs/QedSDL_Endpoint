"""Data for Fig. 1 (anatomy panels): real FlyWire v783 morphology and wiring of the mushroom-body model.

Downloads public skeletons (precomputed://gs://flywire_v141_m783, skeletons_mip_1, no login) of the
right-hemisphere uniglomerular olfactory projection neurons (one per glomerulus), a random sample of
right-hemisphere Kenyon cells and the MBON alpha'3 output neurons, projects them to the frontal (x, y)
plane in micrometres (quantised to 0.25 um, int16 in units of the stored "q_um"), and writes them with a sample of all neuron positions (brain outline) and the
measured glomerulus x Kenyon-cell synapse-count matrix used by the model (both hemispheres, the same
PN->KC entries as load_w in s1_spectra.py) to results/anatomy/fig1_anatomy.npz.

Run on the DGX:  python -I fig1_anatomy_prep.py  (needs cloud-volume, pandas, pyarrow)
"""
import os, sys
import numpy as np
import pandas as pd

D = os.path.expanduser("~/flywire_v783")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "anatomy")
Q = 0.25  # um, quantisation of projected skeletons (far below print resolution); stored as int16 counts of Q
N_KC, SEED, VOX = 250, 0, np.array([4.0, 4.0, 40.0])  # annotation positions are 4x4x40 nm voxels

a = pd.read_csv(f"{D}/neuron_annotations.tsv", sep="\t", low_memory=False)
a["root_id"] = a.root_id.astype(np.int64)
a = a.drop_duplicates("root_id")
rng = np.random.default_rng(SEED)

# brain outline: positions of a sample of all neurons, micrometres
pos = a[["pos_x", "pos_y", "pos_z"]].dropna().values * VOX / 1000
outline = np.round(pos[rng.choice(len(pos), 40000, replace=False), :2]).astype(np.int16)  # 1 um

# measured glomerulus x KC synapse matrix (uniglomerular PNs; glomerulus = cell_type prefix)
p = a[a.cell_class == "ALPN"].sort_values("root_id")
ug = p[p.cell_sub_class == "uniglomerular"].copy()
ug["glom"] = ug.cell_type.fillna("").str.split("_").str[0]
gloms = sorted(set(ug.glom))
kc = a[a.cell_class == "Kenyon_Cell"].sort_values("root_id")
con = pd.read_parquet(f"{D}/Connectivity_783.parquet")
ki = {f: i for i, f in enumerate(kc.root_id)}
gi = dict(zip(ug.root_id, ug.glom.map({g: i for i, g in enumerate(gloms)})))
pk = con[con.Presynaptic_ID.isin(gi) & con.Postsynaptic_ID.isin(ki)]
M = np.zeros((len(gloms), len(kc)), np.int16)
np.add.at(M, (pk.Presynaptic_ID.map(gi).values, pk.Postsynaptic_ID.map(ki).values), pk.Connectivity.values)
kc_sub = np.array([str(x) for x in kc.cell_sub_class.fillna("KCapbp")], dtype="U8")
all_pn = con[con.Presynaptic_ID.isin(set(p.root_id)) & con.Postsynaptic_ID.isin(ki)]

# neurons to draw: right side
pn_r = ug[ug.side == "right"].drop_duplicates("glom")
kc_r = kc[kc.side == "right"]
kc_pick = kc_r.iloc[rng.choice(len(kc_r), N_KC, replace=False)]
mb = a[(a.cell_class == "MBON") & a.cell_type.fillna("").str.contains("MBON14|a'3", regex=True)]
mb = mb[mb.side == "right"]
print(len(pn_r), "PN,", len(kc_pick), "KC,", len(mb), "MBON a'3:", list(mb.cell_type))

from cloudvolume import CloudVolume
cv = CloudVolume("precomputed://https://storage.googleapis.com/flywire_v141_m783", use_https=True,
                 progress=False)


def segs(rid, step=1):
    sk = cv.skeleton.get(int(rid))
    v = sk.vertices / 1000.0
    e = sk.edges[::step]
    q = np.round(np.stack([v[e[:, 0], :2], v[e[:, 1], :2]], 1) / Q) * Q  # frontal plane, Q-um grid
    q = q[np.any(q[:, 0] != q[:, 1], axis=1)]
    q = np.unique(np.sort(q.view([("a", q.dtype, 2)]).reshape(-1, 2), axis=1).view(q.dtype).reshape(-1, 2, 2), axis=0)
    return np.round(q / Q).astype(np.int16)  # (n, 2 ends, x y) in units of Q um


out = dict(q_um=np.float32(Q), outline=outline, M=M, gloms=np.array([str(g) for g in gloms], dtype="U16"), kc_sub=kc_sub,
           n_pn_all=np.int64(p.root_id.nunique()), n_kc=np.int64(len(kc)),
           n_pairs_all=np.int64(len(all_pn)), n_syn_all=np.int64(all_pn.Connectivity.sum()))
for name, ids in [("pn", pn_r.root_id), ("kc", kc_pick.root_id), ("mbon", mb.root_id)]:
    ss = [segs(r) for r in ids]
    out[f"{name}_segs"] = np.concatenate(ss); out[f"{name}_n"] = np.array([len(s) for s in ss])
    out[f"{name}_ids"] = np.array(list(ids), np.int64)
    print(name, out[f"{name}_segs"].shape)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, "fig1_anatomy.npz"), **out)
print("pairs", out["n_pairs_all"], "synapses", out["n_syn_all"])
