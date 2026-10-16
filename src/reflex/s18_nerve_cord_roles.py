#!/usr/bin/env python3
"""S18: what the ventral nerve cord (VNC) does in BANC v888, as a template for an instrument-side controller.

Pairs with >= 3 synapses (file already thresholded). Measures, for the three nerve-cord roles:
  1. local reflex arcs: share of motor-neuron (MN) input from VNC sensory neurons directly or via one VNC
     intrinsic neuron, i.e. paths that never enter the brain;
  2. command bottleneck: descending neurons (DN) vs central-brain neurons, DN output landing in the VNC,
     and how many MNs each DN reaches in <= 2 synaptic steps;
  3. state report: ascending neurons (AN) vs VNC sensory neurons, and where AN output lands in the brain.
MaleCNS v1.0 added 12 Oct 2026 (same roles from its superclass/class annotations, weights >= 3 to match BANC).
Usage: python s18_nerve_cord_roles.py BANC_DIR OUT_JSON [banc|malecns]
"""
import json, sys
import numpy as np
import pandas as pd

bd, out = sys.argv[1:3]
animal = sys.argv[3] if len(sys.argv) > 3 else "banc"


def load_banc():
    m = pd.read_feather(f"{bd}/banc_888_meta.feather").dropna(subset=["banc_888_id"])
    m["id"] = m.banc_888_id.astype("int64")
    sc = m.super_class.astype(str)
    reg = m.region.astype(str)
    role = pd.Series("other", index=m.index)
    role[(sc == "motor")] = "MN"
    role[sc.isin(["descending", "sensory_descending"])] = "DN"
    role[sc.isin(["ascending", "sensory_ascending", "ascending_visceral_circulatory"])] = "AN"
    role[(sc == "sensory") & (reg == "ventral_nerve_cord")] = "VNC_sensory"
    role[(sc == "ventral_nerve_cord_intrinsic")] = "VNC_intrinsic"
    role[(sc == "central_brain_intrinsic")] = "brain_intrinsic"
    cls = m.cell_class.astype(str)
    area = pd.Series("other brain", index=m.index)
    area[cls.isin(["kenyon_cell", "mushroom_body_dopaminergic_neuron", "mushroom_body_output_neuron"])] = "mushroom body"
    area[cls.str.startswith("central_complex")] = "central complex"
    area[cls.str.startswith("lateral_horn")] = "lateral horn"
    area[role == "DN"] = "descending"
    area[reg == "ventral_nerve_cord"] = "nerve cord"
    area[reg == "optic_lobe"] = "optic lobe"
    R = dict(zip(m.id, role)); A = dict(zip(m.id, area))

    e = pd.read_parquet(f"{bd}/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet",
                        columns=["pre_root_id", "post_root_id", "num_synapses"])
    e.columns = ["pre", "post", "n"]
    e["rpre"] = e.pre.map(R).fillna("unannotated"); e["rpost"] = e.post.map(R).fillna("unannotated")
    return e, role, R, A


def load_malecns():
    m = pd.read_feather(f"{bd}/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    sc = m.superclass.astype(str); cls = m["class"].astype(str)
    role = pd.Series("other", index=m.index)
    role[sc == "vnc_motor"] = "MN"
    role[sc.isin(["descending_neuron", "sensory_descending", "efferent_descending"])] = "DN"
    role[sc.isin(["ascending_neuron", "sensory_ascending", "efferent_ascending"])] = "AN"
    role[sc == "vnc_sensory"] = "VNC_sensory"
    role[sc == "vnc_intrinsic"] = "VNC_intrinsic"
    role[sc == "cb_intrinsic"] = "brain_intrinsic"
    area = pd.Series("other brain", index=m.index)
    area[cls.isin(["Kenyon_Cell", "DAN", "MBON"])] = "mushroom body"
    area[cls == "CX"] = "central complex"
    area[role == "DN"] = "descending"
    area[sc.str.startswith("vnc_")] = "nerve cord"
    area[sc.str.startswith("ol_")] = "optic lobe"
    R = dict(zip(m.bodyId, role)); A = dict(zip(m.bodyId, area))
    e = pd.read_feather(f"{bd}/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                        columns=["body_pre", "body_post", "weight"])
    e.columns = ["pre", "post", "n"]
    e = e[e.n >= 3].copy()
    e["rpre"] = e.pre.map(R).fillna("unannotated"); e["rpost"] = e.post.map(R).fillna("unannotated")
    return e, role, R, A


if animal == "banc":
    e, role, R, A = load_banc()
else:
    e, role, R, A = load_malecns()
res = {"counts": {k: int(v) for k, v in role.value_counts().items()}}

# 1. motor neuron input
mi = e[e.rpost == "MN"]; tot = mi.n.sum()
res["mn_input_share"] = {k: float(v / tot) for k, v in mi.groupby("rpre").n.sum().sort_values(ascending=False).items()}
vs = set(e.loc[e.rpre == "VNC_sensory", "pre"])
s2i = e[(e.rpre == "VNC_sensory") & (e.rpost == "VNC_intrinsic")]
fed = set(s2i.post)  # VNC intrinsic neurons that receive VNC sensory input
mn_from_fed = mi[mi.pre.isin(fed)].n.sum()
res["reflex"] = dict(mn_cells=int(mi.post.nunique()),
                     mn_direct_from_vnc_sensory=float(mi[mi.rpre == "VNC_sensory"].n.sum() / tot),
                     mn_from_intrinsic_with_sensory_input=float(mn_from_fed / tot),
                     mn_reached_in_2_steps_without_brain=int(mi[mi.pre.isin(fed | vs)].post.nunique()))

# 2. descending bottleneck
do = e[e.rpre == "DN"]
res["descending"] = dict(dn_cells=int(res["counts"].get("DN", 0)),
                         brain_intrinsic_cells=int(res["counts"].get("brain_intrinsic", 0)),
                         dn_output_into_vnc=float(do[do.post.map(A) == "nerve cord"].n.sum() / do.n.sum()),
                         mn_input_direct_from_dn=float(mi[mi.rpre == "DN"].n.sum() / tot))
d1 = do[["pre", "post"]].drop_duplicates()
mnset = set(mi.post)
direct = d1[d1.post.isin(mnset)].groupby("pre").post.nunique()
via = d1.merge(mi[["pre", "post"]].drop_duplicates().rename(columns={"pre": "post", "post": "mn"}), on="post")
reach = pd.concat([d1[d1.post.isin(mnset)].rename(columns={"post": "mn"})[["pre", "mn"]], via[["pre", "mn"]]])\
          .drop_duplicates().groupby("pre").mn.nunique()
dn_ids = [i for i, r in R.items() if r == "DN"]
reach = reach.reindex(dn_ids).fillna(0)
res["descending"].update(mn_reached_le2_median=float(reach.median()),
                         mn_reached_le2_q90=float(reach.quantile(0.9)),
                         dn_reaching_any_mn_le2=float((reach > 0).mean()))

# 3. ascending report
ao = e[e.rpre == "AN"]
dest = ao.groupby(ao.post.map(A).fillna("unannotated")).n.sum()
res["ascending"] = dict(an_cells=int(res["counts"].get("AN", 0)),
                        vnc_sensory_cells=int(res["counts"].get("VNC_sensory", 0)),
                        output_share_by_area={k: float(v / dest.sum()) for k, v in dest.sort_values(ascending=False).items()})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
