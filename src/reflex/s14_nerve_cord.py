#!/usr/bin/env python3
"""S14b: what the ventral nerve cord (VNC) exchanges with the mushroom body in the two whole-CNS connectomes.

BANC v888 (female brain + nerve cord) and MaleCNS v1.0 (male brain + nerve cord), files as in s13_prep.py,
pairs with >= 3 synapses in both. For Kenyon cells (KC), dopaminergic neurons (DAN) and output neurons (MBON):
share of input synapses from olfactory projection neurons (ALPN) and from neurons whose cell body is in the
nerve cord and project up into the brain (ascending, incl. sensory ascending). For MBONs and DANs: share of
output synapses onto descending neurons (brain to nerve cord). Prints a markdown table and writes JSON.
Usage: python s14_nerve_cord.py BANC_DIR MALECNS_DIR OUT_JSON
"""
import json, sys
import pandas as pd

THR = 3


def banc(d):
    m = pd.read_feather(f"{d}/banc_888_meta.feather").dropna(subset=["banc_888_id"])
    m["id"] = m.banc_888_id.astype("int64")
    cls = {"kenyon_cell": "KC", "mushroom_body_dopaminergic_neuron": "DAN", "mushroom_body_output_neuron": "MBON",
           "antennal_lobe_projection_neuron": "ALPN"}
    g = m.cell_class.map(cls)
    sc = pd.Series([str(s) for s in m.super_class], index=m.index)
    g = g.where(g.notna(), sc.map(lambda s: "ASC" if s.startswith("ascending") or s == "sensory_ascending"
                                   else "DN" if s in ("descending", "sensory_descending") else None))
    lab = dict(zip(m.id, g)); typ = dict(zip(m.id, m.cell_type.astype(str)))
    e = pd.read_parquet(f"{d}/synapses_v3_human_readable_sizethresh3_connectioncounts_countthresh3.parquet",
                        columns=["pre_root_id", "post_root_id", "num_synapses"])
    e.columns = ["pre", "post", "n"]
    return e, lab, typ


def malecns(d):
    a = pd.read_feather(f"{d}/body-annotations-male-cns-v1.0-minconf-0.5.feather")
    cls = {"Kenyon_Cell": "KC", "DAN": "DAN", "MBON": "MBON", "ALPN": "ALPN"}
    g = a["class"].map(cls)
    sc = pd.Series([str(s) for s in a.superclass], index=a.index)
    g = g.where(g.notna(), sc.map(lambda s: "ASC" if "ascending" in s else "DN" if "descending" in s else None))
    lab = dict(zip(a.bodyId, g)); typ = dict(zip(a.bodyId, a.type.astype(str)))
    keep = set(k for k, v in lab.items() if v is not None)
    e = pd.read_feather(f"{d}/connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
                        columns=["body_pre", "body_post", "weight"])
    e.columns = ["pre", "post", "n"]
    e = e[(e.n >= THR) & (e.pre.isin(keep) | e.post.isin(keep))]
    return e, lab, typ


def summary(e, lab, typ):
    e = e[e.n >= THR].copy()
    e["gpre"] = e.pre.map(lab); e["gpost"] = e.post.map(lab)
    out = {}
    for t in ("KC", "DAN", "MBON"):
        inn = e[e.gpost == t]; tot = inn.n.sum()
        asc = inn[inn.gpre == "ASC"]
        o = e[e.gpre == t]
        out[t] = dict(cells=int(sum(1 for v in lab.values() if v == t)), input_synapses=int(tot),
                      from_alpn=float(inn[inn.gpre == "ALPN"].n.sum() / tot), from_ascending=float(asc.n.sum() / tot),
                      ascending_cells_in=int(asc.pre.nunique()),
                      cells_with_ascending_input=int(asc.post.nunique()),
                      top_ascending_types=pd.Series(asc.n.values, index=asc.pre.map(typ)).groupby(level=0).sum()
                      .sort_values(ascending=False).head(5).astype(int).to_dict(),
                      output_synapses=int(o.n.sum()),
                      to_descending=float(o[o.gpost == "DN"].n.sum() / max(o.n.sum(), 1)),
                      cells_with_dn_output=int(o[o.gpost == "DN"].pre.nunique()))
    return out


if __name__ == "__main__":
    bd, md, js = sys.argv[1:4]
    res = {"banc": summary(*banc(bd)), "malecns": summary(*malecns(md))}
    json.dump(res, open(js, "w"), indent=1)
    print("| Dataset | Target | cells | input from olfactory PNs | input from ascending (nerve cord) | ascending cells in | "
          "target cells reached | output onto descending | cells with DN output |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a, r in res.items():
        for t, s in r.items():
            print(f"| {a} | {t} | {s['cells']} | {100*s['from_alpn']:.1f} % | {100*s['from_ascending']:.2f} % | "
                  f"{s['ascending_cells_in']} | {s['cells_with_ascending_input']} | {100*s['to_descending']:.2f} % | "
                  f"{s['cells_with_dn_output']} |")
    for a, r in res.items():
        for t, s in r.items():
            print(a, t, "top ascending types:", s["top_ascending_types"])
