#!/usr/bin/env python3
"""Compare the PBE (Quantum ESPRESSO) hop barriers with the MACE-MP-0 values.
Reads results/janus/dft_neb/<sys>.json (copied from the CHPC run directory) and inputs/manifest.json,
writes results/janus/dft_neb/summary.md."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[2] / "results" / "janus" / "dft_neb"
man = json.loads((HERE / "inputs" / "manifest.json").read_text())
rows = ["| System | Hop layer | MACE-MP-0 (eV) | PBE (eV) | PBE - MACE (eV) | NEB converged | NEB wall time (h) |",
        "|---|---|---|---|---|---|---|"]
for name, m in man.items():
    f = RES / f"{name}.json"
    r = json.loads(f.read_text()) if f.exists() else {}
    e = r.get("E_hop_eV")
    rows.append(f"| {name} | {m['layer']} | {m['Ea_mace_eV']:.2f} | {e if e is None else f'{e:.2f}'} | "
                f"{'' if e is None else f'{e - m['Ea_mace_eV']:+.2f}'} | {r.get('neb_converged')} | "
                f"{'' if r.get('neb_wall_s') is None else f'{r['neb_wall_s'] / 3600:.1f}'} |")
RES.mkdir(parents=True, exist_ok=True)
(RES / "summary.md").write_text("# DFT check of the MACE-MP-0 hop barriers\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
