#!/usr/bin/env python3
"""S22c: memory of the foundation models used elsewhere in the project, against data size.

1. MACE-MP-0 medium (the Janus energetics model) on the DGX GPU: peak GPU memory and time of one
   energy + force call on 2H MoS2 monolayer supercells of 48 to 6,912 atoms, float32 and float64.
   Here "data" is the structure the model is asked about: the neighbour graph and the per-edge
   features grow with the number of atoms.
2. Qwen3.5-9B and Qwen3.6-35B-A3B-FP8 (the local language models of the agent tests): weight bytes from the
   checkpoint files, plus the per-sequence memory a language model needs when it keeps instrument
   readings in its context: key/value cache of the full-attention layers (grows with context length)
   and the fixed recurrent state of the linear-attention layers, from each model's config.json.
   One Raman reading written as 56 numbers is counted as TOKENS_PER_READING tokens (measured with the
   model's tokenizer).
Usage: python s22_foundation.py mace OUT_JSON   (venv with mace-torch and CUDA torch)
       python s22_foundation.py llm OUT_JSON    (venv with tokenizers; reads the local Hugging Face cache)
"""
import glob, json, os, sys, time
import numpy as np

HF = os.path.expanduser("~/.cache/huggingface/hub")
LLMS = ["Qwen3.5-9B", "Qwen3.6-35B-A3B-FP8"]


def mace_part():
    import torch
    from ase.build import mx2
    from mace.calculators import mace_mp
    out = {}
    for dt in ("float32", "float64"):
        calc = mace_mp(model="medium", device="cuda", default_dtype=dt)
        npar = int(sum(p.numel() for p in calc.models[0].parameters()))
        rows = []
        for n in (4, 8, 12, 16, 24, 32, 48):
            at = mx2("MoS2", "2H", a=3.18, thickness=3.19, size=(n, n, 1), vacuum=10.0); at.calc = calc
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t = time.perf_counter(); at.get_potential_energy(); at.get_forces(); torch.cuda.synchronize()
            dt_s = time.perf_counter() - t
            at.calc.results = {}; at.positions[0, 0] += 1e-4  # force a second, timed call
            t = time.perf_counter(); at.get_potential_energy(); at.get_forces(); torch.cuda.synchronize()
            rows.append(dict(atoms=len(at), peak_bytes=int(torch.cuda.max_memory_allocated() - base),
                             s_first=dt_s, s_call=time.perf_counter() - t))
            print(dt, rows[-1], flush=True)
        out[dt] = dict(parameters=npar, weight_bytes=npar * (4 if dt == "float32" else 8), runs=rows)
        del calc; torch.cuda.empty_cache()
    return out


def llm_part():
    from tokenizers import Tokenizer
    out = {}
    rng = np.random.default_rng(0)
    txt = " ".join(f"{v:.3g}" for v in rng.lognormal(0, 1, 56))  # one reading written as 56 numbers
    for m in LLMS:
        snap = glob.glob(f"{HF}/models--Qwen--{m}/snapshots/*")[0]
        c = json.load(open(f"{snap}/config.json")); t = c.get("text_config", c)
        wbytes = sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(f"{snap}/*.safetensors"))
        full = sum(1 for x in t["layer_types"] if x == "full_attention"); lin = len(t["layer_types"]) - full
        kv_per_tok = full * 2 * t["num_key_value_heads"] * t["head_dim"] * 2  # bf16 cache
        lin_state = lin * (t["linear_num_value_heads"] * t["linear_key_head_dim"] * t["linear_value_head_dim"] * 4
                           + (2 * t["linear_num_key_heads"] * t["linear_key_head_dim"] + t["linear_num_value_heads"] * t["linear_value_head_dim"])
                           * (t["linear_conv_kernel_dim"] - 1) * 2)
        tok = Tokenizer.from_file(f"{snap}/tokenizer.json"); tpr = len(tok.encode(txt).ids)
        out[m] = dict(weight_bytes=int(wbytes), layers=len(t["layer_types"]), full_attention_layers=full,
                      linear_attention_layers=lin, kv_bytes_per_token=int(kv_per_tok), linear_state_bytes=int(lin_state),
                      max_context=int(t["max_position_embeddings"]), tokens_per_reading=int(tpr),
                      context_bytes={str(n): int(lin_state + kv_per_tok * n) for n in (1024, 8192, 32768, 131072, 262144)})
        print(m, out[m], flush=True)
    return out


if __name__ == "__main__":
    part, path = sys.argv[1], sys.argv[2]
    json.dump(mace_part() if part == "mace" else llm_part(), open(path, "w"), indent=1)
