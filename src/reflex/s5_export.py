#!/usr/bin/env python3
"""Export the FlyWire PN->KC wiring, the glomerulus input map and the S2/S3 configuration for s5_long.py.
Usage: python s5_export.py OUT_NPZ"""
import json, os, sys
os.environ["S1_MODE"] = "multi"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_spectra as S  # noqa: E402
import s2_s3 as H  # noqa: E402
pn, Min, G = S.glom_map(np.random.default_rng(5)); W = S.load_w(pn); r, c = np.nonzero(W)
np.savez_compressed(sys.argv[1], shape=np.array(W.shape), r=r, c=c, w=W[r, c].astype(np.float32), Min=Min.astype(np.float32),
                    full_cfg=json.dumps(H.default_full_cfg()))
