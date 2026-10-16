#!/usr/bin/env python3
"""Export the FlyWire PN->KC wiring and KC->MBON mask as one small file for the edge timing test.
Usage: python t5_edge_export.py OUT_NPZ"""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import common as K
W = K.wirings(np.random.default_rng(0))["fly"]; r, c = np.nonzero(W)
np.savez_compressed(sys.argv[1], r=r.astype(np.int32), c=c.astype(np.int32), w=W[r, c].astype(np.float32),
                    shape=np.array(W.shape), mask=K.mbon_mask().astype(np.int8))
