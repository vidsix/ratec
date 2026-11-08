#!/usr/bin/env python3
"""Re-evaluate the twin study checkpoints with the corrected byte weighting.

The weights (bits) never changed — only the byte normalizer was wrong twice:
  v1: indexed lens by predicted token id   (garbage weights)
  v2: indexed per-domain lens by flat position (empty slices past domain 0)
  v3 (this): per-domain lens at domain-local target positions.

Training is deterministic (seed 42, eval consumes no RNG), so the saved
state_dicts are exactly the weights a fresh run would produce. This script
reloads each checkpoint, recomputes full-coverage BPB, and rewrites the
twin_{kind}.json results in place (curve/wall/params preserved).
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_study import (RUNS, DOMAINS, TwinLM, load_streams, load_bytes_streams,
                        eval_full)

KINDS = ["bpe", "bytes", "ratec_direct", "ratec"]


def main():
    for kind in KINDS:
        if not os.path.exists(f"{RUNS}/twin_{kind}.pt"):
            print(f"[{kind}] no checkpoint — rerun twin_study.py --kind {kind}")
            continue
        res = json.load(open(f"{RUNS}/twin_{kind}.json"))
        if kind == "bytes":
            _, _, va_ids, va_lens, bounds = load_bytes_streams()
        else:
            prefix = "bpe_mix" if kind == "bpe" else "ratec_main"
            _, _, va_ids, va_lens, bounds = load_streams(prefix)
        model_kind = "bpe" if kind == "ratec_direct" else kind
        model = TwinLM(model_kind)
        model.load_state_dict(torch.load(f"{RUNS}/twin_{kind}.pt",
                                         weights_only=True))
        va_flat = np.concatenate(va_ids)
        bpb, per_dom, covered = eval_full(model, va_flat, va_lens, bounds, 256)
        res["final_bpb_overall"] = bpb
        res["final_bpb_per_domain"] = per_dom
        res["val_bytes_covered"] = covered
        res["val_coverage"] = covered / res["val_bytes"]
        json.dump(res, open(f"{RUNS}/twin_{kind}.json", "w"), indent=2)
        print(f"[{kind}] corrected BPB {bpb:.4f}  " +
              " ".join(f"{k}={v:.3f}" for k, v in per_dom.items()) +
              f"  coverage {res['val_coverage']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
