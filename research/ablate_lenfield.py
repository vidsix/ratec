#!/usr/bin/env python3
"""PLAN_v0.8 P2.3: is the factorized arm's length field contributing yet?

Eval-only ablation on the ratec@2400 checkpoint (no training, seed-
deterministic control): the model maps packed ids -> (content slot,
length field) and concatenates [ec(c), el(l)] through in_proj. Zeroing
el.weight makes the length input carry nothing (in_proj's bias remains),
so the BPB delta isolates the length field's current contribution. A
~0 delta at probe scale means the factorized arm's small-scale numbers
carry a trainable-in-longer component the tied arms don't have — an
honest caveat for the ladder's interpretation, pre-registered in
docs/PLAN_v0.8.md P2.3.

Record: research/ablations.json (envelope + per-domain deltas).
"""
import json
import os
import sys

import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_study import TwinLM, DOMAINS, C, RUNS, load_streams, eval_full  # noqa: E402
from record_util import record_envelope  # noqa: E402

CKPT = os.path.join(RUNS, "twin_ratec_l2400.ckpt")
STEPS = 2400


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ctx = 256
    _, _, va_ids, va_lens, bounds = load_streams("ratec_main")
    va_flat = np.concatenate(va_ids)

    # control: weights as trained (must reproduce the record's 3.7774 —
    # the eval path is the same code the ladder reported with)
    m1 = TwinLM("ratec")
    m1.load_state_dict(ck["model"])
    b0, per0, _ = eval_full(m1, va_flat, va_lens, bounds, ctx)

    # ablation: length field zeroed (el.weight = 0 -> el(l) == 0)
    m2 = TwinLM("ratec")
    m2.load_state_dict(ck["model"])
    with torch.no_grad():
        m2.el.weight.zero_()
    b1, per1, _ = eval_full(m2, va_flat, va_lens, bounds, ctx)

    print(f"ratec@{STEPS} control BPB        : {b0:.4f}")
    print(f"ratec@{STEPS} length-zeroed BPB : {b1:.4f}")
    print(f"delta (zeroed - control)        : {b1 - b0:+.4f} BPB overall")
    for d in DOMAINS:
        print(f"  {d:>5}: control {per0[d]:.4f}  zeroed {per1[d]:.4f}  "
              f"delta {per1[d] - per0[d]:+.4f}")

    rec = {
        "protocol": {
            "what": f"length-field contribution ablation, ratec@{STEPS} "
                    "(eval-only; el.weight zeroed at eval)",
            "checkpoint": os.path.relpath(CKPT, os.path.dirname(RUNS)),
            "interpretation": "delta ~ 0 => the factorized arm's length "
                              "field is not yet contributing at probe "
                              "scale; its small-scale BPB carries a "
                              "trainable-in-longer component",
        },
        "control_bpb": b0, "zeroed_bpb": b1,
        "delta_bpb": b1 - b0,
        "per_domain": {d: {"control": per0[d], "zeroed": per1[d]}
                       for d in DOMAINS},
        "steps": STEPS,
    }
    rec.update(record_envelope(
        inputs={os.path.relpath(CKPT, os.path.dirname(RUNS)):
                str(os.path.getsize(CKPT))},
        deps=("torch",)))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ablations.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
