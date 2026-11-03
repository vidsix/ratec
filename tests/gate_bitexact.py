#!/usr/bin/env python3
"""Gate: the ratec library engine must be bit-exact with the validated
reference pipeline (ratec_judge.c / cached bake-off arrays).

Checks per domain (C=20):
  1. surprise(span)   == {dom}.sur.npy            (float32, bitwise)
  2. encode spans     == {dom}.r20starts.npy      (patch starts)
  3. patch_from_sur(sur, 20) == {dom}.r20starts.npy
  4. engine spans are contiguous and tile the input exactly
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/home/z/my-project/ratec")
import ratec

CACHE = "/home/z/my-project/bakeoff_cache"
DATA = "/home/z/my-project/bakeoff_data"
DOMAINS = ["english", "code", "chinese", "base64"]

meta = json.load(open(f"{CACHE}/meta.json"))
ok = True
for dom in DOMAINS:
    raw = open(f"{DATA}/{dom}.txt", "rb").read()
    m = meta[dom]
    span = raw[m["warm_bytes"]: m["warm_bytes"] + m["scored_bytes"]]
    sur_ref = np.load(f"{CACHE}/{dom}.sur.npy")
    starts_ref = np.load(f"{CACHE}/{dom}.r20starts.npy").astype(np.int64)

    tok = ratec.RateC(c=20, store_book=False)
    sur_new = tok.surprise(span)
    e1 = bool(np.array_equal(sur_new, sur_ref))

    tok2 = ratec.RateC(c=20, store_book=False)
    spans = tok2.encode_to_spans(span)
    starts_new = np.array([s for s, _, _ in spans], dtype=np.int64)
    e2 = bool(np.array_equal(starts_new, starts_ref))

    # contiguity: spans must tile [0, n) with no gaps/overlaps
    tile = (spans[0][0] == 0 and
            all(spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1)) and
            spans[-1][1] == len(span))

    sur3 = ratec.RateC(c=20, store_book=False).surprise(span)
    starts_p = ratec.patch_from_sur(sur3, 20).astype(np.int64)
    e3 = bool(np.array_equal(starts_p, starts_ref))

    print(f"{dom:8s} sur_bitexact={e1}  starts_bitexact={e2}  "
          f"tile={tile}  patch_from_sur={e3}  "
          f"({len(span)} B -> {len(spans)} patches)")
    ok &= e1 and e2 and e3 and tile

print("\nGATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
