#!/usr/bin/env python3
"""Gate: the ratec library engine must be bit-exact with the validated
reference pipeline.

Two modes:

  * shipped fixtures (default) — runs anywhere. Uses the scored spans and
    reference arrays in tests/data/fixtures/ (captured from the validated
    reference pipeline on the bake-off corpora):
      {dom}.span.bin   scored span (post-warm bytes)
      {dom}.sur.npy    reference float32 per-byte surprisal
      {dom}.starts.npy reference patch starts at C=20 (uint32)

  * private bake-off (full corpora, all domains) — set
      RATEC_BAKEOFF_DATA=/path/to/bakeoff_data
      RATEC_BAKEOFF_CACHE=/path/to/bakeoff_cache
    (cache needs {dom}.sur.npy, {dom}.r20starts.npy, meta.json).

Checks per domain (C=20):
  1. surprise(span)   == reference surprisal      (float32, bitwise)
  2. encode spans     == reference patch starts
  3. patch_from_sur(sur, 20) == reference starts
  4. engine spans are contiguous and tile the input exactly
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ratec

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "data", "fixtures")
PRIV_DATA = os.environ.get("RATEC_BAKEOFF_DATA")
PRIV_CACHE = os.environ.get("RATEC_BAKEOFF_CACHE")
DOMAINS = ["english", "code", "chinese", "base64"]


def load_domain(dom):
    """-> (span bytes, ref sur f32, ref starts int64)."""
    if PRIV_DATA and PRIV_CACHE:
        meta = json.load(open(os.path.join(PRIV_CACHE, "meta.json")))[dom]
        raw = open(os.path.join(PRIV_DATA, f"{dom}.txt"), "rb").read()
        span = raw[meta["warm_bytes"]: meta["warm_bytes"] + meta["scored_bytes"]]
        sur = np.load(os.path.join(PRIV_CACHE, f"{dom}.sur.npy"))
        starts = np.load(os.path.join(PRIV_CACHE, f"{dom}.r20starts.npy"))
        return span, sur, starts.astype(np.int64)
    span = open(os.path.join(FIXTURES, f"{dom}.span.bin"), "rb").read()
    gm = json.load(open(os.path.join(FIXTURES, "meta.json")))[dom]
    assert hashlib.sha256(span).hexdigest() == gm["span_sha256"], \
        f"{dom}.span.bin corrupt (sha mismatch)"
    sur = np.load(os.path.join(FIXTURES, f"{dom}.sur.npy"))
    starts = np.load(os.path.join(FIXTURES, f"{dom}.starts.npy"))
    return span, sur, starts.astype(np.int64)


mode = ("private bake-off (%s)" % PRIV_CACHE) if (PRIV_DATA and PRIV_CACHE) \
    else "shipped fixtures (%s)" % os.path.relpath(FIXTURES)
print(f"gate mode: {mode}")

ok = True
for dom in DOMAINS:
    span, sur_ref, starts_ref = load_domain(dom)

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
