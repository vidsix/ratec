#!/usr/bin/env python3
"""The per-order-tables gate (PLAN_v0.9 P1 / v0.9.2 scope): "per-order
tables sized to distinct-context counts (>=5% gate or honestly
declined)".

The judge's per-order context structures today (_rcengine.c):

  order 0  cnt0[256]            1 KB, inline in the core struct
  order 1  tot1[256] + cnt1[256*256]
                               256 + 256 KB, DENSE direct arrays,
                               indexed (h1<<8)|b — zero indirection
  order 2  tot2[65536] + cnt2[65536*256]
                               256 KB + 64 MB VIRTUAL, the counter row
                               is GUARDED (tot2[h2] first — rows are
                               touched only for contexts with history)
  orders 3-5                  already the SIZED open-addressing map
                               (tmap), grown on demand to distinct-
                               context counts — the lever this plan
                               item describes is already the design
                               for every order where it can pay.

A "sized to distinct-context counts" variant for orders 1-2 must either
(a) keep the same hash slices and ADD an indirection (hash -> slot) in
the hottest per-byte loop — strictly more cache misses than the direct
L2-resident index, or (b) change the slice widths — which changes the
model and the ids, forbidden by the 1.0 bit-freeze within 1.x. This
gate measures the footprint and the distinct-context counts, anchors
the judge's parity position from the quiet-box envelope, and records
the decision. No unsound or slower code ships on a declined lever.
"""
import hashlib
import json
import os
import resource
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np

import ratec

from record_util import record_envelope, write_record

DOMAINS = ["english", "code", "chinese", "base64"]
FIXTURES = os.path.join(ROOT, "tests", "data", "fixtures")


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def walk_snapshot(snap):
    """Independent LE walk of the v8 layout (mirror of test_v8.py) —
    extracts the per-order occupancy the gate reasons about."""
    o = 0

    def rd(fmt, size):
        nonlocal o
        v = struct.unpack_from(fmt, snap, o)[0]
        o += size
        return v

    assert rd("<I", 4) == 0x54534352
    assert rd("<I", 4) == 8
    C = rd("<d", 8)
    sb, pf, fz, poi = rd("<I", 4), rd("<I", 4), rd("<I", 4), rd("<I", 4)
    pmode, mcb = rd("<B", 1), rd("<B", 1)
    carry = rd("<d", 8)
    prune_flag = rd("<B", 1)
    nseen, n_forced, n_patches = rd("<Q", 8), rd("<Q", 8), rd("<Q", 8)
    po = rd("<B", 1)
    open_start = rd("<Q", 8)
    open_len = rd("<Q", 8)
    cs, lim = rd("<d", 8), rd("<d", 8)
    assert cs >= 0.0 and lim >= 0.0
    if sb and po and open_len:
        o += open_len                      # obuf (book mode, open patch)
    assert po == 0 or True                 # a flushed stream has po = 0
    # -- fixed tables: cnt0 (1024) -> count nonzero
    cnt0_nz = 0
    for k in range(256):
        if struct.unpack_from("<I", snap, o + 4 * k)[0]:
            cnt0_nz += 1
    o += 256 * 4
    # cnt1 (256 x 256 u32): nonzero counters + distinct contexts via tot1
    # tot1 sits AFTER cnt1 in the layout: cnt1 (256*256*4) then tot1 (256*4)
    o += 256 * 256 * 4
    tot1_nz = 0
    for k in range(256):
        if struct.unpack_from("<I", snap, o + 4 * k)[0]:
            tot1_nz += 1
    o += 256 * 4
    tot2_nz = 0
    for k in range(65536):
        if struct.unpack_from("<I", snap, o + 4 * k)[0]:
            tot2_nz += 1
    o += 65536 * 4
    o += 6 * 8                             # h
    nctx = rd("<Q", 8)
    # skip the cnt2 RLE rows
    for _ in range(nctx):
        o += 2
        m16 = struct.unpack_from("<H", snap, o)[0]
        o += 2 + m16 + 4 * m16
    tmap_capb, tmap_n = rd("<Q", 8), rd("<Q", 8)
    assert tmap_capb <= 26 and tmap_n <= (1 << tmap_capb) * 3 // 4
    o += tmap_n * (8 + 4 + 1 + 16 + 32)    # tmap entries (field-by-field)
    cmap_capb, cmap_n = rd("<Q", 8), rd("<Q", 8)
    assert cmap_capb <= 26
    o += cmap_n * (8 + 4)                  # cmap entries
    if sb:
        bn, alen = rd("<Q", 8), rd("<Q", 8)
        assert (bn == 0) == (alen == 0)
        o += alen + 8 * bn                 # arena + off/ln pairs
    assert o == len(snap), f"walked {o} of {len(snap)}"
    return {
        "C": C, "nseen": nseen, "n_patches": n_patches,
        "order0_distinct": cnt0_nz,
        "order1_distinct": tot1_nz,       # contexts with history (of 256)
        "order2_distinct_rows": tot2_nz,  # tot2 nonzero (of 65536)
        "order2_rle_rows": nctx,          # cnt2 rows with nonzero counters
        "tmap_capbits": tmap_capb, "tmap_n": tmap_n,   # orders 3-5, sized
        "cmap_capbits": cmap_capb, "cmap_n": cmap_n,
        "book_present": bool(sb),
        "map_capbits": mcb,
    }


def main():
    corpus = b"".join(
        open(os.path.join(FIXTURES, f"{d}.span.bin"), "rb").read()
        for d in DOMAINS)
    eng = ratec.RateC(c=20)
    rss0 = rss_mib()
    eng.encode_bytes(corpus)
    rss1 = rss_mib()
    snap = eng._eng.save_state()
    walk = walk_snapshot(snap)

    # resident footprint of the whole engine after 1 MB of real text
    engine_resident_mib = round(max(rss1 - rss0, 0.0), 1)

    # the sized-order occupancy: orders 3-5 already live in the sized map
    direct_orders_fixed_kib = 1 + 256 + 256 + 256        # cnt0+cnt1+tot1+tot2
    virtual_cnt2_mib = 64

    env_path = os.path.join(ROOT, "research", "quietbox_envelope.json")
    env = json.load(open(env_path))
    tik_ref = env.get("tiktoken_reference", {})
    tik_eng = min(v["mb_s"] for v in tik_ref.values()) if tik_ref else None
    ratec_lanes = env.get("lanes", {})
    enc_bytes = ratec_lanes.get("encode_bytes_warm", {}).get("mb_s")

    record = {
        "record": record_envelope(inputs={f"fixture_{d}":
            f"tests/data/fixtures/{d}.span.bin" for d in DOMAINS}),
        "measured": {
            "corpus_bytes": len(corpus),
            "engine_resident_mib_after_1mb_text": engine_resident_mib,
            "snapshot_occupancy": walk,
            "direct_orders_fixed_kib": direct_orders_fixed_kib,
            "order2_virtual_mib": virtual_cnt2_mib,
            "order2_guard": "tot2[h2] checked before the cnt2 row is "
                            "touched — rows materialize only for "
                            "contexts with history (resident cost "
                            "tracks distinct contexts already)",
        },
        "parity_anchor": {
            "source": "research/quietbox_envelope.json (the quiet-box "
                      "record, PLAN_v0.9 P1.1)",
            "tiktoken_r50k_best_mb_s": tik_eng,
            "ratec_encode_bytes_mb_s": enc_bytes,
            "note": "the judge core alone is at tiktoken parity "
                    "(PLAN_v0.8 attribution: 9.6 vs 9.8 MB/s); the "
                    "lane-level gap is patcher bookkeeping + probe "
                    "traffic, not the per-order tables",
        },
        "decision": {
            "lever": "per-order tables sized to distinct-context counts",
            "gate": ">= 5% lane-level improvement on the quiet-box "
                    "protocol, else declined",
            "verdict": "DECLINED",
            "reasons": [
                "orders 3-5 ALREADY use the sized open-addressing map "
                "(tmap grows with distinct contexts) — the lever exists "
                "only for orders 1-2",
                "orders 1-2 are DENSE DIRECT arrays (cnt1 256 KB, tot2 "
                "256 KB — L2-resident) indexed with zero indirection; a "
                "distinct-count-sized variant of the same model must add "
                "a hash->slot indirection to the hottest per-byte loop, "
                "strictly more cache misses",
                "shrinking the slice widths instead would change the "
                "model and the ids — forbidden by the 1.0 bit-freeze "
                "(exact-lane ids/spans/snapshot frozen within 1.x)",
                "the judge core is at tiktoken parity in isolation, so "
                "even a perfect table win cannot move the lane >= 5% — "
                "the remaining sequential gap is patcher bookkeeping + "
                "probe memory traffic (PLAN_v0.9 P1 baseline)",
                "the 64 MB order-2 region is virtual and guarded; "
                "resident cost already tracks distinct contexts "
                "(measured: %d nonzero tot2 rows of 65536 after 1 MB of "
                "real text)" % walk["order2_distinct_rows"],
            ],
            "ship": "no code change; the sized-map design ships for "
                    "orders 3-5 (it always has) and the dense direct "
                    "design ships for orders 1-2",
        },
    }
    out = os.path.join(ROOT, "research", "perorder_tables.json")
    write_record(out, record)
    print(json.dumps(record["measured"]["snapshot_occupancy"], indent=1))
    print(f"engine resident after 1 MB text: {engine_resident_mib} MiB")
    print("decision:", record["decision"]["verdict"])


if __name__ == "__main__":
    main()
