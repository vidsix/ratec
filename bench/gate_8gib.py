#!/usr/bin/env python3
"""The 8 GiB chunked_prep gate (PLAN_v0.9 P2.1 / the 1.0 scale gate).

An 8 GiB synthetic prep job runs through `ratec.chunked_prep` — the
uint32 span space forces more than one engine, and the helper's
segmentation must reproduce EXACTLY what explicit per-segment engines
produce: per-segment sha256 over (ids, starts, ends) compared between

  run A: chunked_prep over a lazily generated chunk stream (flat RSS;
         segments hashed inside the sink and released), and
  run B: the same segments regenerated block-for-block and encoded on
         standalone RateC engines (the documented per-chunk pattern).

Also recorded: peak RSS, wall time, per-segment token counts, and the
reserve=/rehash instrument at scale (the growth-rehash delta a big job
pays WITHOUT reserve= — data the unit gate cannot produce on 3 GB RAM).

The sandbox is RAM-limited (3 GB), so the gate segments at 256 MiB
(engine_bytes=1<<28): every segment's arrays (~650 MB transient) fit
comfortably, the 8 GiB stream still crosses 32 engines, and the
byte-identity claim is per-segment sha256 — the same claim a 4 GiB
segment makes, 32 times. A >=4 GiB single-engine segment run belongs on
the 1.0 soak host and is recorded as such.

Output: research/gate_8gib.json (committed) + a printed verdict.
"""
import hashlib
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import numpy as np

import ratec

from record_util import record_envelope, write_record

TOTAL = 8 << 30                    # 8 GiB of synthetic stream
ENGINE_BYTES = 1 << 28             # 256 MiB per engine (RAM-bound sandbox)
CHUNK = 4 << 20                    # 4 MiB feed chunks
NSEG = TOTAL // ENGINE_BYTES       # 32 segments
CHUNKS_PER_SEG = ENGINE_BYTES // CHUNK

LINE = (b"def encode_stream(data: bytes) -> list:  # the quick brown "
        b"fox jumps over the lazy dog\n    out = []\n    for i, b in "
        b"enumerate(data):\n        if b % 7:\n            out.append("
        b"(i * 31 + b) & 0xFFFF)\n# utf8: \xe4\xbd\xa0\xe5\xa5\xbd"
        b" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")


def chunk_of(block):
    """The deterministic chunk #block (1-based): a per-block header line
    phases the content so segments differ; identical bytes every run."""
    head = b"# block %d of the 8 GiB prep gate\n" % block
    unit = head + LINE
    reps = (CHUNK // len(unit)) + 1
    return (unit * reps)[:CHUNK]


def gen_chunks():
    """The 8 GiB stream as lazily generated 4 MiB chunks (flat RSS)."""
    for block in range(1, TOTAL // CHUNK + 1):
        yield chunk_of(block)


def segment_bytes(seg_index):
    """The seg_index-th 256 MiB segment, generated directly (block k is
    whole; segments align to chunk boundaries by construction)."""
    first = seg_index * CHUNKS_PER_SEG + 1
    return b"".join(chunk_of(k) for k in range(first,
                                               first + CHUNKS_PER_SEG))


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def seg_hash(res):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(res["ids"], dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(res["starts"], dtype=np.uint32).tobytes())
    h.update(np.ascontiguousarray(res["ends"], dtype=np.uint32).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------- run A
hashes_a = []
counts_a = []
t0 = time.time()
ratec.chunked_prep(gen_chunks(), engine_bytes=ENGINE_BYTES, threads=2,
                   sink=lambda r: (hashes_a.append(seg_hash(r)),
                                   counts_a.append(len(r["ids"]))))
t_a = time.time() - t0
rss_a = rss_mib()
assert len(hashes_a) == NSEG, f"run A: {len(hashes_a)} segments, want {NSEG}"
assert sum(counts_a) > 0
print(f"run A (chunked_prep): {NSEG} segments in {t_a:.1f}s, "
      f"peak RSS {rss_a:.0f} MiB", flush=True)

# ------------------------------------------- run B: per-segment engines
hashes_b = []
counts_b = []
t0 = time.time()
for seg in range(NSEG):
    data = segment_bytes(seg)
    eng = ratec.RateC(c=20)
    starts_a, ends_a, tok = eng.encode_stream_parallel(
        data, threads=2, ids=True)
    res = {"ids": np.asarray(tok, dtype=np.int32),
           "starts": np.asarray(starts_a, dtype=np.uint32),
           "ends": np.asarray(ends_a, dtype=np.uint32)}
    hashes_b.append(seg_hash(res))
    counts_b.append(len(res["ids"]))
    del eng, res, data, tok, starts_a, ends_a
t_b = time.time() - t0
rss_b = rss_mib()
print(f"run B (per-segment engines): {NSEG} segments in {t_b:.1f}s, "
      f"peak RSS {rss_b:.0f} MiB", flush=True)

identical = hashes_a == hashes_b
counts_ok = counts_a == counts_b

# ------------------------------------------- reserve instrument at scale
eng = ratec.RateC(c=20)
sample = LINE * 60000            # ~8 MB probe
eng.encode_bytes(sample)
rehash_no_reserve = eng.rehash_count()
eng2 = ratec.RateC(c=20, reserve=TOTAL)
eng2.encode_bytes(sample)
rehash_reserve = eng2.rehash_count()

env = record_envelope(inputs={})
env["notes"] = ("8 GiB synthetic stream, 32 x 256 MiB engines "
                "(RAM-bound sandbox: segment arrays ~650 MB transient; "
                "a >=4 GiB single-engine segment run belongs on the "
                "1.0 soak host)")
record = {
    "record": env,
    "gate": {
        "total_bytes": TOTAL,
        "engine_bytes": ENGINE_BYTES,
        "segments": NSEG,
        "byte_identical": bool(identical),
        "counts_match": bool(counts_ok),
        "run_a_seconds": round(t_a, 1),
        "run_b_seconds": round(t_b, 1),
        "run_a_peak_rss_mib": round(rss_a, 0),
        "run_b_peak_rss_mib": round(rss_b, 0),
        "tokens_total": int(sum(counts_a)),
        "rehash_no_reserve_8mb_probe": int(rehash_no_reserve),
        "rehash_reserve_8gib_8mb_probe": int(rehash_reserve),
    },
    "segment_hashes_a": hashes_a,
    "segment_counts_a": counts_a,
}
out = os.path.join(ROOT, "research", "gate_8gib.json")
write_record(out, record)

print(f"\nbyte-identical: {identical} | token counts match: {counts_ok}")
print(f"tokens total: {sum(counts_a):,}")
print(f"rehash instrument (8 MB probe): no-reserve {rehash_no_reserve}, "
      f"reserve(8 GiB) {rehash_reserve}")
print("VERDICT:", "PASS" if identical and counts_ok else "FAIL")
sys.exit(0 if identical and counts_ok else 1)
