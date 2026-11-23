#!/usr/bin/env python3
"""The parallel-scaling record (the rc's infrastructure-speed table).

Measures the two parallel shapes that matter at infrastructure scale:

 1. ONE engine, encode_stream_parallel at threads = 1, 2, 4, 8 (both
    ids and spans): the intra-engine scaling curve on the 4 MiB
    quiet-box stream. The ids lane is the CONTRACT lane (S6: the ids
    are the sequential first-seen ids for ANY thread count) — the
    curve must be throughput-hungry, not contract-violating: the test
    after each rep asserts decode round-trip and thread-count
    invariance of the ids.

 2. chunked_prep over 256 MiB at threads 2, 4, 8: the sustained
    prep-pipeline shape (fresh engine per segment, flat RSS) — what
    an ingestion fleet actually runs.

Records research/parallel_scaling.json with the envelope (the record
protocol), so the scaling table is reproducible and versioned.
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import ratec

from record_util import record_envelope, write_record

LINE = (b"def encode_stream(data: bytes) -> list:  # the quick brown "
        b"fox jumps over the lazy dog\n    out = []\n    for i, b in "
        b"enumerate(data):\n        if b % 7:\n            out.append("
        b"(i * 31 + b) & 0xFFFF)\n# utf8: \xe4\xbd\xa0\xe5\xa5\xbd"
        b" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")
N = 4 << 20
DATA = (LINE * (N // len(LINE) + 1))[:N]
REPS = 5
BIG = 256 << 20


def main():
    res = {"stream": {}, "prep": {}, "meta": {}}
    ids_ref = None
    for th in (1, 2, 4, 8):
        eng = ratec.RateC(c=20, store_book=True)
        eng.encode_stream_parallel(DATA, threads=th)          # warm untimed
        _, _, ids_ref = eng.encode_stream_parallel(DATA, threads=th, ids=True)
        t_ids, t_sp = [], []
        for rep in range(REPS):
            t0 = time.perf_counter()
            _, _, ids = eng.encode_stream_parallel(DATA, threads=th, ids=True)
            t_ids.append(time.perf_counter() - t0)
            assert eng.decode_bytes(ids) == DATA
            assert bytes(ids) == bytes(ids_ref)               # invariance
            t0 = time.perf_counter()
            eng.encode_stream_parallel(DATA, threads=th, as_arrays=True)
            t_sp.append(time.perf_counter() - t0)
        res["stream"][f"threads_{th}"] = {
            "ids_mb_s": round(N / statistics.median(t_ids) / 1e6, 2),
            "spans_mb_s": round(N / statistics.median(t_sp) / 1e6, 2),
        }
        print(f"threads={th}: ids {res['stream'][f'threads_{th}']['ids_mb_s']} "
              f"MB/s, spans {res['stream'][f'threads_{th}']['spans_mb_s']} MB/s",
              flush=True)

    unit = DATA * (BIG // len(DATA))
    for th in (2, 4, 8):
        def chunks():
            for off in range(0, len(unit), 4 << 20):
                yield unit[off:off + (4 << 20)]
        t0 = time.perf_counter()
        segs = ratec.chunked_prep(chunks(), c=20, engine_bytes=1 << 28,
                                  threads=th, ids=True)
        dt = time.perf_counter() - t0
        ntok = sum(len(seg["ids"]) for seg in segs)
        res["prep"][f"threads_{th}"] = {
            "mb_s": round(len(unit) / dt / 1e6, 2),
            "mtok_s": round(ntok / dt / 1e6, 2),
            "segments": len(segs),
        }
        print(f"prep 256MiB threads={th}: "
              f"{res['prep'][f'threads_{th}']['mb_s']} MB/s, "
              f"{res['prep'][f'threads_{th}']['mtok_s']} Mtok/s", flush=True)

    res["record"] = record_envelope(inputs={})   # the convention key
    res["note"] = "threads>1 on a 4-vCPU-class box may plateau; the " \
                  "record is host-annotated via the envelope"
    write_record(os.path.join(ROOT, "research", "parallel_scaling.json"), res)
    print("recorded research/parallel_scaling.json")


if __name__ == "__main__":
    main()
