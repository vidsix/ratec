#!/usr/bin/env python3
"""Gigatoken bench: sustained GB-scale throughput through the engine.

Streams ~1 GB (the 4 fixture domains tiled) through the two production
paths — spans with book (the ids-prep path) and judge-only (surprise) —
as MANY sequential chunked calls (the real ingestion pattern: engines
stay warm, maps reach steady state), fresh protocol: one engine per
arm, throughput measured over the WHOLE gigabyte, engine reset between
reps only. Records sustained MB/s + Mtok/s + patches, with the record
envelope. Output: bench/bench_gigatoken.json (raw) — the caller wraps
the envelope via record_util.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ratec

FX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "tests/data/fixtures")
DOMAINS = ("english", "code", "chinese", "base64")
TARGET_BYTES = 1 << 30          # 1 GiB per arm
CHUNK = 4 << 20                 # 4 MiB ingestion chunks (real pattern)

def main():
    docs = [open(os.path.join(FX, f"{d}.span.bin"), "rb").read() for d in DOMAINS]
    unit = b"".join(docs)
    tile = TARGET_BYTES // len(unit)
    stream = unit * tile
    n = len(stream)
    print(f"gigatoken bench: {n/1e9:.2f} GB ({n} bytes), chunk {CHUNK>>20} MiB")

    res = {"bytes_total": n, "chunk": CHUNK, "c": 20}
    for arm in ("spans_book", "judge_only"):
        e = ratec.RateC(c=20, store_book=(arm == "spans_book"))
        t0 = time.perf_counter()
        ntok = 0
        for off in range(0, n, CHUNK):
            piece = stream[off:off + CHUNK]
            if arm == "spans_book":
                sp = e.encode_to_spans(piece)
                ntok += len(sp)
            else:
                e.surprise(piece)
        dt = time.perf_counter() - t0
        res[arm] = {
            "sec": round(dt, 3),
            "MB_per_s": round(n / dt / 1e6, 2),
            "tokens": ntok if arm == "spans_book" else None,
            "Mtok_per_s": round(ntok / dt / 1e6, 3) if arm == "spans_book" else None,
            "GB_processed": round(n / 1e9, 3),
        }
        print(f"  {arm}: {res[arm]['MB_per_s']} MB/s sustained"
              + (f", {res[arm]['Mtok_per_s']} Mtok/s" if ntok else ""))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bench", "bench_gigatoken.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()
