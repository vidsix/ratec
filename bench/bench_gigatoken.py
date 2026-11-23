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

    # the infrastructure arm: the ingestion-fleet pattern — chunked_prep
    # over the whole GiB (fresh engine per 256 MiB segment, threads=2
    # per segment), ids streamed, tokens counted across segments
    # warm pass with a counting sink (flat RSS: nothing retained)
    ntok = [0]; nseg = [0]
    ratec.chunked_prep((stream[off:off + CHUNK]
                        for off in range(0, n, CHUNK)),
                       c=20, engine_bytes=1 << 26, threads=2, ids=True,
                       sink=lambda sg: (ntok.__setitem__(0, ntok[0] + len(sg["ids"])),
                                        nseg.__setitem__(0, nseg[0] + 1)))
    t0 = time.perf_counter()
    ratec.chunked_prep((stream[off:off + CHUNK]
                        for off in range(0, n, CHUNK)),
                       c=20, engine_bytes=1 << 26, threads=2, ids=True,
                       sink=lambda sg: None)
    dt = time.perf_counter() - t0
    res["chunked_prep_fleet"] = {
        "mb_s": round(n / dt / 1e6, 2),
        "mtok_s": round(ntok[0] / dt / 1e6, 2),
        "segments": nseg[0], "tokens": ntok[0],
    }
    print(f"chunked_prep fleet (threads=2): {res['chunked_prep_fleet']['mb_s']} MB/s, "
          f"{res['chunked_prep_fleet']['mtok_s']} Mtok/s, {nseg[0]} segments")

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
