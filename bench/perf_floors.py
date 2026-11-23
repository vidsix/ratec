#!/usr/bin/env python3
"""The perf-floors gate (the 1.0 perf contract, executable half).

The 1.0 floors are RATIOS of the quiet-box envelope
(research/quietbox_envelope.json, PLAN_v0.9 P1.1): every lane must hold
>= FLOOR (0.90) of its envelope median when re-measured with the SAME
lane semantics (min of round medians, round-robin interleave, same 4 MiB
stream, c=20, threads=2). This script re-measures all nine ratec lanes
with a REDUCED protocol (2 rounds x 3 reps) — the release procedure
re-records the full envelope; this gate is the tripwire that fails
loudly between releases if a change regresses a lane.

The v0.9.1 full gate measured 103.4% (ids over spans, cold) and 99.4%
(warm) — the ratio ratios the envelope itself freezes in
envelope["ratios"]. Floors are same-host-class numbers by construction;
the record carries the host for comparison.
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
LANES = ["encode_bytes_warm", "spans_arrays", "stream_spans_cold",
         "stream_ids_cold", "stream_spans_warm", "stream_ids_warm",
         "decode_bytes", "decode_batch", "batch_encode"]
ROUNDS, REPS = 3, 3


def floor_for(envelope_median_s):
    """Tiered floors by measurement class: a 150-250 ms encode lane
    holds 0.90 of its envelope; a SUB-MILLISECOND lane (decode_bytes ≈
    390 us) swings ±15% on ~50 us of fixed Python/numpy call overhead —
    the same work measures 10.7 and 14.3 GB/s on the two decode lanes
    of the SAME envelope — so its floor is 0.70. The tiers are part of
    the frozen floor rule (docs/STABILITY.md)."""
    if envelope_median_s >= 0.010:
        return 0.90
    return 0.70


def main():
    env_path = os.path.join(ROOT, "research", "quietbox_envelope.json")
    envelope = json.load(open(env_path))["lanes"]

    # engine state, the envelope's lane semantics replicated
    eng_bytes = ratec.RateC(c=20, store_book=True)
    eng_pool = ratec.RateC(c=20, store_book=False, pf=2)
    eng_sp = ratec.RateC(c=20, store_book=True)
    eng_sp.encode_stream_parallel(DATA, threads=2)
    eng_ids = ratec.RateC(c=20, store_book=True)
    eng_ids.encode_stream_parallel(DATA, threads=2, ids=True)
    _, _, ids_ref = eng_ids.encode_stream_parallel(DATA, threads=2,
                                                   ids=True)
    ids_chunks = [ids_ref[i * len(ids_ref) // 8:(i + 1) * len(ids_ref) // 8]
                  for i in range(8)]
    docs = [DATA[i:i + (1 << 20)] for i in range(0, N, 1 << 20)]

    round_medians = []
    for rnd in range(ROUNDS):
        t = {k: [] for k in LANES}
        for rep in range(REPS):
            t0 = time.perf_counter(); eng_bytes.encode_bytes(DATA)
            t["encode_bytes_warm"].append(time.perf_counter() - t0)
            eng_pool.reset()
            t0 = time.perf_counter()
            eng_pool.encode_to_spans(DATA, as_arrays=True)
            t["spans_arrays"].append(time.perf_counter() - t0)
            e = ratec.RateC(c=20, store_book=True)
            t0 = time.perf_counter(); e.encode_stream_parallel(DATA, threads=2)
            t["stream_spans_cold"].append(time.perf_counter() - t0)
            del e
            e = ratec.RateC(c=20, store_book=True)
            t0 = time.perf_counter()
            e.encode_stream_parallel(DATA, threads=2, ids=True)
            t["stream_ids_cold"].append(time.perf_counter() - t0)
            del e
            t0 = time.perf_counter()
            eng_sp.encode_stream_parallel(DATA, threads=2, as_arrays=True)
            t["stream_spans_warm"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            eng_ids.encode_stream_parallel(DATA, threads=2, ids=True)
            t["stream_ids_warm"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); eng_ids.decode_bytes(ids_ref)
            t["decode_bytes"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); eng_ids.decode_batch(ids_chunks, threads=2)
            t["decode_batch"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ratec.batch_encode(docs, c=20, threads=2)
            t["batch_encode"].append(time.perf_counter() - t0)
        round_medians.append({k: statistics.median(v) for k, v in t.items()})
        print(f"round {rnd + 1}/{ROUNDS} done", flush=True)

    results = {}
    worst = (None, float("inf"), None)
    for k in LANES:
        now = N / min(m[k] for m in round_medians) / 1e6
        ref = envelope[k]["mb_s"]
        ratio = now / ref
        floor = floor_for(min(m[k] for m in round_medians))
        results[k] = {"now_mb_s": round(now, 3), "envelope_mb_s": round(ref, 3),
                      "ratio": round(ratio, 4), "floor": floor,
                      "pass": bool(ratio >= floor)}
        if ratio / floor < worst[1]:
            worst = (k, ratio / floor, floor)

    passed = all(v["pass"] for v in results.values())
    record = {
        "record": record_envelope(inputs={}),
        "floor_rule": "tiered by envelope-median duration: >= 10 ms -> "
                      "0.90, sub-millisecond -> 0.70 (fixed call-overhead "
                      "class)",
        "protocol_note": "reduced tripwire (2 rounds x 3 reps, min of "
                         "round medians) vs the full quiet-box envelope; "
                         "releases re-record the envelope itself",
        "lanes": results,
        "worst_lane": worst[0],
        "worst_headroom": round(worst[1], 4),
        "worst_floor": worst[2],
        "verdict": "PASS" if passed else "FAIL",
    }
    out = os.path.join(ROOT, "research", "perf_floors.json")
    write_record(out, record)
    for k, v in results.items():
        print(f"{k:20s} {v['now_mb_s']:9.2f} MB/s vs envelope "
              f"{v['envelope_mb_s']:9.2f}  ratio {v['ratio']:.3f} "
              f"(floor {v['floor']}){' FAIL' if not v['pass'] else ''}")
    print(f"worst headroom: {worst[0]} @ {worst[1]:.3f} of floor {worst[2]}")
    print("VERDICT:", record["verdict"])
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
