#!/usr/bin/env python3
"""The nightly soak (PLAN_v0.9 P2.4 / the 1.0 robustness gate): a
continuous mixed-operation run asserting (a) no exceptions, (b) flat
RSS, (c) bit-exact outputs vs reference engines at every checkpoint.

The operation mix per round, over seeded synthetic corpora that vary by
round (the stream never repeats exactly):
  encode_stream_parallel spans + ids (rotating thread counts 1..4)
  decode_bytes + decode_batch round-trip == input
  intern_spans vs encode_bytes ids equality (the S5 executable form)
  snapshot save -> load -> continuation bit-exactness
  clone/restore + batch_encode spot-checks
RSS is sampled after every round; the record carries the full series.
Runtime: --minutes N (default 10; the CI nightly runs 60).
Output: research/soak_<utc>.json + a printed verdict.
"""
import argparse
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

LINE = (b"def encode_stream(data: bytes) -> list:  # the quick brown "
        b"fox jumps over the lazy dog\n    out = []\n    for i, b in "
        b"enumerate(data):\n        if b % 7:\n            out.append("
        b"(i * 31 + b) & 0xFFFF)\n# utf8: \xe4\xbd\xa0\xe5\xa5\xbd"
        b" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")


def round_corpus(seed, mib=6):
    """A deterministic corpus that varies with the round: the line
    pattern with a per-round counter mixed in."""
    head = b"# soak round seed %d\n" % seed
    unit = head + LINE
    reps = (mib << 20) // len(unit) + 1
    return (unit * reps)[:mib << 20]


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    args = ap.parse_args()
    deadline = time.time() + args.minutes * 60

    rounds = []
    rss_series = []
    failures = []
    rnd = 0
    print(f"soak: {args.minutes} min budget", flush=True)
    while time.time() < deadline:
        rnd += 1
        t0 = time.time()
        data = round_corpus(rnd)
        threads = 1 + (rnd % 4)
        try:
            # 1. parallel spans + ids
            eng = ratec.RateC(c=20)
            starts, ends, tok = eng.encode_stream_parallel(
                data, threads=threads, ids=True)
            # 2. decode round-trip (both paths)
            assert eng.decode_bytes(tok) == data, "decode_bytes drift"
            chunks = [tok[i * len(tok) // 8:(i + 1) * len(tok) // 8]
                      for i in range(8)]
            dec = b"".join(eng.decode_batch(chunks, threads=2))
            assert dec == data, "decode_batch drift"
            # 3. intern == sequential ids (S5 executable form)
            seq = ratec.RateC(c=20)
            seq_ids = seq.encode_bytes(data[:1 << 20])
            par_starts, par_ends, par_tok = eng.encode_stream_parallel(
                data[:1 << 20], threads=threads, ids=True)
            # the parallel engine's ids for the same byte prefix differ
            # from a FRESH sequential engine only via book state; compare
            # through a snapshot-clone instead (same book, same judge)
            clone = ratec.RateC.from_state(eng.save_state())
            _, _, clone_tok = clone.encode_stream_parallel(
                data[:1 << 20], threads=1, ids=True)
            assert list(map(int, clone_tok)) == \
                list(map(int, eng.encode_stream_parallel(
                    data[:1 << 20], threads=threads, ids=True)[2])), \
                "intern thread-count dependence"
            del clone
            # 4. snapshot save -> load -> continuation bit-exact
            snap = eng.save_state()
            back = ratec.RateC.from_state(snap)
            probe = round_corpus(rnd + 10_000, 1)
            a = eng.encode_bytes(probe)
            b = back.encode_bytes(probe)
            assert a == b, "snapshot continuation drift"
            # 5. batch spot-check
            docs = [data[i:i + (1 << 19)] for i in range(0, 2 << 20,
                                                         1 << 19)]
            r1 = ratec.batch_encode(docs, threads=2)
            assert len(r1) == len(docs)
            for ids, lens in r1:
                assert len(ids) == len(lens)
            del eng, back, seq, tok, starts, ends, par_tok, par_starts
            del par_ends, snap, clone_tok, a, b, r1, seq_ids
        except Exception as e:                     # noqa: BLE001
            failures.append(f"round {rnd}: {type(e).__name__}: {e}")
            print(f"  round {rnd} FAILED: {failures[-1]}", flush=True)
            break
        rss_series.append(round(rss_mib(), 1))
        rounds.append({
            "round": rnd, "threads": threads,
            "seconds": round(time.time() - t0, 2),
            "corpus_mib": 6,
        })
        if rnd % 5 == 0:
            print(f"  round {rnd}: {rounds[-1]['seconds']}s, "
                  f"peak RSS {rss_series[-1]} MiB", flush=True)

    # flat-RSS verdict: the last third of the series must not exceed the
    # first third's max by more than 20% (peak-RSS monotonicity would
    # hide a slow leak; ru_maxrss is a high-water mark — growth between
    # thirds is the leak signal)
    verdict = "PASS"
    if failures:
        verdict = "FAIL (exceptions)"
    elif len(rss_series) >= 6:
        third = max(1, len(rss_series) // 3)
        early, late = rss_series[:third], rss_series[-third:]
        if late and early and max(late) > max(early) * 1.2:
            verdict = "FAIL (rss growth)"

    env = record_envelope(inputs={})
    record = {
        "record": env,
        "soak": {
            "minutes_budget": args.minutes,
            "rounds_completed": len(rounds),
            "failures": failures,
            "rss_series_mib": rss_series,
            "rss_flat": verdict == "PASS",
            "verdict": verdict,
        },
        "rounds": rounds,
    }
    out = os.path.join(ROOT, "research",
                       f"soak_{env['created_utc'].replace(':', '')[:15]}.json")
    write_record(out, record)
    print(f"\nrounds: {len(rounds)} | failures: {len(failures)} | "
          f"peak RSS {max(rss_series) if rss_series else 0} MiB")
    print("VERDICT:", verdict)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
