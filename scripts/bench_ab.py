#!/usr/bin/env python3
"""bench_ab.py — interleaved A/B benchmark for two engine builds.

Single-run benchmarks on a shared box drift by 10-20% between runs,
which has repeatedly produced false keeps and false rejections in the
v0.8.x ledger (see CHANGELOG v0.8.6). This harness interleaves A and B
measurements over multiple rounds and takes the per-lane median, so
host-load drift cancels instead of biasing one side.

Usage:
    python3 scripts/bench_ab.py <pathA> <pathB> [--rounds 4] [--json out]

Each path must be an importable checkout (ratec built in place). The
fixtures come from pathB/tests/data/fixtures/. Lanes: spans (book),
spans (bookless), warm bookless, ids (book), decode_bytes.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np


def load(path):
    for m in list(sys.modules):
        if m.startswith("ratec"):
            del sys.modules[m]
    sys.path.insert(0, path)
    import ratec
    sys.path.pop(0)
    return sys.modules.pop("ratec")


def best(f, k=3):
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def lanes(rc, BIG):
    res = {}
    res["spans_book"] = best(lambda: rc.RateC(c=20, store_book=True).encode_to_spans(BIG))
    res["spans_nobook"] = best(lambda: rc.RateC(c=20, store_book=False).encode_to_spans(BIG))
    e = rc.RateC(c=20, store_book=False)
    e.encode_to_spans(BIG)
    res["warm"] = best(lambda: e.encode_to_spans(BIG))
    e2 = rc.RateC(c=20, store_book=True)
    e2.encode_to_spans(BIG)                      # warm the judge
    ids2 = e2.encode(BIG)
    res["ids_book"] = best(lambda: e2.encode(BIG))
    res["decode"] = best(lambda: e2.decode_bytes(ids2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pathA")
    ap.add_argument("pathB")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    A, B = load(args.pathA), load(args.pathB)
    BIG = open(f"{args.pathB}/tests/data/fixtures/english.span.bin", "rb").read() * 10
    N = len(BIG)
    rb = B.RateC(c=20, store_book=True)
    ids = np.array(rb.encode(BIG), dtype=np.uint32)
    ra = A.RateC(c=20, store_book=True)
    assert np.array_equal(np.array(ra.encode(BIG), dtype=np.uint32), ids), \
        "A and B engines disagree on ids — not a benchmark, a bug"

    keys = ("spans_book", "spans_nobook", "warm", "ids_book", "decode")
    samples = {k: {"A": [], "B": []} for k in keys}
    for r in range(args.rounds):
        # ABBA: alternate the run order per round so host-load drift
        # cannot systematically favor whichever build runs second
        if r % 2 == 0:
            a = lanes(A, BIG); b = lanes(B, BIG)
        else:
            b = lanes(B, BIG); a = lanes(A, BIG)
        for k in keys:
            samples[k]["A"].append(a[k])
            samples[k]["B"].append(b[k])
        print(f"round {r + 1}/{args.rounds} done")

    print(f"\n{'lane':<14}{'A MB/s':>10}{'B MB/s':>10}{'time delta':>12}")
    out = {"N": N, "rounds": args.rounds}
    for k in keys:
        ma = statistics.median(samples[k]["A"])
        mb = statistics.median(samples[k]["B"])
        unit = 1e6 if k != "decode" else 1e6
        out[k] = {"A_s": ma, "B_s": mb, "delta_pct": (mb / ma - 1) * 100,
                  "A_per_s": N / ma / unit, "B_per_s": N / mb / unit}
        print(f"{k:<14}{out[k]['A_per_s']:>10.2f}{out[k]['B_per_s']:>10.2f}"
              f"{out[k]['delta_pct']:>11.1f}%")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
