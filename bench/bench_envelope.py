#!/usr/bin/env python3
"""The quiet-box envelope (PLAN_v0.9 P1.1) — the reference record every
per-lane perf floor is a ratio of.

Why this exists: absolute MB/s on the recycled sandbox swing tens of
percent between sessions (BENCHMARKS.md host-variance warning), so the
1.0 stability contract needs an ANCHOR, not a number. The anchor is a
protocol-quieted record:

  * round-robin interleave — every lane advances one rep per rotation,
    so host drift hits all lanes equally instead of biasing whoever
    runs second (the bench_vs_tiktoken discipline, generalized to all
    lanes);
  * min-of-N — ROUNDS independent rounds, each lane reduced to its
    per-round median, the envelope = the minimum of the medians. The
    minimum is the quietest window the host gave us, which is exactly
    the floor semantics the 1.0 gates need;
  * cold AND warm stream pairs — the cold pair (fresh engine per rep)
    is the prep-pipeline shape and carries the P1.2 gate ratio
    (ids >= 95% of spans-only); the warm pair is the repeated-stream
    LM-prep shape where the intern pass pre-probe hits a saturated
    book.

"Quiet-box" is honest about what it is: this sandbox is a shared host
and exposes no cpufreq governor; the quieting is the protocol above,
recorded with full provenance (load average before/after, governor
probe result, affinity, CPU state) so a genuinely controlled host can
re-issue the record later.

Outputs: research/quietbox_envelope.json (committed) + a printed table.
The BENCHMARKS.md lane table is regenerated from this record by
`make tables` (drift-guarded, like the README tables).
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
import tiktoken

from record_util import record_envelope, write_record

ROUNDS = 3             # independent rounds; envelope = min of round medians
REPS = 9               # per lane per round (round-robin rotation)
TIK_REPS = 3           # tiktoken reference reps per domain per round
DOMAINS = ["english", "code", "chinese", "base64"]
FIXTURES = os.path.join(ROOT, "tests", "data", "fixtures")


def loadavg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return [float(p) for p in parts[:3]]


def governor():
    """CPU governor probe — a container usually has none; record the
    answer either way (the provenance must not invent a quiet host)."""
    p = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return "unavailable (container: no cpufreq sysfs)"


def cpufreq_khz():
    vals = []
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("cpu MHz"):
                    vals.append(float(line.split(":", 1)[1]))
    except OSError:
        pass
    return {"min": min(vals), "max": max(vals)} if vals else None


# ----------------------------------------------------------- the workload
LINE = (b"def encode_stream(data: bytes) -> list:  # the quick brown "
        b"fox jumps over the lazy dog\n    out = []\n    for i, b in "
        b"enumerate(data):\n        if b % 7:\n            out.append("
        b"(i * 31 + b) & 0xFFFF)\n# utf8: \xe4\xbd\xa0\xe5\xa5\xbd"
        b" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")
N = 4 << 20
DATA = (LINE * (N // len(LINE) + 1))[:N]

# scored spans ship with the repo (identical bytes everywhere)
spans = {d: open(os.path.join(FIXTURES, f"{d}.span.bin"), "rb").read()
         for d in DOMAINS}
rtext = {d: spans[d].decode("utf-8") for d in DOMAINS}

LANES = ["encode_bytes_warm", "spans_arrays", "stream_spans_cold",
         "stream_ids_cold", "stream_spans_warm", "stream_ids_warm",
         "decode_bytes", "decode_batch", "batch_encode"]
TIKKEY = "tiktoken_r50k"

# lane state (engines created once; fresh-per-rep engines built inline)
eng_bytes = ratec.RateC(c=20, store_book=True)
eng_pool = ratec.RateC(c=20, store_book=False, pf=2)
eng_sp = ratec.RateC(c=20, store_book=True)          # warm stream pair
eng_sp.encode_stream_parallel(DATA, threads=2)        # warm it untimed
eng_ids = ratec.RateC(c=20, store_book=True)
eng_ids.encode_stream_parallel(DATA, threads=2, ids=True)  # warm untimed
_, _, ids_ref = eng_ids.encode_stream_parallel(DATA, threads=2, ids=True)
# contiguous slices (decode_batch requires C-contiguous arrays)
ids_chunks = [ids_ref[i * len(ids_ref) // 8:(i + 1) * len(ids_ref) // 8]
              for i in range(8)]
r50k = tiktoken.get_encoding("r50k_base")

# round-robin timing: one rotation = one rep of every lane
times = {k: [] for k in LANES}
times_tik = {d: [] for d in DOMAINS}
round_medians = []          # per round: {lane: median}

la_start = loadavg()

for rnd in range(ROUNDS):
    t = {k: [] for k in LANES}
    tt = {d: [] for d in DOMAINS}
    for rep in range(REPS):
        # -- encode_bytes, persistent engine (book persists = bench() semantics)
        t0 = time.perf_counter(); eng_bytes.encode_bytes(DATA)
        t["encode_bytes_warm"].append(time.perf_counter() - t0)
        # -- spans as_arrays, pool semantics (reset per rep)
        eng_pool.reset()
        t0 = time.perf_counter(); eng_pool.encode_to_spans(DATA, as_arrays=True)
        t["spans_arrays"].append(time.perf_counter() - t0)
        # -- cold stream pair: fresh engine per rep (prep-pipeline shape)
        e = ratec.RateC(c=20, store_book=True)
        t0 = time.perf_counter(); e.encode_stream_parallel(DATA, threads=2)
        t["stream_spans_cold"].append(time.perf_counter() - t0)
        del e
        e = ratec.RateC(c=20, store_book=True)
        t0 = time.perf_counter()
        e.encode_stream_parallel(DATA, threads=2, ids=True)
        t["stream_ids_cold"].append(time.perf_counter() - t0)
        del e
        # -- warm stream pair: persistent engine, saturated book
        t0 = time.perf_counter()
        eng_sp.encode_stream_parallel(DATA, threads=2, as_arrays=True)
        t["stream_spans_warm"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        eng_ids.encode_stream_parallel(DATA, threads=2, ids=True)
        t["stream_ids_warm"].append(time.perf_counter() - t0)
        # -- decode lanes
        t0 = time.perf_counter(); eng_ids.decode_bytes(ids_ref)
        t["decode_bytes"].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); eng_ids.decode_batch(ids_chunks, threads=2)
        t["decode_batch"].append(time.perf_counter() - t0)
        # -- batch_encode: 4 x 1 MiB docs, threads=2
        docs = [DATA[i:i + (1 << 20)] for i in range(0, N, 1 << 20)]
        t0 = time.perf_counter(); ratec.batch_encode(docs, c=20, threads=2)
        t["batch_encode"].append(time.perf_counter() - t0)
        # -- tiktoken reference, interleaved into the same rotation
        if rep < TIK_REPS:
            for d in DOMAINS:
                t0 = time.perf_counter(); r50k.encode_ordinary(rtext[d])
                tt[d].append(time.perf_counter() - t0)
    med = {k: statistics.median(v) for k, v in t.items()}
    med[TIKKEY] = {d: statistics.median(v) for d, v in tt.items()}
    round_medians.append(med)
    print(f"round {rnd + 1}/{ROUNDS} done "
          f"(loadavg {loadavg()[0]:.2f})", flush=True)

la_end = loadavg()

envelope = {}
for k in LANES:
    envelope[k] = {
        "mb_s": min(N / m[k] / 1e6 for m in round_medians if k in m),
        "round_medians_s": [m[k] for m in round_medians],
    }
tik_ref = {}
for d in DOMAINS:
    n = len(spans[d])
    tik_ref[d] = {"bytes": n,
                  "mb_s": min(n / m[TIKKEY][d] / 1e6
                              for m in round_medians),
                  "round_medians_s": [m[TIKKEY][d] for m in round_medians]}

# the P1.2 gate ratio, cold and warm, from the SAME record
ratio = {
    "ids_cold_over_spans_cold":
        envelope["stream_ids_cold"]["mb_s"] /
        envelope["stream_spans_cold"]["mb_s"],
    "ids_warm_over_spans_warm":
        envelope["stream_ids_warm"]["mb_s"] /
        envelope["stream_spans_warm"]["mb_s"],
}

# exactness cross-check on the shipped domains (the record must prove
# the engine it timed is the exact engine)
exactness = {}
for d in DOMAINS:
    ref = None
    p = os.path.join(FIXTURES, f"{d}.starts.npy")
    if os.path.exists(p):
        import numpy as np
        ref = np.load(p)
        sp = ratec.RateC(c=20, store_book=False, pf=2).encode_to_spans(spans[d])
        exactness[d] = bool(np.array_equal(
            np.array([s for s, _, _ in sp], dtype=np.int64),
            ref.astype(np.int64)))
    else:
        exactness[d] = None

# the documented entry point, recorded for cross-check (same box, same run)
bench_call = ratec.bench(n=N)

env = record_envelope(inputs={f"fixture_{d}": f"tests/data/fixtures/{d}.span.bin"
                              for d in DOMAINS})
env["host_state"] = {
    "loadavg_start": la_start, "loadavg_end": la_end,
    "governor": governor(), "cpufreq_khz": cpufreq_khz(),
    "affinity_cpus": len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity") else os.cpu_count(),
    "quieting": "protocol-quieted: shared sandbox, min-of-round-medians, "
                "round-robin interleave; no cpufreq governor exposed",
}
env["runs"] = {
    "rounds": ROUNDS, "reps_per_round": REPS,
    "tiktoken_reps_per_round": TIK_REPS,
    "interleave": "round-robin across all lanes within each rep rotation",
    "aggregate": "median per round; envelope = min of round medians (min-of-N)",
}

record = {
    "record": env,
    "protocol": {
        "name": "quietbox_envelope",
        "stream": "4 MiB deterministic mixed code/text/UTF-8 (ratec.bench construction)",
        "c": 20, "threads": 2, "lane": "exact",
        "lanes": LANES + [TIKKEY],
        "lane_semantics": {
            "encode_bytes_warm": "persistent engine, book persists across reps (ratec.bench semantics)",
            "spans_arrays": "pool engine store_book=False, reset() per rep",
            "stream_*_cold": "fresh engine per rep — the prep-pipeline cold-book shape",
            "stream_*_warm": "persistent engine saturated with one untimed call",
            "decode_bytes": "single-thread full-book read of the stream ids",
            "decode_batch": "8-way id fan, threads=2",
            "batch_encode": "4 x 1 MiB docs, threads=2",
            "tiktoken_r50k": "reference judge on the 4 shipped fixture domains",
        },
        "gates": {
            "ids_lane_ratio": "cold ids >= 95% of cold spans-only (PLAN_v0.9 P1.2)",
        },
    },
    "lanes": envelope,
    "ratios": ratio,
    "tiktoken_reference": tik_ref,
    "exactness": exactness,
    "bench_call": bench_call,
}

out = os.path.join(ROOT, "research", "quietbox_envelope.json")
write_record(out, record)
print(f"\nquiet-box envelope written: {out}")
print(f"{'lane':28s} {'MB/s':>10s}   round medians (MB/s)")
for k in LANES:
    meds = [N / m[k] / 1e6 for m in round_medians]
    print(f"{k:28s} {envelope[k]['mb_s']:10.2f}   "
          + " ".join(f"{v:.2f}" for v in meds))
for d in DOMAINS:
    meds = [len(spans[d]) / m[TIKKEY][d] / 1e6 for m in round_medians]
    print(f"{TIKKEY + ':' + d:28s} {tik_ref[d]['mb_s']:10.2f}   "
          + " ".join(f"{v:.2f}" for v in meds))
print(f"\nids/spans cold: {ratio['ids_cold_over_spans_cold'] * 100:.1f}%  "
      f"(gate >= 95%)")
print(f"ids/spans warm: {ratio['ids_warm_over_spans_warm'] * 100:.1f}%")
print("exactness:", exactness)
