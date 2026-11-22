# RateC benchmarks — the performance contract with clear numbers

Every entry point, what it costs, and where the design promises the
speed. Numbers are **measured, not aspirational**: reproduce them on any
box with `python -c "import ratec; ratec.bench()"`.

**Host-variance warning (read first).** This project's numbers come from
a recycled cloud sandbox whose CPU state swings between sessions: the
SAME v0.8.x sequential encoder measured 7.3, 8.7, 9.3, 12.4 and 17.8
MB/s on different days on this one box. Absolute MB/s are therefore
host-day-dependent; what the design promises — and what the ABBA
interleaved harness actually verifies — is the RELATIVE ordering and the
scaling behavior below. Always compare entries within one `bench()` run,
never across sessions. The quiet-box envelope below is the anchor that
makes this survivable: the 1.0 per-lane floors are RATIOS of the
envelope, so a re-issue on a different host lifts the floors, never the
ratios.

## The quiet-box envelope (the 1.0 floors anchor — PLAN_v0.9 P1.1)

Produced by `bench/bench_envelope.py` into
`research/quietbox_envelope.json` (drift-guarded: regenerate the table
with `make tables`). "Quiet" is a protocol, not a host claim: 3
independent rounds, every lane advancing one rep per rotation
(round-robin across lanes, so drift hits all lanes equally instead of
biasing whoever runs second), each lane reduced to its per-round
median, and the envelope = the minimum of the medians — the quietest
window the host gave us. Cold AND warm stream pairs are recorded: the
cold pair (fresh engine per rep) is the prep-pipeline shape and carries
the P1.2 gate ratio (ids ≥ 95% of spans-only); the warm pair is the
repeated-stream LM-prep shape. The record carries full provenance:
load average before/after, the governor probe result (this container
exposes no cpufreq sysfs — recorded as such, never invented), CPU
state, affinity, and exactness cross-checks on all four shipped
domains.

<!-- table:quietbox generated from research/quietbox_envelope.json — do not edit; run make tables -->
Envelope: min-of-3-round medians, round-robin interleave, 2026-11-22; provenance (loadavg, governor probe, affinity, exactness cross-checks) in the record. The 1.0 per-lane floors are RATIOS of these numbers, so a re-issue on a better host lifts the floors, never the ratios.

| lane (exact, 4 MiB stream) | envelope MB/s | round medians |
|---|---:|---|
| `RateC.encode_bytes` | 17.78 | 17.78 18.59 18.52 — sequential judge+patcher; persistent engine, book persists across reps (the `bench()` semantics) |
| `RateC.encode_to_spans` | 18.34 | 18.35 18.40 18.34 — tuple-free `as_arrays=True`; pool engine, `reset()` per rep |
| `encode_stream_parallel` (spans) | 22.37 | 22.37 23.97 22.44 — fresh engine per rep — the prep-pipeline cold shape |
| `…(ids=True)` cold | 23.14 | 23.75 23.30 23.14 — fresh engine per rep — includes the cold-book intern merge |
| `encode_stream_parallel` (spans, warm) | 23.27 | 24.13 23.99 23.27 — persistent engine, saturated book |
| `…(ids=True)` warm | 23.13 | 23.50 23.13 23.79 — saturated book — the intern pre-probe hits |
| `RateC.decode_bytes` | 10757.58 | 10801.93 10757.58 10802.29 — single-thread full-book read of the stream ids |
| `RateC.decode_batch(threads=2)` | 13724.82 | 13724.82 13835.60 13958.45 — 8-way id fan |
| `ratec.batch_encode(threads=2)` | 23.17 | 24.04 24.07 23.17 — 4 x 1 MiB docs, per-doc cold engines |

| gate ratio (P1.2) | value | gate |
|---|---:|---|
| ids/spans cold | 103.4% | ≥ 95%: PASS |
| ids/spans warm | 99.4% | ≥ 95%: PASS |

| tiktoken r50k reference | envelope MB/s |
|---|---:|
| english | 14.10 |
| code | 12.16 |
| chinese | 13.62 |
| base64 | 9.12 |

The judge-parity anchor: ratec's judge core at tiktoken parity in isolation is an attribution claim (PLAN_v0.8) — these rows are the same-corpus reference the next attribution re-run starts from.
<!-- /table:quietbox -->

## The contract per entry point

**`encode_bytes` / `encode` (sequential, exact lane)** — the reference
path: judge adaptation + patcher + book interning in one pass. Output is
bit-frozen (see CHANGELOG: `-ffp-contract=off` discipline, snapshot
fuzz gate). Use it when you need the exact single-engine segmentation:
it is also the ids ground truth that `intern_spans` reproduces.

**`encode_to_spans(as_arrays=True)`** — same judge work, tuple-free
output: two uint32 arrays handed to Python without per-span objects and
without re-acquiring the GIL per span. Cost ≈ `encode_bytes` minus the
book (store_book=False engines skip interning); the win over the old
tuple path was +16% sequential (12.42 → 14.48 MB/s in the v0.8.9 ABBA
on a slower host-day; the tuple path had serialized every worker before
that).

**`encode_stream_parallel` (runs mode)** — the throughput mode for one
long stream: `head_chunks` sequential chunks adapt the judge, then each
worker clones the head state ONCE and encodes one contiguous run —
`workers − 1` approximation boundaries instead of one per chunk.
Scales with cores until the sequential head dominates; spans tile the
input exactly, so decode round-trips.

**`ids=True` + `intern_threads` (v0.9.0)** — adds the LM-prep token
stream to the parallel path. The workers stay bookless and GIL-free;
a thread-parallel intern pass (`Engine.intern_spans_parallel`,
`intern_threads` workers, default = the stream's `threads`) interns
contiguous chunks of the span array into PRIVATE open-addressing
tables (zero content copies), pre-probes its distinct patches against
the frozen book (hits resolve immediately), then ONE ordered merge
interns only the misses into this instance's book and a parallel
gather remaps. Because interning is a pure function of span contents
+ order, the ids are bit-exactly the ids a sequential engine produces
for the same spans — for ANY thread count (machine-checked in
`RatecIntern.lean`, verified by tests for K = 2…16), and the tiling is
validated in C before any interning, so `decode_bytes(ids)`
round-trips exactly. The book persists across calls: repeated streams
share one vocabulary. Measured at the pass level on the 10.7 MiB
corpus: cold book 1.37–1.43×, warm book (the repeated-stream LM-prep
case) 2.23–2.62× vs the sequential intern.

**`decode_bytes` / `decode_batch`** — pure book read (no judge).
`decode_bytes` takes the whole id array through the C buffer fast path;
`decode_batch(ids_list, threads=k)` fans per-document decodes over a
3-phase thread pool with the book shared read-only.

**`ratec.batch_encode`** — many independent documents, each encoded
cold-start on its own engine, GIL released. Identical per-doc ids to a
fresh sequential engine; the fastest path when the corpus is many
small/medium docs instead of one long stream.

**`exact=False` (the fast lane)** — `RateC(c=20, exact=False)`: identical
structure, approximate log2. Historically +2.5–7.5% over the exact lane
on this box (host-dependent); its outputs are NOT stable across
versions and may differ from the exact lane's segmentation (documented
deviation). The LEAN companions (`RatecTiling.lean` v0.9.0b,
`RatecIntern.lean` v0.9.0) prove the structural guarantees: the
patcher's tiling is correct for ANY surprisal function, so
approximation can never break the round-trip, and the parallel intern
assigns the sequential first-seen ids for ANY chunking.

## What was tried and rejected (the honest ledger)

- **FST-lane AVX2 float32 term pass (v0.9.0b experiment).** The six
  per-order double divisions were replaced by one 8-wide float32 pass
  with `vrcpps` + one Newton step. Isolated microbench: **1.7× faster**
  (5.6 vs 9.8 ns per byte). Lane-level: **−4% vs legacy** (helper-call
  integration), **−8.9%** (inline AVX2 clone of the whole loop). The
  judge loop is probe/memory-bound — the FP chain is ~9% of the byte
  cost — and the −mavx2 clone's codegen hurt the surrounding loop more
  than the FP savings. Fully reverted; the exact lane stayed
  bit-frozen throughout (verified by the suite's bit-exactness gate).
- **BET C (cmap prefetch, v0.8.6):** +2.3–8.7% time on every lane.
  REJECTED. **BET B (order-2 exact-line prefetch):** shipped.
  **BET E (tot2 prefetch):** noise-level, reverted.

## Historical milestones (same 10.7 MiB english corpus unless noted)

| version | change | effect (that host-day) |
|---------|--------|------------------------|
| v0.7.0 | spans-as-bytes API, judge-store strip | sequential base |
| v0.8.6 | prefetch sweeps (BET B shipped) | +2.3–8.7% lane |
| v0.8.8 | runs mode (zero per-chunk resets) | 14.6 → 20.6 MB/s thr=8 |
| v0.8.9 | tuple-free span output + parallel decode_batch | spans 12.42 → 14.48 MB/s seq; decode 120 → 177 Mids/s thr=2 |
| v0.9.0b | `ids=True` intern pass + tiling theorems | token stream ≈ free over spans-only |
| v0.9.0 | thread-parallel intern (`intern_threads`) + S5 theorems | intern pass 2.2–2.6× warm / 1.4× cold; ids lane ≈ spans lane |

Reproduce everything: `make check` (suite + bit-exactness + fuzz gates),
`python -c "import ratec; ratec.bench()"` for the table, and the LEAN
files under `lean/` with a Lean 4 toolchain (`lean lean/RatecIntern.lean`
must print nothing).
