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
never across sessions.

## Measured on this host (2026-11-21, `ratec.bench()`, 4 MiB mixed
text/code/UTF-8 stream, c=20, threads=2, best-of-3)

| entry point                        | MB/s   | note |
|------------------------------------|--------:|------|
| `RateC.encode_bytes`               | 17.78  | sequential judge+patcher, ids + book |
| `RateC.encode_to_spans`            | 18.01  | tuple-free `as_arrays=True` output — ≈ encode_bytes (the v0.8.9 GIL-free raw path) |
| `RateC.encode_stream_parallel`     | 24.52  | runs mode, threads=2, spans only |
| `…encode_stream_parallel(ids=True)`| 23.93  | v0.9.0: spans **+ token stream**; the intern pass is itself thread-parallel (`intern_threads`, default = threads) — warm books make it ≈ free, cold books pay the ordered distinct-insert merge |
| `RateC.decode_bytes`               | 13423  | memcpy/hash-bound book read; scales with `decode_batch(threads=k)` (v0.8.9: 120 → 177 Mids/s at 2 threads on the same corpus shape) |
| `ratec.batch_encode`               | 28.60  | per-doc cold engines, thread-parallel, 1 MiB docs |

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

**`fast=True` (the FST lane)** — `RateC(c=20, exact=False)`: identical
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

Reproduce everything: `make check` (suite + bit-exactness + fuzz gates)
and `python -c "import ratec; ratec.bench()"`.
