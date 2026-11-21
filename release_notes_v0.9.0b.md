# v0.9.0b release notes — the token-stream beta

v0.9.0b is the beta for v0.9.0. It continues from v0.8.9 with three
deliverables: the parallel path now produces the token stream, the
interface documents its own performance contract with measured numbers,
and a third machine-checked file makes the fast lane's approximation
provably safe. The exact lane never left its bit-frozen state.

## What ships

**`encode_stream_parallel(..., ids=True)`** — the LM-prep token stream
for the parallel segmentation. Returns `(starts, ends, ids)` uint32
arrays. The parallel workers stay bookless and GIL-free; one
sequential intern-only pass (`Engine.intern_spans`) interns the
finished spans into the instance's book. Gates, all verified by the
suite:

- interning the SEQUENTIAL engine's own spans reproduces the
  sequential `encode_bytes` ids bit-exactly (interning is a pure
  function of span contents + order);
- the span tiling is validated in C (spans must partition the stream)
  before any interning — bad tilings raise, never silently corrupt;
- `decode_bytes(ids)` round-trips exactly in both runs and snapshot
  modes;
- the book persists across calls: repeated streams share one
  vocabulary;
- `ids=True` without `store_book=True` raises.

**`ratec.bench()` + `BENCHMARKS.md`** — the performance contract with
clear numbers: six entry points, best-of-N, a host-variance warning
(the same code measured 7.3–17.8 MB/s across this sandbox's sessions —
the RELATIVE ordering is the contract), per-entry GIL/threading
behavior, the honest tried-and-rejected ledger, and the historical
milestone table. Reproduce: `python -c "import ratec; ratec.bench()"`.

**LEAN `RatecTiling.lean`** — the tiling theorem, machine-checked
(no sorry): the patcher emits spans that partition the stream for ANY
surprisal function. This is the structural guarantee behind the fast
lane: approximation can change WHERE cuts fall, never WHETHER the
segmentation is valid, so decode always round-trips.

## What was tried and rejected

The ledger's deferred FST-lane AVX2 item was built, measured three
ways, and rejected on evidence: the isolated float32 term pass is 1.7×
faster (5.6 vs 9.8 ns/byte), but the judge loop is probe/memory-bound
— FP is ~9% of the byte cost — and lane-level integration measured
−4% (helper call) and −8.9% (whole-loop AVX2 clone). Fully reverted;
legacy FST stays (+2.5–7.5% over exact, host-dependent). Details in
CHANGELOG and BENCHMARKS.md.

## Gates

`make test` green (7 scripts, fuzz gate 12/12, bit-exactness gate),
`pytest tests/` 35/35 (30 + 5 new ids tests). Engine version 0.9.0b0,
single-sourced from pyproject.toml.
