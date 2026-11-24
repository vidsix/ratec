# v0.9.1 — the hardening release

The first stop on the road to v1.0.0 (the stability contract lives in
[docs/PLAN_v0.9.md](docs/PLAN_v0.9.md)). v0.9.0 made the ids lane fast;
v0.9.1 makes it *trustworthy at scale*: a measured perf anchor, proofs
discharged before the code they license, an endianness-honest snapshot
format, a coded error taxonomy, the beyond-4-GiB prep helper, and the
first ecosystem bridge. Every number below is in a committed record.

## The quiet-box envelope (the 1.0 floors anchor)

`research/quietbox_envelope.json` — min-of-3-round medians, round-robin
interleave across all lanes, cold AND warm stream pairs, full
provenance (loadavg, governor probe, affinity, exactness 4/4). The 1.0
per-lane floors are **ratios** of this record: a re-issue on a better
host lifts the floors, never the ratios. On the same record, the P1.2
gate: ids lane **103.4% of spans-only cold** / **99.4% warm** (gate
≥ 95%: PASS). BENCHMARKS.md's lane table now regenerates from the
record (`make tables` guards README *and* BENCHMARKS).

## Proofs, discharged proof-first (S6, S7, S9)

- **S6 — the read-stable merge protocol** (`lean/RatecIntern.lean`):
  a reader probing the live tables while the ordered merge inserts
  resolves the sequential id at ANY linearization step; a miss has no
  third outcome. This is the license the cold-book parallel intern
  needed — discharged before any read-stable C lands, because the
  pipelined attempt was unsound (the v0.9.0 finding).
- **S7 — decode completeness** (`lean/RatecDecode.lean`): with the
  intern's book and ids, decode reproduces the stream byte for byte;
  a decode fails exactly when some id is out of range — never a
  partial output.
- **S9 — snapshot rejection totality** (`lean/RatecSnapshot.lean`):
  exactly three load outcomes; the version check precedes body
  validation, so `SNAPSHOT_VERSION` means "format version", never
  "corrupt bytes".

CI now runs the proof gate: a **lean job** (elan from the committed
`lean/lean-toolchain` pin, active-sorry scan, all six files compile
no-sorry) and an **aarch64 job** (build + full gates on arm64).

## Snapshot v8 — byte order is the format's, not the host's

The v7 layout, explicitly little-endian; the prune section's presence
moves into a header flag. v2..v7 still load (read-side back-compat);
the write path is v8-only. Three gates: the sha256-bound golden battery
+ bit-exact continuations, an independent pure-Python LE parser walking
the golden to exact EOF, and an **executable big-endian-host
simulation** — a v8 blob's bytes are identical across the normal and
BE-simulated builds, and a LE-written v6 golden correctly fails on the
simulated BE host (the historical limitation v8 removes, demonstrated
instead of folklore).

## The coded error taxonomy

Every `RatecError` carries `.code` from `ratec.RatecErrorCode` —
TILING, BOOKLESS, CAP_EXHAUSTED, BUFFER_LIMIT, SNAPSHOT_INVALID,
SNAPSHOT_VERSION, ARGUMENT, STATE — 34 raise sites mapped, asserted
per site by `tests/test_errors.py` plus a source-level audit. Pipelines
branch on codes, never on message strings.

## Scale and pre-sizing

- **`chunked_prep`**: an arbitrarily long stream (lazy chunks, flat
  RSS via `sink=`) split into ≤ `engine_bytes` segments on fresh
  engines — byte-identical to explicit per-segment engines. Gate: an
  **8 GiB** synthetic prep job (32 × 256 MiB engines on this 3 GB-RAM
  sandbox) with per-segment sha256-identical ids/spans.
- **Public `reserve=`** on `RateC` / `batch_encode` / `batch_spans`,
  with the instrumented half of the gate: `rehash_count()`.

## The ecosystem bridge

**`export_tokenizer_json`** — a trained engine's book becomes an HF
`tokenizer.json` WordLevel vocab (latin-1 byte↔char, invertible by
construction). Tested contract: `vocab[token] == id` for every id the
engine produces, and the vocab inverse decodes a ratec ids stream to
exactly `decode_bytes(ids)`.

## Gates

make test green (suite + bit-exactness gate + fuzz + snapshot fuzz +
tables drift), pytest green (43 legacy + 17 new), `make tables --check`
PASS on README and BENCHMARKS, all six LEAN files compile no-sorry on
4.36.0.
