# v0.9.0 release notes — parallel intern, machine-checked token streams

Shipped from one release candidate (v0.9.0rc, kept in
`release_notes_v0.9.0rc.md`). The RC's polish pass caught a real
warm-engine bug (below), updated the README to v0.9, moved the
machine-checked LEAN files into the repo (`lean/`), and re-ran every
gate; the code ships as verified there.

v0.9.0 closes the gap the beta named: the token stream (ids=True) no
longer pays a serial intern pass — interning is multi-core, its
correctness is machine-checked, and the exact lane remains bit-frozen.

## What ships

**Thread-parallel intern (`intern_threads`)** — `encode_stream_parallel
(..., ids=True)` now interns on `intern_threads` workers (default =
the stream's `threads`). Design: private per-worker tables over
contiguous span chunks (zero content copies) → read-only pre-probe
against the frozen book (hits resolve immediately) → one ordered merge
interning only the misses → parallel gather. The ids are bit-exactly
the sequential pass's for ANY thread count.

Three layers of verification:
- tests: K ∈ {2,3,4,8,16} all reproduce the sequential ids bit-exactly;
  warm-book second interns are identical; api round-trips at
  intern_threads 1 and 4; stress shapes (single span, all-identical
  spans, empty) and rejections (bad tiling, bookless) covered;
- LEAN `RatecIntern.lean` (S5): the chunked merge assigns the
  sequential first-seen ids for ANY chunking — no sorry;
- the same bit-exactness discipline as every release (the exact lane's
  outputs never changed in this cycle).

**Measured** (10.7 MiB english corpus, this host-day): intern pass
cold book 1.37–1.43×, warm book 2.23–2.62× vs sequential; ids lane
23.93 vs spans 26.04 MB/s in the shipped `ratec.bench()` run (the
remaining gap is the ordered distinct-insert merge on a cold book —
warm books, the repeated-stream LM-prep pattern, make it nearly free).

## Honest ledger

- The first implementation pipelined the merge into the worker pool
  (measured 1.39–1.49×) and was UNSOUND: pre-probes read the book
  while the merge inserted and grew its hash table — segfault on cold
  books. Reverted to barriered phases; the warm-book win survives
  because the pre-probe (not the overlap) is what shrinks the serial
  merge.
- The FST-lane AVX2 rejection from the beta stands.

## Fixed in the polish pass

- **`encode_stream_parallel(ids=True)` failed on a continued engine**:
  the api's simple path handed `encode_to_spans` output (LIFETIME-
  ABSOLUTE offsets — the documented convention the goldens and the
  dataset builder rely on) to the intern pass, which validates
  call-local tiling. The ids path now rebases its span arrays to the
  call before interning (ids are content-derived — the rebasing cannot
  change them); both calls of a bench-style loop round-trip and the
  simple path matches the chunked path's call-relative spans.
  Regression test: `test_warm_engine_stream_ids_call_relative`.
- **README brought to v0.9**: the status block, the API rows
  (`ids=True` / `intern_threads` / `ratec.bench()`), a new
  "The ids lane (v0.8–v0.9)" section, roadmap items 15–16, the layout
  (tests, `lean/`, docs), and a torn paragraph repaired.
- **`lean/` now ships in the repo** (RateC, RatecSerial, RatecTiling,
  RatecIntern — pure Lean 4, no mathlib, no `sorry`; verify with
  `lean lean/RatecIntern.lean`).

## Gates

`make test` green (tables 4/4, fuzz 12/12, bit-exactness gate),
`pytest tests/` 43/43. Version 0.9.0, single-sourced from
pyproject.toml.
