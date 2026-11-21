# Changelog

## v0.8.9 — the GIL-free + parallel-decode release (internal)

Two structural multithreaded costs were named, measured, and removed:
the per-span Python-object build that held the GIL inside every stream
worker, and the single-threaded decode path (decode is a pure book read
nobody had parallelized). Plus the project's first machine-checked
execution-model theorems in LEAN.

### Kept

- **Tuple-free span output (C)**: `encode_spans_raw` / `feed_raw` /
  `flush_raw` return two little-endian uint32 buffers (memcpy from the
  emit vectors — zero per-span Python objects); Python
  `as_arrays=True` on `encode_to_spans` and `encode_stream_parallel`
  wraps + rebases in numpy. The tuple path costs ~3 PyObjects per span
  built GIL-HELD inside every stream worker (~2.6M objects per
  10.7 MiB). Gates: arrays == tuples bit-exact on the sequential lane
  and both stream modes (sampled position-wise, 0.8-1.2M spans).
  ABBA: sequential spans lane 12.42 -> **14.48 MB/s (+17% time)** —
  the new fastest spans path; runs lane +1% on this 2-vCPU box (the
  tuple build hides behind the other worker's C work at nworkers=2;
  the structural gain is per-worker GIL-freedom, which removes the
  scaling wall on many-core boxes where worker count > 2).
- **Thread-parallel decode_batch (C)**: decoding is a PURE READ of the
  book (ids -> ln/off -> arena; no judge, no state change), so one
  read-only core is SHARED across the C thread pool — no clones, no
  state sync. Three phases: parallel bounds+size pass, GIL-held
  output allocation, parallel pipelined gather (the decode_bytes
  kernel, replicated per job). `RateC.decode_batch(ids_list,
  threads)` with sequential fallback for unvalidated int lists.
  Gates: output == decode_bytes loop (24 docs), deterministic error
  reporting (first bad id, index and sequence reported), book
  required. ABBA: 600K ids / 300 docs — py-loop 120.2 Mids/s ->
  C pool **176.9 Mids/s at thr=2 (1.5x)**; box caps at ~1.5x per the
  SMT 2-vCPU budget.
- **LEAN theorems (download/ratec-lean/RatecSerial.lean, compiles
  clean, no sorry)**: S1 the fold decomposition lemma (streams
  `xs ++ ys` fold exactly as `ys` from the post-`xs` state — the
  mathematical foundation of batch/runs parallelism), S2 the adaptive
  step does not commute (no multiset map-reduce can reproduce the
  serial chain), S3 surprisal is history-dependent (concrete 256/1 vs
  256/2), S4 the judge+patcher pipeline is itself a fold inheriting
  S1+S2. Together: *the cheapest exact algorithm is a serial fold per
  stream, and parallelism must (and may exactly) decompose by
  contiguous runs* — the prior RateC.lean (Kraft/cut-band/codes)
  still compiles alongside.
- Suite 30/30 (24 existing + 6 new: arrays equality, parallel decode
  equality + error paths + fallback).

### Measured and rejected (ledger continues)

- The runs lane gains only +1% from tuple removal at nworkers=2 on
  this box — NOT a rejection: the cost model is per-worker GIL-held
  time (linear in spans) vs C work (parallel); the wall appears when
  workers > GIL-throughput allows (~2-3 here, ~6-8 on real boxes).
  The ABBA harness on THIS box cannot demonstrate a many-core win;
  recorded as structural.
- decode's bounds-check pass (random ln[] reads) is O(total ids) and
  stays inside the C pool's parallel phase 1 — no measurable serial
  residue at 600K ids.

### Findings for the record

- The abstracted step function is provably non-commutative on
  concrete bytes (LEAN S2), which closes the "could a cheaper
  reordering/multiset algorithm exist?" question for the exact lane:
  any exact algorithm must apply the state chain in stream order.
  Cheaper single-stream algorithms must therefore come from constant
  factors (SIMD/FP restructuring), not reorganization — consistent
  with the v0.8.7 neutered-orders decomposition and the deferred
  FST-lane AVX2 note.
- encode_to_spans is a stream CONTINUATION (fresh semantics only via
  reset()) — benchmark gates must reset() between calls or they
  measure a warm judge with fewer, longer patches (833K vs 1046K
  spans in a scouting pass that initially looked like an equality
  failure).

## v0.8.8 — the multicore-throughput release (internal)

The stream-parallel mode was advertised as "the throughput mode" while
actually measuring 0.45x SEQUENTIAL on two cores — every chunk paid a
full snapshot wire-parse + map rebuild plus mid-chunk map rehashes.
This version rebuilds the multicore path around engine cloning and a
better decomposition, and documents exactly where the remaining costs
live.

### Kept

- **Engine clone / restore / reserve (C)**: `core_restore` syncs one
  bookless core onto another bit-identically via union-of-touched-pages
  memcpy (dirty-bitmap driven, maps + cnt2 + small tables + patcher
  state); `core_clone_from` allocates a layout-matching copy; exposed as
  Engine `clone()` / `restore_from()` / `reserve(n)`. Verified: a
  restored worker produces the parent's continuation spans exactly
  (unit-tested), and snapshot-mode stream-parallel output is
  span-identical to the v0.8.7 wire roundtrip (1.2M spans, 10.7 MiB).
- **runs mode (new stream-parallel default)**: each worker clones the
  head state ONCE and encodes ONE CONTIGUOUS RUN, continuing its own
  judge — zero per-chunk state resets, zero rehashes, and only
  workers-1 approximation boundaries. ABBA vs v0.8.7 (4 rounds,
  medians, 2 threads, 10.7 MiB english): auto-chunk lane 7.45 ->
  **9.75 MB/s (+31%)**; 1 MiB-chunk lane 7.42 -> 9.78 (+32%). Boundary
  agreement with sequential (exact-span metric, docstring parity):
  **0.59 vs snapshot's 0.24** (head_chunks=2: 0.65) — closer to the
  single-engine view AND faster. Sequential stays 12.35 MB/s on this
  SMT-limited 2-vCPU box; the runs-mode advantage compounds per core
  on real multi-core boxes (setup is one-time per worker).
- **snapshot mode preserved** (`mode="snapshot"`): v0.8.x semantics
  bit-exact, now on the clone/restore machinery (+12% ABBA at 1 MiB
  chunks — rehash elimination via reserve).
- Suite 24/24 (18 existing + 6 new clone/restore/stream tests).

### Measured and rejected (ledger continues)

- restore_from (82 ms at a 5.4 MiB-head state) is NOT cheaper than the
  old load_state wire rebuild (66 ms) — the state copy is ~256 MB of
  map pages; the 1 MiB-chunk speedup comes from REHASH elimination
  (reserve pre-sizing), not setup cost. Runs mode wins by avoiding
  the per-chunk reset entirely; snapshot mode keeps it for semantics
  parity. A fork-style CoW clone would make snapshot-mode resets cheap
  but has no portable thread implementation.
- The 2-vCPU box caps 2-thread encode at ~1.3x sequential (matches the
  batch-path scaling from v0.8.7) — thr=4 "regressions" in past ledgers
  were oversubscription, now documented as such.

### Findings for the record

- Baseline hygiene: with english.span.bin at 267 KB, a x4 corpus
  (1.07 MB) falls through `n <= chunk` to the sequential fallback —
  an earlier scouting run mistook that for stream-parallel and
  measured 0.23x sequential. Real baselines need n > chunk (the
  docstring's own 10.7 MiB). All numbers in this entry are at 10.7 MiB.
- The FST-lane FP chain (AVX2 restructure) was scoped and deferred:
  the FST lane is only +3.9% over exact, so the FP chain is not a
  dominant cost; the multicore axis pays more.

## v0.8.7 — the batch-throughput release (internal)

The structural work this version started as the order-3 direct array
and ended somewhere better: two big decomposition experiments rewrote
what we know about where encode time actually goes, and the kept bet
delivers the largest single win of the 0.8.x line.

### Kept

- **Batch engine pooling** (C, bworker): one POOLED engine per worker
  thread, `core_reset_keep` between docs, destroy-and-rebuild only on
  a failed job. Previously every doc paid a full cold start (64 MB
  cnt2 first-touch faults, map alloc + growth, book pages). Jobs of
  one rc_batch call share uniform params, so one core serves the
  thread. ABBA-interleaved A/B vs v0.8.5 (4 rounds, medians):
  - 200x5 KB docs, 2 threads (the LM-prep pattern): **2.76 ->
    17.92 MB/s, -84.6% time (6.5x)**.
  - 10x267 KB docs, 1 thread: -16.1% time. 2 threads: -20.6% time.
  Bit-exactness: per-doc cold-judge semantics unchanged (reset path
  is suite-tested; per-doc outputs identical to fresh engines).
- **bench_ab.py ABBA upgrade**: rounds alternate A/B run order so
  host-load drift cannot favor whichever build runs second — caught
  a machine-drift artifact that had briefly looked like a pooled
  batch regression on 2 threads.

### Measured and rejected (ledger continues)

- **Order-3 direct array** (16 GB virtual rows, v7-wire-preserving
  synthesis, entry-count-tracked growth): fully implemented, ALL
  exactness gates passed (ids equality, snapshot cross-compat both
  directions, roundtrip semantics), then ABBA-rejected: +0.7-3.3%
  time on EVERY lane including untouched decode. Mechanism: order-3
  contexts are numerous AND sparse — dense 1 KB rows touch far more
  cache lines and ~16x the TLB pages than the map's packed 64 B
  entries. The probe elimination cannot pay for the storage blowup.
  **This also kills the naive v0.9 fused-window-slot group design**
  (same sparse-context blowup applies); exact fused slots are further
  blocked by shared-prefix state fragmentation (a ctx3 entry cannot
  live inside one of its 256 ctx4 children without splitting).
- **popcnt window index** (u16 presence mask in Ent.pad): rejected on
  paper before implementation — the presence mask is over BYTE VALUES
  (256 bits), which does not fit the 2 pad bytes; a 64 B Ent cannot
  hold 16x u16 counts plus a value mask. win_scan is already SWAR and
  branchless — near-optimal for the layout.
- Decomposition experiments (throwaway builds, warm lanes): removing
  orders 4-5 ENTIRELY moves warm throughput only +1% (english) /
  +4% (base64). The v0.7-era "47% map traffic" figure is obsolete;
  the warm loop is a serial spine (order 0-2 arrays + FP chain +
  patcher) with no dominant component. **The v0.9 fused-slot headliner
  targets a ghost; the roadmap needs rewriting around the batch/
  parallel-throughput axis this version just proved.**

### Findings for the record

- Snapshot save/load is NOT a byte-level fixpoint (+4 bytes per
  roundtrip) — verified PRE-EXISTING on v0.8.5, semantically neutral
  (loaded states continue identically; suite unchanged).
- core_reserve (6n map presize) is called by the batch path only; the
  one-shot path still rehashes mid-encode for large corpora — future
  candidate, no bench-lane effect at current sizes.
- Suite 18/18 (5:08) on the 0.8.7 engine.

## v0.8.6 — the measurement-integrity release (internal)

Finding first: single-run benchmarks on this box drift 10-20%
between runs. Re-measuring the v0.8.x line with an interleaved A/B
harness (multi-round, per-lane median — scripts/bench_ab.py, now part
of the repo) reclassified results:

### Ledger corrections (interleaved A/B, 4 rounds, median)

- **Decode gather kernel CONFIRMED**: v0.8.2 -> v0.8.6 decode_bytes
  91.3 -> 105.3 Mids/s (-13.3% time, interleaved). The software-
  pipelining mechanism is real; the decode line stands.
- **BET B (order-2 exact-line prefetch) RECLASSIFIED as noise**:
  v0.8.4 vs v0.8.5 differ by -0.3..+0.9% on every lane — the
  "−6.7..−10%" v0.8.5 claims were single-run artifacts. The build is
  kept (harmless, mechanically sound, exact) but the win claim is
  withdrawn. Encode throughput is unchanged across v0.8.4..v0.8.6.
- **BET C (cmap prefetch) REJECTED by A/B**: +2.3..+8.7% time on all
  lanes — it had "won" −14..−17% in single runs. tc slots are
  covered by locality after promotion; the extra hash+prefetch
  chains cost more than they save. Not shipped.
- **BET D (fast lane) REJECTED**: only +3-4% vs exact — the cost is
  KT-state-bound, not log2-bound; strengthens the v0.9 fused-slot
  case. Not exposed.
- **BET E (tot2 prefetch) REJECTED**: noise level.
- **PF default re-confirmed interleaved**: pf=2 beats pf=1 by 1.9-
  2.4% and pf=3 by 2.6-3.5% (both lanes); pf=0 collapses (~30%).
  Default stays 2.

### Shipped

- scripts/bench_ab.py: the interleaved A/B harness (rounds, medians,
  ids-agreement assert between builds, decode lane). All future
  keep/reject decisions run through it.
- v0.8.6 engine == v0.8.5 engine modulo a PF-block comment
  (sub-1% A/B verified); suite 18/18.

## v0.8.5 — the absolute-optimization pass (internal)

Campaign: attack the per-byte memory-traffic block directly. Two
structural bets implemented, measured on the 4-lane ladder with a
full bit-exactness gate (per-fixture ids + 0.8.4-snapshot
continuation ids captured BEFORE any edit).

### Kept

- **BET B — order-2 exact-line prefetch** (C, run_body PF block):
  the future byte buf[i+p] is known at prefetch distance p, so the
  exact cnt2 line it will touch is prefetched (was: only the order
  3-5 map slots). Final build vs 0.8.4 baseline, best-of-3:
  spans_book −7.9%, spans_nobook −6.7%, warm −9.4%, ids_book −10.0%
  (6.79/8.69/10.86/5.73 -> 7.37/9.32/11.98/6.36 MB/s; run noise on
  this box is ±5-10%, an earlier run showed ids up to −21%).
  Bit-exact: ids identical on all four fixtures; v0.8.4 snapshot
  loads and continues identically.
- **PF sweep re-run under BET B**: pf=0 collapses (4.93/4.36/5.95
  MB/s — prefetch dominates), pf=1 vs default 2 is mixed within
  noise. PF_DEFAULT stays 2. The 0.8.1 "PF=1 −10%" verdict does not
  transfer to the BET-B regime; default unchanged anyway.

### Measured and rejected (ledger continues)

- **BET A — tot-into-row fusion** (Ct1/Ct2 {tot, cnt[256]} rows,
  v6-wire-preserving writer/loader reshape): all exactness gates
  passed, but LOST 27-45% on the ladder. Root cause: the 1028 B row
  stride destroys the power-of-2 row indexing (h2*1024 was a pure
  shift; h2*1028 needs a multiply) and un-aligns every row start
  from 64 B cache lines. The saved tot-stream traffic does not pay
  for making every access more expensive. Reverted via git checkout;
  the writer/loader reshape code is preserved in this entry for any
  v0.9 revisit (the fused-window-slot design must keep 1024 B rows —
  e.g. tot in a parallel head array per row GROUP, not per row).
- **Order-1 exact-line prefetch**: +2-7% on every lane — the 256 KB
  cnt1 table is L2-resident; extra prefetches cost instructions and
  pollute. Reverted to order-2-only.

### Gates

- Suite 18/18 (5:10) on the 0.8.5 engine.

## v0.8.4 — introspection + validate surface, approximation-quality data (internal)

### Kept

- **`book_counts` / `counts()`** (C + api): per-id occurrence counts
  (window counts since last prune, index == id) as raw uint32 bytes.
  Verified: sum(counts) == n_patches on the english fixture;
  prune preview (survivors at prune_min=2: 9255/29640) makes prune
  tuning measurable without serializing.
- **`validate(data)`** (api): gigatoken CLI --validate parity —
  roundtrip exactness on a state COPY (caller's streaming position
  untouched), patch statistics, bytes/patch, book size. Verified on
  english: roundtrip_ok=True, 6.86 bytes/patch.
- **head_chunks agreement curve documented** (encode_stream_parallel
  docstring): span-boundary agreement with sequential encode rises
  0.23 / 0.35 / 0.46 / 0.56 for head_chunks = 1 / 2 / 3 / 4 on a
  10.7 MiB corpus with 1 MiB chunks. Default stays 1 (minimal
  sequential fraction); the knob is now an informed choice instead
  of a guess.

### Measured and deprioritized (ledger continues)

- **C-level decode_batch**: python-loop batch decode already runs at
  119.8 Mids/s aggregate (L2-warm small chunks beat the single-call
  117.6); per-sequence cost 4.6 us total. A C batch saves ~1-2 us of
  Python entry per call — not worth the C surface now.
- No new external research this pass (wordchipper/TokenDagger/gigatoken
  lessons already mined in 0.8.2-0.8.3); increments drawn from the
  existing v0.9.0 roadmap items 2/4.
- Suite 18/18 (5:28) on the 0.8.4 engine.

## v0.8.3 — decode gather kernel + export/interop surface (internal)

Research pass: TokenDagger (repo unreachable — 404, nothing on
record) and wordchipper (docs.rs: compile-time DFA lexers via logos,
layered vocab views, token-width shrinkage, foldhash/SwissTable,
aho-corasick special-token scan). Transferable discipline: copy
volume + dependent-load latency dominate short-record decode; the
rest has no RateC analog (no regex pretokenizer, ids already uint32,
maps already open-addressed with rolling hashes).

### Kept

- **decode gather kernel** (C): software-pipelined memcpy fan-out —
  the next patch's (off, ln) load + arena prefetch issue while the
  current patch copies. Code fixture: 94.5 -> **117.6 Mids/s
  (+24%)**; worst-case random-id gather 115.8 (pipeline holds under
  cache-hostile ids). First implementation shipped a one-patch
  offset bug (registers overwritten before the copy) — caught by the
  roundtrip assert, fixed with double-buffered registers.
- **book export surface**: `Engine.book_dump()` (C, vocabulary as
  [bytes] with index == id), `to_ranks()` ({bytes: id}, tiktoken
  ranks semantics), `save_ranks(path)` (.tiktoken format, base64
  piece + rank). The introspection/interop half of the v0.9 compat
  story; snapshot wire format untouched.
- **`encode_files(paths)`** (api.py): gigatoken encode_files parity —
  multi-file driver over encode_file (verified: separator split [1,2]
  docs, 54572/54572/1458 patches).
- **`encode_stream_parallel(chunk=None)`** auto-tune: n/threads
  clamped to [1, 8] MiB; verified exact tiling (0..n, no gaps) with
  legitimately different chunk boundaries at cpu_count=2.

### Notes

- Batch persistent thread pool considered and deprioritized: spawn
  cost (~2 ms/call) is noise against 4 MiB chunk work; revisit only
  if many-small-docs workloads show up in the profile.
- Suite 18/18 (5:20) on the 0.8.3 engine.

## v0.8.2 — the research-driven increments (internal)

Sourced from a survey of tiktoken-adjacent speed work and the
gigatoken library (marcelroed/gigatoken, Rust, static-vocab BPE at
GB/s via SIMD pretokenization + a long-tailed repeat-word cache +
whole-file self-managed parallel ingestion).

### Kept

- **decode buffer-protocol fast path** (C): numpy uint32 / buffer
  inputs decode straight from memory — no per-id PyLong boxing.
  Measured on the code fixture: 14.9 (array via sequence protocol) /
  44.7 (list) -> **94.5 Mids/s** (2.1x over list). Bounds checks and
  error messages unchanged; list path untouched.
- **`decode_bytes` numpy wiring + `decode_batch`** (api.py): buffer
  objects route to the fast path automatically; batch decode helper.
- **`encode_file(path, separator=...)`** (api.py): gigatoken
  TextFileSource parity — mmap the file, split on the document
  separator, batch-encode on the C thread pool (independent cold
  judges per doc, same documented semantics as encode_batch).

### Measured and rejected (ledger continues)

- **Patch-intern recurrence memo** (gigatoken repeat-cache analog):
  direct-mapped, memcmp-verified, prune/reset-invalidated — LOST 3%
  on the spans-with-book path. Their cache skips an entire BPE
  pipeline for repeated words; ours would skip one probe whose first
  slot is usually already the answer, while costing a line fetch per
  patch. Reverted clean (BMemoEnt count in source: 0).
- Confirmed already-present: batch worker **map presizing**
  (`2^bits >= 6n` per doc, v0.8.0) — gigatoken's warm-start lesson
  was already applied here.
- Structural note for the record: gigatoken's SIMD pretokenization
  has no RateC analog — our pretokenization-equivalent (the judge)
  is per-byte adaptive and inherently serial; the v0.9 fused window
  slots attack the same cost center (per-byte map traffic) from the
  memory-layout side instead.

## v0.9.0 roadmap (draft — the polished release)

1. **Fused window slots** (headliner): one 128 B slot per order-5
   window holding all three higher orders' KT state — 2 cache lines
   per byte instead of 3, one probe instead of three; v6-wire
   compatible (per-order logical entries on the wire, orphan head
   contexts to an overflow map). Targets the profiled 47% block.
2. **Features**: `encode_files` (multi-file driver), book/vocab
   export (tiktoken-ranks-style introspection), compat validate
   mode in the bench, snapshot v7 only if the fused slots demand it.
3. **Speed**: `encode_stream_parallel` chunk-size auto-tuning,
   decode gather kernel (memcpy fan-out unrolled), gigatoken bench
   on the 0.9 engine for the before/after record.
4. **Polish**: full API docs pass, error-message audit, packaging
   (sdist/wheel), CI matrix, release page for 0.9.0.

## v0.8.1 — the speed campaign (internal)

### Kept

- **fos hit-path reorder**: the map probe's hot path now checks the key
  first (insert branch demoted off the fetch/predict stream). Measured
  on the 4-domain ladder (61 MB, best-of-3): surprise 9.79 -> 11.16
  MB/s (+14%), spans 7.54 -> 8.07 (+7%), spans-with-book 5.63 -> 5.90
  (+5%). Bit-exact by construction (same slots, same semantics); full
  suite green.

### Measured and rejected (the ledger is the deliverable)

- **Per-window slot locator** (one tagged 32 B entry caching the three
  order-3/4/5 tmap slot indices, flushed on growth): LOST 31% on
  surprise (9.79 -> 6.76). The locator line is a fourth dependent miss
  that serializes the three Ent fetches that previously issued in
  parallel; the probe "overhead" the profile blamed is actually the
  data-fetch cost. Reverted.
- **PF default 1** (prefetch depth): +4% in a warm-state C harness but
  -10% in the real cold-engine Python path (6.75 vs 7.54) — the harness
  regime lied. Reverted; pf=2 stands.
- **Hugepages on the cnt2 table**: nil (24.10 vs 24.23 MB/s).

### Profile (block-excision, 16 MB, C harness)

full 23.4 | -map-probes 43.8 (47% of runtime) | -log2 25.3 (8%) |
-direct-orders 25.4 (9%) | pf=0 20.6 (prefetch worth +14%).

### New bench

- **bench/bench_gigatoken.py**: sustained GB-scale throughput (1.07 GB
  tiled 4-domain stream, one warm engine per arm, 4 MiB ingestion
  chunks). Engine 0.8.1 sustained: spans+book 4.26 MB/s (0.254 Mtok/s),
  judge-only 6.18 MB/s. A gigatoken (1e9 tokens) at the spans rate
  needs ~65 min today.

### v0.9 headliner (design locked by today's profile)

**Fused window slots**: one 128 B slot per order-5 window holding all
three higher orders' KT state (2 cache lines per byte instead of 3,
one probe instead of three), wire-format-compatible (v6 on the wire:
per-order logical entries; orphan head contexts to an overflow map).
Targets the 47% probe block; the locator lesson says the win must come
from FEWER fetched lines, not from caching fetches.

## Unreleased

### Speed: merged book hash slots (+12% ids path on top of the intern)

- The book's split probe arrays (8-byte hashes, 4-byte values — two
  cache lines per probe) merge into one 16-byte slot array; probe
  arithmetic, values and ids are bit-exact by construction. Ids path:
  cold 4.9 -> 5.5 MB/s, warm 6.6 -> 7.4 MB/s. Cumulative this cycle:
  cold 1.8 -> 5.5 (+205%), warm 4.1 -> 7.4 (+80%).
- A fused lockstep emit was attempted and REVERTED: the engine's
  persistent OutVecs diverge in capacity across encode modes (spans-only
  never grows the ids vector) and the fused capacity check segfaulted —
  caught by the fuzz gate, root-caused, per-vector growth documented as
  load-bearing.

### Feature: batch_encode — thread-parallel LM-prep tokens

- `ratec.batch_encode(docs, c, threads)` returns (ids int32, lens int32)
  per doc, bit-exact vs cold one-shot engines; the C batch worker gains
  an ids mode (store_book + all three output vectors per job). On this
  2-core box, aggregate throughput exceeds the single-engine warm rate.

### Trainer: chunked + checkpointed cross-entropy for big-vocab heads

- The logits tensor is batch*ctx*vocab*4B — 2.4 GB at C=14's 147,712-row
  head, a global OOM on a 3 GB box (root-caused from dmesg). TwinLM
  gains trunk()/forward split; the trainer applies the head per
  16-position chunk under gradient checkpointing (identical math; one
  chunk's logits in memory; C<=11 rows keep the unchunked path
  bit-identically). Unblocks the pre-registered C=14 sweep arm.

### Gates + docs

- P3.3 record-drift hardening: the tables gate re-verifies sha256 of
  every bound record input present on disk (36 files, no drift) — stale
  provenance now fails the suite.
- `docs/ARCHITECTURE.md`: the definitive architecture document — layers,
  invariants, concurrency model, defaults audit; every number cited from
  a committed record.
- README: batch_encode, current ids-path numbers, roadmap refresh.

### Finding: live-engine heap pin vs batch throughput (documented, unregressed in fresh process)

- A long-lived engine (e.g. the single-thread bench pool) pins its judge
  maps in the heap; batch workers spawned afterwards crawl against it:
  4x slowdown, fully recovered by freeing the engine (0.47s -> 0.10s for
  the 4-doc batch). The bench_vs_tiktoken protocol now frees the pool
  engine before the batch arm; `bench_tiktoken.json` re-recorded
  accordingly (batch 2T: ratec 5.51 vs tiktoken 14.0 MB/s). A
  fresh-process batch (the LM-prep usage) measures ~9 MB/s aggregate.
  An engine-side allocator hardening (worker-side trim / mmap-backed
  maps) is queued for v0.9. A mallopt(M_MMAP_THRESHOLD) attempt was
  tested and reverted as ineffective.

### C-sweep extension (pre-registered, in flight)

- `RATEC_SWEEP=14` extension arm registered BEFORE its run (its
  interpretations live in the runner + the record's extension block);
  C=12's recorded rows and the frozen protocol are untouched. Training
  is checkpoint-resumable (every 50 steps).

### Speed: rolling-hash book intern (PLAN_v0.8 P1, landed early)

- `book_intern` re-ran fnv1a64 over the whole patch content at every
  patch close; `_rcengine.c` now maintains the rolling hash per
  extension byte (identical value by the fnv recurrence — bit-exact by
  construction; the full gate battery passes, including the pinned
  id/span fixtures; snapshot bytes stay format v2..v7 unchanged — the
  hash is derived state, recomputed from the restored obuf on load).
  Measured on a 2 MB corpus mix, ids path (LM data prep): warm
  4.1 -> 6.6 MB/s (+61%), cold 1.8 -> 4.9 MB/s (+172%). Honest cost
  attribution (PLAN_v0.8): the judge alone runs 9.6 MB/s vs tiktoken's
  9.8 — prediction is at parity; the remaining ids-path gap is patcher
  bookkeeping. Earlier book-on/off and memoization speed claims in the
  plan are retracted as measurement artifacts / already-built features.
- New gates: `tests/test_streams.py` (producer protocol invariants:
  budget-bounded id space, exact byte conservation, straddle-allowance
  partition, prune-arm segmentation identity, id-arithmetic round-trip,
  manifest sha bindings) and `tests/test_speed_floor.py` (loose 2.5 MB/s
  floor on the ids path).

## v0.7.0 — 2026-11 (the scale-ladder release)

The s2400 probe's two points become a measured trend: the 4-arm × 4-rung
scale ladder, the engine feature its hypothesis points at (frequency-aware
book pruning), and the LM-optimal C sweep — every record provenance-
enveloped, every number table-generated and drift-guarded. Plan:
`docs/PLAN_v0.7.md`.

### P1 — the scale ladder
- **`research/run_ladder.py`** — 4 arms (`bpe`, `ratec_direct`,
  `ratec` factorized, `bytes`) × 4 rungs (600/1200/2400/4800 steps,
  log-spaced ×2), the twin-study protocol frozen everywhere else
  (d=96/3 layers/4 heads/ctx 256/batch 16/seed 42, identical bytes,
  full-coverage byte-weighted BPB). Rung 600 is seeded from the committed
  `research/twin_{arm}.json` records (protocol-identity asserted field by
  field); every rung ≥ 1200 is a fresh run (OneCycle recomputed per rung —
  no warm-starting across rungs, which would confound the trend), executed
  through `twin_study.py` so all training/eval/checkpoint machinery stays
  in the one protocol file. Resumable at two levels (completed rung
  records are skipped; an interrupted run resumes its checkpoint), and
  `research/twin_ladder.json` is rewritten after EVERY completed run — a
  killed runner never loses more than the run in flight. Both trend
  interpretations were pre-registered in the plan before any rung above
  600 was trained, so the record cannot be curve-fit post hoc.
- **Protocol bug found by the first ladder execution (P0-2 follow-up).**
  The v0.6.0 coverage assert assumed each domain's first val token is one
  byte; RateC val streams can open with a multi-byte patch, so the
  `ratec`/`ratec_direct` arms legitimately cover `val_bytes − 5` and the
  tightened assert failed by one byte. The assert now checks the exact
  invariant (`covered == val_bytes − first-token bytes`); recorded numbers
  were never affected (first-token bytes were never targets — the
  committed 600-step records carry `val_bytes_covered` consistent with
  the corrected invariant).

### P2 — frequency-aware book pruning (opt-in, default bit-exact)
- **Engine (`ratec.RateC(prune=True, prune_after=1<<20, prune_min=2)`).**
  The book gains per-entry window counts; at every patch close whose end
  offset crosses a multiple of `prune_after` bytes, entries whose
  since-last-prune count is below `prune_min` are dropped (default: drop
  singletons — the starvation mechanism the probe hypothesized) and the
  survivors are renumbered dense in first-seen order (in-place compaction,
  allocation-free, hash rebuilt). Boundaries are absolute stream offsets,
  so the prune sequence is identical for chunked and one-shot ingestion.
  Byte segmentation is unaffected — the patcher is book-independent —
  spans are identical with and without pruning; only id assignment churns
  at prune boundaries.
- **Default path byte-identical, snapshot format v6 unchanged; prune-mode
  snapshots are v7** (v6 layout + trailing prune section: interval,
  threshold, schedule, cumulative removals, window counts). v2–v6 readers
  reject v7 cleanly (version check); the reader accepts v2–v7 and keeps
  the strict trailing-garbage check — which is why the section rides on a
  version bump rather than trailing on v6 (the v0.6.0 hardening must not
  move). v7 load validation: interval/threshold ≥ 1, interval ≤ 2^48,
  schedule within one interval past `nseen`.
- **Invariants gated (`tests/test_prune.py`, 25 checks):** prune off ==
  prune kwarg absent (ids, spans, snapshot bytes); spans invariant to
  pruning; ids identical up to the first boundary; determinism across
  fresh engines AND feed chunkings (output level — snapshot bytes are
  layout-dependent and were never chunking-invariant, v0.6.0 included);
  within-epoch decode round-trips; cross-epoch ids cleanly rejected
  (RatecError); v7 snapshot resumes pruning exactly (open patch spans the
  save point); v6 snapshots load unchanged; `prune_min` semantics
  deterministic; empty-book continuation; bookless no-op; carry-mode
  composition; reset restores virgin behavior.
- **Snapshot fuzz gate extended to v7** (`tests/test_snapfuzz.py` §4b):
  targeted mutants over the prune section (version flips both directions,
  interval/threshold/schedule bounds, n_pruned, cnt array, section
  truncations), seeded bit flips and truncations over a live v7 payload,
  save→load→save identity.
- **The LM measurement** (`research/run_prune_lm.py` →
  `research/book_pruning.json`): ratec_direct at C=10, 2400 steps, pruned
  vs unpruned single-engine packed-layout streams — identical byte
  segmentation asserted per domain (token counts and lens), only the
  book's content survival and slot allocation differ;
  engine stats (book sizes, `n_pruned`) reported beside the BPB delta.
  Pre-registered: if the win is within the determinism band, the feature
  stays opt-in and the doc says so.

### P3 — the LM-optimal C sweep
- **`research/run_c_sweep.py` → `research/lm_c_sweep.json`**: ratec_direct
  at C ∈ {8, 10, 12}, 2400 steps each, on single-engine streams in the
  SHIPPED PACKED id layout (`research/build_streams.py --c C`: one engine
  tokenizes the whole corpus — chunked feed, one-shot-equivalent — and
  the val streams continue the SAME engine so val ids live in the train
  book, the shipped streams' proven design). A protocol bug in the first
  draft (raw dense book ids) would have grown the LM table with the book
  (145k+ rows at C=10, breaking compute-matching and filling the head
  softmax with dead rows); the producer now collapses book ids into the
  shipped `256 + (id−256) % 2^C + (len−2)·2^C` layout BEFORE the sweep
  ran — zero compute wasted — so every arm shares the fixed 256+9·2^C-row
  table and the comparison is parameter-exact across C. The library's own C sweep optimizes tokenizer
  rate; this one asks the downstream question at probe scale. Parameter
  growth with C is reported per run as an explicit confound; interpretations
  pre-registered (interior argmin vs monotone-to-edge).
- **`twin_study.py` gains `--c`, `--prefix`, `--lens_lidx`** (all
  default-preserving; `--lens_lidx` retained as the documented dense-id
  interface, unused by the sweep after the packing fix); provenance for
  both stream generations documented in `research/README.md`.

### P4 — tests
- `tests/test_prune.py` added to the suite (the 8th gate) and registered
  in `conftest.py` / `tests/test_suite.py`; the suite manifest records it
  like every gate.

## v0.6.0 — 2026-11 (the information release)

Everything published is now regenerable, every hard claim is guarded by a
test, and the snapshot load path — the attack surface — is hardened and
fuzz-gated. Full review and plan: `docs/PLAN_v0.6.md`. The default encode
path is bit-identical to v0.5.0 (bit-exactness gate green on all domains;
scale record re-produced row-identical after the hardening).

### P0 — honest numbers (protocol corrections)
- **P0-1 · scale-bench protocol correction (re-measured).** The v0.5.0
  scale record was produced by a flawed protocol: the "held-out" 128 KB
  probe was fed into the warming engine at every scale before being
  measured (up to 8 × 128 KB of self-ingestion), and prefixes were re-fed
  cumulatively. `bench/bench_scale.py` now feeds each prefix exactly once,
  snapshots the warming engine, and scores the probe on a disposable
  engine loaded from that snapshot; per-row `bytes_seen`, probe-tail
  UTF-8 trimming, and ratec/tiktoken versions + corpus sha256 are recorded.
  **Corrected result on the same 15.8 MB mix (C=20):** cold 0.157 →
  64 KB 0.151 (−4%) → 1 MB 0.146 (−7%) → 8 MB 0.147 (−7%, saturated) —
  the honest adaptation effect is ~−7%, not −52%; warm RateC holds ~2.9×
  fewer LM positions than tiktoken r50k_base, not 4.2–5.7× (the ~2.7×
  cold-architectural gap is unchanged). Engine untouched — measurement
  only. Old record kept verbatim:
  `research/scale_mix15.8MB_v0.5.0_superseded.json`.
- **P0-2 · twin study claims made assertable.** Eval coverage is now
  asserted in code (`covered >= val_bytes − domains`), the false
  "coverage asserted 100.00%" wording replaced everywhere (coverage is
  every val byte except each domain's first token), and the ≤9-byte
  patch-length cap of the twin streams is documented and asserted
  (`max(id) < V`) on every loaded stream.
- **P0-3 · README fertility sentence made table-derived** ("26–50%" →
  "26–60%", numbers identical to the per-domain table).
- **P0-4 · bench records emit actuals.** `bench/bench_vs_tiktoken.py`
  no longer writes false protocol fields (pf/reps now come from the
  running code, both library versions recorded, prebits actual); batch
  arms timed interleaved round-robin (medians) like the single-thread
  arms; dead `if False` blocks removed; hardcoded font path guarded (also
  in `bench_c_sweep.py`); figure label version from `__version__`.
- **P0-5 · parallel-fertility check demoted to print-only** (1.5 sanity
  guard); the 10% "informational" bound still failed foreign boxes.
- **P0-6 · API/CLI robustness.** `surprise --patch` no longer crashes
  (missing `--c` on the subparser); `--c` is float everywhere (fractional
  C reachable from the CLI); `patch_from_sur` dtype guard (1-D float32
  required, float64 cast, other dtypes rejected — float64 bytes silently
  reinterpreted as float32 produced garbage starts); `pf` docstring now
  matches the code (auto → `PF_DEFAULT` 2).

### P1 — snapshot hardening + format v6 (`load_state` was the attack surface)
- **P1-1 (HIGH)** `Ent.distinct` must be PROMOTED or 1..16 — values
  17..254 reached `win_scan`/the cold-byte branch and could write past
  the 64-byte slot. `tot >= distinct` and every inline count in
  `[1, tot]` are enforced; `distinct == 0` (impossible occupancy) and
  duplicate keys rejected; `tot == 0` rejected (occupancy marker).
- **P1-2 (HIGH)** the map rebuild's probe loop is now provably
  terminating: entry count bounded to 75% of capacity for tmap and cmap.
  A crafted 1 KB snapshot previously spun the interpreter forever with
  the GIL held. Regression-tested in a 30 s subprocess.
- **P1-3 (MED)** every book `(off, ln)` validated against `alen`;
  `bn == 0 ⇔ alen == 0`; ids bounded by the u32 contract.
- **P1-4 (MED)** snapshot format **v6** (write v6, read v2..v6): the
  per-row nonzero counter is u16 — the u8 counter silently dropped rows
  with all 256 counts nonzero (save→load not bit-exact on dense tables).
  Byte layout otherwise = v5.
- **P1-5 (MED)** book-hash sizing cap raised 26→31 with load factor < ½
  enforced, so legitimate large books load and the rebuild terminates;
  `map_capbits` is now SERIALIZED (a tuned engine used to restore as the
  default reserve ceiling); `cs`/`lim`/`carry` must be finite and in
  range; dirty bitmaps no longer leak on map re-alloc.
- **P1-6 (LOW)** `load_state` raises `RatecError` (was ValueError);
  big-endian builds fail at compile time (SWAR + snapshots are LE by
  construction); `Engine_surprise`/`Engine_encode_bytes`/`Engine_feedflush`
  map the recoverable map-exhaustion (−3) to `RatecError` WITHOUT
  poisoning — the −3 abort runs between bytes, so the engine stays
  usable and the documented "reset() recovers" is reachable from Python;
  `rc_batch` error paths release all buffers (were leaking the first k);
  `core_reset_keep` re-inits book tables lost to OOM.
- **New gate: `tests/test_snapfuzz.py` (9/9).** A Python-side v6 layout
  parser, 37 targeted field mutants (every P1 class), 214 truncations,
  1200 seeded bit flips with a raise-or-load-clean oracle, the P1-2 hang
  regression in a subprocess, and the RCZ1 container error-surface check
  — ≥1000 mutants per run, all deterministic. New fixtures:
  `v6_c20` / `v6_c33` / `v6_c20carry` snapshots + continuation goldens
  (generator `tests/gen_v6_goldens.py`, bound in `fixtures_meta.json`).

### P2 — record convention ("so we can put more information in it")
- `research/record_util.py`: the provenance envelope (ratec_version,
  deps, git_rev, host, created_utc, inputs sha256-bound) every producer
  emits in three lines.
- **README tables are GENERATED**: `make tables` rewrites the sentinel
  blocks from the committed records; `tests/test_tables.py` (new gate)
  fails on hand-edited drift and audits every `research/*.json` for the
  envelope (legacy records kept verbatim, explicitly marked). The
  fertility table now shows all four domains (code was silently omitted:
  0.412 → 0.202, −51%).
- Committed bench records: `research/bench_tiktoken.json`,
  `bench_c_sweep.json`; `make records` copies fresh results with
  envelopes and refuses silent result changes.
- `research/build_streams.py`: the twin study's ratec + bytes streams
  are producible in-repo from sha-bound corpora (book-mode C=10, len-cap
  9 documented, manifest binds every output); the bpe arm is honestly
  documented as out-of-repo. `RATEC_RUNS`/`RATEC_TWIN_DATA` defaults are
  repo-relative (P2-6); `RATEC_SCALE_PROBE`/`RATEC_SCALE_DATA` documented.

### P3 — tests that generate information
- `tests/test_fuzz.py` (new gate, 12/12): 118 seeded stream×C property
  cases (tiling, round-trip, chunked==one-shot, patcher==engine, batch==
  serial); judge C-independence bitwise across c∈{1,10,20,33,64}; fast
  lane measured (max rel. deviation 1.18e-07, record
  `research/fastlane_parity.json`); pf 0..4 bitwise-identical (prefetch
  depth only); boundary C; map exhaustion recovery; reset-vs-fresh
  bit-exactness; book-mode snapshot after `feed()`; carry+reset; servo
  brackets; `warm_state` path; CLI smoke with fractional `--c`.
- Suite manifest (`tests/test_manifest.json`): per-script exit + wall
  time, so green-skipped is distinguishable from green-passed.

### P5 — packaging
- `-march=native` out of the default build (plain x86-64 baseline;
  `RATEC_ARCH=native|x86-64-v2` opt-in) — wheels stop being SIGILL-bound;
  bit-exactness re-verified on the baseline build.
- Version single-sourced from `pyproject.toml` (injected via
  `-DRC_VERSION`, drift asserted); `[project.urls]` + license/OS
  classifiers.
- `MANIFEST.in`: the sdist ships the full gate (tests + fixtures +
  conftest + Makefile); sdist install smoke in CI.

### CI
- `.github/workflows/ci.yml`: ubuntu 3.10/3.12 + macOS 3.12 matrix, full
  `make test` + pytest + scale smoke; sdist install smoke; ASan/UBSan
  build fed the snapshot-fuzz corpus; a transformers-installed job so
  `test_smollm2_port` runs instead of skipping everywhere.
  **Landed:** the first push was scope-blocked (the deploy token lacked
  the `workflow` scope GitHub requires for `.github/` paths — the file
  was withheld, see the v0.6.0 release notes), and the identical commit
  landed on `main` with the next workflow-scoped push. Until the first
  Actions run is green, the same commands run locally via `make test`
  (validated in the v0.6.0 round, 7-gate PASS).

## v0.5.0 — 2026-11 (the design thesis, measured + CI)

- **Efficiency vs scale, measured** (`bench/bench_scale.py`, record in
  `research/scale_mix15.8MB.json`): the design claim — BPE's fertility is
  frozen at train time, RateC's adapts as it read — is now a number.
  One engine, log-spaced adaptation prefixes, the same fixed 128 KB
  held-out probe at every scale, C=20, 15.8 MB en/hr/zh/code mix:
  cold 0.157 → 256 KB 0.082 (−48%) → 1 MB 0.076 units/byte (−52%),
  saturating at ~0.076 through 8 MB while the book grows to 1.17 M
  entries. tiktoken r50k_base on the same probe: 0.431 flat — warm
  RateC holds 4.2–5.7× fewer LM positions (64 KB → 8 MB scales).
  **[Superseded in v0.6.0: the probe was not actually held out — see the
  protocol correction at the top of this file.]**
  Portable by default (runs on the shipped fixtures; also smoke-verified
  end-to-end on a fresh tree);
  `RATEC_SCALE_DATA=<corpus>` for the full curve, `make scale`.
- **Twin-study training-scale probe folded in** (same arms/arch/seed/bytes,
  4× steps): bpe 3.112 → 2.607 BPB, ratec_direct 4.796 → 3.818 BPB —
  the tokenization gap narrows 1.684 → 1.212 BPB (−28.1%); RateC improved
  1.94× more than BPE for the same 4× compute, and the gap narrowed in
  every domain. Direction confirmed; trend not established (two points,
  1.27M params). Digest: `research/twin_probe_s2400.json`; record:
  `docs/twin_study.md`.
- **Portability completion of the straggler suites**: `test_smollm2_port.py`
  **skips cleanly** without the optional `transformers` dependency
  (`pip install 'ratec[lm]'`) so a fresh-clone `make pytest` is green with
  zero extra installs; both straggler suites keep their sha256-bound
  fixture assertions; `research/twin_study.py` resolves its data/run
  directories from env vars (`RATEC_RUNS`, `RATEC_TWIN_DATA`) with
  resumable `--tag`/`--ckpt_every` checkpoints for long runs.
- **Subtree hygiene**: `ratec/.gitignore` (in-place extension, build/,
  results/), so fresh clones stay clean after `make build`.
- No engine changes: the default path is bit-identical to v0.4.1
  (gate green on all four domains).

## v0.4.1 — 2026-11 (external-replication round)

Shipped after an independent third-party replication of v0.4.0 (fresh box,
gcc 14.2 / Python 3.12): their external gate passed on all five domains
(tiling, `patch_from_sur` == engine, streaming parity, carry round-trip,
fractional-C monotonicity 14.0→23.75), base64 fertility replicated exactly
(0.500 vs tiktoken 0.773 units/byte), carry patcher measured −23% patches on
high-entropy streams, exact-lane speed landed inside the claimed 0.4–0.65× of
tiktoken. Findings from that round, all fixed here:

- **Portability**: no more absolute author paths. The scored spans, reference
  surprisal/starts (C=20), binary round-trip inputs, the u8_image raster
  (`ascent.raw`, 256 kB) and the 320 kB `lm_corpus.bin` snapshot-training
  corpus now ship under `tests/data/` (~11 MB total incl. reference arrays,
  sha256-bound in `fixtures_meta.json`); `tests/gate_bitexact.py`, both
  benches and all four `test_*.py` suites resolve data repo-relative. Env
  overrides: `RATEC_FIXTURES`, `RATEC_BAKEOFF_DATA` / `RATEC_BAKEOFF_CACHE`
  (full per-C goldens), `RATEC_OUT`.
- **Machine-independent goldens**: `test_v04.py` loads the captured corpus
  slice (`gate_slice.bin`, 160k) and the v4 continuation slice
  (`v4_cont_slice.bin`, 36k) from `tests/data/` instead of author-local
  corpora — the v4→v5 continuation goldens are now replayable byte-exactly on
  any machine, with sha256 assertions binding slices to goldens.
- **Parallel-fertility check**: hard invariants stay tiling + determinism;
  fertility is informational with a 15% soft bound (an external box measured
  ×1.105 against the previous 10% bound — snapshot-head judge state per chunk
  worker is box-dependent, not an engine invariant).
- **First-run DX**: `conftest.py` builds the engine in place when pytest runs
  from the repo root; `Makefile` adds `make build / test / pytest / bench`.
- C-sweep gate degrades honestly when per-C goldens are absent: golden
  equality is asserted at C=20 (shipped) and everywhere in private-cache mode;
  at other C it asserts tiling, strict monotonicity and
  `patch_from_sur == engine starts` (judge/patcher independence), recorded as
  `golden_mode` in `c_sweep.json`.
- Perf: default path bit-identical to v0.4.0 (gate green on all four domains).
- **Twin study, first result** (the standing BPE-vs-RateC LM experiment):
  compute-matched, identical 15.8M train / 659k val bytes, seed 42,
  full-coverage byte-weighted BPB — BPE 3.112 < raw bytes 4.147 <
  ratec_direct 4.796 ≈ ratec (factorized) 4.816. Interface is free
  (Δ 0.02 BPB); the tokenization gap is long-tail patch-type frequency
  starving tiny embeddings. Protocol, numbers and harness:
  `docs/twin_study.md`, `research/`.

## v0.4.0 — 2026-11

- **Fractional C**: `C` is a double everywhere (constructor, patcher,
  `patch_from_sur`, `batch`, snapshots). Snapshot format bumped to
  RCST v5 (`f64 C`, `pmode`, `carry`); v2–v4 blobs load unchanged —
  bit-exact continuation verified against fixtures captured with the
  v0.3.0 binary (`tests/data/v4_c{20,33}.bin`).
- **Credit carry-over patcher** (`RateC(carry=True)`, `patch_from_sur(
  ..., carry=True)`, `batch(..., carry=True)`): each patch may spend the
  previous patch's unused budget (capped at +C extra); forced escapes
  burn the carry; mean rate → C exactly. Bit-exact vs a Python golden on
  5 surprisal distributions × 4 C values; streaming parity holds.
  Measured −15.1% patches at C=20 on the mixed gate corpus; the carry
  patcher itself is cheaper than under-shoot (0.75×).
- **Rate servo**: `solve_c` (patch-count target — strictly monotone in C,
  exact bisection, judge-once/patch-many) and `solve_c_for_book`
  (book-size target — unimodal in C, coarse scan + local bisection,
  achieved value reported exactly).
- **Modality profiles** (`ratec.profiles`): registry-as-data profiles
  (`raw`, `pcm16`, `f32`, `u8_image`) with lossless byte transforms
  upstream of the untouched core. Measured patch-count wins at C=20:
  pcm16 −37.1%, f32 −9.5%, u8_image −9.6%; all round-trips bit-exact
  (47/47 checks incl. real photo, PCM, NaN/inf/denormal f32).
- **Docs**: `docs/v04_design.md` (landed / measured / declined record —
  run-detection fast path declined for exact mode: predict+update are
  fused in one probe; SIMD patcher deferred: 5.5% of runtime; GPU batch
  judge design note) and `docs/container.md` (container spec draft).
- Perf: default path bit-identical to v0.3.0 (gate green on all four
  domains); exact lane 9.15 MB/s judge on the mixed reference corpus.

## v0.3.0 — 2026-11

- `ratec.smollm2`: `RateCTokenizer` (frozen-vocab `PreTrainedTokenizer`),
  `encode_for_lm` contract — `(patch_id, patch_byte_length)` pairs with
  `sum(lens) == len(utf8_bytes)` —, `build_smollm2_ratec_config`
  (SmolLM2-135M-shaped `LlamaConfig`, torch-free), torch-guarded
  `SmolLM2RateCAdapter`.
- Interface low-latency pass: hot method binding, ids-only encode path.
- Fair C-sweep bench (`bench/bench_c_sweep.py`): tokens/byte and MB/s
  measured together, golden-verified per C.
- Docs: `docs/PROFILING.md`, `docs/smollm2_port.md`.

## v0.3 — 2026-11

- Two-stage judge/patch pipeline (RP=8) + reusable emit vectors;
  API latency wins kept; SIMD/pipeline variants tried and rejected by
  fair ablation (see `docs/PROFILING.md`).

## v0.2 — 2026-11

- v2 engine: u16 occurrence windows, 64 B line-aligned slots (+22% MB/s
  over v1 on the reference corpus); fast non-exact lane `exact=False`
  (≤ 1 float32 ulp, 100% patch-start agreement on test corpora);
  `encode_stream_parallel` — one stream across cores, deterministic;
  snapshot format RCZ1/v4.

## v0.1 — 2026-11

- First C core: fixed-weight 6-order KT mixture judge (orders 0–5) +
  greedy under-shoot patcher at a C-bit budget; dynamic patch book;
  streaming bit-exact vs batch; parallel batch spans; portable
  snapshots; bit-exactness gate (`tests/gate_bitexact.py`).
