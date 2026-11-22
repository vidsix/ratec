# PLAN_v0.9 — hardening, proofs, and the road to v1.0.0 (production at scale)

Status: implemented in v0.9.1 (the hardening scope below shipped;
the v1.0.0 contract and the v0.9.2/1.0 scopes remain the roadmap).
Two grounded refinements recorded against the draft: (1) the error
taxonomy gained an 8th code, STATE — two OOM-adjacent engine-state
sites (poisoned after MemoryError, batch worker allocation failure)
fit no caller-facing code; (2) the "quiet-box" is a protocol, not a
host claim — the sandbox exposes no cpufreq governor, so the quieting
is min-of-round-medians + round-robin interleave, recorded with full
provenance in the envelope. Original draft below, unchanged.

Draft status: written against v0.9.0 (b828df2). Every baseline below is
measured and lives in BENCHMARKS.md / CHANGELOG / the research records;
absolute MB/s are host-day-dependent on this recycled sandbox — the
deltas and orderings are the stable signal. Nothing here is approved
until its gate is defined; every lever ships proof-first where the house
style demands it.

## Why this plan exists

v0.9.0 closed the token-stream cycle: LM-prep ids went from a serial
afterthought (1.8 MB/s cold at the cycle's start) to a multi-core path
that sits within 8% of the spans-only lane (23.93 vs 26.04 MB/s in the
shipped bench run), with the intern pass 2.23–2.62× warm / 1.37–1.43×
cold and its id-equivalence machine-checked (LEAN S5). What v0.9.0 did
NOT do — and what the first stable release must — is earn the right to
be called production-ready:

1. **The speed story is un-anchored**: the reference comparison
   (bench_vs_tiktoken, 0.4–0.65× per-byte) was recorded on shared
   sandbox hosts that drift tens of percent; there are no perf FLOORS
   a downstream user can hold the library to.
2. **The scale ceiling is undocumented in the api**: span offsets are
   uint32 — one engine covers ≤ 4 GiB of lifetime stream — and
   production corpora blow past that. The chunking pattern exists but
   is folklore, not a tested helper.
3. **Portability is a claim, not a build artifact**: snapshots are
   native-endian x86-64; CI has no LEAN job and no aarch64 story;
   `reserve=` pre-sizing exists inside the engine but is not public.
4. **Errors are one class**: `RatecError` carries no codes — a prep
   pipeline cannot distinguish a tiling violation from book exhaustion.
5. **The ecosystem bridges (GGUF / llama.cpp / vLLM) are roadmap
   promises, not code.**

v0.9.1 and v0.9.2 buy those; v1.0.0 freezes them. The v1.0.0 definition
of "stable" is fixed first, so the increments can be gated against it.

## The v1.0.0 definition of stable (the contract this plan builds to)

* **Bit-freeze**: the exact lane's ids/spans/surprisal and the snapshot
  format are frozen within the 1.x major version; a 1.0-trained snapshot
  loads in every 1.x. The fast lane's instability stays documented, not
  frozen.
* **Perf floors**: `ratec.bench()` defines per-lane floors relative to a
  recorded quiet-box envelope (not absolute MB/s); `test_speed_floor.py`
  fails the build when a lane regresses past its floor.
* **Robustness gates**: fuzz (snapshot mutants, stream interleavings,
  decode of arbitrary id arrays), ASan/UBSan CI, a multi-GiB soak with
  flat memory, and a defined error taxonomy every raise site maps to.
* **Proof coverage**: every execution-model claim in the docs is a
  no-`sorry` LEAN theorem compiled in CI (S1–S5 exist; S6–S9 below).
* **Ecosystem**: a frozen snapshot loads into HF `transformers`
  (tokenizer.json export) and llama.cpp (GGUF) and vLLM adapters, each
  with a round-trip gate.
* **Scale**: an 8 GiB prep job completes with byte-identical ids to
  per-chunk engines, using the shipped helper.

## P1 — performance (the quiet-box anchor + the last ids-lane gap)

Baseline (v0.9.0 bench run, 4 MiB mixed stream, threads=2):
encode_bytes 17.78, spans raw 18.01, stream-parallel 24.52, ids lane
23.93, decode_bytes 13423, batch_encode 28.60 MB/s. Known cost split:
the judge core is at tiktoken parity in isolation (9.6 vs 9.8 MB/s,
PLAN_v0.8 attribution); the remaining sequential gap is patcher
bookkeeping + probe memory traffic; the judge's FP chain is ~9% of byte
cost (the AVX2 rejection analysis).

Levers, in expected-value order:

1. **Quiet-box re-record (v0.9.1, first — everything calibrates on
   it).** bench_vs_tiktoken + ratec.bench() on a controlled, non-shared
   host (min-of-N, ABBA-interleaved, medians), producing the reference
   envelope the 1.0 floors are ratios of. Gate: the envelope committed
   as a record with provenance (machine, CPU state, governor, runs),
   and BENCHMARKS.md's tables re-issued from it.
2. **Cold-book parallel intern — proof-first (v0.9.1, the last big
   gap).** The pipelined variant was measured 1.39–1.49× and was
   UNSOUND (pre-probes raced the merge's insert + growth — segfault on
   cold books); the shipped design barriers the phases, which keeps
   warm books fast but leaves cold books paying the ordered
   distinct-insert merge (the 23.93 vs 26.04 gap). The 1.0 design:
   a **read-stable merge** — the pre-probe and the cross-worker
   duplicate resolution read a frozen secondary table while the primary
   grows behind an epoch check; soundness is proved in LEAN (S6,
   below) BEFORE any C lands. Gate: ids bit-exact vs sequential for
   K = 2…16 (existing battery), plus cold-book ids lane ≥ 95% of the
   spans-only lane on the quiet-box envelope. If the proof cannot be
   discharged, the barriered design ships to 1.0 and this is recorded —
   no unsound code in the stable line.
3. **Per-order tables sized to distinct-context counts + per-order
   prefetch schedules (README roadmap item 4, v0.9.2).** The shared
   orders-3..5 map costs up to ~96 B per hot byte and `map_capbits`
   (2^22 slots) is a one-size cap; distinct-context counts let each
   order get a right-sized table (fewer collisions = fewer probes) and
   a per-order prefetch depth beats the global pf knob. Gate: the ABBA
   harness shows ≥ 5% sequential lane improvement with bit-exact
   outputs and flat-or-better memory on the 10.7 MiB corpus; else the
   lever is recorded as measured-and-declined (house style).
4. **Fast-lane parity decision (v0.9.2).** `exact=False` is historically
   +2.5–7.5% on this box (host-dependent) with a documented deviation;
   either (a) the quiet-box shows a real double-digit win worth
   recommending, with its segmentation-difference surface fully
   characterized, or (b) it is parked with an honest "not a speed win"
   note. Gate: a decision record in BENCHMARKS.md; no silent drift.
5. **Public `reserve=` pre-sizing (v0.9.1).** The engine already
   reserves (`eng_head.reserve`); expose it on `RateC`/module batch
   entry points for known-scale jobs (cold-start tax is 26–50% from
   arena/map growth). Gate: a sized cold encode shows measurably fewer
   realloc/growth events (instrumented count) and ≥ 10% cold throughput
   on the quiet-box; docstrings state the store_book truth (book-on is
   FASTER for patch reuse — a memory note, not a speed note).

## P2 — robustness (the production gates)

1. **Span-offset scale story (v0.9.1).** Keep uint32 internally (64-bit
   span records cost cache lines on the hottest path); ship the
   official pattern instead: `ratec.chunked_prep(blob, engine_bytes)`
   — snapshot-continue + offset-rebase chunking, returning per-chunk
   (ids, global_start, global_end) with the engine-lifetime limit
   enforced loudly. Gate: an 8 GiB synthetic prep job completes,
   and its per-chunk ids are byte-identical to the same chunks encoded
   on fresh engines (the determinism property prep pipelines need).
2. **Snapshot format v8 — explicit endianness (v0.9.1).** RCZ2 with a
   declared little-endian payload: same tables, decoded by explicit
   byte order on any host. v7 readers keep working (read path
   back-compat, write path moves). Gate: a golden snapshot round-trips
   bit-exactly through a big-endian emulation (qemu or cross build) in
   CI; the v4/v6/v7 golden battery still passes unchanged.
3. **Error taxonomy (v0.9.1).** `RatecError` gains `.code` (an enum:
   TILING, BOOKLESS, CAP_EXHAUSTED, BUFFER_LIMIT, SNAPSHOT_INVALID,
   SNAPSHOT_VERSION, ARGUMENT) with every raise site mapped; the
   messages stay human. Gate: a test asserts each code at each raise
   site; the fuzz gates assert codes, not string matches.
4. **Fuzz + soak expansion (v0.9.1→v0.9.2).** Existing: snapshot
   mutants ≥ 1000/run, targeted field mutants, v4/v6/v7 goldens. Add:
   stream-interleaving fuzz (feed/flush/encode_stream_parallel on
   shared state), decode fuzz (arbitrary id arrays decode to bytes or
   a coded error — never corrupt output; pairs with S7), and a nightly
   multi-GiB soak (encode + decode + snapshot cycle) asserting flat RSS
   and no fd growth. Gate: green nightly for a week before 1.0.0rc.
5. **CI hardening (v0.9.1).** ASan/UBSan/sdist jobs exist; add the
   LEAN job (compile all of lean/, plus a gate that fails on ACTIVE
   `sorry`/`admit` — matching the word only outside comments), and an
   aarch64 build+test job (cross or hosted runner) behind P2.2's v8.
   Gate: all green on main; the 1.0 tag refuses to cut on a red matrix.

## P3 — ecosystem features (the "feature-full enough for production" pillar)

1. **tokenizer.json export (v0.9.1).** From a frozen snapshot:
   book ids + the 256 byte-fallbacks + special tokens, in HF
   tokenizer.json shape (the RateCTokenizer port already exists; this
   is the interop artifact). Gate: an exported tokenizer round-trips
   arbitrary unicode through the HF shape byte-exactly, and
   `encode_for_lm` lengths still satisfy sum(lens) == len(utf8).
2. **GGUF / llama.cpp bridge (v0.9.2).** RateC has no merge table —
   the GGUF tokenizer segment carries the book as vocab with
   (id, byte_length) pairs; llama.cpp consumes it through a small
   patch/PR. Gate: llama.cpp loads the exported GGUF and produces
   RateC-identical ids on the golden battery subset that llama's
   vocab model can express (documented subset; the tiling is the
   decoder's, so byte-lengths tile exactly).
3. **vLLM adapter note + validation (v0.9.2).** The (id, byte_length)
   factorized contract needs a decode-side hook; ship the adapter
   recipe and a validation script against a serving-config fixture.
   Gate: documented recipe + a passing validation script in the repo;
   no vendored vLLM fork.
4. **Prep-pipeline cookbook (v1.0.0rc).** The production shapes, with
   numbers from the quiet-box envelope: many-small-docs (batch_encode),
   one-long-stream (encode_stream_parallel + reserve), mixed shards
   (chunked_prep), warm-vs-cold expectations. Gate: every recipe in
   the cookbook runs as a marked test against the committed fixtures.

## P4 — the LEAN layer for 1.0 (proof coverage S6–S9)

House rule unchanged: the theorem lands before the code it licenses.

* **S6 — the read-stable merge protocol (before P1.2's C).** In
  `RatecIntern.lean`: concurrent readers of a frozen table + one
  growing primary + ordered merge assign the sequential first-seen
  ids for any interleaving that respects the epoch check. Gate: no
  `sorry`, compiled in CI, and the C implementation's comments cite
  the theorem name it implements.
* **S7 — decode completeness (v0.9.1).** In `RateC.lean`: every id
  resolves to either its exact byte sequence, a defined raw-byte slab
  value, or the documented error — no third outcome (the property the
  decode fuzz asserts in C). Gate: theorem + the fuzz cross-check.
* **S8 — servo spec (v0.9.2).** `solve_c`'s contract as a theorem:
  the returned C reproduces the target patch count on the measured
  corpus within the documented tolerance (monotonicity already
  externally replicated). Gate: theorem + the existing servo tests
  become its executable form.
* **S9 — snapshot v8 totality (v0.9.1, after P2.2).** save/load is a
  total function on valid states and load rejects everything else with
  a coded error (mirrors the fuzz gate). Gate: theorem + the mutant
  battery enumerating the rejection paths.

## P5 — release engineering for 1.0

* **API freeze audit (v1.0.0rc).** Every public name, its docstring,
  its version origin, and its freeze class (frozen ids / frozen shape /
  may move) in one generated table; the drift test gains an
  api-surface section.
* **Wheels + aarch64 (v1.0.0rc).** sdist exists; add manylinux wheels
  for x86-64 + aarch64 built from the CI matrix. Gate: pip install
  from wheel passes the full suite on both arches.
* **Semver + stability policy doc (v1.0.0rc).** The bit-freeze and
  deprecation policy, written into docs/ and the release notes.
* **The 1.0 record.** bench() envelope, floors, soak results, proof
  inventory (S1–S9), ecosystem gate results — one provenance-enveloped
  record like the rest.

## Milestones

| release | scope (from above) | headline gate |
|---|---|---|
| **v0.9.1 — the hardening release** | P1.1 quiet-box, P1.2 cold-book intern (proof-first), P1.5 reserve=, P2.1 chunked_prep, P2.2 snapshot v8, P2.3 error taxonomy, P2.5 CI (LEAN + aarch64), P3.1 tokenizer.json, P4 S6+S7 (+S9 if v8 lands) | 8 GiB prep byte-identical; ids lane ≥ 95% of spans on the quiet box; all fuzz + proof gates green in CI |
| **v0.9.2 — the production-shape release** | P1.3 per-order tables, P1.4 fast-lane decision, P2.4 fuzz/soak completion, P3.2 GGUF, P3.3 vLLM validation, P4 S8 | nightly soak green ≥ 1 week; ≥ 5% sequential lane win recorded or declined; GGUF loads in llama.cpp |
| **v1.0.0rc** | P3.4 cookbook, P5 freeze audit + wheels + policy doc + the 1.0 record | every table in the record reproducible from committed artifacts |
| **v1.0.0** | the freeze | the stable contract above, all gates green |

## Non-goals (scope control for the stable line)

* No new snapshot semantics beyond v8's endianness; no format v9 in 1.x.
* No LM-quality claims move in the README — the twin/ladder records stay
  research (BPB vs training scale remains open; C=14 stands at probe
  scale).
* No SIMD judge rewrite: the FP chain is ~9% of byte cost and the loop
  is probe/memory-bound — revisit only if the quiet-box attribution
  changes the split.
* No distributed-prep framework: determinism per shard is already
  guaranteed (batch_encode ids are per-doc); orchestration is the
  user's, the cookbook documents the pattern.
* No BPE-replacement benchmarks: the code-first no-SOTA-claims rule
  stands in 1.0.
