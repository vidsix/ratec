# RateC v0.6.0 plan — the information release

Status: drafted 2026-11-08 from a full-repo review (engine, Python surface,
tests, benches, research records, packaging, CI) after the v0.5.0
reconciliation release. Every claim below cites `file:line` in the v0.5.0
tree. Sequencing rule that does not change: **the default encode path stays
bit-exact** — anything touching it ships with gate + goldens, or it does not
ship.

Theme for v0.6.0, per the project direction: **turn the repo into an
information generator** — every published number regenerable from a
committed record, every hard claim guarded by a test, and the open LM-side
scale lever actually measured. First, though, three published claims need
repair (P0); the repo's honest-caveat culture is worth more than any single
number.

---

## 1. Review findings

### P0 — correctness / honesty of published numbers (fix first)

- **P0-1 · bench_scale protocol flaw (HIGH — affects the v0.5.0 headline).**
  `bench/bench_scale.py:125-127` warms ONE engine cumulatively
  (`TRAIN_POOL[:s]` re-fed at every scale, so prefixes are seen multiple
  times) and then runs `encode_to_spans(PROBE)` on that same engine at
  every scale — the probe's 128 KB is ingested into judge **and book**
  (`store_book=True`) up to 8 times before its last measurement. The
  probe is therefore not held out, and small-scale rows are flattered the
  most (at scale 32 KB the engine has already absorbed ~1 MB of probe
  re-ingestion). The record's `protocol.adaptation`
  (`research/scale_mix15.8MB.json`) also misdescribes the code. The −52%
  saturated number is plausibly close to truth (prefix dominates at 8 MB);
  the small-scale rows are not.
  **Fix:** keep one continuing engine, but take a `save_state()` after
  warming on `pool[:s]` and score the probe on a **disposable engine
  loaded from that snapshot** (also exercises the snapshot path every
  run); never feed the probe into the warming engine; record
  `bytes_seen == s` per row; trim the probe to a UTF-8 boundary once
  (the unguarded `PROBE.decode("utf-8")` at `bench_scale.py:140` can kill
  a run after measurement but before the JSON dump); add
  `ratec_version`, `tiktoken_version`, corpus sha256 to the record.
  Then **regenerate** `scale_mix15.8MB.json` and the README table with a
  note about the protocol correction.
- **P0-2 · twin "coverage 100%" claim is false (MED).**
  `research/twin_study.py` (`eval_full`) never scores each domain's first
  token; records show 659,468–659,473 covered of 659,469–659,473, yet
  `twin_summary.json` and `research/README.md` claim "coverage asserted
  100.00%" — and there is no assert. Fix: assert
  `covered >= val_bytes - domains` in code; reword to "every val byte
  except each domain's first token". Also document the implicit
  patch-length cap: `V = 256 + 9·2^10` (`twin_study.py:50`) caps
  `len_idx <= 9`, i.e. patches ≤ 9 bytes at C=10 — a library-patcher
  divergence that must be stated and asserted (`max(val_ids) < V`).
- **P0-3 · probe digest + README drift.** `docs/twin_study.md:90,98` and
  the CHANGELOG cite `research/twin_probe_s2400.json` (exists in the repo
  tree — kept through the reconciliation). But the README fertility
  sentence "26–50% fewer LM positions" (`README.md:57`) contradicts the
  same document's table (chinese −60%, `README.md:249`). Fix: make the
  sentence table-derived ("26–60%") — and let P2-2's generator make such
  drift impossible.
- **P0-4 · bench_vs_tiktoken record fields are false (MED).**
  `bench/bench_vs_tiktoken.py:194-199` writes `"pf": 3` (arms run `pf=2`,
  `:60,:85,:97,:112`) and `reps {25,31}` (loop is `range(21)`, `:67`);
  neither library version is recorded; the batch arms are timed
  back-to-back (`:128-151`) — the exact drift bias the docstring warns
  about (`:53-56`). Fix: emit actuals from running code, record both
  versions, round-robin the batch arms. Also delete the dead
  `if False` blocks (`:87-90,:109`) and wrap the hardcoded font path
  (`:229`, `bench_c_sweep.py:326`) in try/except.
- **P0-5 · "informational" fertility still fails the suite (MED).**
  `tests/test_ratec.py:259-263` says informational but appends to
  `FAILURES` → exit 1; a foreign box already measured ×1.105. Fix:
  print-only (or a 1.5 sanity guard), keeping tiling + determinism as the
  hard invariants.
- **P0-6 · API/CLI robustness quickies (MED).**
  `python -m ratec surprise FILE --patch` crashes (`cli.py:87` reads
  `a.c`; the subparser at `:32-35` defines no `--c`) — probe-verified;
  also make `--c` `type=float` everywhere so fractional C is reachable
  from the CLI (`cli.py:24,40,44`). `patch_from_sur`
  (`ratec/api.py:285-287`) silently reinterprets float64 input as float32
  bytes (probe-verified: garbage starts) — require float32/1-D, accept
  float64 by casting, else raise. Fix the `pf` docstring drift
  (`api.py:44` says "Default 3"; constructor default is auto/−1,
  PF_DEFAULT=2).

### P1 — snapshot robustness (load_state is the attack surface)

`load_state` accepts arbitrary bytes and currently validates almost
nothing. Probed exploits from the review:

- **P1-1 (HIGH)** `Ent.distinct` accepted in 17..254 → OOB write up to
  239 bytes past a 64-byte slot (`_rcengine.c:937-941` → `win_scan:167`,
  cold-byte branch `:453-454`).
- **P1-2 (HIGH)** map rebuild probes until an empty slot with nothing
  tying `cnt` to capacity (`:943,:958`; only `capb > 26` checked at
  `:934`) → a crafted snapshot spins forever **with the GIL held**
  (`rc_load_state:1530-1544`) — a 1 KB file freezes the interpreter.
- **P1-3 (MED)** book entries unvalidated: file-supplied `off`/`ln` not
  checked against `alen` (`:996-1008`) → OOB reads in `fnv1a64` and in
  `decode_bytes`' memcpy (`:1289-1291`); `alen==0 && bn>0` → NULL deref.
- **P1-4 (MED)** snapshot RLE counter is u8 and wraps at 256 nonzero
  followers (`save_core:835-838`) → a full row is silently dropped,
  save→load not bit-exact on binary-ish corpora.
- **P1-5 (MED)** book hash rebuild caps at 2^26 slots (`:970`) while
  `book_grow_hash` grows unbounded (`:210`) → legitimate >2^26-entry
  books hang on load. Also `book_intern` truncates arena offsets to u32
  (`:257-258`) — guard at the documented 4 GiB limit.
- **P1-6 (LOW)** `map_capbits` is not serialized (`load_core:887`
  hardcodes the default) — a tuned engine restores as default and can
  hit exhaustion later; NaN/absurd `cs`/`lim`/carry accepted from
  snapshots (`:874-896`); `load_core` leaks dirty bitmaps on re-alloc
  (`:935,:952`); `reset()` doesn't clear `book.err` (`core_reset_keep`,
  `:372-418`) so the advertised "reset() recovers" is false after a book
  OOM; `Engine_surprise` maps the recoverable −3 exhaustion to
  `PyErr_NoMemory` (`:1184`); `rc_batch` error paths leak job outputs
  and skip `PyBuffer_Release` (`:1476-1492`); little-endian is assumed
  silently (SWAR `:152-169`, snapshot memcpy) — add a compile-time
  `#error` on BE or document.

**Plan:** one hardened load path — validate `distinct ∈ {PROMOTED} ∪
[0..WIN]` and `tot >= distinct` per entry; `cnt <= (1<<capb)·3/4` before
inserting (tmap + cmap + book hash); every `(off, ln)` pair against
`alen`; `bn==0 ⇒ alen==0`; reject non-finite C/carry/limits. Snapshot
format **v6** (write v6, read v2–v6): u16 RLE counter (P1-4), serialize
`map_capbits` (P1-6), keep v5 byte-layout otherwise. Ship a
**snapshot-fuzz test** (bit-flips/truncations/field-corruptions of the
existing fixtures — every mutant must raise `RatecError`, never crash or
hang) and make `_maybe_decompress` raise `RatecError` on bad zlib
(`api.py:268-271`). Add v6 continuation goldens to the fixtures so the
format bump is gate-verified like v4→v5.

### P2 — information infrastructure ("so we can put more information in it")

- **P2-1 · Record convention.** Every committed record JSON gains:
  `ratec_version`, `deps` {numpy, tiktoken, torch, transformers,
  tiktoken-encoding files}, `git_rev`, `host` {cpu, cores},
  `inputs` {path: sha256}, `protocol_actuals` (what actually ran, not
  what the docstring says). Producer helper `research/record_util.py` so
  every script emits this in three lines.
- **P2-2 · README tables from records.** `make tables` regenerates the
  fertility/speed/carry/profile/scale/twin tables in README.md from
  `research/*.json`; a test asserts README == records (the drift guard
  that would have caught P0-3 and the missing bench records). Numbers
  that exist only in prose today and must become records: fertility/speed
  table (only in uncommitted `results/bench.json`), carry −15.1%/−23.2%,
  profile wins −37.1%/−9.5%/−9.6%, v1→v2 9.0→11.1 MB/s.
- **P2-3 · Commit the bench records.** `bench_vs_tiktoken.py` /
  `bench_c_sweep.py` write to unversioned `results/` — copy final
  records into `research/` (small JSONs) at release time; add
  `make records`.
- **P2-4 · Twin-stream producer + corpus binding.** Nothing in the repo
  produces `runs/{bpe_mix,ratec_main}_*_{train,val}_{ids,lens}.npy`;
  the corpora behind every twin number (`data/_train_mix.txt`,
  `data/{d}_val.txt`) have no sha256 anywhere. Ship
  `research/build_streams.py` (BPE-9472 arm via tiktoken trained-free
  definition, RateC arm via the library, length-cap ≤ 9 documented) and
  bind all inputs by sha256 into the records.
- **P2-5 · Pin bench deps** in pyproject extras (`tiktoken>=x,<y` as
  measured) and record exact versions at run time (P2-1).
- **P2-6 · Env-var hygiene.** `RATEC_RUNS`/`RATEC_TWIN_DATA` default
  CWD-relative (`twin_study.py:47-48`) → default to
  `os.path.dirname(__file__)/../{runs,data}`; document
  `RATEC_SCALE_PROBE` (`bench_scale.py:43`) in the README env list;
  `RATEC_FIXTURES` override must still find `fixtures_meta.json`
  (tests read `DATA/../fixtures_meta.json`) — search upward or document.

### P3 — test suite expansion (tests that generate information)

- **P3-1 · Seeded property/fuzz suite** (`tests/test_fuzz.py`, no new
  deps): N≥200 seeded streams (iid uniform, low-entropy repeats, UTF-8
  text, all-zero, 0xFF runs, lengths 0..100k) × C ∈ {1..64,
  fractional 1.5/17.5/63.5}: tiling, round-trip, stream parity at random
  chunk boundaries, `patch_from_sur(surprise(x)) == engine starts`,
  `batch_spans == serial`, snapshot-continuation equality.
- **P3-2 · Judge C-independence, directly:** `surprise(x)` bitwise-equal
  across c ∈ {1,10,20,33,64} on identical state — the claim underneath
  `patch_from_sur`, both servos and the C-sweep G3 check.
- **P3-3 · Fast-lane parity:** exact vs `exact=False` patch-start
  equality on all fixtures + fuzz corpus, and the measured max relative
  deviation vs the claimed 1.1e-7 (README:78) — recorded in the record.
- **P3-4 · pf equality:** pf=0..4 produce identical spans/sur
  (compile-time fan `:705-715`; only pf=2 tested today).
- **P3-5 · Snapshot negative tests** (see P1): every corrupt-mutant
  class → `RatecError`; reset-vs-fresh bit-exactness on spans+sur;
  book-mode snapshot after `feed()` (obuf path, only store_book=False
  is tested today); map-exhaustion (−3) via tiny `map_capbits`;
  boundary C (64.0 accepted, 64.0001 rejected, 0.5, 1.5).
- **P3-6 · Untested-API sweep:** `solve_c`/`solve_c_for_book` bracket
  edges (`api.py:316-319,:374-375,:393`), `encode(c=...)` rejection,
  `warm_state` as path, `batch_spans(fast=True/carry=True)`,
  `encode_stream_parallel` carry + `head_chunks>1` + `n<=chunk`,
  `save_state(path=...)`, carry+reset, CLI smoke (`encode/decode/
  surprise/state` incl. fractional `--c`), `test_suite.py` timeouts,
  `make test` += `test_smollm2_port.py` (it now skips cleanly — make
  `make test` and pytest agree).
- **P3-7 · Test manifest.** Every suite run writes a machine-readable
  manifest (pass/skip per file + counts) so a green-skipped suite is
  distinguishable from a green-passed one — feeds the release checklist.

### P4 — experiments (the big numbers v0.6 exists to produce)

- **E0 · Regenerate the scale record** under the fixed P0-1 protocol
  (same 15.8 MB mix) — replaces the v0.5.0 table; expect small-scale
  gains to shrink, saturation to hold; publish both old/new in the
  v0.6.0 notes as the protocol correction.
- **E1 · Twin 4-arm matrix at 2400 steps × 3 seeds (42/43/44):** bpe /
  ratec (factorized) / ratec_direct / bytes. The 4× probe showed the gap
  narrowing −28.1% (1.684 → 1.212 BPB); this establishes the trend with
  error bars. Harness already supports `--tag`/resume; run in
  background, records land in `research/twin_s2400_seed*.json`.
- **E2 · Twin scale ladder (the open lever):** corpus sizes {0.5, 2, 8,
  32} MB × steps {600, 2400, 9600} × arms {bpe, ratec_direct} — tests
  "the tokenization gap shrinks with scale" as a curve, not a sentence.
  Long-running; lands in v0.6.x records + README.
- **E3 · LM-optimal C sweep:** twin arms at C ∈ {8, 10, 12, 14, 16} at
  600 steps — is C=10 right for LM quality, or just for fertility?
- **E4 · Frequency-aware book pruning probe:** the v0.4.1 finding says
  long-tail patch-type frequency starves tiny embeddings; prototype a
  pruning/frequency-reordering pass over the book and measure BPB at
  1.3M params before committing to it.
- **E5 · Per-C goldens for one domain:** ship tiny per-C reference
  arrays (C ∈ {8,12,16,24,33}) so `bench_c_sweep`'s "golden-verified per
  C" is externally true at more than one C.

### P5 — packaging / portability

- **P5-1 · `-march=native` out of the default build** (`setup.py:9`,
  `conftest.py:28`): baseline `-march=x86-64-v2` (or none), opt-in
  `RATEC_ARCH=native`; wheels stop being SIGILL-bound, builds become
  reproducible; gate re-run to confirm bit-exactness holds (it is
  enforced by `-ffp-contract=off -fno-fast-math`, not ISA).
- **P5-2 · sdist completeness:** build an sdist today and the gate
  cannot run from it — `tests/data/`, `gate_bitexact.py`, `conftest.py`,
  `Makefile`, `CHANGELOG.md` missing; a stale in-place `.so` leaks into
  it. Add `MANIFEST.in`, exclude `*.so`, add an sdist-install smoke test
  to CI.
- **P5-3 · CI matrix (lands with the workflow file):** ubuntu
  {3.9, 3.12} + macOS {3.12} (P3 bit-exactness is glibc-`log2`-bound —
  macOS is the cheapest probe of that), an ASAN/UBSAN job fed by the
  P1 fuzz corpus, and a `transformers`-installed job so `test_smollm2_port`
  actually runs somewhere instead of skipping. Declare Windows and
  big-endian unsupported (classifiers + README), matching P1-6.
- **P5-4 · Version single-sourcing:** `RC_VERSION` (`_rcengine.c:53`)
  vs `pyproject.toml:7` — make pyproject the source, inject into the
  build; add `[project.urls]` + license classifier.

---

## 2. Sequencing

| step | contents | exit criteria |
|---|---|---|
| 1 | P0-1..P0-6 + E0 rerun + record regeneration | suite green; scale record + README table regenerated under the fixed protocol with a correction note |
| 2 | P1 hardened load + v6 format + snapshot fuzz + goldens | fuzz corpus (≥1000 mutants) raises cleanly, never crashes/hangs; v4/v5/v6 continuation goldens green |
| 3 | P2 infrastructure (convention, records, `make tables`, drift test, stream producer) | every README table regenerates from committed records; drift test fails on hand-edited numbers |
| 4 | P3 test expansion (fuzz, C-independence, fast-lane, pf, negative, API sweep, manifest) | listed paths covered; `make test` and pytest agree |
| 5 | P5 packaging (arch, MANIFEST, version) | sdist runs the gate; baseline-arch build passes the gate |
| 6 | CI workflow file lands (needs a `workflow`-scoped token — staged locally, validated) | first green Actions run incl. macOS + ASAN |
| 7 | **v0.6.0 release** | all of the above |
| 8 | E1 (2400×3 seeds), then E2/E3/E4 ladders in v0.6.x | records committed; README updated from records |

## 3. Non-goals (unchanged)

No SOTA claims; no LM-quality claims without the twin harness; no engine
bit-path change without gate + new goldens; fast lane stays opt-in and
bit-incomparable; Windows/big-endian stay unsupported (documented);
no new runtime dependencies.

## 4. What the review says is already right (do not touch)

The single-hot-loop engine core with commit-at-done state; poisoned-engine
discipline; page-granular reset via dirty bitmaps; sha256-bound fixtures +
v4 continuation goldens; the bit-exactness gate; interleaved round-robin
bench timing with medians and honest `golden_mode` degradation; the carry
patcher's Python-golden differential fuzz (the model for P3-1); the twin
harness's byte-budget tripwires, full-coverage eval and checkpoint/resume;
the honest-caveat culture — which is exactly why P0 gets fixed before
anything else.
