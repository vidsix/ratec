# Changelog

## v0.6.0 — unreleased (the information release, step 1: honest numbers)

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
