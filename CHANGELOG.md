# Changelog

## Unreleased — twin-study training-scale probe

The singleton-starvation hypothesis (v0.4.1 finding) makes a testable
prediction: more training steps ⇒ more samples per patch type ⇒ the
BPE-vs-RateC gap narrows. Measured (same arms/arch/seed/bytes, 4× steps,
OneCycle recomputed, identical full-coverage eval):

- bpe 3.112 → **2.607** BPB (−0.505, −16.2%)
- ratec_direct 4.796 → **3.818** BPB (−0.978, −20.4%)
- tokenization gap **1.684 → 1.212 BPB (−28.1%)**; RateC improved
  **1.94× more** than BPE for the same 4× compute
- gap narrowed in every domain: en −25.1%, hr −23.6%, zh −35.5%,
  code −31.1% relative

Direction confirmed; trend not established (two points, 1.27M params, gap
still large). `bytes` and factorized-`ratec` arms not re-run at this scale.
Digest: `research/twin_probe_s2400.json`; record: `docs/twin_study.md`.

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
