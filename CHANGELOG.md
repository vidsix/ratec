# Changelog

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
