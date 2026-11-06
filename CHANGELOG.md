# Changelog

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
