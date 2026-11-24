# v0.9.2 — the production-shape release

The v1.0 levers get their measured verdicts — including one honestly
DECLINED — and the ecosystem bridges gain their documented subsets.
Everything below is in a committed record; nothing is aspirational.

## Two verdicts from the 1.0 gate list

- **Per-order tables sized to distinct-context counts: DECLINED, with
  measurements.** The sized design already exists where it can pay:
  orders 3–5 live in the open-addressing map (1,037,058 entries at
  capbits 22 for 1 MB of mixed text). Orders 1–2 are dense DIRECT
  arrays (cnt1/tot1/tot2, 768 KB total — L2-resident, zero
  indirection); a same-model sized variant must add a hash→slot
  indirection to the hottest per-byte loop, and shrinking the slice
  widths changes the ids — forbidden by the 1.0 bit-freeze. The 64 MB
  order-2 region is virtual and tot2-guarded: 9,226/65,536 rows touched
  after 1 MB of real text — resident cost already tracks distinct
  contexts. (`research/perorder_tables.json`; the occupancy comes from
  an independent v8-snapshot walk with an exact-EOF assert.)
- **The fast-lane parity decision: KEEP.** Measured on this build:
  surprisal deviation ≤ 1.19e-07 (bound 5e-07), segmentation IDENTICAL
  to the exact lane on all four shipped domains, speed delta
  −1.1%…+4.0%. The lane's only promise is STRUCTURAL and
  machine-checked for any surprisal function (LEAN T2/S5/S6); it stays
  formally outside the 1.0 bit-freeze, and a future parity failure
  flips the decision to REMOVE. (`research/fastlane_decision.json`.)

## The nightly soak (the 1.0 robustness clock starts)

`bench/soak_nightly.py`: continuous mixed operations — parallel
spans+ids at rotating thread counts, decode round-trips on both paths,
intern thread-independence, snapshot-continuation bit-exactness, batch
spot-checks — over seed-varied 6 MiB corpora, RSS sampled per round
with a flat-RSS verdict. First record: **198 rounds, 0 failures, flat
220.4 MiB peak — PASS**. The `nightly.yml` workflow runs the 60-minute
soak plus fresh fuzz batteries every night; the "nightly green ≥ 1 week
before rc" clock starts at this release.

## The ecosystem bridges, documented subsets

- **`export_gguf`**: a valid GGUF v3 metadata container (zero tensors,
  typed KVs) carrying the book as `tokenizer.ggml.tokens/scores/
  token_type` plus the full v8 snapshot as `ratec.snapshot.v8`
  (sha256-bound). The subset: any GGUF parser can READ the vocab table;
  llama.cpp must NOT re-tokenize with it — the vocab table alone cannot
  reproduce ratec's segmentation. Gate: an independent pure-Python GGUF
  parser walks the container to exact EOF, asserts vocab equality, and
  reloads the embedded snapshot bit-exactly (continuation identity).
- **The vLLM validation recipe** (`docs/vllm_recipe.md`): the contract
  is "encoding is ratec's, decoding is anyone's" — vLLM's tokenizer
  layer decodes ratec ids byte-exactly through the exported
  tokenizer.json; server-side encoding is forbidden (documented) —
  pre-encode with ratec, feed `prompt_token_ids`.

## S8 — the servo spec, machine-checked

`lean/RatecServo.lean`: `solve_c`'s contract as a theorem on the grid
model with a monotone nonincreasing patch count — S8a totality (always
a bracket inside [lo, hi]), S8b the crossing-bracket invariant (the hi
side stays ≤ target at every bracket; the lo side > target whenever
non-degenerate — the degenerate bracket IS the exact-hit edge), S8c
termination (each probe halves the width; fuel ≥ width converges to a
bracket of width ≤ 1 — the tightest grid bracket of the crossing), S8d
the achieved count is exact. The existing servo tests are its
executable form. **Seven LEAN files compile no-sorry** on 4.36.0.

## Gates

make test green (suite + bit-exactness + fuzz + snapfuzz + tables),
pytest green, `make tables --check` PASS on README and BENCHMARKS, all
seven LEAN files no-sorry, the soak PASS at 198 rounds.
