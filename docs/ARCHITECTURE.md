# RateC — architecture

This document is the definitive description of the shipped system as of
v0.8.0. Every design decision listed here is backed by a number in a
committed record (`research/*.json`) or by a gate in `tests/` — no
statement without evidence.

## What RateC is

A byte-level adaptive tokenizer: a 6-order KT mixture (**the judge**)
scores every byte with a surprisal, and a greedy rate-packing rule (**the
patcher**) groups bytes into patches under a C-bit budget. Patches become
tokens; a content-addressed **book** gives multi-byte patches stable ids.
There is no training run, no vocabulary file, no corpus dependency: the
tokenizer adapts to the stream as it reads it, and the adaptation is
deterministic — the same bytes produce the same tokens on any machine.

The LM-facing interface is **factorized**: a token is
`(content_id, length_class)` where content lives in a table of
256 + 9·2^C rows (single bytes are their own raw value; multi-byte
patches index one of 2^C content slots × 9 length classes) and the
length field carries the byte length of the patch. The LM consumes
`ids` and `lens` as separate aligned streams.

## The layers (bottom up)

```
bytes ──judge──> per-byte surprisal ──patcher──> spans (start,end)
        │                                              │
        └── book (content-addressed id table) ──> ids  │
                                                       ▼
                                    packed streams (ids, lens) ──> LM
```

1. **Judge (`Ent` slots + context maps).** A 6-order byte-context model.
   Contexts of order 1..6 are hashed (multiplicative golden hashing) into
   open-addressed maps holding a 64-byte `Ent` slot per context: an
   inline window of 16 recently-seen next-bytes with their counts, and a
   promotion path to full counters when a context exceeds 16 distinct
   continuations. The prediction is a KT-style adaptive mixture over the
   orders; the per-byte output is a float32 surprisal. Judge-only speed:
   **9.6 MB/s** — tiktoken-CLA parity without any vocabulary.

2. **Patcher.** Greedy under-shoot: extend a patch while the running
   bit-cost stays under the C-bit budget; close at the first byte that
   overshoots. C is the only knob. The incremental streaming patcher is
   bit-identical to the reference `patch_from_sur` over the stored
   surprisal array (gated in `tests/test_fuzz.py` P3-1).

3. **Book (intern).** Multi-byte patches are interned into an
   append-only arena; ids are the arena entry index. Lookup is FNV-1a-64
   rolling-hash open addressing in a single 16-byte slot array (hash +
   id in one cache line — worth +12% ids-path MB/s, record: this cycle's
   commits). Single-byte patches skip the book and carry their raw byte
   value, which bounds any token to one of 256 + 2^C ids.
   The rolling hash makes interning O(1) amortized per byte instead of
   per-close re-hashing (measured: ids path 4.1 → 5.5 MB/s cold,
   6.6 → 7.4 MB/s warm; +61% / +172% then +12% on top).

4. **Packed streams.** `ids:int32, lens:int32` aligned arrays, one pair
   per token, one pair per domain/split. Byte conservation (Σ lens ==
   corpus bytes) and id bounds (id < 256 + 9·2^C, or raw bytes for
   single-byte tokens) are gated per C in `tests/test_streams.py`.

5. **LM interface (`ratec_direct`).** A tied embedding/head over the
   256 + 9·2^C rows: the LM's logits ARE the content table, so "tied
   head" is not an ablation here — it is structurally what the
   interface is. An ablation zeroing the length field measured
   **−0.172 BPB** (the length field contributes); per-domain ordering
   zh/code-first. The ladder (`research/twin_ladder.json`) records the
   honest architecture comparison: at 2400 steps the tied interface
   trails the untied factorized arm (`ratec`) by 0.006 BPB and the gap
   inverts past that rung — **3.5270 (tied) vs 3.5214 (factorized) at
   4800** — so the shipped `ratec_direct` is the simplicity winner, not
   the BPB winner at larger budgets; the sweep and the doc say so
   rather than quietly preferring the home arm.
   Frozen trainer protocol: d=96, 3 layers, 4 heads, ctx 256, batch 16,
   lr 3e-4 OneCycle, seed 42, 2400 steps.

## The invariants (what the gates enforce)

- **Bit-exactness.** Streaming == one-shot == reference-patcher, judge
  C-independence, chunked-encode identity, snapshot round-trips
  including mid-patch save/load (the rolling-hash recompute path),
  reset-vs-fresh, prune-arm segmentation identity. 18 gates, all
  deterministic, all seed-fixed.
- **Protocol pre-registration.** Every measurement record carries its
  interpretation block BEFORE the run (`lm_c_sweep.json` extension),
  and `make_tables.py --check` + the drift gate fail the suite if a
  README number or a bound input hash moves without a re-record.
- **Recoverability.** Long jobs checkpoint; commits push before compute;
  records are never hand-edited.
- **Honest speed.** The speed floor gate is deliberately loose (2.5 MB/s
  ids-path) because concurrent load halves observed throughput; the
  true unloaded numbers live in README, sourced from records.

## Choosing C (the one knob)

| use case | C | why |
| --- | --- | --- |
| tiny probe LM (~4M params) | **12** | 3.3804 BPB — best measured; the pre-registered C=14 arm (v0.8) decides whether the optimum climbs further |
| general tokenization / compression | **20** (default) | 3.24 bytes/token over the 4 bake-off domains (bench_c_sweep record); judge bandwidth dominates |
| very long streams, warm caches | 20-24 | fertility keeps improving with C; map grows on demand |

The sweep record compares budgets, not parameter-matched models —
head rows grow as 256 + 9·2^C, and the record states that confound per
row. Full rationale: `research/lm_c_sweep.json` (protocol block),
`docs/PLAN_v0.7.md` P3.

## Defaults (audited v0.8 — API and CLI agree on every value)

| knob | default | justification |
| --- | --- | --- |
| `c` | 20 | the choosing-C table above; 3.24 B/token on mixed corpora |
| `exact` | True | bit-exactness is the contract; the fast lane is opt-in and bounded by fastlane_parity.json (≤1.1e-7 rel. deviation) |
| `store_book` | True | ids consumers are the default audience; spans-only bulk work sets False |
| `pf` | auto (−1) | prefetch depth auto-tunes; pf 0..4 are measured-equal variants (fuzz P3-4: spans and surprisal bitwise-identical) |
| `prebits` | 16 | map grows on demand via core_reserve; initial capacity is output-invisible (verified bit-exact across 16/21) |
| `map_capbits` | 22 | 4.2M slots ≈ judge footprint bound; exhaustion is a RECOVERABLE RatecError (fuzz P3-5) |
| `threads` | 0 (auto) | batch paths scale to cpu_count; 0 = all cores |

## Concurrency model

- Single engine (`RateC`) — one judge, sequential, warm across the
  whole stream. The correct choice for one long document.
- `encode_stream_parallel` — head chunks sequential (exact), tail
  chunks parallel from a snapshot, a documented approximation
  (deterministic, tiling, round-trips).
- `batch_encode` — one cold engine per independent document, threads
  across docs, GIL released; bit-exact vs cold one-shot engines. The
  choice for many small/medium docs (LM data prep).

## What is deliberately NOT here

- No learned merges / vocab files (the judge replaces both).
- No BPE-vs-ratec training-time parity claims: the ladder
  (`research/twin_ladder.json`) compares trained systems end to end.
- No warm-state seeding in `batch_encode`: per-doc independence is the
  contract; the streaming path is the warm path.
