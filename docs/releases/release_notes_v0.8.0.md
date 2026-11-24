# v0.8.0 — the speed-and-standard release: notes

Status: **EXECUTED — complete.** All items below are committed, gated,
and the C=14 record is in (`research/lm_c_sweep.json`, extension block
+ row). Executed across five sandbox recycles; every recycle recovered
in minutes via checkpoints and push-before-compute.

## What ships

1. **Speed — merged book hash slots** (on top of the rolling-hash intern
   that opened the cycle): the book's split hash/value probe arrays
   become one 16-byte slot array — one cache line per probe, bit-exact
   by construction. Ids path (LM data prep): **cold 1.8 → 5.5 MB/s
   (+205%), warm 4.1 → 7.4 MB/s (+80%)** across the cycle. A fused
   lockstep emit was attempted and REVERTED: the engine's persistent
   output vectors diverge in capacity across encode modes; the fuzz
   gate caught the segfault, the per-vector growth is documented as
   load-bearing. Honest cost attribution unchanged: the judge alone
   runs 9.6 MB/s vs tiktoken's 9.8 — prediction at parity; the ids-path
   gap is patcher bookkeeping.
2. **Feature — `batch_encode(docs, c, threads)`**: thread-parallel
   per-document `(ids, lens)` for LM data prep, bit-exact vs cold
   one-shot engines, GIL released across docs. Aggregate throughput on
   2 cores exceeds the single-engine warm rate.
3. **Trainer — chunked + checkpointed cross-entropy** for big-vocab
   heads: the logits tensor is batch·ctx·vocab·4B (2.4 GB at C=14's
   147,712-row head — a global OOM on a 3 GB box, root-caused from
   dmesg). TwinLM gains trunk()/forward split; the head applies per
   16-position chunk under gradient checkpointing — identical math,
   one chunk's logits in memory, C≤11 rows bit-identical on the
   unchunked path.
4. **Gates — P3.3 record-drift hardening**: the tables gate re-verifies
   the sha256 of every bound record input present on disk (36 files);
   stale provenance now fails the suite. Speed-floor docstring updated
   to the merged-slot reality (floor stays 2.5 MB/s: concurrent load
   halves observed throughput — measured range under training load
   3.3–9.5 on the floor's own sample).
5. **Docs — `docs/ARCHITECTURE.md`**: the definitive architecture
   document (layers, invariants, concurrency model, defaults audit);
   every number cited from a committed record. README refreshed:
   batch_encode, current ids-path numbers, roadmap.
6. **C-sweep extension (pre-registered) — RESULT**: the C=14 arm was
   registered BEFORE its run (interpretations in the record's
   `extension_v0_8` block, 2026-11-18T17:42:06Z); C=12's rows and the
   frozen protocol untouched. **C=14 lands at 3.3142 BPB — below C=12's
   3.3804 — so the first pre-registered interpretation fires: the
   LM-optimal budget at this scale sits inside {12,14}, with C=14 the
   best measured arm (14.54M params; head rows 256+9·2^14 = 147,712 —
   the param-count confound grows and is reported per run).** The sweep
   stays monotone C=8→14: 3.5279 / 3.4451 / 3.3804 / 3.3142 at
   2.11 / 2.68 / 3.32 / 4.01 bytes per token. Per-domain at C=14:
   en 3.088, hr 3.278, zh 4.251, code 3.127. The run survived four
   sandbox recycles on its checkpoints and reproduced its quickBPB
   trajectory exactly across resumes.

## Finding worth its own paragraph

A long-lived RateC engine pins its judge maps in the process heap and
batch workers spawned afterwards crawl against it — 4x slower, fully
recovered once the engine is freed. The tiktoken-comparison protocol
now frees the single-thread pool engine before the batch arm and the
record was re-taken (batch 2T: ratec 5.51 vs tiktoken 14.0 MB/s);
fresh-process batch — the LM-prep usage — measures ~9 MB/s aggregate.
Allocator hardening is queued for v0.9.

## Provenance

Executed across sandbox recycles on a 3 GB shared box; code pushed
before compute; the trainer checkpoints every 50 steps; the post-train
benchmark re-record runs unattended on the quiet box. The published
token was used once for push and should be rotated after the release.
