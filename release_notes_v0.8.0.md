# v0.8.0 — the speed-and-standard release: notes

Status: **DRAFT — numbers pending the C=14 record** (training in flight,
checkpoint-resumable; the record + interpretation land in this file when
the arm completes). Everything else in this list is committed and gated.

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
6. **C-sweep extension (pre-registered)**: the C=14 arm registered
   BEFORE its run — interpretations live in the runner and the record's
   extension block; C=12's rows and the frozen protocol untouched.
   RESULT PENDING: C=14 below C=12's 3.3804 → interior optimum inside
   {12,14}; at/above → the direction continues (stated as a direction,
   never extrapolated). Param confound per run (256+9·2^14 = 147,712
   head rows, 14.54M params).

## Provenance

Executed across sandbox recycles on a 3 GB shared box; code pushed
before compute; the trainer checkpoints every 50 steps; the post-train
benchmark re-record runs unattended on the quiet box. The published
token was used once for push and should be rotated after the release.
