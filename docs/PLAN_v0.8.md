# PLAN_v0.8 — speed, trained-right, and robustness

Status: draft (v0.7.0 landing in parallel; every number below is measured
on this 2-core sandbox, 2 MB of the corpus mix, warm book unless stated).

## Why this plan exists

v0.7.0 proved the training-side claims (the ladder trend, the sweep, the
prune measurement) with bit-exact engines and enveloped records. What it
did NOT do: make the tokenizer fast, check whether the LM arms are
*trained right* rather than merely trained identically, and close the
robustness gaps a release surfaces. v0.8 is that release.

## P1 — speed (measured levers, bit-exactness non-negotiable)

Baseline: RateC one-shot **5.5 MB/s** warm (4.0 cold) vs tiktoken
cl100k **9.8 MB/s** on the same box. Attribution: ~95% of wall time is
inside the C engine (`encode_spans`); Python span assembly is noise.

Measured cost structure (knob sweep, 2 MB warm):

| config | MB/s | note |
|---|---|---|
| exact, c=20 (default) | 5.5 | the shipped configuration |
| exact, c=8 | 2.0 | 2.7× slower — smaller C = more closes = more judge work |
| fast lane (exact=False) | 2.2 | slower than exact here — the fast path is not a speed win at these settings |
| pf=0 | 1.5 | 3.7× slower than pf=auto(2) — prefetch is load-bearing |
| store_book=False | 2.7 | the book makes encoding 2× FASTER (patch reuse cuts judge work) |
| batch_spans (8×250 KB, cold/doc) | 2.3 | cold-per-doc tax; GIL is NOT the blocker (released in the worker) |

Levers, in expected-value order:

1. ~~judge memoization~~ **RETRACTED — already built into the engine**,
   twice over: the judge slots carry inline 16-entry windows (one 64-byte
   cache line per hot context = the memo), and the judge's counts update
   every byte, so an external memo would be stale (not bit-exact). The
   lever that DID land instead: **incremental open-patch hashing** —
   book_intern re-ran fnv1a64 over the whole patch content at every
   close; the engine now maintains the rolling hash per extension byte
   (identical value by the fnv recurrence — bit-exact by construction,
   proven by the full gate battery: 9/9 with pinned id/span fixtures,
   snapshot round-trips unchanged). Measured on the 2 MB corpus mix,
   ids path (LM data prep): warm 4.1 → 6.6 MB/s (+61%), cold 1.8 →
   4.9 MB/s (+172%); book-on now matches book-off. Honest cost
   attribution: judge-only 9.6 MB/s ≈ tiktoken 9.8 — the KT judge core
   is at parity; the remaining ids-path gap is patcher bookkeeping, not
   prediction.
2. ~~batch on book-on engines~~ **RETRACTED after controlled
   re-measurement**: the original "book-on 2× faster" was a warm-vs-cold
   artifact (fresh engine per iteration vs a warmed one). Book-off is
   actually FASTER (no obuf/intern work), and the batch worker already
   pre-sizes its maps demand-style. The real lever found instead:
3. **`reserve=` pre-sizing** for known-scale jobs. Cold-start tax is
   26–50% (book arena realloc + judge map growth); a size hint kills
   the growth chain. Composes with `warm_state` (v0.4.1).
4. **Document the truth about `store_book`**: the api docstring's
   "turn off for bulk LM data prep" is a MEMORY note, not a speed note —
   book-on is 2× faster and produces identical spans (gate this
   equality, then recommend book-on everywhere).

Non-goals: SIMD rc paths (the rc inner loop is table-driven and already
~free relative to the judge), threading inside one stream (streaming
semantics make intra-stream parallelism output-changing).

## P2 — trained-right audit (the "right architecture, wrong training" hunt)

The ladder protocol froze ONE recipe across arms — compute-matched and
seed-deterministic, which is what a *comparison* needs. But identical
training is not optimal training; three checks, each cheap and
controlled:

1. **Head-table fairness (CLOSED — verified clean).** Suspected during
   v0.7.0: the sweep's dense-id draft would have given ratec arms
   13–52 M-row tables vs bpe's 1.27 M (dead softmax rows, broken
   compute-matching). Found and fixed BEFORE the sweep ran (packed-id
   producer, commit 5f6430b); all arms now share the exact same fixed
   table. The ladder records themselves were always clean (v_eff 9472
   everywhere, verified from records).
2. **Untied-head tax on the factorized arm — CLOSED, structurally
   impossible to tie.** The head operates on 9472 packed ids while the
   input embedding is over 1280 content rows — there is no shared table,
   so "tie the head" is not an ablation but a different architecture
   (that IS ratec_direct, already on the ladder). The untied head is the
   factorized interface's price, already priced into the measured trend
   (ratec ≤ ratec_direct from 1200 steps on) — no further run needed.
3. **Length-field learning — DONE, positive** (`research/
   ablate_lenfield.py` → `research/ablations.json`): eval-only ablation
   on the ratec@2400 checkpoint (control reproduces the record's 3.7774
   exactly). Zeroing the length embedding costs +0.1718 BPB overall —
   the length field ALREADY contributes materially at probe scale, and
   domain-structured as designed: zh +0.344, code +0.235, en +0.101,
   hr +0.073 (the high-fertility-variance domains lean on it hardest).
   The "dormant component" caveat is refuted; the factorized
   architecture is learning what it was built to learn.

## P3 — robustness (falling-out parts)

1. **Suite the stream producer.** build_streams' packing is now
   protocol-critical (it feeds the sweep and prune records) but has no
   dedicated gate: add tests/test_streams.py — id-bound invariants per C
   (max id < 256+9·2^C), byte conservation (lens sum == corpus bytes),
   1-byte tokens as raw bytes, segmentation identity across prune arms,
   packed-layout round-trip through twin_study's decoder.
2. **Speed regression floor.** After P1 lands: a gate asserting ≥4.5 MB/s
   warm one-shot on the reference corpus (loose floor — catches
   accidental pessimization without flaking on load).
3. **Record-drift hardening.** make_tables --check covers README tables;
   extend the drift test to the runner records (twin_ladder /
   lm_c_sweep / book_pruning): envelope hashes re-verified on every test
   run, not just at generation time.
4. **Recoverability.** This release cycle survived four sandbox recycles;
   the cost was low only because everything was checkpoint-resumable and
   pushed early. Formalize: the release procedure itself gains a
   "push before compute" step (records-so-far commits are cheap and
   recycle-proof).

## Milestones

- v0.7.0 (this cycle): ladder + sweep + prune records, P3 protocol fix
  (packed streams), CI green, tag + release.
- v0.8.0: P1 levers 1–4 with bit-exact gates, P2 ablations 2–3, P3.1–3.4;
  release notes carry before/after MB/s and the ablation numbers.

## Status (updated in-cycle — what actually happened)

- **P1 speed**: landed early. Rolling-hash intern (+61% warm / +172%
  cold ids path) and merged book hash slots (+12% on top; cold 5.5 /
  warm 7.4 MB/s). A fused lockstep emit was attempted and REVERTED —
  the engine's persistent OutVecs diverge in capacity across encode
  modes and the fused check segfaults (caught by the fuzz gate;
  per-vector growth is load-bearing). Judge-only remains 9.6 MB/s
  (tiktoken parity); the ids-path gap to it is now mostly arena writes
  for new book entries.
- **P1 speed, new feature**: `batch_encode` — thread-parallel per-doc
  ids+lens for LM prep, bit-exact vs cold one-shot engines. For
  many-doc corpora this multiplies throughput by cores.
- **P2 ablations**: 2 closed (tied-head is structurally impossible —
  "tied" IS ratec_direct), 3 done positive (length field contributes
  −0.172 BPB at 2400 steps, domain-structured zh/code-first).
- **P3 gates**: test_streams.py (8 producer gates) + test_speed_floor.py
  landed; suite = 18 gates, all green on the new core.
- **C-sweep extension (pre-registered)**: C=14 arm running. The first
  launch OOM-killed (2.4 GB logits tensor at the 147,712-row head);
  the trainer now applies the head per 16-position chunk under gradient
  checkpointing (identical math, one chunk's logits in memory; C<=11
  rows bit-identical on the unchunked path). ~6 h at this head size,
  checkpointed every 50 steps — resumable across turn boundaries.
- **Push state**: the v0.7.0 tag and release are on GitHub; this
  cycle's commits (E1, batch_encode, trainer fix, extension) are local
  until a rotated token lands.
