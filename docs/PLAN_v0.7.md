# PLAN v0.7 — the scale-ladder release

From the post-v0.6.0 review (pushed tip `2242c15`, full 7-gate PASS on the
exact pushed tree). Theme: the repo's one open roadmap item is
**LM-side scale** (README item 12: "LM-side scale (BPB vs training scale)
remains open"), and the s2400 probe's own verdict is *"direction
confirmed, trend not established: two points"*. v0.7 turns that two-point
probe into a measured trend, and lands the engine feature the probe's
hypothesis points at.

## P0 — review findings (already landed / operational)

- **Makefile recipe tabs** — `make test` had been broken since `ed4d7bb`
  (v0.5.0 reconciliation rewrote recipes with 8 spaces; every invocation
  died with `missing separator`). Fixed on the published line
  (`2242c15`), full gate re-verified on that tree. The CI job would have
  caught this at push time — the workflow file is the one commit the
  deploy token couldn't push (no `workflow` scope); CHANGELOG documents
  the withhold, and the next workflow-scoped token lands it as a single
  commit (`4211e34` in the local export line).
- **Review sweep of the pushed tree**: no author-local paths in code or
  docs (the v0.4.1/P2 portability work held: `RATEC_*` env vars are the
  only opt-ins, shipped fixtures are the default gate inputs); no stale
  tolerances (parallel-fertility stays informational); version
  single-sourcing verified live (pyproject == engine == 0.6.0).

## P1 — the LM-side scale ladder (headline)

Goal: one committed record that establishes — or honestly refutes — the
trend the probe suggested, with the same fixed protocol everywhere.

- **Arms (4, all re-run at every rung):** `bpe`, `ratec_direct`,
  `ratec` (factorized book, the shipped architecture), `bytes` (no
  tokenizer control). The s2400 probe re-ran only 2 arms; the factorized
  arm — the one that ships — has never been on a ladder.
- **Rungs:** steps ∈ {600, 1200, 2400, 4800} (log-spaced ×2, matching the
  probe's 4×-improvement-per-4×-compute finding). Everything else frozen:
  d=96, 3 layers, 4 heads, ctx 256, batch 16, OneCycle recomputed per
  rung, seed 42, the same 15,811,609-B train / 659,473-B val bytes,
  full-coverage byte-weighted BPB (the corrected protocol from v0.6.0's
  P0 round).
- **Deliverables:** `research/twin_ladder.json` (provenance envelope:
  ratec_version, git_rev, host, inputs sha256 per the P2 convention);
  a generated README table (gap vs steps, 4 arms) wired into
  `make tables` + the drift test; `docs/twin_study.md` gains a
  "scale trend" section with the fit.
- **The number to publish:** gap-vs-log-steps slope + the
  improvement-slope ratio (RateC ΔBPB per doubling vs BPE's), per domain
  and pooled. Two interpretations pre-registered, so the record can't be
  curve-fit post hoc: if the gap keeps narrowing with an approximately
  constant log-slope, that *is* the "more efficient with scale" claim,
  stated with its measured exponent and range; if it plateaus, the doc
  says where it plateaus. Either way the record is the contribution.
- **Compute budget:** 16 runs (4 arms × 4 rungs) at probe size (~1.3M
  params) — hours on one GPU, resumable per run (the existing
  `twin_study.py` checkpoint/resume machinery carries over).

## P2 — frequency-aware book pruning (the engine feature the hypothesis points at)

The probe's stated mechanism: *singleton starvation* — patch types seen
rarely contribute little to the LM but occupy the fixed C budget.
Pruning = book entries whose recent-window sample count is below a
threshold stop competing for budget.

- **Engine (opt-in, default bit-exact):** `prune=True` (or `prune_after=n`
  bytes) on `Ratec(book=...)`; the default path stays byte-identical
  (the bit-exactness gate must not move). Snapshot: v6 gains an optional
  trailing prune-metadata section — readers of v6 without it load
  unchanged (backward compatible both directions).
- **Invariant tests (P4):** prune-determinism (same stream → same pruned
  book); snapshot round-trip with and without the section; encode
  continues bit-exact across prune boundaries; `reset()` semantics
  unchanged.
- **Measured claims to ship (or drop):** units/byte and LM BPB at equal
  C budget, cold-vs-warm, before/after pruning — one record
  `research/book_pruning.json`, one README row. If the win is < noise,
  the feature stays opt-in and the doc says so.

## P3 — LM-optimal C sweep (cheap, informs every doc)

At probe scale (2400 steps, ratec_direct): C ∈ {8, 12, 16, 20, 28} —
does the C=20 default sit at the LM optimum, or is the tokenizer-quality
optimum (C=33 per the bake-off) different from the LM-quality optimum?
Record `research/lm_c_sweep.json`; the answer goes into `docs/twin_study.md`
and the README's C-choice note.

## P4 — tests that generate information (continued)

- Ladder protocol regression: the ladder record is regenerable from
  sha-bound corpora + committed seeds (extends P2-6's
  `build_streams.py` machinery); drift test covers the new table.
- Pruning invariants (above).
- `make test` gates grow by at most one script; the suite manifest
  keeps green-passed distinguishable from green-skipped.

## P5 — explicitly out of scope (deferred, with reasons)

- **Tokenizer ecosystem bridges (GGUF / llama.cpp / vLLM)** — the README
  roadmap says "once the training side settles"; the ladder is exactly
  that settling. Prep notes only (`docs/container.md` gets the
  ratec-stream-inside-container mapping sketch).
- **Parallelism / SIMD patcher** — measured and declined in v0.4
  (patcher = 5.5% of runtime; the judge's orders 3..5 are fused); the
  revisit conditions in `docs/v04_design.md` still hold.
- **New modality profiles** — registry-as-data makes them cheap, but
  none has a user yet; add when there is one.

## Operational (not code)

1. Land CI: a token with `workflow` scope pushes `4211e34` (rebased
   trivially) — CI then guards exactly the gates this release ran by
   hand. **The current token must be rotated regardless** (it transited
   chat).
2. `docs/PLAN_v0.6.md` → keep verbatim (record of the round);
   `PLAN_v0.7.md` (this file) becomes the Layout pointer.
