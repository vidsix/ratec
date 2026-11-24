# v0.7.0 — the scale-ladder release: notes + outcome

Status: **EXECUTED — complete.** Code `91ae140` (CI green), protocol
fix `5f6430b`, plan + docs `bfc8268`/`5936cab`, ladder-complete records
`51a4e19`, speed + gates `fa467bd`/`f9ed436`, final records `03c1e8f`,
tag `v0.7.0`.

## What ships

1. **P1 — the scale ladder** (`research/run_ladder.py` →
   `research/twin_ladder.json`): 4 arms × 4 rungs (600/1200/2400/4800
   steps), the twin-study protocol frozen everywhere else, rung 600
   seeded from the committed records, every higher rung a fresh
   OneCycle-recomputed run (no warm-starting across rungs), the record
   rewritten after every completed run, resumable per run. Both
   interpretations pre-registered in `docs/PLAN_v0.7.md` before any
   rung above 600 was trained.
2. **P2 — frequency-aware book pruning** (opt-in engine feature):
   `RateC(prune=True, prune_after=1<<20, prune_min=2)`. Window counts per
   book entry; at absolute-offset boundaries, entries below `prune_min`
   are dropped and survivors renumber dense; byte segmentation is
   unaffected (spans invariant). Default path byte-identical (v6
   snapshots unchanged, bit-exactness gate green); prune-mode snapshots
   are v7, rejected cleanly by v2–v6 readers. `tests/test_prune.py`
   (25 checks) gates the invariants; the snapshot fuzz gate grew a v7
   section (9 checks; mutant corpus ≥ 1000 incl. the v7 families).
3. **P3 — the LM-optimal C sweep** (`research/run_c_sweep.py` →
   `research/lm_c_sweep.json`): ratec_direct at C ∈ {8, 10, 12}, 2400
   steps, on single-engine streams in the shipped packed-id layout
   (`build_streams.py --c C`; val continues the train engine so ids stay
   consistent — the shipped streams' proven design). A protocol bug in
   the first draft (raw dense ids → the LM table would have grown with
   the book) was caught and fixed BEFORE the sweep ran; every arm shares
   the fixed 256+9·2^C-row table, parameter budgets by design.
4. **P4 — suite**: `tests/test_prune.py` registered as the 8th gate;
   v0.8-wip adds `tests/test_streams.py` (producer protocol gates) and
   `tests/test_speed_floor.py`.
5. **Protocol fix found by the ladder itself**: the v0.6.0 eval-coverage
   assert assumed 1-byte first tokens; RateC val streams can open with a
   multi-byte patch. The assert now checks the exact invariant;
   recorded numbers were never affected.
6. **Speed (PLAN_v0.8 P1, landed early)**: rolling-hash book intern —
   bit-exact by construction, full gate battery green, ids path +61%
   warm / +172% cold.

## The results (final)

**The scale ladder (16 runs, BPB lower = better):**

| steps | bpe | ratec_direct | ratec | bytes | gap bpe−ratec_direct |
|---|---|---|---|---|---|
| 600 | 3.112 | 4.796 | 4.816 | 4.147 | −1.684 |
| 1200 | 2.837 | 4.183 | 4.181 | 3.885 | −1.346 |
| 2400 | 2.607 | 3.818 | 3.777 | 3.675 | −1.211 |
| 4800 | 2.396 | 3.527 | 3.521 | 3.185 | −1.131 |

- The tokenization deficit narrows monotonically with training compute:
  **+0.179 BPB per log2-doubling over 4 rungs** (pre-registered).
- Arm improvement slopes (BPB per doubling): ratec_direct −0.417,
  ratec −0.429, bpe −0.238 → **the ratec arms improve 1.75× faster per
  compute-doubling than bpe**.
- The factorized interface overtakes direct at scale (3.5214 vs 3.5270
  at 4800).

**The C sweep (2400 steps):** C=8 3.5279 (606k params) → C=10 3.4451
(1.27M) → C=12 3.3804 (3.92M) — monotone through the edge; bigger patch
budgets keep helping the LM at this scale (C=14+ queued for v0.8 to find
the interior optimum). Protocol determinism proven: the prune stage's
unpruned arm reproduced the sweep's C=10 number exactly (3.4451) across
independent invocations.

**Book pruning (pre-registered negative):** pruned 3.8138 vs unpruned
3.4451 — pruning COSTS the LM +0.3686 BPB at equal compute; the
starvation-relief hypothesis is refuted at this scale. The feature stays
opt-in (default off) with the negative result documented — the engine
ships on its invariants, not on this number.

**Provenance note:** this release was executed across five sandbox
recycles; every recycle was recovered in minutes because code was
pushed early, all compute artifacts live in a recycle-proof runs/
directory, and every runner is checkpoint-resumable. The rebuilt engine
reproduced the lost session's numbers bit-for-bit (rung 1200: bpe
2.8365, ratec_direct 4.1826 — identical) and the stream manifests
reproduce identical engine stats (C=10 book 145,802; pruned 44,556
after 195,923 removals). Determinism held end to end.

## Verification status

- Local gates: 9 suite gates (incl. test_prune 25/25, snapfuzz 9/9 with
  the v7 families) + 8 producer gates + the speed floor — all green.
- CI green on `91ae140` (the full matrix, first time with the prune gate
  and v7 fuzz); records commits carry the same code.
- `make tables` regenerated and the drift check passes (README tables
  match the records).
