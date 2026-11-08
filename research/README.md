# Research records

Result records and the harness for the LM-side questions the tokenizer
bench alone cannot answer. Everything here is small and committed; the
large artifacts (tokenized `.npy` streams, `.pt` checkpoints) stay outside
the repo — point `RATEC_RUNS` / `RATEC_TWIN_DATA` at your own copies to
reproduce.

## Twin study — `twin_study.py`, `twin_reeval.py`, `twin_*.json`

The standing rent since the v0.3 LM track: the same transformer trained on
the **same bytes** under three tokenizations, compute-matched, seed 42.
Full protocol is in the `twin_study.py` docstring; the committed
`twin_summary.json` is the record of the run summarized below.

**Setup.** d=96, 3 layers, 4 heads, ctx 256, batch 16, 600 steps, AdamW
(3e-4, OneCycle), 2 torch threads. Identical 15,811,609-B train corpus and
659,473-B held-out val stream for every arm. Eval is full-coverage
byte-weighted BPB on the identical val bytes (coverage asserted 100.00%
per arm — this fixed two earlier byte-weighting bugs, see
`twin_reeval.py`).

**Arms.** `bpe` (tied 9472-row embedding/head, 1.27M params) /
`ratec_direct` (same shape, tied, on RateC C=10 patch ids — isolates the
interface) / `ratec` (factorized MegaByte/BLT-style `(id, byte_len)`
input, untied head, 1.41M params) / `bytes` (256-row tied, the raw-byte
floor).

**Result (BPB over the identical val bytes):**

| arm | params | BPB | interface |
|---|---|---|---|
| bpe | 1,269,600 | **3.112** | tied, 9472 rows |
| bytes | 384,864 | 4.147 | tied, 256 rows |
| ratec_direct | 1,269,600 | 4.796 | tied, on the same C=10 ids |
| ratec | 1,405,248 | 4.816 | factorized input + untied head |

**Reading.**
- The factorized interface is free: 0.02 BPB between `ratec` and
  `ratec_direct` on identical ids. The (id, byte_len) contract costs
  nothing measurable at this scale.
- The tokenization gap is the whole story: −1.7 BPB. The committed
  hypothesis is long-tail patch-type frequency starving tiny embeddings —
  C=10 book singletons were ~30–65% in the early bake-off, BPE tokens
  average ~528 train samples/type vs RateC ~267.
- **Caveat, stated plainly:** 600 steps / 1.3M params is tiny. This study
  quantifies the starting deficit honestly; it does not bound the method.
  Patch-based interfaces (MegaByte, BLT) close such gaps with scale and
  local models.

**Next levers** (in expected-value order): frequency-aware book pruning,
LM-optimal C sweep (the tokenizer-optimal C need not be the LM-optimal
one), scale. Checkpoint files (`runs/twin_{kind}.pt`) are deterministic
seed-42 weights — re-eval is exact via `twin_reeval.py`.
