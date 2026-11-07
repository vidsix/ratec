# The twin study — same LM, same bytes, different tokenization

*Status: first result (v0.4.1, 600-step compute-matched run). This is the
standing experiment that separates "excellent library" from "important
result": does a RateC-tokenized LM beat a BPE-tokenized LM trained on the
same bytes? The honest answer at 1.3M params today is **no — and the numbers
below say exactly where the loss comes from.***

## Protocol

- **Identical transformer** for every arm: `d=96, layers=3, heads=4,
  ctx=256, batch=16, AdamW(3e-4, β=(0.9,0.95), wd=0.1), OneCycle, seed 42,
  2 torch threads`.
- **Identical byte budget** for every arm: the same 15,811,609-byte mixed
  train corpus (en/hr/zh/code) and the same 659,473-byte held-out val
  stream. No arm sees more or fewer bytes.
- **Identical eval** for every arm: full-coverage byte-weighted BPB — every
  val byte is a prediction target exactly once (the previous harness
  subsampled per arm and scored different byte subsets; that parity gap is
  fixed and coverage is asserted).
- **Compute-matched**: same tokens/step × steps, so the same FLOPs. Arms
  differ in bytes-per-token, so identical FLOPs buy each arm a different
  amount of unique-byte coverage. Params and train-FLOPs are reported per
  arm; BPB is the tokenization-agnostic metric.

## Arms

| arm | interface | params | val bytes/token |
|---|---|---|---|
| `bpe` | tied emb/head over 9,472 rows | 1,269,600 | 3.0395 |
| `bytes` | tied emb/head over 256 rows (raw-byte floor) | 384,864 | 1.0 |
| `ratec_direct` | tied emb/head over 9,472 rows on RateC C=10 patch ids | 1,269,600 | 1.3967 |
| `ratec` | factorized input (256 + 2^10 content rows, 12 length rows @32, in_proj) + untied head over 9,472 | 1,405,248 | 1.3967 |

`ratec` and `ratec_direct` train on the **same** C=10 patch ids — the pair
isolates the interface cost from the tokenization cost.

## Results (600 steps, seed 42)

| arm | BPB overall | en | hr | zh | code | train FLOPs (est) |
|---|---|---|---|---|---|---|
| **bpe** | **3.112** | 2.734 | 3.219 | 3.335 | 3.487 | 18.7 T |
| bytes | 4.147 | 3.703 | 4.055 | 5.036 | 4.348 | 5.7 T |
| ratec_direct | 4.796 | 4.562 | 4.868 | 5.286 | 4.820 | 18.7 T |
| ratec (factorized) | 4.816 | 4.545 | 4.910 | 5.316 | 4.875 | 20.7 T |

## Findings

1. **The interface is free.** Factorized (4.816) vs direct tied (4.796) on
   the same C=10 patch ids: Δ ≈ 0.02 BPB. The MegaByte/BLT-style
   `(id, byte_len)` factorized input costs nothing measurable at this
   scale.
2. **Tokenization is the whole story.** BPE 3.112 vs RateC 4.80–4.82 vs
   raw bytes 4.147 BPB, compute-matched. At this scale raw bytes beat
   RateC patches in every domain, despite 102 fewer bytes of context per
   window for the bytes arm.
3. **Working hypothesis: long-tail patch-type frequency.** BPE tokens
   average ~528 train samples per type; RateC C=10 patch types average
   ~267 with a heavy singleton tail (early bake-off: ~30–65% singletons);
   raw bytes average ~9,600. Tiny embeddings starve on singleton-heavy
   vocabularies.
4. **Caveat — scale is not bounded by this.** 600 steps / 1.3M params is a
   starting-deficit measurement, not a bound on the method. Patch-based
   interfaces (MegaByte, BLT) close such gaps with scale + local models.
5. **Next levers:** training-scale probe (more steps ⇒ more samples per
   patch type — directly tests finding 3), frequency-aware book pruning,
   and a C sweep for the LM optimum (not the tokenizer optimum).

## Training-scale probe (2400 steps, post-v0.4.1)

Directly tests finding 3: same two matched-params arms, same everything,
**4× the training steps** (2400; OneCycle schedule recomputed for the longer
run; seed 42; identical bytes; identical full-coverage eval).

| arm | BPB @600 | BPB @2400 | improvement |
|---|---|---|---|
| bpe | 3.112 | 2.607 | −0.505 (−16.2%) |
| ratec_direct | 4.796 | 3.818 | **−0.978 (−20.4%)** |
| **gap** | **1.684** | **1.212** | **−28.1%** |

- RateC improved **1.94× more** than BPE for the same 4× compute.
- The gap narrowed in **every domain**: en 1.828→1.370 (−25%), hr
  1.649→1.260 (−24%), zh 1.951→1.257 (−36%), code 1.332→0.918 (−31%).
- Honest framing: two points show **direction, not a trend line**. The gap
  is still large (1.21 BPB — BPE still wins decisively at 1.27M params),
  and the `bytes` / factorized-`ratec` arms were not re-run at this scale.
  But the direction is exactly what the singleton-starvation hypothesis
  predicts: 4× steps ≈ 4× samples per patch type.

Artifacts: `research/twin_probe_s2400.json` (digest), plus per-arm JSONs,
curves and checkpoints in the authoring workspace
(`runs/twin_*_s2400.*`).

## Artifacts

- `research/twin_summary.json` — machine-readable digest of the 600-step
  result above.
- `research/twin_probe_s2400.json` — machine-readable digest of the
  2400-step training-scale probe.
- `research/twin_study.py` — the exact harness (arch, streams, eval,
  checkpoint-resume for long runs). It
  reads workspace-local tokenized streams (`runs/ratec_main_*` /
  `runs/bpe_mix_*` .npy) and is published here so the protocol is
  auditable line-by-line; the full per-arm JSONs/curves and checkpoints
  live in the authoring workspace (`runs/twin_*.json`, `runs/twin_*.pt`).
