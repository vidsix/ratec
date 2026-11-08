# RateC — the rate-C information-vocabulary tokenizer, as a library

`ratec` replaces BPE's learned merge table with an **adaptive entropy model
(the judge)** that scores every byte, and a **greedy rate-packing rule (the
patcher)** that cuts patches at a fixed C-bit budget. There is **no training
data, no vocabulary file, no regex pre-tokenizer**: the model is the
algorithm, it adapts as it reads, and any C works at any time.

> **Status:** v0.5.0, experimental. The C core and byte-level semantics are
> stable and bit-exact gate-verified; the LM-side surface (`ratec.smollm2`)
> and the patcher modes/servos are new and may still move. Code-first
> repo: run `tests/` and `bench/` yourself — no benchmark claims against
> SOTA in this README. Independently replicated: an external gate
> (tiling/parity/carry/fractional-C monotonicity) passes on foreign boxes,
> and the base64 fertility number replicates exactly.

```
pip install .            # C extension, gcc >= 9, x86-64 (or any 64-bit)
make test                # or: python -m pytest tests/  (conftest builds in place)
```

Tests run anywhere: the scored spans, reference surprisal/starts, the
v4-continuation slices, the u8_image raster and the snapshot-training corpus
ship under `tests/data/` (sha256-bound, ~11 MB incl. reference arrays). Env
overrides: `RATEC_FIXTURES`, `RATEC_BAKEOFF_DATA` + `RATEC_BAKEOFF_CACHE`
(full per-C golden gate), `RATEC_OUT`.

```python
import ratec
tok = ratec.RateC(c=20)                  # any c=1..64, any time

ids = tok.encode("hello 你好!")          # patch ids (book-backed)
tok.decode(ids)                          # -> "hello 你好!"  (round-trips)

tok.encode_to_spans(b"...")              # [(start, end, id|None), ...]
tok.surprise(b"...")                     # per-byte float32 surprisal
ratec.batch_spans(docs, c=20, threads=N) # parallel bulk encoding
ratec.RateC(c=20, exact=False)           # fast lane (poly log2, ~1 ulp)
ratec.RateC(c=20, store_book=False)      \
    .encode_stream_parallel(             # ONE stream on many cores
        blob, chunk=1<<20, threads=N)
```

## Why it is different

| | BPE (tiktoken) | RateC |
|---|---|---|
| vocabulary | fixed, fitted offline | none — patches are defined by information, not by a table |
| C (tokens-per-byte dial) | retrain for every vocab size | any `c=1..64` at runtime |
| modalities | text only (byte-fallback for the rest) | bytes are bytes: text, code, CJK, base64, PNG, PCM — one encoder |
| adaptation | none | judge adapts continuously; optional snapshot = "warm start" |
| new domain | out-of-vocabulary fury | just feed it |
| LM interface | ids | ids **and** raw byte spans (the LM-visible structure) |

Fertility (units per byte, C=20, this repo's bake-off corpora): ratec
0.204 (en) / 0.292 (zh) / 0.500 (b64) vs tiktoken r50k 0.275 / 0.722 /
0.770 — **26–60% fewer LM positions** (same numbers as the per-domain
table in "What RateC buys" below; 0.204/0.275 = −26%, 0.292/0.722 =
−60%, 0.500/0.770 = −35%). On high-entropy streams
the credit carry-over patcher compounds the win: base64 0.500 → **0.384**
units/byte (−23.2% patches; an independent replication measured −23%) —
reproduce with `patch_from_sur(RateC(store_book=False).surprise(b64), 20,
carry=True)` against `tests/data/fixtures/base64.span.bin`.

## API

### `ratec.RateC(c=20, store_book=True, pf=2, warm_state=None)`

* **`c`** — patch budget in bits (1..64). Smaller C → shorter patches, more
  of them. The judge is C-independent, so the choice is purely downstream
  (embedding width, context budget).
* **`store_book`** — keep the dynamic patch book (content → id, first-seen
  order) so `encode`/`decode` round-trip without the source. Turn off for
  bulk LM data prep; spans alone need no book.
* **`pf`** — software prefetch depth 0..4 for the order-3..5 maps
  (default: auto-tuned; 2–3 measured best on x86-64).
* **`exact`** — `False` selects the fast lane: polynomial log2 (max
  deviation 1.1e-7 relative ≈ one float32 ulp, 100% patch-start agreement
  with exact on our test corpora) plus a relaxed summation tree. Not
  bit-comparable with exact mode; round-trips and fertility unchanged.
* **`warm_state`** — snapshot bytes/path from `save_state()` (see below).

Core methods (tiktoken-shaped where it makes sense):

| method | semantics |
|---|---|
| `encode(text)` / `decode(ids)` | utf-8 ⇄ book ids |
| `encode_bytes(b)` / `decode_bytes(ids)` | raw bytes ⇄ ids |
| `encode_to_spans(b)` | `[(start, end, id)]` — patches tile the input exactly |
| `surprise(b)` | float32 per-byte surprisal (judge state advances) |
| `feed(b)` / `flush()` | streaming: patches closed by each chunk; spans are absolute |
| `encode_batch(docs, threads)` | parallel, GIL released, one cold engine per doc |
| `encode_stream_parallel(data, chunk, threads, head_chunks)` | ONE stream on many cores: sequential head + snapshot-seeded chunk workers (ffmpeg-style approximation, deterministic, tiles exactly) |
| `save_state()` / `RateC.from_state(x)` | full snapshot (judge + patcher + book) |
| `reset()` | in-place clear, page-granular via dirty-bitmaps (cost ∝ touched pages) |

Module-level: `ratec.patch_from_sur(sur, c)` (re-patch a stored surprisal at
any C — the judge pass is C-independent, so C-sweeps are nearly free),
`ratec.batch_spans(docs, c, threads)`, `ratec.RatecError`.

CLI: `python -m ratec encode|decode|surprise|state|bench …`

## LM interface (v0.3): HF tokenizer + SmolLM2 port

`ratec.smollm2` turns a RateC snapshot into a first-class HF tokenizer for
Llama-family training — details in `docs/smollm2_port.md`. The shape:

```python
import ratec
from ratec.smollm2 import RateCTokenizer, build_smollm2_ratec_config

tok = ratec.RateC(c=20)
for chunk in corpus:                 # 1. "training" = ingesting bytes;
    tok.encode_bytes(chunk)          #    judge + book adapt as they read
snapshot = tok.save_state()

tokenizer = RateCTokenizer(snapshot=snapshot)   # 2. freeze into an HF tokenizer
tokenizer.save_pretrained("ratec-tokenizer")    #    frozen vocab + 256 byte-fallbacks

cfg = build_smollm2_ratec_config(tokenizer)     # 3. SmolLM2-135M-shaped LlamaConfig
# train any LlamaForCausalLM on (patch_id, patch_len) pairs from
# tokenizer.encode_for_lm(text) — sum(lens) == len(utf8 bytes), always
```

Key properties: any text round-trips **byte-exactly** with a frozen vocab;
`encode_for_lm` is the MegaByte/BLT-style `(id, byte_length)` contract;
special tokens live in a reserved id block that never collides with the book.

## v0.4: fractional C, carry-over patcher, servo, profiles

```python
import ratec
from ratec import patch_from_sur, solve_c, solve_c_for_book
from ratec.profiles import apply, restore

tok = ratec.RateC(c=17.5)                # fractional C — the budget is a float
ratec.RateC(c=20, carry=True)            # credit carry-over patcher (opt-in):
                                         # each patch may spend the previous
                                         # patch's unused budget (cap +C);
                                         # mean rate -> C exactly. Measured
                                         # -15.1% patches at C=20 (mixed).

c, n = solve_c(corpus, 50_000)           # servo: C for a patch-count target
c, b = solve_c_for_book(corpus, 16_384)  # servo: C for a book-size target
                                         # (frozen-vocab donor match)

t = apply("pcm16", pcm_bytes)            # modality profiles: lossless byte
restore("pcm16", t) == pcm_bytes         # transforms upstream, core untouched;
                                         # -37% patches on PCM, -9.5% on f32
```

Patch counts and MB/s are measured together in `bench/bench_c_sweep.py`;
the carry/deferred-performance record (incl. why the SIMD patcher is a
5.5%-ceiling defer) lives in `docs/v04_design.md`. Container spec draft:
`docs/container.md`.

## Semantics (the precise bits)

* **Judge.** Fixed-weight mixture of six adaptive KT estimators over byte
  contexts of order 0..5. Per byte: `pm = Σ_oi W[oi]·q_oi(b)` with
  `W = [.02,.04,.08,.16,.30,.40]`, KT smoothing `(cb+0.2)·inv/(1+tot·inv)`,
  `inv = 1/(256·0.2)`, cold-start uniform `1/256`. Surprisal
  `sur = -log2(pm)` (float32). Orders 0–2 use direct tables; orders 3–5 use
  one shared open-addressing map with inline 16-entry occurrence windows
  (u16 counts: contexts stay single-probe up to 65535 total visits; a
  windowed context predicts and updates in ONE 64-byte — one cache line —
  probe. Only contexts exceeding the window promote to the 16-byte count
  slots of the shared map).
* **Patcher.** Greedy under-shoot: a patch is the longest run with
  `Σ sur ≤ C` (searchsorted right semantics), forced single-byte escapes
  when one byte alone exceeds C.
* **Streaming.** `feed(a); feed(b); flush()` is *bit-exact* with
  `encode(a+b)`: the running prefix sum, open-patch state and judge tables
  persist across calls. `flush()` hard-closes the open tail patch.
* **Bit-exactness.** Built with `-ffp-contract=off -fno-fast-math` and the
  same double-expression shapes as the reference pipeline; validated
  bitwise (surprisal arrays and patch starts) against the baked-off
  reference on four domains — see `tests/gate_bitexact.py`.
* **Snapshot format.** `save_state()` returns a zlib-compressed blob
  (magic `RCZ1`, inner format v4) containing the full judge tables
  (order-2 rows RLE'd), maps, patcher state and book. A 120 KB-trained
  state is ~0.7 MB. Snapshots are portable across processes with the same
  build; endianness is native (x86-64).
* **Limits.** Span offsets are uint32 → one engine covers ≤ 4 GiB of
  stream. Orders-3..5 maps cost up to ~96 B of physical memory per hot
  byte; `map_capbits` (default 2^22 slots) bounds them, with graceful
  slowdown and a clean error if truly exhausted. Engines are not
  thread-safe (use one per thread or `batch_spans`).

## Efficiency vs scale (the design thesis, measured)

BPE's fertility is frozen at train time — a fixed merge table performs
identically on byte 1 and byte 10^12. RateC's judge and book adapt as they
read, so its positions-per-byte on **held-out** data should improve with
the size of the stream it has already adapted to. `bench/bench_scale.py`
measures exactly that: one engine, log-spaced adaptation prefixes, the
**same fixed 128 KB probe** (disjoint from every prefix) scored at each
scale. On a 15.8 MB en/hr/zh/code pretraining mix (C=20, full record in
`research/scale_mix15.8MB.json`):

| adaptation seen | units/byte (probe) | vs cold |
|---|---|---|
| 0 (cold engine) | 0.157 | — |
| 64 KB | 0.102 | −35% |
| 256 KB | 0.082 | −48% |
| 1 MB | 0.076 | −52% |
| 8 MB | 0.076 | −52% (saturated) |
| tiktoken r50k_base (same probe) | 0.431 | flat by construction |

Honest reading: the effect is real and front-loaded — −48 of the −52
points arrive within the first 256 KB, and the curve saturates at ~0.076
even though the book keeps growing (1.17 M entries at 8 MB, no further
fertility benefit). A warm RateC holds **4.2–5.7× fewer LM positions**
than tiktoken r50k_base on the same bytes (ratio over the 64 KB → 8 MB
scales). This measures positions-per-byte, not LM quality — the quality
question is the twin study's (`research/`), and its scale lever is still
open.

Reproduce: `RATEC_SCALE_DATA=<your-corpus> make scale` (portable shipped
fixtures by default; the table above used the same 15.8 MB mix as the
twin study).

## Performance (2-core sandbox, C=20)

Controlled comparison (interleaved min-of-5, pooled steady-state,
5 MB mixed corpus, bytes → patch spans, bit-exact both sides):

| engine | MB/s |
|---|---|
| v1 (u8 windows, 32 B slots) | 9.0 |
| **v2 (u16 windows, 64 B line-aligned slots, 16 B count slots)** | **11.1** (**+22%**) |

Ablation profile (see `docs/PROFILING.md`): orders-3..5 map machinery
was ~72% of runtime — hot order-3 contexts promoted early and paid a
second cache probe per visit; the widened windows keep contexts
single-probe up to 65535 visits, and every probe now touches exactly
one cache line.

Library bench vs `tiktoken r50k_base` (same bytes, same process,
round-robin interleaved reps, medians — reproducible via
`bench/bench_vs_tiktoken.py`): exact RateC sits at **0.4–0.65× of
tiktoken's per-byte speed depending on box state** (this 2-core shared
sandbox drifts tens of percent; memory-bound adaptive workloads suffer
more under contention than tiktoken's compact lookup loops). What
RateC buys instead, per byte:

| domain | tiktoken r50k units/byte | ratec units/byte | fewer LM positions |
|---|---|---|---|
| english | 0.275 | 0.204 | −26% |
| chinese | 0.722 | 0.292 | −60% |
| base64 | 0.770 | 0.500 | −35% |

plus: no vocab file, any C at runtime, byte-native multimodality, and
real multi-core throughput via `batch_spans` (independent docs) and
`encode_stream_parallel` (one stream, chunked from a snapshot — measured
in this sandbox at ~1.7× on 2 threads for the parallel section, scaling
with cores on real machines). The honest verdict: exact RateC does ~100×
more arithmetic per byte than a merge lookup and currently trades speed
for compression; the fast lane and the profiler roadmap below are the
paths to raw speed.

## Roadmap (updated after the profiling round — see docs/PROFILING.md)

1. ~~SIMD mixture terms~~ tried, measured, reverted: scalar `divsd`
   already overlap via OoO; the lane round-trip was a net loss on this
   microarchitecture. Documented in PROFILING.md.
2. ~~Fast (non-exact) lane~~ **done** — `RateC(exact=False)`; ≤ 1 float32
   ulp deviation, 100% patch-start agreement on test corpora.
3. ~~Approximate multi-stream mode~~ **done** — `encode_stream_parallel`.
4. **Next levers, in expected-value order:** per-order prefetch
   schedules; per-order tables sized to distinct-context counts; on
   real multi-core boxes, batch/parallel modes dominate everything
   single-threaded.
5. **LM track (active):** twin-study training harness — same byte
   budget, RateC vs BPE vs raw bytes, measured on BPB / params /
   FLOPs-per-byte. **First result (v0.4.1, `research/twin_summary.json`):**
   compute-matched at 1.3M params on identical 15.8M train / 659k val bytes,
   BPE 3.112 < raw bytes 4.147 < RateC C=10 4.80–4.82 BPB; the factorized
   interface is free (Δ 0.02 BPB vs direct ids) — the tokenization gap is
   long-tail patch-type frequency starving tiny embeddings. Full record:
   `docs/twin_study.md` + `research/`. **Training-scale probe (4× steps,
   post-v0.4.1):** the gap narrows 1.684 → 1.212 BPB (−28.1%); RateC
   improves 1.94× more than BPE for the same extra compute, in every
   domain. Next: full 4-arm matrix at 2400 steps, frequency-aware book
   pruning, LM-optimal C sweep, wider scale ladder. Tokenizer
   ecosystem bridges (GGUF/llama.cpp, vLLM) once the training side settles.
6. **~~Fractional C~~ done** — double C everywhere, snapshot v5.
7. **~~Credit carry-over patcher~~ done** — `carry=True`; −15.1% patches
   at C=20, mean rate → C exactly, streaming bit-exact.
8. **~~Rate servo~~ done** — `solve_c` (exact, patch-count target) and
   `solve_c_for_book` (unimodal target, coarse+refine).
9. **~~Modality profiles~~ done** — `ratec.profiles`, registry-as-data,
   lossless transforms (pcm16 −37.1% patches, f32 −9.5%, image −9.6%).
10. **SIMD prefix-sum patcher / run-detection fast path** — measured and
    deferred: the patcher is 5.5% of runtime and the judge's orders-3..5
    probes are fused predict+update (an exact skip cannot avoid them).
    Numbers and the revisit conditions are in `docs/v04_design.md`.
11. **~~External-replication hardening~~ done (v0.4.1)** — shippable
    fixtures replace author-local paths, v4-continuation goldens replay
    byte-exactly on any machine, parallel-fertility is informational
    (tiling/determinism stay hard), `conftest.py` + `make test` for the
    first-run experience.
12. **~~Efficiency vs scale, tokenizer side~~ measured (v0.5.0)** —
    adaptation improves held-out fertility −52% with a 256 KB-scale
    knee and a ~0.076 units/byte floor (5.7× under tiktoken r50k);
    saturated by ~1 MB. Numbers, honest caveats and reproduce command
    in the section above; record in `research/scale_mix15.8MB.json`.
    LM-side scale (BPB vs training scale) remains open.

## Layout

```
ratec/
  _rcengine.c        the whole C core (engine, book, batch, snapshots)
  api.py / cli.py    pythonic surface + CLI
  smollm2.py         HF tokenizer (frozen vocab) + SmolLM2 config builder
  profiles.py        modality profiles: lossless transforms + registry
tests/
  gate_bitexact.py   bitwise gate vs the validated reference
  test_ratec.py      40+ checks: round-trips (text/binary), streaming
                     parity, C sweep, batch, book, snapshots, errors
  test_v04.py        fractional C, carry patcher (vs python golden),
                     snapshot v5 + v4 back-compat, servos
  test_profiles.py   profile round-trips + predictability wins
  test_smollm2_port.py   freeze / roundtrip / fallback / specials / config
bench/
  bench_vs_tiktoken.py   fertility table above, reproducible
  bench_c_sweep.py       tokens/byte + MB/s across C, golden-verified per C
  bench_scale.py         efficiency vs adaptation scale (the thesis, measured)
docs/
  smollm2_port.md    the LM port, precisely
  PROFILING.md       the ablation profile behind the v2 engine
  v04_design.md      what landed / was measured / was declined, with numbers
  twin_study.md      the paired BPE-vs-RateC LM experiment, with numbers
  PLAN_v0.6.md       the next version's plan (from the full-repo review)
  container.md       container format spec (draft v0.1)
research/
  twin_study.py      the twin-study harness (protocol in the docstring)
  twin_*.json        committed result records (seed-42 run + scale probe)
  scale_mix15.8MB.json   efficiency-vs-scale record behind the table above
```

Rebuild in place:

```
gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fPIC -shared \
    -I"$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")" \
    ratec/_rcengine.c -o ratec/_rcengine.cpython-*.so -lm -lpthread
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
