# RateC — the rate-C information-vocabulary tokenizer, as a library

`ratec` replaces BPE's learned merge table with an **adaptive entropy model
(the judge)** that scores every byte, and a **greedy rate-packing rule (the
patcher)** that cuts patches at a fixed C-bit budget. There is **no training
data, no vocabulary file, no regex pre-tokenizer**: the model is the
algorithm, it adapts as it reads, and any C works at any time.

> **Status:** v0.3.0, experimental. The C core and byte-level semantics are
> stable and bit-exact gate-verified; the LM-side surface (`ratec.smollm2`)
> is new and may still move. Code-first repo: run `tests/` and `bench/`
> yourself — no benchmark claims against SOTA in this README.

```
pip install .            # C extension, gcc >= 9, x86-64 (or any 64-bit)
```

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

Fertility (units per byte, C=20, this repo's bake-off corpora): ratec 0.20
(en) / 0.29 (zh) / 0.50 (b64) vs tiktoken r50k 0.28 / 0.72 / 0.77 —
**26–50% fewer LM positions** for the same bytes.

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
   FLOPs-per-byte; tokenizer ecosystem bridges (GGUF/llama.cpp, vLLM)
   once the training side settles.

## Layout

```
ratec/
  _rcengine.c        the whole C core (engine, book, batch, snapshots)
  api.py / cli.py    pythonic surface + CLI
  smollm2.py         HF tokenizer (frozen vocab) + SmolLM2 config builder
tests/
  gate_bitexact.py   bitwise gate vs the validated reference
  test_ratec.py      40+ checks: round-trips (text/binary), streaming
                     parity, C sweep, batch, book, snapshots, errors
  test_smollm2_port.py   freeze / roundtrip / fallback / specials / config
bench/
  bench_vs_tiktoken.py   fertility table above, reproducible
  bench_c_sweep.py       tokens/byte + MB/s across C, golden-verified per C
docs/
  smollm2_port.md    the LM port, precisely
  PROFILING.md       the ablation profile behind the v2 engine
```

Rebuild in place:

```
gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fPIC -shared \
    -I"$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")" \
    ratec/_rcengine.c -o ratec/_rcengine.cpython-*.so -lm -lpthread
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
