# The ratec cookbook (P3.4)

Task-oriented recipes, each grounded in the frozen API surface
(`research/api_manifest.json`) and the guarantees it relies on. Every
recipe here runs as written against the committed surface.

## 1. Train, encode, decode — the basic loop

```python
import ratec

tok = ratec.RateC(c=20)              # c = bits per patch (the rate dial)
tok.encode_bytes(corpus)             # adapt + tokenize in one pass
ids = tok.encode_bytes(new_text)     # the frozen book tokenizes new text
back = tok.decode_bytes(ids)         # byte-exact inverse
assert back == new_text
```

Guarantees: `decode_bytes(ids) == data` is the decode contract (S7,
machine-checked totality iff; the C decode fuzz asserts it propertywise).
The ids are content-addressed — re-encoding the same bytes against the
same book state gives the same ids (S5/S6).

## 2. Parallel tokenization (multi-thread, one engine)

```python
starts, ends, ids = tok.encode_stream_parallel(
    data, chunk=1 << 20, threads=4, ids=True)
# or arrays-only (no book growth in the returned arrays):
starts, ends = tok.encode_stream_parallel(data, threads=4, as_arrays=True)
```

The parallel intern assigns the SEQUENTIAL first-seen ids for ANY
chunking (S6 — read-stable merge, no third outcome); the tiling is
correct for any chunk boundaries (T2). Thread-count choice does not
change the ids — `bench/soak_nightly.py` asserts it every night.

## 3. Batch many documents

```python
docs = [open(p, "rb").read() for p in paths]
results = ratec.batch_encode(docs, c=20, threads=4)
for ids, lens in results:            # (ids, doc_lengths) per doc
    ...

# or straight from files, split on the separator:
tok = ratec.RateC(c=20)
out = tok.encode_file("corpus.bin", separator=b"<|endoftext|>",
                      threads=4, max_docs=1000)
```

## 4. Multi-GiB prep (flat memory, byte-identical)

```python
def chunks():                        # any iterable of byte chunks
    with open("big.bin", "rb") as f:
        while True:
            b = f.read(1 << 28)      # 256 MiB
            if not b:
                return
            yield b

ids = ratec.chunked_prep(chunks(), c=20, engine_bytes=1 << 32,
                         threads=2, ids=True)
```

`chunked_prep` encodes each segment on a FRESH engine and the result
is per-segment identical to encoding that segment standalone (the
8 GiB scale gate: 32 × 256 MiB segments, sha256 equality against
explicit per-segment engines — `research/gate_8gib.json`). RSS stays
at the engine size + chunk, never the stream size; a uint32 span space
means ONE engine covers at most 4 GiB of lifetime stream — chunking is
the documented way past it. Pass `sink=` to stream ids out per segment
instead of accumulating.

## 5. Rate control — hit a token budget

```python
c, n = ratec.solve_c(corpus, target_patches=500_000)   # ~500k patches
# n is EXACT for the returned c (S8d); the bisection keeps a
# crossing bracket (S8b) and converges (S8c)

c2, book = ratec.solve_c_for_book(corpus, target_book=50_000)
# book size vs C is unimodal — coarse scan then local bisection
```

Patch count is monotone nonincreasing in C (the property the servo
tests replicate externally); the judge pass runs once, so each
candidate costs only the cheap re-patch (`ratec.patch_from_sur`).

## 6. Books — dump, transplant, reserve

```python
ranks = tok.to_ranks()               # dict[bytes, int]: piece -> id
pieces = [p for p, _ in sorted(ranks.items(), key=lambda kv: kv[1])]

# exact transfer of the whole trained state:
clone = ratec.RateC.from_state(tok.save_state())

# or ingest a donor book piece-wise (approximate vocabulary transfer):
other = ratec.RateC(c=20, reserve=len(pieces))   # pre-size the book map
for piece in pieces:
    other.feed(piece)
other.flush()
ids = other.encode_bytes(text)       # tokenize against the donor vocab
print(tok.book_size, tok.n_patches, tok.rehash_count())
```

`to_ranks()` exposes the piece→id map (portable books); `from_state`
is the EXACT state transfer (continuation identity — the bridge tests
assert it). `reserve=` pre-sizes the context/book maps (the v0.9.1
lever — no rehash growth during ingestion); `rehash_count()` reports
how many grows happened.

## 7. Snapshots — freeze, reload, prune

```python
tok.save_state("ratec-c20.bin")      # the RCZ1 container (snapshot v8)
back = ratec.RateC.from_state(open("ratec-c20.bin", "rb").read())
# `back` continues exactly like `tok` from the same state
# (warm-book continuation identity — the GGUF/tokenizer tests assert it)

tok2 = ratec.RateC(c=20, prune=True, prune_after=1 << 20, prune_min=2)
tok2.encode_bytes(large_stream)      # prunes cold contexts as it grows
print(tok2.n_pruned)
```

Snapshots carry the book + judge state + the open patch; load rejects
corrupt payloads with `SNAPSHOT_INVALID` and foreign versions with
`SNAPSHOT_VERSION` (S9 — total on valid states, classified rejection
otherwise). v2…v8 payloads read back compatibly; the writer stays v8.

## 8. Ecosystem bridges

```python
ratec.export_tokenizer_json(tok, "tokenizer.json")   # HF WordLevel vocab
ratec.export_gguf(tok, "ratec-c20.gguf")             # GGUF v3 carriage
```

- **tokenizer.json**: WordLevel vocab, `vocab[latin1(piece)] == id`,
  decode round-trips byte-exactly (the vLLM recipe:
  `docs/vllm_recipe.md` — encoding is ratec's, decoding is anyone's).
- **GGUF**: a metadata container — any GGUF parser reads the vocab
  table; llama.cpp must NOT re-tokenize with it (the vocab alone cannot
  reproduce ratec's segmentation). The embedded `ratec.snapshot.v8`
  (sha256-bound) is the authoritative state.

## 9. Errors — the coded taxonomy

```python
from ratec import RatecError, RatecErrorCode

try:
    ratec.RateC.from_state(b"garbage")
except RatecError as e:
    if RatecErrorCode(e.code) is RatecErrorCode.SNAPSHOT_INVALID:
        ...      # corrupt payload (vs SNAPSHOT_VERSION: foreign version)
```

The 8 codes — `TILING`, `BOOKLESS`, `CAP_EXHAUSTED`, `BUFFER_LIMIT`,
`SNAPSHOT_INVALID`, `SNAPSHOT_VERSION`, `ARGUMENT`, `STATE` — are
frozen (docs/STABILITY.md §5); every site maps to exactly one code
(`tests/test_errors.py` audits the mapping).

## 10. Maintenance and sanity

```python
report = tok.validate(sample)        # decode round-trip + state check
print(tok.c, tok.carry, tok.carry_mode, tok.bytes_seen)
print(tok.n_forced, tok.prune_after, tok.prune_min)
sur = tok.surprise(data)             # the surprisal stream (numpy array)
counts = tok.counts()                # per-id usage counts (numpy array)
tok.save_ranks("ranks.json")         # persist the piece->id ranking
tok.reset()                          # drop judge state, keep nothing
```

`validate` re-decodes and compares — the fastest way to assert a
healthy engine after loading a snapshot. `surprise` exposes the exact
surprisal stream (the servo's raw material); `counts()` the per-id
usage histogram.
