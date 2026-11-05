# SmolLM2 port — training SmolLM2 on RateC tokens

`ratec.smollm2` makes RateC a first-class tokenizer for SmolLM2 (any
Llama-family model) in the Hugging Face `transformers` ecosystem. It
works without torch; the model side needs torch and is torch-guarded.

## What the port provides

| piece | what it is |
| --- | --- |
| `RateCTokenizer` | `PreTrainedTokenizer` subclass (transformers 5.x `PythonBackend`), backed by a **frozen** RateC engine snapshot |
| `encode_for_lm` | text → `(patch_ids, patch_byte_lengths)` — the LM-facing contract (MegaByte/BLT-style patching) |
| `build_smollm2_ratec_config` | SmolLM2-135M-shaped `LlamaConfig` sized to the RateC vocab (torch-free) |
| `SmolLM2RateCAdapter` | torch-side adapter stub; raises a clear error until `pip install torch` |

## Frozen-vocab id spaces (stable after construction)

```
[0, book)          frozen book ids — the patch content exactly
[book, book+256)   byte-fallback ids — single byte (id - book), latin-1
                   transparent; used when live patching produces a patch
                   whose content is not in the frozen book (the patch is
                   emitted as its bytes; single bytes already in the book
                   reuse their book id)
[0xFFFFFFF0 + i]   special tokens (up to 16), transformers added-token
                   machinery, never colliding with book growth
```

Any text round-trips byte-exactly with a frozen vocab; `vocab_size` is
`book + 256` and never changes. (A few byte-block ids may be unused when
that single byte is already a book patch — `get_vocab()` covers every
emittable id and is injective.)

## 3-step training recipe

```python
import ratec
from ratec.smollm2 import RateCTokenizer, build_smollm2_ratec_config

# 1. Ingest the corpus to grow the dynamic book (this is "RateC training")
tok = ratec.RateC(c=20)
for chunk in corpus_chunks:            # bytes
    tok.encode_bytes(chunk)
snapshot = tok.save_state()

# 2. Freeze it into an HF tokenizer
tokenizer = RateCTokenizer(snapshot=snapshot)
tokenizer.save_pretrained("ratec-tokenizer")        # ratec_snapshot.bin + configs

# 3. SmolLM2-shaped config with the RateC vocab; train with any trainer
cfg = build_smollm2_ratec_config(tokenizer)         # LlamaConfig, no torch needed
# model = LlamaForCausalLM(cfg)                     # needs torch
# trainer(model, tokenizer=RateCTokenizer.from_pretrained("ratec-tokenizer"), ...)
```

Feed the model `(patch_id, patch_len)` pairs from
`tokenizer.encode_for_lm(text)`: one embedding per patch,
`sum(lens) == len(text.encode("utf-8"))` always holds.

## Loading

* `RateCTokenizer.from_pretrained(dir)` — supported path; restores the
  engine snapshot, frozen book, and special tokens; ids reproduce exactly.
* transformers **5.x** `AutoTokenizer` always converts to fast
  (Rust) tokenizers and cannot load slow-only custom classes (`use_fast`
  is ignored by design) — pass the tokenizer object explicitly to
  `Trainer`/`model.from_pretrained(..., tokenizer=...)` instead.

## Special tokens

Llama-style wrapping is enabled automatically once `bos_token`/`eos_token`
exist (`special_tokens_pattern = "bos_eos"` → `[BOS] seq [EOS]`).
`decode(..., skip_special_tokens=True)` strips them; ids live in the
reserved `0xFFFFFFF0+` block.

## Tests

`python3 tests/test_smollm2_port.py` — snapshot freeze, byte-exact
round-trips, ids == raw-engine ids for in-book content, byte-fallback for
held-out content, specials, save/from_pretrained id equality, LM helper,
config shape (SmolLM2-135M: hidden 576 / layers 30 / heads 9 / kv 3).
