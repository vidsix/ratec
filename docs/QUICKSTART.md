# QUICKSTART — ratec in the first ten minutes

ratec replaces BPE's learned merge table with an adaptive 6-order KT
mixture (the judge) that scores every byte, and a greedy rate-packing
rule (the patcher) that cuts patches at a C-bit budget. No training
data, no vocab files: the tokenizer IS an algorithm plus a saved state.

## 0. Install

```bash
# A. release wheels — no compiler (cp39-cp312, x86_64 + aarch64):
#    grab the wheel for your interpreter from
#    https://github.com/vidsix/ratec/releases (assets auto-attached)
pip install ./ratec-1.0.0rc3-cp312-cp312-manylinux2014_x86_64.whl

# B. straight from GitHub (builds the C extension, needs gcc >= 9):
pip install git+https://github.com/vidsix/ratec@v1.0.0rc3

# C. from a clone:
git clone https://github.com/vidsix/ratec && cd ratec && pip install .
```

## 1. The tiktoken-shaped flow

```python
import ratec

tok = ratec.get_ratec(c=20)          # one call, cached per config
tok.train(corpus_bytes)              # one bulk pass: encode IS the adaptation
ids = tok.encode("hello world")      # str -> list[int]
tok.decode(ids)                      # -> "hello world"
tok.n_vocab                          # the adaptive book size (grows in use)
```

That is the whole API surface a BPE/tiktoken user needs. The one
conceptual difference: **ratec adapts on every encode** — a ratec
engine is a state, not a static table. Train once (`train`),
then encode freely; when you want to PIN the behavior (serving,
release, bridging to other engines), save the state:

```python
blob = tok.save_state()              # bytes (the RCZ1 container)
open("my-model.ratec", "wb").write(blob)
tok2 = ratec.load("my-model.ratec")  # path | bytes | http(s) URL, memoized
```

`load` continues the engine BIT-EXACTLY — the loaded tokenizer encodes
identically to the one that saved, byte for byte (that invariant is a
release gate, not a hope).

## 2. Reproducible config

`c` is the per-patch bit budget — the only real hyperparameter. Bigger
`c` = bigger patches = smaller ids-per-byte. `ratec.solve_c(data,
target_patches=...)` searches it for you; `pf` (pre-filter depth) and
`store_book` (build the decode table) round out the config that
`get_ratec(c=..., pf=..., store_book=...)` accepts. Full parameter
reference: README "API".

## 3. Bridge to the engines (day 1)

```python
ratec.export_tokenizer_json(tok, "tokenizer.json")   # HF / vLLM / SGLang
ratec.export_gguf(tok, "my-model.gguf")              # llama.cpp metadata subset
```

- **HF transformers / vLLM / SGLang**: the exported `tokenizer.json`
  loads through the `tokenizers` layer they all share; decode is
  byte-exact (proven, `research/integration_proofs.json`). Pre-encode
  prompts with ratec and feed `prompt_token_ids`.
- **llama.cpp**: the GGUF carries the vocab + the snapshot; it must NOT
  re-tokenize with the file (the segmentation is the algorithm). The
  snapshot decode recipe: `docs/vllm_recipe.md`.
- The contract in one line: **encoding is ratec's, decoding is
  anyone's** — the vocabulary is a table, the segmentation is the
  algorithm.

## 4. LM data prep at speed (the ids lane)

```python
ids = ratec.chunked_prep(token_iter, c=20)    # 1 GiB+ at flat RSS
docs = ratec.batch_encode(list_of_docs, c=20) # thread-parallel spans
```

Details and the measured floors: README "The ids lane",
BENCHMARKS.md.

## 5. Where to go next

- README — full API, semantics, the efficiency thesis
- docs/vllm_recipe.md — the engine matrix + the GGUF snapshot recipe
- docs/COOKBOOK.md — error taxonomy, streams, parallelism, recipes
- STABILITY.md — what is frozen (API, snapshot v8, formats) and what
  may still grow
- BENCHMARKS.md — the measured record
