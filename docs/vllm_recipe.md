# The vLLM validation recipe (PLAN_v0.9 P3.3)

How a frozen ratec snapshot feeds a vLLM deployment, and what each side
is allowed to do. The contract: **encoding is ratec's, decoding is
anyone's** — the vocabulary is a table, the segmentation is the
algorithm.

## 1. Freeze the snapshot

```python
import ratec
tok = ratec.RateC(c=20)
tok.encode_bytes(corpus)            # adapt on the prep corpus
tok.save_state("ratec-c20.bin")     # RCZ1 container, snapshot v8
ratec.export_tokenizer_json(tok, "tokenizer.json")
```

`export_tokenizer_json` writes a WordLevel vocab (`vocab[latin1(book_bytes)]
== id` for every id — the round-trip gate in `tests/test_v091.py`
asserts the inverse reconstructs `decode_bytes(ids)` exactly).

## 2. Serve with vLLM

Point vLLM's `--tokenizer` at the exported `tokenizer.json`:

```
vllm serve <model> --tokenizer ./ --tokenizer-mode slow
```

- **Decoding**: vLLM decodes generated ids through the WordLevel vocab
  — byte-exact for ids the ratec engine produced (latin-1 is a
  byte↔char bijection; the JSON round-trip gate proves it).
- **Encoding**: vLLM's tokenizer must NOT be used to encode prompts —
  WordLevel with no pre-tokenizer would segment on whole-token
  boundaries, not ratec's rate patching. Pre-encode with ratec and feed
  ids directly (`prompt_token_ids`), the same way any custom tokenizer
  bypasses the server-side encoding.

## 3. The validation gate (what CI and the recipe assert)

`tests/test_v091.py::test_export_tokenizer_json_contract` validates the
loadable shape: the JSON parses, `model.type == "WordLevel"`, the vocab
size equals the book size, every id round-trips, and the decode
contract holds on a fresh ids stream. If the `transformers`/`tokenizers`
package is installed, load the file through
`tokenizers.Tokenizer.from_str` and decode — the same path vLLM's
tokenizer layer takes (skipped when the package is absent; the JSON
contract is the required half).

## The day-1 engine matrix

| engine | support | path |
| --- | --- | --- |
| HF transformers | proven (the tokenizers layer it wraps) | `tokenizers.Tokenizer.from_file` — decode byte-exact (`research/integration_proofs.json`) |
| vLLM / SGLang | proven (the tokenizers layer both call) | pre-encode with ratec, feed `prompt_token_ids`; decode via the vocab |
| llama.cpp | proven (metadata subset) | `export_gguf`; the GGUF carries vocab + the snapshot; llama.cpp must NOT re-tokenize |
| plain tokenizers | proven | same file as transformers |

The GGUF snapshot rides as ONE base64 string (`ratec.snapshot.v8`,
sha256-bound by `ratec.snapshot.sha256` over the raw bytes):

```python
import base64, gguf                      # gguf-py ships with llama.cpp
r = gguf.GGUFReader("ratec-c20.gguf")
kv = r.fields["ratec.snapshot.v8"]
raw = bytes(kv.parts[kv.data[0]])
eng = ratec.RateC.from_state(base64.b64decode(raw))   # continues bit-exact
```

(Why a string and not a u8 array: gguf-py walks arrays PER ELEMENT —
the rc1 form, a ~5M-element array, was 5M recursive calls and an OOM
on a 3 GiB box. One string is one part in every reader.)

## The documented subset

| capability | status |
|---|---|
| vocab table (id → token bytes) via tokenizer.json | validated round-trip |
| decode of ratec ids through HF tokenizers | validated round-trip |
| encode via vLLM/HF tokenizers | FORBIDDEN (documented above) |
| GGUF carriage (vocab + snapshot) | `export_gguf` — metadata subset; llama.cpp must not re-tokenize |
| training-side (BPB, servos) | ratec-only; no bridge claim |
