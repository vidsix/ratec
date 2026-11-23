# v1.0.0rc2 — the day-1 support candidate

The rc that makes ratec models loadable by the STANDARD tooling on day
1 — and removes the one blocker the rc1 integration proof uncovered.

## The GGUF carriage is now readable by the standard reader (the fix)

The rc1 proof recorded that the official `gguf` package (llama.cpp's
own Python reader, gguf-py) OOM-killed a 3 GiB box on our 5 MB export.
Root cause FOUND: **gguf-py walks every array PER ELEMENT** — `for idx
in range(alen)` recursing per element with its own numpy object. Our
snapshot rode as a ~5M-element u8 array: 5M recursive calls. The file
was structurally valid GGUF; the ecosystem's reader just cannot carry
large element arrays.

The fix: `ratec.snapshot.v8` rides as **ONE base64 string** — a single
length-prefixed part in every GGUF reader, pure ASCII, byte-exact,
sha256-bound by `ratec.snapshot.sha256` over the raw snapshot bytes.
Verified: the official gguf package loads the 13 MB export (23k vocab
+ 10 MB snapshot) under a 2 GiB memory cap, and the proof records
`gguf_official_reader_ok: True` as a first-class check. The snapshot
v8 FILE format is untouched (this is the container's value encoding,
not the frozen engine state). Decode recipe:

```python
import base64, gguf                      # gguf-py ships with llama.cpp
r = gguf.GGUFReader("ratec-c20.gguf")
kv = r.fields["ratec.snapshot.v8"]
eng = ratec.RateC.from_state(base64.b64decode(bytes(kv.parts[kv.data[0]])))
```

## The day-1 engine story, written down

`docs/vllm_recipe.md` gains the day-1 matrix:

| engine | support | path |
| --- | --- | --- |
| HF transformers | proven (the tokenizers layer it wraps) | `tokenizers.Tokenizer.from_file` — decode byte-exact |
| vLLM / SGLang | proven (the same tokenizers layer) | pre-encode with ratec, feed `prompt_token_ids`; decode via the vocab |
| llama.cpp | proven (metadata subset) | `export_gguf`; re-tokenization forbidden |
| plain tokenizers | proven | same file as transformers |

README gains the release-wheel install commands (cp39–cp312 ×
x86_64/aarch64 manylinux2014): infra loads ratec without a compiler.

## Gates (executed at this tag)

make test green, full pytest green (64 tests), tables drift PASS, all
seven LEAN files no-sorry, floors gate PASS against the rc1 envelope,
the integration proof PASS (tokenizers decode byte-exact, official
GGUF reader OK, vocab == book, snapshot continuation bit-exact),
provenance 22/22 with the audit re-run at 1.0.0rc2 — **zero API
drift** (9 functions / 20 methods / the 8-code taxonomy / 6 C entry
points, identical to the v0.9.3 freeze).

## Known-open for 1.0.0

- The nightly-green ≥ 1 week clock (STABILITY.md step 8) — CI has been
  green on main since the rc1 repair; the nightly cadence continues.
- The wheels release job ships with its `contents:write` fix from the
  rc1 session — this tag's artifacts attach automatically for the
  first time.
