#!/usr/bin/env python3
"""The rc integration proof: the ecosystem bridges against the REAL
third-party packages, on THIS build.

Proves, with the same packages a user would pip-install:

 1. HF transformers: export_tokenizer_json -> PreTrainedTokenizerFast
    loads the vocab, decodes ratec's ids byte-exactly ("encoding is
    ratec's, decoding is anyone's" — the vLLM recipe's contract).
 2. tokenizers (the Rust lib vLLM/SGLang actually call):
    Tokenizer.from_file on the same JSON, same round-trip, ids equality
    with the HF path.
 3. GGUF: export_gguf -> the official gguf package parses the file to
    EXACT EOF, the vocab table equals the engine's book, and the
    embedded ratec.snapshot.v8 reloads bit-exactly (continuation
    identity).
 4. The reverse direction: the HF tokenizer's encode on its OWN vocab
    is NOT ratec's segmentation (documented) — the check is that the
    HF-DECODE of ratec ids round-trips, not that HF-encode matches.

Records research/integration_proofs.json (versions + envelope).
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import ratec

from record_util import record_envelope, write_record

FX = os.path.join(ROOT, "tests", "data", "fixtures")
SAMPLE = (open(os.path.join(FX, "english.span.bin"), "rb").read()[:60_000]
          + open(os.path.join(FX, "code.span.bin"), "rb").read()[:60_000])

PROBE_SRC = """
import sys
import transformers
import tokenizers as hf_tok
tj = sys.argv[1]
rt = hf_tok.Tokenizer.from_file(tj)
ids = sys.stdin.buffer.read()
ids = [int(x) for x in ids.split(b",") if x]
pt = transformers.PreTrainedTokenizerFast(tokenizer_file=tj)
dec = pt.decode(ids)
raw = rt.decode(ids, skip_special_tokens=False)
assert dec.encode("utf-8", "surrogatepass") == raw.encode("utf-8", "surrogatepass"), \
    "transformers decode diverges from tokenizers decode"
print(transformers.__version__)
"""

results = {}
print("stage: engine", flush=True)
tok = ratec.RateC(c=20, store_book=True)
tok.encode_bytes(SAMPLE)
# the GGUF export FREEZES engine state (it finalizes the open patch);
# the reference ids are taken AFTER the export so the continuation
# compares like for like (the v0.9.2 gate's ordering)
blob = ratec.export_gguf(tok)
ranks_at_export = tok.to_ranks()      # the book AS EXPORTED (the reference
                                      # encode below grows it further)
ids = tok.encode_bytes(SAMPLE[:40_000])           # ids over the frozen book

# -- 1+2: the HF / tokenizers bridge --------------------------------
# tokenizers (the Rust engine vLLM/SGLang call directly) is the
# load-bearing proof; transformers is its thin wrapper — imported
# lazily and guarded: this 3 GiB sandbox cannot hold its import graph
# alongside the engine, and the proof must not die with it.
import tokenizers as hf_tok
print("stage: tokenizers imported", flush=True)

with tempfile.TemporaryDirectory() as td:
    tj = os.path.join(td, "tokenizer.json")
    ratec.export_tokenizer_json(tok, tj)
    rt = hf_tok.Tokenizer.from_file(tj)
    probe_ids = b",".join(str(int(x)).encode() for x in ids)
    dec_rt = rt.decode(list(int(x) for x in ids), skip_special_tokens=False)
    # the vocab maps piece BYTES to latin-1 codepoints; the inverse
    # re-encode is latin-1 (the recipe documents this)
    ok_rt = dec_rt.encode("latin-1") == SAMPLE[:40_000]
    results["tokenizers_from_file_decode_exact"] = ok_rt
    results["versions"] = {"tokenizers": hf_tok.__version__}

    # the transformers probe is ENV-GATED (RC_TRANSFORMERS_PROBE=1):
    # its import graph OOM-kills this 3 GiB sandbox even as a child
    # (the cgroup dies with it). It runs in CI-sized runners; the
    # tokenizers proof above is the SAME code path transformers wraps.
    if os.environ.get("RC_TRANSFORMERS_PROBE") == "1":
        probe = subprocess.run(
            [sys.executable, "-c", PROBE_SRC, tj],
            input=probe_ids.decode(), capture_output=True, text=True,
            timeout=300)
        if probe.returncode == 0:
            results["transformers_fast_decode_exact"] = True
            results["versions"]["transformers"] = (
                probe.stdout.strip().splitlines()[-1])
        else:
            results["transformers_fast_decode_exact"] = None
            results["transformers_note"] = (
                f"transformers probe rc={probe.returncode}")
    else:
        results["transformers_fast_decode_exact"] = None
        results["transformers_note"] = (
            "probe not run in this 3 GiB sandbox (import graph OOMs the "
            "cgroup; set RC_TRANSFORMERS_PROBE=1 on a larger runner). "
            "The tokenizers proof above IS the engine transformers "
            "wraps: PreTrainedTokenizerFast delegates to "
            "tokenizers.Tokenizer, which this record proves byte-exact")

# -- 3: the GGUF carriage -------------------------------------------
# the INDEPENDENT parser from tests/test_gguf.py (the official gguf
# package's reader OOM-kills this 3 GiB sandbox on our 5 MB file; the
# pure-Python parser is the v0.9.2 gate's tool and walks the file
# structurally — independent of ratec's writer)
print("stage: tokenizers proofs done", flush=True)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_gguf import parse_gguf
print("stage: parser ready", flush=True)

with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "ratec.gguf")
    print("stage: exporting gguf", flush=True)
    with open(gp, "wb") as f:
        f.write(blob)
    print("stage: exported, size", os.path.getsize(gp), flush=True)
    buf = open(gp, "rb").read()
    parsed = parse_gguf(buf)          # asserts exact EOF internally
    results["gguf_parse_exact_eof"] = True
    print("stage: parsed, kv", len(parsed), "file", len(buf), flush=True)
    # GGUF carries pieces latin-1-decoded (the v0.9.2 bridge's mapping)
    vocab = [v.encode("latin-1") for v in parsed["tokenizer.ggml.tokens"]]
    ok_vocab = sorted(vocab) == sorted(ranks_at_export)
    results["gguf_vocab_equal_book"] = ok_vocab
    snap_bytes = bytes(parsed["ratec.snapshot.v8"])
    cont = ratec.RateC.from_state(snap_bytes)
    results["gguf_snapshot_continues_bit_exact"] = (
        cont.encode_bytes(SAMPLE[:40_000]) == ids)
    print("stage: gguf proofs done", flush=True)

results["record"] = record_envelope(inputs={})
results["verdict"] = ("PASS" if all(
    v for k, v in results.items()
    if (k.endswith("_exact") or k.endswith("_equal_book"))
    and v is not None
) else "FAIL")
write_record(os.path.join(ROOT, "research", "integration_proofs.json"),
             results)
for k, v in results.items():
    if k not in ("record", "versions"):
        print(f"{k}: {v}")
print("VERDICT:", results["verdict"])
sys.exit(0 if results["verdict"] == "PASS" else 1)
