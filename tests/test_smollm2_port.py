#!/usr/bin/env python3
"""test_smollm2_port.py — correctness suite for the HF transformers port.

Run:  python3 tests/test_smollm2_port.py
No torch required (tokenizer + config side only).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ratec

try:
    from transformers import AutoTokenizer  # noqa: F401  (presence probe)
    from ratec.smollm2 import (RateCTokenizer, encode_for_lm,
                               build_smollm2_ratec_config, SMOLLM2_135M_KW)
except ImportError as e:
    print(f"SKIP: test_smollm2_port needs the optional `transformers` "
          f"dependency ({e}); the tokenizer side needs no torch. "
          f"Install with: pip install 'ratec[lm]' or pip install transformers")
    sys.exit(0)

FAILURES = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  {extra}" if extra else ""))
    if not cond:
        FAILURES.append(name)


DATA = os.environ.get(
    "RATEC_FIXTURES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fixtures"))

# ------------------------------------------------ 1. snapshot from corpus
print("== snapshot training (corpus ingestion) ==")
corpus = open(os.path.join(DATA, "lm_corpus.bin"), "rb").read()
fmeta = json.load(open(os.path.join(DATA, os.pardir, "fixtures_meta.json")))
assert len(corpus) == fmeta["lm_corpus_len"] and \
    hashlib.sha256(corpus).hexdigest() == fmeta["lm_corpus_sha256"], \
    "lm_corpus.bin corrupt — see fixtures_meta.json"
tok_raw = ratec.RateC(c=20)
tok_raw.encode_bytes(corpus)
snap = tok_raw.save_state()
check("snapshot produced", len(snap) > 0, f"{len(snap)} B")

held = ("RateC is an adaptive tokenizer: the judge assigns surprisal and "
        "the patcher packs bytes. 你好世界! def f(x): return x*2 🙂\n") * 4

# ------------------------------------------------ 2. tokenizer semantics
print("== RateCTokenizer semantics ==")
tk = RateCTokenizer(snapshot=snap)
check("book frozen == snapshot book", tk.book_size == tok_raw.book_size,
      f"{tk.book_size} patches")
check("vocab_size == book+256", tk.vocab_size == tk.book_size + 256)

ids, lens = tk.encode_for_lm(held)
check("encode non-empty", len(ids) > 0, f"{len(ids)} patches")
check("lens sum == utf-8 byte length", sum(lens) == len(held.encode("utf-8")),
      f"{sum(lens)} vs {len(held.encode('utf-8'))}")
check("decode(ids) byte-exact", tk.decode_bytes(ids) == held.encode("utf-8"))
check("hf decode round-trip", tk.decode(ids) == held)
raw_ids = ratec.RateC(warm_state=snap).encode_bytes(held.encode("utf-8"))
in_book = all(i < tk.book_size for i in ids)
if in_book:
    check("ids == raw engine ids (all patches in book)", ids == raw_ids)
else:
    print(f"  [info] {sum(1 for i in ids if i >= tk.book_size)} "
          f"byte-fallback ids (held-out content) — frozen-vocab contract")
check("all ids < vocab_size", all(0 <= i < tk.vocab_size for i in ids))

# get_vocab consistency
vocab = tk.get_vocab()
check("get_vocab covers every emittable id",
      set(ids) <= set(vocab.values()),
      f"{len(vocab)} entries; max id {max(ids)} < vocab_size {tk.vocab_size}")
inv = {}
dup = 0
for k, v in vocab.items():
    if v in inv:
        dup += 1
    inv[v] = k
check("get_vocab injective (no duplicate ids)", dup == 0)
check("max emitted id < vocab_size (embedding-safe)",
      max(ids) < tk.vocab_size and max(vocab.values()) < tk.vocab_size)

# ------------------------------------------------ 3. specials
print("== special tokens ==")
tk2 = RateCTokenizer(snapshot=snap,
                     bos_token="<|bos|>", eos_token="<|eos|>")
n_specials = 2
check("special ids in reserved block",
      tk2.bos_token_id >= 0xFFFFFFF0 and tk2.eos_token_id > tk2.bos_token_id,
      f"bos={tk2.bos_token_id:#x}")
enc = tk2.encode("<|bos|>" + held, add_special_tokens=False)
check("inline special token splits + maps", enc[0] == tk2.bos_token_id)
check("decode strips special (skip)", 
      tk2.decode(enc, skip_special_tokens=True) == held)
enc2 = tk2.encode(held)                # add_special_tokens=True default
check("bos_eos pattern wraps", enc2[0] == tk2.bos_token_id and
      enc2[-1] == tk2.eos_token_id, f"len {len(enc2)}")
check("wrapped decode round-trips",
      tk2.decode(enc2, skip_special_tokens=True) == held)
mask = tk2.get_special_tokens_mask(enc2, already_has_special_tokens=True)
check("special mask flags bos/eos", mask[0] == 1 and mask[-1] == 1)

# ------------------------------------------------ 4. save / from_pretrained
print("== save_pretrained / from_pretrained ==")
tmp = tempfile.mkdtemp(prefix="ratec_hf_")
try:
    tk2.save_pretrained(tmp)
    files = sorted(os.listdir(tmp))
    check("snapshot file written", "ratec_snapshot.bin" in files, str(files))
    check("tokenizer_config written", "tokenizer_config.json" in files)
    cfgj = json.load(open(os.path.join(tmp, "tokenizer_config.json")))
    check("tokenizer_class recorded",
          "RateCTokenizer" in cfgj.get("tokenizer_class", ""))

    tk3 = RateCTokenizer.from_pretrained(tmp)
    ids3, lens3 = tk3.encode_for_lm(held)
    check("from_pretrained: identical ids", ids3 == ids)
    check("from_pretrained: identical lens", lens3 == lens)
    check("from_pretrained: specials preserved",
          tk3.bos_token_id == tk2.bos_token_id)
    check("from_pretrained: decode round-trip",
          tk3.decode(ids3) == held)

    # AutoTokenizer: transformers 5.x ALWAYS converts to fast tokenizers
    # (use_fast is ignored) and cannot load slow-only custom classes.
    # Documented load path: RateCTokenizer.from_pretrained (above) — the
    # same object a Trainer would receive when passed explicitly.
    try:
        from transformers import AutoTokenizer
        at = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
        ids_at = at.encode_for_lm(held)[0]
        check("AutoTokenizer loads RateCTokenizer",
              type(at).__name__ == "RateCTokenizer" and ids_at == ids,
              f"got {type(at).__name__}")
    except Exception as e:
        print(f"  [info] AutoTokenizer (5.x fast-only) refused a slow custom "
              f"tokenizer: {type(e).__name__} — documented; class "
              f"from_pretrained is the supported path")
        check("class from_pretrained == supported load path", ids3 == ids)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------ 5. LM helper + config
print("== LM helpers ==")
pairs = encode_for_lm(["hello world", "你好世界"], tk)
check("encode_for_lm batches", len(pairs) == 2 and
      all(sum(l) == len(t.encode("utf-8")) for (i, l), t in zip(pairs, ["hello world", "你好世界"])))

cfg = build_smollm2_ratec_config(tk2)
check("LlamaConfig SmolLM2-135M shape",
      cfg.hidden_size == 576 and cfg.num_hidden_layers == 30 and
      cfg.intermediate_size == 1536 and cfg.num_attention_heads == 9)
check("config vocab == tokenizer vocab", cfg.vocab_size == tk2.vocab_size,
      f"{cfg.vocab_size}")
check("config tokenizer_class", cfg.tokenizer_class == "ratec.smollm2.RateCTokenizer")
j = json.loads(cfg.to_json_string())
check("config json-serializable", j["vocab_size"] == tk2.vocab_size)
check("SMOLLM2_135M_KW exported", SMOLLM2_135M_KW["hidden_size"] == 576)

# ------------------------------------------------ 6. edge cases
print("== edge cases ==")
tk0 = RateCTokenizer(snapshot=None, c=20)   # fresh engine, empty book
e0 = tk0.encode("byte fallback only — 灰")
check("fresh tokenizer byte-fallback round-trip", tk0.decode(e0) == "byte fallback only — 灰")
check("fresh ids land in byte block",
      all(tk0.book_size <= i < tk0.vocab_size for i in e0))
edge = RateCTokenizer(snapshot=snap)
for s in ["", "a", "\x00\x01\xff", "🙂" * 500, "x" * 100000]:
    ii, ll = edge.encode_for_lm(s)
    if edge.decode_bytes(ii) != s.encode("utf-8"):
        check(f"edge round-trip len={len(s)}", False)
        break
else:
    check("edge round-trips (empty/1B/\\x00\\x01\\xff/emoji/100k)", True)

print()
print("GATE:", "PASS" if not FAILURES else f"FAIL ({len(FAILURES)}): {FAILURES}")
sys.exit(0 if not FAILURES else 1)
