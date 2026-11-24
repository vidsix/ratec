# v1.0.0rc1 — the release candidate

The candidate for the actual ratec: the freeze audit executed AT the
tag, the speed records that define the performance story, and the
ecosystem integration proof that caught a real detokenization bug on
its way in.

## The freeze audit (executed per docs/STABILITY.md)

- **ZERO API drift.** `api_freeze` regenerated at 1.0.0rc1: 9 module
  functions, the RateC constructor + 20 methods + `from_state`, the
  8-code error taxonomy, 6 C entry points — IDENTICAL to the v0.9.3
  freeze. The only manifest delta is the recorded version. The 1.0
  surface is now formally frozen.
- **Provenance diff, every hash explained.** The record covers 22
  artifacts (three new: `integration_proofs.json`,
  `parallel_scaling.json`, the gigatoken record). The changed hashes:
  the re-recorded envelope (the new floor reference per checklist step
  5), the regenerated manifest, and the three new records. Nothing
  changed silently.
- The quiet-box envelope re-recorded on this build:
  encode_bytes_warm 18.67 MB/s, exactness 4/4 domains, ids/spans warm
  99.7%. Floors gate PASS against it.

## The speed story (all measured on this build)

- **-fno-math-errno** (the exact lane): the per-byte judge calls
  -log2(pm) once per byte; the flag drops ONLY glibc's errno
  bookkeeping — the computed value is bit-identical by spec, and the
  bit-exactness gates confirm the outputs. +5% on the warm encode lane
  (18.0 → 18.9 MB/s); every other lane within noise.
- **The parallel-scaling record** (`research/parallel_scaling.json`):
  one engine, threads 1/2/4/8, the S6 ids-invariance asserted every
  rep (the contract holds at every thread count). On this 2-vCPU box
  the curve plateaus at threads=2 (+13%) — the record is
  host-annotated via the record envelope; the fleet shape is
  chunked_prep at threads × N processes.
- **The gigatoken record** (`bench/bench_gigatoken.json`): 1 GiB
  sustained — 7.8 MB/s spans / 12.17 MB/s judge serial, and the fleet
  arm: chunked_prep over the whole GiB with a counting `sink=` (flat
  RSS — per-segment results stream, nothing retained; 64 MiB segments
  after the 256 MiB config OOM'd the 3 GiB sandbox), **9.94 MB/s /
  1.13 Mtok/s on 2 vCPUs**. The earlier 0.0 Mtok/s bug: chunked_prep
  returns a list of per-segment dicts, not a flat array — the cookbook
  recipe was wrong the same way and is fixed.

## The integration proof — and the bug it caught

`bench/integration_proofs.py` → `research/integration_proofs.json`,
**PASS**:

- **tokenizers.Tokenizer.from_file decodes ratec ids byte-exact** —
  the exact path vLLM and SGLang call (`vocab[latin1(piece)] == id`,
  the documented subset).
- **GGUF**: the independent pure-Python parser (the official gguf
  package's reader OOM-kills a 3 GiB box on our 5 MB file — the repo's
  parser is the v0.9.2 gate's tool) walks the file to EXACT EOF,
  vocab == book, and the embedded snapshot.v8 reloads and CONTINUES
  BIT-EXACT (continuation identity).
- **transformers**: its own probe is env-gated
  (RC_TRANSFORMERS_PROBE=1) — the import graph exceeds this sandbox's
  3 GiB even as a child process. The record documents that the
  tokenizers proof IS the code transformers wraps
  (PreTrainedTokenizerFast delegates to tokenizers.Tokenizer).

**The bug the proof caught**: with `decoder: null` in tokenizer.json,
HF tokenizers / vLLM / transformers JOIN DECODED PIECES WITH SPACES —
200 ratec ids decoded to 2065 bytes. Every model bridged through
tokenizer.json would have silently corrupted detokenization. The
export now carries an identity Replace decoder (never matches; the
piece join becomes "") and `rt.decode(ids).encode("latin-1")`
round-trips exactly. The v0.9.1 contract test passed through the raw
JSON — which never inserts spaces — so the suite stayed green while
the real engines were broken; a bridge-level proof with the real
packages is what found it. Compatibility: test_v091 4/4, test_gguf
2/2.

The GGUF proof also fixed an ordering trap: the export FREEZES engine
state (it finalizes the open patch) — reference ids must be taken
AFTER the export, or the continuation check compares different states.

## CI repair (the rc cannot ship on red CI)

GitHub Actions was RED on main since v0.9.2's push — four failures,
all diagnosed from the Actions logs and fixed:

1. **lean active-sorry gate**: the module doc headers SAY "with no
   sorry" — the gate stripped line comments but not block docs.
2. **test_stream_ids_share_vocabulary**: a +10_000 fixed bound on a
   book-size assertion that is platform-marginal (macOS: 365419 vs
   332834+10000, ~9% swing from map-growth timing) — now a
   proportional bound; test-only, the engine untouched.
3. **wheels sdist**: setup-python 3.12 no longer bundles setuptools.
4. **wheels selectors**: `cp39-abi3-manylinux_*` is not a valid
   cibuildwheel identifier — cp39..cp312-manylinux_{x86_64,aarch64}.

## Known-open for 1.0.0 (honest)

- **The nightly-green ≥ 1 week clock** (STABILITY.md checklist step 8)
  restarted at the CI repair — the rc exists precisely to gather that
  window. The first soak record (198 rounds, flat RSS, PASS) stands;
  the nightly workflow now runs on green CI.
- The aarch64 manylinux WHEELS build on GitHub's arm runners (the
  wheels workflow fires on this tag); this x86_64 sandbox cannot
  cross-compile them — the CI aarch64 job proves the gate suite green
  on the architecture.

## Gates

make test green, full pytest green (64 tests), tables drift PASS, all
seven LEAN files no-sorry on the pinned toolchain, floors gate PASS
against the new envelope, the integration proof PASS, provenance 22/22
with every delta explained in this document.
