# v1.0.0rc3 — the standardization candidate

The rc that makes ratec FEEL like the most popular BPE tokenizers —
as easy to pick up as tiktoken, installable straight from GitHub, and
documented end to end.

## The tiktoken-grade front door (additive surface)

- `ratec.get_ratec(c=20, pf=-1, store_book=True, cache=True)` — the
  `get_encoding` analogue: one call, one working tokenizer, cached per
  config. The cached engine is SHARED and adaptive (right for infra
  serving one model); `cache=False` for a private engine.
- `ratec.load(source, cache=True)` — the infra one-liner: any saved
  state from bytes, a path, or an http(s) URL. URL fetches land in
  `RATEC_CACHE` (else `XDG_CACHE_HOME/ratec`, else `~/.cache/ratec`);
  loads are memoized in-process keyed by (source, size, mtime) so
  per-request `load()` does not re-read. Continues bit-exactly.
- `RateC.train(text_or_bytes)` — the explicit one-time bulk-adapt
  call; encode IS the adaptation, `train` names the one big pass
  before decode-exact use. Returns self.
- `RateC.n_vocab` — the tiktoken-facing name for the adaptive book
  size (0 on a cold engine; pinned by save_state).

All four are ADDITIVE: the frozen core (byte semantics, snapshot v8,
the exact lane) is untouched, the manifest records the new surface
(21 methods / 1 classmethod / 11 functions), and
`tests/test_ergonomics.py` arrives the way a tiktoken user would —
train, encode/decode, unicode round-trips, save/load continuation,
URL fetch — all passing.

## Install straight from GitHub — verified

`pip install git+https://github.com/vidsix/ratec@v1.0.0rc2` was
executed in a CLEAN venv this session: build, install, and the
encode/decode smoke all pass. README documents all three install
paths (git+https, release wheels with the real asset URLs, clone);
the wheels pipeline auto-attaches cp39-cp312 × x86_64/aarch64
manylinux2014 artifacts to this release, as it did for rc2.

## The project, cleaned and documented

- `docs/QUICKSTART.md` (new): the first ten minutes — install, the
  tiktoken-shaped flow, reproducible config via `solve_c`, the engine
  bridges (HF / vLLM / SGLang / llama.cpp), the LM data-prep lane.
- README opens with the quickstart and the one conceptual difference
  from BPE: a ratec engine is a state, not a static table.
- The nine historical release notes moved from the repo root to
  `docs/releases/` (nothing referenced them; history preserved).

## Gates (executed at this tag)

make test green, full pytest green (74 tests incl. the 10 new
ergonomics tests), tables drift PASS, all seven LEAN files no-sorry,
floors gate PASS against the standing envelope, the integration proof
PASS (tokenizers decode byte-exact, official GGUF reader OK, vocab ==
book, snapshot continuation bit-exact), provenance 22/22, freeze
audit re-run at 1.0.0rc3 — the additions are additive; the frozen
core is identical to the v0.9.3 freeze.

## Known-open for 1.0.0

- The nightly-green ≥ 1 week clock (STABILITY.md step 8) — CI has
  been green on main since the rc1 repair; the rc2 and rc3 tags join
  the same green window.
