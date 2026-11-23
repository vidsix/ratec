# v0.9.3 — the rc-readiness release

Everything the 1.0rc gate list needs, delivered as executable
artifacts — the freeze stops being a promise and becomes tests,
records, and a pipeline.

## The freeze, made executable

- **The API freeze audit.** `bench/api_freeze.py` introspects the
  ENTIRE public surface into `research/api_manifest.json`: 9 module
  functions, the `RateC` constructor with its 12 parameters
  (names/kinds/defaults, annotations as strings so the manifest is
  interpreter-stable), 20 methods + `from_state` + the properties, the
  8-code `RatecErrorCode` taxonomy, `RatecError`'s public attributes,
  `__all__`, and the 6 C extension entry points — each with provenance.
  `tests/test_api_freeze.py` regenerates the manifest in memory and
  asserts exact equality: any rename, signature change, default
  change, or entry-point change fails the suite. A deliberate API
  change is a visible manifest-regeneration commit under the rules of
  `docs/STABILITY.md`.
- **The perf-floors gate.** `bench/perf_floors.py` re-measures all
  nine ratec lanes with the envelope's exact lane semantics (min of
  round medians, round-robin interleave, same 4 MiB stream) and holds
  each to a TIERED floor: ≥ 0.90 of the quiet-box envelope for lanes
  with median ≥ 10 ms, ≥ 0.70 for sub-millisecond lanes — where ~50 µs
  of fixed Python/numpy call overhead is a ±15% class (the same
  envelope's two decode lanes measure 10.7 and 14.3 GB/s for the same
  work; encoding that measurement class into the rule is the honest
  form). First run: **all nine lanes PASS** — encode lanes
  0.99–1.22 of envelope, decode_bytes 0.845 (floor 0.70), decode_batch
  0.970.
- **The 1.0 provenance record.** `bench/provenance.py` hashes all 19
  contract artifacts — the seven LEAN proof files (T2, S1, S5/S6, S7,
  S9, S8, the scaffold), the quiet-box envelope, the floor check, the
  8 GiB scale gate, the fast-lane parity and KEEP-decision records, the
  per-order DECLINED record, the API manifest, the first soak record,
  both fuzz batteries, and the CI workflows — each with its sha256,
  byte size, what it establishes, and HOW it is reproduced. The audit:
  re-run and diff against the committed record; a changed hash is a
  contract-artifact change, a reviewable commit, never silent drift.

## The policy and the cookbook

- **`docs/STABILITY.md`** — the versioning contract, written BEFORE
  1.0 so the freeze is explicit: the six frozen things (exact-lane
  byte-freeze, snapshot v8 read-compat, the API manifest, the floor
  rule, the error taxonomy, the record protocol), the four explicitly
  unfrozen (the fast lane with its 5e-07 parity tripwire — a failure
  flips the KEEP decision to REMOVE; absolute perf numbers; research
  records; docstrings), the SemVer mapping, the deprecation path, and
  the 8-step release checklist.
- **`docs/COOKBOOK.md`** (P3.4) — ten recipes grounded in the frozen
  signatures: the basic loop, parallel tokenization, batch files,
  multi-GiB prep (the uint32 span-space limit documented honestly —
  one engine covers at most 4 GiB of lifetime stream), rate control,
  books/transplants (`to_ranks()` + `from_state`; `book_dump` is NOT
  public — the recipes use what ships), snapshots/pruning, the
  ecosystem bridges, the error taxonomy, maintenance. **Every recipe
  was executed against the current build before commit**: solve_c hit
  an exact 250-patch target, the piece-wise transplant tokenizes, the
  snapshot continuation is bit-exact, and `SNAPSHOT_INVALID` maps to
  the right code.

## The wheel pipeline

`.github/workflows/wheels.yml`, triggered on `v*` tags: the sdist job
(plus a wheel built FROM the sdist — the provenance check),
cibuildwheel manylinux wheels (cp39–cp312-abi3) on x86_64 AND aarch64
(`ubuntu-24.04-arm` — the architecture the CI gate suite already proves
green), per-wheel encode/decode smoke tests, and automatic attachment
to the GitHub release. Verified locally end-to-end: the sdist contains
the C source and build files, a wheel builds FROM the sdist, installs
into a clean venv, and round-trips byte-exactly. The aarch64 BUILDS run
on GitHub's arm runners — this x86_64 sandbox cannot cross-compile
them; the workflow is the deliverable and the release uses it.

## Housekeeping

The branch push law caught a silent failure this cycle: three milestone
commits had landed on the previous release's merge branch while the
working ref sat stale at v0.9.2 — the commits were safe locally and
recovered by pushing HEAD explicitly; remote state is now verified by
ls-remote after every push.

## Gates

make test green (suite + bit-exactness + fuzz + snapfuzz + tables),
full pytest green (64 tests: the 62 of v0.9.2 + the two freeze-audit
tests), tables drift PASS, all seven LEAN files no-sorry, the
perf-floors gate PASS on all nine lanes, the provenance record covers
19/19 artifacts with zero missing.
