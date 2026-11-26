# The stability policy (road to 1.0, then the 1.x line)

This document is the versioning contract. It is written BEFORE 1.0 so
the freeze is explicit, and the 1.0 freeze audit executes it. Anything
promised here is machine-checked; anything not promised is stated as
not-promised.

## What is frozen at 1.0 (within 1.x)

1. **The exact lane's outputs, byte-for-byte.** For a fixed book state
   and fixed C, `encode_bytes` / `encode_to_spans` / `surprise` /
   `encode_stream_parallel` (both spans and ids) / `decode_bytes` /
   `decode_batch` produce identical bytes on every supported platform
   and every endianness (v0.9.1's BE-host simulation gate). The ids are
   content-addressed and order-stable (S5/S6 proofs); the tiling is
   correct for ANY surprisal function (T2).
2. **Snapshot format v8, read-compatible forever.** v2…v8 payloads
   load in 1.x; the writer stays v8; prune-presence is the header flag
   (offset 42); load classifies invalid vs wrong-version with the
   coded errors (S9). A NEW snapshot version is a MAJOR release.
3. **The public API surface** — `research/api_manifest.json` is the
   freeze: module functions with full signatures (names, kinds,
   defaults), the `RateC` constructor and every public method, the
   `RatecErrorCode` taxonomy (8 codes), `RatecError`'s public
   attributes, `__all__`, and the C extension's entry points. The
   drift test (`tests/test_api_freeze.py`) fails on any difference.
   The manifest changes only in a visible regeneration commit.
4. **The perf floor RULE** (not the numbers — the rule): every ratec
   lane holds ≥ 0.90 of its quiet-box-envelope median (lanes with
   envelope median ≥ 10 ms) or ≥ 0.70 (sub-millisecond lanes, where
   fixed call overhead is a ±15% class). The envelope is re-recorded
   with the full quiet-box protocol at every minor release
   (`research/quietbox_envelope.json`); `bench/perf_floors.py` is the
   between-release tripwire.
5. **The coded error taxonomy**: the 8 codes and which conditions map
   to which code do not change meaning within 1.x (new codes are
   additive, a MINOR).
6. **The record protocol**: `research/provenance_1.0.json` pins the
   sha256 of every contract artifact. Re-running `bench/provenance.py`
   and diffing against the committed record is the freeze audit — a
   changed hash is a contract-artifact change and MUST be a reviewable
   commit.

## What is explicitly NOT frozen

- **The fast lane** (`exact=False`): its outputs are NOT stable across
  versions and may differ from the exact lane's segmentation. Its only
  promise is STRUCTURAL and machine-checked for any surprisal function
  (tiling, round-trip, intern ids — T2/S5/S6). The parity bound
  (surprisal deviation ≤ 5e-07, `research/fastlane_parity.json`) is its
  tripwire: a parity failure flips the v0.9.2 KEEP decision to REMOVE.
- **Absolute perf numbers** — host-dependent by nature; only the floor
  ratios are contract.
- **Research record numbers** (twins, ladders, probe sweeps) — they are
  recorded measurements, not guarantees; regenerated records may differ
  within their protocol's noise.
- **Docstrings and documentation content.**

## Versioning

SemVer, applied to the frozen surface:

- **MAJOR** — any frozen-surface change: a new snapshot write version,
  an API signature change or removal, an error-code meaning change, a
  floor-rule change, an exact-lane output change (this one should be
  impossible and would carry a full provenance investigation).
- **MINOR** — additive only: new functions/methods, new optional
  parameters (defaults preserve old behavior), new error codes, new
  proof files, new records.
- **PATCH** — fixes with no frozen-surface change.

**Deprecation path**: the replacement ships additively first; the old
path is documented-deprecated for at least one minor; removal happens
at the next MAJOR with a CHANGELOG migration note. Deprecations are
visible in the manifest diff (the old signature stays until removal).

## The 1.0 freeze audit (release checklist)

1. `python bench/api_freeze.py` → diff `research/api_manifest.json` —
   MUST be empty unless the release is the deliberate MAJOR/MINOR.
2. `python -m pytest tests/test_api_freeze.py -q` — green.
3. `python bench/provenance.py` → diff `research/provenance_1.0.json` —
   every changed hash is explained in the release notes.
4. `python bench/perf_floors.py` — PASS.
5. The full quiet-box re-record — the new envelope becomes the floor
   reference for the next cycle.
6. All seven LEAN files compile no-sorry on the pinned toolchain.
7. `make test` + full pytest green on x86_64 and aarch64 CI.
8. The memory-safety proof gate is green AT the release tag:
   `bench/sanitizer_gate.py` says SAN-GO — zero ASan errors, zero UBSan
   runtime errors, and non-scaling LSan leaks across the torture scales
   — over the full test suite, the integration proofs, and the
   15k-mutant blitz, with its machine record committed as
   `bench/sanitizer_record.json` on the tagged line. The nightly soak
   stays scheduled as a free tripwire; the gate reports its streak but
   no longer gates on it. (Amended 2026-11-26: per-path LSan proof
   replaces the original "≥ 1 green week" — the criterion's purpose,
   no leaks at 1.0.0, is served more directly. See docs/PROOF_STATUS.md.)
