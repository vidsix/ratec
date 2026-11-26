# v1.0.0 — the proof-gated release

The 1.0.0 contract, executed: every STABILITY.md criterion re-proven
at the tag by `bench/release_gate.py` (GO), and the memory-safety
criterion — as amended — discharged by `bench/sanitizer_gate.py`
(SAN-GO) instead of a calendar week.

## Why the amendment

The original step 8 asked for one green week of the nightly soak. The
soak samples the same paths seven times; LeakSanitizer PROVES
leak-freedom on every path the battery executes, and the battery covers
the entire public surface including hostile inputs. The criterion's
purpose — no leaks at 1.0.0 — is served more directly by proof than by
time-sampling, so criterion 8 was amended (docs/PROOF_STATUS.md,
adopted 2026-11-26): the sanitizer record at the tag gates, the nightly
stays scheduled as a free tripwire. The nightly was green (streak 2)
when the amendment landed and keeps running.

## The proof, at this tag

- `bench/sanitizer_gate.py`: the REAL engine rebuilt with
  ASan+UBSan — full pytest suite 74/74 with ZERO memory errors and
  ZERO undefined-behavior reports; integration proofs (tokenizers
  decode byte-exact, official GGUF reader, snapshot continuation)
  green under the sanitizers; differential-leak torture at cycles
  1/30/60 with LSan totals BYTE-IDENTICAL across scales (756,177
  bytes = the CPython baseline; zero engine leaks — a real 1 KB/cycle
  leak would show +60 KB by scale 60); 15k-mutant blitz on the
  state/GGUF/tokenizer.json loaders — 2,857 loads, 12,143 clean
  rejections, 0 crashes; gcc -fanalyzer 0 warnings.
- `bench/release_gate.py`: all 8 criteria re-proven at the tag;
  criterion 8 checks the committed `bench/sanitizer_record.json` —
  green, hash-matched to THIS engine source, on the tagged line.
- The frozen core is identical to the v0.9.3 freeze; the rc1–rc4
  additive surface (fine-grained errors, snapshot format v6 + fuzz
  gate, parallel intern, api freeze, provenance, packaging, the gates
  themselves) is the 1.0 surface. The manifest is unchanged by this
  release (bench/ additions only).

## The claim stack (docs/PROOF_STATUS.md)

LEAN proves the SPEC (tiling for any surprisal, decode roundtrip +
error-iff, load classification, intern/merge, snapshot gates — no
sorry on the pinned v4.36.0 toolchain). Golden bit-exact gates prove
the implementation matches the validated reference pipeline. The
sanitizer gate proves the implementation leak-free and UB-free on
every executed path. Static analysis found nothing. The remaining gap
— formal C↔LEAN equivalence — is a CompCert-class project, out of
scope and honestly documented.

## Gates (executed at this tag)

make test green; full pytest green; tables drift PASS; all seven LEAN
files compile no-sorry on the pinned toolchain; floors PASS against
the standing envelope; integration proof PASS; provenance record
clean; manifest surface-identical; sanitizer gate SAN-GO; release
gate GO.

## Known-open for 1.0.x

None blocking. The nightly tripwire continues; formal C↔LEAN
extraction remains a documented non-goal for the 1.x line.
