# PROOF_STATUS — what is proven, how, and what is claimed

The question this document answers: **can correctness be established
by proof instead of by wall-clock soaking, and where exactly does each
method's power end?** The honest answer is a STACK of four evidence
layers — each covers a different failure class, and only the stack
covers the code "everywhere".

## The claim stack (as of v1.0.0rc4 + this gate)

| Layer | Tool | What it proves | What it cannot prove |
|---|---|---|---|
| 1. Specification | LEAN, 7 files, no-sorry on the pinned v4.36.0 | the ALGORITHM is right: tiling correctness for ANY surprisal (T2), decode roundtrip + error-iff, load classification total, intern/merge semantics, snapshot magic/version gates | nothing about the C implementation — there is no extractor connecting C to LEAN, and memory behavior is outside the model |
| 2. Implementation equivalence | golden bit-exact gates (`tests/gate_bitexact.py`: bitwise float32 surprisal + exact patch starts + exact tiling vs the validated reference pipeline's captured arrays), reference cross-checks inside the suite, byte-exact clone/restore and continuation tests | the C engine reproduces the validated reference behavior on every captured input, byte for byte | inputs outside the captured/fixture set (sampled, not universal) |
| 3. Memory & UB safety | **`bench/sanitizer_gate.py`** — the REAL engine rebuilt with ASan+UBSan: full pytest suite, integration proofs (tokenizers bridge + GGUF official reader + snapshot continuation), the 3-scale differential-leak torture (cycles 1/30/60; a real leak scales with cycles, the CPython baseline does not), a 15k-mutant blitz on the state/GGUF/tokenizer.json loaders | **no memory error and no undefined behavior on any executed path; no leak that scales with workload** — per-path proof, not per-night sampling | paths the battery never executes (fuzz + suite + torture cover the entire public surface, but "every path" means every EXECUTED path) |
| 4. Static analysis | gcc -fanalyzer over the engine source | potential leak/UAF paths visible statically | noisy; report-only, it informs and does not gate |

The nightly soak's evidence class — "RSS stayed flat on 7 sampled
nights" — is strictly WEAKER than layer 3 for the leak question: it
samples the same paths 7 times and can miss a leak that needs a
specific workload; LSan proves leak-freedom on every path the battery
executes. What the soak uniquely adds is environment variance
(runner states, allocator behavior across images) — which is why the
proposal below keeps it, free, as a tripwire.

## The proposed STABILITY.md criterion-8 amendment

Current (step 8): *the nightly soak has been green (flat RSS) for
≥ 1 week before the rc tag.*

Proposed replacement:

> 8. Memory-safety proof at the release tag: `bench/sanitizer_gate.py`
>    says SAN-GO — zero ASan errors, zero UBSan runtime errors, and
>    non-scaling leaks (LSan, differential across torture scales) over
>    the full test suite, the integration proofs, and the mutant
>    blitz. The nightly soak stays scheduled as a free tripwire; its
>    greenness is reported but is no longer the gate.

Rationale: the criterion's PURPOSE is "no leaks at 1.0.0". Per-path
LSan proof serves that purpose more directly than time-sampling. The
amendment keeps the deprecation discipline of the contract: the
replacement ships additively (the gate lands in rc4 first, the nightly
continues), the old path stays documented, and the change is visible
in the manifest diff (bench/ additions only — the frozen surface
does not move).

## What "right everywhere provably" honestly means here

- The SPEC is proven everywhere (LEAN, all properties, no sorry).
- The IMPLEMENTATION is proven UB-free and leak-free on every path the
  gate executes — and the gate executes the entire public surface,
  including hostile inputs (15k mutants).
- The IMPLEMENTATION matches the validated reference bit-exactly on
  every captured fixture, and the suite re-derives reference equality
  on live inputs.
- The remaining gap — formal equivalence between the C source and the
  LEAN spec — is a full extractor project (CompCert/Cogent-class) and
  is out of scope for 1.0. Nobody ships that for a CPython extension;
  the stack above is the industry-standard ceiling.

## How to re-verify (one command each)

    python3 bench/sanitizer_gate.py        # layers 3+4, ~30 min
    python3 bench/release_gate.py          # the 8-point freeze audit
    make test && pytest tests/ -q          # layers 1+2 execution
