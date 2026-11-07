# v0.4.0 design notes — what landed, what was measured, what was declined

Engineering record for the v0.4 cycle. This is the "expected-value order"
list evaluated against measurements on the 2-core reference sandbox;
every decline is documented with the number that declined it.

## Landed

### Fractional C (nearly free, quietly useful)
`C` is now a double everywhere (`RcCore.C`, patcher thresholds, snapshots,
`batch`, `patch_from_sur`). `c=17.5` costs nothing — the budget was always
a float compare (`lim = cs + C`). Snapshot format bumped to **RCST v5**
(C stored as f64; v2–v4 blobs still load, integer C widened to double).
Killer use case: sweeping fractional C to hit an exact book size.

### Credit carry-over patcher (opt-in, `RateC(c=..., carry=True)`)
Replaces the proposed "closest to C" mode, which would have broken the
`Σ ≤ C` invariant the LM story leans on. Formal spec (golden, mirrored in
`tests/test_v04.py::ref_carry_starts`):

```
carry_0 = 0;  carry ∈ [0, C]  (a register, not an accumulator)
open:   lim = cs_open + C + carry            (spend cap = +C extra)
close:  waste = lim − cs_close ≥ 0
        carry = clamp(waste, 0, C)           (deposit unused budget)
forced (single byte > budget): carry = 0    (escape burns the carry)
flush:  same deposit as a normal close      (streaming parity)
```

Telescoping: Σ_n = C + carry_{n−1} − carry_n for normal closes ⇒ mean
bits/patch → C exactly as the carry is bounded. Measured on the mixed
160 KB gate corpus at C=20: **41,791 → 35,494 patches (−15.1%)**, mean
bits/patch 19.99949, carry patcher costs **0.75×** the under-shoot
patcher (it does less work per patch). Streaming `feed/flush == one-shot`
holds bit-exactly; snapshot v5 carries `pmode` + `carry`.

### Rate servo (`ratec.solve_c`, `ratec.solve_c_for_book`)
The judge pass is C-independent, so one `surprise()` pass serves every
candidate C — the servo is judge-once/patch-many bisection.
Design correction found by testing: **patch count** is strictly monotone
decreasing in C (exact bisection ⇒ `solve_c` lands the target exactly,
e.g. 28,007/28,007 at c*=17.3173), but **book size vs C is unimodal**
(→256 as C→1; peaks mid-range; falls back toward patch count as patches
become long and unique) ⇒ `solve_c_for_book` does a coarse ascending
scan + local bisection and returns the achieved book size honestly.

### Modality profiles (`ratec.profiles`, registry-as-data)
Architecture copied from codecs: **filters + entropy coder**. RateC is
the entropy coder; a profile is `(byte transform, suggested C, note)`;
transforms live upstream and are lossless; the core never branches on
modality. Measured predictability wins at C=20 (same judge, patch count):

| profile | transform | patch-count win |
|---|---|---|
| pcm16 | deinterleave channels + per-plane delta | **−37.1%** |
| f32   | byte-plane separation (exp plane smooth, mantissa honest) | −9.5% |
| u8_image | per-row previous-pixel delta | −9.6% |

Losslessness is absolute: every transform round-trips bit-for-bit
(47/47 profile checks, incl. real photo / PCM / NaN-inf-denormal f32).

## Measured, then declined or deferred (with the numbers)

### Run-detection fast path — declined for exact mode, deferred for fast lane
The proposal: skip order-3..5 map probes for highly predictable bytes.
Blocker (structural, not tuning): v2 **fuses predict-and-update into one
cache-line probe** per order. Skipping the prediction still pays the
probe (adaptivity requires the update), and skipping the update diverges
the judge state — so an *exact* fast path cannot skip the expensive part,
only the mixture arithmetic around it. Ceiling measured via the lean
decomposition: the entire patcher is 5.5% of runtime; the arithmetic
around the probes is a fraction of that. Revisit only with a split
predict/write probe design (v0.5 candidate).

### SIMD prefix-sum patcher — deferred (5.5% ceiling)
Measured: judge 164.0 ms vs patcher 9.6 ms on 1.5 MB (5.5% of runtime).
A vectorized patcher caps total speedup at ~5%. Also a correction to the
"bit-exact trivially" assumption: float addition is not associative, so a
reassociated SIMD scan changes cut positions near thresholds; bit-exact
SIMD needs speculative scan + exact verification windows. Worth doing
only after the judge gets faster (then the patcher share grows) or as
the GPU-offload foundation. Design kept in this note; not landed.

### Pipelined judge/patcher — deferred
Hides the same 5.5%, at the cost of a thread pair and chunk-buffer
ring. `encode_stream_parallel` already covers the multi-core case for
bulk prep. Revisit with the SIMD patcher.

## GPU batch judge — design note (no GPU on the reference box)
The judge is sequential *per stream* but independent across streams.
A batched kernel steps D documents forward one byte at a time:
state = D × (cnt0[256] + tot/h per stream shared tables are the issue —
per-stream direct tables cnt1/cnt2 residency vs. shared read-only maps).
The map for orders 3–5 is the hard part: open addressing with per-stream
probes randomizes access; either partition the map per stream (D × 96 B/hot
byte — memory-bound) or accept cross-stream contention. Batch D=256 ×
1 byte/step keeps HBM saturated on the direct-table portion; expected
throughput ≫ CPU for corpus prep. This is the LM-track item: tokenize
corpora at GB/s where the dataloader would otherwise idle.

## Compatibility
- RCST v5 snapshots: `double C`, `pmode`, `carry`; v2–v4 load unchanged
  (bit-exact continuation verified against fixtures captured with the
  v0.3.0 binary: `tests/data/v4_c{20,33}.bin` + continuation goldens).
- Default path (under-shoot, integer or fractional C) is bit-identical
  to v0.3.0: `tests/gate_bitexact.py` green on all four domains.
