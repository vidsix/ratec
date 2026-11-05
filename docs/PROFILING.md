# Profiling the RateC engine — what the numbers said, what we changed

All measurements on the 2-core development sandbox; the box drifts by
tens of percent over minutes, so every conclusion below comes from
**interleaved A/B runs** (variants alternating within the same minutes)
and **min-of-N** medians, never from back-to-back single runs.

## Tools

* `perf`/`valgrind` unavailable in the sandbox; `-pg` gprof was useless
  because the hot loop is one `always_inline` function (attribution
  collapsed into `main`/`map_grow`).
* What actually worked, in order of value:
  1. **Ablation builds** — compile variants that skip one subsystem
     (`NOO345`, `NOO3`, `NOO4`, `NOO5`, `NOO012`, `NOLOG2`, `NOPATCH`)
     and diff the runtimes. Exact cost attribution, zero measurement
     overhead.
  2. **RDTSC section counters** (`__rdtscp` pairs around six sections) —
     useful shares, but serialization destroys the OoO overlap, so
     absolute times are meaningless (11.1 → 2.2 MB/s with counters on);
     use for *shares only*.
  3. **Bit-exact checksum gate** — FNV-1a over the float32 surprisal
     array + patch starts (`rcsa_pool … c` prints it). Every optimization
     variant must reproduce `5c8f58bed54ea07d` on the 5 MB mixed corpus
     before its timing counts.

## Round 1 — where the time goes (baseline v1, pooled steady-state)

Ablation on the 5 MB mixed corpus (percent of total runtime):

| subsystem                      | share |
|--------------------------------|-------|
| orders 3–5 map machinery       | **~72%** |
| log2                           | ~10%  |
| patcher + emit                 | ~6%   |
| flat orders 0–2                | ~0% (THP already absorbed the TLB cost) |
| loop + everything else         | ~12%  |

Per-order split of the map cost: **order 3 ≈ 22%, order 5 ≈ 8%,
order 4 ≈ 5%.** Order 3 dominates because its hot contexts promote
early (u8 window counts saturate at `tot = 255`), and every visit to a
promoted context pays a **second cache probe** into the count map.

RDTSC shares agreed: o345 maps ~28–34%, prefetch block ~15–19%, flat
orders ~15%, log2 ~7%, patcher ~4%, growth checks ~4%.

## Round 2 — what we tried, what won

| change | verdict |
|---|---|
| AVX-512 `vdivpd` for the 6 mixture terms (lanes bit-exact, cold lanes blended) | **reverted** — bit-exact ✓ but 7.25 vs 11.14 MB/s: the six scalar `divsd` already overlap through OoO, and the lane stack arrays added store-forwarding stalls |
| current-byte cmap/cnt2 prefetches at loop top | **reverted** — 8.3 vs 11.1 MB/s: 4 extra prefetches/byte saturated the MSHRs and delayed the useful future-tmap prefetches |
| **u16 window counts (IMAX 255→65535), window 8→16, 64 B line-aligned tmap slots, 16 B cmap slots** | **kept** — +22% (min-of-5 interleaved: 11.08 vs 9.04 MB/s). 40 B slots alone were *slower* (line straddling); the 64 B slot makes every probe exactly one line |
| SWAR `win_scan` (u64 xor + zero-byte trick) | kept — identical first-match semantics, fewer branches |
| dirty-page bitmaps for `reset()` | kept — reset cost now ∝ touched pages, not table size; fixes a 64–288 MB scan per pooled rep (the first cut had a real bug: bitmap bit was `slot & page_mask` (up to 255 → UB shift for 16 B slots); corrected to `page_index` with word = page>>6) |
| PGO + LTO | dropped — gcda naming under LTO kept missing the use-phase, and any gain drowned in the drift noise |
| fast lane (poly log2 + relaxed tree) | kept as a documented option — ≤ 1 float32 ulp deviation, 100% patch-start agreement on test corpora; ~+5–10% on this box (memory-bound, so the log2 win is mostly hidden) |

Bugs found by the process (all fixed, all now regression-tested):
`save_state` after `feed()` with `store_book=False` memcpy'd a NULL
`obuf`; snapshot v4 stores the fast flag; `load_core` rebuilds the
order-2 dirty bitmap (reset-after-load was corrupting cnt2 rows).

## Where the remaining time goes

After round 2 the profile is still map-probe dominated: three tmap
probes/byte (orders 3–5) plus count probes for promoted contexts, mostly
L3/DRAM. The exact lane's floor on this box is the serial chain
`probe → term → 6 sequential pm += (bit-exact order) → log2 → patcher`,
partially hidden by the depth-2 future prefetch. The honest levers left:
more cores (`batch_spans`, `encode_stream_parallel`), the fast lane, and
—if someone wants to go deeper—per-order tables with per-order-sized
prefetch schedules.

## Reproducing

```bash
cd ratec/build
gcc -O3 -march=native -ffp-contract=off -fno-fast-math -DRC_STANDALONE \
    -o rcsa_pool ../ratec/_rcengine.c -lm -lpthread
./rcsa_pool prof5mb.bin 20 2 21 3 c     # pooled steady-state + checksum
```

## v0.3 campaign — SIMD / pipeline experiments (measured and rejected)

The v0.3 directive was "SIMD-optimize the kernel so it reads multiple
tokens ahead and coordinates them". Every candidate was built, gated
bit-exact, and measured with interleaved best-of-9 A/B on the standalone
harness (english/code/chinese/base64, 20 pf/prebits points). Ground
truth about the box first, because it explains the results:

### VM ground truth (micro-calibrators, scripts/calib.c, scripts/ipctest.c)

| probe | result |
| --- | --- |
| serial imul+add chain | 3.6 cy/iter → effective ~3.55 GHz |
| 4 independent ALU chains | ~4.6 uops/cyc — the CPU is NOT IPC-capped |
| random pointer-chase 512 KB | 2.3 ns/hop (L2, normal) |
| random pointer-chase 64 MB | **52.2 ns/hop ≈ 185 cy** (DRAM-class) |
| divsd throughput | 3.66 cy/div (normal) |

### Where the time goes

* Repetitive input ('aaaa'*300k, working set ≈ L1): **20.2 MB/s** — the
  hot-loop compute/dependency floor is ~160 cy/byte on this uarch.
* Real text (300 KB english): ~6.5 MB/s → **~70% of real-text time is
  DRAM-class latency** on the random table probes (tmap slots + the
  64 MB order-2 row space).
* Prefetch depth sweep (scalar, sequential body): pf0 4.25 → pf1 5.24 →
  **pf2 6.51** → pf3 5.98 → pf4 5.63 MB/s. Deeper prefetch *hurts*
  (RFO pollution of L1/L2 from write-intent prefetches).
* Clamping the order-2 row space into L2 (ablation only): +17-18% → the
  order-2 table costs about a fifth of the wall; the tmap probes the rest.
* Table size (prebits): 16-18 ≈ 21 > 20; prebits=24 collapses to
  2.05 MB/s (1 GB table). Residency beats capacity.
* log2 (fast poly lane vs glibc): +1-7%; divisions (6×divsd/byte,
  ~22 cy divider occupancy): not the bottleneck (nodiv ablation within
  noise). NOT worth vectorizing.

### Candidate 1 — two-stage judge/patch pipeline (RP=8 ring)

Judge and patcher are provably independent chains; a ring decouples
them so log2 + patch branches + book interning "leave" the judge path.
Result: the OoO window (≈512 uops ≈ 3-4 bytes in flight) already
overlaps those chains; the ring adds 2 stores + 2 loads per byte.
Measured: english 6.78 vs 7.40 seq (-8%), chinese -10%, code -2%,
base64 +3%. **Rejected** (kept in git history, commit b942079).

### Candidate 2 — AVX-512 address-generation unit (pfau)

3-lane contexts for orders 3-5 stepped S bytes ahead, one vpmullq per
horizon, scalar prefetch issue. Single-horizon (S=12..20) and dual-
horizon (S=4 + S=4+4·pf) variants both lose to the scalar loop
(6 prefetches/byte, addresses from register-resident h[]): the vector
path pays ~8-10 512-bit uops/byte plus store/reload extraction, while
the scalar math is ~25 cheap ALU uops the scheduler hides anyway.
Measured: AU 6.23/5.45/6.24/6.74 vs scalar 6.78/5.71/6.45/6.87 MB/s on
english/code/chinese/base64. **Rejected** (same commit).

### Kept from v0.3 (pure wins, bit-exact)

* Reusable emit vectors in the core — steady-state `feed` does zero
  malloc/free (was 3 alloc/free per call).
* Ids-only `encode_bytes` path — no span tuples materialized.
* Python-side hot-method binding via an engine-slot property
  (encode("hello world 123") end-to-end: **1.28 µs**, decode: 0.43 µs).
* Poison-on-any-failed-feed policy (reset() recovers).

### Remaining levers (not taken)

* Working-set reduction is semantics-constrained: Ent is 64 B (one
  line) and WIN=16 window promotion is part of the golden behavior;
  shrinking either changes surprisal values and breaks bit-exactness.
* The order-2 row space could become a 16 B-entry map (~10x smaller
  touched set) at the cost of a probe on the hot path — plausible
  future experiment, needs a snapshot format bump (v5).
