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
