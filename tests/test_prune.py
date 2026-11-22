#!/usr/bin/env python3
"""v0.7.0 gate: frequency-aware book pruning (PLAN_v0.7 P2) — the opt-in
engine feature and every invariant the plan demands.

Run:  python3 tests/test_prune.py
Exit 0 only if every check passes.

What is asserted here (and what is deliberately NOT):
  - the DEFAULT path is untouched: prune off == prune kwarg absent (ids,
    spans); non-prune snapshots are format v8 (v0.9.1 write path: the
    v7 layout explicit-LE, prune presence as a header flag — the v6
    goldens still LOAD unchanged, read-side back-compat). The broader default-path bit-exactness vs
    the reference pipeline is gate_bitexact.py's job and must not move.
  - prune mechanics: singletons are dropped at absolute-offset boundaries,
    survivors renumber dense, counts restart per window, n_pruned counts.
  - determinism at the OUTPUT level: same stream -> same spans, same ids,
    same pruned book — across fresh engines AND across feed chunkings
    (boundaries are absolute stream offsets). Snapshot BYTES are not
    compared across chunkings: the judge map section serializes in slot
    order and slot layout follows the map-growth trajectory, which
    core_reserve sizes per chunk — true of v0.6.0 too, not a prune effect.
  - spans are INVARIANT to pruning (the patcher is book-independent); ids
    are identical up to the first prune boundary.
  - snapshot v7 round-trips and resumes pruning exactly (output-level); v6
    snapshots (no prune) load unchanged.
  - decode semantics: within a prune epoch decode round-trips; ids emitted
    before a prune are cleanly REJECTED (RatecError) by the bounds check —
    the documented opt-in trade-off.
  - reset() restores virgin BEHAVIOR (outputs equal a fresh engine's);
    map capacity legitimately persists across reset (kept pages), so
    snapshots are not compared here.
  - prune on a bookless engine is an accepted no-op; carry mode composes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ratec
from ratec import RatecError

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# deterministic corpus: frequent phrases plus a crowd of one-off tokens
# (~185 KB), prune every 4 KiB
COMMON = ("the quick brown fox jumps over the lazy dog and the signal is "
          "the rate at which information flows through the channel. ").encode()
RARE = [f" rare-token-{k:04d}-xy ".encode() for k in range(400)]
parts = []
for k in range(1400):
    parts.append(COMMON)
    if k % 3 == 0:
        parts.append(RARE[(k * 7) % len(RARE)])
        parts.append(RARE[(k * 13 + 1) % len(RARE)])
DATA = b"".join(parts)
PA = 4096


def fresh(prune=True, **kw):
    if prune:
        return ratec.RateC(c=20, prune=True, prune_after=PA,
                           prune_min=kw.get("prune_min", 2))
    return ratec.RateC(c=20)


def spans_of(t, data):
    return [s[:2] for s in t.encode_to_spans(data)]


# ============================================================ 1. default path
print("1. default path untouched (prune off)")
t_ref = ratec.RateC(c=20)
spans_ref = spans_of(t_ref, DATA)
t_ref2 = ratec.RateC(c=20)
ids_ref = t_ref2.encode_bytes(DATA)       # fresh cold engine
snap_ref = t_ref2._eng.save_state()       # post-encode v6 snapshot
# v0.9.1: the write path is v8-only (explicit LE) — the version byte is
# 8 for prune AND non-prune engines; the prune section's presence moved
# into the header flag byte (offset 42), so "default path untouched"
# means: same version, flag 0, byte-identical payload across configs.
check("prune kwarg absent == prune=False (format v8, flag 0)",
      int.from_bytes(snap_ref[4:8], "little") == 8
      and snap_ref[42] == 0
      and ratec.RateC(c=20, prune=False)._eng.save_state()[42] == 0)
t_off = ratec.RateC(c=20, prune=False)
ids_off = t_off.encode_bytes(DATA)
t_off2 = ratec.RateC(c=20, prune=False)
_ = t_off2.encode_bytes(DATA)
check("prune=False ids identical + snapshot byte-identical",
      ids_off == ids_ref and t_off2._eng.save_state() == snap_ref)
check("prune=False spans identical",
      spans_of(ratec.RateC(c=20, prune=False), DATA) == spans_ref)
check("prune off: n_pruned stays 0", t_off.n_pruned == 0
      and t_off.prune_after == 0)

# ============================================================ 2. prune mechanics
print("2. prune mechanics (singleton starvation at work)")
t_on = fresh()
spans_on = spans_of(t_on, DATA)
check("spans invariant to pruning", spans_on == spans_ref,
      f"({len(spans_on)} patches)")
# ids must match the un-pruned run until the first boundary crossing: the
# first prune fires at the first close with end >= PA
pre = [s for s in spans_ref if s[0] < PA - 64]     # safely before boundary
check("ids identical before the first prune boundary",
      spans_on[:len(pre)] == pre)
t_cnt = fresh()
_ = t_cnt.encode_bytes(DATA)
check("book shrinks + n_pruned counts",
      t_cnt.book_size < 3000 and t_cnt.n_pruned > 0,
      f"book {t_cnt.book_size} (unpruned ~3269), pruned {t_cnt.n_pruned}")
check("prune-mode snapshot is format v8 with prune flag",
      int.from_bytes(t_cnt._eng.save_state()[4:8], "little") == 8
      and t_cnt._eng.save_state()[42] == 1)

# ============================================================ 3. determinism
print("3. determinism (outputs, across engines and chunkings)")
a = fresh(); a.encode_bytes(DATA)
b = fresh(); b.encode_bytes(DATA)
check("two fresh engines: identical ids + counters",
      a.encode_bytes(b"") == b.encode_bytes(b"") == []
      and a.n_patches == b.n_patches
      and a.n_pruned == b.n_pruned and a.book_size == b.book_size)
c = fresh()
CHUNK = 997
chunk_spans = []
for k in range(0, len(DATA), CHUNK):
    chunk_spans.extend(c.feed(DATA[k:k + CHUNK]))
chunk_spans.extend(c.flush())
check("chunked feed+flush == one-shot spans; counters equal",
      [(s, e) for s, e, _ in chunk_spans] == spans_ref
      and c.n_patches == a.n_patches and c.n_pruned == a.n_pruned
      and c.book_size == a.book_size and c.bytes_seen == a.bytes_seen)
d = fresh()
CHUNK2 = 65536
chunk_spans2 = []
for k in range(0, len(DATA), CHUNK2):
    chunk_spans2.extend(d.feed(DATA[k:k + CHUNK2]))
chunk_spans2.extend(d.flush())
check("different chunk size: same spans + counters",
      [(s, e) for s, e, _ in chunk_spans2] == spans_ref
      and d.n_pruned == a.n_pruned and d.book_size == a.book_size)

# ============================================================ 4. decode semantics
print("4. decode semantics (within-epoch round-trip, cross-epoch rejected)")
t_e = fresh()
small = COMMON * 3                       # well inside one prune interval
ids_small = t_e.encode_bytes(small)
check("within-epoch decode round-trips",
      t_e.decode_bytes(ids_small) == small)
t_f = fresh()
full_ids = t_f.encode_bytes(DATA)        # ids spanning many prune epochs
rejected = 0
try:
    t_f.decode_bytes(full_ids)
except RatecError:
    rejected = 1
check("cross-epoch ids are cleanly rejected (RatecError)", rejected == 1)
t_g = fresh()
t_g.encode_bytes(DATA[:-PA])             # warm through all but the tail
tail = DATA[-PA // 2:]
tail_ids = t_g.encode_bytes(tail)
check("post-prune (current-epoch) ids decode",
      t_g.decode_bytes(tail_ids) == tail)

# ============================================================ 5. snapshot v7
print("5. snapshot v7 round-trip + resume + v6 back-compat")
t_h = fresh()
prefix_spans = t_h.feed(DATA[:len(DATA) // 2])   # NO flush: the open patch
snap7 = t_h._eng.save_state()                    # spans the save point
t_i = ratec.RateC.from_state(snap7)
rest_spans = t_i.feed(DATA[len(DATA) // 2:]) + t_i.flush()
t_f2 = fresh()
full_spans = t_f2.feed(DATA) + t_f2.flush()
n_pre = len(prefix_spans)
check("v7 snapshot resumes pruning exactly (spans, ids, counters)",
      [(s, e, i) for s, e, i in
       ([(s, e, i) for s, e, i in prefix_spans]
        + [(s, e, i) for s, e, i in rest_spans])]
      == [(s, e, i) for s, e, i in full_spans]
      and t_i.n_pruned == t_f2.n_pruned
      and t_i.book_size == t_f2.book_size
      and t_i.n_patches == t_f2.n_patches)
t_k = ratec.RateC.from_state(snap_ref)   # v6 from the default engine
check("v6 snapshot loads unchanged; stays non-prune",
      t_k._eng.save_state() == snap_ref and t_k.prune_after == 0)

# ============================================================ 6. params/edges
print("6. parameters, edges, composition")
# deterministic prune_min semantics: "AA" runs carry high counts; the
# content "Bx" is seen EXACTLY twice inside the first window; "By"/"Cz"
# once. One prune fires in the trailing run. min=2 keeps Bx; min=3 drops
# it -> n_pruned differs by exactly 1, book_size by exactly 1.
Ax = b"A" * 100
stream6 = Ax + b"Bx" + Ax + b"Bx" + b"A" * 3900
t_m2 = ratec.RateC(c=20, prune=True, prune_after=4096, prune_min=2)
_ = t_m2.encode_bytes(stream6)
t_m3 = ratec.RateC(c=20, prune=True, prune_after=4096, prune_min=3)
_ = t_m3.encode_bytes(stream6)
check("prune_min=2 keeps the 2-seen entry; prune_min=3 drops it",
      t_m3.n_pruned == t_m2.n_pruned + 1
      and t_m2.book_size == t_m3.book_size + 1,
      f"(min2 pruned {t_m2.n_pruned} book {t_m2.book_size}; "
      f"min3 pruned {t_m3.n_pruned} book {t_m3.book_size})")
t_n = ratec.RateC(c=20, prune=True, prune_after=1 << 40, prune_min=2)
ids_n = t_n.encode_bytes(DATA)
check("interval beyond the stream: never fires, ids identical",
      t_n.n_pruned == 0 and ids_n == ids_ref)
t_o1 = ratec.RateC(c=20, prune=True, prune_after=64, prune_min=1 << 30)
_ = t_o1.encode_bytes(DATA)
check("huge prune_min empties the book; engine continues",
      t_o1.book_size == 0 or t_o1.book_size < 5)
tail2 = b"the quick brown fox"
check("post-emptying encode still round-trips",
      t_o1.decode_bytes(t_o1.encode_bytes(tail2)) == tail2)
t_p0 = ratec.RateC(c=20, store_book=False, prune=True, prune_after=PA)
check("bookless prune is a no-op",
      len(t_p0.encode_to_spans(DATA[:4096])) > 0 and t_p0.n_pruned == 0)
t_pc = ratec.RateC(c=20, prune=True, prune_after=PA, prune_min=2, carry=True)
ids_pc = t_pc.encode_bytes(DATA[:PA * 3])
check("carry mode composes with pruning (round-trip within epoch)",
      t_pc.decode_bytes(t_pc.encode_bytes(COMMON * 2)) == COMMON * 2
      and len(ids_pc) > 0)
t_q = fresh()
t_q.encode_bytes(DATA)
t_q.reset()
r_q = t_q.encode_bytes(DATA)
check("reset restores virgin behavior (ids == fresh run)",
      r_q == fresh().encode_bytes(DATA) and t_q.n_pruned > 0)
bad = 0
try:
    ratec.RateC(c=20, prune=True, prune_after=0)
except ValueError:
    bad = 1
check("prune=True with prune_after=0 rejected", bad == 1)

# ============================================================ 7. streaming path
print("7. feed/flush streaming path")
t_r = fresh()
n_closed = 0
for k in range(0, len(DATA), 65536):
    n_closed += len(t_r.feed(DATA[k:k + 65536]))
n_closed += len(t_r.flush())
check("streaming path prunes identically",
      n_closed == len(spans_ref) and t_r.n_pruned == a.n_pruned
      and t_r.book_size == a.book_size)

print(f"\nPRUNE GATE: {'PASS' if all(RESULTS) else 'FAIL'} "
      f"({sum(RESULTS)}/{len(RESULTS)} checks)")
sys.exit(0 if all(RESULTS) else 1)
