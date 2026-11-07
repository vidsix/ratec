#!/usr/bin/env python3
"""v0.4.0 gate: fractional C, credit carry-over patcher, snapshot v5
(with v4 back-compat fixtures), and the rate servo.

Run:  python3 tests/test_v04.py
Exit 0 only if every check passes.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ratec
from ratec import patch_from_sur, solve_c, solve_c_for_book

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


TD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
fmeta = json.load(open(os.path.join(TD, "fixtures_meta.json")))
corpus = open(os.path.join(TD, "gate_slice.bin"), "rb").read()   # 160k mixed bytes
assert len(corpus) == fmeta["gate_slice_len"] and hashlib.sha256(corpus).hexdigest() == fmeta["gate_slice_sha256"], \
    "gate_slice.bin corrupt — see fixtures_meta.json"
# the slice was captured as code[:80k]+zh[:50k]+mix[:30k]; the raw files are
# workspace-local, so the bound slice ships instead — identical bytes everywhere

# ============================================================ 1. fractional C
print("1. fractional C")
tok = ratec.RateC(c=17.5)
ids = tok.encode("hello 你好!")
check("c=17.5 roundtrip", tok.decode(ids) == "hello 你好!")
check("c property is float", isinstance(tok.c, float) and abs(tok.c - 17.5) < 1e-12)

sub = corpus[:200000]                    # (corpus is 160k, so sub == corpus)
N = len(sub)
spans = ratec.RateC(c=17.5, store_book=False).encode_to_spans(sub)
tile = spans[0][0] == 0 and spans[-1][1] == N and all(
    spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1))
check("c=17.5 spans tile exactly", tile, f"{len(spans)} patches")

sur = ratec.RateC(c=1.0, store_book=False).surprise(sub)
n16 = len(patch_from_sur(sur, 16))
n175 = len(patch_from_sur(sur, 17.5))
n20 = len(patch_from_sur(sur, 20))
check("patch count monotone 16 > 17.5 > 20", n16 > n175 > n20,
      f"{n16} > {n175} > {n20}")

eng_starts = np.array(
    [s for s, _, _ in ratec.RateC(c=17.5, store_book=False)
     .encode_to_spans(sub)], dtype=np.uint32)
check("engine c=17.5 == patch_from_sur(17.5)",
      np.array_equal(eng_starts, patch_from_sur(sur, 17.5)))

try:
    ratec.RateC(c=0.5)
    check("c out of range rejected", False)
except ValueError:
    check("c out of range rejected", True)

# ============================================== 2. carry patcher vs reference
print("2. credit carry-over patcher")


def ref_carry_starts(sur_f32: np.ndarray, C: float) -> np.ndarray:
    """Python golden: sequential float64 cumsum, same expression shapes
    as the C spec (lim = (cs_open + C) + carry; carry = clamp(waste))."""
    n = len(sur_f32)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(sur_f32.astype(np.float64), out=cs[1:])
    starts = []
    s, carry = 0, 0.0
    while s < n:
        lim = cs[s] + C + carry
        e, j = s, s + 1
        while j <= n and cs[j] <= lim:
            e = j
            j += 1
        if e <= s:
            e, carry = s + 1, 0.0            # forced escape
        else:
            waste = lim - cs[e]
            carry = 0.0 if waste < 0.0 else (C if waste > C else waste)
        starts.append(s)
        s = e
    return np.array(starts, dtype=np.uint32)


rng = np.random.default_rng(7)
fuzz = {
    "text-like": sur,
    "exp(4)": rng.exponential(4.0, 50000).astype(np.float32),
    "tiny(0,0.01)": rng.uniform(0.0, 0.01, 50000).astype(np.float32),
    "huge(30,40)": rng.uniform(30.0, 40.0, 20000).astype(np.float32),
    "alternating": np.tile([40.0, 0.01], 25000).astype(np.float32),
}
ok_all = True
for name, arr in fuzz.items():
    for c in (12, 17.5, 20, 33):
        got = patch_from_sur(arr, c, carry=True)
        want = ref_carry_starts(arr, c)
        if not np.array_equal(got, want):
            ok_all = False
            print(f"    mismatch: {name} c={c} "
                  f"({len(got)} vs {len(want)} starts)")
check("fuzz: C carry patcher == python golden (5 arrays x 4 C)", ok_all)

# engine (streaming, book mode) vs reference on real text
eng = ratec.RateC(c=20, carry=True)
spans_c = eng.encode_to_spans(corpus)
starts_c = np.array([s for s, _, _ in spans_c], dtype=np.uint32)
sur_full = ratec.RateC(c=1.0, store_book=False).surprise(corpus)
check("engine carry starts == reference",
      np.array_equal(starts_c, ref_carry_starts(sur_full, 20.0)))
check("carry register within [0, C]", 0.0 <= eng.carry <= 20.0,
      f"carry={eng.carry:.4f}")

# streaming parity: feed in irregular chunks + flush == one-shot
st = ratec.RateC(c=20, carry=True)
chunked = []
pos = k = 0
steps = [33333, 1, 77777, 5]
while pos < len(corpus):
    step = steps[k] if k < len(steps) else 65537
    chunked.extend(st.feed(corpus[pos:pos + step]))
    pos += step
    k += 1
chunked.extend(st.flush())
starts_stream = np.array([s for s, _, _ in chunked], dtype=np.uint32)
check("streaming feed/flush == one-shot (carry)",
      np.array_equal(starts_stream, starts_c))

# invariants: per-patch bits <= 2C for multi-byte patches; mean -> C
cs64 = np.empty(len(sur_full) + 1, dtype=np.float64)
cs64[0] = 0.0
np.cumsum(sur_full.astype(np.float64), out=cs64[1:])
edges = np.append(starts_c.astype(np.int64), len(corpus))
sums = np.diff(cs64[edges])
lens = np.diff(edges)
check("multi-byte patch bits <= 2C", bool((sums[lens > 1] <= 40.0 + 1e-9).all()),
      f"max={sums[lens > 1].max():.3f}")
mean_bits = float(cs64[-1] / len(sums))
# telescoping gives C exactly for normal closes; forced escapes overspend
# (sur > budget), so allow a small positive offset
check("mean bits/patch -> C (telescoping)", abs(mean_bits - 20.0) < 0.25,
      f"mean={mean_bits:.5f}")
check("carry fills more than under-shoot",
      len(starts_c) < n20, f"carry {len(starts_c)} < undershoot {n20} patches")

# ================================================= 3. snapshot v5 + v4 compat
print("3. snapshots")
t5 = ratec.RateC(c=17.5, carry=False)
t5.encode_bytes(corpus[:100000])
snap5 = t5.save_state()
t5b = ratec.RateC.from_state(snap5)
cont = corpus[100000:]
check("v5 snapshot: c preserved", abs(t5b.c - 17.5) < 1e-12)
check("v5 snapshot: continuation identical",
      t5b.encode_to_spans(cont) == t5.encode_to_spans(cont))

tc = ratec.RateC(c=20, carry=True)
tc.encode_bytes(corpus[:100000])
snapc = tc.save_state()
tcb = ratec.RateC.from_state(snapc)
check("v5 snapshot: carry mode preserved",
      tcb.carry_mode and tcb.encode_to_spans(cont) == tc.encode_to_spans(cont))

ok_v4 = True
cont_v4 = open(os.path.join(TD, "v4_cont_slice.bin"), "rb").read()   # the fixture capture's slice
assert len(cont_v4) == fmeta["v4_cont_len"] and hashlib.sha256(cont_v4).hexdigest() == fmeta["v4_cont_sha256"], \
    "v4_cont_slice.bin corrupt"
for c in (20, 33):
    blob = open(os.path.join(TD, f"v4_c{c}.bin"), "rb").read()
    gold = json.load(open(os.path.join(TD, f"v4_c{c}_cont_golden.json")))
    assert gold["cont_sha256"] == fmeta["v4_cont_sha256"]
    t4 = ratec.RateC.from_state(blob)
    if abs(t4.c - float(c)) > 1e-12:
        ok_v4 = False
        print(f"    v4 c{c}: c restored as {t4.c}")
    got = [int(s) for s, _, _ in t4.encode_to_spans(cont_v4)]
    if got != gold["starts_absolute"]:
        ok_v4 = False
        print(f"    v4 c{c}: continuation diverges "
              f"({len(got)} vs {len(gold['starts_absolute'])} patches)")
check("v4 back-compat: load + bit-exact continuation (c=20, c=33)", ok_v4)

# ================================================================== 4. servo
print("4. rate servo")
sl = corpus[:100000]
sur_sl = ratec.RateC(c=1.0, store_book=False).surprise(sl)
n20_sl = len(patch_from_sur(sur_sl, 20))
target = int(n20_sl * 1.15)                   # 15% more patches than C=20
c_star, n_star = solve_c(sl, target)
check("servo lands near target",
      abs(n_star - target) <= max(1, target // 500),
      f"c*={c_star:.4f} -> {n_star} patches (target {target})")
c_lo, n_lo = solve_c(sl, 4 * target)
check("servo higher target -> smaller C", c_lo < c_star,
      f"c*={c_lo:.4f} -> {n_lo}")
c_carry, n_carry = solve_c(sl, target, carry=True)
check("servo works in carry mode",
      abs(n_carry - target) <= max(1, target // 500),
      f"c*={c_carry:.4f} -> {n_carry}")

# book-size servo (unimodal target): 10% above the C=20 book
eng_b20 = ratec.RateC(c=20, store_book=True)
eng_b20.encode_bytes(sl)
target_book = int(eng_b20.book_size * 1.10)
c_b, book_b = solve_c_for_book(sl, target_book)
check("book servo within 2% of target",
      abs(book_b - target_book) <= max(2, target_book // 50),
      f"c*={c_b:.3f} -> book {book_b} (target {target_book})")

# servo patch-count == engine truth at the returned C
eng_check = ratec.RateC(c=c_star, store_book=True)
eng_check.encode_bytes(sl)
st2 = patch_from_sur(sur_sl, c_star)
edges2 = np.append(st2.astype(np.int64), len(sl))
distinct = len(set(sl[edges2[k]:edges2[k + 1]]
                   for k in range(len(edges2) - 1)))
check("servo C reproduces book from re-patch",
      eng_check.book_size == distinct,
      f"book={eng_check.book_size} distinct={distinct}")

print()
print(f"V04 GATE: {'PASS' if all(RESULTS) else 'FAIL'} "
      f"({sum(RESULTS)}/{len(RESULTS)} checks)")
sys.exit(0 if all(RESULTS) else 1)
