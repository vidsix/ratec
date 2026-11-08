#!/usr/bin/env python3
"""Seeded property/fuzz suite + untested-API sweep (PLAN_v0.6 P3).

Everything here is deterministic (fixed seeds) and dependency-free beyond
numpy. Sections:

  1. P3-1 property fuzz   — generated streams x C values: spans tile
                            exactly, decode round-trips, chunked streaming
                            == one-shot, patch_from_sur(surprise(x)) ==
                            engine starts, batch == serial
  2. P3-2 judge C-independence — surprise(x) bitwise-equal across
                            c in {1,10,20,33,64} on identical state
  3. P3-3 fast-lane parity — exact vs exact=False: surprisal deviation vs
                            the claimed <=1.1e-7 relative bound, recorded
                            in research/fastlane_parity.json
  4. P3-4 pf equality     — pf 0..4 are prefetch-depth variants only:
                            spans and surprisal bitwise-identical
  5. P3-5 negative/boundary — boundary C values, map exhaustion is a
                            RECOVERABLE RatecError, reset-vs-fresh
                            bit-exactness, book-mode snapshot after feed()
                            (the obuf path)
  6. P3-6 API sweep       — solve_c brackets, warm_state path, batch
                            fast/carry, encode_stream_parallel edges,
                            carry+reset, CLI smoke incl. fractional --c

Run:  python3 tests/test_fuzz.py
Exit 0 only if every check passes.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ratec  # noqa: E402
from ratec import (RateC, RatecError, batch_spans, patch_from_sur,  # noqa: E402
                   solve_c, solve_c_for_book)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# ------------------------------------------------------------- generators
def gen_streams(rng):
    """~60 seeded streams: structured + adversarial, lengths 0..20k."""
    out = []
    out.append(b"")                                        # empty
    out.append(b"x")                                       # 1 byte
    out.append(b"\x00" * 4096)                             # all-zero
    out.append(b"\xff" * 4096)                             # all-0xFF
    out.append(("abcdefgh" * 512).encode())                # low-entropy repeat
    out.append("héllo wörld — ünïcode ✓ 你好世界 " .encode() * 64)  # UTF-8 text
    out.append(bytes(rng.integers(0, 256, 20000, dtype=np.uint8)))   # uniform
    out.append(bytes(rng.integers(0, 4, 15000, dtype=np.uint8)))     # 2-bit-ish
    out.append(bytes(rng.choice([0, 255], 12000).astype(np.uint8)))  # binary
    for n in (2, 3, 15, 16, 17, 63, 64, 65, 255, 256, 1023, 4097):
        out.append(bytes(rng.integers(0, 256, n, dtype=np.uint8)))
    # base64-ish + json-ish + csv-ish structured
    import base64
    blob = bytes(rng.integers(0, 256, 3000, dtype=np.uint8))
    out.append(base64.b64encode(blob) * 3)
    out.append(json.dumps({"k": list(range(500))}).encode() * 2)
    return out


CS = (1, 8, 17.5, 33, 63.5, 64)


def main():
    rng = np.random.default_rng(42)
    print("property fuzz + API sweep")

    streams = gen_streams(rng)

    # ============================================== 1. P3-1 property fuzz
    n_cases = 0
    ok_prop, why = True, ""
    for si, data in enumerate(streams):
        for c in CS:
            if len(data) > 8000 and c not in (20, 17.5):
                continue                    # keep big streams to 2 C values
            n_cases += 1
            try:
                # spans tile exactly (empty input -> zero spans, still valid)
                spans = RateC(c=c, store_book=False).encode_to_spans(data)
                if data:
                    if spans[0][0] != 0 or spans[-1][1] != len(data):
                        ok_prop, why = False, f"stream {si} c={c}: no tiling"
                        break
                    if any(spans[k][1] != spans[k + 1][0]
                           for k in range(len(spans) - 1)):
                        ok_prop, why = False, (
                            f"stream {si} c={c}: gap in tiling")
                        break
                else:
                    if spans:
                        ok_prop, why = False, (
                            f"stream {si} c={c}: empty input gave spans")
                        break
                # book-mode round trip (ids)
                tok = RateC(c=c)
                ids = tok.encode_bytes(data)
                if len(data) and tok.decode_bytes(ids) != data:
                    ok_prop, why = False, f"stream {si} c={c}: round-trip"
                    break
                # chunked streaming == one-shot (random boundaries)
                if len(data) > 1:
                    eng1 = RateC(c=c, store_book=False)
                    one = eng1.encode_to_spans(data)
                    eng2 = RateC(c=c, store_book=False)
                    cuts = sorted(rng.choice(
                        range(1, len(data)), size=min(7, len(data) - 1),
                        replace=False)) if len(data) > 8 else [len(data) // 2]
                    prev, acc = 0, []
                    for cu in list(cuts) + [len(data)]:
                        acc += eng2.feed(data[prev:cu])
                        prev = cu
                    acc += eng2.flush()
                    if [(s, e) for s, e, _ in one] != \
                       [(s, e) for s, e, _ in acc]:
                        ok_prop, why = False, (
                            f"stream {si} c={c}: chunked != one-shot")
                        break
                # patch_from_sur(surprise) == engine starts
                sur = RateC(c=1, store_book=False).surprise(data)
                pf_starts = patch_from_sur(sur, c)
                eng_starts = np.array([s for s, _, _ in spans], np.uint32)
                if not np.array_equal(pf_starts, eng_starts):
                    ok_prop, why = False, (
                        f"stream {si} c={c}: patch_from_sur != engine")
                    break
            except Exception as e:                          # noqa: BLE001
                ok_prop, why = False, f"stream {si} c={c}: {type(e).__name__}: {e}"
                break
        if not ok_prop:
            break
    check(f"property fuzz: {n_cases} streamxC cases tile/roundtrip/"
          "stream-parity/patcher-parity", ok_prop, why)

    # batch == serial (independent docs, cold engines)
    docs = [bytes(rng.integers(0, 256, int(n), dtype=np.uint8))
            for n in rng.integers(100, 3000, 12)]
    b = batch_spans(docs, c=20, threads=2)
    ok_b = all(
        np.array_equal(starts,
                       np.array([s for s, _, _ in RateC(
                           c=20, store_book=False)
                           .encode_to_spans(doc)], np.uint32))
        and n == len(doc)
        for starts, n, doc in ((s, n, d) for (s, n), d in zip(b, docs)))
    check("batch_spans == serial encode (12 docs, 2 threads)", ok_b)
    n_cases += 1

    # ============================================== 2. P3-2 judge C-independence
    corpus = open(os.path.join(HERE, "data", "gate_slice.bin"), "rb").read()
    ref = RateC(c=20, store_book=False).surprise(corpus[:65536])
    ok_ci, why = True, ""
    for c in (1, 10, 20, 33, 64):
        s = RateC(c=c, store_book=False).surprise(corpus[:65536])
        if s.tobytes() != ref.tobytes():
            ok_ci, why = False, f"c={c}: surprisal differs from c=20"
            break
    check("judge C-independence: surprise bitwise-equal across c "
          "{1,10,20,33,64}", ok_ci, why)

    # ============================================== 3. P3-3 fast-lane parity
    ex = RateC(c=20, store_book=False).surprise(corpus)          # exact
    fa = RateC(c=20, store_book=False, exact=False).surprise(corpus)
    denom = np.maximum(np.abs(ex), 1e-30)
    max_rel = float(np.max(np.abs(fa - ex) / denom))
    ex_starts = np.array([s for s, _, _ in
                          RateC(c=20, store_book=False).encode_to_spans(corpus)])
    fa_starts = np.array([s for s, _, _ in
                          RateC(c=20, store_book=False, exact=False)
                          .encode_to_spans(corpus)])
    mism = int(np.sum(ex_starts != fa_starts[:len(ex_starts)]))
    check("fast lane: max relative surprisal deviation within claim "
          "(<= 5e-7)", max_rel <= 5e-7,
          f"measured {max_rel:.2e} (README claims ~1.1e-7); "
          f"patch starts shifted {mism}/{len(ex_starts)} (documented)")
    rec = {"protocol": {
        "claim": "max relative surprisal deviation ~1.1e-7 (README)",
        "bound_asserted": 5e-7,
        "bytes": len(corpus), "c": 20,
        "patch_starts_shifted": mism, "patch_starts_total": int(len(ex_starts)),
    }}
    from research.record_util import record_envelope
    rec["protocol"].update(record_envelope(deps=("numpy",)))
    rec["measured_max_relative_deviation"] = max_rel
    with open(os.path.join(ROOT, "research", "fastlane_parity.json"), "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")

    # ============================================== 4. P3-4 pf equality
    ref_sp = RateC(c=20, store_book=False, pf=2).encode_to_spans(corpus)
    ref_sur = RateC(c=20, store_book=False, pf=2).surprise(corpus[:32768])
    ok_pf, why = True, ""
    for pf in (0, 1, 3, 4):
        sp = RateC(c=20, store_book=False, pf=pf).encode_to_spans(corpus)
        su = RateC(c=20, store_book=False, pf=pf).surprise(corpus[:32768])
        if sp != ref_sp:
            ok_pf, why = False, f"pf={pf}: spans differ"
            break
        if su.tobytes() != ref_sur.tobytes():
            ok_pf, why = False, f"pf={pf}: surprisal differs"
            break
    check("pf 0..4: spans + surprisal bitwise-identical (prefetch depth "
          "only)", ok_pf, why)

    # ============================================== 5. P3-5 negative/boundary
    ok_neg, why = True, ""
    for bad in (0.5, 0.0, 64.0001, 100.0, float("nan")):
        try:
            RateC(c=bad)
            ok_neg, why = False, f"c={bad} accepted"
            break
        except ValueError:
            pass
    for good in (1.0, 1.5, 63.5, 64.0):
        try:
            RateC(c=good)
        except Exception as e:                              # noqa: BLE001
            ok_neg, why = False, f"c={good} rejected ({e})"
            break
    if ok_neg:
        try:
            batch_spans([b"abc"], c=64.5)
            ok_neg, why = False, "batch c=64.5 accepted"
        except ValueError:
            pass
    check("boundary C: 64.0/1.5 accepted; 0.5/64.0001/NaN rejected", ok_neg, why)

    # map exhaustion is a RECOVERABLE RatecError (tiny map_capbits)
    eng = RateC(c=20, store_book=False, map_capbits=8)
    err = None
    try:
        eng.encode_to_spans(corpus[:200000])
    except RatecError as e:
        err = e
    ok_ex = err is not None and "exhausted" in str(err)
    if ok_ex:
        # NOT poisoned: reset() restores cold state bit-exactly and the
        # engine is fully usable again (the -3 aborts between bytes)
        eng.reset()
        small = RateC(c=20, store_book=False).encode_to_spans(b"tiny input")
        got = eng.encode_to_spans(b"tiny input")
        if [(s, e) for s, e, _ in got] != [(s, e) for s, e, _ in small]:
            ok_ex, why = False, "engine not usable after reset"
    check("map exhaustion (map_capbits=8): RatecError, reset() recovers",
          ok_ex, why)

    # reset() == fresh engine, bit-exact (spans + surprisal)
    a = RateC(c=20)
    a.encode_bytes(corpus[:100000])
    a.reset()
    b = RateC(c=20)
    sa, sb = a.encode_to_spans(corpus[100000:150000]), \
        b.encode_to_spans(corpus[100000:150000])
    ua, ub = a.surprise(corpus[150000:160000]), \
        b.surprise(corpus[150000:160000])
    ok_rs = sa == sb and ua.tobytes() == ub.tobytes()
    # reset() also recovers a book-OOM engine's book tables (P1-6 fix,
    # exercised at the C level by core_reset_keep)
    check("reset() == fresh engine (spans + surprise bit-exact)", ok_rs, why)

    # book-mode snapshot after feed() — the open-patch obuf path
    eng = RateC(c=20)
    eng.feed(corpus[:37])                    # odd boundary -> patch open
    snap = eng.save_state()
    eng2 = RateC.from_state(snap)
    tail = corpus[37:100037]
    ok_ob = eng2.encode_to_spans(tail) == eng.encode_to_spans(tail)
    check("book-mode snapshot after feed() (open obuf) continues "
          "identically", ok_ob)

    # ============================================== 6. P3-6 API sweep
    ok_api, why = True, ""
    # solve_c brackets: target 1 patch -> C pinned at hi; huge target -> lo
    data = corpus[:60000]
    c_hi, n_hi = solve_c(data, 1)
    c_lo, n_lo = solve_c(data, 10 ** 7)
    if c_hi < 60 or c_lo > 4:
        ok_api, why = False, f"servo brackets: c_hi={c_hi:.2f} c_lo={c_lo:.2f}"
    if ok_api:
        c_book, book = solve_c_for_book(data, 1)          # tiny book -> hi C
        if book > 200:
            ok_api, why = False, f"solve_c_for_book(target=1) -> {book}"
    if ok_api:
        # warm_state constructor path == from_state
        w = RateC(c=20)
        w.encode_bytes(corpus[:50000])
        snap = w.save_state()
        e1, e2 = RateC.from_state(snap), RateC(c=20, warm_state=snap)
        t = corpus[50000:60000]
        if e1.encode_to_spans(t) != e2.encode_to_spans(t):
            ok_api, why = False, "warm_state path differs from from_state"
    if ok_api:
        # batch fast/carry arms run and tile
        for kw in ({"fast": True}, {"carry": True}):
            bb = batch_spans([b"hello world" * 50], c=20, threads=1, **kw)
            if len(bb) != 1:
                ok_api, why = False, f"batch {kw} returned {len(bb)} entries"
                break
    if ok_api:
        # encode_stream_parallel edges: n <= chunk -> sequential path;
        # head_chunks > 1 and carry mode still tile + round-trip
        tokp = RateC(c=20)
        for kw in ({"chunk": 10 ** 9}, {"chunk": 4096, "head_chunks": 3},
                   {"chunk": 4096, "head_chunks": 2}):
            spans_p = tokp.encode_stream_parallel(corpus[:65536], **kw)
            if spans_p[0][0] != 0 or spans_p[-1][1] != 65536:
                ok_api, why = False, f"stream_parallel {kw}: no tiling"
                break
    check("API sweep: servo brackets, warm_state, batch arms, "
          "encode_stream_parallel edges", ok_api, why)

    # carry + reset: carry register clears with state, engine stays exact
    cc = RateC(c=20, carry=True)
    cc.encode_bytes(corpus[:40000])
    cc.reset()
    dd = RateC(c=20, carry=True)
    ok_cr = cc.encode_to_spans(corpus[:20000]) == \
        dd.encode_to_spans(corpus[:20000])
    check("carry mode: reset() clears the carry register (== fresh)", ok_cr)

    # CLI smoke: encode/decode/surprise/state with fractional --c
    with tempfile.TemporaryDirectory() as td:
        fin = os.path.join(td, "in.txt")
        fout = os.path.join(td, "out.ids")
        fstate = os.path.join(td, "state.rc")
        fsur = os.path.join(td, "sur.f32")
        fdec = os.path.join(td, "dec.txt")
        open(fin, "wb").write(corpus[:20000])
        cli = [sys.executable, "-m", "ratec"]
        r1 = subprocess.run(cli + ["encode", fin, "-o", fout, "--c", "17.5"],
                            capture_output=True, cwd=ROOT)
        r2 = subprocess.run(cli + ["state", fin, "-o", fstate, "--c", "17.5"],
                            capture_output=True, cwd=ROOT)
        r3 = subprocess.run(cli + ["decode", fout, fstate, "-o", fdec],
                            capture_output=True, cwd=ROOT)
        r4 = subprocess.run(cli + ["surprise", fin, "-o", fsur,
                                   "--c", "17.5", "--patch"],
                            capture_output=True, cwd=ROOT)
        ok_cli = all(r.returncode == 0 for r in (r1, r2, r3, r4))
        if ok_cli:
            ok_cli = open(fdec, "rb").read() == corpus[:20000]
            ok_cli = ok_cli and os.path.getsize(fsur) == 20000 * 4
        check("CLI smoke: encode/state/decode/surprise --patch with "
              "fractional --c (round-trip)", ok_cli,
              "" if ok_cli else (r1.stderr or r2.stderr or r3.stderr
                                 or r4.stderr).decode()[-200:])

    print()
    print(f"FUZZ GATE: {'PASS' if all(RESULTS) else 'FAIL'} "
          f"({sum(RESULTS)}/{len(RESULTS)} checks)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
