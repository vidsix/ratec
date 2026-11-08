#!/usr/bin/env python3
"""Snapshot-hardening gate (v0.6.0): `load_state` is the attack surface —
arbitrary bytes must raise ratec.RatecError, never crash, never hang.

Deterministic, seeded, no new deps. Four mutant families over the shipped
v6 goldens (raw RCST payloads, decompressed):

  1. truncations       — every proper prefix is invalid (trailing-garbage
                         check); sampled to keep the run fast
  2. seeded bit flips  — the oracle is raise-or-load-clean: a flip inside a
                         don't-care region (e.g. stale window bytes beyond
                         `distinct`) may load, but then save->load->save
                         must round-trip and the engine must still encode;
                         anything else must raise RatecError
  3. targeted fields   — every P1 class from the review, located by parsing
                         the v6 layout: header scalars (C/carry/pf/po/pmode/
                         map_capbits), cnt2 rows (m16 stack bound, nctx),
                         tmap (distinct/tot/c window invariants, cnt vs
                         capacity), cmap (cnt), book (bn/alen/off/ln)
  4. the P1-2 hang     — cnt far above capacity (the crafted-1KB spin with
                         the GIL held) in a subprocess with a timeout

Also asserts save->load->save byte-identity on live engines and that the
RCZ1 container surfaces zlib corruption as RatecError.

Run:  python3 tests/test_snapfuzz.py
Exit 0 only if every check passes.
"""
import json
import hashlib
import os
import struct
import subprocess
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ratec  # noqa: E402
from ratec import RatecError  # noqa: E402

TD = os.path.join(HERE, "data")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------ raw snapshots
def load_raw(name):
    """Committed RCZ1 container -> raw RCST payload bytes."""
    meta = json.load(open(os.path.join(TD, "fixtures_meta.json")))
    blob = open(os.path.join(TD, f"{name}.bin"), "rb").read()
    assert sha(blob) == meta[f"{name}_sha256"], f"{name}: fixture corrupt"
    assert blob[:4] == b"RCZ1"
    return zlib.decompress(blob[4:])


def load_via_api(raw):
    """load raw payload through the module API (returns engine or raises)."""
    return ratec._rcengine.load_state(raw)


def roundtrip_ok(raw):
    """load(raw) must yield a self-consistent, working engine:
    save->load->save of its own state is byte-stable and it still encodes."""
    eng = load_via_api(raw)
    snap2 = eng.save_state()
    if snap2[:4] == b"RCZ1":
        snap2 = zlib.decompress(snap2[4:])
    eng2 = load_via_api(snap2)
    snap3 = eng2.save_state()
    if snap3[:4] == b"RCZ1":
        snap3 = zlib.decompress(snap3[4:])
    if snap3 != snap2:
        return False
    out = eng2.encode_spans(b"round trip probe bytes, nothing exotic")
    return len(out) > 0


# ------------------------------------------------------------ v6 structure
def parse_v6(raw):
    """Walk the v6 layout; return dict of section offsets (for mutants)."""
    o = {"ver": 4, "off_C": 8}
    o["off_sb"], o["off_pf"], o["off_fz"], o["off_poisoned"] = 16, 20, 24, 28
    o["off_pmode"], o["off_mcb"] = 32, 33
    o["off_carry"] = 34
    o["off_nseen"], o["off_nforced"], o["off_npatches"] = 42, 50, 58
    o["off_patch_open"] = 66
    o["off_open_start"], o["off_open_len"] = 67, 75
    o["off_cs"], o["off_lim"] = 83, 91
    at = 99
    if raw[o["off_sb"]]:
        pass  # store_book flag; obuf presence decided below
    po = raw[o["off_patch_open"]]
    olen = struct.unpack_from("<Q", raw, 75)[0]
    sb = raw[o["off_sb"]]
    if sb and po and olen:
        o["off_obuf"] = at
        at += olen
    at += 256 * 4 + 256 * 256 * 4 + 256 * 4 + 65536 * 4 + 48  # fixed tables
    (nctx,) = struct.unpack_from("<Q", raw, at)
    o["off_nctx"] = at
    at += 8
    o["cnt2_rows"] = []
    for _ in range(nctx):
        row_off = at
        (m16,) = struct.unpack_from("<H", raw, at + 2)
        at += 4 + m16 + 4 * m16
        o["cnt2_rows"].append((row_off, m16))
    o["off_tmap_capb"] = at
    (capb, cnt) = struct.unpack_from("<QQ", raw, at)
    at += 16
    o["tmap_capb"], o["tmap_cnt"] = capb, cnt
    o["off_tmap_ent0"] = at
    o["tmap_entsize"] = 8 + 4 + 1 + 16 + 32
    at += cnt * o["tmap_entsize"]
    (capb, cnt) = struct.unpack_from("<QQ", raw, at)
    o["off_cmap_capb"] = at
    o["cmap_capb"], o["cmap_cnt"] = capb, cnt
    at += 16
    o["off_cmap_ent0"] = at
    o["cmap_entsize"] = 12
    at += cnt * 12
    if sb:
        (bn, alen) = struct.unpack_from("<QQ", raw, at)
        o["off_book"] = at
        o["book_bn"], o["book_alen"] = bn, alen
        at += 16
        o["off_book_arena"] = at
        at += alen
        o["off_book_off"] = at
        at += bn * 8                     # off/ln pairs (4+4 each)
    o["end"] = at
    assert at == len(raw), f"parser walked to {at}, file is {len(raw)}"
    return o


# ---------------------------------------------------------------- mutants
def truncations(raw, limit):
    """Yield proper prefixes (generator: one mutant alive at a time)."""
    n = len(raw)
    step = max(1, n // limit)
    for k in range(0, n, step):
        yield raw[:k]
    yield raw[:-1]


def bitflips(raw, count, seed):
    """Yield (pos, mutant) pairs — generator, one mutant alive at a time."""
    x = seed
    n = len(raw)
    for _ in range(count):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        pos = (x >> 16) % n
        bit = (x >> 8) & 7
        b = bytearray(raw)
        b[pos] ^= 1 << bit
        yield pos, bytes(b)


def main():
    print("snapshot fuzz + hardening gate")

    # ---------------------------------------------- 0. live round-trips
    corpus = open(os.path.join(TD, "gate_slice.bin"), "rb").read()
    ok_rt = True
    for kw in (dict(c=20), dict(c=17.5, carry=True), dict(c=8, store_book=False),
               dict(c=64, pf=0), dict(c=1, pf=4)):
        eng = ratec.RateC(**kw)
        eng.encode_to_spans(corpus[:60000])
        raw = eng.save_state()
        if raw[:4] == b"RCZ1":
            raw = zlib.decompress(raw[4:])
        again = load_via_api(raw).save_state()
        if again[:4] == b"RCZ1":
            again = zlib.decompress(again[4:])
        if again != raw:
            ok_rt = False
            print(f"    round-trip drift at {kw}")
    check("save->load->save byte-identity (5 configs)", ok_rt)

    # ---------------------------------------------- 1. committed goldens parse
    snaps = {}
    for name in ("v6_c20", "v6_c33", "v6_c20carry"):
        raw = load_raw(name)
        p = parse_v6(raw)          # asserts the parser walks exactly to end
        snaps[name] = (raw, p)
    check("v6 layout parser walks all 3 goldens to exact EOF", True,
          f"{', '.join(f'{k}: {len(v[0])} B' for k, v in snaps.items())}")

    # v6 golden continuation: load, feed the sha-bound continuation slice,
    # and require the exact absolute patch starts recorded at capture time
    cont = open(os.path.join(TD, "v4_cont_slice.bin"), "rb").read()
    meta = json.load(open(os.path.join(TD, "fixtures_meta.json")))
    assert len(cont) == meta["v4_cont_len"]
    ok_gold, why_gold = True, ""
    for name in ("v6_c20", "v6_c33", "v6_c20carry"):
        gold = json.load(open(os.path.join(TD, f"{name}_cont_golden.json")))
        if sha(cont) != gold["cont_sha256"]:
            ok_gold, why_gold = False, f"{name}: cont slice sha mismatch"
            break
        if int.from_bytes(snaps[name][0][4:8], "little") != 6:
            ok_gold, why_gold = False, f"{name}: snapshot is not v6"
            break
        eng = ratec.RateC.from_state(snaps[name][0])
        got = [int(s) for s, _, _ in eng.encode_to_spans(cont)]
        if got != gold["starts_absolute"]:
            ok_gold, why_gold = False, (
                f"{name}: continuation diverges "
                f"({len(got)} vs {gold['n_patches_cont']} patches)")
            break
    check("v6 goldens: load + bit-exact continuation (c20 book, c33 stream, "
          "c20 carry)", ok_gold, why_gold)

    base_raw, P = snaps["v6_c20"]

    # ---------------------------------------------- 2. targeted field mutants
    def put(raw, off, data):
        b = bytearray(raw)
        b[off:off + len(data)] = data
        return bytes(b)

    M = []  # (label, mutant_bytes, must_raise)
    # header scalars (P1-6)
    M += [
        ("C = NaN", put(base_raw, P["off_C"], struct.pack("<d", float("nan"))), True),
        ("C = 100", put(base_raw, P["off_C"], struct.pack("<d", 100.0)), True),
        ("C = 0.5", put(base_raw, P["off_C"], struct.pack("<d", 0.5)), True),
        ("carry = NaN", put(base_raw, P["off_carry"], struct.pack("<d", float("nan"))), True),
        ("carry = C+1", put(base_raw, P["off_carry"], struct.pack("<d", 21.0)), True),
        ("carry = -1", put(base_raw, P["off_carry"], struct.pack("<d", -1.0)), True),
        ("pf = 9", put(base_raw, P["off_pf"], struct.pack("<I", 9)), True),
        ("po = 2", put(base_raw, P["off_patch_open"], bytes([2])), True),
        ("pmode = 2", put(base_raw, P["off_pmode"], bytes([2])), True),
        ("sb = 2", put(base_raw, P["off_sb"], struct.pack("<I", 2)), True),
        ("fz = 2", put(base_raw, P["off_fz"], struct.pack("<I", 2)), True),
        ("map_capbits = 30", put(base_raw, P["off_mcb"], bytes([30])), True),
        ("map_capbits = 3", put(base_raw, P["off_mcb"], bytes([3])), True),
        ("cs = NaN", put(base_raw, P["off_cs"], struct.pack("<d", float("nan"))), True),
        ("lim = NaN", put(base_raw, P["off_lim"], struct.pack("<d", float("nan"))), True),
        ("cs = -1", put(base_raw, P["off_cs"], struct.pack("<d", -1.0)), True),
        ("bad magic", b"XCST" + base_raw[4:], True),
        ("ver = 7 (future)", put(base_raw, 4, struct.pack("<I", 7)), True),
        ("ver = 1 (too old)", put(base_raw, 4, struct.pack("<I", 1)), True),
    ]
    # cnt2 rows (P1-4 family)
    if P["cnt2_rows"]:
        row_off, m16 = P["cnt2_rows"][0]
        M += [
            ("cnt2 m16 = 300 (stack bound)",
             put(base_raw, row_off + 2, struct.pack("<H", 300)), True),
            ("nctx = 70000",
             put(base_raw, P["off_nctx"], struct.pack("<Q", 70000)), True),
        ]
    # tmap invariants (P1-1, P1-2) — built from the first entry's actual
    # fields, because slot 0 in file order may be inline OR promoted
    e0 = P["off_tmap_ent0"]
    esz = P["tmap_entsize"]
    cap = 1 << P["tmap_capb"]
    d0 = base_raw[e0 + 12]
    (t0,) = struct.unpack_from("<I", base_raw, e0 + 8)
    inline = d0 not in (0, 0xFF)
    M += [
        ("tmap distinct = 17 (past WIN, was OOB)",
         put(base_raw, e0 + 12, bytes([17])), True),
        ("tmap distinct = 200 (was OOB write)",
         put(base_raw, e0 + 12, bytes([200])), True),
        ("tmap distinct = 0 (impossible occupancy)",
         put(base_raw, e0 + 12, bytes([0])), True),
        ("tmap distinct = 254 (not PROMOTED)",
         put(base_raw, e0 + 12, bytes([254])), True),
        ("tmap tot = 0 (occupancy marker)",
         put(base_raw, e0 + 8, struct.pack("<I", 0)), True),
        ("tmap cnt = capacity (P1-2 hang shape)",
         put(base_raw, P["off_tmap_capb"] + 8, struct.pack("<Q", cap)), True),
        ("tmap cnt = 4*capacity (P1-2 hang shape)",
         put(base_raw, P["off_tmap_capb"] + 8,
             struct.pack("<Q", 4 * cap)), True),
        ("tmap capb = 40",
         put(base_raw, P["off_tmap_capb"], struct.pack("<Q", 40)), True),
    ]
    if inline and d0 >= 2:
        M.append(("tmap tot = distinct-1 (under-counted)",
                  put(base_raw, e0 + 8, struct.pack("<I", d0 - 1)), True))
    if inline:
        # Ent layout: key(8) tot(4) distinct(1) b[16] c[16] -> c[0] at +29
        M.append(("tmap c[0] = 0 (empty inline count)",
                  put(base_raw, e0 + 29, struct.pack("<H", 0)), True))
        if t0 < 0xFFFF:
            M.append(("tmap c[0] = tot+1 (over-count)",
                      put(base_raw, e0 + 29, struct.pack("<H", t0 + 1)), True))
    # cmap
    c0 = P["off_cmap_ent0"]
    M += [
        ("cmap cnt = capacity",
         put(base_raw, P["off_cmap_capb"] + 8,
             struct.pack("<Q", 1 << P["cmap_capb"])), True),
        ("cmap tot = 0",
         put(base_raw, c0 + 8, struct.pack("<I", 0)), True),
    ]
    # book (P1-3, P1-5)
    if P["book_bn"] > 0:
        boff = P["off_book_off"]
        off0, ln0 = struct.unpack_from("<II", base_raw, boff)
        M += [
            ("book off beyond alen",
             put(base_raw, boff, struct.pack("<I", P["book_alen"] + 1)), True),
            ("book ln past arena end (from off0)",
             put(base_raw, boff + 4,
                 struct.pack("<I", P["book_alen"] - off0 + 1)), True),
            ("book bn = 2^32-1 (id overflow)",
             put(base_raw, P["off_book"], struct.pack("<Q", 0xFFFFFFFF)), True),
            ("book alen = 2^40 (offset truncation)",
             put(base_raw, P["off_book"] + 8, struct.pack("<Q", 1 << 40)), True),
        ]
    n_struct = 0
    n_loaded = 0
    bad = []
    for label, mut, must_raise in M:
        try:
            load_via_api(mut)
            raised = False
        except RatecError:
            raised = True
        except Exception as e:                     # noqa: BLE001
            bad.append((label, f"wrong exception {type(e).__name__}: {e}"))
            continue
        if raised:
            n_struct += 1
        elif must_raise:
            bad.append((label, "loaded but should have been rejected"))
        else:
            n_loaded += 1
    check(f"targeted mutants: {len(M) - len(bad)}/{len(M)} behave "
          f"({n_struct} rejected, {n_loaded} tolerated)",
          not bad, "; ".join(f"{l}: {d}" for l, d in bad[:4]) or "")

    # ---------------------------------------------- 3. truncations
    ok_tr, bad_tr, n_tried = True, "", 0
    for t in truncations(base_raw, 150):
        n_tried += 1
        try:
            load_via_api(t)
            ok_tr = False
            bad_tr = f"length {len(t)} loaded"
            break
        except RatecError:
            pass
    for t in truncations(snaps["v6_c33"][0], 60):
        n_tried += 1
        try:
            load_via_api(t)
            ok_tr = False
            bad_tr = f"length {len(t)} loaded"
            break
        except RatecError:
            pass
    check(f"truncations: all {n_tried} rejected", ok_tr, bad_tr)

    # ---------------------------------------------- 4. seeded bit flips
    ok_fl, n_raise, n_clean, n_fl, bad_fl = True, 0, 0, 0, ""
    for pos, mut in bitflips(base_raw, 700, 42):
        n_fl += 1
        try:
            eng = load_via_api(mut)
            if not roundtrip_ok(mut):
                ok_fl = False
                bad_fl = f"flip at {pos}: loaded but round-trip drifted"
                break
            n_clean += 1
        except RatecError:
            n_raise += 1
        except Exception as e:                     # noqa: BLE001
            ok_fl = False
            bad_fl = f"flip at {pos}: {type(e).__name__}: {e}"
            break
    if ok_fl:
        for pos, mut in bitflips(snaps["v6_c33"][0], 500, 7):
            n_fl += 1
            try:
                eng = load_via_api(mut)
                if not roundtrip_ok(mut):
                    ok_fl = False
                    bad_fl = f"flip at {pos}: loaded but round-trip drifted"
                    break
                n_clean += 1
            except RatecError:
                n_raise += 1
            except Exception as e:                 # noqa: BLE001
                ok_fl = False
                bad_fl = f"flip at {pos}: {type(e).__name__}: {e}"
                break
    check(f"bit flips: {n_fl} seeded mutants, "
          f"{n_raise} rejected + {n_clean} load-clean",
          ok_fl, bad_fl)

    # ---------------------------------------------- 5. the P1-2 hang probe
    hang = put(base_raw, P["off_tmap_capb"] + 8,
               struct.pack("<Q", 1 << P["tmap_capb"]))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(b"RCZ1" + zlib.compress(hang, 1))
        hang_path = tf.name
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys, zlib; sys.path.insert(0, %r); "
             "import ratec;"
             "raw = zlib.decompress(open(sys.argv[1],'rb').read()[4:]);"
             "ratec._rcengine.load_state(raw)" % (ROOT,),
             hang_path],
            capture_output=True, timeout=30)
    finally:
        os.unlink(hang_path)
    check("P1-2 regression: cnt=capacity rejected in <30s subprocess "
          "(was: infinite probe with GIL held)",
          r.returncode != 0 and b"RatecError" in r.stderr,
          f"exit={r.returncode}")

    # ---------------------------------------------- 6. container + totals
    try:
        ratec.RateC.from_state(b"RCZ1" + b"\x78\x9c" + b"garbage-not-zlib")
        ok_c = False
        why = "zlib garbage loaded"
    except RatecError:
        ok_c = True
        why = ""
    check("RCZ1 container corruption -> RatecError (not zlib.error)", ok_c, why)

    total = len(M) + n_tried + n_fl + 1
    check("mutant corpus >= 1000 cases", total >= 1000, f"{total} cases")

    print()
    print(f"SNAPFUZZ GATE: {'PASS' if all(RESULTS) else 'FAIL'} "
          f"({sum(RESULTS)}/{len(RESULTS)} checks)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
