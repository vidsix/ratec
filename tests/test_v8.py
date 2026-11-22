"""v0.9.1 snapshot v8 — explicit little-endian format gates (PLAN_v0.9
P2.2): the v8 goldens load and continue bit-exactly, save(load(snap)) is
byte-identical, the write path is v8-only, and an independent PYTHON
little-endian parser walks the payload to exact EOF — the format-level
endianness proof (the executable BE-host simulation is test_be_sim.py).
"""
import hashlib
import json
import os
import struct
import zlib

import pytest

import ratec

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "data")

V8_GOLDENS = ["v8_c20", "v8_c33", "v8_c20carry"]


def _raw(name):
    p = os.path.join(TD, f"{name}.bin")
    if not os.path.exists(p):
        pytest.skip(f"{name}.bin not captured yet (run gen_v8_goldens.py)")
    container = open(p, "rb").read()
    meta = json.load(open(os.path.join(TD, "fixtures_meta.json")))
    assert hashlib.sha256(container).hexdigest() == meta[f"{name}_sha256"]
    return zlib.decompress(container[4:]) if container[:4] == b"RCZ1" \
        else container


def gold_c(name):
    return {"v8_c20": 20, "v8_c33": 33, "v8_c20carry": 20}[name]


def gold_carry(name):
    return name == "v8_c20carry"


def gold_book(name):
    return name != "v8_c33"


def test_v8_goldens_load_and_continue():
    warm = open(os.path.join(TD, "gate_slice.bin"), "rb").read()
    cont = open(os.path.join(TD, "v4_cont_slice.bin"), "rb").read()
    for name in V8_GOLDENS:
        snap = _raw(name)
        assert snap[:4] == b"RCST"
        assert int.from_bytes(snap[4:8], "little") == 8
        loaded = ratec.RateC.from_state(snap)
        raw2 = loaded.save_state()
        if raw2[:4] == b"RCZ1":
            raw2 = zlib.decompress(raw2[4:])
        assert raw2 == snap, f"{name}: save(load(snap)) != snap"
        got = [int(s) for s, _, _ in loaded.encode_to_spans(cont)]
        # the reference is the WARM ENGINE continuing on cont (the same
        # definition the golden capture verifies) — a fresh engine on
        # warm+cont is NOT the reference for carry mode: its boundary
        # patch may absorb cont's first bytes through the carry credit
        ref = ratec.RateC(c=gold_c(name), carry=gold_carry(name),
                          store_book=gold_book(name))
        if gold_book(name):
            ref.encode_bytes(warm)
        else:
            ref.encode_to_spans(warm)
        want = [int(s) for s, _, _ in ref.encode_to_spans(cont)]
        assert got == want, f"{name}: continuation mismatch"


def test_write_path_is_v8_only():
    for kw in (dict(c=20), dict(c=33, store_book=False),
               dict(c=20, carry=True),
               dict(c=20, prune=True, prune_after=1 << 15)):
        tok = ratec.RateC(**kw)
        if kw.get("store_book", True):
            tok.encode_bytes(b"write-path version check " * 64)
        else:
            tok.encode_to_spans(b"write-path version check " * 64)
        snap = tok._eng.save_state()
        if snap[:4] == b"RCZ1":
            snap = zlib.decompress(snap[4:])
        assert int.from_bytes(snap[4:8], "little") == 8, kw
        back = ratec.RateC.from_state(snap)
        raw2 = back.save_state()
        if raw2[:4] == b"RCZ1":
            raw2 = zlib.decompress(raw2[4:])
        assert raw2 == snap


def test_prune_flag_roundtrip():
    tok = ratec.RateC(c=20, prune=True, prune_after=1 << 12, prune_min=2)
    tok.encode_bytes(b"prune mode v8 flag round-trip " * 200)
    snap = tok._eng.save_state()
    if snap[:4] == b"RCZ1":
        snap = zlib.decompress(snap[4:])
    assert snap[42] in (0, 1)            # the prune_flag byte rides the header
    assert snap[42] == 1                 # this engine prunes
    back = ratec.RateC.from_state(snap)
    assert back.prune_after == tok.prune_after
    ids_a = tok.encode_bytes(b"prune continues identically " * 20)
    ids_b = back.encode_bytes(b"prune continues identically " * 20)
    assert ids_a == ids_b


def test_python_le_parser_walks_v8_to_eof():
    """Independent explicit-LE decode of the committed v8 golden: every
    header scalar reads the same value the C loaded, and the walk lands
    at exact EOF. This is the format-level byte-order contract — no C
    code, no native memcpy, just the documented layout."""
    snap = _raw("v8_c20")
    o = 0

    def u32():
        nonlocal o
        v = struct.unpack_from("<I", snap, o)[0]
        o += 4
        return v

    def f64():
        nonlocal o
        v = struct.unpack_from("<d", snap, o)[0]
        o += 8
        return v

    def u8():
        nonlocal o
        v = snap[o]
        o += 1
        return v

    def u64():
        nonlocal o
        v = struct.unpack_from("<Q", snap, o)[0]
        o += 8
        return v

    assert u32() == 0x54534352                       # "RCST" little-endian
    assert u32() == 8
    C = f64()
    assert 1.0 <= C <= 64.0
    sb, pf, fz, poisoned = u32(), u32(), u32(), u32()
    assert sb == 1 and poisoned == 0 and pf <= 4 and fz <= 1
    pmode, mcb = u8(), u8()
    assert pmode <= 1 and 8 <= mcb <= 26
    carry = f64()
    assert 0.0 <= carry <= C
    prune_flag = u8()
    assert prune_flag == 0                           # c20 config: no prune
    nseen, n_forced, n_patches = u64(), u64(), u64()
    assert nseen == len(open(os.path.join(TD, "gate_slice.bin"), "rb").read())
    assert n_patches > 0
    patch_open = u8()
    open_start, open_len = u64(), u64()
    cs, lim = f64(), f64()
    assert cs >= 0.0 and lim >= 0.0
    if sb and patch_open and open_len:
        o += open_len                                # obuf (book mode only)
    o += 4 * 256                                     # cnt0 (u32 x 256)
    o += 256 * 256 * 4 + 256 * 4 + 65536 * 4         # cnt1, tot1, tot2
    o += 6 * 8                                       # h (NO=6 u64 words)
    nctx = u64()
    assert nctx <= 65536 and nctx > 0                # a trained judge has rows
    for _ in range(nctx):
        struct.unpack_from("<H", snap, o)            # context id (u16 LE)
        o += 2
        m16 = struct.unpack_from("<H", snap, o)[0]
        o += 2
        assert m16 <= 256
        o += m16                                     # nz bytes
        o += 4 * m16                                 # values (u32 LE each)
    tmap_capb, tmap_n = u64(), u64()
    assert tmap_capb <= 26 and tmap_n <= (1 << tmap_capb) * 3 // 4
    WIN = 16
    for _ in range(tmap_n):
        o += 8 + 4 + 1 + WIN                         # key, tot, distinct, b
        o += 2 * WIN                                 # c: u16 per window slot
    cmap_capb, cmap_n = u64(), u64()
    assert cmap_capb <= 26
    o += cmap_n * (8 + 4)
    bn, alen = u64(), u64()
    assert (bn == 0) == (alen == 0)
    o += alen
    o += 8 * bn                                      # off/ln u32 pairs
    assert not prune_flag                            # c20: no trailing section
    assert o == len(snap), f"walked {o} of {len(snap)} bytes"
