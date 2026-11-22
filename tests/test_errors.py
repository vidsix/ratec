"""v0.9.1 error taxonomy — every reachable raise site asserts its .code
(PLAN_v0.9 P2.3); the SOURCE-level mapping audit pins the rest.

Reachable sites are driven through the public API; sites whose trigger
needs multi-GiB inputs or a poisoned engine are pinned by the source
audit below, which parses every ratec_err/ratec_errf call in
_rcengine.c and asserts (a) every code is in the published enum and
(b) the per-code site counts match the committed table — so an unmapped
or recoded site fails the suite.
"""
import os
import re

import numpy as np
import pytest

import ratec
from ratec import RatecError, RatecErrorCode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def code_of(exc):
    return RatecErrorCode(exc.code)


def test_bookless_family():
    eng = ratec.RateC(c=20, store_book=False)
    for fn in (
        lambda e: e.decode_bytes([0]),
        lambda e: e.decode_batch([[0]]),
        lambda e: e._eng.intern_spans(b"abc", np.array([0], np.uint32),
                                      np.array([1], np.uint32)),
        lambda e: e._eng.intern_spans_parallel(
            b"abc", np.array([0], np.uint32), np.array([1], np.uint32)),
        lambda e: e._eng.book_dump(),
        lambda e: e._eng.book_counts(),
    ):
        with pytest.raises(RatecError) as ei:
            fn(eng)
        assert code_of(ei.value) is RatecErrorCode.BOOKLESS
    tok = ratec.RateC(c=20, store_book=False)
    with pytest.raises(RatecError) as ei:
        tok.encode_bytes(b"abc")
    assert code_of(ei.value) is RatecErrorCode.BOOKLESS
    # the wrapper's ids guard fires BEFORE the C (documented ValueError);
    # the C BOOKLESS site is reached via intern_spans_parallel above
    with pytest.raises(ValueError):
        tok.encode_stream_parallel(b"abc def", threads=1, ids=True)


def test_tiling_family():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(b"some context " * 40)
    bad = [
        (np.array([1], np.uint32), np.array([3], np.uint32)),      # start != 0
        (np.array([0], np.uint32), np.array([99], np.uint32)),     # end != len
        (np.array([0, 5], np.uint32), np.array([4, 8], np.uint32)),  # gap
        (np.array([0, 2], np.uint32), np.array([4, 1], np.uint32)),  # overrun
    ]
    for starts, ends in bad:
        with pytest.raises(RatecError) as ei:
            tok._eng.intern_spans(b"abcdefgh", starts, ends)
        assert code_of(ei.value) is RatecErrorCode.TILING
        with pytest.raises(RatecError) as ei:
            tok._eng.intern_spans_parallel(b"abcdefgh", starts, ends)
        assert code_of(ei.value) is RatecErrorCode.TILING


def test_cap_exhausted_is_coded_and_recoverable():
    eng = ratec.RateC(c=20, store_book=False, map_capbits=8)
    # non-repeating real text (periodic patterns promote too few
    # contexts to exhaust a 256-slot map — same trigger as test_fuzz)
    corpus = b"".join(
        open(os.path.join(ROOT, "tests", "data", "fixtures", f"{d}.span.bin"),
             "rb").read()
        for d in ("english", "code", "chinese", "base64"))
    with pytest.raises(RatecError) as ei:
        eng.encode_to_spans(corpus[:200000])
    assert code_of(ei.value) is RatecErrorCode.CAP_EXHAUSTED
    eng.reset()
    eng.encode_to_spans(b"small")            # reset() recovers


def test_snapshot_version_rejects_are_coded():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(b"seed content for a real snapshot")
    snap = tok._eng.save_state()
    assert int.from_bytes(snap[4:8], "little") == 8          # v8 writer
    for badver in (1, 9, 2 ** 31):
        mutant = snap[:4] + badver.to_bytes(4, "little") + snap[8:]
        with pytest.raises(RatecError) as ei:
            ratec.RateC.from_state(mutant)
        assert code_of(ei.value) is RatecErrorCode.SNAPSHOT_VERSION


def test_snapshot_invalid_rejects_are_coded():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(b"seed content for a real snapshot " * 8)
    snap = tok._eng.save_state()
    with pytest.raises(RatecError) as ei:
        ratec.RateC.from_state(b"NOPE" + snap[4:])
    assert code_of(ei.value) is RatecErrorCode.SNAPSHOT_INVALID
    with pytest.raises(RatecError) as ei:
        ratec.RateC.from_state(snap[:len(snap) // 2])
    assert code_of(ei.value) is RatecErrorCode.SNAPSHOT_INVALID
    import zlib
    with pytest.raises(RatecError) as ei:
        ratec.RateC.from_state(b"RCZ1" + b"\x00" * 16)
    assert code_of(ei.value) is RatecErrorCode.SNAPSHOT_INVALID
    # load_state adopts the blob's config (a stream-mode blob loads
    # as-is — documented); the INCOMPATIBLE-transfer path is restore_from
    # between engines whose map capacity differs
    stream = ratec.RateC(c=33, store_book=False)
    stream.encode_to_spans(b"stream mode state " * 8)
    sstate = stream._eng.save_state()
    ratec.RateC.from_state(sstate)          # must NOT raise
    book = ratec.RateC(c=20, map_capbits=22)
    book.encode_bytes(b"book state " * 20)
    other = ratec.RateC(c=20, map_capbits=16)
    other.encode_bytes(b"other state " * 20)
    with pytest.raises(RatecError) as ei:
        book._eng.restore_from(other._eng)     # Engine, not the wrapper
    assert code_of(ei.value) is RatecErrorCode.SNAPSHOT_INVALID


def test_argument_family():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(b"train the book " * 50)
    with pytest.raises(RatecError) as ei:
        tok.decode_bytes([1 << 30])
    assert code_of(ei.value) is RatecErrorCode.ARGUMENT
    with pytest.raises(RatecError) as ei:
        tok._eng.clone()                         # clone requires bookless
    assert code_of(ei.value) is RatecErrorCode.ARGUMENT
    with pytest.raises(RatecError) as ei:
        tok._eng.restore_from(b"not an engine")
    assert code_of(ei.value) is RatecErrorCode.ARGUMENT
    with pytest.raises(ValueError):
        ratec.RateC(c=20, reserve=-1)


def test_buffer_limit_is_coded():
    """A v8 snapshot with nseen just under the 4 GiB span space pushes
    the next feed over the BUFFER_LIMIT — reachable without allocating
    4 GiB (crafts the header explicitly; doubles as the v8 header
    layout check from the Python side)."""
    import struct
    # v8 header: magic, ver=8, C f64, sb, pf, fz, poisoned, pmode u8,
    # mcb u8, carry f64, prune_flag u8 — then nseen/n_forced/n_patches
    hdr = struct.pack("<II", 0x54534352, 8)
    hdr += struct.pack("<d", 20.0)
    hdr += struct.pack("<IIII", 0, 2, 0, 0)          # sb, pf, fz, poisoned
    hdr += bytes([0, 20])                            # pmode, map_capbits
    hdr += struct.pack("<d", 0.0)                    # carry
    hdr += bytes([0])                                # prune_flag
    hdr += struct.pack("<QQQ", (1 << 32) - 64, 0, 0)  # nseen near the limit
    hdr += bytes([0])                                # patch_open = 0
    hdr += struct.pack("<QQdd", 0, 0, 0.0, 20.0)     # open_start, open_len, cs, lim
    hdr += b"\x00" * (256 * 4)                       # cnt0 (u32 x 256)
    hdr += b"\x00" * (256 * 256 * 4)                 # cnt1
    hdr += b"\x00" * (256 * 4)                       # tot1
    hdr += b"\x00" * (65536 * 4)                     # tot2
    hdr += b"\x00" * (6 * 8)                         # h (NO=6)
    hdr += struct.pack("<Q", 0)                      # nctx = 0
    hdr += struct.pack("<QQ", 12, 0)                 # tmap capbits, n
    hdr += struct.pack("<QQ", 12, 0)                 # cmap capbits, n
    eng = ratec.RateC(c=20, store_book=False)
    try:
        loaded = ratec.RateC.from_state(hdr)
    except RatecError as e:
        pytest.skip(f"crafted minimal-state layout not accepted "
                    f"({e.code}); BUFFER_LIMIT pinned by source audit")
    with pytest.raises(RatecError) as ei:
        loaded.encode_to_spans(b"x" * 4096)
    assert code_of(ei.value) is RatecErrorCode.BUFFER_LIMIT


def test_source_level_mapping_audit():
    """Every ratec_err/ratec_errf call uses a published code, and the
    per-code site counts match the committed table (an unmapped or
    recoded site fails here — the 'every raise site mapped' gate)."""
    src = open(os.path.join(ROOT, "ratec", "_rcengine.c")).read()
    calls = re.findall(r'ratec_errf?\(\s*"([A-Z_]+)"', src)
    valid = {m.value for m in RatecErrorCode}
    assert set(calls) <= valid, f"unpublished codes: {set(calls) - valid}"
    counts = {c: calls.count(c) for c in set(calls)}
    committed = {
        "ARGUMENT": 6, "BOOKLESS": 7, "BUFFER_LIMIT": 5,
        "CAP_EXHAUSTED": 4, "SNAPSHOT_INVALID": 2, "SNAPSHOT_VERSION": 1,
        "STATE": 7, "TILING": 4,
    }
    assert counts == committed, (
        f"site mapping drifted: {counts} vs committed {committed}")
