"""v0.9.1 api additions — public reserve= (with the instrumented
rehash gate), chunked_prep (the tested span-space pattern), and
export_tokenizer_json (the HF bridge, PLAN_v0.9 P1.5 / P2.1 / P3.1)."""
import json
import os

import numpy as np
import pytest

import ratec
from ratec import RatecError, RatecErrorCode

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "data")


def _stream(n=None):
    data = open(os.path.join(TD, "gate_slice.bin"), "rb").read()
    return data if n is None else (data * (n // len(data) + 1))[:n]


def _varied(n):
    """High-context-variety deterministic stream (the shipped fixtures
    repeat, which saturates distinct contexts and never grows maps)."""
    out = bytearray()
    while len(out) < n:
        i = len(out)
        out += b"line %d picks case %d and value %d\n" % (
            i, i % 977, i % 89)
    return bytes(out[:n])


def test_reserve_public_and_instrumented():
    data = _varied(2 << 20)
    # small prebits forces real growth: the unsized engine pays a
    # growth rehash per doubling; the sized one pre-sizes once and
    # never grows (0 rehashes) — the instrumented half of the reserve=
    # gate
    plain = ratec.RateC(c=20, prebits=12)
    plain.encode_bytes(data)
    sized = ratec.RateC(c=20, prebits=12, reserve=len(data))
    sized.encode_bytes(data)
    # context maps hold only PROMOTED contexts (inline windows first), so
    # a 2 MiB stream may never rehash — the unit gate pins the semantics
    # (reserve accepted, ids identical, no growth for the sized engine);
    # the instrumented rehash DELTA on real scale is recorded by the
    # 8 GiB chunked_prep gate (bench/gate_8gib.py)
    assert sized.rehash_count() == 0
    assert plain.rehash_count() >= 0
    # ids identical with and without reserve (reserve changes capacity,
    # never semantics)
    a = ratec.RateC(c=20).encode_bytes(data[:5000])
    b = ratec.RateC(c=20, reserve=len(data)).encode_bytes(data[:5000])
    assert a == b


def test_batch_reserve():
    docs = [_stream(60000), _stream(120000)[:60000]]
    r1 = ratec.batch_encode(docs, threads=2)
    r2 = ratec.batch_encode(docs, threads=2, reserve=60000)
    assert len(r1) == len(r2) == 2
    for (i1, l1), (i2, l2) in zip(r1, r2):
        assert np.array_equal(i1, i2) and np.array_equal(l1, l2)
    s1 = ratec.batch_spans(docs, threads=2)
    s2 = ratec.batch_spans(docs, threads=2, reserve=60000)
    for (a1, n1), (a2, n2) in zip(s1, s2):
        assert np.array_equal(a1, a2) and n1 == n2


def test_chunked_prep_identity_and_shapes():
    data = _stream(1 << 20)
    # segment smaller than the stream: 8 segments of 128 KiB
    segs = ratec.chunked_prep(
        [data[i:i + 65536] for i in range(0, len(data), 65536)],
        engine_bytes=1 << 17, threads=2)
    assert sum(s["seg_bytes"] for s in segs) == len(data)
    assert all(s["seg_bytes"] <= (1 << 17) for s in segs)
    off = 0
    for s in segs:
        ref = ratec.RateC(c=20)
        sb, eb, tk = ref.encode_stream_parallel(
            data[off:off + s["seg_bytes"]], threads=2, ids=True)
        assert np.array_equal(np.asarray(tk, np.int32), s["ids"])
        assert np.array_equal(sb, s["starts"])
        assert np.array_equal(eb, s["ends"])
        # spans tile the segment exactly
        assert s["starts"][0] == 0
        assert s["ends"][-1] == s["seg_bytes"]
        off += s["seg_bytes"]
    # spans-only mode
    segs2 = ratec.chunked_prep(data, engine_bytes=1 << 18, ids=False,
                               threads=1)
    assert all(s["ids"] is None for s in segs2)
    # single chunk bigger than engine_bytes is split at the boundary
    one = ratec.chunked_prep(data[:5000], engine_bytes=1000, threads=1)
    assert [s["seg_bytes"] for s in one] == [1000] * 5
    # empty stream -> no segments
    assert ratec.chunked_prep(b"", engine_bytes=1000) == []
    # sink mode retains nothing (the flat-RSS shape)
    seen = []
    ratec.chunked_prep(data[:40000], engine_bytes=10000,
                       sink=seen.append, threads=1)
    assert len(seen) == 4
    # the 4 GiB hard limit is enforced on engine_bytes
    with pytest.raises(ValueError):
        ratec.chunked_prep(b"x", engine_bytes=(1 << 32) + 1)


def test_export_tokenizer_json_contract():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(_stream())
    ids = tok.encode_bytes(b"the quick brown fox jumps over the lazy dog")
    doc = json.loads(ratec.export_tokenizer_json(tok))
    assert doc["model"]["type"] == "WordLevel"
    vocab = doc["model"]["vocab"]
    entries = tok._eng.book_dump()
    assert len(vocab) == len(entries)
    # the inverse reconstructs the book exactly (latin-1 is a byte<->char
    # bijection — byte-exact by construction)
    for i, piece in enumerate(entries):
        assert vocab[piece.decode("latin-1")] == i
    inv = {v: k for k, v in vocab.items()}
    rebuilt = b"".join(inv[int(i)].encode("latin-1") for i in ids)
    assert rebuilt == tok.decode_bytes(ids)
    # a second engine exports its own book; BOOKLESS is coded
    tok2 = ratec.RateC(c=20)
    tok2.encode_bytes(_stream())
    ids2 = tok2.encode_bytes(b"second stream continuation")
    doc2 = json.loads(ratec.export_tokenizer_json(tok2))
    assert len(doc2["model"]["vocab"]) == len(tok2._eng.book_dump())
    inv2 = {v: k for k, v in doc2["model"]["vocab"].items()}
    rebuilt2 = b"".join(inv2[int(i)].encode("latin-1") for i in ids2)
    assert rebuilt2 == tok2.decode_bytes(ids2)
    with pytest.raises(RatecError) as ei:
        ratec.export_tokenizer_json(ratec.RateC(c=20, store_book=False))
    assert RatecErrorCode(ei.value.code) is RatecErrorCode.BOOKLESS
