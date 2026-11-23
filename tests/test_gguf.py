"""GGUF bridge tests (PLAN_v0.9 P3.2): export_gguf writes a valid GGUF
v3 container carrying the ratec book as tokenizer metadata; an
INDEPENDENT pure-Python GGUF parser in this test walks the container
back and asserts vocab equality + the decode contract. llama.cpp
itself is not runnable here — the container validity + content
equality IS the gate; the documented subset is stated in the
exporter's docstring and BENCHMARKS.md.
"""
import base64
import hashlib
import struct

import pytest

import numpy as np

import ratec
from ratec import RatecError, RatecErrorCode

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types (gguf-py GGUFType / llama.cpp enum gguf_type)
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL = range(8)
T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = 8, 9, 10, 11, 12

SZ = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
      T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}


def _read_str(buf, o):
    (n,) = struct.unpack_from("<Q", buf, o)
    o += 8
    s = buf[o:o + n].decode("utf-8")
    return s, o + n


def _read_val(buf, o, typ):
    if typ == T_STRING:
        return _read_str(buf, o)
    if typ == T_ARRAY:
        et, n = struct.unpack_from("<IQ", buf, o)
        o += 12
        vals = []
        for _ in range(n):
            v, o = _read_val(buf, o, et)
            vals.append(v)
        return vals, o
    fmt = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h",
           T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<B",
           T_U64: "<Q", T_I64: "<q", T_F64: "<d"}[typ]
    (v,) = struct.unpack_from(fmt, buf, o)
    return v, o + SZ[typ]


def parse_gguf(buf):
    """Independent GGUF v3 reader: magic, version, KVs (tensor_count=0
    for the ratec bridge — no tensor infos, no data block)."""
    assert buf[:4] == GGUF_MAGIC
    (ver,) = struct.unpack_from("<I", buf, 4)
    assert ver == GGUF_VERSION
    tensor_count, kv_count = struct.unpack_from("<QQ", buf, 8)
    o = 24
    assert tensor_count == 0, "the ratec bridge carries metadata only"
    kvs = {}
    for _ in range(kv_count):
        key, o = _read_str(buf, o)
        (typ,) = struct.unpack_from("<I", buf, o)
        o += 4
        val, o = _read_val(buf, o, typ)
        kvs[key] = val
    assert o == len(buf), f"trailing bytes after KVs: {len(buf) - o}"
    return kvs


def test_gguf_export_roundtrip():
    tok = ratec.RateC(c=20)
    tok.encode_bytes(b"a small training corpus for the gguf bridge " * 40)
    tok.encode_bytes(b"the quick brown fox jumps over the lazy dog")
    blob = ratec.export_gguf(tok)      # snapshot taken at THIS state
    kvs = parse_gguf(blob)

    entries = tok._eng.book_dump()
    tokens = kvs["tokenizer.ggml.tokens"]
    assert len(tokens) == len(entries)
    for i, piece in enumerate(entries):
        assert tokens[i] == piece.decode("latin-1")
    assert kvs["tokenizer.ggml.model"] == "ratec-adaptive"
    assert kvs["ratec.format.version"] == 1
    assert abs(kvs["ratec.c"] - 20.0) < 1e-9
    assert kvs["ratec.book.count"] == len(entries)
    assert kvs["general.architecture"] == "ratec"
    # the embedded snapshot is the engine's own state, sha-bound
    snap = base64.b64decode(kvs["ratec.snapshot.v8"])
    assert hashlib.sha256(snap).hexdigest() == kvs["ratec.snapshot.sha256"]
    # snapshot-reload semantics: the reloaded engine CONTINUES exactly
    # like the source engine from the same state (ids are warm-book ids)
    back = ratec.RateC.from_state(snap)
    probe = b"second stream continuation probe bytes"
    ids2 = back.encode_bytes(probe)
    ids3 = tok.encode_bytes(probe)
    assert list(map(int, ids2)) == list(map(int, ids3))

    # types as the documented subset promises
    assert kvs["tokenizer.ggml.token_type"] == [1] * len(entries)
    assert all(s == 0.0 for s in kvs["tokenizer.ggml.scores"])


def test_gguf_bookless_raises():
    import pytest
    with pytest.raises(RatecError) as ei:
        ratec.export_gguf(ratec.RateC(c=20, store_book=False))
    assert RatecErrorCode(ei.value.code) is RatecErrorCode.BOOKLESS
