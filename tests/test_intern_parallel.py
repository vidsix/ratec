"""v0.9.0: the thread-parallel intern pass.

ids must be bit-exactly the sequential pass's for ANY thread count
(chunks-in-order x local-first-seen == global first-seen), warm books
must resolve through the read-only pre-probe, and the api path must
round-trip for intern_threads in {1, 4}.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ratec  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "fixtures"


def _fix(name: str) -> bytes:
    return (DATA / f"{name}.span.bin").read_bytes()


def test_parallel_ids_equal_sequential_all_k():
    data = _fix("english") * 4
    r = ratec.RateC(c=20, store_book=True)
    r.reset()
    starts, ends = r.encode_to_spans(data, as_arrays=True)
    sb, eb = starts.tobytes(), ends.tobytes()
    ids_seq = np.frombuffer(
        r._eng.intern_spans(data, sb, eb), dtype=np.uint32)
    for k in (2, 3, 4, 8, 16):
        ids_par = np.frombuffer(
            r._eng.intern_spans_parallel(data, sb, eb, k), dtype=np.uint32)
        assert np.array_equal(ids_seq, ids_par), f"K={k}"


def test_parallel_ids_thread_count_invariant():
    data = _fix("english") * 40
    r = ratec.RateC(c=20, store_book=True)
    r.reset()
    starts, ends = r.encode_to_spans(data, as_arrays=True)
    sb, eb = starts.tobytes(), ends.tobytes()
    ids2 = np.frombuffer(
        r._eng.intern_spans_parallel(data, sb, eb, 2), dtype=np.uint32)
    ids8 = np.frombuffer(
        r._eng.intern_spans_parallel(data, sb, eb, 8), dtype=np.uint32)
    assert np.array_equal(ids2, ids8)


def test_warm_book_second_intern_identical():
    data = _fix("english") * 4
    r = ratec.RateC(c=20, store_book=True)
    r.reset()
    starts, ends = r.encode_to_spans(data, as_arrays=True)
    sb, eb = starts.tobytes(), ends.tobytes()
    first = np.frombuffer(
        r._eng.intern_spans_parallel(data, sb, eb, 4), dtype=np.uint32)
    # second intern: everything hits through the read-only pre-probe —
    # the ids must come out identical (and the book must not duplicate)
    second = np.frombuffer(
        r._eng.intern_spans_parallel(data, sb, eb, 4), dtype=np.uint32)
    assert np.array_equal(first, second)


def test_api_roundtrip_intern_threads():
    data = _fix("english") * 40
    r = ratec.RateC(c=20, store_book=True)
    for it in (1, 4):
        s, e, tok = r.encode_stream_parallel(
            data, threads=2, ids=True, intern_threads=it)
        assert len(tok) == len(s) == len(e)
        assert tok.dtype == np.uint32
        assert r.decode_bytes(tok) == data


def test_parallel_intern_stress_shapes():
    r = ratec.RateC(c=20, store_book=True)
    # single span
    d = b"hello world"
    ids = np.frombuffer(r._eng.intern_spans_parallel(
        d, np.asarray([0], dtype=np.uint32).tobytes(),
        np.asarray([len(d)], dtype=np.uint32).tobytes(), 4), dtype=np.uint32)
    assert len(ids) == 1 and r.decode_bytes(ids) == d
    # all-identical spans (one distinct patch, many chunks)
    d = b"abc" * 20
    m = 20
    starts = np.arange(m, dtype=np.uint32) * 3
    ends = starts + 3
    ids = np.frombuffer(r._eng.intern_spans_parallel(
        d, starts.tobytes(), ends.tobytes(), 8), dtype=np.uint32)
    assert len(set(ids.tolist())) == 1
    assert r.decode_bytes(ids) == d
    # empty
    ids = r._eng.intern_spans_parallel(b"", b"", b"", 4)
    assert ids == b""


def test_parallel_intern_rejects_bad_tiling():
    data = _fix("english") * 2
    r = ratec.RateC(c=20, store_book=True)
    starts = np.asarray([0, 100], dtype=np.uint32)
    ends = np.asarray([50, 5001], dtype=np.uint32)
    with pytest.raises(ValueError):
        r._eng.intern_spans_parallel(
            data[:5001], starts.tobytes(), ends.tobytes(), 4)


def test_parallel_intern_requires_book():
    r = ratec.RateC(c=20, store_book=False)
    with pytest.raises(Exception):
        r._eng.intern_spans_parallel(
            b"abc", np.asarray([0], dtype=np.uint32).tobytes(),
            np.asarray([3], dtype=np.uint32).tobytes(), 2)


def test_warm_engine_stream_ids_call_relative():
    """v0.9.0rc regression: a continued engine must not break the ids
    path — encode_stream_parallel rebases its span arrays to the call
    (encode_to_spans itself keeps the documented lifetime-absolute
    offsets that the goldens and the dataset builder rely on)."""
    data = b"abc" * 1000
    r = ratec.RateC(c=20, store_book=True)
    first = r.encode_stream_parallel(data, ids=True)
    assert r.decode_bytes(first[2]) == data
    # second call: the engine is warm — still round-trips, spans 0-based
    second = r.encode_stream_parallel(data, ids=True)
    assert r.decode_bytes(second[2]) == data
    assert int(second[0][0]) == 0 and int(second[1][-1]) == len(data)
    # ids count tracks the (adapted) segmentation, not the first call
    assert len(second[2]) == len(second[0])
    # encode_to_spans keeps lifetime-absolute offsets on a warm engine
    s, e = r.encode_to_spans(data[:300], as_arrays=True)
    assert int(s[0]) == 2 * len(data) and int(e[-1]) == 2 * len(data) + 300
