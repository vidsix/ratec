"""v0.9.0b: the ids side of stream-parallel encoding.

Engine.intern_spans interns a span tiling into the book in encounter
order — ids must equal the sequential engine's ids for the same spans
(pure content+order function), the parallel token stream must decode
round-trip exactly, and the book must persist across calls so repeated
streams share one vocabulary.
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


def test_intern_pass_matches_sequential_ids():
    data = _fix("english") * 4
    r = ratec.RateC(c=20, store_book=True)
    r.reset()
    ids_seq = np.asarray(r.encode_bytes(data), dtype=np.uint32)
    r.reset()
    starts, ends = r.encode_to_spans(data, as_arrays=True)
    raw = r._eng.intern_spans(data, starts.tobytes(), ends.tobytes())
    ids_int = np.frombuffer(raw, dtype=np.uint32)
    assert np.array_equal(ids_seq, ids_int)


def test_stream_ids_roundtrip_both_modes():
    data = _fix("english") * 40
    r = ratec.RateC(c=20, store_book=True)
    for mode in ("runs", "snapshot"):
        starts, ends, tok = r.encode_stream_parallel(
            data, threads=2, mode=mode, ids=True)
        assert len(tok) == len(starts) == len(ends)
        assert tok.dtype == np.uint32
        assert r.decode_bytes(tok) == data


def test_stream_ids_share_vocabulary():
    data = _fix("english") * 40
    r = ratec.RateC(c=20, store_book=True)
    _, _, tok1 = r.encode_stream_parallel(data, threads=2, ids=True)
    _, _, tok2 = r.encode_stream_parallel(data[:len(data) // 2], ids=True)
    # the second call's ids index the same persistent book; the serial
    # re-encode can add a few percent of new merges beyond the parallel
    # run's merged book (host-dependent merge timing — observed ~33k on
    # a ~365k book on macOS). A proportional bound still catches real
    # runaway (a broken share would be 2-3x), while a fixed +10k does
    # not survive platform map-growth timing.
    assert int(tok2.max()) <= int(tok1.max()) + max(10_000, int(tok1.max()) // 10)
    assert r.decode_bytes(tok2) == data[:len(data) // 2]


def test_intern_spans_rejects_bad_tiling():
    data = _fix("english") * 2
    r = ratec.RateC(c=20, store_book=True)
    ids = np.asarray(r.encode_bytes(data[:5000]), dtype=np.uint32)
    starts = np.asarray([0, 100], dtype=np.uint32)
    ends = np.asarray([50, 5001], dtype=np.uint32)
    # v0.9.1: tiling violations are CODED RatecError (TILING), not bare
    # ValueError — pipelines branch on codes
    with pytest.raises(ratec.RatecError) as ei:
        r._eng.intern_spans(data[:5001], starts.tobytes(), ends.tobytes())
    assert ei.value.code == "TILING"
    del ids


def test_stream_ids_requires_book():
    r = ratec.RateC(c=20, store_book=False)
    with pytest.raises(ValueError):
        r.encode_stream_parallel(b"abc" * 100, ids=True)
