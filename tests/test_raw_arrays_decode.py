"""v0.8.9: tuple-free span output + thread-parallel decode.

The raw-array span path (encode_spans_raw / as_arrays=True) must carry
EXACTLY the tuple path's values — same spans, same counts. The parallel
decode pool shares one read-only book across threads; its output must
equal the sequential decode_bytes loop. These are the gates that make
the GIL-free paths safe to default into.
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


def test_as_arrays_matches_tuples_sequential():
    data = _fix("english") * 4
    r = ratec.RateC(c=20, store_book=False)
    r.reset()
    tup = r.encode_to_spans(data)
    r.reset()
    starts, ends = r.encode_to_spans(data, as_arrays=True)
    assert len(starts) == len(tup) == len(ends)
    assert starts.dtype == np.uint32 and ends.dtype == np.uint32
    assert all((int(starts[k]), int(ends[k])) == (tup[k][0], tup[k][1])
               for k in range(len(tup)))
    assert int(ends[-1]) == len(data)


def test_as_arrays_matches_tuples_stream_parallel():
    data = _fix("english") * 40            # 10.7 MiB: real chunking
    r = ratec.RateC(c=20, store_book=False)
    for mode in ("runs", "snapshot"):
        tup = [(s, e, i) for s, e, i in
               r.encode_stream_parallel(data, threads=2, mode=mode)]
        starts, ends = r.encode_stream_parallel(
            data, threads=2, mode=mode, as_arrays=True)
        assert len(starts) == len(tup)
        assert all((int(starts[k]), int(ends[k])) == (tup[k][0], tup[k][1])
                   for k in range(len(tup)))
        assert int(starts[0]) == 0 and int(ends[-1]) == len(data)


def test_decode_batch_matches_sequential():
    data = _fix("english") * 4
    r = ratec.RateC(c=20, store_book=True)
    docs = [data[i * 3000:(i + 1) * 3000] for i in range(24)]
    ids = [np.asarray(r.encode_bytes(d), dtype=np.uint32) for d in docs]
    ref = [r.decode_bytes(x) for x in ids]
    for threads in (1, 2):
        out = r.decode_batch(ids, threads=threads)
        assert out == ref
        assert all(isinstance(b, bytes) for b in out)


def test_decode_batch_rejects_bad_ids():
    data = _fix("english") * 2
    r = ratec.RateC(c=20, store_book=True)
    good = np.asarray(r.encode_bytes(data[:5000]), dtype=np.uint32)
    bad = good.copy()
    bad[3] = 10 ** 7                      # beyond the book
    with pytest.raises(ratec.RatecError):
        r.decode_batch([good, bad], threads=2)


def test_decode_batch_falls_back_for_lists():
    data = _fix("english") * 2
    r = ratec.RateC(c=20, store_book=True)
    ids = r.encode_bytes(data[:5000])
    ref = r.decode_bytes(ids)
    out = r.decode_batch([ids, ids], threads=2)   # plain int lists
    assert out == [ref, ref]


def test_decode_requires_book():
    r = ratec.RateC(c=20, store_book=False)
    with pytest.raises(ratec.RatecError):
        r.decode_batch([np.zeros(4, dtype=np.uint32)], threads=1)
