"""v0.8.8: engine clone/restore + encode_stream_parallel modes.

The clone/restore machinery (C core_restore / core_clone_from) must carry
judge+patcher state bit-identically — a restored engine must produce the
same spans the parent would produce. The stream-parallel modes must tile
exactly, be deterministic, and the snapshot mode must keep the pre-0.8.8
per-chunk semantics (every chunk restarts from the head state).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ratec  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "fixtures"


def _fix(name: str) -> bytes:
    return (DATA / f"{name}.span.bin").read_bytes()


def test_restore_state_equality():
    """clone + restore_from must reproduce the parent's continuation spans
    exactly — the union-clear page sync is state-identical."""
    data = _fix("english")
    head, tail = data[:100_000], data[100_000:]
    parent = ratec._rcengine.Engine(c=20, store_book=False, exact=True)
    parent.feed(head)
    parent.flush()
    parent.reserve(len(tail))
    worker = parent.clone()
    sp_parent = parent.encode_spans(tail)
    sp_worker = worker.encode_spans(tail)
    assert sp_parent == sp_worker
    # restore rewinds the worker; the parent has moved on — restoring the
    # PARENT'S CURRENT state must now reproduce its further continuation
    parent.feed(tail[:10_000])
    parent.flush()
    worker.restore_from(parent)
    sp_p2 = parent.encode_spans(tail[10_000:20_000])
    sp_w2 = worker.encode_spans(tail[10_000:20_000])
    assert sp_p2 == sp_w2


def test_restore_rejects_book_engines():
    eng = ratec._rcengine.Engine(c=20, store_book=True, exact=True)
    with pytest.raises(ratec.RatecError):
        eng.clone()


def test_stream_parallel_tiling_determinism():
    data = _fix("english") * 40          # 10.7 MiB: forces real chunking
    r = ratec.RateC(c=20, store_book=False)
    n = len(data)
    for mode in ("runs", "snapshot"):
        sp1 = r.encode_stream_parallel(data, threads=2, mode=mode)
        sp2 = r.encode_stream_parallel(data, threads=2, mode=mode)
        assert sp1 == sp2                      # deterministic
        assert sp1[0][0] == 0 and sp1[-1][1] == n
        assert all(sp1[k][1] == sp1[k + 1][0] for k in range(len(sp1) - 1))
        # every span is nonempty
        assert all(e > s for s, e, _ in sp1)


def test_snapshot_mode_matches_pre088_semantics():
    """snapshot mode: every chunk restarts from the head state — a chunk's
    spans must equal what a from-scratch engine loaded with the head
    snapshot produces for the same bytes."""
    data = _fix("chinese") * 6
    chunk = 1 << 18
    r = ratec.RateC(c=20, store_book=False)
    sp = r.encode_stream_parallel(data, threads=2, chunk=chunk,
                                  mode="snapshot")
    n = len(data)
    assert sp[0][0] == 0 and sp[-1][1] == n
    # head chunk = chunk 0: its spans are exactly sequential
    head_sp = ratec.RateC(c=20, store_book=False).encode_to_spans(data[:chunk])
    assert [(s, e) for s, e, _ in sp[:len(head_sp)]] == \
           [(s, e) for s, e, _ in head_sp]


def test_runs_mode_fewer_boundaries_than_snapshot():
    """runs mode continues the judge across chunks, so it must agree with
    the sequential view at least as well as snapshot mode."""
    data = _fix("english") * 40
    r = ratec.RateC(c=20, store_book=False)
    seq = set(r.encode_to_spans(data))
    snap = r.encode_stream_parallel(data, threads=2, mode="snapshot")
    runs = r.encode_stream_parallel(data, threads=2, mode="runs")
    a_snap = sum(1 for x in snap if x in seq) / len(snap)
    a_runs = sum(1 for x in runs if x in seq) / len(runs)
    assert a_runs >= a_snap


def test_invalid_mode_rejected():
    r = ratec.RateC(c=20, store_book=False)
    with pytest.raises(ValueError):
        r.encode_stream_parallel(b"abc", mode="nope")
