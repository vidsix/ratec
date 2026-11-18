"""P3.2 (PLAN_v0.8): speed regression floor.

Loose floor on the ids path (the LM data-prep path) with the rolling-
hash intern and merged book slots landed: catches accidental
pessimizations (e.g. a revert to per-close full re-hashing, which
measured 4.1 MB/s warm before the fix vs 7.4 after) without flaking on
loaded machines — the floor stays 2.5 because a concurrently running
benchmark or training job can halve the observed number (measured range
under load: 3.3-9.5 MB/s on this very sample; unloaded it is stable
around 7). 512 KB sample keeps the judge map allocation unit-test-sized.
"""
import os
import sys
import time

import pytest

HERE = __file__.rsplit("/tests/", 1)[0]
sys.path.insert(0, HERE)

import ratec                      # noqa: E402

pytestmark = pytest.mark.slow


def test_ids_path_speed_floor():
    # deterministic mixed sample (English words, UTF-8 codepoints, code)
    words = b"the quick brown fox jumps over the lazy dog and then some "
    uni = " привет тест данных ещё немного текста для скорости ".encode("utf-8")
    code = b"def f(x):\n    return x * 2 + y\nclass A:\n    pass\n"
    sample = (words * 1600 + uni * 800 + code * 1600)[:512_000]
    eng = ratec.RateC(c=20)
    eng.encode_to_spans(sample[:64_000])          # warm the book
    t0 = time.perf_counter()
    eng.encode_to_spans(sample)
    dt = time.perf_counter() - t0
    mbs = len(sample) / dt / 1e6
    assert mbs >= 2.5, f"ids path regressed: {mbs:.2f} MB/s (< 2.5 floor)"
