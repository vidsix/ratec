"""P3.1 (PLAN_v0.8): gate the sweep/prune stream producer.

build_streams.py's packed-id layout is protocol-critical — it feeds the
c_sweep and book_pruning records — so its invariants are pinned here:
the id space is bounded by the C budget (not by the book), bytes are
conserved exactly, single-byte tokens carry raw byte values, the prune
arms share byte segmentation, and the ids decode through twin_study's
original id arithmetic.

The full-size streams live in runs/ (produced once, sha-bound); these
tests run the same producer code path in-process on a tiny corpus with
the module's partition constants patched (fast, hermetic).
"""
import sys

import numpy as np
import pytest

HERE = __file__.rsplit("/tests/", 1)[0]
RS = HERE + "/research"
sys.path.insert(0, RS)
sys.path.insert(0, HERE)

import build_streams as bs          # noqa: E402

CORPUS = (b"the quick brown fox jumps over the lazy dog. " * 40
          + "\u041f\u0440\u0438\u0432\u0435\u0442 ".encode("utf-8") * 20
          + b"def f(x):\n    return x * 2\n" * 30)
DOMAINS = ["en", "hr", "zh", "code"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Patch the module's corpus constants onto a tiny corpus, then run
    the producer's regime function directly (unpruned + pruned)."""
    n = len(CORPUS)
    offs = [0, n // 4, n // 2, (3 * n) // 4, n]
    d = tmp_path_factory.mktemp("streams")
    (d / "_train_mix.txt").write_bytes(CORPUS)
    for k, name in enumerate(DOMAINS):
        (d / f"{name}_val.txt").write_bytes(CORPUS[offs[k]:offs[k + 1]][:4096])
    saved = (bs.DATA, bs.RUNS, bs.MIX_OFFSETS, bs.TRAIN_BYTES, bs.VAL_BYTES)
    bs.DATA = str(d)
    bs.RUNS = str(d)
    bs.MIX_OFFSETS = offs
    bs.TRAIN_BYTES = n
    bs.VAL_BYTES = sum(len((d / f"{name}_val.txt").read_bytes())
                       for name in DOMAINS)
    try:
        bs.regime_ratec_c(10, False, 1 << 20, 2)
        bs.regime_ratec_c(10, True, 256, 2)
        bs.regime_ratec_c(8, False, 1 << 20, 2)
    finally:
        (bs.DATA, bs.RUNS, bs.MIX_OFFSETS, bs.TRAIN_BYTES,
         bs.VAL_BYTES) = saved
    return d, offs


def _load(d, prefix, dom, split):
    return (np.load(d / f"{prefix}_{dom}_{split}_ids.npy"),
            np.load(d / f"{prefix}_{dom}_{split}_lens.npy"))


@pytest.mark.parametrize("c", [8, 10])
def test_id_space_bounded_by_budget(built, c):
    """Max id < 256 + 9*2^C: the LM table is bound by the C budget, not
    by the book (the P3 protocol fix)."""
    d, _ = built
    V = 256 + 9 * (2 ** c)
    for dom in DOMAINS:
        for split in ["train", "val"]:
            ids, _ = _load(d, f"ratec_c{c}", dom, split)
            assert int(ids.max()) < V, (dom, split)


def test_byte_conservation(built):
    """Sum of lens over train+val == corpus + val bytes, exactly."""
    d, _ = built
    tot = sum(int(_load(d, "ratec_c10", dom, split)[1].sum())
              for dom in DOMAINS for split in ["train", "val"])
    vals = sum(len((d / f"{dom}_val.txt").read_bytes()) for dom in DOMAINS)
    assert tot == len(CORPUS) + vals


def test_single_bytes_are_raw_values(built):
    """1-byte tokens carry their byte value (< 256), the shipped
    layout's raw-byte slab."""
    d, _ = built
    ids, lens = _load(d, "ratec_c10", "en", "train")
    b1 = ids[lens == 1]
    assert len(b1) > 0 and int(b1.max()) < 256


def test_train_slice_partition(built):
    """Slices partition the mix EXACTLY in total; per-domain sums differ
    from their partition sizes only by straddling patches assigned to the
    slice their START byte falls in (the documented contract — a patch
    starting near a boundary spills into the next slice's byte range)."""
    d, offs = built
    sums = []
    for k, dom in enumerate(DOMAINS):
        _, lens = _load(d, "ratec_c10", dom, "train")
        s = int(lens.sum())
        sums.append(s)
        assert abs(s - (offs[k + 1] - offs[k])) < 512, (dom, s)
    assert sum(sums) == offs[-1]          # exact partition of the mix


def test_prune_arms_share_segmentation(built):
    """Pruned vs unpruned: identical per-domain token counts and lens
    (the patcher is book-independent — the prune measurement's control)."""
    d, _ = built
    for dom in DOMAINS:
        for split in ["train", "val"]:
            a = _load(d, "ratec_c10", dom, split)
            b = _load(d, "ratec_c10_pr", dom, split)
            assert len(a[0]) == len(b[0]) and (a[1] == b[1]).all(), dom


def test_packed_ids_decode_via_id_arithmetic(built):
    """Round-trip through twin_study's original id arithmetic: content
    slot = 256 + (id-256) % 2^C, length field = 1 + (id-256)//2^C; the
    length field must match the TRUE lens for every multi-byte token
    (lens > 10 collapse into the top slab — eval weighting is unaffected)."""
    d, _ = built
    m = 2 ** 10
    for dom in DOMAINS:
        ids, lens = _load(d, "ratec_c10", dom, "train")
        mb = lens >= 2
        lidx = 1 + (ids[mb] - 256) // m
        expect = np.minimum(lens[mb], 10) - 1
        assert (lidx == expect).all(), dom
        content = 256 + (ids[mb] - 256) % m
        assert int(content.max()) < 256 + m


def test_manifest_binds_files_and_stats(built):
    """The manifest sha256-binds every file and records the engine stats
    the records quote."""
    import hashlib
    import json
    d, _ = built
    m = json.load(open(d / "streams_manifest_c10.json"))
    assert {"protocol", "files", "engine_stats"} <= set(m)
    for name, rec in m["files"].items():
        h = hashlib.sha256((d / name).read_bytes()).hexdigest()
        assert h == rec["sha256"], name
    assert m["engine_stats"]["book_size_final"] > 0
