"""ratec.api — the pythonic surface of the RateC tokenizer.

RateC is an adaptive rate-based tokenizer: a fixed-weight 6-order KT mixture
(judge) assigns each byte a surprisal; a greedy under-shoot patcher packs
bytes into patches worth at most C bits. There is NO training data and NO
vocabulary file — the model is the algorithm, and any C works at any time.

Example
-------
>>> import ratec
>>> tok = ratec.RateC(c=20)
>>> ids = tok.encode("hello world")
>>> tok.decode(ids) == "hello world"
True
"""
from __future__ import annotations

import zlib
from typing import Iterable, Sequence

import numpy as np

from . import _rcengine as _rc

__all__ = ["RateC", "RatecError", "patch_from_sur", "batch_spans", "__version__"]
RatecError = _rc.RatecError


class RateC:
    """RateC tokenizer with a tiktoken-shaped API.

    Parameters
    ----------
    c : float
        Patch budget in bits (default 20; fractional values like 17.5 are
        allowed). Smaller C -> shorter patches and more of them; larger C ->
        longer patches. The judge is C-independent, so the choice is purely
        downstream (embedding width, context budget).
    store_book : bool
        Keep the dynamic patch book (content -> id) so `encode`/`decode`
        round-trip without the source text. Off is marginally faster for
        spans-only output; for LM data prep you need ids, which requires
        the book — with the rolling-hash intern (v0.8) the book's cost
        over spans-only is just the per-new-entry arena write, so id
        throughput roughly equals span throughput.
    pf : int
        Software prefetch depth (0-4); -1 = auto, which resolves to the
        measured default 2 on x86-64 (`PF_DEFAULT` in `_rcengine.c`).
    warm_state : bytes | str | None
        Snapshot bytes or path produced by `save_state` — cold-start is
        avoided and the judge starts already adapted (RateC's optional
        "model file"; entirely optional).
    prune : bool
        Frequency-aware book pruning (v0.7, opt-in; default off). At every
        `prune_after` bytes of stream position, book entries whose
        since-last-prune occurrence count is below `prune_min` are dropped
        and the surviving ids renumbered dense (first-seen order kept).
        Targets singleton starvation: patch types seen rarely occupy the
        fixed id budget downstream LMs must spend embedding rows on.
        Byte segmentation is UNAFFECTED (the patcher is book-independent) —
        spans are identical with and without pruning; only id assignment
        churns at prune boundaries. Consequences, documented and tested:
        ids emitted before a prune may stop decoding on the pruned engine
        (`decode_bytes` rejects them cleanly), and prune-mode snapshots
        use snapshot format v7 (older ratec versions reject them cleanly;
        non-prune snapshots remain byte-identical v6).
    prune_after : int
        Prune interval in bytes (default 1 MiB; must be >= 1 when `prune`).
    prune_min : int
        Since-last-prune count threshold (default 2: drop singletons).

    Notes
    -----
    * `encode`/`decode` require the book. The book is instance state: ids
      from tokenizer A mean nothing to tokenizer B unless B was created
      with A's state snapshot.
    * Engines are not thread-safe; use one instance per thread or
      `ratec.batch_spans` for parallel bulk work.
    """

    def __init__(self, c: float = 20, store_book: bool = True, pf: int = -1,
                 warm_state=None, prebits: int = 16, map_capbits: int = 22,
                 exact: bool = True, carry: bool = False,
                 prune: bool = False, prune_after: int = 1 << 20,
                 prune_min: int = 2):
        if prune and prune_after < 1:
            raise ValueError("prune_after must be >= 1 when prune=True")
        if prune_min < 1:
            raise ValueError("prune_min must be >= 1")
        self._eng = _rc.Engine(c=c, store_book=store_book, pf=pf,
                               prebits=prebits, map_capbits=map_capbits,
                               exact=exact, carry=carry,
                               prune_after=prune_after if prune else 0,
                               prune_min=prune_min)
        if warm_state is not None:
            state = warm_state
            if isinstance(state, str):
                with open(state, "rb") as f:
                    raw = f.read()
            else:
                raw = bytes(state)
            # swap in the loaded core, preserving nothing (it is the state);
            # a snapshot carries its own C (judge state is shared, C only
            # affects the patcher and changing it mid-stream is ill-defined)
            self._eng = _rc.load_state(_maybe_decompress(raw))
        # bind hot methods (property setter below keeps them fresh if a
        # caller swaps self._eng afterwards): per-call attribute lookup on
        # the engine object is the largest Python-side cost for small inputs
        self._eng = self._eng

    # engine slot with re-binding: swapping self._eng refreshes the hot
    # bound methods used by encode/decode/spans
    @property
    def _eng(self):
        return self._engine_obj

    @_eng.setter
    def _eng(self, value):
        self._engine_obj = value
        self._enc = value.encode_bytes
        self._dec = value.decode_bytes
        self._spans = value.encode_spans

    # ------------------------------------------------------------ text API
    def encode(self, text, *, c: int | None = None) -> list[int]:
        """str | bytes -> patch ids (utf-8 is the byte interface)."""
        if c is not None and c != self.c:
            raise ValueError(
                "per-call c is not supported on a stateful engine; build a "
                "RateC(c=...) or use ratec.patch_from_sur for offline sweeps")
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        return self._enc(data)

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8")

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        """ids -> bytes. numpy uint32 arrays (and other buffer objects) go
        through the engine's buffer-protocol fast path (no per-id boxing;
        ~2x over list input on the 0.8.1 engine)."""
        arr = getattr(ids, "dtype", None)
        if arr is not None and ids.dtype == np.uint32 and ids.ndim == 1:
            return self._dec(ids)
        return self._dec(list(ids))

    def decode_batch(self, ids_list: Sequence[Sequence[int]],
                     threads: int = 0) -> list[bytes]:
        """Decode many id sequences on the engine's C thread pool (v0.8.9).

        Decoding is a PURE READ of the book (ids -> ln/off -> arena, no
        judge, no state change), so all sequences share ONE engine with
        the GIL released — parallel bounds-check + parallel gather, no
        clones. Buffer inputs (numpy uint32 / array('I') / bytes) take
        the parallel path; other sequences fall back to the sequential
        loop (decode_bytes validates arbitrary int lists). The book is
        shared, so ids must come from this instance's state."""
        arrs = []
        for ids in ids_list:
            arr = getattr(ids, "dtype", None)
            if arr is not None and ids.dtype == np.uint32 and ids.ndim == 1:
                arrs.append(ids)
            elif isinstance(ids, (bytes, bytearray, memoryview)):
                arrs.append(ids)
            else:
                # unvalidatable int lists: keep the exact sequential path
                return [self.decode_bytes(ids) for ids in ids_list]
        return list(self._eng.decode_batch(arrs, threads=threads))

    # --------------------------------------------------- parallel stream
    def encode_stream_parallel(self, data, chunk: int | None = 1 << 20,
                               threads: int = 0, head_chunks: int = 1,
                               mode: str = "runs", as_arrays: bool = False):
        """Encode ONE long stream on many cores by chunking.

        The first `head_chunks` chunks run sequentially (exact judge
        adaptation); a state snapshot taken after them seeds all parallel
        workers. Deterministic, spans still tile the input exactly and
        decode round-trips. For bulk LM data prep on real (multi-core)
        boxes this is the throughput mode.

        mode="runs" (v0.8.8 default): each worker clones the head state
        ONCE and encodes ONE CONTIGUOUS RUN of the remainder, continuing
        its own judge — zero per-chunk state resets, and only
        `workers - 1` approximation boundaries total (fewer than one per
        chunk, so spans track the single-engine view closer while also
        being faster).

        mode="snapshot": the v0.8.x per-chunk model — every chunk restarts
        from the head snapshot (clone + restore, bit-identical to the
        pre-0.8.8 wire roundtrip), one approximation boundary per chunk.

        chunk=None auto-tunes: n/threads clamped to [1, 8] MiB so workers
        stay balanced and the sequential head stays a small fraction.

        head_chunks trades sequential work for approximation quality.
        Measured (10.7 MiB english corpus, 1 MiB chunks): span-boundary
        agreement with sequential encode_to_spans rises 0.23 / 0.35 /
        0.46 / 0.56 for head_chunks = 1 / 2 / 3 / 4 (snapshot mode);
        runs mode measures 0.59 (head_chunks=1) / 0.65 (head_chunks=2) —
        continuing the judge across chunks both removes per-chunk state
        resets and tracks the single-engine view closer. Default 1 keeps
        the sequential fraction minimal; raise it when patch boundaries
        must track the single-engine view more closely.

        Returns [(start, end, id|None), ...] with absolute offsets, or
        with as_arrays=True (v0.8.9) a (starts, ends) pair of uint32
        ndarrays carrying the same values — tuple-free output keeps the
        workers off the GIL (the per-span tuple build serialized all
        workers and cost ~0.3-0.5 s per 10.7 MiB).
        """
        if mode not in ("runs", "snapshot"):
            raise ValueError("mode must be 'runs' or 'snapshot'")
        data = _as_bytes(data)
        n = len(data)
        if threads <= 0:
            import os
            threads = os.cpu_count() or 1
        if chunk is None:
            chunk = max(1 << 20, min(8 << 20, n // max(1, threads)))
        if n <= chunk:
            return self.encode_to_spans(data, as_arrays=as_arrays)
        from concurrent.futures import ThreadPoolExecutor

        head = min(max(0, head_chunks), (n + chunk - 1) // chunk - 1)
        # 1) sequential head on a private engine; it emits its own spans
        eng_head = _rc.Engine(c=self.c, store_book=False,
                              pf=self._eng.pf, exact=True,
                              carry=self._eng.carry_mode)
        out = []
        head_starts, head_ends = [], []
        consumed = 0
        for k in range(head):
            seg = data[k * chunk: (k + 1) * chunk]
            if as_arrays:
                sb, eb = eng_head.feed_raw(seg)
                head_starts.append(np.frombuffer(sb, dtype=np.uint32))
                head_ends.append(np.frombuffer(eb, dtype=np.uint32))
            else:
                out.extend(eng_head.feed(seg))
            consumed += len(seg)
        if as_arrays:
            sb, eb = eng_head.flush_raw()   # close open tail at the boundary
            head_starts.append(np.frombuffer(sb, dtype=np.uint32))
            head_ends.append(np.frombuffer(eb, dtype=np.uint32))
        else:
            out.extend(eng_head.flush())
        # v0.8.8: seed workers by CLONING the head engine (bit-identical
        # judge state, no wire roundtrip, no map rebuild) and pre-sizing
        # its maps for head + one chunk, so no chunk ever rehashes mid-run.
        eng_head.reserve(chunk)
        nseen0 = consumed      # snapshot's absolute stream position
        njobs = (n - consumed + chunk - 1) // chunk
        nworkers = max(1, min(threads, njobs))
        starts_all = []
        import queue as _queue
        pool = _queue.Queue()
        for _ in range(nworkers):
            pool.put(eng_head.clone())

        if mode == "runs":
            # contiguous runs: each worker clones the head state once and
            # CONTINUES encoding its run — no per-chunk restore at all
            spans_parts: list = [None] * nworkers
            run_len = (n - consumed + nworkers - 1) // nworkers

            def run(w: int, a: int, b: int):
                eng = pool.get()
                try:
                    # a fresh clone already carries the head state; restores
                    # are only needed on REUSE (never in this mode)
                    if as_arrays:
                        sb, eb = eng.encode_spans_raw(data[a:b])
                        off = a - nseen0
                        return w, (np.frombuffer(sb, dtype=np.uint32) + off,
                                   np.frombuffer(eb, dtype=np.uint32) + off)
                    sp = eng.encode_spans(data[a:b])
                finally:
                    pool.put(eng)
                off = a - nseen0
                return w, [(s + off, e + off, i) for s, e, i in sp]

            jobs = []
            for w in range(nworkers):
                a = consumed + w * run_len
                b = min(a + run_len, n)
                if a < b:
                    jobs.append((w, a, b))
            with ThreadPoolExecutor(max_workers=nworkers) as ex:
                for w, spans_part in ex.map(lambda j: run(*j), jobs):
                    spans_parts[w] = spans_part
            if as_arrays:
                sa = head_starts + [p[0] for p in spans_parts if p is not None]
                ea = head_ends + [p[1] for p in spans_parts if p is not None]
                if len(sa) > 1:
                    return (np.concatenate(sa), np.concatenate(ea))
                return (sa[0].copy(), ea[0].copy())
            for part in spans_parts:
                if part:
                    starts_all.extend(part)
            return out + starts_all

        def run(a: int, b: int):
            eng = pool.get()
            try:
                eng.restore_from(eng_head)
                if as_arrays:
                    sb, eb = eng.encode_spans_raw(data[a:b])
                    off = a - nseen0
                    return (np.frombuffer(sb, dtype=np.uint32) + off,
                            np.frombuffer(eb, dtype=np.uint32) + off)
                sp = eng.encode_spans(data[a:b])
            finally:
                pool.put(eng)
            # engine spans continue the snapshot's absolute offsets; rebase
            # to this chunk's true position
            off = a - nseen0
            return [(s + off, e + off, i) for s, e, i in sp]

        jobs = []
        pos = consumed
        while pos < n:
            nxt = min(pos + chunk, n)
            jobs.append((pos, nxt))
            pos = nxt
        with ThreadPoolExecutor(max_workers=nworkers) as ex:
            parts = list(ex.map(lambda ab: run(*ab), jobs))
        if as_arrays:
            sa = head_starts + [p[0] for p in parts]
            ea = head_ends + [p[1] for p in parts]
            if len(sa) > 1:
                return (np.concatenate(sa), np.concatenate(ea))
            return (sa[0].copy(), ea[0].copy())
        for spans_part in parts:
            starts_all.extend(spans_part)
        return out + starts_all

    # ----------------------------------------------------------- bytes API
    def encode_bytes(self, data) -> list[int]:
        return self._enc(bytes(data))

    def encode_to_spans(self, data, as_arrays: bool = False):
        """byte spans [(start, end, id|None), ...] — the LM-facing view.

        as_arrays=True (v0.8.9) returns (starts, ends) little-endian
        uint32 ndarrays instead of per-span tuples — the same values
        with ~50x less Python-object allocation (the tuple path costs
        ~3 objects per span, GIL-held; arrays are two memcpys + two
        numpy wraps)."""
        data = _as_bytes(data)
        if as_arrays:
            sb, eb = self._eng.encode_spans_raw(data)
            return (np.frombuffer(sb, dtype=np.uint32),
                    np.frombuffer(eb, dtype=np.uint32))
        return self._spans(data)

    def encode_batch(self, docs: Sequence, threads: int = 0):
        """Parallel spans for independent docs (bookless, GIL released).

        Returns a list of (starts ndarray uint32, n_bytes) per doc; patch k
        covers [starts[k], starts[k+1]) and the last patch ends at n_bytes.
        """
        raws = _rc.batch([bytes(d) for d in docs], c=self.c, threads=threads)
        out = []
        for raw, d in zip(raws, docs):
            starts = np.frombuffer(raw, dtype=np.uint32)
            out.append((starts, len(d)))
        return out

    def encode_file(self, path, *, separator: bytes = b"<|endoftext|>",
                    threads: int = 0, max_docs: int | None = None):
        """Whole-file ingestion (gigatoken TextFileSource parity): mmap the
        file, split on the byte separator, batch-encode the documents on
        the engine's C-side thread pool. Every document gets an INDEPENDENT
        cold judge (same semantics as encode_batch) — deterministic, spans
        tile each document exactly. Returns the encode_batch view
        ([(starts ndarray, n_bytes), ...]) plus the document byte lengths
        consumed; docs longer than 1 GiB are rejected by the batch path."""
        import mmap as _mmap
        with open(path, "rb") as f:
            with _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
                data = bytes(mm)
        if separator:
            docs, start = [], 0
            while True:
                j = data.find(separator, start)
                if j < 0:
                    docs.append(data[start:])
                    break
                docs.append(data[start:j])
                start = j + len(separator)
            if not docs[-1]:
                docs.pop()
        else:
            docs = [data]
        if max_docs is not None:
            docs = docs[:max_docs]
        return self.encode_batch(docs, threads=threads)

    def encode_files(self, paths, *, separator: bytes = b"<|endoftext|>",
                     threads: int = 0, max_docs: int | None = None) -> list:
        """Multi-file driver (gigatoken encode_files parity): encode_file
        each path; returns one encode_batch view per file, in order."""
        return [self.encode_file(p, separator=separator, threads=threads,
                                 max_docs=max_docs) for p in paths]

    # ------------------------------------------------------------- export
    def to_ranks(self) -> dict[bytes, int]:
        """The learned vocabulary as {bytes: id} (tiktoken-ranks parity:
        introspection, vocab diffs, offline analysis)."""
        return {piece: i for i, piece in enumerate(self._eng.book_dump())}

    def save_ranks(self, path: str) -> int:
        """Write the vocabulary in tiktoken's .tiktoken format (base64
        piece + ' ' + rank per line). Returns the number of entries."""
        import base64
        n = 0
        with open(path, "wb") as f:
            for i, piece in enumerate(self._eng.book_dump()):
                f.write(base64.b64encode(piece) + b" %d\n" % i)
                n += 1
        return n

    def counts(self) -> np.ndarray:
        """Per-id occurrence counts (window counts since the last prune,
        index == id) — prune-tuning introspection: how many patches
        survive a given prune_min."""
        raw = self._eng.book_counts()
        if raw is None:
            raise RuntimeError("book counts unavailable")
        return np.frombuffer(raw, dtype=np.uint32)

    def validate(self, data, *, decode_check: bool = True) -> dict:
        """Integrity report for a corpus against this instance's state
        (gigatoken CLI --validate parity): roundtrip exactness, patch
        statistics, compression. Runs on a COPY of the judge/patcher
        state, so the caller's streaming position is untouched."""
        data = _as_bytes(data)
        eng = _rc.load_state(self._eng.save_state())
        ids = eng.encode_bytes(data)
        rep: dict = {"n_bytes": len(data), "n_patches": len(ids),
                     "bytes_per_patch": (len(data) / len(ids)) if ids else 0.0}
        if decode_check:
            out = eng.decode_bytes(np.array(ids, dtype=np.uint32))
            rep["roundtrip_ok"] = (out == data)
        rep["n_book"] = self._eng.book_size
        return rep

    # ----------------------------------------------------------- research
    def surprise(self, data) -> np.ndarray:
        """Per-byte surprisal (float32) from the current judge state."""
        return np.frombuffer(self._eng.surprise(_as_bytes(data)), dtype=np.float32)

    def feed(self, data):
        """Streaming: feed a chunk, get patches closed by it (spans)."""
        return self._eng.feed(_as_bytes(data))

    def flush(self):
        return self._eng.flush()

    # -------------------------------------------------------------- state
    def save_state(self, path: str | None = None) -> bytes | None:
        raw = _maybe_compress(self._eng.save_state())
        if path is None:
            return raw
        with open(path, "wb") as f:
            f.write(raw)
        return None

    @classmethod
    def from_state(cls, state, **kw) -> "RateC":
        return cls(warm_state=state, **kw)

    def reset(self) -> None:
        self._eng.reset()

    # -------------------------------------------------------------- misc
    @property
    def c(self) -> float:
        return self._eng.c

    @property
    def carry(self) -> float:
        """Current carry-over register value (0 unless carry mode)."""
        return self._eng.carry

    @property
    def carry_mode(self) -> bool:
        """True when the credit carry-over patcher is active."""
        return bool(self._eng.carry_mode)

    @property
    def book_size(self) -> int:
        return self._eng.book_size

    @property
    def bytes_seen(self) -> int:
        return self._eng.bytes_seen

    @property
    def n_patches(self) -> int:
        return self._eng.n_patches

    @property
    def n_forced(self) -> int:
        return self._eng.n_forced

    @property
    def n_pruned(self) -> int:
        """Book entries removed by pruning so far (0 unless prune=True)."""
        return self._eng.n_pruned

    @property
    def prune_after(self) -> int:
        """Prune interval in bytes (0 = pruning disabled)."""
        return self._eng.prune_after

    @property
    def prune_min(self) -> int:
        """Since-last-prune count threshold."""
        return self._eng.prune_min

    def __repr__(self):
        return (f"RateC(c={self.c}, store_book={self._eng.store_book}, "
                f"bytes_seen={self.bytes_seen}, book={self.book_size})")


_MAGIC = b"RCZ1"


def _as_bytes(data) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


def _maybe_compress(raw: bytes) -> bytes:
    return _MAGIC + zlib.compress(raw, 6)


def _maybe_decompress(raw: bytes) -> bytes:
    if raw[:4] == _MAGIC:
        try:
            return zlib.decompress(raw[4:])
        except zlib.error as e:
            # a corrupt RCZ1 container must surface as RatecError, not an
            # unrelated zlib.error leaking out of load_state (v0.6.0)
            raise RatecError(f"corrupt ratec state container: {e}") from None
    return raw


def patch_from_sur(sur: np.ndarray | bytes, c: float,
                   carry: bool = False) -> np.ndarray:
    """Reference patcher over a stored surprisal array.

    The patcher is C-independent of the judge, so one `surprise()` pass
    serves every C — this is the cheap way to run C sweeps. `c` may be
    fractional (e.g. 17.5). `carry=True` selects the credit carry-over
    patcher (v0.4): each patch may spend the previous patch's unused
    budget (capped at +C extra); the long-run rate converges to exactly
    C bits/patch.

    `sur` must be 1-D float32 (what `RateC.surprise` returns) or raw
    float32 bytes. A float64 array is accepted and cast; any other dtype
    is REJECTED — silently reinterpreting e.g. float64 bytes as float32
    produced garbage patch starts (v0.6.0 dtype guard).
    """
    if isinstance(sur, np.ndarray):
        if sur.ndim != 1:
            raise ValueError(
                f"sur must be 1-D float32 surprisal, got shape {sur.shape}")
        if sur.dtype == np.float64:
            sur = sur.astype(np.float32)          # accepted, cast
        elif sur.dtype != np.float32:
            raise ValueError(
                f"sur must be float32 surprisal, got dtype {sur.dtype} — "
                "other dtypes are rejected because their bytes would be "
                "silently reinterpreted as float32 (cast float64 explicitly)")
        raw = sur.tobytes()
    else:
        raw = bytes(sur)
    return np.frombuffer(_rc.patch_from_sur(raw, float(c), bool(carry)),
                         dtype=np.uint32)


def solve_c(data, target_patches: int, lo: float = 1.0, hi: float = 64.0,
            carry: bool = False, max_probes: int = 40) -> tuple[float, int]:
    """Rate servo: find the C whose patch count hits `target_patches`.

    Patch count is strictly monotone decreasing in C, so this is an exact
    bisection. The judge pass runs ONCE (it is C-independent); every
    candidate C then costs only the cheap re-patch (`patch_from_sur`),
    so the sweep is nearly free compared to re-ingesting the corpus.

    Returns (c, achieved_patch_count) — the count is exact for the
    returned C. For a BOOK-SIZE target (frozen-vocab donor match) use
    `solve_c_for_book`; note book size <= patch count (repeated contents
    share one entry).
    """
    data = _as_bytes(data)
    if target_patches <= 0:
        raise ValueError("target_patches must be positive")
    if not (1.0 <= lo < hi <= 64.0):
        raise ValueError("need 1.0 <= lo < hi <= 64.0")
    sur = RateC(c=lo, store_book=False).surprise(data)

    def patches_at(c: float) -> int:
        return int(len(patch_from_sur(sur, c, carry=carry)))

    lo_c, hi_c = float(lo), float(hi)
    lo_n, hi_n = patches_at(lo_c), patches_at(hi_c)
    if lo_n <= target_patches:          # even the smallest C gives fewer
        return lo_c, lo_n
    if hi_n >= target_patches:          # even the largest C gives more
        return hi_c, hi_n
    for _ in range(max_probes):
        mid = 0.5 * (lo_c + hi_c)
        if mid <= lo_c or mid >= hi_c:
            break
        n = patches_at(mid)
        if n > target_patches:
            lo_c, lo_n = mid, n
        else:
            hi_c, hi_n = mid, n
        if hi_c - lo_c < 1e-9:
            break
    # return whichever bracket side landed closer to the target
    if abs(lo_n - target_patches) <= abs(hi_n - target_patches):
        return lo_c, lo_n
    return hi_c, hi_n


def solve_c_for_book(data, target_book: int, lo: float = 1.0,
                     hi: float = 64.0, carry: bool = False,
                     coarse_steps: int = 32,
                     max_probes: int = 24) -> tuple[float, int]:
    """Rate servo for frozen-vocab targets: find C with book size ≈ target.

    "Tokenize this corpus so the frozen book has exactly N entries" —
    the donor-vocab match for transplants. Book size vs C is UNIMODAL
    (→ 256 distinct single bytes as C→1; peaks mid-range where patches
    are diverse but repeated; falls back toward the patch count as C
    grows and patches become long and unique), so this does a coarse
    ascending scan for the rising bracket around the target, then a
    local bisection. Returns (c, achieved_book); the achieved value is
    exact for the returned C, the match is as close as book(C) allows.
    """
    data = _as_bytes(data)
    if target_book <= 0:
        raise ValueError("target_book must be positive")
    if not (1.0 <= lo < hi <= 64.0):
        raise ValueError("need 1.0 <= lo < hi <= 64.0")
    n = len(data)
    sur = RateC(c=lo, store_book=False).surprise(data)

    def book_at(c: float) -> int:
        starts = patch_from_sur(sur, c, carry=carry)
        edges = np.append(starts.astype(np.int64), n)
        seen = set()
        for k in range(len(edges) - 1):
            seen.add(data[edges[k]:edges[k + 1]])
        return len(seen)

    grid = [lo + (hi - lo) * k / (coarse_steps - 1)
            for k in range(coarse_steps)]
    vals = [book_at(c) for c in grid]
    best_v, best_c = max(zip(vals, grid))
    if vals[0] >= target_book:
        return grid[0], vals[0]
    if best_v < target_book:            # target above the attainable peak
        return best_c, best_v
    for k in range(1, coarse_steps):
        if vals[k] >= target_book > vals[k - 1]:
            a, fa, b, fb = grid[k - 1], vals[k - 1], grid[k], vals[k]
            for _ in range(max_probes):
                mid = 0.5 * (a + b)
                if mid <= a or mid >= b:
                    break
                fm = book_at(mid)
                if fm < target_book:
                    a, fa = mid, fm
                else:
                    b, fb = mid, fm
                if b - a < 1e-4:
                    break
            if abs(fb - target_book) <= abs(fa - target_book):
                return b, fb
            return a, fa
    return best_c, best_v


def batch_spans(docs: Sequence, c: float = 20, threads: int = 0,
                fast: bool = False, carry: bool = False):
    """Parallel rate-patching of independent byte documents.

    Returns a list of (starts ndarray uint32, n_bytes); patch k spans
    [starts[k], starts[k+1]), the last ends at n_bytes. Docs are encoded
    cold-start and independently — identical results to `RateC(
    store_book=False)` per doc, across all cores. `fast=True` uses the
    non-exact fast lane (see RateC(exact=False)).
    """
    raws = _rc.batch([bytes(d) for d in docs], c=float(c), threads=threads,
                     fast=bool(fast), carry=bool(carry))
    return [(np.frombuffer(raw, dtype=np.uint32), len(d))
            for raw, d in zip(raws, docs)]


def batch_encode(docs: Sequence, c: float = 20, threads: int = 0,
                 fast: bool = False, carry: bool = False):
    """Thread-parallel LM-prep tokenization of independent byte documents.

    Returns a list of (ids ndarray int32, lens ndarray int32) per doc:
    `ids` are book ids (single-byte patches carry their raw byte value,
    multi-byte patches are book ids — the shipped packed-stream content
    semantics), `lens` the per-token byte lengths (ends - starts; the
    spans tile each doc exactly). Each doc is encoded cold-start and
    independently — identical ids to `RateC(c=c).encode_to_spans(doc)`
    per doc, with all cores working in parallel (the GIL is released).
    For one long stream prefer a single `RateC` engine (its judge stays
    warm across the whole stream); for many small/medium docs this is
    the fastest path to training-ready tokens.
    """
    raws = _rc.batch([bytes(d) for d in docs], c=float(c), threads=threads,
                     fast=bool(fast), carry=bool(carry), ids=True)
    out = []
    for starts_b, ends_b, ids_b in raws:
        starts = np.frombuffer(starts_b, dtype=np.uint32)
        ends = np.frombuffer(ends_b, dtype=np.uint32)
        ids = np.frombuffer(ids_b, dtype=np.uint32)
        out.append((ids.astype(np.int32),
                    ends.astype(np.int32) - starts.astype(np.int32)))
    return out


__version__ = _rc.version()
