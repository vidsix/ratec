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

import enum
import hashlib
import zlib
from typing import Iterable, Sequence

import numpy as np

from . import _rcengine as _rc

__all__ = ["RateC", "RatecError", "RatecErrorCode", "patch_from_sur",
           "batch_spans", "chunked_prep", "export_tokenizer_json",
           "export_gguf", "__version__"]
RatecError = _rc.RatecError


class RatecErrorCode(str, enum.Enum):
    """The error taxonomy (PLAN_v0.9 P2.3) — every RatecError carries a
    .code from this fixed set; pipelines branch on codes, never on
    message strings. .code is the plain string value (the C sets it);
    `RatecErrorCode(err.code)` converts. The C raise-site mapping is
    asserted per site by tests/test_errors.py and the mapping table
    lives in _rcengine.c's ratec_err comment block."""
    TILING = "TILING"                        # spans must partition the stream
    BOOKLESS = "BOOKLESS"                    # operation needs store_book=True
    CAP_EXHAUSTED = "CAP_EXHAUSTED"          # raise map_capbits / chunk it
    BUFFER_LIMIT = "BUFFER_LIMIT"            # 4 GiB uint32 span space hit
    SNAPSHOT_INVALID = "SNAPSHOT_INVALID"    # corrupt/truncated/incompatible state
    SNAPSHOT_VERSION = "SNAPSHOT_VERSION"    # state format version not understood
    ARGUMENT = "ARGUMENT"                    # caller-side contract violation
    STATE = "STATE"                          # engine poisoned/busy/internal failure


# NOTE: STATE is the plan's 7-code draft + 1 — two OOM-adjacent engine-
# state sites (poisoned after MemoryError, batch worker allocation
# failure) fit no caller-facing code; recorded in CHANGELOG as the
# documented refinement.


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
                 prune_min: int = 2, reserve: int | None = None):
        if prune and prune_after < 1:
            raise ValueError("prune_after must be >= 1 when prune=True")
        if prune_min < 1:
            raise ValueError("prune_min must be >= 1")
        if reserve is not None and reserve < 0:
            raise ValueError("reserve must be >= 0")
        self._eng = _rc.Engine(c=c, store_book=store_book, pf=pf,
                               prebits=prebits, map_capbits=map_capbits,
                               exact=exact, carry=carry,
                               prune_after=prune_after if prune else 0,
                               prune_min=prune_min)
        if reserve is not None:
            # v0.9.1 public reserve=: pre-size the context maps for an
            # n-byte run before the first encode — a sized cold encode
            # shows measurably fewer growth rehashes (the instrumented
            # half of the gate is RateC.rehash_count; the timing half is
            # the bench). Internally this is the same core_reserve the
            # stream head and batch pool already call.
            self._eng.reserve(int(reserve))
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
                               mode: str = "runs", as_arrays: bool = False,
                               ids: bool = False,
                               intern_threads: int | None = None):
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

        ids=True (v0.9.0) adds the token stream: returns (starts, ends,
        ids) uint32 ndarrays where ids[k] indexes THIS instance's book in
        first-seen order. The parallel workers stay bookless; a single
        sequential intern-only pass (Engine.intern_spans) assigns ids
        from the span contents — the judge is never consulted, so ids
        depend only on span contents and order. Requires store_book=True
        (the ids decode via decode_bytes/decode_batch on this instance;
        the book persists across calls, so repeated calls share one
        vocabulary). ids=True implies the array form.

        intern_threads (v0.9.0): worker count for the intern pass
        (default = threads). The pass splits the span array into
        contiguous chunks, interns each into a private table (zero
        copies), merges in chunk order into the real book, and remaps —
        the ids are bit-exactly the sequential pass's for ANY thread
        count, so threads only change speed, never output.
        """
        if mode not in ("runs", "snapshot"):
            raise ValueError("mode must be 'runs' or 'snapshot'")
        if ids and not self._eng.store_book:
            raise ValueError(
                "ids=True requires store_book=True on this instance (the "
                "ids index this engine's book and decode through it)")
        data = _as_bytes(data)
        n = len(data)
        if threads <= 0:
            import os
            threads = os.cpu_count() or 1
        if chunk is None:
            chunk = max(1 << 20, min(8 << 20, n // max(1, threads)))
        want_arrays = as_arrays or ids
        if intern_threads is None:
            intern_threads = threads
        if n <= chunk:
            starts, ends = self.encode_to_spans(data, as_arrays=True)
            return self._stream_ids_tail(data, starts, ends,
                                         ids=ids, as_arrays=as_arrays,
                                         intern_threads=intern_threads)
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
            if want_arrays:
                sb, eb = eng_head.feed_raw(seg)
                head_starts.append(np.frombuffer(sb, dtype=np.uint32))
                head_ends.append(np.frombuffer(eb, dtype=np.uint32))
            else:
                out.extend(eng_head.feed(seg))
            consumed += len(seg)
        if want_arrays:
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
                    if want_arrays:
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
            if want_arrays:
                sa = head_starts + [p[0] for p in spans_parts if p is not None]
                ea = head_ends + [p[1] for p in spans_parts if p is not None]
                sf = np.concatenate(sa) if len(sa) > 1 else sa[0].copy()
                ef = np.concatenate(ea) if len(ea) > 1 else ea[0].copy()
                return self._stream_ids_tail(data, sf, ef,
                                             ids=ids, as_arrays=as_arrays,
                                             intern_threads=intern_threads)
            for part in spans_parts:
                if part:
                    starts_all.extend(part)
            return out + starts_all

        def run(a: int, b: int):
            eng = pool.get()
            try:
                eng.restore_from(eng_head)
                if want_arrays:
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
        if want_arrays:
            sa = head_starts + [p[0] for p in parts]
            ea = head_ends + [p[1] for p in parts]
            sf = np.concatenate(sa) if len(sa) > 1 else sa[0].copy()
            ef = np.concatenate(ea) if len(ea) > 1 else ea[0].copy()
            return self._stream_ids_tail(data, sf, ef,
                                         ids=ids, as_arrays=as_arrays,
                                         intern_threads=intern_threads)
        for spans_part in parts:
            starts_all.extend(spans_part)
        return out + starts_all

    def _stream_ids_tail(self, data, starts, ends, *, ids, as_arrays,
                         intern_threads=1):
        """Shared tail of encode_stream_parallel: attach the intern-pass
        token stream (ids=True) or convert back to tuples. The span
        arrays are rebased to THIS call's input (the intern pass
        validates call-local tiling; a continued engine emits
        lifetime-absolute offsets — v0.9.0rc fix, ids are
        content-derived so the rebasing cannot change them)."""
        if len(starts):
            base = int(starts[0])
            if base:
                starts = starts - base
                ends = ends - base
        if ids:
            if intern_threads > 1:
                raw = self._eng.intern_spans_parallel(
                    data, starts.tobytes(), ends.tobytes(),
                    int(intern_threads))
            else:
                raw = self._eng.intern_spans(data,
                                             starts.tobytes(),
                                             ends.tobytes())
            return (starts, ends, np.frombuffer(raw, dtype=np.uint32))
        if not as_arrays:
            return [(s, e, None) for s, e in zip(starts.tolist(),
                                                 ends.tolist())]
        return (starts, ends)

    # ----------------------------------------------------------- bytes API
    def encode_bytes(self, data) -> list[int]:
        return self._enc(bytes(data))

    def encode_to_spans(self, data, as_arrays: bool = False):
        """byte spans [(start, end, id|None), ...] — the LM-facing view.

        as_arrays=True (v0.8.9) returns (starts, ends) little-endian
        uint32 ndarrays instead of per-span tuples — the same values
        with ~50x less Python-object allocation (the tuple path costs
        ~3 objects per span, GIL-held; arrays are two memcpys + two
        numpy wraps). Spans are lifetime-absolute stream offsets: a
        continued engine's spans run past earlier calls (subtract
        starts[0] for call-local tiling) — the same convention as
        feed()/flush()."""
        data = _as_bytes(data)
        # Spans are LIFETIME-ABSOLUTE stream offsets (as the v4/v6 goldens
        # and the dataset builder rely on): a continued engine's spans run
        # past earlier calls — subtract starts[0] for call-local tiling.
        # encode_stream_parallel does that rebasing for the ids path.
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

    def rehash_count(self) -> int:
        """Context-map growth rehashes since engine birth (tmap + cmap;
        runtime stat, not serialized). The instrumented counter behind
        the public reserve= gate (PLAN_v0.9 P1.5): a sized cold encode
        must show measurably fewer rehashes, and the count is the
        honest half of that gate (the timing half is the bench)."""
        return int(self._eng.rehash_count())

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
            # a corrupt RCZ1 container must surface as a coded RatecError,
            # not an unrelated zlib.error leaking out of load_state (v0.6.0;
            # coded in v0.9.1)
            err = RatecError(f"corrupt ratec state container: {e}")
            err.code = "SNAPSHOT_INVALID"
            raise err from None
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
                fast: bool = False, carry: bool = False,
                reserve: int | None = None):
    """Parallel rate-patching of independent byte documents.

    Returns a list of (starts ndarray uint32, n_bytes); patch k spans
    [starts[k], starts[k+1]), the last ends at n_bytes. Docs are encoded
    cold-start and independently — identical results to `RateC(
    store_book=False)` per doc, across all cores. `fast=True` uses the
    non-exact fast lane (see RateC(exact=False)). `reserve=n` pre-sizes
    the shared pool's context maps for an n-byte doc before the first
    encode (v0.9.1 public reserve=: reset() keeps grown capacity, so
    one reserve serves the whole batch; see RateC(reserve=) and
    `Engine.rehash_count` for the instrumented gate).
    """
    raws = _rc.batch([bytes(d) for d in docs], c=float(c), threads=threads,
                     fast=bool(fast), carry=bool(carry),
                     **({"reserve": int(reserve)} if reserve is not None
                        else {}))
    return [(np.frombuffer(raw, dtype=np.uint32), len(d))
            for raw, d in zip(raws, docs)]


def batch_encode(docs: Sequence, c: float = 20, threads: int = 0,
                 fast: bool = False, carry: bool = False,
                 reserve: int | None = None):
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
    the fastest path to training-ready tokens. `reserve=n` pre-sizes
    the shared pool's context maps for an n-byte doc (v0.9.1 public
    reserve= — fewer growth rehashes on cold pools; see
    `RateC.rehash_count`).
    """
    raws = _rc.batch([bytes(d) for d in docs], c=float(c), threads=threads,
                     fast=bool(fast), carry=bool(carry), ids=True,
                     **({"reserve": int(reserve)} if reserve is not None
                        else {}))
    out = []
    for starts_b, ends_b, ids_b in raws:
        starts = np.frombuffer(starts_b, dtype=np.uint32)
        ends = np.frombuffer(ends_b, dtype=np.uint32)
        ids = np.frombuffer(ids_b, dtype=np.uint32)
        out.append((ids.astype(np.int32),
                    ends.astype(np.int32) - starts.astype(np.int32)))
    return out


__version__ = _rc.version()


# ------------------------------------------------------- scale + ecosystem
def chunked_prep(chunks, c: float = 20, engine_bytes: int = 1 << 32,
                 threads: int = 2, ids: bool = True, exact: bool = True,
                 carry: bool = False, reserve: int | None = None,
                 sink=None):
    """Prep an arbitrarily long stream within the uint32 span space.

    Span offsets are uint32 — ONE engine covers at most 4 GiB of
    lifetime stream (the documented BUFFER_LIMIT). `chunked_prep` turns
    that limit from folklore into a tested helper: it consumes `chunks`
    (bytes, or an iterable of byte chunks — the streaming shape keeps
    RSS flat on corpora bigger than RAM), splits the stream into
    segments of at most `engine_bytes` bytes, and encodes each segment
    on a FRESH engine (a fresh judge + book per segment — exactly the
    per-chunk-engines pattern the scale record documents, now with the
    bookkeeping handled).

    Per segment the result is identical to encoding that segment's
    bytes on a standalone engine: ids bit-exact, spans tiling the
    segment exactly (the PLAN_v0.9 P2.1 gate asserts sha256 equality
    against explicit per-segment engines on an 8 GiB synthetic stream —
    bench/gate_8gib.py). Segment results carry `seg_bytes` so offsets
    can be re-based into a u64 global space downstream; the arrays
    themselves stay uint32 per segment.

    Parameters
    ----------
    chunks : bytes | bytearray | iterable of bytes-like
        The stream. An iterable is consumed lazily; a chunk may exceed
        `engine_bytes` (it is split at segment boundaries).
    engine_bytes : int
        Per-engine span budget, <= the 4 GiB hard limit. Operators
        choose it from RAM: a segment's ids+spans arrays are held only
        for the duration of that segment (and only when `sink` is None
        are they all retained).
    ids : bool
        Produce the LM-prep token stream per segment (ids + lens)
        alongside spans.
    sink : callable | None
        Called as `sink(result_dict)` per segment as soon as it is
        encoded; when given, results are NOT retained (the flat-RSS
        shape for multi-GiB jobs). Each result dict:
        ``{"seg_index", "seg_bytes", "starts", "ends", "ids", "lens"}``
        (ids/lens None when ids=False).

    Returns
    -------
    list of result dicts (empty-safe to ignore when sink is used).
    """
    if engine_bytes < 1 or engine_bytes > (1 << 32):
        raise ValueError("engine_bytes must be 1 .. 2**32 (the uint32 "
                         "span space)")
    if isinstance(chunks, (bytes, bytearray, memoryview)):
        chunks = [bytes(chunks)]
    out = [] if sink is None else None
    seg = bytearray()
    seg_index = 0

    def finish():
        nonlocal seg_index, seg
        data = bytes(seg)
        seg.clear()
        # ids=True needs a stored book (the intern pass interns into it),
        # so segment engines are always book mode — discarded per segment.
        # ONE encode_stream_parallel call per segment: a second call would
        # run on a warmed judge + book (a continued engine), which is NOT
        # the per-fresh-engine pattern this helper documents.
        eng = RateC(c=c, store_book=True, exact=exact, carry=carry, pf=2)
        if ids:
            starts_a, ends_a, tok = eng.encode_stream_parallel(
                data, threads=threads, ids=True)
            res = {"seg_index": seg_index, "seg_bytes": len(data),
                   "starts": np.asarray(starts_a, dtype=np.uint32),
                   "ends": np.asarray(ends_a, dtype=np.uint32),
                   "ids": np.asarray(tok, dtype=np.int32),
                   "lens": (np.asarray(ends_a, dtype=np.uint32) -
                            np.asarray(starts_a, dtype=np.uint32)
                            ).astype(np.int32)}
        else:
            starts_a, ends_a = eng.encode_stream_parallel(
                data, threads=threads, as_arrays=True)
            res = {"seg_index": seg_index, "seg_bytes": len(data),
                   "starts": np.asarray(starts_a, dtype=np.uint32),
                   "ends": np.asarray(ends_a, dtype=np.uint32),
                   "ids": None, "lens": None}
        seg_index += 1
        if sink is not None:
            sink(res)
        else:
            out.append(res)
        del eng

    for chunk in chunks:
        chunk = bytes(chunk)
        view = memoryview(chunk)
        while len(view):
            take = min(len(view), engine_bytes - len(seg))
            seg += view[:take]
            view = view[take:]
            if len(seg) >= engine_bytes:
                finish()
    if seg:
        finish()
    return out if sink is None else []


# GGUF metadata value types (llama.cpp enum gguf_type)
_T_U8, _T_I8, _T_U16, _T_I16, _T_U32, _T_I32, _T_F32, _T_BOOL = range(8)
_T_STRING, _T_ARRAY, _T_U64, _T_I64, _T_F64 = 8, 9, 10, 11, 12


def export_gguf(engine, path=None):
    """Export a trained engine's book as a GGUF v3 metadata container.

    The llama.cpp bridge (PLAN_v0.9 P3.2) — as a DOCUMENTED SUBSET: the
    container is a valid GGUF v3 file (magic, version, zero tensors,
    typed KVs) that any GGUF parser can read, carrying

      tokenizer.ggml.model      "ratec-adaptive" (NOT a llama.cpp BPE:
                                llama.cpp must not be used to
                                re-tokenize with this file — ratec's
                                segmentation is the algorithm, and the
                                vocab table alone cannot reproduce it)
      tokenizer.ggml.tokens     the book, index == id, latin-1 strings
                                (byte-exact, invertible)
      tokenizer.ggml.scores     0.0 per token (no learned scores — the
                                vocabulary is adaptive, not trained)
      tokenizer.ggml.token_type 1 (NORMAL) per token
      ratec.c / ratec.book.count / ratec.book.sha256
      ratec.snapshot.v8         the engine's full v8 snapshot, byte
                                array — the authoritative state; sha256
                                bound by ratec.snapshot.sha256

    The round-trip gate (tests/test_gguf.py) parses the container with
    an independent GGUF reader and asserts vocab equality + that the
    embedded snapshot reloads bit-exactly. `engine` needs
    store_book=True (BOOKLESS otherwise, like decode). Returns the
    bytes; writes to `path` when given.
    """
    dump = (engine.book_dump() if hasattr(engine, "book_dump")
            else engine._eng.book_dump())
    snapshot = engine._eng.save_state()
    digest = hashlib.sha256(snapshot).hexdigest()

    import struct as _s

    def gstr(x):
        b = x.encode("utf-8")
        return _s.pack("<Q", len(b)) + b

    def garr(elem_type, packer, vals):
        return _s.pack("<IQ", elem_type, len(vals)) + b"".join(
            packer(v) for v in vals)

    kvs = []
    u32 = lambda v: _s.pack("<I", v)
    f64 = lambda v: _s.pack("<d", v)
    u64 = lambda v: _s.pack("<Q", v)
    f32 = lambda v: _s.pack("<f", v)
    i32 = lambda v: _s.pack("<i", v)

    def kv(key, typ, packer, val):
        kvs.append(gstr(key) + u32(typ) + packer(val))

    kv("general.architecture", _T_STRING, gstr, "ratec")
    kv("general.name", _T_STRING, gstr, "ratec-adaptive-book")
    kv("ratec.format.version", _T_U32, u32, 1)
    kv("ratec.c", _T_F64, f64, float(engine.c))
    kv("ratec.book.count", _T_U64, u64, len(dump))
    kv("ratec.book.sha256", _T_STRING, gstr,
       hashlib.sha256(b"".join(dump)).hexdigest())
    kv("ratec.snapshot.v8", _T_ARRAY,
       lambda vals: _s.pack("<IQ", _T_U8, len(vals))
       + b"".join(_s.pack("<B", v) for v in vals), list(snapshot))
    kv("ratec.snapshot.sha256", _T_STRING, gstr, digest)
    kv("tokenizer.ggml.model", _T_STRING, gstr, "ratec-adaptive")
    kv("tokenizer.ggml.tokens", _T_ARRAY,
       lambda vals: _s.pack("<IQ", _T_STRING, len(vals))
       + b"".join(gstr(v.decode("latin-1")) for v in vals), list(dump))
    kv("tokenizer.ggml.scores", _T_ARRAY,
       lambda vals: _s.pack("<IQ", _T_F32, len(vals))
       + b"".join(f32(v) for v in vals), [0.0] * len(dump))
    kv("tokenizer.ggml.token_type", _T_ARRAY,
       lambda vals: _s.pack("<IQ", _T_I32, len(vals))
       + b"".join(i32(v) for v in vals), [1] * len(dump))

    out = (b"GGUF" + _s.pack("<I", 3) + _s.pack("<Q", 0)
           + _s.pack("<Q", len(kvs)) + b"".join(kvs))
    if path is not None:
        with open(path, "wb") as f:
            f.write(out)
    return out

def export_tokenizer_json(engine, path=None, pretty: bool = True):
    """Export a trained engine's book as an HF `tokenizer.json` vocab.

    The bridge that lets a frozen ratec snapshot load into
    HF `transformers` (PLAN_v0.9 P3.1): the book's (id -> token bytes)
    mapping becomes a WordLevel vocab. Token bytes are mapped to JSON
    strings one-to-one via latin-1 (every byte is a valid latin-1
    codepoint, so the mapping is invertible and byte-exact — the
    round-trip gate in tests/test_v091.py reconstructs the book bytes
    from the JSON alone). Consumers that tokenize WITH this vocab must
    use ratec's segmentation (the vocab is the book's content table,
    not a whitespace splitter); the documented contract is: for every
    id the engine produces, `vocab[latin1(book_bytes[id])] == id`, and
    decoding a ratec ids stream through that inverse yields
    `engine.decode_bytes(ids)` exactly.

    `engine` needs store_book=True (BOOKLESS otherwise, like decode).
    Returns the JSON string; writes to `path` when given.
    """
    dump = (engine.book_dump() if hasattr(engine, "book_dump")
            else engine._eng.book_dump())   # ordered token bytes, index == id
    vocab = {}
    for tid, content in enumerate(dump):
        vocab[content.decode("latin-1")] = tid
    if len(vocab) != len(dump):
        raise RatecError("book contains duplicate token strings; "
                         "vocab would collide")
    doc = {
        "version": "1.0",
        "truncation": None,
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        # the identity Replace decoder is load-bearing: with
        # `decoder: null`, tokenizers/transformers/vLLM joins pieces
        # with spaces on decode and the byte-exact contract breaks
        # (an rc integration proof caught this: 200 ids decoded to
        # 2065 bytes). A never-matching Replace keeps every piece
        # verbatim and switches the join to "" — decode(ids) then
        # latin-1-encodes back to the exact source bytes.
        "decoder": {"type": "Replace",
                    "pattern": {"String": "\u0000"},
                    "content": ""},
        "added_tokens": [],
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }
    import json as _json
    s = _json.dumps(doc, ensure_ascii=False,
                    indent=2 if pretty else None)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


# ------------------------------------------------------------------ bench
def bench(n: int = 4 << 20, c: int = 20, threads: int = 2, rounds: int = 3):
    """Throughput of the main entry points, measured on this machine NOW.

    Self-contained deterministic mixed-content stream (no fixture files),
    best-of-`rounds` per entry. Prints a table and returns the dict so
    scripts/CI can diff. Absolute numbers are host-dependent; the
    RELATIVE ordering is what the design promises (see BENCHMARKS.md):

        encode_stream_parallel(spans) > encode_to_spans > encode_bytes
        stream ids adds the sequential intern pass (+~10-15% time)
        decode_bytes ~ 150+ Mids/s single-thread, decode_batch scales

    Entries (all exact lane unless noted):
      encode_bytes      sequential judge+patcher, ids + book (store_book)
      encode_to_spans   sequential spans, tuple-free as_arrays output
      stream_parallel   encode_stream_parallel, runs mode, spans
      stream_ids        the same with ids=True (spans + intern pass)
      decode_bytes      full book read of the stream's ids
      batch             ratec.batch_encode, thread-parallel, spans
    """
    import time
    line = (b"def encode_stream(data: bytes) -> list:  # the quick brown "
            b"fox jumps over the lazy dog\n    out = []\n    for i, b in "
            b"enumerate(data):\n        if b % 7:\n            out.append("
            b"(i * 31 + b) & 0xFFFF)\n# utf8: \xe4\xbd\xa0\xe5\xa5\xbd"
            b" \xd0\xbf\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82\n")
    data = (line * (n // len(line) + 1))[:n]

    def best(f):
        ts = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    r = RateC(c=c, store_book=True)
    t_enc = best(lambda: r.encode_bytes(data))
    r.reset()
    t_spans = best(lambda: r.encode_to_spans(data, as_arrays=True))

    rp = RateC(c=c, store_book=True)
    t_par = best(lambda: rp.encode_stream_parallel(
        data, threads=threads, as_arrays=True))
    t_ids = best(lambda: rp.encode_stream_parallel(
        data, threads=threads, ids=True))
    starts, ends, tok = rp.encode_stream_parallel(
        data, threads=threads, ids=True)
    t_dec = best(lambda: rp.decode_bytes(tok))

    t_batch = best(lambda: batch_encode(
        [data[i:i + (1 << 20)] for i in range(0, len(data), 1 << 20)],
        c=c, threads=threads))

    res = {
        "n_bytes": n, "c": c, "threads": threads, "rounds": rounds,
        "encode_bytes_mb_s": n / t_enc / 1e6,
        "encode_to_spans_mb_s": n / t_spans / 1e6,
        "stream_parallel_mb_s": n / t_par / 1e6,
        "stream_ids_mb_s": n / t_ids / 1e6,
        "decode_bytes_mb_s": n / t_dec / 1e6,
        "batch_mb_s": n / t_batch / 1e6,
    }
    print(f"ratec {__version__} bench: {n >> 20} MiB, c={c}, "
          f"threads={threads}, best-of-{rounds}")
    print(f"  encode_bytes        {res['encode_bytes_mb_s']:8.2f} MB/s")
    print(f"  encode_to_spans     {res['encode_to_spans_mb_s']:8.2f} MB/s")
    print(f"  stream_parallel     {res['stream_parallel_mb_s']:8.2f} MB/s")
    print(f"  stream ids (+intern){res['stream_ids_mb_s']:8.2f} MB/s")
    print(f"  decode_bytes        {res['decode_bytes_mb_s']:8.2f} MB/s")
    print(f"  batch_encode        {res['batch_mb_s']:8.2f} MB/s")
    return res
