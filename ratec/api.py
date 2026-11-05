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
    c : int
        Patch budget in bits (default 20). Smaller C -> shorter patches and
        more of them; larger C -> longer patches. Changeable per instance;
        the judge is C-independent.
    store_book : bool
        Keep the dynamic patch book (content -> id) so `encode`/`decode`
        round-trip without the source text. Turn off for bulk LM data prep
        (use `encode_to_spans`/`batch_spans`, which only need offsets).
    pf : int
        Software prefetch depth (0-4). Default 3 (best on x86-64).
    warm_state : bytes | str | None
        Snapshot bytes or path produced by `save_state` — cold-start is
        avoided and the judge starts already adapted (RateC's optional
        "model file"; entirely optional).

    Notes
    -----
    * `encode`/`decode` require the book. The book is instance state: ids
      from tokenizer A mean nothing to tokenizer B unless B was created
      with A's state snapshot.
    * Engines are not thread-safe; use one instance per thread or
      `ratec.batch_spans` for parallel bulk work.
    """

    def __init__(self, c: int = 20, store_book: bool = True, pf: int = -1,
                 warm_state=None, prebits: int = 16, map_capbits: int = 22,
                 exact: bool = True):
        self._eng = _rc.Engine(c=c, store_book=store_book, pf=pf,
                               prebits=prebits, map_capbits=map_capbits,
                               exact=exact)
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
        return self._dec(list(ids)).decode("utf-8")

    # --------------------------------------------------- parallel stream
    def encode_stream_parallel(self, data, chunk: int = 1 << 20,
                               threads: int = 0, head_chunks: int = 1):
        """Encode ONE long stream on many cores by chunking.

        The first `head_chunks` chunks run sequentially (exact judge
        adaptation); a state snapshot taken after them seeds all parallel
        workers. Chunks therefore see a judge warmed by the head but not
        by each other — a documented approximation of sequential
        `encode_to_spans` (ffmpeg-style), deterministic, spans still tile
        the input exactly and decode round-trips. For bulk LM data prep
        on real (multi-core) boxes this is the throughput mode.

        Returns [(start, end, id|None), ...] with absolute offsets.
        """
        data = _as_bytes(data)
        n = len(data)
        if n <= chunk:
            return self.encode_to_spans(data)
        from concurrent.futures import ThreadPoolExecutor

        head = min(max(0, head_chunks), (n + chunk - 1) // chunk - 1)
        # 1) sequential head on a private engine; it emits its own spans and
        #    the snapshot (taken after flush) seeds the parallel workers
        eng_head = _rc.Engine(c=self.c, store_book=False,
                              pf=self._eng.pf, exact=True)
        out = []
        consumed = 0
        for k in range(head):
            seg = data[k * chunk: (k + 1) * chunk]
            out.extend(eng_head.feed(seg))
            consumed += len(seg)
        out.extend(eng_head.flush())          # close open tail at the boundary
        snap = eng_head.save_state()
        eng_head = None
        # 2) parallel workers from the snapshot
        starts_all = []

        nseen0 = consumed      # snapshot's absolute stream position

        def run(a: int, b: int):
            eng = _rc.load_state(snap)
            sp = eng.encode_spans(data[a:b])
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
        with ThreadPoolExecutor(max_workers=threads or None) as ex:
            for spans_part in ex.map(lambda ab: run(*ab), jobs):
                starts_all.extend(spans_part)
        return out + starts_all

    # ----------------------------------------------------------- bytes API
    def encode_bytes(self, data) -> list[int]:
        return self._enc(bytes(data))

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        return self._dec(list(ids))

    def encode_to_spans(self, data) -> list[tuple[int, int, int | None]]:
        """byte spans [(start, end, id|None), ...] — the LM-facing view."""
        return self._spans(_as_bytes(data))

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
    def c(self) -> int:
        return self._eng.c

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
        return zlib.decompress(raw[4:])
    return raw


def patch_from_sur(sur: np.ndarray | bytes, c: int) -> np.ndarray:
    """Reference greedy under-shoot patcher over a stored surprisal array.

    The patcher is C-independent of the judge, so one `surprise()` pass
    serves every C — this is the cheap way to run C sweeps.
    """
    raw = sur.tobytes() if isinstance(sur, np.ndarray) else bytes(sur)
    return np.frombuffer(_rc.patch_from_sur(raw, int(c)), dtype=np.uint32)


def batch_spans(docs: Sequence, c: int = 20, threads: int = 0,
                fast: bool = False):
    """Parallel rate-patching of independent byte documents.

    Returns a list of (starts ndarray uint32, n_bytes); patch k spans
    [starts[k], starts[k+1]), the last ends at n_bytes. Docs are encoded
    cold-start and independently — identical results to `RateC(
    store_book=False)` per doc, across all cores. `fast=True` uses the
    non-exact fast lane (see RateC(exact=False)).
    """
    raws = _rc.batch([bytes(d) for d in docs], c=int(c), threads=threads,
                     fast=bool(fast))
    return [(np.frombuffer(raw, dtype=np.uint32), len(d))
            for raw, d in zip(raws, docs)]


__version__ = _rc.version()
