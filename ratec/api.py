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
        round-trip without the source text. Turn off for bulk LM data prep
        (use `encode_to_spans`/`batch_spans`, which only need offsets).
    pf : int
        Software prefetch depth (0-4); -1 = auto, which resolves to the
        measured default 2 on x86-64 (`PF_DEFAULT` in `_rcengine.c`).
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

    def __init__(self, c: float = 20, store_book: bool = True, pf: int = -1,
                 warm_state=None, prebits: int = 16, map_capbits: int = 22,
                 exact: bool = True, carry: bool = False):
        self._eng = _rc.Engine(c=c, store_book=store_book, pf=pf,
                               prebits=prebits, map_capbits=map_capbits,
                               exact=exact, carry=carry)
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
                              pf=self._eng.pf, exact=True,
                              carry=self._eng.carry_mode)
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


__version__ = _rc.version()
