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
                 warm_state=None, prebits: int = 16, map_capbits: int = 22):
        self._eng = _rc.Engine(c=c, store_book=store_book, pf=pf,
                               prebits=prebits, map_capbits=map_capbits)
        if warm_state is not None:
            state = warm_state
            if isinstance(state, (str, bytes, bytearray)) and not isinstance(state, str):
                pass
            if isinstance(state, str):
                with open(state, "rb") as f:
                    raw = f.read()
            else:
                raw = bytes(state)
            new = _rc.load_state(_maybe_decompress(raw))
            # swap in the loaded core, preserving nothing (it is the state)
            self._eng = new
            if c is not None and new.c != c:
                # caller asked for a C different than the snapshot's: keep the
                # snapshot's C (judge state is shared, C only affects the
                # patcher and changing it mid-stream is ill-defined).
                pass

    # ------------------------------------------------------------ text API
    def encode(self, text, *, c: int | None = None) -> list[int]:
        """str | bytes -> patch ids (utf-8 is the byte interface)."""
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if c is not None and c != self.c:
            raise ValueError(
                "per-call c is not supported on a stateful engine; build a "
                "RateC(c=...) or use ratec.patch_from_sur for offline sweeps")
        return self._eng.encode_bytes(data)

    def decode(self, ids: Sequence[int]) -> str:
        return self._eng.decode_bytes(list(ids)).decode("utf-8")

    # ----------------------------------------------------------- bytes API
    def encode_bytes(self, data) -> list[int]:
        return self._eng.encode_bytes(bytes(data))

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        return self._eng.decode_bytes(list(ids))

    def encode_to_spans(self, data) -> list[tuple[int, int, int | None]]:
        """byte spans [(start, end, id|None), ...] — the LM-facing view."""
        return self._eng.encode_spans(_as_bytes(data))

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


def batch_spans(docs: Sequence, c: int = 20, threads: int = 0):
    """Parallel rate-patching of independent byte documents.

    Returns a list of (starts ndarray uint32, n_bytes); patch k spans
    [starts[k], starts[k+1]), the last ends at n_bytes. Docs are encoded
    cold-start and independently — identical results to `RateC(
    store_book=False)` per doc, across all cores.
    """
    raws = _rc.batch([bytes(d) for d in docs], c=int(c), threads=threads)
    return [(np.frombuffer(raw, dtype=np.uint32), len(d))
            for raw, d in zip(raws, docs)]


__version__ = _rc.version()
