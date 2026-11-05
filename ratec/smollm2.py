"""ratec.smollm2 — Hugging Face transformers port of the RateC tokenizer.

SmolLM2 (and any Llama-family model) can be trained on RateC tokens. RateC
has no static vocab file — its "model file" is an engine snapshot, and the
dynamic patch book frozen at snapshot time becomes the vocabulary:

  1. Ingest the training corpus with a plain engine to grow the book:
         tok = ratec.RateC(c=20)
         for chunk in corpus: tok.encode_bytes(chunk)
         snapshot = tok.save_state()
  2. Build the HF tokenizer over the frozen snapshot:
         tok = RateCTokenizer(snapshot=snapshot)
  3. Train SmolLM2 with tokenizer.vocab_size embeddings, feeding
     (patch_id, patch_len) pairs (see `encode_for_lm`).

ID SPACES (stable, frozen at construction)
------------------------------------------
  [0, book)          frozen book ids: the patch content exactly
  [book, book+256)   byte-fallback ids: single byte `id - book` (latin-1
                     transparent); used when live patching produces a
                     patch whose content is not in the frozen book — the
                     patch is emitted as its bytes (MegaByte-style)
  [0xFFFFFFF0+i]     special tokens (up to 16), handled by transformers'
                     added-token machinery

Any text therefore round-trips byte-exactly with a frozen vocab, and
`vocab_size` never changes after construction.
"""
from __future__ import annotations

import json
import os

from . import _rcengine as _rc
from .api import _maybe_decompress

try:  # transformers is an optional dependency of ratec
    from transformers import AddedToken, LlamaConfig, PreTrainedTokenizer
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "ratec.smollm2 requires `pip install transformers` "
        "(the tokenizer side needs no torch)"
    ) from _e

__all__ = ["RateCTokenizer", "encode_for_lm", "build_smollm2_ratec_config",
           "SMOLLM2_135M_KW"]

_SPECIAL_BASE = 0xFFFFFFF0           # up to 16 special-token ids
_BYTE_FALLBACK = 256
_SNAP_FILE = "ratec_snapshot.bin"
_VOCAB_FILE = "ratec_vocab.json"


def _book_from_engine(engine) -> list[bytes]:
    """id -> content bytes, via decode_bytes on one id at a time.

    decode_bytes on a single id is exact; doing it one id at a time is
    O(n) memcpys and only happens at construction."""
    out = []
    for i in range(engine.book_size):
        out.append(engine.decode_bytes([i]))
    return out


class RateCTokenizer(PreTrainedTokenizer):
    """Hugging Face tokenizer backed by a frozen RateC engine snapshot.

    Args:
        snapshot: engine snapshot — bytes, a file path, or None (fresh
            engine with an empty book; byte-fallback still covers all
            text, the judge just starts cold).
        c: patch budget in bits when no snapshot is given.
    """

    vocab_files_names = {_SNAP_FILE: _SNAP_FILE}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, snapshot=None, c: int = 20, **kwargs):
        # ---- engine + frozen book (must exist before super().__init__,
        # ---- which adds special tokens and calls get_vocab/vocab_size)
        if snapshot is None:
            self._engine = _rc.Engine(c=int(c), store_book=True)
            self._snapshot_bytes = None
        elif isinstance(snapshot, (bytes, bytearray)):
            self._snapshot_bytes = bytes(snapshot)
            self._engine = _rc.load_state(_maybe_decompress(self._snapshot_bytes))
        else:  # path
            with open(snapshot, "rb") as f:
                self._snapshot_bytes = f.read()
            self._engine = _rc.load_state(_maybe_decompress(self._snapshot_bytes))
        self._book = _book_from_engine(self._engine)   # frozen: id -> bytes
        self._book_n = len(self._book)
        self._book_tok = {b.decode("latin-1"): i for i, b in enumerate(self._book)}
        self._id2tok = {i: b.decode("latin-1") for i, b in enumerate(self._book)}

        # clean_up would silently rewrite " ." -> "." etc. and break the
        # byte-exact round-trip guarantee
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(**kwargs)
        # Llama-style [BOS] seq [EOS] wrapping once bos/eos exist
        if self.bos_token_id is not None and self.eos_token_id is not None \
                and self.special_tokens_pattern is None:
            self.special_tokens_pattern = "bos_eos"

    # --------------------------------------------------------- id spaces
    def _special_id(self, i: int) -> int:
        return _SPECIAL_BASE + i

    @property
    def book_size(self) -> int:
        return self._book_n

    @property
    def vocab_size(self) -> int:
        return self._book_n + _BYTE_FALLBACK

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self._book_tok)
        for byte in range(256):
            tok = bytes([byte]).decode("latin-1")
            if tok not in vocab:
                vocab[tok] = self._book_n + byte
        vocab.update(self._added_tokens_encoder)
        return vocab

    # ------------------------------------------------------ tokenization
    def _encode_bytes(self, data: bytes) -> tuple[list[int], list[int]]:
        """bytes -> (ids, lens) over the frozen vocab with byte fallback."""
        base = self._engine.bytes_seen       # spans are absolute stream offs
        spans = self._engine.encode_spans(data)
        ids: list[int] = []
        lens: list[int] = []
        for s, e, _ in spans:
            content = data[s - base: e - base]
            bid = self._book_tok.get(content.decode("latin-1"))
            if bid is not None:
                ids.append(bid)
                lens.append(e - s)
            else:
                for byte in content:
                    btok = bytes([byte]).decode("latin-1")
                    bid = self._book_tok.get(btok)
                    if bid is not None:      # single byte already in book
                        ids.append(bid)
                    else:
                        ids.append(self._book_n + byte)
                    lens.append(1)
        return ids, lens

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        data = text.encode("utf-8")
        ids, _ = self._encode_bytes(data)
        return [self._id2tok.get(i) or bytes([i - self._book_n]).decode("latin-1")
                if i < self._book_n + _BYTE_FALLBACK else i
                for i in ids]

    def _convert_token_to_id(self, token: str) -> int | None:
        bid = self._book_tok.get(token)
        if bid is not None:
            return bid
        if len(token) == 1:
            return self._book_n + ord(token)   # latin-1 transparent
        return None

    def _convert_id_to_token(self, index: int) -> str:
        if index < self._book_n:
            return self._id2tok[index]
        if index < self._book_n + _BYTE_FALLBACK:
            return bytes([index - self._book_n]).decode("latin-1")
        return ""

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raw = b"".join(t.encode("latin-1") for t in tokens)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")

    def decode_bytes(self, ids) -> bytes:
        """ids -> bytes over the frozen vocab (byte-exact, no utf-8 view)."""
        out = []
        for i in ids:
            if 0 <= i < self._book_n:
                out.append(self._book[i])
            elif self._book_n <= i < self._book_n + _BYTE_FALLBACK:
                out.append(bytes([i - self._book_n]))
            elif i >= _SPECIAL_BASE:
                continue                       # special token: no bytes
            else:
                raise ValueError(f"id {i} outside the frozen vocab")
        return b"".join(out)

    # ------------------------------------------------------ specials API
    def _add_tokens(self, new_tokens, special_tokens: bool = False) -> int:
        """Assign special-token ids from the reserved block 0xFFFFFFF0+i
        instead of `len(vocab)+i`, so book growth can never collide."""
        added = 0
        new_tokens = new_tokens or []
        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token!r} is not a string.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                token = AddedToken(token, lstrip=False, rstrip=False,
                                   normalized=not special_tokens,
                                   special=special_tokens)
            if token.content in self._added_tokens_encoder:
                continue
            idx = self._special_id(len(self._added_tokens_decoder))
            token.__setstate__({"special": True, "normalized": False})
            self._added_tokens_decoder[idx] = token
            self._added_tokens_encoder[token.content] = idx
            added += 1
        self._update_trie()
        self._update_total_vocab_size()
        return added

    # ------------------------------------------------------------- saving
    def save_vocabulary(self, save_directory: str,
                        filename_prefix: str | None = None) -> tuple[str, ...]:
        save_directory = str(save_directory)
        snap_path = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + _SNAP_FILE)
        raw = self._snapshot_bytes
        if raw is None:                       # never-saved fresh engine
            raw = self._engine.save_state()
        with open(snap_path, "wb") as f:
            f.write(_maybe_decompress(raw))
        vocab_path = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + _VOCAB_FILE)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.get_vocab(), f, ensure_ascii=False)
        return (snap_path, vocab_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if os.path.isdir(str(pretrained_model_name_or_path)):
            snap = os.path.join(str(pretrained_model_name_or_path), _SNAP_FILE)
            if os.path.exists(snap) and "snapshot" not in kwargs:
                with open(snap, "rb") as f:
                    kwargs["snapshot"] = f.read()
        return super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs)

    # --------------------------------------------------------------- LM
    def encode_for_lm(self, text: str) -> tuple[list[int], list[int]]:
        """(patch_ids, patch_byte_lengths) — the LM-facing contract."""
        return self._encode_bytes(text.encode("utf-8"))

    def __repr__(self):
        return (f"RateCTokenizer(c={self._engine.c}, book={self._book_n}, "
                f"vocab_size={self.vocab_size}, "
                f"specials={len(self._added_tokens_decoder)})")


# ---------------------------------------------------------------- helpers
def encode_for_lm(texts, tokenizer: RateCTokenizer):
    """Batch helper: texts -> list of (ids, lens) pairs (no torch needed)."""
    if isinstance(texts, str):
        texts = [texts]
    return [tokenizer.encode_for_lm(t) for t in texts]


SMOLLM2_135M_KW = dict(
    hidden_size=576, intermediate_size=1536, num_hidden_layers=30,
    num_attention_heads=9, num_key_value_heads=3,
    max_position_embeddings=8192, rms_norm_eps=1e-5,
    rope_theta=1000000.0, tie_word_embeddings=True,
)


def build_smollm2_ratec_config(tokenizer: RateCTokenizer) -> "LlamaConfig":
    """SmolLM2-135M-shaped LlamaConfig sized for a RateC vocabulary."""
    cfg = LlamaConfig(vocab_size=tokenizer.vocab_size, **SMOLLM2_135M_KW)
    cfg.architectures = ["LlamaForCausalLM"]
    cfg._name_or_path = "HuggingFaceTB/SmolLM2-135M-Instruct"
    cfg.tokenizer_class = "ratec.smollm2.RateCTokenizer"
    return cfg


class SmolLM2RateCAdapter:                                # pragma: no cover
    """Torch-side adapter (SmolLM2 model <-> RateC tokens).

    The LM consumes one embedding per RateC patch; patch byte-lengths are
    provided to the model so it can apply intra-patch byte-position
    handling (MegaByte/BLT-style) if desired. Requires torch.
    """

    def __init__(self, *args, **kwargs):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "SmolLM2RateCAdapter needs torch (`pip install torch`) — "
                "the tokenizer side (RateCTokenizer) works without it") from e
