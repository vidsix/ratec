"""ratec — the rate-C information-vocabulary tokenizer, as a library.

RateC replaces BPE's learned merge table with an adaptive 6-order KT
mixture (the judge) that scores every byte, and a greedy rate-packing rule
(the patcher) that cuts patches at a C-bit budget. Zero training data, zero
vocab files, any C, any byte stream (text, code, CJK, images, audio, ...).

    >>> import ratec
    >>> tok = ratec.RateC(c=20)
    >>> tok.decode(tok.encode("你好, hello!")) == "你好, hello!"
    True
"""
from .api import (RateC, RatecError, patch_from_sur, batch_spans, solve_c,
                  solve_c_for_book, __version__)

__all__ = ["RateC", "RatecError", "patch_from_sur", "batch_spans", "solve_c",
           "solve_c_for_book", "__version__"]
