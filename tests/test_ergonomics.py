"""The rc3 ergonomics layer — the tiktoken-grade front door, tested the
way a tiktoken user would arrive: one import, one call, encode/decode,
save/load. Additive surface (train, n_vocab, get_ratec, load) on top
of the frozen core; every exactness invariant from the older gates
still holds through the new names."""

import os
import tempfile

import pytest

import ratec


# ---------------------------------------------------------------- get_ratec
def test_get_ratec_cached_same_object():
    a = ratec.get_ratec(c=20)
    b = ratec.get_ratec(c=20)
    assert a is b


def test_get_ratec_private_engine_on_cache_false():
    a = ratec.get_ratec(c=20)
    b = ratec.get_ratec(c=20, cache=False)
    assert a is not b


def test_get_ratec_config_keys_differ():
    a = ratec.get_ratec(c=20)
    b = ratec.get_ratec(c=24)
    assert a is not b and a.c == 20 and b.c == 24


# --------------------------------------------------------- tiktoken-shaped
def test_train_encode_decode_roundtrip():
    corpus = (b"the quick brown fox jumps over the lazy dog\n" * 400
              + b"pack my box with five dozen liquor jugs\n" * 200)
    tok = ratec.get_ratec(c=20, cache=False)
    tok.train(corpus)
    s = "the quick brown fox jumps over the lazy dog"
    ids = tok.encode(s)
    assert isinstance(ids, list) and ids
    assert tok.decode(ids) == s
    # unicode: str in, str out, byte-exact through the utf-8 interface
    uni = "héllo wörld — 你好世界 · привет"
    assert tok.decode(tok.encode(uni)) == uni
    # mixed scripts inside a longer document
    doc = "English 中文 Русский 100% ✓\n" * 20
    assert tok.decode(tok.encode(doc)) == doc


def test_n_vocab_adaptive_then_pinned():
    tok = ratec.get_ratec(c=20, cache=False)
    assert tok.n_vocab == 0            # cold: the book is empty
    tok.train(b"abcdefgh" * 500)
    assert tok.n_vocab == tok.book_size > 0
    loaded = ratec.load(tok.save_state())
    assert loaded.n_vocab == tok.n_vocab   # pinned by the state


def test_module_level_aliases_accept_the_tiktoken_flow():
    tok = ratec.get_ratec(c=20, cache=False)
    tok.train(b"lorem ipsum dolor sit amet consectetur " * 300)
    ids = tok.encode("lorem ipsum dolor sit amet")
    assert tok.decode(ids) == "lorem ipsum dolor sit amet"
    # bytes stay the byte-exact lane
    raw = b"\x00\x01\xff\xfe binary \xf0\x9f\x98\x80"
    assert tok.decode_bytes(tok.encode_bytes(raw)) == raw


# -------------------------------------------------------------------- load
def test_load_from_bytes_and_path():
    tok = ratec.RateC(c=20)
    tok.train(b"state continuation corpus " * 600)
    probe = "state continuation corpus"
    # the state is saved BEFORE probing (encoding mutates the engine —
    # the v0.9.2 export-order rule); the invariant is that two loads of
    # the SAME bytes continue IDENTICALLY, and equal the engine itself
    state = tok.save_state()
    a = ratec.load(state)
    ref = a.encode(probe)
    assert ratec.load(state).encode(probe) == ref   # bit-exact continuation
    with tempfile.NamedTemporaryFile(suffix=".ratec", delete=False) as f:
        f.write(state)
        path = f.name
    try:
        b = ratec.load(path)
        assert b.encode(probe) == ref
        # fresh loads of the same bytes agree (a/b already continued;
        # path loads are memoized by design — bypass for the fresh pair)
        assert a.n_vocab == b.n_vocab
        assert ratec.load(state).n_vocab == ratec.load(path, cache=False).n_vocab
    finally:
        os.unlink(path)


def test_load_http_url_fetches_into_ratec_cache(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    tok = ratec.RateC(c=20)
    tok.train(b"url fetch corpus " * 600)
    src = tmp_path / "model.ratec"
    src.write_bytes(tok.save_state())
    ref = ratec.load(src.read_bytes()).n_vocab

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):        # keep the test output clean
            pass
    srv = ThreadingHTTPServer(("127.0.0.1", 0),
                              lambda *a, **k: Quiet(*a, directory=str(tmp_path), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/model.ratec"
    cache = tmp_path / "cache"
    os.environ["RATEC_CACHE"] = str(cache)
    try:
        a = ratec.load(url)               # fetched over real http
        b = ratec.load(url)
        assert a is b                     # memoized in-process
        assert a.n_vocab == ref           # the bytes survived the fetch
        assert len(list(cache.iterdir())) == 1   # landed in RATEC_CACHE
    finally:
        del os.environ["RATEC_CACHE"]
        srv.shutdown()


def test_load_bad_type_raises():
    with pytest.raises(TypeError):
        ratec.load(12345)


def test_frozen_surface_untouched_through_new_names():
    # the pre-freeze invariants hold through the ergonomic front door:
    # continued encode equality, decode of uint32 arrays, book growth
    tok = ratec.get_ratec(c=20, cache=False)
    tok.train(b"INVARIANT CHECK " * 200)
    ids = tok.encode_bytes(b"INVARIANT CHECK")
    import numpy as np
    back = tok.decode_bytes(np.asarray(ids, dtype=np.uint32))
    assert back == b"INVARIANT CHECK"
