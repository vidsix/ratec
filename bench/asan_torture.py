#!/usr/bin/env python3
"""asan_torture.py — the differential-leak torture workload.

Runs the FULL engine surface N cycles (train, encode/decode, batch,
save/load, tokenizer.json, gguf, chunked_prep, reserve/rehash), samples
RSS, and exits normally so LeakSanitizer reports at process exit.

The sanitizer gate runs this at cycles = 1, 30, 60 and compares the
LeakSanitizer byte totals: a REAL leak scales with the cycle count; a
flat total across scales is the CPython interpreter baseline. That
differential converts "RSS looked flat for a week" into "no leak on
any executed path, proven per-path by LSan".

usage: python3 bench/asan_torture.py <cycles>
"""

import json
import os
import random
import sys
import tempfile
import threading
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ratec

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
         "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
         "пси омега Привет мир 中文测试 トークナイザ emoji ✓ ☂ 😀 "
         "def fn(x): return x*2  class Foo: pass  while True:").split()


def corpus(seed: int, mb: float = 1.5) -> bytes:
    rng = random.Random(seed)
    words = list(WORDS) + [f"tok{rng.randrange(10**6)}" for _ in range(2000)]
    n = int(mb * 1024 * 1024 / 8)
    return " ".join(rng.choice(words) for _ in range(n)).encode("utf-8")


def docs(seed: int, n: int = 50) -> list:
    rng = random.Random(seed ^ 0xBEEF)
    out = []
    for i in range(n):
        size = rng.choice([100, 1000, 5000, 20000, 50000])
        out.append((" ".join(rng.choice(WORDS)
                            for _ in range(size // 6))).encode("utf-8"))
    return out


class RSSWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            kb = int(line.split()[1])
                            self.peak = max(self.peak, kb)
                            self.samples.append(kb)
                            break
            except OSError:
                pass
            time.sleep(0.25)


def one_cycle(i: int):
    c = 20 + (i % 14)
    eng = ratec.get_ratec(c=c, cache=False)
    eng.train(corpus(i))
    for d in docs(i):
        ids = eng.encode(d)
        assert eng.decode(ids) == d.decode('utf-8')
    ds = docs(i ^ 1, 20)
    b = ratec.batch_encode(ds, c=c, threads=2)
    assert len(b) == 20
    import numpy as np
    ib = [np.asarray(eng.encode_bytes(d), dtype=np.uint32) for d in ds]
    assert eng.decode_batch(ib, threads=2) == ds
    st = eng.save_state()
    back = ratec.load(st)
    assert back.encode(b"roundtrip probe") == eng.encode(b"roundtrip probe")
    js = json.loads(ratec.export_tokenizer_json(eng, pretty=(i % 2 == 0)))
    assert "model" in js
    gg = os.path.join(tempfile.gettempdir(), f"rc_tort_{i}.gguf")
    ratec.export_gguf(eng, gg)
    os.unlink(gg)
    if hasattr(eng, "reserve"):
        eng.reserve(eng.n_vocab + 4096)
    if hasattr(eng, "rehash_count"):
        _ = eng.rehash_count()
    # chunked_prep exercises the lazy-chunk path if this build has it
    cp = getattr(ratec, "chunked_prep", None)
    if cp is not None and i % 5 == 0:
        list(cp(corpus(i ^ 3, 0.6), c=c))
    del eng, back


def main(cycles: int):
    w = RSSWatch()
    w.start()
    warm = corpus(999, 0.5)
    eng0 = ratec.get_ratec(c=20, cache=False)
    eng0.train(warm)
    ids0 = eng0.encode(warm[:4096])
    assert eng0.decode(ids0) == warm[:4096].decode('utf-8')
    t0 = time.time()
    for i in range(cycles):
        one_cycle(i)
    dt = time.time() - t0
    w.stop.set()
    w.join(timeout=1.0)
    tail = w.samples[-20:] if w.samples else [0]
    print(f"TORTURE cycles={cycles} dt={dt:.1f}s "
          f"rss_peak={w.peak >> 10}MiB "
          f"rss_tail_min={min(tail) >> 10}MiB "
          f"rss_tail_max={max(tail) >> 10}MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1))
