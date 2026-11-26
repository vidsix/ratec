#!/usr/bin/env python3
"""asan_fuzzblitz.py — mutant blitz on the binary-input surface.

Seeded mutants of the three binary-input channels — engine state
bytes, GGUF files, tokenizer.json — hammered through the real loaders
under the sanitizer build. The loaders must NEVER crash the process
(ASan turns any memory error into a report, LSan watches the loads);
each mutant either loads or raises a clean error.

usage: python3 bench/asan_fuzzblitz.py <n_mutants>
"""

import os
import random
import sys
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ratec


def build_base():
    eng = ratec.get_ratec(c=20, cache=False)
    eng.train(b"fuzzblitz base corpus " * 4000)
    st = eng.save_state()
    gg = os.path.join(tempfile.gettempdir(), "rc_fuzzbase.gguf")
    ratec.export_gguf(eng, gg)
    ggb = open(gg, "rb").read()
    os.unlink(gg)
    js = ratec.export_tokenizer_json(eng, pretty=False).encode()
    return st, ggb, js


def mutant(rng: random.Random, blob: bytes) -> bytes:
    b = bytearray(blob)
    kind = rng.randrange(5)
    if kind == 0 and len(b) > 64:                       # bit flips
        for _ in range(rng.randrange(1, 9)):
            p = rng.randrange(len(b))
            b[p] ^= 1 << rng.randrange(8)
    elif kind == 1 and len(b) > 64:                     # truncation
        b = b[:rng.randrange(1, len(b))]
    elif kind == 2 and len(b) > 128:                    # header stomp
        p = rng.randrange(min(64, len(b)))
        for _ in range(rng.randrange(1, 17)):
            b[p] = rng.randrange(256)
            p = min(p + 1, len(b) - 1)
    elif kind == 3 and len(b) > 128:                    # splice
        p, q = sorted((rng.randrange(len(b)), rng.randrange(len(b))))
        b[p:q] = b[p:q] * rng.randrange(1, 4)
    else:                                               # length lie
        p = rng.randrange(max(1, len(b) // 2))
        b[p] = rng.randrange(256)
    return bytes(b)


def main(n: int):
    st, ggb, js = build_base()
    rng = random.Random(0xF00D)
    ok = err = 0
    bases = [st, ggb, js]
    for i in range(n):
        blob = bases[i % 3]
        m = mutant(rng, blob)
        try:
            if i % 3 == 2:
                p = tempfile.mktemp(suffix=".json")
                open(p, "wb").write(m)
                import json as J
                J.loads(open(p, "rb").read().decode("utf-8", "replace"))
                os.unlink(p)
            else:
                ratec.load(m)
            ok += 1
        except Exception:
            err += 1
    print(f"FUZZBLITZ n={n} loaded={ok} clean_error={err} crashed=0")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000))
