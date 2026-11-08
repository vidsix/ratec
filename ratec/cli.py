"""ratec.cli — command line interface.

    python -m ratec encode FILE [-o OUT.ids] [--c 20]
    python -m ratec decode IDS FILE.state -o out.bin
    python -m ratec surprise FILE [-o OUT.f32] [--c 20] [--patch]
    python -m ratec state FILE -o model.rcstate      # snapshot ("warm start")
    python -m ratec bench FILE [--reps 15]

`--c` is a float everywhere (fractional budgets like 17.5 are first-class).
"""
import argparse
import sys
import time
import numpy as np

from .api import RateC, patch_from_sur, __version__


def main(argv=None):
    p = argparse.ArgumentParser("ratec", description="RateC tokenizer CLI")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="file -> ids (or span count with --spans)")
    pe.add_argument("file"); pe.add_argument("-o", "--out", default="-")
    pe.add_argument("--c", type=float, default=20)
    pe.add_argument("--spans", action="store_true",
                    help="print spans instead of ids")

    pd = sub.add_parser("decode", help="ids file + state file -> bytes")
    pd.add_argument("ids"); pd.add_argument("state")
    pd.add_argument("-o", "--out", required=True)

    ps = sub.add_parser("surprise", help="file -> float32 surprisal")
    ps.add_argument("file"); ps.add_argument("-o", "--out", default="-")
    ps.add_argument("--c", type=float, default=20)
    ps.add_argument("--patch", action="store_true",
                    help="also print patch count at --c")

    psx = sub.add_parser("state", help="build a warm-start snapshot")
    psx.add_argument("file", nargs="+")
    psx.add_argument("-o", "--out", required=True)
    psx.add_argument("--c", type=float, default=20)

    pb = sub.add_parser("bench", help="throughput of this file")
    pb.add_argument("file"); pb.add_argument("--reps", type=int, default=15)
    pb.add_argument("--c", type=float, default=20)

    a = p.parse_args(argv)

    if a.cmd == "encode":
        tok = RateC(c=a.c)
        data = open(a.file, "rb").read()
        t0 = time.perf_counter()
        spans = tok.encode_to_spans(data)
        dt = time.perf_counter() - t0
        mb = len(data) / dt / 1e6
        if a.spans:
            text = "\n".join(f"{s} {e}" for s, e, _ in spans)
            out = text + "\n"
        else:
            out = "\n".join(str(i) for _, _, i in spans) + "\n"
        if a.out == "-":
            sys.stdout.write(out)
        else:
            open(a.out, "w").write(out)
        print(f"# {len(data)} bytes -> {len(spans)} patches, "
              f"{mb:.2f} MB/s, book={tok.book_size}, "
              f"forced={tok.n_forced}", file=sys.stderr)

    elif a.cmd == "decode":
        tok = RateC(c=20)
        tok._eng = __import__("ratec")._rcengine.load_state(
            __import__("ratec").api._maybe_decompress(open(a.state, "rb").read()))
        ids = [int(x) for x in open(a.ids).read().split()]
        data = tok.decode_bytes(ids)
        open(a.out, "wb").write(data)
        print(f"# decoded {len(ids)} patches -> {len(data)} bytes", file=sys.stderr)

    elif a.cmd == "surprise":
        tok = RateC(c=a.c)
        data = open(a.file, "rb").read()
        sur = tok.surprise(data)
        raw = sur.tobytes()
        if a.out == "-":
            sys.stdout.buffer.write(raw)
        else:
            open(a.out, "wb").write(raw)
        if a.patch:
            starts = patch_from_sur(sur, a.c)
            print(f"# {len(data)} bytes -> {len(starts)} patches "
                  f"(C={a.c:g})", file=sys.stderr)

    elif a.cmd == "state":
        tok = RateC(c=a.c)
        for path in a.file:
            tok._eng.feed(open(path, "rb").read())
        tok.flush()
        tok.save_state(a.out)
        print(f"# state: {tok.bytes_seen} bytes seen, book={tok.book_size} "
              f"-> {a.out}", file=sys.stderr)

    elif a.cmd == "bench":
        data = open(a.file, "rb").read()
        tok = RateC(c=a.c)
        best = float("inf")
        for _ in range(a.reps):
            tok.reset()
            t0 = time.perf_counter()
            tok.encode_to_spans(data)
            best = min(best, time.perf_counter() - t0)
        mb = len(data) / best / 1e6
        print(f"{len(data)} bytes  C={a.c}  best {best*1e3:.2f} ms  "
              f"{mb:.2f} MB/s  patches={tok.n_patches}  "
              f"units/byte={tok.n_patches/max(1,len(data)):.4f}")


if __name__ == "__main__":
    main()
