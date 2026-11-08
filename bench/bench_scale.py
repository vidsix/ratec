#!/usr/bin/env python3
"""Bench: RateC efficiency vs adaptation scale — does it get better with more data?

The design thesis: BPE's fertility is frozen at train time (a fixed merge
table performs identically on byte 1 and byte 1e12), while RateC's judge and
book adapt as they read — so its units-per-byte on held-out data should
*improve* with the size of the stream it has already seen. This bench
measures exactly that, holding the probe bytes fixed:

  for each adaptation scale s (log-spaced):
      warm one engine on corpus[:s]
      measure units/byte on the SAME fixed probe (never seen at any s)
      record book size and singleton fraction (patch types seen once)

Arms: ratec warm@scale s (one engine, adaptation continues), ratec cold
(scale 0: fresh engine per rep), tiktoken r50k_base (flat by construction;
measured, not asserted — and only when tiktoken is importable).

Portable by default: the corpus is the concatenation of the shipped scored
spans (english+code+chinese+base64 ≈ 959 KB). Point RATEC_SCALE_DATA at a
bigger plain file (e.g. a 16 MB LM pretraining mix) for the full curve —
the published README numbers come from that mode.

Outputs: table to stdout; scale.json into RATEC_OUT (default: ./results).
"""
import json
import os
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import ratec

FIXTURES = os.environ.get(
    "RATEC_FIXTURES",
    os.path.join(ROOT, "tests", "data", "fixtures"))
OUT = os.environ.get("RATEC_OUT") or os.path.join(os.getcwd(), "results")
os.makedirs(OUT, exist_ok=True)

PROBE_BYTES = int(os.environ.get("RATEC_SCALE_PROBE", 128 * 1024))

# --------------------------------------------------------------- corpus load
# Portable default: the four shipped scored spans (~959 KB of raw bytes).
# The probe is the FIXED per-domain tail (a quarter each), so every
# adaptation prefix is disjoint from it and the probe is domain-mixed; the
# adaptation pool is round-robin interleaved so every prefix is mixed too.
# RATEC_SCALE_DATA replaces all of this with one big plain file (the probe
# is then that file's tail).
def load_corpus():
    big = os.environ.get("RATEC_SCALE_DATA")
    if big:
        data = open(big, "rb").read()
        print(f"corpus: {big} ({len(data)/1e6:.2f} MB) [RATEC_SCALE_DATA]")
        pool, probe = data[:-PROBE_BYTES], data[-PROBE_BYTES:]
        return pool, probe, "RATEC_SCALE_DATA"
    doms = [open(os.path.join(FIXTURES, f"{d}.span.bin"), "rb").read()
            for d in ("english", "code", "chinese", "base64")]
    q = PROBE_BYTES // len(doms)
    probe = b"".join(d[-q:] for d in doms)
    pools = [d[:-q] for d in doms]
    block = 4096
    parts, idx = [], [0] * len(pools)
    while any(idx[i] < len(pools[i]) for i in range(len(pools))):
        for i in range(len(pools)):
            if idx[i] < len(pools[i]):
                parts.append(pools[i][idx[i]:idx[i] + block])
                idx[i] += block
    pool = b"".join(parts)
    print(f"corpus: shipped spans, round-robin interleaved "
          f"({len(pool)/1e6:.2f} MB pool + {len(probe)/1e6:.2f} MB probe) "
          "[english+code+chinese+base64; RATEC_SCALE_DATA overrides]")
    return pool, probe, "shipped_fixtures"

CORPUS, PROBE, CORPUS_KIND = load_corpus()

TRAIN_POOL = CORPUS

SCALES = [32_000, 64_000, 128_000, 256_000, 512_000, 1_048_576,
          2_097_152, 4_194_304, 8_388_608]
SCALES = [s for s in SCALES if s <= len(TRAIN_POOL)]
if not SCALES:                                 # tiny corpus fallback
    SCALES = [max(8_000, len(TRAIN_POOL) // 4)]

# ------------------------------------------------------------------- helpers
def units_per_byte(engine, blob):
    """LM positions per byte: number of patches the engine cuts blob into.
    encode_to_spans works with or without the patch book; the count equals
    len(encode_bytes(blob)) exactly."""
    return len(engine.encode_to_spans(blob)) / len(blob)

def book_stats(engine, probe_ids):
    n = engine.book_size
    if not n:
        return 0, 0.0
    c = Counter(i for i in probe_ids if i is not None)   # escapes carry None
    in_book = [i for i in c if i < n]
    singletons = sum(1 for i in in_book if c[i] == 1)
    return n, (singletons / max(1, len(in_book)))

# ------------------------------------------------------------------- tiktoken
tik = None
try:
    import tiktoken
    tik = tiktoken.get_encoding("r50k_base")
except ImportError:
    print("tiktoken not installed — baseline row omitted (pip install tiktoken)")

# ------------------------------------------------------------------- run
rows = []

cold = ratec.RateC(c=20, store_book=False)
cold_fert = units_per_byte(cold, PROBE)
rows.append({"arm": "ratec cold", "scale": 0, "units_per_byte": cold_fert,
             "book": 0, "singleton_frac": None})
print(f"\nprobe: {len(PROBE)/1024:.0f} KB (fixed tail, disjoint from every "
      f"adaptation prefix)\n")
hdr = f"{'adaptation':>12} {'units/byte':>10} {'vs cold':>8} {'book':>8} {'sing.':>6}"
print(hdr); print("-" * len(hdr))
print(f"{'cold (0 B)':>12} {cold_fert:>10.4f} {'—':>8} {'—':>8} {'—':>6}")

eng = ratec.RateC(c=20, store_book=True)
for s in SCALES:
    eng.encode_to_spans(TRAIN_POOL[:s])       # adaptation continues (deployment)
    ids = eng.encode_to_spans(PROBE)          # then the unseen probe
    fert = len(ids) / len(PROBE)
    book, sing = book_stats(eng, [sp[2] for sp in ids])
    rows.append({"arm": "ratec warm", "scale": s, "units_per_byte": fert,
                 "book": book, "singleton_frac": sing})
    gain = 100.0 * (cold_fert - fert) / cold_fert
    sing_s = "—" if sing is None else f"{sing:.2f}"
    print(f"{s:>12,} {fert:>10.4f} {gain:>+7.1f}% {book:>8,} {sing_s:>6}")

if tik is not None:
    # tiktoken is byte-level: encode(s) tokenizes utf8(s) exactly, so the
    # token count equals the count over the raw probe bytes (all four
    # domains are valid utf-8, same trick as bench_vs_tiktoken.py)
    tfert = len(tik.encode_ordinary(PROBE.decode("utf-8"))) / len(PROBE)
    rows.append({"arm": "tiktoken r50k_base", "scale": None,
                 "units_per_byte": tfert})
    print(f"{'tiktoken r50k':>12} {tfert:>10.4f} "
          f"{100*(cold_fert-tfert)/cold_fert:>+7.1f}% {'fixed':>8}    — "
          "(fertility is train-time-frozen)")

best = max((r for r in rows if r["arm"] == "ratec warm"),
           key=lambda r: r["scale"], default=None)
print()
if best:
    gain = 100.0 * (cold_fert - best["units_per_byte"]) / cold_fert
    print(f"scale effect: cold {cold_fert:.4f} -> warm@{best['scale']:,} "
          f"{best['units_per_byte']:.4f} units/byte ({gain:+.1f}% vs cold)")

out = {"protocol": {
           "c": 20,
           "corpus_kind": CORPUS_KIND,
           "probe_bytes": len(PROBE),
           "probe": "fixed tail of the corpus, disjoint from every prefix",
           "adaptation": "one engine, encode_bytes continues through scales",
           "scales": SCALES,
       },
       "rows": rows}
with open(os.path.join(OUT, "scale.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {os.path.join(OUT, 'scale.json')}")
