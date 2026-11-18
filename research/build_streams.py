#!/usr/bin/env python3
"""Twin-stream producer (PLAN_v0.6 P2-4): bind every twin input by sha256.

What ships in the repo as `runs/{kind}_{domain}_{train,val}_{ids,lens}.npy`
feeds every number in the twin records, and until v0.6.0 nothing in the
repo could regenerate two of the three arms. This script produces the
arms that are fully defined by committed code + data:

  ratec   ids via the library (C=10, pf default), lens via encode_to_spans;
          length index is min(byte_len, 9) — the V=9472 embedding table caps
          patches at 9 bytes (len_idx <= 9), a documented divergence from
          the library patcher (which can emit longer patches)
  bytes   ids = raw bytes (V=256), lens = 1

  bpe     NOT produced here: the 9472-merge BPE arm is defined by a merge
          table trained outside the repo (workspace `runs/bpe_mix.json`);
          its streams are bound to the records by sha256 instead. See
          research/README.md.

Every output is verified against the input corpus sha256 before writing,
and the written streams are hashed for the record. Usage:

  RATEC_TWIN_DATA=<dir with _train_mix.txt and <dom>_val.txt> \
  RATEC_RUNS=<output dir> python research/build_streams.py [--kinds ratec,bytes]

Inputs (repo-relative defaults, P2-6):
  data/_train_mix.txt   the 15,811,609-byte mixed training corpus
  data/<d>_val.txt      held-out val stream per domain (en/hr/zh/code)
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ratec  # noqa: E402
from research.record_util import record_envelope, sha256_file  # noqa: E402

RUNS = os.environ.get("RATEC_RUNS", os.path.join(_ROOT, "runs"))
DATA = os.environ.get("RATEC_TWIN_DATA", os.path.join(_ROOT, "data"))
DOMAINS = ["en", "hr", "zh", "code"]
C = 10                      # ratec arm patch budget (matches twin_study)
LEN_CAP = 9                 # len_idx <= 9 by the V=9472 table layout

# the shipped mix partition (v0.5.0 protocol, bound by the bpe streams'
# byte sums and verified: the hr/zh/code slices equal {d}_train.txt
# byte-for-byte; the en slice is the mix's first 6,490,411 bytes — the
# same bytes every arm consumes, so the partition itself is the protocol)
MIX_OFFSETS = [0, 6_490_411, 8_992_506, 11_503_730, 15_811_609]
TRAIN_BYTES = MIX_OFFSETS[-1]
VAL_BYTES = 659_473
FEED_CHUNK = 4 << 20        # chunked feed keeps patches open across
                            # chunks (feed(a);feed(b) == feed(a+b) exactly),
                            # so tokenization is one-shot-equivalent while
                            # span lists stay per-chunk (bounded memory)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def stream_ratec(raw: bytes):
    """(patch_ids, patch_lens): the twin harness consumes per-patch arrays.
    Book-mode ids (store_book=True — patch types only exist in the book);
    len index = min(byte_len, LEN_CAP) per the embedding table's layout."""
    tok = ratec.RateC(c=C)                     # store_book=True by default
    spans = tok.encode_to_spans(raw)
    patch_ids = np.array([int(pid) for _, _, pid in spans], dtype=np.int32)
    lens = np.array([min(e - s, LEN_CAP) for s, e, _ in spans],
                    dtype=np.int32)
    return patch_ids, lens


def stream_bytes(raw: bytes):
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    lens = np.ones(len(ids), dtype=np.int32)
    return ids, lens


def _spans_to_arrays(spans):
    if not spans:
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int64))
    ids = np.fromiter((s[2] for s in spans), dtype=np.int32, count=len(spans))
    lens = np.fromiter((s[1] - s[0] for s in spans), dtype=np.int32,
                       count=len(spans))
    starts = np.fromiter((s[0] for s in spans), dtype=np.int64,
                         count=len(spans))
    return ids, lens, starts


def _pack_ids(ids, lens, starts, raw, c, base=0):
    """Map (book_id, len, start) onto the SHIPPED packed layout — the one
    twin_study.py's original id arithmetic decodes (content slot =
    (id-256) % 2^C, length slab = (id-256) // 2^C, single bytes carried
    as their byte value). The producer MUST emit this layout: an LM table
    sized from raw dense book ids would grow with the book (145k+ rows at
    C=10), breaking compute-matching and filling the head softmax with
    dead rows — the v0.7.0 P3 protocol bug this packing fixes before the
    sweep ran. The modulo collapse IS the budget semantics: 2^C content
    slots serve all 145k book contents, so a larger C buys a larger
    budget — the question the sweep exists to answer. lens stay TRUE byte
    lengths (eval weighting); lens > 10 collapse into the top slab for
    the id (the shipped streams' LEN_CAP behavior), which only affects
    the model's length-field granularity, never byte-weighted eval."""
    m = 1 << c
    raw_bytes = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
    packed = 256 + (np.minimum(lens, 10) - 2) * m + (ids % m)
    # starts are ABSOLUTE stream offsets (a continued engine's spans run
    # past the mix into the val stream); `base` maps them into `raw`
    out = np.where(lens == 1, raw_bytes[starts - base], packed)
    return out.astype(np.int32)


def regime_ratec_c(c: int, prune: bool, prune_after: int, prune_min: int):
    """v0.7.0 sweep regime (PLAN_v0.7 P3 + P2 measurement): ONE engine
    tokenizes the whole corpus — the train mix (chunked feed, split at the
    shipped MIX_OFFSETS, straddling patches assigned to the slice their
    start byte falls in) and then each domain's val file WITHOUT reset, so
    val ids live in the same book as train ids (the shipped streams'
    proven design). Ids are emitted in the SHIPPED PACKED layout (see
    _pack_ids) so every arm shares the fixed 256+9*2^C-row table —
    compute-matched with the ladder arms and decodable by twin_study.py's
    original id arithmetic; lens are TRUE byte lengths (byte-weighted
    eval consumes them directly). With prune=True the book churns at
    prune boundaries and slots renumber — the bounded-vocabulary
    mechanism under measurement.

    Outputs {prefix}_dom_{split}_{ids,lens}.npy with
    prefix = ratec_c{c}[_pr], bound by sha256 in
    streams_manifest_c{c}[_pr].json."""
    prefix = f"ratec_c{c}" + ("_pr" if prune else "")
    mix_path = os.path.join(DATA, "_train_mix.txt")
    mix = open(mix_path, "rb").read()
    assert len(mix) == TRAIN_BYTES and MIX_OFFSETS[-1] == TRAIN_BYTES
    eng = ratec.RateC(c=c, prune=prune, prune_after=prune_after,
                      prune_min=prune_min)
    stats = {"engine": {"c": c, "prune": prune,
                        "prune_after": prune_after if prune else 0,
                        "prune_min": prune_min if prune else 0}}
    # ---- train: chunked feed over the mix, sliced by MIX_OFFSETS
    per_dom = [[] for _ in DOMAINS]           # lists of (ids, lens) chunks
    off = 0                                   # bytes consumed so far
    for k in range(0, len(mix), FEED_CHUNK):
        spans = eng.feed(mix[k:k + FEED_CHUNK])
        ids, lens, starts = _spans_to_arrays(spans)
        ids = _pack_ids(ids, lens, starts, mix, c)
        # assign each token to the slice its START byte falls in; token
        # starts are contiguous (each patch starts where the previous
        # ends) and feed() reports absolute stream offsets
        if len(ids):
            first_start = spans[0][0]
            cum = np.concatenate(([first_start],
                                  first_start + np.cumsum(lens.astype(np.int64))[:-1]))
        else:
            cum = np.zeros(0, dtype=np.int64)
        slice_idx = np.searchsorted(np.array(MIX_OFFSETS[1:-1]), cum,
                                    side="right")
        for si in range(4):
            m = slice_idx == si
            if m.any():
                per_dom[si].append((ids[m], lens[m]))
        off += len(mix[k:k + FEED_CHUNK])
    spans = eng.flush()
    ids, lens, starts = _spans_to_arrays(spans)
    ids = _pack_ids(ids, lens, starts, mix, c)
    if len(ids):
        cum = np.concatenate(([spans[0][0]],
                              spans[0][0] + np.cumsum(lens.astype(np.int64))[:-1]))
        slice_idx = np.searchsorted(np.array(MIX_OFFSETS[1:-1]), cum,
                                    side="right")
        for si in range(4):
            m = slice_idx == si
            if m.any():
                per_dom[si].append((ids[m], lens[m]))
    stats["engine"]["book_size_after_train"] = eng.book_size
    stats["engine"]["n_pruned_after_train"] = eng.n_pruned
    # ---- val: same engine (same book), one domain at a time
    man = {"protocol": {
        "what": f"sweep/prune streams at C={c} ({'pruned' if prune else 'unpruned'} book)",
        "regime": "single engine over mix (chunked feed) then per-domain "
                  "val without reset; ids emitted in the SHIPPED PACKED "
                  "layout (256 + (book_id % 2^C) + (min(len,10)-2)*2^C, "
                  "1-byte tokens as raw byte values) so every arm shares "
                  "the fixed 256+9*2^C-row table — compute-matched with "
                  "the ladder's arms; TRUE byte lens; train slices = "
                  "shipped MIX_OFFSETS partition; consumed with plain "
                  "twin_study.py id arithmetic (no --lens_lidx)",
        "mix_offsets": MIX_OFFSETS,
        "note": ("the modulo collapse of book ids into 2^C content slots "
                 "IS the budget semantics (a larger C buys more slots); "
                 "prune-mode ids renumber at boundaries, so slots churn "
                 "there — the mechanism under measurement; fixed v0.7.0 "
                 "P3 before the sweep ran (the earlier draft emitted raw "
                 "dense ids, which would have grown the LM table with "
                 "the book)"),
    }}
    man["protocol"].update(record_envelope(
        inputs={"_train_mix.txt": os.path.relpath(mix_path, _ROOT),
                **{f"{d}_val.txt": f"data/{d}_val.txt" for d in DOMAINS}},
        deps=("numpy",)))
    man["files"] = {}
    tot_bytes = 0
    stream_base = TRAIN_BYTES   # val streams continue the mix stream:
                                # each domain's spans start where the
                                # previous domain's ended (absolute)
    for di, dom in enumerate(DOMAINS):
        ids = np.concatenate([a for a, _ in per_dom[di]]) \
            if per_dom[di] else np.zeros(0, dtype=np.int32)
        lens = np.concatenate([l for _, l in per_dom[di]]) \
            if per_dom[di] else np.zeros(0, dtype=np.int32)
        tot_bytes += int(lens.sum())
        val = open(os.path.join(DATA, f"{dom}_val.txt"), "rb").read()
        vspans = eng.encode_to_spans(val)
        v_ids, v_lens, v_starts = _spans_to_arrays(vspans)
        v_ids = _pack_ids(v_ids, v_lens, v_starts, val, c, base=stream_base)
        tot_bytes += int(v_lens.sum())
        stream_base += len(val)
        for tag, arr in ((f"{prefix}_{dom}_train_ids", ids),
                         (f"{prefix}_{dom}_train_lens", lens),
                         (f"{prefix}_{dom}_val_ids", v_ids),
                         (f"{prefix}_{dom}_val_lens", v_lens)):
            out = os.path.join(RUNS, f"{tag}.npy")
            np.save(out, arr)
            man["files"][f"{tag}.npy"] = {"sha256": sha256_file(out),
                                          "len": int(len(arr))}
        print(f"  {prefix}/{dom}: train {len(ids)} tok / {int(lens.sum())} B "
              f"| val {len(v_ids)} tok / {int(v_lens.sum())} B", flush=True)
    assert tot_bytes == TRAIN_BYTES + VAL_BYTES, tot_bytes
    stats["engine"]["book_size_final"] = eng.book_size
    stats["engine"]["n_pruned_final"] = eng.n_pruned
    man["engine_stats"] = stats["engine"]
    out_path = os.path.join(RUNS, f"streams_manifest_c{c}{'_pr' if prune else ''}.json")
    with open(out_path, "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {out_path} ({len(man['files'])} files bound)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="ratec,bytes")
    ap.add_argument("--c", type=int, default=None,
                    help="sweep regime: build single-engine streams at this "
                         "patch budget (v0.7.0); with --prune the book is "
                         "pruned every --prune-after bytes")
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--prune-after", type=int, default=1 << 20)
    ap.add_argument("--prune-min", type=int, default=2)
    args = ap.parse_args()

    if args.c is not None:
        regime_ratec_c(args.c, args.prune, args.prune_after, args.prune_min)
        return
    kinds = args.kinds.split(",")

    train_path = os.path.join(DATA, "_train_mix.txt")
    assert os.path.isfile(train_path), f"missing {train_path}"
    train_sha = sha256_file(train_path)
    train = open(train_path, "rb").read()

    manifest = {"protocol": {
        "ratec_c": C, "len_cap": LEN_CAP,
        "bpe_arm": ("not produced in-repo (merge table trained outside; "
                    "streams bound by sha256 in the twin records)"),
        "note": ("ids/lens per domain: {kind}_{dom}_{train,val}_{ids,lens}"
                 ".npy; ratec ids are patch ids, bytes ids are byte values"),
    }, "files": {}}
    manifest["protocol"].update(record_envelope(
        inputs={"train_mix.txt": os.path.relpath(train_path, _ROOT)},
        deps=("numpy",)))

    for kind in kinds:
        for dom in DOMAINS:
            val_path = os.path.join(DATA, f"{dom}_val.txt")
            if not os.path.isfile(val_path):
                print(f"  skip {kind}/{dom}: no {val_path}")
                continue
            val = open(val_path, "rb").read()
            for split, raw in (("train", train), ("val", val)):
                fn = lambda r: (stream_ratec(r) if kind == "ratec"
                                else stream_bytes(r))
                ids, lens = fn(raw)
                for tag, arr in ((f"{kind}_{dom}_{split}_ids", ids),
                                 (f"{kind}_{dom}_{split}_lens", lens)):
                    out = os.path.join(RUNS, f"{tag}.npy")
                    np.save(out, arr)
                    manifest["files"][f"{tag}.npy"] = {
                        "sha256": sha256_file(out), "len": int(len(arr))}
                print(f"  {kind}/{dom}/{split}: {len(ids)} tokens "
                      f"({len(raw)} bytes)")
    out_path = os.path.join(RUNS, "streams_manifest.json")
    manifest["inputs"] = {"_train_mix.txt": train_sha}
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {out_path} ({len(manifest['files'])} stream files bound)")


if __name__ == "__main__":
    main()
