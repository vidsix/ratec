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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="ratec,bytes")
    args = ap.parse_args()
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
