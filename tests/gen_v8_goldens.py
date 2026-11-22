#!/usr/bin/env python3
"""Regenerate the v8 snapshot goldens (tests/data/v8_*.bin + *_cont_golden.json).

Same discipline as gen_v6_goldens.py — everything is derived from
sha256-bound shipped fixtures, so the goldens are reproducible in-repo:

  warm = tests/data/gate_slice.bin     (160k mixed corpus)
  cont = tests/data/v4_cont_slice.bin  (36k continuation slice)

v8 = the v7 layout explicitly little-endian, prune presence carried by a
header flag (PLAN_v0.9 P2.2). v0.9.1 writes ONLY v8, so this script is the
format's golden capture. Run from the repo root after `make build`:

  python tests/gen_v8_goldens.py

The script verifies save->load->continue == warm-engine continue and
save(load(snap)) == snap byte-identity before writing anything, then
updates fixtures_meta.json. The v2..v7 goldens stay frozen: regenerating
them requires a pre-v0.9.1 checkout (their writer is retired on purpose).
"""
import hashlib
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ratec  # noqa: E402

TD = os.path.join(HERE, "data")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    warm = open(os.path.join(TD, "gate_slice.bin"), "rb").read()
    cont = open(os.path.join(TD, "v4_cont_slice.bin"), "rb").read()
    meta_path = os.path.join(TD, "fixtures_meta.json")
    meta = json.load(open(meta_path))
    assert len(warm) == meta["gate_slice_len"] and sha(warm) == meta["gate_slice_sha256"]
    assert len(cont) == meta["v4_cont_len"] and sha(cont) == meta["v4_cont_sha256"]

    configs = [
        ("v8_c20", dict(c=20, store_book=True)),
        ("v8_c33", dict(c=33, store_book=False)),
        ("v8_c20carry", dict(c=20, store_book=True, carry=True)),
    ]
    for name, kw in configs:
        eng = ratec.RateC(**kw)
        if kw["store_book"]:
            eng.encode_bytes(warm)
        else:
            eng.encode_to_spans(warm)          # stream mode: no book
        container = eng.save_state()           # RCZ1 zlib container
        # snapshot header sanity on the raw payload: magic "RCST" + version 8
        assert container[:4] == b"RCZ1", name
        snap = zlib.decompress(container[4:])
        assert snap[:4] == b"RCST", name
        ver = int.from_bytes(snap[4:8], "little")
        assert ver == 8, f"{name}: snapshot version {ver} != 8"
        # loaded engine continues bit-identically to the warm engine
        loaded = ratec.RateC.from_state(snap)
        raw2 = loaded.save_state()
        if raw2[:4] == b"RCZ1":
            raw2 = zlib.decompress(raw2[4:])
        assert raw2 == snap, f"{name}: save(load(snap)) != snap"
        got = [int(s) for s, _, _ in loaded.encode_to_spans(cont)]
        want = [int(s) for s, _, _ in eng.encode_to_spans(cont)]
        assert got == want, f"{name}: continuation mismatch"

        gold = {
            "warm_sha256": sha(warm),
            "cont_sha256": sha(cont),
            "warm_len": len(warm),
            "cont_len": len(cont),
            "ratec_version_at_capture": ratec.__version__,
            "c": kw["c"],
            "carry": bool(kw.get("carry", False)),
            "store_book": bool(kw["store_book"]),
            # encode spans are absolute stream offsets (nseen = warm_len)
            "starts_absolute": got,
            "n_patches_cont": len(got),
        }
        with open(os.path.join(TD, f"{name}.bin"), "wb") as f:
            f.write(container)
        with open(os.path.join(TD, f"{name}_cont_golden.json"), "w") as f:
            json.dump(gold, f)
        meta[f"{name}_sha256"] = sha(container)
        meta[f"{name}_golden_sha256"] = sha(
            json.dumps(gold, sort_keys=True).encode())
        meta[f"{name}_len"] = len(container)
        print(f"  wrote {name}.bin ({len(container)} B container, "
              f"{len(got)} cont patches, ver={ver})")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    print("v8 goldens regenerated + fixtures_meta.json updated")


if __name__ == "__main__":
    main()
