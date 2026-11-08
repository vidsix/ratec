#!/usr/bin/env python3
"""make records (PLAN_v0.6 P2-3): copy fresh bench outputs into research/.

The bench producers write machine-local outputs to `results/` (unver-
sioned); at release time the final records travel with the code. This
script copies each results/*.json into research/ under its release name,
adds the provenance envelope (record_util), and refuses to overwrite an
existing committed record whose protocol changed silently — a protocol
change deserves a diff in the release notes, not a drive-by rewrite.

  results/scale.json        -> research/scale_mix15.8MB.json   (only when
                               RATEC_SCALE_DATA ran; shipped-fixture smoke
                               runs stay in results/)
  results/bench_tiktoken.json -> research/bench_tiktoken.json
  results/bench_c_sweep.json  -> research/bench_c_sweep.json

Usage: python research/capture_records.py [--force]
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "research"))

from record_util import record_envelope, write_record  # noqa: E402

RESULTS = os.environ.get("RATEC_RESULTS", os.path.join(_ROOT, "results"))

MAP = {
    "scale.json": "scale_mix15.8MB.json",
    "bench_tiktoken.json": "bench_tiktoken.json",
    "bench_c_sweep.json": "bench_c_sweep.json",
}


def main():
    force = "--force" in sys.argv
    if not os.path.isdir(RESULTS):
        print(f"no results/ directory at {RESULTS} — run the benches first")
        return
    copied = skipped = 0
    for src_name, dst_name in MAP.items():
        src = os.path.join(RESULTS, src_name)
        dst = os.path.join(_ROOT, "research", dst_name)
        if not os.path.isfile(src):
            continue
        rec = json.load(open(src))
        if "record" not in rec and "protocol" in rec:
            rec["protocol"].update(record_envelope())
        elif "record" not in rec:
            rec["record"] = record_envelope()
        if os.path.isfile(dst) and not force:
            old = json.load(open(dst))
            old_rows = json.dumps(old.get("rows") or old.get("results") or {},
                                  sort_keys=True)
            new_rows = json.dumps(rec.get("rows") or rec.get("results") or {},
                                  sort_keys=True)
            if old_rows != new_rows:
                print(f"  SKIP {dst_name}: results differ from the committed "
                      f"record (pass --force to overwrite consciously)")
                skipped += 1
                continue
        write_record(dst, rec)
        print(f"  {src_name} -> research/{dst_name}")
        copied += 1
    print(f"capture_records: {copied} copied, {skipped} skipped")


if __name__ == "__main__":
    main()
