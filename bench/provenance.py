#!/usr/bin/env python3
"""The 1.0 provenance record (the 1.0rc gate: "every table in the
record reproducible from committed artifacts").

Hashes EVERY artifact the 1.0 contract rests on — the seven LEAN proof
files, the quiet-box envelope and its floor rule, the 8 GiB scale gate,
the soak records, the parity/decision records, the API freeze manifest,
the fuzz batteries, the CI workflows, the snapshot-format constants —
into research/provenance_1.0.json with, for each, the committed path,
its sha256, and HOW it is reproduced (the gate that regenerates or
re-checks it). The 1.0 freeze audit re-runs this generator and diffs
against the committed record: a changed hash on a contract artifact is
a deliberate, reviewable event (the artifact changed → the contract
changed), never a silent drift.

The record is self-describing: it embeds the generator's own path and
the reproduction command for the record itself.
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from record_util import record_envelope, write_record

OUT = os.path.join(ROOT, "research", "provenance_1.0.json")

# artifact path -> (what it is, how it is reproduced/verified)
ARTIFACTS = {
    "lean/RatecTiling.lean": ("T2 — tiling correctness for ANY surprisal",
                              "elan: lean lean/RatecTiling.lean (no-sorry)"),
    "lean/RatecIntern.lean": ("S5/S6 — parallel intern ids",
                              "elan: lean lean/RatecIntern.lean (no-sorry)"),
    "lean/RatecDecode.lean": ("S7 — decode totality iff",
                              "elan: lean lean/RatecDecode.lean (no-sorry)"),
    "lean/RatecSerial.lean": ("S1 — serial family totality",
                              "elan: lean lean/RatecSerial.lean (no-sorry)"),
    "lean/RatecSnapshot.lean": ("S9 — snapshot v8 totality/rejection",
                                "elan: lean lean/RatecSnapshot.lean (no-sorry)"),
    "lean/RateC.lean": ("the C spec scaffold",
                        "elan: lean lean/RateC.lean (no-sorry)"),
    "lean/RatecServo.lean": ("S8 — solve_c contract (v0.9.2)",
                             "elan: lean lean/RatecServo.lean (no-sorry)"),
    "research/quietbox_envelope.json": (
        "P1.1 — the quiet-box envelope (the 1.0 perf reference)",
        "release procedure: full quiet-box re-record"),
    "research/perf_floors.json": (
        "the executable floor check vs the envelope (tiered floors)",
        "python bench/perf_floors.py"),
    "research/gate_8gib.json": (
        "the scale gate — 8 GiB prep byte-identical via chunked_prep",
        "python bench/gate_8gib.py (32 x 256 MiB engines)"),
    "research/fastlane_parity.json": (
        "the fast-lane surprisal-deviation bound (5e-07)",
        "the parity bench (research/ablate_lenfield.py)"),
    "research/fastlane_decision.json": (
        "the v0.9.2 fast-lane KEEP decision, measured",
        "python bench/fastlane_decision.py"),
    "research/perorder_tables.json": (
        "the v0.9.2 per-order-tables DECLINED record",
        "python bench/perorder_gate.py"),
    "research/api_manifest.json": (
        "the frozen public API surface (the freeze audit's manifest)",
        "python bench/api_freeze.py + tests/test_api_freeze.py"),
    "research/soak_2026-11-23T0941.json": (
        "the first nightly-soak record (198 rounds flat-RSS PASS)",
        "python bench/soak_nightly.py --minutes 60 (nightly)"),
    "tests/test_fuzz.py": ("the property fuzz battery",
                           "python tests/test_fuzz.py"),
    "tests/test_snapfuzz.py": ("the snapshot mutant battery",
                               "python tests/test_snapfuzz.py"),
    ".github/workflows/ci.yml": ("CI: suite + bit-exactness + tables",
                                 "GitHub Actions on push"),
    ".github/workflows/nightly.yml": ("CI: 60-min soak + fresh fuzz nightly",
                                      "GitHub Actions nightly cron"),
    ".github/workflows/ci.yml": (
        "CI: suite + bit-exactness + tables + LEAN compile-all + aarch64",
        "GitHub Actions on push"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    entries = {}
    missing = []
    for rel, (what, repro) in ARTIFACTS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        entries[rel] = {
            "sha256": sha256(path), "bytes": os.path.getsize(path),
            "what": what, "reproduce": repro,
        }
    lean = open(os.path.join(ROOT, "lean", "lean-toolchain")).read()
    record = {
        "record": record_envelope(inputs={}),
        "git_rev": git_rev(),
        "lean_toolchain": lean.strip(),
        "generator": "bench/provenance.py",
        "reproduce_record": "python bench/provenance.py  # re-run and diff "
                            "against research/provenance_1.0.json — any "
                            "changed hash is a contract-artifact change",
        "artifacts": entries,
        "missing": missing,
        "count": len(entries),
    }
    write_record(OUT, record)
    print(f"artifacts hashed: {len(entries)}"
          + (f" | MISSING: {missing}" if missing else ""))
    for rel, e in entries.items():
        print(f"  {rel}: {e['sha256'][:12]}  {e['what']}")
    print("lean toolchain:", record["lean_toolchain"])


if __name__ == "__main__":
    main()
