#!/usr/bin/env python3
"""Record-convention gate (PLAN_v0.6 P2-2): the README's published numbers
must equal the committed records, and every research record must carry its
provenance envelope.

  1. `research/make_tables.py --check` — README tables (scale, fertility)
     regenerate byte-identically from research/*.json; hand-edited numbers
     inside the sentinels fail here.
  2. Every research/*.json parses and either carries the record envelope
     (ratec_version/deps/git_rev/host/created_utc/inputs) or is explicitly
     marked as a legacy/superseded record — no unattributed numbers.

Run:  python3 tests/test_tables.py
Exit 0 only if every check passes.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = []

ENVELOPE_KEYS = ("ratec_version", "deps", "git_rev", "host", "created_utc",
                 "inputs")
LEGACY_OK = {
    # pre-convention records kept verbatim as historical evidence
    "scale_mix15.8MB_v0.5.0_superseded.json",
    "twin_bpe.json", "twin_bytes.json", "twin_ratec.json",
    "twin_ratec_direct.json", "twin_summary.json", "twin_probe_s2400.json",
}


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    print("record convention + README drift gate")

    r = subprocess.run([sys.executable,
                        os.path.join(ROOT, "research", "make_tables.py"),
                        "--check"], capture_output=True, text=True)
    check("README tables regenerate from records (make tables --check)",
          r.returncode == 0, (r.stdout + r.stderr).strip().splitlines()[-1]
          if (r.stdout + r.stderr).strip() else "")

    rdir = os.path.join(ROOT, "research")
    n_env, n_legacy, bad = 0, 0, []
    for fn in sorted(os.listdir(rdir)):
        if not fn.endswith(".json"):
            continue
        try:
            rec = json.load(open(os.path.join(rdir, fn)))
        except Exception as e:                          # noqa: BLE001
            bad.append(f"{fn}: unparseable ({e})")
            continue
        env = rec.get("record") or rec.get("protocol", {})
        if all(k in env for k in ENVELOPE_KEYS):
            n_env += 1
        elif fn in LEGACY_OK:
            n_legacy += 1
        else:
            bad.append(f"{fn}: no provenance envelope")
    check(f"records carry provenance ({n_env} enveloped, "
          f"{n_legacy} marked legacy)", not bad,
          "; ".join(bad[:3]))

    print()
    print(f"TABLES GATE: {'PASS' if all(RESULTS) else 'FAIL'} "
          f"({sum(RESULTS)}/{len(RESULTS)} checks)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
