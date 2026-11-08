"""Pytest wrapper: run each standalone check script as a subprocess and fail
if its exit code is nonzero. The scripts print their own detailed PASS/FAIL
lines; this file turns them into collectable pytest items.

Also writes tests/test_manifest.json — the machine-readable suite manifest
(PLAN_v0.6 P3-7): per-script pass/fail + wall time, so a green-SKIPPED run
(e.g. test_smollm2_port without transformers) is distinguishable from a
green-PASSED one.
"""
import datetime
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["test_ratec.py", "test_v04.py", "test_profiles.py",
           "test_smollm2_port.py", "test_snapfuzz.py", "test_tables.py", "test_fuzz.py",
           "gate_bitexact.py"]

MANIFEST = os.path.join(HERE, "test_manifest.json")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    dt = time.monotonic() - t0
    # merge this script's outcome into the manifest (parallel-safe enough:
    # pytest runs scripts sequentially by default)
    entry = {"script": script, "exit": r.returncode, "seconds": round(dt, 2),
             "at": datetime.datetime.now(
                 datetime.timezone.utc).isoformat(timespec="seconds")}
    try:
        man = json.load(open(MANIFEST))
    except Exception:
        man = {"scripts": {}}
    man.setdefault("scripts", {})[script] = entry
    man["updated_utc"] = entry["at"]
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    assert r.returncode == 0, f"{script} failed (exit {r.returncode})"
