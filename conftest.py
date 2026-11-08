"""Pytest bootstrap: make `pytest` work from the repo root on a fresh clone.

Running pytest from the repo root shadows any installed ratec with the local
package directory — which contains no built engine extension (it is
gitignored), so the first run fails with a confusing import error. This file
builds the extension in place once, then re-checks.
"""
import importlib
import os
import subprocess
import sys


def _engine_ok():
    try:
        from ratec import _rcengine  # noqa: F401
        return True
    except Exception:
        return False


if not _engine_ok():
    _root = os.path.dirname(os.path.abspath(__file__))
    print("conftest: local engine extension missing — building in place "
          "(needs gcc >= 9) ...", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, os.path.join(_root, "setup.py"),
             "build_ext", "--inplace"], cwd=_root)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "in-place engine build failed; run `make build` and check the "
            "compiler output") from e
    importlib.invalidate_caches()
    if not _engine_ok():
        raise RuntimeError("engine still not importable after build")


# tests/test_*.py and gate_bitexact.py are standalone scripts (they print
# their own PASS/FAIL lines and sys.exit at the end) — pytest must not collect
# them directly. tests/test_suite.py wraps each as a subprocess test instead.
_HERE = os.path.dirname(os.path.abspath(__file__))
collect_ignore = [os.path.join(_HERE, "tests", f) for f in
                  ("test_ratec.py", "test_v04.py", "test_profiles.py",
                   "test_smollm2_port.py", "test_snapfuzz.py",
                   "test_tables.py", "gate_bitexact.py")]
