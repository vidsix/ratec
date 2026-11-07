"""Pytest wrapper: run each standalone check script as a subprocess and fail
if its exit code is nonzero. The scripts print their own detailed PASS/FAIL
lines; this file turns them into collectable pytest items.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["test_ratec.py", "test_v04.py", "test_profiles.py",
           "test_smollm2_port.py", "gate_bitexact.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    assert r.returncode == 0, f"{script} failed (exit {r.returncode})"
