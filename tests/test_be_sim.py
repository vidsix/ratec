"""The big-endian-host simulation gate (PLAN_v0.9 P2.2).

RATEC_SIMULATE_BIG_ENDIAN swaps the NATIVE scalar reads/writes so a
plain x86 build executes what a big-endian host would do. The locked
contract:

  * a v8 snapshot's BYTES are identical whether they were written by
    the normal or the BE-simulated build (explicit-LE: byte order is
    the format's, not the host's);
  * both builds load the same v8 bytes to the same state;
  * a v2..v7 NATIVE-endian golden (written by an LE host) does NOT
    load on the simulated BE host — the historical limitation v8
    exists to remove, now demonstrated instead of folklore.

The simulated engine builds into a SEPARATE directory and runs in a
fresh interpreter via PYTHONPATH — the installed/normal extension is
never replaced (a mapped .so must not be rewritten under a live
process).
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

import pytest

import ratec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROBE = r'''
import sys, json, zlib, hashlib
sys.path.insert(0, sys.argv[1])
import ratec
data = (b"def encode_stream(data):\n    return [x * 31 for x in data]\n"
        b"\xe4\xbd\xa0\xe5\xa5\xbd " * 4000)
tok = ratec.RateC(c=20)
tok.encode_bytes(data)
raw = tok._eng.save_state()
if raw[:4] == b"RCZ1":
    raw = zlib.decompress(raw[4:])
ids = tok.encode_bytes(b"continuation probe")
back = ratec.RateC.from_state(raw)
ids2 = back.encode_bytes(b"continuation probe")
v6 = open(sys.argv[2], "rb").read()
if v6[:4] == b"RCZ1":
    v6 = zlib.decompress(v6[4:])
out = {"v8_ver": int.from_bytes(raw[4:8], "little"),
       "v8_sha256": hashlib.sha256(raw).hexdigest(),
       "ids": [int(i) for i in ids[:32]],
       "ids2": [int(i) for i in ids2[:32]]}
try:
    ratec.RateC.from_state(v6)
    out["v6_loaded_on_be"] = True
except Exception as e:
    out["v6_loaded_on_be"] = False
    out["v6_err_code"] = getattr(e, "code", None)
print(json.dumps(out))
'''


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="setup.py build flow assumes POSIX")
def test_be_simulation_contract():
    tmp = tempfile.mkdtemp(prefix="ratec_be_sim_")
    try:
        pkg = os.path.join(tmp, "ratec")
        build_out = os.path.join(tmp, "buildlib")
        shutil.copytree(os.path.join(ROOT, "ratec"), pkg,
                        ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        r = subprocess.run(
            [sys.executable, "setup.py", "build_ext",
             "--build-lib", build_out],
            cwd=ROOT, env={**os.environ,
                           "CFLAGS": "-DRATEC_SIMULATE_BIG_ENDIAN"},
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        hits = glob.glob(os.path.join(build_out, "ratec", "_rcengine*.so"))
        assert hits, "sim build produced no extension"
        shutil.copy(hits[0], os.path.join(pkg, os.path.basename(hits[0])))

        v6p = os.path.join(HERE, "data", "v6_c20.bin")
        r2 = subprocess.run([sys.executable, "-c", PROBE, tmp, v6p],
                            env={**os.environ, "PYTHONPATH": tmp},
                            capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr[-2000:]
        res = json.loads(r2.stdout.strip().splitlines()[-1])
        assert res["v8_ver"] == 8

        # the SAME bytes from the NORMAL build: sha must match the sim's
        data = (b"def encode_stream(data):\n    return [x * 31 for x in data]\n"
                b"\xe4\xbd\xa0\xe5\xa5\xbd " * 4000)
        tok = ratec.RateC(c=20)
        tok.encode_bytes(data)
        raw = tok._eng.save_state()
        if raw[:4] == b"RCZ1":
            raw = zlib.decompress(raw[4:])
        assert hashlib.sha256(raw).hexdigest() == res["v8_sha256"], (
            "v8 bytes differ between the normal and BE-simulated builds")
        # both builds load the same v8 bytes to the same state
        back = ratec.RateC.from_state(raw)
        ids2 = back.encode_bytes(b"continuation probe")
        assert [int(i) for i in ids2[:32]] == res["ids2"] == res["ids"]
        # an LE-written native-format (v6) golden does NOT load on the
        # simulated BE host — the historical limitation v8 removes
        assert res["v6_loaded_on_be"] is False
        assert res["v6_err_code"] in ("SNAPSHOT_INVALID", "SNAPSHOT_VERSION")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
