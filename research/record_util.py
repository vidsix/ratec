"""Record envelope helper — every committed record JSON carries provenance.

The convention (PLAN_v0.6 P2-1): a published number is only as good as the
record behind it, so every `research/*.json` produced from running code
gains, in addition to its protocol and results:

    ratec_version   float — ratec.__version__ at capture time
    deps            {name: version} — measurement-relevant imports
    git_rev         str — `git rev-parse --short HEAD` ("" if unavailable)
    host            {python, machine, cpu_count} — capture box summary
    created_utc     ISO-8601 UTC timestamp
    inputs          {label: sha256} — every byte the run consumed

Usage (three lines inside any producer script):

    from research.record_util import record_envelope, sha256_file
    rec = {"protocol": {...}, "rows": [...]}
    rec.update(record_envelope(inputs={"corpus": "data/_train_mix.txt"}))
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys


def sha256_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    """Short HEAD revision, best-effort (records must not fail on tarballs).
    Handles worktrees too (.git is a file pointing at the real gitdir)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        for up in range(4):
            root = os.path.normpath(
                os.path.join(here, *([".."] * up)))
            if os.path.exists(os.path.join(root, ".git")):
                out = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=root,
                    capture_output=True, text=True, timeout=10)
                if out.returncode == 0:
                    return out.stdout.strip()
                break
    except Exception:
        pass
    return ""


def host_info() -> dict:
    info = {"python": sys.version.split()[0], "machine": platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


def _dep_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def record_envelope(inputs: dict = None,
                    deps: tuple = ("numpy", "tiktoken", "torch",
                                   "transformers")) -> dict:
    """Return the provenance block. `inputs` maps label -> path (relative
    paths resolve against the repo root); only existing files are bound."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bound = {}
    for label, path in (inputs or {}).items():
        p = path if os.path.isabs(path) else os.path.join(root, path)
        if os.path.isfile(p):
            bound[label] = {"sha256": sha256_file(p), "path": path,
                            "bytes": os.path.getsize(p)}
        else:                      # missing input: say so, never invent
            bound[label] = {"path": path, "missing": True}
    env = {
        "ratec_version": _dep_version("ratec") or "",
        "deps": {n: v for n in deps if (v := _dep_version(n))},
        "git_rev": git_rev(),
        "host": host_info(),
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "inputs": bound,
    }
    if not env["ratec_version"]:    # ratec not importable where this runs
        try:
            sys.path.insert(0, root)
            import ratec
            env["ratec_version"] = ratec.__version__
        except Exception:
            pass
    return env


def write_record(path: str, record: dict) -> None:
    with open(path, "w") as f:
        json.dump(record, f, indent=1, sort_keys=False)
        f.write("\n")
