#!/usr/bin/env python3
"""release_gate.py — the machine-checkable 1.0 freeze audit.

Executes the docs/STABILITY.md release checklist ("The 1.0 freeze
audit", 8 criteria) and prints a GO / NO-GO verdict; exit 0 iff GO.
This is the gate that decides the 1.0.0 push — the criteria are not
remembered, they are re-proven:

  1. api_freeze regen leaves research/api_manifest.json surface-identical
     (rev/timestamp-immune; the frozen surface did not move).
  2. tests/test_api_freeze.py green.
  3. provenance regen leaves every artifact hash identical
     (timestamp/rev-immune comparison; any drift is listed).
  4. bench/perf_floors.py PASS.
  5. the quiet-box envelope is present and is the floor reference.
  6. every lean/*.lean compiles no-sorry on the PINNED toolchain
     (docstring mentions of the word are not active sorries).
  7. make test AND the full pytest suite green.
  8. the memory-safety proof at the tag: bench/sanitizer_record.json is
     committed, says go=true with every criterion green, was produced
     from THIS engine source (sha256 match), and on a commit that is an
     ancestor of (or equal to) HEAD. The nightly soak stays scheduled as
     a free tripwire; its streak is REPORTED, not gating (the criterion-8
     amendment — docs/PROOF_STATUS.md, adopted 2026-11-26 — per-path
     LSan proof replaces the >= 1 green week).

usage:
    python3 bench/release_gate.py                  # verdict table
    python3 bench/release_gate.py --json out.json  # + machine record
    python3 bench/sanitizer_gate.py \
        --json bench/sanitizer_record.json         # produce the evidence

Only the nightly tripwire REPORT touches https://api.github.com
(GITHUB_TOKEN optional; the repo is public). The verdict itself is
local and offline.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "vidsix/ratec"


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          **kw)


def _toolchain_env():
    """Env for the pinned lean binary: ELAN_HOME search + LD_LIBRARY_PATH."""
    pin = (ROOT / "lean" / "lean-toolchain").read_text().strip()  # e.g. leanprover/lean4:v4.36.0
    fs_name = pin.replace("/", "--").replace(":", "---")          # elan's on-disk name
    candidates = []
    if os.environ.get("ELAN_HOME"):
        candidates.append(Path(os.environ["ELAN_HOME"]))
    candidates.append(Path.home() / ".elan")
    candidates.append(ROOT.parent.parent / "tools" / "elan")
    for base in candidates:
        tc = base / "toolchains" / fs_name
        if (tc / "bin" / "lean").exists():
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = str(tc / "lib" / "lean")
            env["ELAN_HOME"] = str(base)
            return str(tc / "bin" / "lean"), env, pin
    return None, None, pin


def _strip_lean_comments(src: str) -> str:
    """Remove /- -/ block and -- line comments so docstring mentions of
    the word 'sorry' do not read as active sorries."""
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("/-", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif src.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _strip_volatile(obj):
    """Drop fields that legitimately change between regens (revision,
    timestamps) — the surface itself must not move."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in ("git_rev", "created_utc")}
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def c1_manifest():
    # the regen needs a buildable engine (the .so is gitignored; conftest
    # rebuilds for pytest, standalone scripts must not assume it)
    _run(["make", "build"])
    before = json.loads((ROOT / "research" / "api_manifest.json").read_text())
    r = _run([sys.executable, "bench/api_freeze.py"])
    after = json.loads((ROOT / "research" / "api_manifest.json").read_text())
    same = _strip_volatile(before) == _strip_volatile(after)
    ok = r.returncode == 0 and same
    m = after
    meta = m.get("python", m)
    ev = (f"surface identical to the committed record (rev/timestamp-immune); "
          f"{len(meta.get('functions', []))} fns / "
          f"{len(meta.get('RateC', {}).get('methods', []))} methods "
          f"@ {m.get('meta', {}).get('__version__', '?')}")
    if r.returncode != 0:
        ev = f"api_freeze failed: {r.stderr.strip().splitlines()[-1:]}"
    elif not same:
        ev = "SURFACE DRIFT: the regenerated manifest differs from the committed record"
    return ok, ev


def c2_freeze_test():
    r = _run([sys.executable, "-m", "pytest", "tests/test_api_freeze.py",
              "-q", "--no-header", "-x"])
    return r.returncode == 0, r.stdout.strip().splitlines()[-1:] or ["no output"]


def c3_provenance():
    rec = ROOT / "research" / "provenance_1.0.json"
    before = json.loads(rec.read_text())
    r = _run([sys.executable, "bench/provenance.py"])
    after = json.loads(rec.read_text())
    ab, aa = before.get("artifacts", {}), after.get("artifacts", {})
    # the research/*.json records are timestamp-stamped BY DESIGN (every
    # regen rewrites created_utc, and upstream regens by other gate steps
    # re-stamp them) — their bytes change legitimately, so they cannot be
    # hash-compared; the freeze-relevant artifacts (engine, lean proofs,
    # bench sources, corpora) are compared strictly.
    VOLATILE = lambda k: k.startswith("research/") and k.endswith(".json")  # noqa: E731
    hb = {k: v.get("sha256") for k, v in ab.items() if not VOLATILE(k)}
    ha = {k: v.get("sha256") for k, v in aa.items() if not VOLATILE(k)}
    drift = sorted({k for k in hb.keys() | ha.keys()
                    if hb.get(k) != ha.get(k)})
    nvol = sum(1 for k in aa if VOLATILE(k))
    ok = r.returncode == 0 and not drift
    ev = (f"{len(ha)} artifacts hash-identical to the committed record "
          f"(rev/timestamp-immune; {nvol} research/*.json records "
          f"excluded — timestamp-stamped by design)")
    if drift:
        ev = f"HASH DRIFT in {len(drift)} artifact(s): {drift[:4]}..."
        ok = False
    return ok, ev


def c4_floors():
    r = _run([sys.executable, "bench/perf_floors.py"])
    tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or []
    return r.returncode == 0, tail


def c5_envelope():
    p = ROOT / "research" / "quietbox_envelope.json"
    if not p.exists():
        return False, "research/quietbox_envelope.json missing"
    env = json.loads(p.read_text())
    lanes = env.get("record", env)
    n = len(lanes) if isinstance(lanes, dict) else 0
    ok = n >= 4 and bool(env.get("meta") or env.get("record"))
    return ok, f"envelope present, {n} lane entries — the floor reference"


def c6_lean():
    lean, env, pin = _toolchain_env()
    if lean is None:
        return False, (f"pinned toolchain {pin} not found — reinstall: "
                       f"elan toolchain install {pin}")
    files = sorted((ROOT / "lean").glob("*.lean"))
    fails, active = [], []
    for f in files:
        r = _run([lean, str(f.relative_to(ROOT))], env=env)
        if r.returncode != 0:
            fails.append(f.name)
        body = _strip_lean_comments(f.read_text())
        if re.search(r"\bsorry\b", body):
            active.append(f.name)
    ok = not fails and not active
    ev = f"{len(files)} files compile no-sorry on {pin}"
    if fails:
        ev = f"COMPILE FAIL: {fails}"
    elif active:
        ev = f"ACTIVE SORRY: {active}"
    return ok, ev


def c7_suite():
    mk = _run(["make", "test"])
    pt = _run([sys.executable, "-m", "pytest", "tests/", "-q",
               "--no-header"])
    m = re.search(r"(\d+) passed", pt.stdout)
    counts = f"{m.group(1)} tests passed" if m else "pytest output unparsed"
    ok = mk.returncode == 0 and pt.returncode == 0
    ev = f"make test exit {mk.returncode}; pytest exit {pt.returncode} ({counts})"
    return ok, ev


def c8_nightly(min_n):
    # the list endpoint filtered client-side (the named-workflow runs
    # endpoint 404s on the short name; the list form is stable)
    url = (f"https://api.github.com/repos/{REPO}/actions/runs"
           f"?per_page=100")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
           if os.environ.get("GITHUB_TOKEN") else {})})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:                                   # noqa: BLE001
        return False, f"Actions API unreachable: {e}", {"streak": None}
    runs = sorted((r for r in data.get("workflow_runs", [])
                   if r.get("event") == "schedule"
                   and r.get("status") == "completed"
                   and r.get("name", "").lower() == "nightly"
                   and r.get("head_branch") == "main"),
                  key=lambda r: r["created_at"], reverse=True)
    if not runs:
        return False, "no completed scheduled nightlies on main", {
            "streak": 0, "min": min_n}
    streak = 0
    for r in runs:                    # green runs since the last red run
        if r["conclusion"] == "success":
            streak += 1
        else:
            break
    days = [(_dt.datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")))
            for r in runs[:streak]]
    gaps = [(days[i] - days[i + 1]).days for i in range(len(days) - 1)]
    cadence_ok = all(0 <= g <= 2 for g in gaps)   # daily cron, slack for delays
    latest = days[0].date() if days else None
    projected = (str(latest + _dt.timedelta(days=min_n - streak))
                 if latest and streak < min_n else "—")
    ok = streak >= min_n and cadence_ok
    last_seen = ", ".join(f"{r['created_at'][:10]}:{r['conclusion'][:4]}"
                          for r in runs[:min(streak + 2, 8)])
    ev = (f"streak {streak}/{min_n} green scheduled nightlies on main "
          f"(cadence {'ok' if cadence_ok else 'BROKEN'}: gaps {gaps}); "
          f"latest first: {last_seen}; "
          + (f"week completes earliest {projected} if green continues"
             if streak < min_n else "requirement met"))
    return ok, ev, {"streak": streak, "min": min_n, "cadence_ok": cadence_ok,
                    "projected": projected,
                    "runs": [{"date": r["created_at"][:10],
                              "conclusion": r["conclusion"]}
                             for r in runs[:streak]]}


def c8_memory_safety(record="bench/sanitizer_record.json"):
    # the criterion-8 amendment: per-path proof at the tag, not weekly
    # sampling. The record must be green, hash-match THIS engine source,
    # and descend from the recorded commit's line (evidence freshness).
    rec_path = ROOT / record
    if not rec_path.exists():
        return False, (f"{record} missing — produce it at this commit: "
                       f"python3 bench/sanitizer_gate.py --json {record}"), {}
    rec = json.loads(rec_path.read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    src = ROOT / "ratec" / "_rcengine.c"
    src_sha = (hashlib.sha256(src.read_bytes()).hexdigest()
               if src.exists() else None)
    crit = rec.get("criteria", [])
    go_rec = rec.get("go") is True and bool(crit) and all(
        c.get("pass") for c in crit)
    sha_ok = src_sha is not None and rec.get("engine_src_sha256") == src_sha
    rh = rec.get("head")
    anc = bool(rh) and (rh == head or subprocess.run(
        ["git", "merge-base", "--is-ancestor", rh, head],
        cwd=ROOT, capture_output=True).returncode == 0)
    ok = go_rec and sha_ok and anc
    ev = (f"record go={rec.get('go')}, criteria "
          f"{sum(1 for c in crit if c.get('pass'))}/{len(crit)}; "
          f"engine-src sha {'match' if sha_ok else 'MISMATCH'}; "
          f"record head {str(rh)[:7]} is{' an' if anc and rh != head else ''}"
          f"{' equal to' if rh == head else ''} ancestor of HEAD "
          f"{head[:7]}: {anc}")
    return ok, ev, {"record_head": rh, "head": head}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default="bench/sanitizer_record.json")
    ap.add_argument("--json", metavar="OUT")
    args = ap.parse_args(argv)

    results = [("1 manifest immutability", *c1_manifest()),
               ("2 freeze test", *c2_freeze_test()),
               ("3 provenance hashes", *c3_provenance()),
               ("4 perf floors", *c4_floors()),
               ("5 quiet-box envelope", *c5_envelope()),
               ("6 lean no-sorry (pinned)", *c6_lean()),
               ("7 make test + full pytest", *c7_suite())]
    ok8, ev8, meta8 = c8_memory_safety(args.record)
    results.append(("8 memory-safety proof at the tag (SAN-GO)", ok8, [ev8]))
    tw_ok, tw_ev, tw_meta = c8_nightly(7)      # tripwire REPORT only

    go = all(ok for _, ok, _ in results)
    print(f"release gate @ {ROOT.name} — the docs/STABILITY.md 1.0 checklist")
    for name, ok, ev in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
              + "; ".join(str(x) for x in ev))
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'}"
          + ("" if go else " — resolve the FAIL rows, then re-run"))
    print(f"tripwire (report-only, not gating): nightly streak "
          f"{tw_meta.get('streak')} green — {tw_ev}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"go": go, "nightly_tripwire": tw_meta,
             "criteria": [{"name": n, "pass": ok, "evidence": [str(x) for x in ev]}
                          for n, ok, ev in results]}, indent=2))
        print(f"record written: {args.json}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
