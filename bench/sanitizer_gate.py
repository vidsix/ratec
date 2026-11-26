#!/usr/bin/env python3
"""sanitizer_gate.py — the memory-safety proof gate (criterion-8 upgrade).

Replaces black-box RSS soaking with per-path proof on the REAL C
engine, compiled with AddressSanitizer + UndefinedBehaviorSanitizer:

  A. the engine is rebuilt with -fsanitize=address,undefined
  B. the FULL pytest suite runs under the sanitizers (fuzz gates,
     golden bit-exact gates, clone/restore, error taxonomy included)
  C. the integration proofs (tokenizers bridge, GGUF official reader,
     snapshot continuation) run under the sanitizers
  D. the torture workload (train/encode/decode/batch/save/load/
     tokenizer.json/gguf/chunked_prep) runs at cycles 1 / 30 / 60;
     LeakSanitizer byte totals are compared across scales — a REAL
     leak scales with the cycle count, the CPython baseline does not
  E. a mutant blitz (state / gguf / tokenizer.json) runs under the
     sanitizers — loaders must never crash
  F. gcc -fanalyzer static pass over the engine source (report-only:
     the analyzer is false-positive-prone, so it informs, it does not
     gate)
  G. the normal (un-instrumented) build is restored

GO iff: zero ASan errors, zero UBSan runtime errors, non-scaling
leaks, suite + proofs + blitz green. This is the evidence behind the
proposed STABILITY.md criterion-8 amendment: proven leak-free and
UB-free per executed path, with the nightly kept as a free tripwire.

usage:
    python3 bench/sanitizer_gate.py            # verdict table
    python3 bench/sanitizer_gate.py --json OUT # machine record
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path("/tmp/asan_gate")


def _sh(cmd, env=None, timeout=7200):
    return subprocess.run(cmd, cwd=ROOT, shell=True, text=True,
                          capture_output=True, timeout=timeout, env=env)


def _san_env(tag: str):
    env = dict(os.environ)
    libasan = subprocess.run(["gcc", "-print-file-name=libasan.so"],
                             capture_output=True, text=True).stdout.strip()
    libubsan = subprocess.run(["gcc", "-print-file-name=libubsan.so"],
                              capture_output=True, text=True).stdout.strip()
    WORK.mkdir(exist_ok=True)
    env["LD_PRELOAD"] = f"{libasan}:{libubsan}"
    env["ASAN_OPTIONS"] = ("detect_leaks=1:log_path=" + str(WORK / f"{tag}")
                           + ":halt_on_error=0:abort_on_error=0"
                           ":exitcode=3:detect_stack_use_after_return=1")
    env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=0"
    return env


def _reports(tag: str):
    """(asan_error_lines, ubsan_error_lines, leak_bytes) across the tag's logs."""
    asan, ubsan, leaks = [], [], 0
    for f in glob.glob(str(WORK / f"{tag}.*")) + glob.glob(str(WORK / f"{tag}")):
        txt = Path(f).read_text(errors="replace")
        for ln in txt.splitlines():
            if ("ERROR: AddressSanitizer" in ln
                    and "leaked" not in ln and "SUMMARY" not in ln):
                asan.append(ln)
            if "runtime error:" in ln:
                ubsan.append(ln)
        m = re.search(r"SUMMARY: AddressSanitizer: (\d+) byte\(s\) leaked",
                      txt)
        if m:
            leaks += int(m.group(1))
    return asan, ubsan, leaks


def _leak_scale_ok(l1, l30, l60):
    """Real leaks scale with cycles; interpreter baselines stay flat."""
    slack = max(4096, (l1 or 0) // 4)
    return (l60 - l1) <= slack, slack


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="OUT")
    ap.add_argument("--torture-cycles", type=int, nargs=3,
                    default=[1, 30, 60])
    args = ap.parse_args(argv)
    results = []

    def rec(name, ok, ev):
        results.append({"name": name, "pass": bool(ok), "evidence": ev})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {ev}")

    # A. instrumented build
    env = os.environ.copy()
    env["CFLAGS"] = "-fsanitize=address,undefined -fno-omit-frame-pointer -g"
    env["LDFLAGS"] = "-fsanitize=address,undefined"
    r = _sh("touch ratec/_rcengine.c && python3 setup.py build_ext "
            "--inplace --force", env=env)
    linked = _sh("ldd ratec/_rcengine*.so | grep -c -i 'asan\\|ubsan'")
    okA = r.returncode == 0 and int(linked.stdout.strip() or 0) >= 1
    rec("A instrumented build", okA,
        f"build exit {r.returncode}; asan/ubsan linked: "
        f"{linked.stdout.strip()} libs")

    # B. full suite under sanitizers
    r = _sh(f"{sys.executable} -m pytest tests/ -q --no-header -x",
            env=_san_env("suite"), timeout=7200)
    a, u, lk = _reports("suite")
    rec("B full pytest under ASan/UBSan", r.returncode == 0 and not a and not u,
        f"pytest exit {r.returncode}; asan_errors={len(a)} "
        f"ubsan_errors={len(u)}; tail "
        f"{(r.stdout.strip().splitlines() or [''])[-1][:60]}")

    # C. integration proofs under sanitizers
    r = _sh(f"{sys.executable} bench/integration_proofs.py",
            env=_san_env("proof"))
    a, u, lk = _reports("proof")
    verdict = "PASS" if "VERDICT: PASS" in (r.stdout + r.stderr) else "NOT-PASS"
    rec("C integration proofs under ASan/UBSan",
        verdict == "PASS" and not a and not u,
        f"verdict {verdict}; asan_errors={len(a)} ubsan_errors={len(u)}")

    # D. differential-leak torture at three scales
    cyc = args.torture_cycles
    leaks = {}
    rss = {}
    for n in cyc:
        r = _sh(f"{sys.executable} bench/asan_torture.py {n}",
                env=_san_env(f"tort{n}"), timeout=7200)
        a, u, lk = _reports(f"tort{n}")
        leaks[n] = lk
        m = re.search(r"rss_peak=(\d+)MiB rss_tail_min=(\d+)MiB "
                      r"rss_tail_max=(\d+)MiB", r.stdout)
        rss[n] = m.groups() if m else ("?", "?", "?")
        if a or u or r.returncode != 0:
            rec(f"D torture cycles={n}", False,
                f"exit {r.returncode}; asan={len(a)} ubsan={len(u)}")
            leaks_ok = False
            break
    else:
        leaks_ok, slack = _leak_scale_ok(*leads(leaks, cyc))
        rec("D differential-leak torture",
            leaks_ok,
            f"leaked bytes at cycles {cyc[0]}/{cyc[1]}/{cyc[2]}: "
            f"{leaks[cyc[0]]}/{leaks[cyc[1]]}/{leaks[cyc[2]]} "
            f"(slack {slack}); rss peak/tail per scale: {rss}")

    # E. mutant blitz under sanitizers
    r = _sh(f"{sys.executable} bench/asan_fuzzblitz.py 15000",
            env=_san_env("fuzz"), timeout=7200)
    a, u, lk = _reports("fuzz")
    rec("E mutant blitz (15k) under ASan/UBSan",
        r.returncode == 0 and not a and not u,
        f"exit {r.returncode}; asan={len(a)} ubsan={len(u)}; "
        f"{(r.stdout.strip().splitlines() or [''])[-1][:60]}")

    # F. gcc -fanalyzer static pass (report-only)
    inc = subprocess.run([sys.executable, "-c",
                          "import sysconfig; print(sysconfig.get_paths()['include'])"],
                         capture_output=True, text=True).stdout.strip()
    r = _sh(f"gcc -fanalyzer -c ratec/_rcengine.c -I{inc} "
            f"-DRC_VERSION='\"sanitizer_gate\"' -o /tmp/fan.o",
            timeout=1800)
    warn = len(re.findall(r"\[-Wanalyzer-", r.stderr))
    rec("F gcc -fanalyzer static pass (report-only)", True,
        f"exit {r.returncode}; {warn} analyzer warnings (informative, "
        f"not gating)")

    # G. restore the normal build
    _sh("touch ratec/_rcengine.c && make build")

    go = all(x["pass"] for x in results)
    print(f"\nVERDICT: {'SAN-GO' if go else 'SAN-NO-GO'} — memory safety "
          f"{'proven on every executed path' if go else 'NOT proven; fix above'}")
    if args.json:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        src = ROOT / "ratec" / "_rcengine.c"
        src_sha = (hashlib.sha256(src.read_bytes()).hexdigest()
                   if src.exists() else None)
        Path(args.json).write_text(json.dumps(
            {"go": go, "head": head, "engine_src_sha256": src_sha,
             "criteria": results}, indent=2))
        print(f"record written: {args.json} "
              f"(head {head[:7]}, engine-src sha {str(src_sha)[:12]}…)")
    return 0 if go else 1


def leads(leaks, cyc):
    return leaks[cyc[0]], leaks[cyc[1]], leaks[cyc[2]]


if __name__ == "__main__":
    sys.exit(main())
