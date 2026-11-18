#!/usr/bin/env python3
"""LM-optimal C sweep runner (PLAN_v0.7 P3): ratec_direct at C in {8,10,12},
2400 steps each, on single-engine dense-id streams built by
build_streams.py --c {C} (sha-bound in runs/streams_manifest_c{C}.json).

The library's own C sweep (research/bench_c_sweep.json) optimizes the
TOKENIZER rate; this one asks the downstream question: at probe scale,
which C best feeds a fixed tiny LM under the factorized interface
(256+2^C content rows, the shipped architecture's budget)?

Pre-registered interpretations (set before any run — the record cannot be
curve-fit post hoc):
  - interior argmin over {8,10,12} -> that C is the LM-optimal budget at
    this scale, stated with its BPB;
  - monotone decrease to the C=12 edge -> the optimum sits above the
    swept range; the doc says so and marks the direction, no extrapolation;
  - params grow with C (head rows 256+9*2^C) — reported per run as an
    explicit confound: the sweep compares budgets, not parameter-matched
    models.

Resumable: a C with an existing valid record is skipped; interrupted runs
resume from their checkpoints inside twin_study.py. The record
research/lm_c_sweep.json is rewritten after every completed run.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from research.record_util import record_envelope, write_record  # noqa: E402

RUNS = os.environ.get("RATEC_RUNS", os.path.join(ROOT, "runs"))
# RATEC_SWEEP env override (v0.8 extension: RATEC_SWEEP=14 chases the
# pre-registered "optimum lies above the swept range" direction; the
# extension's own interpretation is registered below, before any run)
SWEEP = [int(x) for x in
         os.environ.get("RATEC_SWEEP", "8,10,12").split(",") if x.strip()]
STEPS = 2400
RECORD = os.path.join(ROOT, "research", "lm_c_sweep.json")
PRE_REGISTERED = [
    "interior argmin over {8,10,12} -> LM-optimal C at probe scale, "
    "stated with its BPB;",
    "monotone decrease to the C=12 edge -> the optimum lies above the "
    "swept range; stated as a direction, never extrapolated;",
    "params grow with C (head rows 256+9*2^C): reported per run as an "
    "explicit confound. Written before any sweep run (PLAN_v0.7.md P3).",
]


def run_json(c):
    return os.path.join(RUNS, f"twin_ratec_direct_csweep_c{c}.json")


def load_record():
    if os.path.exists(RECORD):
        with open(RECORD) as f:
            return json.load(f)
    return {
        "protocol": {
            "what": "LM-optimal C sweep: ratec_direct at C in {8,10,12}, "
                    "2400 steps, probe protocol otherwise frozen "
                    "(PLAN_v0.7 P3)",
            "arm": "ratec_direct (tied embedding/head over 256+9*2^C rows)",
            "streams": "single-engine streams in the SHIPPED PACKED id "
                       "layout (build_streams.py --c C; sha-bound in "
                       "runs/streams_manifest_c{C}.json) — every arm "
                       "shares the fixed 256+9*2^C-row table, so the "
                       "comparison is parameter-EXACT across C (the "
                       "producer's modulo collapse of book ids into 2^C "
                       "content slots IS the budget semantics)",
            "frozen": {"d": 96, "layers": 3, "heads": 4, "ctx": 256,
                       "batch": 16, "lr": 3e-4, "seed": 42,
                       "steps": STEPS,
                       "train_bytes": 15_811_609, "val_bytes": 659_473},
            "interpretations_pre_registered": PRE_REGISTERED,
        },
        "extension_v0_8": {
            "what": "C=14 arm added via RATEC_SWEEP=14 — chases the "
                    "C=12-edge direction without touching the v0.7 "
                    "protocol or its recorded rows",
            "pre_registered_utc": None,   # filled on first extension run
            "interpretations_pre_registered": [
                "C=14 below C=12's 3.3804 -> interior optimum inside "
                "{12,14}; stated with its C and BPB;",
                "C=14 at/above 3.3804 -> the direction continues; stated "
                "as a direction, never extrapolated;",
                "the param-count confound grows again (256+9*2^14 rows) "
                "and is reported per run.",
            ],
        },
        "rows": {},
        "status": {"runs_complete": 0, "runs_total": len(SWEEP)},
    }


def rewrite(rec):
    rec["status"]["updated_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["protocol"].pop("envelope", None)
    rec["protocol"].update(record_envelope(inputs={
        f"streams_manifest_c{c}": os.path.join(RUNS,
                                               f"streams_manifest_c{c}.json")
        for c in SWEEP
    }, deps=("numpy", "torch")))
    write_record(RECORD, rec)


def main():
    rec = load_record()
    rec.setdefault("status", {})["runs_total"] = max(
        int(rec["status"].get("runs_total", 0)), len(SWEEP))
    if not rec.get("extension_v0_8"):
        rec["extension_v0_8"] = {
            "what": "C=14 arm added via RATEC_SWEEP=14",
            "pre_registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
            "interpretations_pre_registered": [
                "C=14 below C=12's 3.3804 -> interior optimum inside "
                "{12,14}; stated with its C and BPB;",
                "C=14 at/above 3.3804 -> the direction continues; stated "
                "as a direction, never extrapolated;",
                "the param-count confound grows again (256+9*2^14 rows) "
                "and is reported per run.",
            ],
        }
    print(f"[c_sweep] runs complete: {rec['status']['runs_complete']}"
          f"/{rec['status']['runs_total']}", flush=True)
    for c in SWEEP:
        out = run_json(c)
        key = str(c)
        if key in rec["rows"]:
            print(f"[c_sweep] skip C={c} (recorded)", flush=True)
            continue
        if not os.path.exists(out):
            cmd = [sys.executable, os.path.join(HERE, "twin_study.py"),
                   "--kind", "ratec_direct", "--c", str(c),
                   "--prefix", f"ratec_c{c}",
                   "--steps", str(STEPS), "--tag", f"_csweep_c{c}"]
            print(f"[c_sweep] RUN {' '.join(cmd)}", flush=True)
            p = subprocess.run(cmd, env=os.environ.copy())
            if p.returncode != 0:
                print(f"[c_sweep] C={c} exited rc={p.returncode} "
                      f"(checkpoint kept; resume on next invocation)",
                      flush=True)
                rewrite(rec)
                sys.exit(2)
        with open(out) as f:
            r = json.load(f)
        assert r["steps"] == STEPS and r["c_eff"] == c
        man = json.load(open(os.path.join(RUNS,
                                          f"streams_manifest_c{c}.json")))
        rec["rows"][key] = {
            "final_bpb_overall": r["final_bpb_overall"],
            "final_bpb_per_domain": r["final_bpb_per_domain"],
            "params": r["params"], "v_eff": r["v_eff"],
            "train_tokens": r["train_tokens"],
            "bytes_per_token": r["train_bytes"] / r["train_tokens"],
            "book_size_engine": man["engine_stats"]["book_size_final"],
            "wall_sec": r.get("wall_sec"), "source": "run"}
        rec["status"]["runs_complete"] += 1
        rewrite(rec)
        best = min((v["final_bpb_overall"], k) for k, v in
                   rec["rows"].items())
        print(f"[c_sweep] C={c} done: {r['final_bpb_overall']:.4f} BPB "
              f"(params {r['params']:,}); best so far: C={best[1]} "
              f"@ {best[0]:.4f}", flush=True)
    print("[c_sweep] sweep complete", flush=True)
    rewrite(rec)


if __name__ == "__main__":
    main()
