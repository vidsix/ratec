#!/usr/bin/env python3
"""Ladder runner: 4 arms x 4 rungs (PLAN_v0.7 P1).

Turns the s2400 two-point probe into a measured trend. Protocol frozen
in docs/PLAN_v0.7.md and repeated in the record itself:

  arms   bpe | ratec_direct | ratec (factorized, shipped) | bytes (floor)
  rungs  steps in {600, 1200, 2400, 4800}, log-spaced x2
  fixed  d=96 layers=3 heads=4 ctx=256 batch=16 lr=3e-4 OneCycle
         recomputed per rung, seed 42, the same 15,811,609-B train /
         659,473-B val bytes, full-coverage byte-weighted BPB

The 600 rung is NOT re-run: the committed research/twin_{arm}.json records
are seeded into the ladder as source "committed_600" (they are the exact
protocol at 600 steps — asserted field by field). Every rung >= 1200 is a
fresh run (OneCycle recomputed per rung; no warm-starting across rungs,
which would confound the trend), executed by shelling out to
research/twin_study.py --tag _l{rung} so ALL training/eval/checkpoint
machinery stays in the one protocol file.

Resumable at two levels:
  - completed runs load their runs/twin_{arm}_l{rung}.json and are skipped;
  - an interrupted run resumes from its checkpoint inside twin_study.py.
The record research/twin_ladder.json is rewritten after EVERY completed
run, so a killed runner never loses more than the run in flight.

Pre-registered interpretations (copied into the record, set before any
rung > 600 was trained — the record cannot be curve-fit post hoc):
  narrow w/ ~constant log-slope -> "more efficient with scale" claim,
    stated with its measured slope; plateau -> the doc says where.

usage:
    python3 research/run_ladder.py                 # run all remaining
    python3 research/run_ladder.py --only 1200     # one rung
    python3 research/run_ladder.py --status        # print, run nothing
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from research.record_util import record_envelope, write_record  # noqa: E402

RUNS = os.environ.get("RATEC_RUNS", os.path.join(ROOT, "runs"))
ARMS = ["bpe", "ratec_direct", "ratec", "bytes"]
RUNGS = [600, 1200, 2400, 4800]
LADDER = os.path.join(ROOT, "research", "twin_ladder.json")
FROZEN = {"d": 96, "layers": 3, "heads": 4, "ctx": 256, "batch": 16,
          "lr": 3e-4, "seed": 42,
          "train_bytes": 15_811_609, "val_bytes": 659_473}
PRE_REGISTERED = [
    "if the tokenization gap (bpe - ratec arm) keeps narrowing with an "
    "approximately constant log-slope, that IS the 'more efficient with "
    "scale' claim, stated with its measured slope and range;",
    "if the gap plateaus, the doc states where it plateaus; either way the "
    "record is the contribution. Written before any rung above 600 was "
    "trained (PLAN_v0.7.md P1).",
]
GAPS = [("bpe", "ratec_direct"), ("bpe", "ratec"), ("bpe", "bytes")]
DOMAINS = ["en", "hr", "zh", "code"]


def run_json(arm, rung):
    return os.path.join(RUNS, f"twin_{arm}_l{rung}.json")


def seed_600():
    """Load the committed 600-step records, asserting protocol identity."""
    out = {}
    for arm in ARMS:
        with open(os.path.join(ROOT, "research", f"twin_{arm}.json")) as f:
            r = json.load(f)
        for k, v in FROZEN.items():
            if k in ("heads", "train_bytes", "val_bytes"):
                continue           # not fields of the per-run record
            assert r.get(k) == v, (arm, k, r.get(k), v)
        assert r["steps"] == 600, (arm, r["steps"])
        if arm != "bytes":
            assert r["train_bytes"] == FROZEN["train_bytes"]
            assert r["val_bytes"] == FROZEN["val_bytes"]
        out[arm] = {"final_bpb_overall": r["final_bpb_overall"],
                    "final_bpb_per_domain": r["final_bpb_per_domain"],
                    "params": r["params"],
                    "wall_sec": r.get("wall_sec"),
                    "source": "committed_600"}
    return out


def load_ladder():
    if os.path.exists(LADDER):
        with open(LADDER) as f:
            return json.load(f)
    rec = {
        "protocol": {
            "what": "LM-side scale ladder: 4 arms x 4 rungs, one frozen "
                    "protocol (PLAN_v0.7 P1)",
            "arms": ARMS,
            "rungs_steps": RUNGS,
            "frozen": FROZEN,
            "rung_600_source": "the committed research/twin_{arm}.json "
                               "records, asserted protocol-identical",
            "rung_>=1200": "fresh runs (OneCycle recomputed per rung; no "
                           "warm-start across rungs) via twin_study.py "
                           "--tag _l{rung}, resumable per run",
            "eval": "full-coverage byte-weighted BPB, identical held-out "
                    "bytes for all arms (twin_study.py eval_full)",
            "interpretations_pre_registered": PRE_REGISTERED,
        },
        "rungs": {},
        "status": {"rungs_complete": [], "runs_complete": 0,
                   "runs_total": len(ARMS) * len(RUNGS)},
    }
    rec["rungs"]["600"] = seed_600()
    return rec


def gaps_and_slopes(rec):
    """Gap table + least-squares slopes over the completed rungs."""
    rungs = rec["rungs"]
    done = sorted(int(r) for r in rungs
                  if all(a in rungs[r] for a in ARMS))
    table = {}
    for r in done:
        e = rungs[str(r)]
        row = {}
        for hi, lo in GAPS:
            row[f"{hi}_minus_{lo}"] = (e[hi]["final_bpb_overall"]
                                       - e[lo]["final_bpb_overall"])
        table[r] = row
    slopes = {}
    if len(done) >= 2:
        xs = [math.log2(r) for r in done]
        for hi, lo in GAPS:
            name = f"{hi}_minus_{lo}"
            ys = [table[r][name] for r in done]
            xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
            den = sum((x - xm) ** 2 for x in xs)
            slopes[name] = {"bpb_per_doubling": sum(
                (x - xm) * (y - ym) for x, y in zip(xs, ys)) / den,
                "points": len(done)}
        # per-arm BPB-vs-log-steps slope and the improvement-slope ratio
        # (ratec_direct's BPB improvement per doubling vs bpe's — the
        # probe's headline ratio, now with >=2 rungs behind each point)
        arm_slopes = {}
        for arm in ARMS:
            ys = [rungs[str(r)][arm]["final_bpb_overall"] for r in done]
            ym = sum(ys) / len(ys)
            arm_slopes[arm] = sum(
                (x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
        slopes["arm_bpb_per_doubling"] = arm_slopes
        if arm_slopes["bpe"] != 0:
            slopes["improvement_ratio_ratec_direct_over_bpe"] = (
                arm_slopes["ratec_direct"] / arm_slopes["bpe"])
    return table, slopes


def rewrite(rec):
    table, slopes = gaps_and_slopes(rec)
    rec["gap_table"] = {str(r): {k: round(v, 4) for k, v in row.items()}
                        for r, row in table.items()}
    rec["slopes"] = slopes
    rec["status"]["updated_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # engine provenance, restated every rewrite: the version tag was
    # bumped 0.6.0 -> 0.7.0 mid-ladder; the bump changes the version
    # string and adds the opt-in prune feature (unused by the ladder) —
    # the training code path is identical
    rec["protocol"]["engine_note"] = (
        "training runs executed by the engine binary tagged 0.6.0 "
        "(the 0.7.0 bump changes the version string and adds the opt-in "
        "prune feature, which the ladder does not use); ratec_version in "
        "the envelope reflects the runner process's import at rewrite")
    rec["protocol"].pop("envelope", None)
    streams = [f"{pfx}_{d}_{split}_{kind}.npy"
               for pfx in ("bpe_mix", "ratec_main")
               for d in ("en", "hr", "zh", "code")
               for split in ("train", "val") for kind in ("ids", "lens")]
    inputs = {s: os.path.join(RUNS, s) for s in streams}
    inputs["bytes_train_mix"] = os.path.join(
        os.environ.get("RATEC_TWIN_DATA", os.path.join(ROOT, "data")),
        "_train_mix.txt")
    rec["protocol"].update(record_envelope(inputs=inputs,
                                           deps=("numpy", "torch")))
    write_record(LADDER, rec)


def run_one(arm, rung):
    cmd = [sys.executable, os.path.join(HERE, "twin_study.py"),
           "--kind", arm, "--steps", str(rung), "--tag", f"_l{rung}"]
    print(f"[ladder] RUN {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, env=os.environ.copy())
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"[ladder] {arm}@{rung} exited rc={p.returncode} after "
              f"{dt:.0f}s (interrupted runs keep their checkpoint and "
              f"resume on the next invocation)", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=RUNGS[1:], default=None,
                    help="train only this rung")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    rec = load_ladder()
    rungs = rec["rungs"]
    done_rungs = sorted(int(r) for r in rungs
                        if all(a in rungs[r] for a in ARMS))
    print(f"[ladder] rungs complete: {done_rungs or 'none beyond 600'}; "
          f"runs complete: {rec['status']['runs_complete']}/"
          f"{rec['status']['runs_total']}", flush=True)
    if args.status:
        rewrite(rec)
        table, slopes = gaps_and_slopes(rec)
        for r, row in sorted(table.items()):
            print(f"  rung {r}: " + "  ".join(
                f"{k}={v:+.3f}" for k, v in row.items()), flush=True)
        if slopes:
            print(f"  slopes: " + "  ".join(
                f"{k}={v['bpb_per_doubling']:+.4f}" if isinstance(v, dict)
                else f"{k}={v:+.4f}" for k, v in slopes.items()), flush=True)
        return

    rungs.setdefault("600", seed_600())
    rung_list = [args.only] if args.only else [r for r in RUNGS if r > 600]
    for rung in rung_list:
        for arm in ARMS:
            key = str(rung)
            if arm in rungs.get(key, {}):
                print(f"[ladder] skip {arm}@{rung} (recorded)", flush=True)
                continue
            out = run_json(arm, rung)
            if not os.path.exists(out):
                if not run_one(arm, rung):
                    rewrite(rec)
                    sys.exit(2)
            with open(out) as f:
                r = json.load(f)
            assert r["steps"] == rung and r["kind"] == arm
            rungs.setdefault(key, {})[arm] = {
                "final_bpb_overall": r["final_bpb_overall"],
                "final_bpb_per_domain": r["final_bpb_per_domain"],
                "params": r["params"], "wall_sec": r.get("wall_sec"),
                "train_flops_est": r.get("train_flops_est"),
                "source": "run"}
            rec["status"]["runs_complete"] += 1
            if all(a in rungs[key] for a in ARMS) \
                    and key not in rec["status"]["rungs_complete"]:
                rec["status"]["rungs_complete"].append(key)
            rewrite(rec)
            table, slopes = gaps_and_slopes(rec)
            print(f"[ladder] {arm}@{rung} done: "
                  f"{r['final_bpb_overall']:.4f} BPB", flush=True)
            if rung in table:
                print(f"[ladder] gap table rung {rung}: " + "  ".join(
                    f"{k}={v:+.3f}" for k, v in table[rung].items()),
                    flush=True)
            if "bpe_minus_ratec_direct" in slopes:
                s = slopes["bpe_minus_ratec_direct"]["bpb_per_doubling"]
                n = slopes["bpe_minus_ratec_direct"]["points"]
                print(f"[ladder] gap slope (bpe-ratec_direct): "
                      f"{s:+.4f} BPB per log2-doubling over {n} rungs",
                      flush=True)
    print("[ladder] all requested rungs complete", flush=True)
    rewrite(rec)


if __name__ == "__main__":
    main()
