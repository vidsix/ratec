#!/usr/bin/env python3
"""Book-pruning LM measurement runner (PLAN_v0.7 P2): does frequency-aware
pruning of the patch book improve LM BPB at equal compute?

Design (fully controlled, same producer, same protocol):
  arm A  ratec_direct, C=10, streams ratec_c10   (dense book, unpruned)
  arm B  ratec_direct, C=10, streams ratec_c10_pr (prune_after=1 MiB,
         prune_min=2: singletons dropped, ids renumbered at boundaries)
Both arms consume identical byte segmentation — the patcher is
book-independent (asserted: identical per-domain token counts) — so the
ONLY difference is id assignment: unbounded rare-heavy vocabulary vs a
pruned, more-frequent one with id churn.

Engine stats (runs/streams_manifest_c10*.json engine_stats) tell the
mechanism side: the unpruned C=10 book holds ~146k contents; pruning
removes ~196k singletons and keeps ~45k.

Pre-registered interpretation (set before any run): if the pruned arm's
BPB win is within the run-to-run determinism band (~0.00-0.01 BPB — the
protocol is seed-42 deterministic, so 'noise' here means the churn cost
trading off against starvation relief, not sampling noise), the feature
stays opt-in and the doc says so plainly; the engine feature ships on the
strength of its invariants, not on this number.

Resumable per arm; record research/book_pruning.json rewritten after
every completed run.
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
STEPS = 2400
RECORD = os.path.join(ROOT, "research", "book_pruning.json")
ARMS = [("unpruned", "ratec_c10", False),
        ("pruned", "ratec_c10_pr", True)]
PRE_REGISTERED = [
    "if the pruned arm's win is within the determinism band (~0.01 BPB), "
    "the feature stays opt-in and the doc says so plainly — the engine "
    "feature ships on its invariants, not on this number;",
    "the mechanism claim is reported from the engine stats (book sizes, "
    "n_pruned) alongside the LM outcome. Written before any run "
    "(PLAN_v0.7.md P2).",
]


def run_json(tag):
    return os.path.join(RUNS, f"twin_ratec_direct_{tag}.json")


def assert_segmentation_identical():
    """The control, asserted: the patcher is book-independent, so pruned
    and unpruned streams must have IDENTICAL per-domain token counts and
    (ids apart) identical spans. Prune changes id assignment only."""
    import numpy as np
    for dom in ["en", "hr", "zh", "code"]:
        a = np.load(os.path.join(RUNS, f"ratec_c10_{dom}_train_lens.npy"))
        b = np.load(os.path.join(RUNS, f"ratec_c10_pr_{dom}_train_lens.npy"))
        assert len(a) == len(b), (dom, len(a), len(b))
        assert (a == b).all(), f"{dom}: lens differ — segmentation moved?!"
    print("[prune_lm] control: per-domain segmentation identical across "
          "arms (token counts + lens)", flush=True)


def load_record():
    if os.path.exists(RECORD):
        with open(RECORD) as f:
            return json.load(f)
    return {
        "protocol": {
            "what": "book-pruning LM measurement: ratec_direct C=10, "
                    "2400 steps, pruned vs unpruned single-engine dense "
                    "streams (PLAN_v0.7 P2)",
            "arms": {name: {"prefix": pfx, "pruned": pr}
                     for name, pfx, pr in ARMS},
            "frozen": {"d": 96, "layers": 3, "heads": 4, "ctx": 256,
                       "batch": 16, "lr": 3e-4, "seed": 42,
                       "steps": STEPS, "c": 10,
                       "train_bytes": 15_811_609, "val_bytes": 659_473,
                       "prune_after": 1 << 20, "prune_min": 2},
            "control": "identical byte segmentation across arms (patcher "
                       "is book-independent; per-domain token counts "
                       "asserted equal); only id assignment differs",
            "interpretations_pre_registered": PRE_REGISTERED,
        },
        "arms": {},
        "status": {"runs_complete": 0, "runs_total": len(ARMS)},
    }


def rewrite(rec):
    rec["status"]["updated_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if len(rec["arms"]) == 2:
        u = rec["arms"]["unpruned"]["final_bpb_overall"]
        p = rec["arms"]["pruned"]["final_bpb_overall"]
        rec["result"] = {
            "pruned_minus_unpruned_bpb": p - u,
            "note": "negative = pruning helps the LM at equal compute",
        }
    rec["protocol"].pop("envelope", None)
    rec["protocol"].update(record_envelope(inputs={
        "streams_manifest_c10": os.path.join(RUNS,
                                             "streams_manifest_c10.json"),
        "streams_manifest_c10_pr": os.path.join(
            RUNS, "streams_manifest_c10_pr.json"),
    }, deps=("numpy", "torch")))
    write_record(RECORD, rec)


def main():
    rec = load_record()
    assert_segmentation_identical()
    print(f"[prune_lm] runs complete: {rec['status']['runs_complete']}"
          f"/{rec['status']['runs_total']}", flush=True)
    for name, pfx, pruned in ARMS:
        if name in rec["arms"]:
            print(f"[prune_lm] skip {name} (recorded)", flush=True)
            continue
        tag = "prune_noprune" if not pruned else "prune_pruned"
        out = run_json(tag)
        if not os.path.exists(out):
            cmd = [sys.executable, os.path.join(HERE, "twin_study.py"),
                   "--kind", "ratec_direct", "--c", "10",
                   "--prefix", pfx, "--lens_lidx",
                   "--steps", str(STEPS), "--tag", f"_{tag}"]
            print(f"[prune_lm] RUN {' '.join(cmd)}", flush=True)
            p = subprocess.run(cmd, env=os.environ.copy())
            if p.returncode != 0:
                print(f"[prune_lm] {name} exited rc={p.returncode} "
                      f"(checkpoint kept; resume on next invocation)",
                      flush=True)
                rewrite(rec)
                sys.exit(2)
        with open(out) as f:
            r = json.load(f)
        man = json.load(open(os.path.join(
            RUNS, f"streams_manifest_c10{'_pr' if pruned else ''}.json")))
        rec["arms"][name] = {
            "final_bpb_overall": r["final_bpb_overall"],
            "final_bpb_per_domain": r["final_bpb_per_domain"],
            "params": r["params"], "v_eff": r["v_eff"],
            "train_tokens": r["train_tokens"],
            "engine_book_size": man["engine_stats"]["book_size_final"],
            "engine_n_pruned": man["engine_stats"]["n_pruned_final"],
            "wall_sec": r.get("wall_sec"), "source": "run"}
        rec["status"]["runs_complete"] += 1
        rewrite(rec)
        print(f"[prune_lm] {name} done: {r['final_bpb_overall']:.4f} BPB "
              f"(book {man['engine_stats']['book_size_final']:,})", flush=True)
    u = rec["arms"]["unpruned"]["final_bpb_overall"]
    p = rec["arms"]["pruned"]["final_bpb_overall"]
    print(f"[prune_lm] pruned - unpruned = {p - u:+.4f} BPB "
          f"({'pruning helps' if p < u else 'pruning does not help'} "
          f"the LM at equal compute)", flush=True)
    rewrite(rec)


if __name__ == "__main__":
    main()
