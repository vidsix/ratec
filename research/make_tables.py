#!/usr/bin/env python3
"""Regenerate the record-backed README tables (PLAN_v0.6 P2-2).

Every table between `<!-- table:NAME ... -->` / `<!-- /table:NAME -->`
sentinels in README.md is GENERATED from a committed research record —
hand-editing a number inside the sentinels fails `make tables --check`
(wired into the suite as tests/test_tables.py). This is the drift guard
that would have caught the v0.5.0 supersession before it shipped.

  tables:
    scale      <- research/scale_mix15.8MB.json
    fertility  <- research/bench_tiktoken.json   (ALL four domains)

  usage:
    python research/make_tables.py            # rewrite README between sentinels
    python research/make_tables.py --check    # exit 1 if README is stale
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
README = os.path.join(ROOT, "README.md")

MINUS = "\u2212"  # the typographic minus used across the README tables


def load(name):
    with open(os.path.join(ROOT, "research", name)) as f:
        return json.load(f)


# ------------------------------------------------------------------ scale
def render_scale(rec):
    rows = rec["rows"]
    cold = next(r for r in rows if r["arm"] == "ratec cold")["units_per_byte"]
    tik = next(r for r in rows if r["arm"].startswith("tiktoken"))
    lines = ["| adaptation seen | units/byte (probe) | vs cold |",
             "|---|---|---|"]
    lines.append("| 0 (cold engine) | %.3f | — |" % cold)
    shown = (64000, 256000, 1048576, 8388608)     # log-spaced display subset
    for r in rows:
        if r["arm"] != "ratec warm" or r["scale"] not in shown:
            continue
        s = r["scale"]
        label = f"{s // 1048576} MB" if s >= 1048576 else f"{s // 1024} KB"
        pct = (r["units_per_byte"] / cold - 1.0) * 100.0
        lines.append("| %s | %.3f | %s%.0f%% |"
                     % (label, r["units_per_byte"], MINUS, -pct))
    lines.append("| tiktoken r50k_base (same probe) | %.3f | flat by "
                 "construction |" % tik["units_per_byte"])
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------- fertility
def render_fertility(rec):
    lines = ["| domain | tiktoken r50k units/byte | ratec units/byte "
             "| fewer LM positions |", "|---|---|---|---|"]
    for dom in ("english", "code", "chinese", "base64"):
        pd = rec["per_domain"][dom]
        bpe = pd["bpe50k"]["units_per_byte"]
        rc = pd["ratec"]["units_per_byte"]
        lines.append("| %s | %.3f | %.3f | %s%.0f%% |"
                     % (dom, bpe, rc, MINUS, (1.0 - rc / bpe) * 100.0))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- ladder
def _sgn(v):
    """Signed value with the README's typographic minus (never doubled)."""
    return ("%s%.3f" % (MINUS, abs(v))) if v < 0 else ("%.3f" % v)


def render_ladder(rec):
    """The LM-side scale ladder (PLAN_v0.7 P1): BPB per arm at each rung,
    with the headline gap (bpe − ratec_direct) and its log-steps slope."""
    rungs = sorted(int(r) for r in rec.get("rungs", {}))
    arms = ("bpe", "ratec_direct", "ratec", "bytes")
    lines = ["| steps | bpe | ratec_direct | ratec (factorized) | bytes floor "
             "| gap bpe−ratec_direct |", "|---|---|---|---|---|---|"]
    for r in rungs:
        e = rec["rungs"][str(r)]
        vals = [e.get(a, {}).get("final_bpb_overall") for a in arms]
        gap = (vals[0] - vals[1]) if None not in (vals[0], vals[1]) else None
        lines.append("| %d | %s | %s | %s | %s | %s |" % (
            r,
            "—" if vals[0] is None else "%.3f" % vals[0],
            "—" if vals[1] is None else "%.3f" % vals[1],
            "—" if vals[2] is None else "%.3f" % vals[2],
            "—" if vals[3] is None else "%.3f" % vals[3],
            "—" if gap is None else _sgn(gap)))
    s = rec.get("slopes", {}).get("bpe_minus_ratec_direct", {})
    if "bpb_per_doubling" in s:
        # a positive slope means the (negative) gap shrinks toward zero:
        # the tokenization deficit narrows by that many BPB per doubling
        # of training compute; negative would mean it widens
        lines.append("")
        lines.append(
            "Gap slope: **%s BPB per log2-doubling** (least squares "
            "over the %d completed rungs; the gap %s with training scale)."
            % (_sgn(s["bpb_per_doubling"]), s["points"],
               "narrows" if s["bpb_per_doubling"] > 0 else "WIDENS"))
    return "\n".join(lines) + "\n"


def render_c_sweep(rec):
    """The LM-optimal C sweep (PLAN_v0.7 P3 + the v0.8 extension): BPB at
    each C, with the param count as the explicit budget confound."""
    rows = rec.get("rows", {})
    cs = sorted(int(k) for k in rows)
    lines = ["| C | head rows | params | final BPB | bytes/token |",
             "|---|---|---|---|---|"]
    for c in cs:
        e = rows[str(c)]
        head = 256 + 9 * (1 << c)
        lines.append("| %d | %d | %s | %.4f | %.2f |" % (
            c, head, format(e["params"], ","), e["final_bpb_overall"],
            e.get("bytes_per_token", 0.0)))
    lines.append("")
    lines.append(
        "The sweep compares budgets, not parameter-matched models — head "
        "rows grow as 256 + 9·2^C and the record states that confound per "
        "row. The v0.8 C=14 extension (pre-registered in the record's "
        "extension_v0_8 block) landed BELOW C=12, firing the first "
        "pre-registered interpretation: the LM-optimal budget at probe "
        "scale sits inside {12,14}, C=14 measured best.")
    return "\n".join(lines) + "\n"


def render_quietbox(rec):
    """The quiet-box envelope (PLAN_v0.9 P1.1): per-lane reference
    numbers the 1.0 perf floors are ratios of, plus the P1.2 gate
    ratios and the tiktoken reference judge."""
    n = rec["bench_call"]["n_bytes"]
    lanes = rec["lanes"]
    NOTES = [
        ("encode_bytes_warm", "`RateC.encode_bytes`",
         "sequential judge+patcher; persistent engine, book persists "
         "across reps (the `bench()` semantics)"),
        ("spans_arrays", "`RateC.encode_to_spans`",
         "tuple-free `as_arrays=True`; pool engine, `reset()` per rep"),
        ("stream_spans_cold", "`encode_stream_parallel` (spans)",
         "fresh engine per rep — the prep-pipeline cold shape"),
        ("stream_ids_cold", "`…(ids=True)` cold",
         "fresh engine per rep — includes the cold-book intern merge"),
        ("stream_spans_warm", "`encode_stream_parallel` (spans, warm)",
         "persistent engine, saturated book"),
        ("stream_ids_warm", "`…(ids=True)` warm",
         "saturated book — the intern pre-probe hits"),
        ("decode_bytes", "`RateC.decode_bytes`",
         "single-thread full-book read of the stream ids"),
        ("decode_batch", "`RateC.decode_batch(threads=2)`",
         "8-way id fan"),
        ("batch_encode", "`ratec.batch_encode(threads=2)`",
         "4 x 1 MiB docs, per-doc cold engines"),
    ]
    lines = [
        "Envelope: min-of-%d-round medians, round-robin interleave, %s; "
        "provenance (loadavg, governor probe, affinity, exactness "
        "cross-checks) in the record. The 1.0 per-lane floors are RATIOS "
        "of these numbers, so a re-issue on a better host lifts the "
        "floors, never the ratios." % (
            rec["record"]["runs"]["rounds"],
            rec["record"]["created_utc"][:10]),
        "",
        "| lane (exact, %d MiB stream) | envelope MB/s | round medians |"
        % (n >> 20),
        "|---|---:|---|",
    ]
    for key, label, note in NOTES:
        meds = lanes[key]["round_medians_s"]
        med_mb = " ".join("%.2f" % (n / m / 1e6) for m in meds)
        lines.append("| %s | %.2f | %s |" % (label, lanes[key]["mb_s"],
                                             med_mb + " — " + note))
    lines += ["", "| gate ratio (P1.2) | value | gate |", "|---|---:|---|"]
    r = rec["ratios"]
    lines.append("| ids/spans cold | %.1f%% | ≥ 95%%: %s |" % (
        r["ids_cold_over_spans_cold"] * 100,
        "PASS" if r["ids_cold_over_spans_cold"] >= 0.95 else "FAIL"))
    lines.append("| ids/spans warm | %.1f%% | ≥ 95%%: %s |" % (
        r["ids_warm_over_spans_warm"] * 100,
        "PASS" if r["ids_warm_over_spans_warm"] >= 0.95 else "FAIL"))
    lines += ["", "| tiktoken r50k reference | envelope MB/s |",
              "|---|---:|"]
    for dom, e in rec["tiktoken_reference"].items():
        lines.append("| %s | %.2f |" % (dom, e["mb_s"]))
    lines.append("")
    lines.append(
        "The judge-parity anchor: ratec's judge core at tiktoken parity "
        "in isolation is an attribution claim (PLAN_v0.8) — these rows "
        "are the same-corpus reference the next attribution re-run "
        "starts from.")
    return "\n".join(lines) + "\n"


TABLES = {
    "scale": ("README.md", "research/scale_mix15.8MB.json", render_scale),
    "fertility": ("README.md", "research/bench_tiktoken.json", render_fertility),
    "ladder": ("README.md", "research/twin_ladder.json", render_ladder),
    "c_sweep": ("README.md", "research/lm_c_sweep.json", render_c_sweep),
    "quietbox": ("BENCHMARKS.md", "research/quietbox_envelope.json",
                 render_quietbox),
}

FILES = {}
for _name, (_path, _record, _render) in TABLES.items():
    FILES.setdefault(_path, []).append(_name)


def regenerate(path, names):
    out = open(path).read()
    for name in names:
        _path, record, render = TABLES[name]
        body = render(load(os.path.basename(record)))
        begin = f"<!-- table:{name} generated from {record} — do not edit; run make tables -->"
        end = f"<!-- /table:{name} -->"
        pre, _sep, rest = out.partition(f"{begin}\n")
        if not _sep:
            raise SystemExit(f"sentinels for {name} missing from {path}")
        body_out, _sep2, post = rest.partition(end)
        if not _sep2:
            raise SystemExit(f"closing sentinel for {name} missing")
        out = f"{pre}{begin}\n{body}{end}{post}"
    return out


def main():
    check = "--check" in sys.argv
    drifted = []
    changed = False
    for path, names in FILES.items():
        text = open(path).read()
        new = regenerate(path, names)
        if new != text:
            changed = True
            for n in names:
                _path, record, render = TABLES[n]
                body = render(load(os.path.basename(record)))
                if body not in text:
                    drifted.append(f"{path}:{n}")
    if not changed:
        print("make tables: README + BENCHMARKS tables match the records")
        return
    if check:
        print("make tables --check: tables STALE: " + ", ".join(drifted))
        sys.exit(1)
    for path in FILES:
        new = regenerate(path, FILES[path])
        with open(path, "w") as f:
            f.write(new)
    print("make tables: README + BENCHMARKS tables regenerated from records")


if __name__ == "__main__":
    main()
