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


TABLES = {
    "scale": ("research/scale_mix15.8MB.json", render_scale),
    "fertility": ("research/bench_tiktoken.json", render_fertility),
    "ladder": ("research/twin_ladder.json", render_ladder),
    "c_sweep": ("research/lm_c_sweep.json", render_c_sweep),
}


def regenerate(readme_text):
    out = readme_text
    for name, (record, render) in TABLES.items():
        body = render(load(os.path.basename(record)))
        begin = f"<!-- table:{name} generated from {record} — do not edit; run make tables -->"
        end = f"<!-- /table:{name} -->"
        pre, _sep, rest = out.partition(f"{begin}\n")
        if not _sep:
            raise SystemExit(f"sentinels for {name} missing from README.md")
        _old, _sep2, post = rest.partition(f"\n{end}")
        if not _sep2:
            raise SystemExit(f"closing sentinel for {name} missing")
        out = f"{pre}{begin}\n{body}{end}{post}"
    return out


def main():
    check = "--check" in sys.argv
    text = open(README).read()
    new = regenerate(text)
    if new == text:
        print("make tables: README tables match the records")
        return
    if check:
        # report which tables drifted for the test's failure message
        drifted = [n for n in TABLES
                   if render(load(os.path.basename(TABLES[n][0])))
                   not in text]
        print("make tables --check: README tables STALE: "
              + ", ".join(drifted))
        sys.exit(1)
    with open(README, "w") as f:
        f.write(new)
    print("make tables: README tables regenerated from records")


if __name__ == "__main__":
    main()
