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


TABLES = {
    "scale": ("research/scale_mix15.8MB.json", render_scale),
    "fertility": ("research/bench_tiktoken.json", render_fertility),
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
