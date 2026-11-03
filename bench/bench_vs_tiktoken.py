#!/usr/bin/env python3
"""Bench: ratec library vs tiktoken, same bytes, same box, same protocol as
the speed2 bake-off (scored spans after 32768B warm edge, medians).

Arms:
  tiktoken r50k_base / cl100k_base / o200k_base   encode_ordinary (25 reps)
  ratec lib (this engine, pf default)             fresh engine per rep (31 reps)
  ratec lib batch(threads=2) vs tiktoken encode_ordinary_batch(num_threads=2)

Also: prefetch-depth sweep pf 0..4 on english (validates the library default).
Outputs: bench.json, bench.csv, fig12_lib.png in OUT.
"""
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, "/home/z/my-project/ratec")
import ratec

DATA = "/home/z/my-project/bakeoff_data"
OUT = "/home/z/my-project/download/ratec_lib_report"
os.makedirs(OUT, exist_ok=True)
DOMAINS = ["english", "code", "chinese", "base64"]

spans = {}
for dom in DOMAINS:
    raw = open(f"{DATA}/{dom}.txt", "rb").read()
    spans[dom] = raw[32768: 32768 + 267232 if dom in ("english", "chinese")
                     else len(raw)]
# use the exact scored spans from meta for comparability with speed2
import json as _json
meta = _json.load(open("/home/z/my-project/bakeoff_cache/meta.json"))
for dom in DOMAINS:
    m = meta[dom]
    raw = open(f"{DATA}/{dom}.txt", "rb").read()
    spans[dom] = raw[m["warm_bytes"]: m["warm_bytes"] + m["scored_bytes"]]

def med(v): return statistics.median(v)

res = {d: {"bytes": len(spans[d])} for d in DOMAINS}
print("spans:", {d: len(spans[d]) for d in DOMAINS})

# ------------------------------------------------------------ tiktoken arms
import tiktoken
TENC = ["r50k_base", "cl100k_base", "o200k_base"]
for name in TENC:
    enc = tiktoken.get_encoding(name)
    key = "bpe50k" if name == "r50k_base" else name.split("_")[0]
    for dom in DOMAINS:
        text = spans[dom].decode("utf-8")
        enc.encode_ordinary(text)
        ts = []
        for _ in range(25):
            t0 = time.perf_counter_ns()
            ids = enc.encode_ordinary(text)
            ts.append((time.perf_counter_ns() - t0) / 1e9)
        res[dom][key] = {"med_s": med(ts), "units": len(ids)}
        print(f"  tiktoken {name:12s} {dom:8s} {med(ts)*1e3:7.2f} ms  "
              f"{len(spans[dom])/med(ts)/1e6:6.2f} MB/s  {len(ids)} units")

# ------------------------------------------------------------- ratec single
# pool steady-state: one engine, in-place reset between reps (reset cost
# included). Supplementary modes recorded for the report.
print()
for dom in DOMAINS:
    data = spans[dom]
    tok = ratec.RateC(c=20, store_book=False, pf=2, prebits=21)
    tok.reset()
    ts = []
    for _ in range(31):
        t0 = time.perf_counter_ns()
        tok.reset()
        spans_out = tok.encode_to_spans(data)
        ts.append((time.perf_counter_ns() - t0) / 1e9)
    units = len(spans_out)
    cold = []
    for _ in range(11):
        t0 = time.perf_counter_ns()
        ratec.RateC(c=20, store_book=False, pf=2, prebits=21).encode_to_spans(data)
        cold.append((time.perf_counter_ns() - t0) / 1e9)
    st_ref = np.load(f"/home/z/my-project/bakeoff_cache/{dom}.r20starts.npy")
    st_new = np.array([s for s, _, _ in spans_out], dtype=np.int64)
    exact = bool(np.array_equal(st_new, st_ref))
    res[dom]["ratec"] = {"med_s": med(ts), "units": units, "exact": exact,
                         "pf": 2,
                         "cold_med_s": med(cold),
                         "cold_MBps": len(data) / med(cold) / 1e6}
    print(f"  ratec lib       {dom:8s} {med(ts)*1e3:7.2f} ms  "
          f"{len(data)/med(ts)/1e6:6.2f} MB/s  {units} units  exact={exact}  "
          f"(cold: {med(cold)*1e3:.1f} ms)")

# ------------------------------------------------------------- batch arms
print()
docs = [spans[d] for d in DOMAINS]
tokbatch = ratec.batch_spans(docs, c=20, threads=2)  # cold engines
tb = sum(len(x[0]) for x in tokbatch)
for dom, (starts, n) in zip(DOMAINS, tokbatch):
    st_ref = np.load(f"/home/z/my-project/bakeoff_cache/{dom}.r20starts.npy")
    exact = bool(np.array_equal(starts.astype(np.int64), st_ref.astype(np.int64)))
    assert exact, dom
t0 = time.perf_counter()
for _ in range(9):
    ratec.batch_spans(docs, c=20, threads=2)
ratec_batch_s = (time.perf_counter() - t0) / 9

enc = tiktoken.get_encoding("r50k_base")
texts = [spans[d].decode("utf-8") for d in DOMAINS]
enc.encode_ordinary_batch(texts, num_threads=2)
t0 = time.perf_counter()
for _ in range(9):
    enc.encode_ordinary_batch(texts, num_threads=2)
tik_batch_s = (time.perf_counter() - t0) / 9
tot_b = sum(len(x) for x in docs)
print(f"  batch x4 docs, 2 threads: ratec {tot_b/ratec_batch_s/1e6:.2f} MB/s "
      f"({tb} patches) | tiktoken r50k {tot_b/tik_batch_s/1e6:.2f} MB/s")

# --------------------------------------------------------------- pf sweep
print()
pf_tab = {}
data = spans["english"]
for pf in range(5):
    tok = ratec.RateC(c=20, store_book=False, pf=pf)
    ts = []
    for _ in range(15):
        tok.reset()
        t0 = time.perf_counter_ns()
        tok.encode_to_spans(data)
        ts.append((time.perf_counter_ns() - t0) / 1e9)
    pf_tab[pf] = med(ts)
    print(f"  pf={pf}: {med(ts)*1e3:7.2f} ms  {len(data)/med(ts)/1e6:6.2f} MB/s")

# --------------------------------------------------------------- aggregate
ARMS = ["bpe50k", "cl100k", "o200k", "ratec"]
tot_b = sum(res[d]["bytes"] for d in DOMAINS)
overall = {}
for a in ARMS:
    t = sum(res[d][a]["med_s"] for d in DOMAINS)
    overall[a] = {"total_bytes": tot_b, "total_s": t,
                  "MB_per_s": tot_b / t / 1e6}
for d in DOMAINS:
    for a in ARMS:
        res[d][a]["MB_per_s"] = res[d]["bytes"] / res[d][a]["med_s"] / 1e6
        res[d][a]["units_per_byte"] = res[d][a]["units"] / res[d]["bytes"]

speed = {d: res[d]["ratec"]["MB_per_s"] / res[d]["bpe50k"]["MB_per_s"]
         for d in DOMAINS}
speed["ALL"] = overall["ratec"]["MB_per_s"] / overall["bpe50k"]["MB_per_s"]
print("\n" + "=" * 64)
print(f"{'domain':9s}" + "".join(f"{a:>13s}" for a in ARMS) + "   (MB/s)")
for d in DOMAINS + ["ALL"]:
    cells = [f"{overall[a]['MB_per_s']:13.2f}" if d == "ALL"
             else f"{res[d][a]['MB_per_s']:13.2f}" for a in ARMS]
    print(f"{d:9s}" + "".join(cells))
print(f"\nratec/tiktoken r50k speedup: " +
      "  ".join(f"{d}={speed[d]:.2f}x" for d in DOMAINS + ["ALL"]))
print(f"batch(2T) ratec/tiktoken: {tot_b/ratec_batch_s / (tot_b/tik_batch_s):.2f}x")

json.dump({"protocol": {
    "c_bits": 20, "pf": 3, "reps": {"tiktoken": 25, "ratec": 31},
    "input": "bake-off scored spans (32768B warm edge trimmed)",
    "ratec_arm": "ratec 0.1.0 engine, pool steady-state (in-place reset, reset cost included), store_book=False, pf=2, prebits=21",
    "exactness": {d: res[d]["ratec"]["exact"] for d in DOMAINS}},
    "per_domain": res, "overall": overall,
    "speedup_vs_r50k": speed,
    "batch_2t": {"ratec_MBps": tot_b / ratec_batch_s / 1e6,
                  "tiktoken_MBps": tot_b / tik_batch_s / 1e6,
                  "ratio": (tot_b / ratec_batch_s) / (tot_b / tik_batch_s),
                  "exact": True},
    "pf_sweep_english": {str(k): {"med_s": v, "MBps": len(data) / v / 1e6}
                          for k, v in pf_tab.items()}},
    open(f"{OUT}/bench.json", "w"), indent=1, default=float)

with open(f"{OUT}/bench.csv", "w") as f:
    f.write("domain,arm,bytes,median_ms,MB_per_s,units_per_byte\n")
    for d in DOMAINS:
        for a in ARMS:
            r = res[d][a]
            f.write(f"{d},{a},{res[d]['bytes']},{r['med_s']*1e3:.3f},"
                    f"{r['MB_per_s']:.2f},{r['units_per_byte']:.4f}\n")
    for a in ARMS:
        f.write(f"ALL,{a},{tot_b},{overall[a]['total_s']*1e3:.1f},"
                f"{overall[a]['MB_per_s']:.2f},\n")
    f.write(f"BATCH2T,ratec,{tot_b},{ratec_batch_s*1e3:.1f},"
            f"{tot_b/ratec_batch_s/1e6:.2f},\n")
    f.write(f"BATCH2T,bpe50k,{tot_b},{tik_batch_s*1e3:.1f},"
            f"{tot_b/tik_batch_s/1e6:.2f},\n")

# ---------------------------------------------------------------- fig12
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C = {"bpe50k": "#888888", "cl100k": "#AAAAAA", "o200k": "#CCCCCC",
     "ratec": "#B03A2E", "ratec_b": "#E74C3C", "tik_b": "#555555"}
LBL = {"bpe50k": "tiktoken r50k", "cl100k": "tiktoken cl100k",
       "o200k": "tiktoken o200k", "ratec": "RateC lib 0.1"}

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3), constrained_layout=True)
ax = axes[0]
groups = DOMAINS + ["ALL"]
x = np.arange(len(groups)); wd = 0.2
for i, a in enumerate(ARMS):
    vals = [overall[a]["MB_per_s"] if g == "ALL" else res[g][a]["MB_per_s"]
            for g in groups]
    ax.bar(x + (i - 1.5) * wd, vals, wd, color=C[a], label=LBL[a])
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylabel("tokenization throughput (MB/s, log)")
ax.set_title("Encode speed, same bytes, 1 thread\nhigher = faster")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.25, which="both")

ax = axes[1]
vals = [speed[d] for d in DOMAINS] + [speed["ALL"]]
bars = ax.bar(np.arange(len(vals)), vals, 0.55,
              color=["#B03A2E" if v >= 1 else "#DD8452" for v in vals])
ax.axhline(1.0, color="#444", ls="--", lw=1.2)
for r, v in zip(bars, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.2f}x",
            ha="center", fontsize=9)
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(DOMAINS + ["ALL"])
ax.set_ylabel("speedup vs tiktoken r50k (x)")
ax.set_title("RateC lib vs GPT-2's BPE, 1 thread\n>1 = RateC wins")
ax.grid(axis="y", alpha=0.25)

ax = axes[2]
w = 0.32
ax.bar(0 - w / 2, overall["ratec"]["MB_per_s"], w, color=C["ratec_b"],
       label="RateC 1T")
ax.bar(0 + w / 2, overall["bpe50k"]["MB_per_s"], w, color=C["tik_b"],
       label="tiktoken 1T")
ax.bar(1 - w / 2, tot_b / ratec_batch_s / 1e6, w, color=C["ratec_b"])
ax.bar(1 + w / 2, tot_b / tik_batch_s / 1e6, w, color=C["tik_b"])
ax.set_xticks([0, 1])
ax.set_xticklabels(["1 thread", "2 threads (4 docs)"])
ax.set_ylabel("aggregate MB/s")
ax.set_title("Aggregate throughput on the\n4 bake-off domains")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.25)

fig.savefig(f"{OUT}/fig12_lib.png", dpi=150)
print(f"\nwrote {OUT}/bench.json, bench.csv, fig12_lib.png")
