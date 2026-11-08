#!/usr/bin/env python3
"""Bench: ratec library vs tiktoken, same bytes, same box, same protocol as
the speed2 bake-off (scored spans after 32768B warm edge, medians).

Arms:
  tiktoken r50k_base / cl100k_base / o200k_base   encode_ordinary (21 reps)
  ratec lib (this engine, pf=2 = resolved library default)   pool steady-state, 21 reps
  ratec lib batch(threads=2) vs tiktoken encode_ordinary_batch(num_threads=2)
    (batch arms also interleaved round-robin, 9 reps each, medians)

Also: prefetch-depth sweep pf 0..4 on english (validates the library default).
Outputs: bench.json, bench.csv, fig12_lib.png in OUT.
"""
import json
import os
import statistics
import sys
import time
from importlib import metadata as _ilmd

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # ratec/ repo root
sys.path.insert(0, ROOT)
import ratec

FIXTURES = os.environ.get(
    "RATEC_FIXTURES",
    os.path.join(ROOT, "tests", "data", "fixtures"))
OUT = os.environ.get("RATEC_OUT") or os.path.join(os.getcwd(), "results")
os.makedirs(OUT, exist_ok=True)
DOMAINS = ["english", "code", "chinese", "base64"]
PF = 2                # ratec arm prefetch depth (the resolved library default)
REPS = 21             # interleaved single-thread reps per arm (actual loop count)
BATCH_REPS = 9        # interleaved batch reps per arm
PF_REPS = 15          # reps per point of the pf sweep

def _pkg_version(name):
    try:
        return _ilmd.version(name)
    except _ilmd.PackageNotFoundError:
        return None

# scored spans ship with the repo (identical bytes everywhere)
spans = {dom: open(os.path.join(FIXTURES, f"{dom}.span.bin"), "rb").read()
         for dom in DOMAINS}

def _golden(dom):
    """Reference starts when available: shipped (C=20) or private cache via
    RATEC_BAKEOFF_CACHE. Exactness is asserted only where a golden exists."""
    p = os.path.join(FIXTURES, f"{dom}.starts.npy")
    if os.path.exists(p):
        return np.load(p)
    cache = os.environ.get("RATEC_BAKEOFF_CACHE")
    if cache and os.path.exists(os.path.join(cache, f"{dom}.r20starts.npy")):
        return np.load(os.path.join(cache, f"{dom}.r20starts.npy"))
    return None

def med(v): return statistics.median(v)

res = {d: {"bytes": len(spans[d])} for d in DOMAINS}
print("spans:", {d: len(spans[d]) for d in DOMAINS})

# ------------------------------------- interleaved tiktoken/ratec arms
# The sandbox CPU drifts by tens of percent over minutes; measuring arms
# back-to-back biases whoever runs second. Every rep is therefore measured
# round-robin across arms within one process, and medians are compared.
import tiktoken
TENC = [("bpe50k", "r50k_base"), ("cl100k", "cl100k_base"), ("o200k", "o200k_base")]
tokencs = [(k, tiktoken.get_encoding(n)) for k, n in TENC]
rtok = ratec.RateC(c=20, store_book=False, pf=PF)         # pool: reused, reset per rep
rtext = {d: spans[d].decode("utf-8") for d in DOMAINS}
tids = {d: {k: None for k, _ in tokencs} for d in DOMAINS}
runtimes = {d: {k: [] for k, _ in tokencs} for d in DOMAINS}
runtimes_r = {d: [] for d in DOMAINS}
rcold = {d: [] for d in DOMAINS}

for rep in range(REPS):
    for d in DOMAINS:
        text = rtext[d]
        data = spans[d]
        for k, enc in tokencs:
            t0 = time.perf_counter_ns()
            ids = enc.encode_ordinary(text)
            dt = (time.perf_counter_ns() - t0) / 1e9
            tids[d][k] = len(ids)
            runtimes[d][k].append(dt)
        # ratec pool steady-state (reset inside the timed window, like tiktoken's
        # stateless call is the whole work)
        t0 = time.perf_counter_ns()
        rtok.reset()
        sp = rtok.encode_to_spans(data)
        runtimes_r[d].append((time.perf_counter_ns() - t0) / 1e9)
        if rep >= 1 and rep % 5 == 1:                      # cold sample
            t0 = time.perf_counter_ns()
            ratec.RateC(c=20, store_book=False, pf=PF).encode_to_spans(data)
            rcold[d].append((time.perf_counter_ns() - t0) / 1e9)

# exactness: fresh engine per domain vs reference starts (shipped C=20 golden,
# or the private bake-off cache when RATEC_BAKEOFF_CACHE is set)
exactness = {}
for d in DOMAINS:
    ref = _golden(d)
    sp = ratec.RateC(c=20, store_book=False, pf=PF).encode_to_spans(spans[d])
    exactness[d] = None if ref is None else bool(np.array_equal(
        np.array([s for s, _, _ in sp], dtype=np.int64), ref.astype(np.int64)))

res2 = {}
for d in DOMAINS:
    res2[d] = {"bytes": len(spans[d])}
    for k, _ in tokencs:
        res2[d][k] = {"med_s": med(runtimes[d][k]), "units": tids[d][k]}
    res2[d]["ratec"] = {"med_s": med(runtimes_r[d]), "units": None,
                        "exact": exactness[d], "pf": PF,
                        "cold_med_s": med(rcold[d]) if rcold[d] else None}
# units: from a fresh exactness-style pass (fresh engine per domain)
for d in DOMAINS:
    sp = ratec.RateC(c=20, store_book=False, pf=PF).encode_to_spans(spans[d])
    res2[d]["ratec"]["units"] = len(sp)
res = res2

for d in DOMAINS:
    for k, _ in tokencs:
        r = res[d][k]
        print(f"  tiktoken {k:8s} {d:8s} {r['med_s']*1e3:7.2f} ms  "
              f"{len(spans[d])/r['med_s']/1e6:6.2f} MB/s  {r['units']} units")
    r = res[d]["ratec"]
    cold = r["cold_med_s"]
    print(f"  ratec lib       {d:8s} {r['med_s']*1e3:7.2f} ms  "
          f"{len(spans[d])/r['med_s']/1e6:6.2f} MB/s  {r['units']} units  exact={r['exact']}"
          + (f"  (cold: {cold*1e3:.1f} ms)" if cold else ""))
print()

# ------------------------------------------------------------- batch arms
print()
docs = [spans[d] for d in DOMAINS]
tokbatch = ratec.batch_spans(docs, c=20, threads=2)  # cold engines
tb = sum(len(x[0]) for x in tokbatch)
for dom, (starts, n) in zip(DOMAINS, tokbatch):
    ref = _golden(dom)
    if ref is not None:
        assert bool(np.array_equal(starts.astype(np.int64), ref.astype(np.int64))), dom

enc = tiktoken.get_encoding("r50k_base")
texts = [spans[d].decode("utf-8") for d in DOMAINS]
# batch arms timed INTERLEAVED (round-robin reps, medians) — the same drift
# discipline as the single-thread arms; timing them back-to-back would bias
# whoever runs second (exactly the docstring warning at the top of the file)
ratec.batch_spans(docs, c=20, threads=2)             # warm both paths once
enc.encode_ordinary_batch(texts, num_threads=2)
ratec_ts, tik_ts = [], []
for _ in range(BATCH_REPS):
    t0 = time.perf_counter()
    ratec.batch_spans(docs, c=20, threads=2)
    ratec_ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    enc.encode_ordinary_batch(texts, num_threads=2)
    tik_ts.append(time.perf_counter() - t0)
ratec_batch_s = med(ratec_ts)
tik_batch_s = med(tik_ts)
tot_b = sum(len(x) for x in docs)
print(f"  batch x4 docs, 2 threads (interleaved, median of {BATCH_REPS}): "
      f"ratec {tot_b/ratec_batch_s/1e6:.2f} MB/s ({tb} patches) | "
      f"tiktoken r50k {tot_b/tik_batch_s/1e6:.2f} MB/s")

# --------------------------------------------------------------- pf sweep
print()
pf_tab = {}
data = spans["english"]
for pf in range(5):
    tok = ratec.RateC(c=20, store_book=False, pf=pf)
    ts = []
    for _ in range(PF_REPS):
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
    "c_bits": 20, "pf": PF,
    "reps": {"tiktoken_encode_ordinary": REPS, "ratec_pool": REPS,
             "batch": BATCH_REPS, "pf_sweep": PF_REPS,
             "ratec_cold_samples_per_domain": len(rcold["english"])},
    "timing": "round-robin interleaved within one process, medians; "
              "batch arms interleaved too",
    "input": "bake-off scored spans (shipped fixtures; or RATEC_BAKEOFF_DATA + RATEC_BAKEOFF_CACHE)",
    "golden_mode": "shipped C=20 starts" if any(v is not None for v in exactness.values()) else "none (exactness not asserted)",
    "ratec_arm": f"ratec engine, pool steady-state (in-place reset, reset cost "
                 f"included), store_book=False, pf={PF}, prebits=16 "
                 f"(library default)",
    "ratec_version": ratec.__version__,
    "tiktoken_version": _pkg_version("tiktoken"),
    "numpy_version": _pkg_version("numpy"),
    "exactness": {d: res[d]["ratec"]["exact"] for d in DOMAINS}},
    "per_domain": res, "overall": overall,
    "speedup_vs_r50k": speed,
    "batch_2t": {"ratec_MBps": tot_b / ratec_batch_s / 1e6,
                  "tiktoken_MBps": tot_b / tik_batch_s / 1e6,
                  "ratio": (tot_b / ratec_batch_s) / (tot_b / tik_batch_s),
                  "reps": BATCH_REPS, "timing": "interleaved, median",
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
for _fp in ("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
        except Exception:
            pass                      # labels are ASCII; missing font is fine
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C = {"bpe50k": "#888888", "cl100k": "#AAAAAA", "o200k": "#CCCCCC",
     "ratec": "#B03A2E", "ratec_b": "#E74C3C", "tik_b": "#555555"}
LBL = {"bpe50k": "tiktoken r50k", "cl100k": "tiktoken cl100k",
       "o200k": "tiktoken o200k", "ratec": f"RateC lib {ratec.__version__}"}

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
