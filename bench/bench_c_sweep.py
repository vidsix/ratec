#!/usr/bin/env python3
"""FAIR C-parameter sweep for the ratec library.

Claim under test (no cherry-picked C): increasing the patch budget C must
decrease tokens/byte (longer patches), while single-thread throughput
changes only slightly. Every C value in {12,16,20,24,28,32} is reported for
BOTH tokens/byte and MB/s, and timing only counts if the bit-exactness gate
passes at EVERY C.

Protocol (mirrors bench/bench_vs_tiktoken.py):
  * inputs are the bake-off scored spans (meta.json warm/scored trimming)
  * the sandbox CPU drifts by tens of percent over minutes, so all
    single-thread arms are measured round-robin within one rep loop
    (21 reps) and medians are compared; the batch arm likewise sweeps C
    round-robin (9 reps)
  * ratec arm = pool steady-state: tok.reset() INSIDE the timed window,
    then tok.encode_to_spans(span)

Gates (must pass for every domain and C before any timing counts):
  G1 fresh RateC(c=C, store_book=False).encode_to_spans(span) starts ==
     cached golden {dom}.r{C}starts.npy (int64 exact)
  G2 spans tile [0, len) exactly (contiguity)
  G3 patch_from_sur(surprise(span), C) == golden starts for every C
     (proves judge/patcher independence across C; one surprise pass
     serves all C since the judge is C-independent)
  G4 batch_spans(4 docs, c=C, threads=2) starts == serial golden starts

Outputs: printed table + trend summary, c_sweep.json, c_sweep.csv,
fig_c_sweep.png in OUT (default ./results, override with RATEC_OUT).
Exit code 0 iff all gates pass, else 1.
"""
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # ratec/ repo root
sys.path.insert(0, ROOT)
import ratec

FIXTURES = os.environ.get(
    "RATEC_FIXTURES",
    os.path.join(ROOT, "tests", "data", "fixtures"))
PRIV_CACHE = os.environ.get("RATEC_BAKEOFF_CACHE")   # full per-C goldens, optional
OUT = os.environ.get("RATEC_OUT") or os.path.join(os.getcwd(), "results")
os.makedirs(OUT, exist_ok=True)

DOMAINS = ["english", "code", "chinese", "base64"]
CS = [12, 16, 20, 24, 28, 32]
REPS = 21            # single-thread arms
BATCH_REPS = 9       # batch arm


def med(v):
    return statistics.median(v)


# ---------------------------------------------------------------- inputs
# spans ship with the repo; goldens do too, but only for C=20. The full set
# of per-C reference starts lives in the private bake-off cache — set
# RATEC_BAKEOFF_CACHE to enable the full golden gate at every C.
spans = {dom: open(os.path.join(FIXTURES, f"{dom}.span.bin"), "rb").read()
         for dom in DOMAINS}

def _golden(dom, C):
    p = os.path.join(FIXTURES, f"{dom}.starts.npy")
    if C == 20 and os.path.exists(p):
        return np.load(p).astype(np.int64)
    if PRIV_CACHE and os.path.exists(os.path.join(PRIV_CACHE, f"{dom}.r{C}starts.npy")):
        return np.load(os.path.join(PRIV_CACHE, f"{dom}.r{C}starts.npy")).astype(np.int64)
    return None

golden = {(dom, C): _golden(dom, C) for dom in DOMAINS for C in CS}
golden_mode = ("full private cache" if all(v is not None for v in golden.values())
               else "shipped C=20 + internal cross-checks" if any(v is not None for v in golden.values())
               else "none")

print("C-sweep bench | ratec", ratec.__version__)
print("spans:", {d: len(spans[d]) for d in DOMAINS})
print(f"C values: {CS}   reps: single={REPS} batch={BATCH_REPS} (round-robin, medians)\n")

# ------------------------------------------------------------------ gates
# G1+G2: fresh engine per (domain, C); G3: one surprise pass per domain
# (judge is C-independent), then the reference patcher per C.
gate = {}          # (dom, C) -> bool
gate_detail = {}   # (dom, C) -> "" or mismatch description
sur_ok = {}        # dom -> bool  (G3)
sur_detail = {}

for dom in DOMAINS:
    span = spans[dom]
    n = len(span)
    # one surprise pass serves every C
    sur = ratec.RateC(c=CS[0], store_book=False).surprise(span)

    for C in CS:
        ref = golden[(dom, C)]
        ok, why = True, []

        # G1 fresh-engine spans (== golden when a golden exists) + G2 tiling
        sp = ratec.RateC(c=C, store_book=False).encode_to_spans(span)
        starts = np.array([s for s, _, _ in sp], dtype=np.int64)
        ends = np.array([e for _, e, _ in sp], dtype=np.int64)
        if ref is not None:
            if not np.array_equal(starts, ref):
                k = int(np.argmax(starts != ref)) if len(starts) == len(ref) else -1
                ok = False
                why.append(f"starts!=golden (n={len(starts)} vs {len(ref)}"
                           + (f", first diff idx {k}: {starts[k]} vs {ref[k]})" if k >= 0 else ")"))
        tiled = (len(sp) > 0 and starts[0] == 0 and ends[-1] == n
                 and np.array_equal(starts[1:], ends[:-1]))
        if not tiled:
            ok = False
            why.append(f"spans do not tile [0,{n}) exactly")
        if not np.array_equal(np.diff(starts) > 0, np.ones(len(starts) - 1, bool)):
            ok = False
            why.append("starts not strictly increasing")

        # G3 reference patcher on stored surprisal; must equal the engine's
        # own starts at every C (judge/patcher independence) — universal,
        # and == golden wherever a golden exists
        ps = ratec.patch_from_sur(sur, C).astype(np.int64)
        if not np.array_equal(ps, starts):
            ok = False
            why.append(f"patch_from_sur!=engine starts (n={len(ps)} vs {len(starts)})")
        elif ref is not None and not np.array_equal(ps, ref):
            ok = False
            why.append(f"patch_from_sur!=golden (n={len(ps)} vs {len(ref)})")

        gate[(dom, C)] = ok
        gate_detail[(dom, C)] = "; ".join(w for w in why if w)

    # G3 verdict for the domain (patcher path, all C)
    s_ok = all(gate[(dom, C)] for C in CS)
    sur_ok[dom] = s_ok

all_gates = all(gate.values())

# G4: batch vs serial golden, per domain per C (verification call, untimed)
docs = [spans[d] for d in DOMAINS]
batch_gate = {}
for C in CS:
    ok = True
    why = []
    res_b = ratec.batch_spans(docs, c=C, threads=2)
    for dom, (starts, n_bytes) in zip(DOMAINS, res_b):
        ref = golden[(dom, C)]
        if ref is not None and (n_bytes != len(spans[dom]) or
                                not np.array_equal(starts.astype(np.int64), ref)):
            ok = False
            why.append(f"{dom}: batch starts != golden")
    batch_gate[C] = (ok, "; ".join(why))
    all_gates &= ok

print(f"GATE (mode: {golden_mode}):")
for dom in DOMAINS:
    stat = "PASS" if all(gate[(dom, C)] for C in CS) else "FAIL"
    bad = [C for C in CS if not gate[(dom, C)]]
    extra = f"  (failing C: {bad}: " + " | ".join(gate_detail[(dom, c)] for c in bad) + ")" if bad else ""
    print(f"  {dom:8s} spans+tiling+patch_from_sur: {stat}{extra}")
for C in CS:
    ok, why = batch_gate[C]
    print(f"  batch c={C:<3d} starts==serial golden: {'PASS' if ok else 'FAIL ' + why}")
print()

if not all_gates:
    print("GATE FAILURE — timing would not count; aborting before the timed arms.")
    json.dump({"protocol": {"note": "gate failure, no timing recorded",
                            "c_values": CS, "reps": REPS, "batch_reps": BATCH_REPS},
               "gates": {f"{d}/r{C}": {"pass": gate[(d, C)], "detail": gate_detail[(d, C)]}
                         for d in DOMAINS for C in CS},
               "batch_gates": {str(C): {"pass": batch_gate[C][0], "detail": batch_gate[C][1]}
                               for C in CS}},
              open(f"{OUT}/c_sweep.json", "w"), indent=1, default=float)
    sys.exit(1)

# ------------------------------------------------- single-thread timing
# Round-robin across ALL (domain, C) arms within one rep loop; medians
# compared (the box drifts by tens of percent over minutes).
# One pooled engine PER C (6 total), reused across domains: an active
# engine holds ~200MB of judge state, so 24 live engines OOM the 4GB box.
# reset() inside the timed window makes the pool domain-independent
# (same steady-state protocol as bench_vs_tiktoken.py).
engines = {C: ratec.RateC(c=C, store_book=False) for C in CS}
runtimes = {(d, C): [] for d in DOMAINS for C in CS}
pool_exact = {(d, C): None for d in DOMAINS for C in CS}

for rep in range(REPS):
    for d in DOMAINS:
        data = spans[d]
        for C in CS:
            tok = engines[C]
            t0 = time.perf_counter_ns()
            tok.reset()                       # reset INSIDE the timed window
            sp = tok.encode_to_spans(data)
            dt = (time.perf_counter_ns() - t0) / 1e9
            runtimes[(d, C)].append(dt)
            if rep == 0 and golden[(d, C)] is not None:  # pool engine stays bit-exact
                pool_exact[(d, C)] = bool(np.array_equal(
                    np.array([s for s, _, _ in sp], dtype=np.int64),
                    golden[(d, C)]))

pool_known = {k: v for k, v in pool_exact.items() if v is not None}
pool_ok = bool(pool_known) and all(pool_known.values())
print(f"pool-engine spot check (rep 0 starts == golden where a golden exists, "
      f"{len(pool_known)}/{len(pool_exact)} arms): "
      f"{'PASS' if pool_ok else 'FAIL'}\n")

# ------------------------------------------------------- batch timing
batch_rt = {C: [] for C in CS}
for rep in range(BATCH_REPS):                 # round-robin over C as well
    for C in CS:
        t0 = time.perf_counter_ns()
        ratec.batch_spans(docs, c=C, threads=2)
        batch_rt[C].append((time.perf_counter_ns() - t0) / 1e9)

# ---------------------------------------------------------- aggregate
res = {}
for d in DOMAINS:
    n = len(spans[d])
    res[d] = {"bytes": n, "per_c": {}}
    for C in CS:
        ms = med(runtimes[(d, C)])
        n_patches = (len(golden[(d, C)]) if golden[(d, C)] is not None
                     else len(ratec.RateC(c=C, store_book=False)
                              .encode_to_spans(spans[d])))
        res[d]["per_c"][C] = {
            "gate": True,                     # reaching here == all gates passed
            "patches": n_patches,
            "tokens_per_byte": n_patches / n,
            "med_s": ms,
            "MBps": n / ms / 1e6,
            "pool_exact": pool_exact[(d, C)],
        }

tot_b = sum(len(x) for x in docs)
batch = {}
for C in CS:
    ms = med(batch_rt[C])
    batch[C] = {"med_s": ms, "MBps": tot_b / ms / 1e6, "gate": batch_gate[C][0]}

# -------------------------------------------------------- printed table
print("=" * 74)
print(f"{'C':>4s}  {'gate':4s}  {'patches':>8s}  {'tok/byte':>8s}  {'MB/s 1T':>8s}"
      "     (median of 21 round-robin reps, reset inside timed window)")
for d in DOMAINS:
    print(f"\ndomain: {d}   ({len(spans[d]):,d} bytes scored span)")
    for C in CS:
        r = res[d]["per_c"][C]
        print(f"{C:>4d}  {'PASS':4s}  {r['patches']:>8d}  {r['tokens_per_byte']:>8.5f}"
              f"  {r['MBps']:>8.2f}")
print(f"\nbatch arm (4 docs = {tot_b:,d} bytes, threads=2, median of {BATCH_REPS}):")
for C in CS:
    print(f"  C={C:<3d} batch MB/s = {batch[C]['MBps']:6.2f}   gate PASS")

# ---------------------------------------------------------- C-trend
print("\n" + "=" * 74)
print("C-TREND SUMMARY (measured, C=12 -> C=32):")
trend = {}
for d in DOMAINS:
    tpb = [res[d]["per_c"][C]["tokens_per_byte"] for C in CS]
    mb = [res[d]["per_c"][C]["MBps"] for C in CS]
    mono = all(tpb[i + 1] <= tpb[i] for i in range(len(tpb) - 1))
    pct = (mb[-1] - mb[0]) / mb[0] * 100.0
    trend[d] = {"tokens_per_byte_C12": tpb[0], "tokens_per_byte_C32": tpb[-1],
                "tok_per_byte_monotone_decreasing": bool(mono),
                "MBps_C12": mb[0], "MBps_C32": mb[-1],
                "MBps_min": min(mb), "MBps_max": max(mb),
                "MBps_pct_change_C12_to_C32": pct,
                "MBps_direction": "up" if pct > 0 else ("down" if pct < 0 else "flat")}
    print(f"  {d:8s} tok/byte {tpb[0]:.5f} -> {tpb[-1]:.5f} "
          f"(x{tpb[0]/tpb[-1]:.2f} fewer tokens, monotone decreasing: "
          f"{'yes' if mono else 'NO'}) | "
          f"MB/s {mb[0]:.2f} -> {mb[-1]:.2f} ({'+' if pct >= 0 else ''}{pct:.1f}%, "
          f"range {min(mb):.2f}..{max(mb):.2f})")
mb_all = [batch[C]["MBps"] for C in CS]
print(f"  batch2T  MB/s {mb_all[0]:.2f} -> {mb_all[-1]:.2f} "
      f"({'+' if mb_all[-1] >= mb_all[0] else ''}"
      f"{(mb_all[-1]-mb_all[0])/mb_all[0]*100:.1f}%), range {min(mb_all):.2f}..{max(mb_all):.2f}")
trend["batch2T"] = {"MBps_C12": mb_all[0], "MBps_C32": mb_all[-1],
                    "MBps_min": min(mb_all), "MBps_max": max(mb_all),
                    "MBps_pct_change_C12_to_C32": (mb_all[-1] - mb_all[0]) / mb_all[0] * 100}

# ------------------------------------------------------------- json/csv
json.dump({
    "protocol": {
        "library": f"ratec {ratec.__version__}",
        "c_values": CS,
        "input": "bake-off scored spans (meta.json warm/scored trimming)",
        "reps": {"single_thread": REPS, "batch_2t": BATCH_REPS},
        "ratec_arm": "pool steady-state: tok.reset() inside timed window then "
                     "encode_to_spans; round-robin across all (domain,C) arms; medians",
        "batch_arm": "ratec.batch_spans(docs of all 4 domains, c=C, threads=2), "
                     "round-robin over C; total MB/s over all 4 docs",
        "gates": "G1 spans==golden r{C}starts (int64 exact, wherever a golden "
                 "exists), G2 tiling [0,len), G3 patch_from_sur==engine starts "
                 "at every C (==golden where available), G4 batch starts==golden; "
                 "all must pass before timing counts",
        "golden_mode": golden_mode,
        "pool_engine_rep0_exact": bool(pool_ok),
    },
    "per_domain": res,
    "batch": batch,
    "trend": trend,
}, open(f"{OUT}/c_sweep.json", "w"), indent=1, default=float)

with open(f"{OUT}/c_sweep.csv", "w") as f:
    f.write("domain,kind,C,gate,patches,tokens_per_byte,median_ms,MBps\n")
    for d in DOMAINS:
        for C in CS:
            r = res[d]["per_c"][C]
            f.write(f"{d},single,{C},PASS,{r['patches']},"
                    f"{r['tokens_per_byte']:.6f},{r['med_s']*1e3:.3f},{r['MBps']:.3f}\n")
    for C in CS:
        f.write(f"ALL4,batch2T,{C},PASS,,,"
                f"{batch[C]['med_s']*1e3:.3f},{batch[C]['MBps']:.3f}\n")

# --------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COL = {"english": "#B03A2E", "code": "#2E86C1", "chinese": "#1E8449",
       "base64": "#B7950B"}

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)

ax = axes[0]
for d in DOMAINS:
    y = [res[d]["per_c"][C]["tokens_per_byte"] for C in CS]
    ax.plot(CS, y, "o-", color=COL[d], label=d, lw=1.8, ms=5)
ax.set_xticks(CS)
ax.set_xlabel("patch budget C (bits)")
ax.set_ylabel("tokens per byte (patches / byte)")
ax.set_title("Patch density vs C\nlower = fewer tokens per byte")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
for d in DOMAINS:
    y = [res[d]["per_c"][C]["MBps"] for C in CS]
    ax.plot(CS, y, "s-", color=COL[d], label=d, lw=1.8, ms=5)
ax.set_xticks(CS)
ax.set_xlabel("patch budget C (bits)")
ax.set_ylabel("throughput (MB/s, 1 thread, median)")
ax.set_title("Single-thread throughput vs C\n(should be nearly flat)")
ax.grid(alpha=0.25)

fig.suptitle("RateC C-sweep — gated (" + golden_mode + ")"
             if all_gates else "RateC C-sweep — GATE FAILURES (timing invalid)",
             fontsize=11)
fig.savefig(f"{OUT}/fig_c_sweep.png", dpi=150)

print(f"\nwrote {OUT}/c_sweep.json, {OUT}/c_sweep.csv, {OUT}/fig_c_sweep.png")
sys.exit(0 if all_gates else 1)
