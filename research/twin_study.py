#!/usr/bin/env python3
"""Twin study: same LM, same bytes, three tokenizations — BPE vs RateC vs raw bytes.

The number that separates "excellent library" from "important result" (the
standing rent since the v0.3 LM track opened). Protocol:

  Identical transformer  d=96, layers=3, heads=4, ctx=256, batch=16,
                         steps=600, AdamW(3e-4, b0.9/b0.95, wd 0.1),
                         OneCycle, seed 42, 2 torch threads.
  Identical byte budget  every arm trains on the same 15,811,609-byte mixed
                         corpus and evaluates on the same 659,473-byte
                         held-out stream (en/hr/zh/code).
  Three arms, V=9472 for bpe/ratec (256 + 9*2^10), V=256 for bytes:
    bpe    tied embedding/head over 9472 rows.
    ratec  factorized input per the spec: content rows from a (256 + 2^C)
           table (C=10; raw-byte block shares rows 0..255), length embedding
           (2+C rows @ 32), concat -> project; untied output head over 9472.
           Tokens are RateC C=10 patch ids; (id, byte_len) contract.
    bytes  tied embedding/head over 256 rows — the raw-byte floor.

  Compute accounting is compute-matched (same tokens/step x steps), which is
  the deployment-relevant cut: the tokenizers differ in bytes-per-token, so
  the same FLOPs buy each arm a different number of unique bytes. Params and
  train-FLOPs are reported per arm; BPB is the tokenization-agnostic metric.

  Eval (the parity fix): the previous harness subsampled each arm's token
  stream independently, so the two arms scored different byte subsets
  (187k vs 389k of 659k). Here every val byte is scored exactly once,
  byte-weighted, per domain — identical held-out bytes for all arms.

Outputs: runs/twin_{kind}.json and runs/twin_summary.json.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tokenized streams (.npy) and checkpoints (.pt) are produced outside the
# repo and are large; override both to rerun. The committed research/*.json
# are the result records for the seed-42 run documented in this docstring.
# Defaults are REPO-relative (not CWD-relative — a run from anywhere finds
# the same layout; PLAN_v0.6 P2-6).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RUNS = os.environ.get("RATEC_RUNS", os.path.join(_ROOT, "runs"))
DATA = os.environ.get("RATEC_TWIN_DATA", os.path.join(_ROOT, "data"))
DOMAINS = ["en", "hr", "zh", "code"]
V = 256 + 9 * (2 ** 10)     # 9472; length rows cap patches at 9 bytes
                            # (len_idx <= 9) — a deliberate divergence from
                            # the library patcher, which can emit longer
                            # patches; asserted on every loaded stream
C = 10                      # ratec arm patch budget (2^C content rows)


# ------------------------------------------------------------------ streams
def load_streams(prefix):
    tr_ids, tr_lens, va_ids, va_lens, bounds = [], [], [], [], [0]
    for d in DOMAINS:
        tr_ids.append(np.load(f"{RUNS}/{prefix}_{d}_train_ids.npy"))
        tr_lens.append(np.load(f"{RUNS}/{prefix}_{d}_train_lens.npy"))
        vi = np.load(f"{RUNS}/{prefix}_{d}_val_ids.npy")
        vl = np.load(f"{RUNS}/{prefix}_{d}_val_lens.npy")
        va_ids.append(vi); va_lens.append(vl)
        bounds.append(bounds[-1] + len(vi))
    return (np.concatenate(tr_ids), np.concatenate(tr_lens),
            va_ids, va_lens, bounds)


def load_bytes_streams():
    """bytes arm: ids are the raw bytes, lens all 1."""
    tr = np.frombuffer(open(f"{DATA}/_train_mix.txt", "rb").read(), dtype=np.uint8)
    tr_ids = tr.astype(np.int64)
    tr_lens = np.ones(len(tr_ids), dtype=np.int64)
    va_ids, va_lens, bounds = [], [], [0]
    for d in DOMAINS:
        raw = open(f"{DATA}/{d}_val.txt", "rb").read()
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        va_ids.append(ids)
        va_lens.append(np.ones(len(ids), dtype=np.int64))
        bounds.append(bounds[-1] + len(ids))
    return tr_ids, tr_lens, va_ids, va_lens, bounds


# -------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, -1).transpose(1, 2)
        k = k.view(B, T, self.h, -1).transpose(1, 2)
        v = v.view(B, T, self.h, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf"))
        x = x + self.proj((F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TwinLM(nn.Module):
    def __init__(self, kind, d=96, layers=3, heads=4, ctx=256,
                 c_eff=None, v_eff=None, lens_lidx=False):
        super().__init__()
        self.kind = kind
        self.ctx = ctx
        self.c_eff = C if c_eff is None else c_eff
        self.v_eff = V if v_eff is None else v_eff
        self.lens_lidx = lens_lidx
        self.vocab = {"bpe": self.v_eff, "ratec": self.v_eff, "bytes": 256}[kind]
        if kind == "bpe":
            self.emb = nn.Embedding(self.v_eff, d)
            self.in_proj = None
        elif kind == "bytes":
            self.emb = nn.Embedding(256, d)
            self.in_proj = None
        else:
            self.ec = nn.Embedding(256 + (2 ** self.c_eff), d)  # bytes + content rows
            self.el = nn.Embedding(2 + self.c_eff, 32)          # length field
            self.in_proj = nn.Linear(d + 32, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.vocab, bias=False)
        if kind in ("bpe", "bytes"):
            self.head.weight = self.emb.weight             # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def content_len_idx(self, ids, lens=None):
        m = 2 ** self.c_eff
        content = torch.where(ids < 256, ids, 256 + (ids - 256) % m)
        if self.lens_lidx:
            # dense-id streams: the length field comes from the token's
            # true byte length (the documented (id, byte_len) contract),
            # not from id arithmetic — which is only well-defined for the
            # shipped packed-id streams. lens: byte length per input token.
            cap = torch.full_like(ids, self.c_eff)
            lidx = torch.where(ids < 256, torch.zeros_like(ids),
                               torch.clamp(lens - 1, min=0) .clamp(max=cap))
            return content, lidx
        lidx = torch.where(ids < 256, torch.zeros_like(ids), 1 + (ids - 256) // m)
        return content, lidx

    def forward(self, idx, lens=None):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        if self.kind == "ratec":
            c, l = self.content_len_idx(idx, lens)
            x = self.in_proj(torch.cat([self.ec(c), self.el(l)], dim=-1)) + self.pos(pos)[None]
        else:
            x = self.emb(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------- eval
@torch.no_grad()
def eval_full(model, va_flat, va_lens, bounds, ctx):
    """Byte-weighted BPB over every val byte EXCEPT each domain's first
    token (that token has no in-domain predecessor to predict it from, so
    it is never a target — see the coverage assert in main()). Bytes are
    counted via lens. va_flat: concatenated val token ids; bounds:
    per-domain token boundaries into it; va_lens: list of per-domain
    byte-length arrays."""
    model.eval()
    overall_bits = overall_bytes = 0.0
    per = {}
    covered = 0
    for di, d in enumerate(DOMAINS):
        b0, b1 = bounds[di], bounds[di + 1]
        bits = byt = 0.0
        s = b0
        while s < b1 - 1:
            T = min(ctx, b1 - s - 1)
            chunk = va_flat[s:s + T + 1]
            x = torch.from_numpy(chunk[:-1].astype(np.int64))[None]
            y = torch.from_numpy(chunk[1:].astype(np.int64))[None]
            xl = torch.from_numpy(
                va_lens[di][s - b0:s - b0 + T].astype(np.int64))[None]
            logp = F.log_softmax(model(x, xl)[0], dim=-1)
            nll = -logp[torch.arange(T), y[0]] / math.log(2)
            bits += float(nll.sum())
            # byte weight = lens of the TARGET POSITIONS, domain-LOCAL
            # (s is a flat-stream position; lens arrays are per-domain)
            lo = s - b0
            byt += int(va_lens[di][lo + 1:lo + T + 1].sum())
            s += T
        per[d] = bits / max(1.0, byt)
        covered += int(byt)
        overall_bits += bits
        overall_bytes += byt
    model.train()
    return overall_bits / max(1.0, overall_bytes), per, covered


@torch.no_grad()
def eval_quick(model, ids, va_ids, va_lens, ctx, max_chunks=200):
    """Subsampled BPB for the training curve only (never reported as final)."""
    model.eval()
    flat = np.concatenate(va_ids)
    flat_lens = np.concatenate(va_lens)
    tot_bits = tot_bytes = 0.0
    step = max(1, (len(flat) - 1) // (max_chunks * ctx))
    for s in range(0, len(flat) - ctx - 1, ctx * step if step > 1 else ctx):
        chunk = flat[s:s + ctx + 1]
        if len(chunk) < ctx + 1:
            break
        x = torch.from_numpy(chunk[:-1].astype(np.int64))[None]
        y = torch.from_numpy(chunk[1:].astype(np.int64))[None]
        xl = torch.from_numpy(flat_lens[s:s + ctx].astype(np.int64))[None]
        logp = F.log_softmax(model(x, xl)[0], dim=-1)
        nll = -logp[torch.arange(ctx), y[0]] / math.log(2)
        tot_bits += float(nll.sum())
        tot_bytes += int(flat_lens[s + 1:s + ctx + 1].sum())
    model.train()
    return tot_bits / max(1.0, tot_bytes)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["bpe", "ratec", "bytes", "ratec_direct"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="",
                    help="output suffix, e.g. --tag _s2400 -> runs/twin_bpe_s2400.json "
                         "(empty keeps the canonical 600-step artifact names)")
    ap.add_argument("--ckpt_every", type=int, default=50,
                    help="save a resumable checkpoint every N steps "
                         "(runs/twin_{kind}{tag}.ckpt)")
    ap.add_argument("--c", type=int, default=C,
                    help="ratec patch-budget C for the sweep (default 10; "
                         "scales the 2^C content rows and the V table)")
    ap.add_argument("--prefix", default="",
                    help="stream filename prefix override (default: "
                         "bpe_mix for bpe, ratec_main otherwise)")
    ap.add_argument("--lens_lidx", action="store_true",
                    help="dense-id stream interface: length field from the "
                         "token's true byte length instead of id arithmetic "
                         "(required when ids are dense engine book ids)")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    if args.kind == "bytes":
        tr_ids, tr_lens, va_ids, va_lens, bounds = load_bytes_streams()
    else:
        prefix = args.prefix or ("bpe_mix" if args.kind == "bpe" else "ratec_main")
        tr_ids, tr_lens, va_ids, va_lens, bounds = load_streams(prefix)
    c_eff = int(args.c)
    if args.lens_lidx:
        # dense-id streams: the head table must cover the book's id range
        # (there is no packed length-slab structure to size it from)
        v_eff = max(256 + 9 * (2 ** c_eff),
                    int(tr_ids.max()) + 1,
                    max(int(a.max()) for a in va_ids) + 1)
    else:
        v_eff = 256 + 9 * (2 ** c_eff)
    train_bytes = int(tr_lens.sum())
    val_bytes = int(sum(v.sum() for v in va_lens))
    val_tokens = int(sum(len(v) for v in va_ids))
    print(f"[{args.kind}] train {len(tr_ids):,} tokens / {train_bytes:,} B | "
          f"val {val_tokens:,} tokens / {val_bytes:,} B", flush=True)

    # byte parity across arms is the whole point — assert it
    assert train_bytes == 15_811_609, train_bytes
    assert val_bytes == 659_473, val_bytes

    # patch-id range: the length-row cap means no stream id may reach V
    # (would be an out-of-range embedding lookup anyway — assert the
    # producer honored the documented <=9-byte patch cap)
    # patch-id range: the length-row cap means no stream id may reach V
    # (would be an out-of-range embedding lookup anyway — assert the
    # producer honored the documented <=9-byte patch cap). Dense-id runs
    # (lens_lidx) size V_eff from the id range itself, so no assert.
    if args.kind != "bytes" and not args.lens_lidx:
        va_max = max(int(a.max()) for a in va_ids)
        assert int(tr_ids.max()) < v_eff and va_max < v_eff, \
            "stream id >= V: patch-length cap violated"

    interface = {
        "bpe": f"tied embedding/head over {v_eff} rows",
        "ratec": f"factorized input (256+2^{c_eff} content rows, {2+c_eff} length rows @32, "
                 f"in_proj) + untied head over {v_eff}; C={c_eff} patch ids",
        "ratec_direct": f"tied embedding/head over {v_eff} rows on the SAME RateC "
                        f"C={c_eff} patch ids as 'ratec' (isolates interface cost)",
        "bytes": "tied embedding/head over 256 rows (raw-byte floor)",
    }[args.kind]
    if args.lens_lidx:
        interface += ("; dense-id streams: content slot = (id-256) mod 2^C, "
                      "length field from true byte length")

    model_kind = "bpe" if args.kind == "ratec_direct" else args.kind
    model = TwinLM(model_kind, d=args.d, layers=args.layers, ctx=args.ctx,
                   c_eff=c_eff, v_eff=v_eff, lens_lidx=args.lens_lidx)
    params = n_params(model)
    print(f"[{args.kind}] params {params:,}  interface: {interface}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1,
        div_factor=10, final_div_factor=10)

    rng = np.random.default_rng(42)
    curve = []
    start_step = 0
    ckpt_path = f"{RUNS}/twin_{args.kind}{args.tag}.ckpt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        torch.set_rng_state(ck["torch_rng"])
        rng.bit_generator.state = ck["np_rng"]
        curve = ck.get("curve", [])
        start_step = ck["step"]
        print(f"[{args.kind}] resumed from {ckpt_path} at step {start_step}",
              flush=True)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "np_rng": rng.bit_generator.state, "curve": curve},
                   ckpt_path)

    t0 = time.time()
    try:
        for step in range(start_step + 1, args.steps + 1):
            offs = rng.integers(0, len(tr_ids) - args.ctx - 1, size=args.batch)
            x = torch.stack([torch.from_numpy(
                tr_ids[o:o + args.ctx].astype(np.int64)) for o in offs])
            y = torch.stack([torch.from_numpy(
                tr_ids[o + 1:o + args.ctx + 1].astype(np.int64)) for o in offs])
            xl = (torch.from_numpy(np.stack(
                [tr_lens[o:o + args.ctx] for o in offs]).astype(np.int64))
                if args.lens_lidx else None)
            logits = model(x, xl)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 50 == 0 or step == 1:
                bpb = eval_quick(model, None, va_ids, va_lens, args.ctx)
                curve.append({"step": step, "val_bpb_quick": bpb})
                el = time.time() - t0
                print(f"  step {step}/{args.steps} loss {float(loss):.4f} "
                      f"quickBPB {bpb:.4f} ({el:.0f}s)", flush=True)
            if step % args.ckpt_every == 0 or step == args.steps:
                save_ckpt(step)
    except KeyboardInterrupt:
        save_ckpt(step)
        print(f"[{args.kind}] interrupted at step {step} — checkpoint saved "
              f"({ckpt_path}); rerun with the same args to resume", flush=True)
        raise SystemExit(2)

    wall = time.time() - t0
    va_flat = np.concatenate(va_ids)
    bpb, per_dom, covered = eval_full(model, va_flat, va_lens, bounds, args.ctx)
    # coverage is exact-by-construction but assert it, never assume it:
    # every val byte except each domain's first TOKEN is a target exactly
    # once — the first token has no in-domain predecessor, and ALL of its
    # bytes are uncovered (ratec val streams can open with a multi-byte
    # patch; the v0.6.0 assert assumed 1-byte first tokens and falsely
    # failed such arms — caught by the first ladder execution, v0.7.0)
    first_tok_bytes = int(sum(int(v[0]) for v in va_lens))
    assert covered == val_bytes - first_tok_bytes, \
        f"coverage hole: {covered} of {val_bytes} bytes covered " \
        f"(expected exactly {val_bytes - first_tok_bytes})"
    nominal_tokens = args.steps * args.batch * args.ctx
    res = {
        "kind": args.kind, "interface": interface,
        "d": args.d, "layers": args.layers, "ctx": args.ctx,
        "batch": args.batch, "steps": args.steps, "lr": args.lr, "seed": 42,
        "params": params, "vocab_effective": model.vocab,
        "c_eff": c_eff, "v_eff": v_eff, "lens_lidx": bool(args.lens_lidx),
        "stream_prefix": prefix if args.kind != "bytes" else "(raw bytes)",
        "train_bytes": train_bytes, "train_tokens": len(tr_ids),
        "val_bytes": val_bytes, "val_tokens": val_tokens,
        "val_bytes_per_token": val_bytes / val_tokens,
        "train_bytes_per_ctx": train_bytes / len(tr_ids) * args.ctx,
        "final_bpb_overall": bpb,
        "final_bpb_per_domain": per_dom,
        "val_bytes_covered": covered,
        "val_coverage": covered / val_bytes,
        "val_coverage_note": "every val byte except each domain's first "
                             "token is a target exactly once (that token "
                             "has no in-domain predecessor); all of its "
                             "bytes are uncovered — asserted EXACTLY in "
                             "code (val_bytes - first-token bytes)",
        "train_flops_est": 6 * params * nominal_tokens,
        "flops_per_train_byte_est": 6 * params * nominal_tokens / train_bytes,
        "wall_sec": wall, "curve": curve,
        "protocol": {
            "compute_matched": "same d/layers/ctx/batch/steps/lr/seed; arms "
                               "differ in bytes-per-token so the same FLOPs "
                               "buy different unique-byte coverage",
            "byte_budget": "identical 15,811,609-B train corpus and "
                           "659,473-B val stream for all arms",
            "eval": "byte-weighted BPB on the identical val bytes; every "
                    "val byte except each domain's first token is a target "
                    "exactly once (coverage asserted in code)",
            "patch_len_cap": "ratec arms cap patches at 9 bytes (length "
                             "rows 2^10, len_idx<=9); asserted max(id) < V",
        },
    }
    out = f"{RUNS}/twin_{args.kind}{args.tag}.json"
    torch.save(model.state_dict(), f"{RUNS}/twin_{args.kind}{args.tag}.pt")
    json.dump(res, open(out, "w"), indent=2)
    print(f"[{args.kind}] FINAL BPB {bpb:.4f}  per-domain "
          + " ".join(f"{k}={v:.3f}" for k, v in per_dom.items())
          + f"  params {params:,}  wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
