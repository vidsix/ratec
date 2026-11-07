# RateC container format — spec v0.1 (draft)

Status: **draft**. Nothing implements this yet; `ratec.profiles` supplies
the transforms. The spec is written down now so implementations (and the
LM-side `(id, byte_len, modality)` contract) can be built against
something stable, and so "ratec-encoded" artifacts become as portable as
a codec.

## Design rules

1. **Lossless in the core.** A conforming decoder reproduces the exact
   input bytes. Any lossy step (quantize → higher effective C) is an
   upstream, explicitly declared transform — never the container's.
2. **Profiles are data.** The header carries a profile id; the decoder
   looks up the transform in the registry; the RateC core never branches
   on modality.
3. **Spans are the universal interface.** Patch k = `[start_k, start_{k+1})`
   byte offsets over the *transformed* byte stream; `(id, byte_len)`
   is what an LM consumes.

## Single-stream container (magic `RCCT`)

```
offset  size  field
0       4     magic "RCCT"
4       2     version (u16 LE, = 1)
6       2     profile id (u16 LE; registry index; 0xFFFF = raw/identity)
8       8     C (f64 LE) — patch budget used for this stream
16      32    snapshot hash (sha256 of the frozen book snapshot;
              all-zero = bookless stream)
48      8     payload length (u64 LE)
56      ...   payload (see modes)
```

Payload modes (flag bit in `version`: 0 = spans, 1 = ids):

- **Spans mode (bookless):** `u32 count`, then `count` u32 LE patch
  starts (first = 0 implied by construction? NO — store all starts;
  the payload ends at the transformed length, which the container does
  not store: the spans themselves reconstruct it from the last start +
  the final patch, so spans mode MUST also store `u32 total_len` after
  the starts). Decoder re-derives contents by transforming nothing —
  contents are the transformed bytes themselves; for spans mode the
  payload IS `total_len` bytes of transformed data followed by the
  starts array (self-contained, no book needed).
- **Ids mode (frozen book):** `u32 count`, then `count` u32 LE patch
  ids into the frozen book pinned by the snapshot hash. Compact; the
  LM-consumable form is `(id, len)` with lens from the book.

## Multiplexed document (magic `RCCM`)

```
0       4     magic "RCCM"
4       2     version (= 1)
6       2     stream count S
8       ...   S × stream header: {u16 profile_id, u8 flags, u8 reserved,
              f64 C, u64 payload_len, u32 name_len, name bytes}
...     ...   concatenated sub-payloads (each an RCCT payload body,
              without the RCCT header)
```

Mixed documents (text + base64 + PCM in one file) become one mux
container; per-stream C is explicit. The LM-side record for each patch
is then `(book_id, byte_len, profile_id)` — the modality tag costs one
u16 per stream, not per patch.

## Versioning rules

- Container `version` bump = breaking change; decoders must reject.
- Profile registry additions are non-breaking; an unknown profile id is
  a hard error (never silently decode as raw).
- The snapshot hash pins frozen-book identity end-to-end: same hash ⇒
  same ids, bit-exact across implementations (the tokenizer is
  deterministic given the snapshot; the patcher is C-exact).
