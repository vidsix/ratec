#!/usr/bin/env python3
"""test_profiles.py — RateC modality profiles: lossless transforms + wins.

Run:  python3 tests/test_profiles.py
Every check prints PASS/FAIL; exit code 0 iff all pass.

Checks:
  a. restore(apply(profile, data)) == data, bit-for-bit, on the real
     ascent.raw asset + synthetic stereo PCM16 sine and float32 randn
  b. same judge budget (C=20): transformed bytes need FEWER patches than raw
     (delta+deinterleave for pcm16, byte planes for f32, row delta for image)
  c. malformed headers / odd lengths raise ValueError
  d. registry API: 4 names, apply/restore accept str or Profile
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ratec
from ratec import profiles
from ratec.profiles import PROFILES, Profile, apply, restore, list_profiles

FAILURES = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILURES.append(name)


def n_patches(data, c=20):
    sur = ratec.RateC(c=c, store_book=False).surprise(data)
    return len(ratec.patch_from_sur(sur, c))


def raises_ve(fn):
    try:
        fn()
        return False
    except ValueError:
        return True
    except Exception:
        return False


DATA = "/home/z/my-project/bakeoff_data"

# ------------------------------------------------------------------- assets
img = open(os.path.join(DATA, "ascent.raw"), "rb").read()
img_width = 512 if len(img) % 512 == 0 else 1   # spec rule: 512 if divisible else 1-row
img_prof = (PROFILES["u8_image"] if img_width == 512
            else profiles.make_u8_image(img_width))

sr = 8000                                       # 1s of 8kHz stereo sine, PCM16
t = np.arange(sr, dtype=np.float64) / sr
ch0 = (18000.0 * np.sin(2 * np.pi * 440.0 * t)).astype("<i2")
ch1 = (18000.0 * np.sin(2 * np.pi * 554.37 * t + 0.6)).astype("<i2")
stereo = np.empty(sr * 2, dtype="<i2")
stereo[0::2], stereo[1::2] = ch0, ch1
pcm = stereo.tobytes()
mono = ch0.tobytes()

f32data = np.random.default_rng(0).standard_normal(4096).astype("<f4").tobytes()
weird = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0,
                  np.finfo(np.float32).tiny, np.finfo(np.float32).max,
                  1e-45, -1e-45, 3.14e-40], dtype="<f4").tobytes()

# --------------------------------------------------- a. round trips (exact)
print("== a. round trips (bit-exact) ==")
assets = [
    ("pcm16 stereo sine 1s@8kHz", PROFILES["pcm16"], pcm),
    ("pcm16 mono via with_channels(1)", PROFILES["pcm16"].with_channels(1), mono),
    ("f32 randn(4096)", PROFILES["f32"], f32data),
    ("f32 NaN/inf/-0.0/denormals", PROFILES["f32"], weird),
    (f"u8_image ascent.raw ({len(img)}B, width={img_width})", img_prof, img),
    ("u8_image width=1 (single row)", profiles.make_u8_image(1), img),
    ("raw identity (mixed bytes)", PROFILES["raw"], pcm + f32data + img[:1024]),
]
for name, prof, data in assets:
    check(f"{name}: restore(apply(p, data)) == data",
          restore(prof, apply(prof, data)) == data)
for name in list_profiles():
    check(f"{name}: empty input round-trips", restore(name, apply(name, b"")) == b"")

# exact wire-format conformance (magic / header / plane order)
check("pcm16 header: b'RCP1' + u8 channels=2", apply("pcm16", pcm)[:5] == b"RCP1\x02")
one = struct.pack("<hh", 1000, -1000)
check("pcm16 micro: planes sequential, first sample of each raw",
      apply("pcm16", one) == b"RCP1\x02" + one)
check("f32 header: b'RCF1' + u32 count=4096",
      apply("f32", f32data)[:8] == b"RCF1" + struct.pack("<I", 4096))
check("f32 micro: 1.0 -> sign+exponent plane first (3f 80 00 00)",
      apply("f32", struct.pack("<f", 1.0)) ==
      b"RCF1\x01\x00\x00\x00" + b"\x3f\x80\x00\x00")
grid = bytes([1, 2, 5, 1])                      # 2x2 raster
check("u8_image micro: per-row previous-pixel delta",
      apply(profiles.make_u8_image(2), grid) ==
      b"RCI1" + struct.pack("<II", 2, 2) + bytes([1, 1, 5, 252]))
check("u8_image header: w/h match asset",
      apply(img_prof, img)[:12] ==
      b"RCI1" + struct.pack("<II", img_width, len(img) // img_width))
p1 = PROFILES["pcm16"].with_channels(1)
check("with_channels(1) returns a NEW Profile bound to mono",
      isinstance(p1, Profile) and p1 is not PROFILES["pcm16"] and
      apply(p1, mono)[:5] == b"RCP1\x01")
check("with_channels does not mutate the registry",
      apply("pcm16", pcm)[:5] == b"RCP1\x02")
check("with_channels on non-pcm16 raises", raises_ve(lambda: PROFILES["raw"].with_channels(1)))

# ------------------------------------------------------------- b. wins
print("== b. predictability: patch counts at C=20 (identical judge budget) ==")
counts = {}
for label, prof, data in [("pcm16", PROFILES["pcm16"], pcm),
                          ("f32", PROFILES["f32"], f32data),
                          ("u8_image", img_prof, img)]:
    n_raw, n_tr = n_patches(data), n_patches(apply(prof, data))
    counts[label] = (n_raw, n_tr)
    pct = 100.0 * (n_raw - n_tr) / n_raw
    print(f"  [info] {label:9s}: {n_raw:7d} -> {n_tr:7d} patches ({pct:+.1f}%)")
check("pcm16: delta + deinterleave reduces patch count",
      counts["pcm16"][1] < counts["pcm16"][0])
check("f32: byte-plane separation reduces patch count",
      counts["f32"][1] < counts["f32"][0])
check("u8_image: per-row delta reduces patch count",
      counts["u8_image"][1] < counts["u8_image"][0])
sample = open(os.path.join(DATA, "english.txt"), "rb").read()[32768:32768 + 20000]
check("raw: identity keeps patch count identical",
      n_patches(apply("raw", sample)) == n_patches(sample))

# ------------------------------------------------------- c. error handling
print("== c. error handling (ValueError on malformed input) ==")
check("pcm16 apply: odd byte count raises",
      raises_ve(lambda: apply("pcm16", b"\x01\x02\x03")))
check("pcm16 apply: samples not divisible by channels raises",
      raises_ve(lambda: apply("pcm16", b"\x00" * 6)))
check("pcm16 restore: bad magic raises",
      raises_ve(lambda: restore("pcm16", b"XXXX\x02\x00\x00")))
check("pcm16 restore: truncated header raises",
      raises_ve(lambda: restore("pcm16", b"RCP1")))
check("pcm16 restore: payload not whole channel planes raises",
      raises_ve(lambda: restore("pcm16", b"RCP1\x02" + b"\x00" * 6)))
check("f32 apply: byte count % 4 raises",
      raises_ve(lambda: apply("f32", b"\x00" * 5)))
check("f32 restore: bad magic raises",
      raises_ve(lambda: restore("f32", b"RCF0" + b"\x00" * 4)))
check("f32 restore: truncated header raises",
      raises_ve(lambda: restore("f32", b"RCF1\x01\x00")))
check("f32 restore: payload != 4*count raises",
      raises_ve(lambda: restore("f32", b"RCF1" + struct.pack("<I", 2) + b"\x00" * 5)))
check("u8_image apply: length not multiple of width raises",
      raises_ve(lambda: apply("u8_image", b"\x00" * 513)))
check("u8_image restore: bad magic raises",
      raises_ve(lambda: restore("u8_image", b"RCI0" + b"\x00" * 8)))
check("u8_image restore: truncated header raises",
      raises_ve(lambda: restore("u8_image", b"RCI1\x01\x00")))
check("u8_image restore: payload != w*h raises",
      raises_ve(lambda: restore("u8_image", b"RCI1" + struct.pack("<II", 4, 4) + b"\x00" * 7)))
check("unknown profile name raises ValueError (apply and restore)",
      raises_ve(lambda: apply("wav", b"x")) and raises_ve(lambda: restore("nope", b"x")))

# ----------------------------------------------------------- d. registry API
print("== d. registry API ==")
names = list_profiles()
check("list_profiles() has exactly the 4 names",
      set(names) == {"raw", "pcm16", "f32", "u8_image"} and len(names) == 4)
check("PROFILES values are Profile instances",
      all(isinstance(v, Profile) for v in PROFILES.values()))
check("apply/restore accept a str name",
      apply("raw", b"abc") == b"abc" and restore("raw", b"abc") == b"abc")
check("apply/restore accept a Profile instance",
      apply(PROFILES["f32"], f32data)[:4] == b"RCF1" and
      restore(PROFILES["f32"], apply(PROFILES["f32"], f32data)) == f32data)
check("bytearray input accepted",
      apply("pcm16", bytearray(pcm)) == apply("pcm16", pcm))
check("every profile carries c (None = library default) and note",
      all(p.c is None or isinstance(p.c, float) for p in PROFILES.values()) and
      all(isinstance(p.note, str) and p.note for p in PROFILES.values()))

# ------------------------------------------------------------------- verdict
print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):", *FAILURES, sep="\n  - ")
    sys.exit(1)
print("ALL TESTS PASS")
