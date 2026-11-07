"""ratec.profiles — modality profiles: data-declared, lossless byte transforms.

Architecture rule: the RateC core (api.py / _rcengine.c) stays UNTOUCHED and
lossless — it is byte-generic and never learns what the bytes mean. Modality
support ships purely as data: the frozen `Profile` dataclass below plus the
module-level `PROFILES` registry. A profile pairs an information-preserving
byte `transform` (applied upstream, before tokenization) with its exact
inverse `untransform` (applied downstream), so the stream the judge scores
changes shape but carries identical information; `restore(apply(p, x)) == x`
holds bit-for-bit for every profile here.

Why transform at all — RateC's 6-order byte contexts must align with structure:
  * pcm16: raw LE int16 audio interleaves channels and sign bytes, so byte
    contexts mix two signals and both byte roles. Deinterleave into channel
    planes, then delta-code each plane as int16 (first sample stored raw):
    deltas of real audio are small and smooth.
  * f32: raw LE float32 interleaves a smooth sign+exponent plane with two
    near-uniform mantissa planes. Byte-plane separation (plane 3 first, then
    2, 1, 0) gives every plane homogeneous contexts.
  * u8_image: neighbouring raster pixels are highly correlated; a
    previous-pixel delta per row (first pixel of each row raw) concentrates
    the mass and lets short contexts predict.

All transforms are pure stdlib + numpy, accept bytes/bytearray, raise
ValueError on malformed input (odd lengths, bad magic, wrong payload size),
and round-trip exactly. The registry is opt-in at the call site:

    from ratec import profiles
    payload = profiles.apply("pcm16", pcm_bytes)
    ids = tok.encode_bytes(payload)          # RateC core, unchanged
    assert profiles.restore("pcm16", payload) == pcm_bytes
"""
from __future__ import annotations

import dataclasses
import struct
from typing import Callable

import numpy as np

__all__ = ["Profile", "PROFILES", "apply", "restore", "list_profiles",
           "make_pcm16", "make_u8_image"]


# ----------------------------------------------------------- profile record

@dataclasses.dataclass(frozen=True)
class Profile:
    """A named modality: an exact transform/untransform pair plus metadata.

    `c` is an optional suggested patch budget (None = library default 20);
    `note` documents the wire format produced by `transform`.
    """

    name: str
    transform: Callable[[bytes], bytes]    # bytes -> bytes (information-preserving)
    untransform: Callable[[bytes], bytes]  # exact inverse of transform
    c: float | None = None
    note: str = ""

    def with_channels(self, channels: int) -> "Profile":
        """pcm16 only: return a NEW Profile bound to a different channel count."""
        if self.name != "pcm16":
            raise ValueError("with_channels() is defined only for the 'pcm16' "
                             f"profile, not {self.name!r}")
        return make_pcm16(channels)


def _identity(data: bytes) -> bytes:
    return data


# ------------------ pcm16: LE int16 PCM, deinterleave + per-channel delta

_PCM_MAGIC = b"RCP1"


def _pcm16_encode(data: bytes, channels: int) -> bytes:
    if channels < 1:
        raise ValueError(f"pcm16: channels must be >= 1, got {channels}")
    if len(data) % 2:
        raise ValueError(f"pcm16: odd byte count {len(data)} (int16 samples)")
    n = len(data) // 2
    if n % channels:
        raise ValueError(f"pcm16: {n} samples not divisible by channels={channels}")
    x = np.frombuffer(data, dtype="<i2").astype(np.int64).reshape(-1, channels)
    d = np.empty_like(x)
    d[:1, :] = x[:1, :]                # first sample of each channel: raw
    d[1:, :] = x[1:, :] - x[:-1, :]    # per-channel-stream delta
    # mod 65536 == two's-complement bit pattern of int16(delta); storing as
    # "<u2" bytes is byte-identical to storing the signed "<i2" value
    planes = (d % 65536).astype("<u2")             # (n_samples, channels)
    return _PCM_MAGIC + bytes([channels]) + planes.T.tobytes()


def _pcm16_decode(data: bytes) -> bytes:
    if len(data) < 5 or bytes(data[:4]) != _PCM_MAGIC:
        raise ValueError("pcm16: bad header (expected b'RCP1' + u8 channels)")
    channels = data[4]
    if channels < 1:
        raise ValueError(f"pcm16: invalid channel count {channels}")
    payload = bytes(data[5:])
    step = 2 * channels
    if len(payload) % step:
        raise ValueError(f"pcm16: payload of {len(payload)} bytes is not whole "
                         f"{channels}-channel planes ({step} bytes each)")
    n = len(payload) // step                       # samples per channel plane
    d = np.frombuffer(payload, dtype="<i2").astype(np.int64).reshape(channels, n)
    planes = (np.cumsum(d, axis=1) % 65536).astype("<u2")   # per-plane cumsum
    return planes.T.tobytes()                      # re-interleave


# ------------------ f32: LE float32 byte planes, most-significant first

_F32_MAGIC = b"RCF1"


def _f32_encode(data: bytes) -> bytes:
    if len(data) % 4:
        raise ValueError(f"f32: byte count {len(data)} not divisible by 4")
    n = len(data) // 4
    if n > 0xFFFFFFFF:
        raise ValueError("f32: more than 2**32-1 floats unsupported")
    b = np.frombuffer(data, dtype=np.uint8).reshape(n, 4)
    planes = b[:, ::-1].T              # (4, n): plane 3 (sign+exp) first, then 2,1,0
    return _F32_MAGIC + struct.pack("<I", n) + planes.tobytes()


def _f32_decode(data: bytes) -> bytes:
    if len(data) < 8 or bytes(data[:4]) != _F32_MAGIC:
        raise ValueError("f32: bad header (expected b'RCF1' + u32 count)")
    (n,) = struct.unpack("<I", data[4:8])
    payload = bytes(data[8:])
    if len(payload) != 4 * n:
        raise ValueError(f"f32: payload {len(payload)} bytes != 4 * count {n}")
    p = np.frombuffer(payload, dtype=np.uint8).reshape(4, n)
    return p[::-1].T.tobytes()         # (n, 4): bytes b0..b3 little-endian


# ------------------ u8_image: u8 raster, previous-pixel delta per row

_U8_MAGIC = b"RCI1"


def _u8_encode(data: bytes, width: int) -> bytes:
    if width < 1:
        raise ValueError(f"u8_image: width must be >= 1, got {width}")
    n = len(data)
    if n % width:
        raise ValueError(f"u8_image: length {n} is not a multiple of width {width}")
    h = n // width
    pix = np.frombuffer(data, dtype=np.uint8).astype(np.int64).reshape(h, width)
    d = np.empty_like(pix)
    d[:, 0] = pix[:, 0]                # first pixel of each row: raw
    d[:, 1:] = pix[:, 1:] - pix[:, :-1]   # previous-pixel delta per row
    body = (d % 256).astype(np.uint8)
    return _U8_MAGIC + struct.pack("<II", width, h) + body.tobytes()


def _u8_decode(data: bytes) -> bytes:
    if len(data) < 12 or bytes(data[:4]) != _U8_MAGIC:
        raise ValueError("u8_image: bad header (expected b'RCI1' + u32 w + u32 h)")
    w, h = struct.unpack("<II", data[4:12])
    payload = bytes(data[12:])
    if len(payload) != w * h:
        raise ValueError(f"u8_image: payload {len(payload)} bytes != "
                         f"width*height = {w}*{h}")
    d = np.frombuffer(payload, dtype=np.uint8).astype(np.int64).reshape(h, w)
    pix = (np.cumsum(d, axis=1) % 256).astype(np.uint8)     # per-row cumsum
    return pix.tobytes()


# ------------------ factories + the data-declared registry

def make_pcm16(channels: int = 2) -> Profile:
    """Build the pcm16 profile bound to `channels` interleaved LE int16 streams."""
    return Profile(
        name="pcm16",
        transform=lambda data: _pcm16_encode(data, channels),
        untransform=_pcm16_decode,     # header self-describes the channel count
        c=None,
        note=(f"LE int16 PCM, {channels} channel(s): deinterleave + per-channel "
              "int16 delta, planes stored sequentially; b'RCP1' + u8 channels."),
    )


def make_u8_image(width: int = 512) -> Profile:
    """Build the u8_image profile for a single-channel raster of `width` px."""
    return Profile(
        name="u8_image",
        transform=lambda data: _u8_encode(data, width),
        untransform=_u8_decode,        # header self-describes width/height
        c=None,
        note=("u8 raster: previous-pixel delta per row; b'RCI1' + u32 width "
              "+ u32 height."),
    )


PROFILES: dict[str, Profile] = {
    "raw": Profile(
        name="raw",
        transform=_identity,
        untransform=_identity,
        c=None,
        note="identity — bytes enter the tokenizer exactly as they are.",
    ),
    "pcm16": make_pcm16(2),
    "f32": Profile(
        name="f32",
        transform=_f32_encode,
        untransform=_f32_decode,
        c=None,
        note=("LE float32 byte planes, most-significant first "
              "(sign+exponent, then 2, 1, 0); b'RCF1' + u32 count."),
    ),
    "u8_image": make_u8_image(512),
}


# ------------------ public entry points

def _resolve(profile_or_name: "Profile | str") -> Profile:
    if isinstance(profile_or_name, Profile):
        return profile_or_name
    if isinstance(profile_or_name, str) and profile_or_name in PROFILES:
        return PROFILES[profile_or_name]
    raise ValueError(f"unknown profile: {profile_or_name!r}; "
                     f"known profiles: {list_profiles()}")


def apply(profile_or_name: "Profile | str", data: bytes) -> bytes:
    """Transform `data` with a profile (accepts a Profile instance or name)."""
    return _resolve(profile_or_name).transform(bytes(data))


def restore(profile_or_name: "Profile | str", data: bytes) -> bytes:
    """Exactly invert `apply` (accepts a Profile instance or name)."""
    return _resolve(profile_or_name).untransform(bytes(data))


def list_profiles() -> list[str]:
    """Names of every registered modality profile."""
    return list(PROFILES)
