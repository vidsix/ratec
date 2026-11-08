import os
import re

from setuptools import Extension, setup


def _pyproject_version():
    """Single source of truth for the version is pyproject.toml (P5-4);
    the C extension receives it via -DRC_VERSION and the drift is
    asserted by the suite."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pyproject.toml")) as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    if not m:
        raise RuntimeError("pyproject.toml has no [project] version")
    return m.group(1)


# ISA baseline (P5-1): bit-exactness is enforced by -ffp-contract=off and
# -fno-fast-math, NOT by the instruction set — so the default build uses
# the plain x86-64 baseline (SSE2, every x86-64 box since 2003) instead of
# -march=native, which produced SIGILL wheels on older CPUs and
# machine-dependent builds. Opt in explicitly:
#   RATEC_ARCH=native      full native (this box only)
#   RATEC_ARCH=x86-64-v2   SSE4.2/POPCNT baseline (Sandy Bridge+, 2008+)
_arch = os.environ.get("RATEC_ARCH", "").strip()
ARCH_ARGS = {"native": ["-march=native"],
             "x86-64-v2": ["-march=x86-64-v2"]}.get(_arch, [])

VERSION = _pyproject_version()

setup(
    version=VERSION,
    ext_modules=[
        Extension(
            "ratec._rcengine",
            sources=["ratec/_rcengine.c"],
            define_macros=[("RC_VERSION", f'"{VERSION}"')],
            extra_compile_args=[
                "-O3", *ARCH_ARGS,
                # mandatory for bit-exactness with the reference pipeline
                "-ffp-contract=off", "-fno-fast-math",
            ],
            libraries=["m", "pthread"],
        )
    ]
)
