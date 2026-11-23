#!/usr/bin/env python3
"""The API freeze audit (the 1.0 freeze contract, executable half).

Introspects the ENTIRE public surface and writes the frozen manifest
(research/api_manifest.json):

  * module-level functions with full signatures (parameter names,
    kinds, defaults — inspect.signature, annotations as strings so the
    manifest is stable across interpreter versions)
  * the RateC class: constructor signature + every public method's
    signature
  * RatecError's public attributes and RatecErrorCode's members (the
    coded error taxonomy is frozen with the API)
  * the C extension's entry-point name list (what api.py's wrappers
    promise)
  * __version__

The drift test (tests/test_api_freeze.py) regenerates the manifest
in-memory and asserts EXACT equality with the committed file: any
rename, signature change, default change, added or removed public name
fails the suite. For 1.0 the manifest IS the freeze — a change means a
regeneration commit (visible in the diff) plus the version bump and
deprecation note the stability policy (docs/STABILITY.md) requires.

Run:  python bench/api_freeze.py            # regenerate the manifest
"""
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

import ratec
from ratec import api as _api
_rc = _api._rc

from record_util import record_envelope, write_record

OUT = os.path.join(ROOT, "research", "api_manifest.json")


def sig_of(obj):
    """Signature as a stable structure: [(name, kind, default)] with
    annotations as strings."""
    sig = inspect.signature(obj)
    params = []
    for p in sig.parameters.values():
        default = (None if p.default is inspect.Parameter.empty
                   else repr(p.default))
        ann = ("" if p.annotation is inspect.Parameter.empty
               else str(p.annotation))
        params.append([p.name, p.kind.name, default, ann])
    ret = ("" if sig.return_annotation is inspect.Signature.empty
           else str(sig.return_annotation))
    return {"params": params, "return": ret}


def main():
    manifest = {"python": {}, "c": {}, "meta": {}}

    # module-level public functions (skip the class + re-exported errors)
    fns = {}
    for name in sorted(ratec.__all__):
        obj = getattr(ratec, name)
        if inspect.isfunction(obj):
            fns[name] = sig_of(obj)
    manifest["python"]["functions"] = fns

    # the RateC class surface
    cls = {"init": sig_of(ratec.RateC.__init__), "methods": {}}
    for name in sorted(ratec.RateC.__dict__):
        if name.startswith("_"):
            continue
        obj = getattr(ratec.RateC, name)
        if inspect.isfunction(obj):
            cls["methods"][name] = sig_of(obj)
    classmethods = {}
    for name in sorted(ratec.RateC.__dict__):
        obj = inspect.getattr_static(ratec.RateC, name)
        if isinstance(obj, classmethod) and not name.startswith("_"):
            classmethods[name] = sig_of(getattr(ratec.RateC, name))
    if classmethods:
        cls["classmethods"] = classmethods
    props = [n for n in sorted(ratec.RateC.__dict__)
             if type(inspect.getattr_static(ratec.RateC, n)) is property]
    if props:
        cls["properties"] = props
    manifest["python"]["RateC"] = cls

    # the coded error taxonomy
    manifest["python"]["RatecErrorCode"] = {
        m.name: str(m.value) for m in ratec.RatecErrorCode}
    # the WRAPPER-DEFINED public attrs only: inherited BaseException
    # attrs (args, with_traceback) are interpreter-level, and add_note
    # exists only on 3.11+ — freezing dir() made the manifest break on
    # older interpreters (the 3.10 CI job caught it). The contract is
    # what the wrapper defines.
    manifest["python"]["RatecError_attrs"] = [
        n for n in sorted(ratec.RatecError.__dict__)
        if not n.startswith("_")]
    manifest["python"]["__all__"] = list(ratec.__all__)

    # C entry points the wrappers promise
    manifest["c"]["functions"] = sorted(
        n for n in dir(_rc) if not n.startswith("_"))
    manifest["c"]["doc_line"] = (_rc.__doc__ or "").strip().splitlines()[0]

    manifest["meta"]["__version__"] = ratec.__version__
    manifest["record"] = record_envelope(inputs={})

    write_record(OUT, manifest)
    n_fns = len(manifest["python"]["functions"])
    n_m = len(manifest["python"]["RateC"]["methods"])
    n_c = len(manifest["c"]["functions"])
    print(f"manifest: {n_fns} module functions, "
          f"{len(manifest['python']['RateC']['classmethods'])} classmethods, "
          f"{n_m} methods, "
          f"{len(manifest['python']['RatecErrorCode'])} error codes, "
          f"{n_c} C entry points")
    print("methods:", ", ".join(manifest["python"]["RateC"]["methods"]))
    print("functions:", ", ".join(manifest["python"]["functions"]))


if __name__ == "__main__":
    main()
