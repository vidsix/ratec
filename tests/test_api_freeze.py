"""The API freeze drift test (the 1.0 freeze contract, executable
half): the public surface is regenerated in memory and asserted EQUAL
to the committed research/api_manifest.json.

Any rename, signature change, default change, added/removed public
name, error-code change, or C entry-point change fails here. A DELIBERATE
API change regenerates the manifest (bench/api_freeze.py) in a visible
commit and follows the versioning rules in docs/STABILITY.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

MANIFEST = os.path.join(ROOT, "research", "api_manifest.json")


def _generated():
    import api_freeze
    return api_freeze


def test_api_manifest_matches_the_frozen_surface():
    af = _generated()
    import ratec
    from ratec import api as _api

    fns = {}
    for name in sorted(ratec.__all__):
        obj = getattr(ratec, name)
        if af.inspect.isfunction(obj):
            fns[name] = af.sig_of(obj)

    cls = {"init": af.sig_of(ratec.RateC.__init__), "methods": {}}
    for name in sorted(ratec.RateC.__dict__):
        if name.startswith("_"):
            continue
        obj = getattr(ratec.RateC, name)
        if af.inspect.isfunction(obj):
            cls["methods"][name] = af.sig_of(obj)
    classmethods = {}
    for name in sorted(ratec.RateC.__dict__):
        obj = af.inspect.getattr_static(ratec.RateC, name)
        if isinstance(obj, classmethod) and not name.startswith("_"):
            classmethods[name] = af.sig_of(getattr(ratec.RateC, name))
    if classmethods:
        cls["classmethods"] = classmethods
    props = [n for n in sorted(ratec.RateC.__dict__)
             if type(af.inspect.getattr_static(ratec.RateC, n)) is property]
    if props:
        cls["properties"] = props

    frozen = json.load(open(MANIFEST))
    assert frozen["python"]["functions"] == fns, \
        "module-level API drift vs research/api_manifest.json"
    assert frozen["python"]["RateC"] == cls, \
        "RateC surface drift vs research/api_manifest.json"
    assert frozen["python"]["__all__"] == list(ratec.__all__)
    assert frozen["python"]["RatecErrorCode"] == {
        m.name: str(m.value) for m in ratec.RatecErrorCode}
    assert frozen["python"]["RatecError_attrs"] == [
        n for n in sorted(dir(ratec.RatecError)) if not n.startswith("_")]
    assert frozen["c"]["functions"] == sorted(
        n for n in dir(_api._rc) if not n.startswith("_")), \
        "C entry-point drift vs research/api_manifest.json"
    assert frozen["meta"]["__version__"] == ratec.__version__


def test_manifest_record_has_provenance():
    frozen = json.load(open(MANIFEST))
    rec = frozen["record"]
    assert rec.get("git_rev") and rec.get("created_utc")
    assert rec.get("host", {}).get("python")
