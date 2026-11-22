/-!
# Rate-C snapshot rejection totality — machine-checked (pure Lean 4, no
mathlib)

Companion to `RatecIntern.lean` / `RatecDecode.lean`. This file is the
machine-checked form of the snapshot-format claim (PLAN_v0.9 P4, S9),
modeling the load path's HEADER DISCRIMINATION in `_rcengine.c`'s
`load_core`:

S9a `load_classify_total` — for ANY inputs, `loadClassify` returns one
of exactly three classes: `ok`, `invalid`, or `version v`. The load
path never has a fourth outcome (the C returns an engine or a coded
error; the fuzz gate's mutant battery enumerates the rejection paths).

S9b `version_dominates` — the version check PRECEDES body validation: a
blob whose version byte is outside [2, 8] classifies as `version v`
EVEN IF its body would fail every other check. This is the precedence
the C implements (the version branch runs before `core_new` and before
any section parse), and it is what makes `SNAPSHOT_VERSION` a
trustworthy code: seeing it means "your file is a NEWER/OLDER ratec
format", never "your file is corrupt".

S9c `magic_gates` — a wrong magic is `invalid` regardless of everything
else (not-ratec bytes never masquerade as a version problem).

S9d `ok_iff_valid` — `ok` exactly when the header sanity bounds hold
(the P1-6 field guards: flags ≤ 1, pf ≤ 4, map_capbits ∈ [8, 26],
prune_flag ≤ 1) and the body validates. The body itself is structural
(rd_get length checks — the C's trailing-garbage + bounds checks, kept
out of this model as the fuzz gate's territory).

The C raise sites this licenses (rc_load_state): kind == 1 → coded
`SNAPSHOT_VERSION`; kind == 0 → coded `SNAPSHOT_INVALID`; `ok` → the
engine. tests/test_errors.py asserts both codes live.
-/

namespace RatecSnapshot

/-- The three outcomes a snapshot load can have — no fourth. -/
inductive SnapResult (α : Type) where
  | ok (state : α)
  | invalid
  | version (v : Nat)

/-- The RCST magic, little-endian ("RCST"). -/
def RCST_MAGIC : Nat := 0x54534352

/-- The header discrimination + sanity guards, one faithful function
over the checks `load_core` performs before touching any body section.
The conditions arrive as the Bool values the C computes:

  badMagic  = (magic != RCST_MAGIC)          — the P0 magic check
  overVer   = (ver < 2 \/ ver > 8)            — the v2..v8 range check
  badBounds = flags > 1 \/ pf > 4 \/ mcb outside [8, 26] \/ pruneFlag > 1
              (the P1-6 field-sanity guards)
  bodyOK    — the body's structural validation (length accounting,
              bounds, trailing-garbage): the fuzz gate's territory.

The field-level derivation of these Bools is 1:1 with `_rcengine.c`;
this function fixes the CLASSIFICATION and its precedence. -/
def loadClassify (ver : Nat) (badMagic overVer badBounds bodyOK : Bool) :
    SnapResult Nat :=
  match badMagic, overVer, badBounds, bodyOK with
  | true, _, _, _ => .invalid
  | false, true, _, _ => .version ver
  | false, false, true, _ => .invalid
  | false, false, false, true => .ok 0
  | false, false, false, false => .invalid

/-- S9a — totality: exactly three outcomes for every input. -/
theorem load_classify_total (ver : Nat) (badMagic overVer badBounds
    bodyOK : Bool) :
    loadClassify ver badMagic overVer badBounds bodyOK = .invalid ∨
    loadClassify ver badMagic overVer badBounds bodyOK = .version ver ∨
    loadClassify ver badMagic overVer badBounds bodyOK = .ok 0 := by
  cases badMagic <;> cases overVer <;> cases badBounds <;> cases bodyOK <;>
    simp [loadClassify]

/-- S9b — version dominance: an out-of-range version classifies as
`.version ver` regardless of the body's validity — the precedence that
makes SNAPSHOT_VERSION mean "format version", never "corrupt bytes".
(Wrong MAGIC still wins first: see `magic_gates` — the C checks magic,
then version.) -/
theorem version_dominates (ver : Nat) (bodyOK : Bool)
    (hM : badMagic = false) (hv : overVer = true) :
    loadClassify ver badMagic overVer badBounds bodyOK = .version ver := by
  simp [loadClassify, hM, hv]

/-- S9c — wrong magic → invalid, before anything else (not-ratec bytes
never report a version). -/
theorem magic_gates (ver : Nat) (bodyOK : Bool) (hM : badMagic = true) :
    loadClassify ver badMagic overVer badBounds bodyOK = .invalid := by
  simp [loadClassify, hM]

/-- S9d — ok exactly when no check fails: magic matches, the version is
in range, the header bounds hold, and the body validates. -/
theorem ok_iff_valid (ver : Nat) :
    loadClassify ver badMagic overVer badBounds bodyOK = .ok 0 ↔
      badMagic = false ∧ overVer = false ∧ badBounds = false ∧ bodyOK := by
  cases badMagic <;> cases overVer <;> cases badBounds <;> cases bodyOK <;>
    simp [loadClassify]


end RatecSnapshot

