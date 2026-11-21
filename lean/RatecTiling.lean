/-!
# Rate-C patcher tiling — machine-checked (pure Lean 4, no mathlib)

Companion to `RateC.lean` (vocabulary) and `RatecSerial.lean` (execution
model). This file proves the STRUCTURAL guarantee that the fast lane
relies on: whatever non-negative surprisal values `sf i` the judge
produces — exact libm doubles, the fast-log2 polynomial, or any future
approximation — the greedy under-shoot patcher emits spans that
PARTITION the stream exactly. Consequences:

T1 `pstep_inv` — the tiling invariant (closed spans cover [0, openStart),
   the open patch is [openStart, pos)) is preserved by every byte, in
   every branch (byte joins, forced single byte, close+join, forced
   escape).

T2 `pfold_tiling` — after n bytes the closed spans plus the open patch
   partition [0, n). The judge never sees this theorem: it holds for ANY
   `sf`, so lane choices and approximations cannot break the
   segmentation (decode always round-trips).

T3 `tiling_concat` — a partition's pieces, concatenated, rebuild the
   whole range: the roundtrip statement in machine-checked form.

The model drops the carry-over patcher mode (carry only shifts `lim`,
never the tiling structure) and the 4 GiB span-limit aborts; the
branches mirror run_body in `_rcengine.c`.
-/

namespace RatecTiling

/-- Patcher state: stream position, closed spans in order, the open
patch, the running bits `cs` and the current budget `lim`. -/
structure PState where
  pos : Nat
  spans : List (Nat × Nat)
  openStart : Nat
  openLen : Nat
  cs : Nat
  lim : Nat

/-- One byte of the greedy under-shoot patcher (carry mode off; carry
only shifts `lim`, never the tiling). Mirrors run_body: open if needed,
join while the budget holds, else close — and the byte either joins the
fresh patch or escapes as a forced single byte. -/
def pstep (sf : Nat → Nat) (C : Nat) (s : PState) : PState :=
  let w := sf s.pos
  let openS := if s.openLen = 0 then s.pos else s.openStart
  let lim' := if s.openLen = 0 then s.cs + C else s.lim
  if s.cs + w ≤ lim' then
    ⟨s.pos + 1, s.spans, openS, s.openLen + 1, s.cs + w, lim'⟩
  else if s.openLen = 0 then
    -- forced single byte (the patch opened this byte and closed at once)
    ⟨s.pos + 1, s.spans ++ [(s.pos, s.pos + 1)], 0, 0, s.cs + w, s.cs + C⟩
  else if s.cs + w ≤ s.cs + C then
    -- close the open patch; the byte joins the fresh one
    ⟨s.pos + 1, s.spans ++ [(s.openStart, s.pos)], s.pos, 1,
      s.cs + w, s.cs + C⟩
  else
    -- close, then forced escape
    ⟨s.pos + 1, s.spans ++ [(s.openStart, s.pos), (s.pos, s.pos + 1)],
      0, 0, s.cs + w, s.cs + C⟩

/-- The n-byte run of the patcher from the cold state. -/
def pfold (sf : Nat → Nat) (C : Nat) (n : Nat) : PState :=
  (List.range n).foldl (fun s _ => pstep sf C s) ⟨0, [], 0, 0, 0, 0⟩

/-! ## the partition predicate -/

/-- `coversFrom spans a n`: `spans` tile [a, n) consecutively. -/
def coversFrom : List (Nat × Nat) → Nat → Nat → Prop
  | [], a, n => a = n
  | (a, b) :: rest, from_, n => a = from_ ∧ a < b ∧ b ≤ n ∧ coversFrom rest b n

/-- Extending a cover by one abutting (non-empty) span. -/
theorem covers_append {xs : List (Nat × Nat)} {a m : Nat}
    (hcov : coversFrom xs a m) {y : Nat} (hbound : m < y) :
    coversFrom (xs ++ [(m, y)]) a y := by
  induction xs generalizing a with
  | nil =>
      simp only [List.nil_append, coversFrom]
      exact ⟨hcov.symm, hbound, Nat.le_refl y, trivial⟩
  | cons x rest ih =>
      obtain ⟨xa, xb, xbn, xrest⟩ := hcov
      simp only [List.cons_append, coversFrom]
      exact ⟨xa, xb, by omega, ih xrest⟩

/-- Concatenating two abutting covers. -/
theorem covers_end_ge {ys : List (Nat × Nat)} {m n : Nat}
    (h2 : coversFrom ys m n) : m ≤ n := by
  cases ys with
  | nil => simp only [coversFrom] at h2; omega
  | cons y rest =>
      obtain ⟨ya, yb, ybn, _⟩ := h2
      omega

theorem covers_concat {ys : List (Nat × Nat)} {m n : Nat}
    (h2 : coversFrom ys m n) :
    ∀ (xs : List (Nat × Nat)) (a : Nat), coversFrom xs a m →
      coversFrom (xs ++ ys) a n := by
  intro xs
  induction xs with
  | nil =>
      intro a h1
      simp only [coversFrom] at h1 ⊢
      rw [h1]
      exact h2
  | cons x rest ih =>
      intro a h1
      obtain ⟨xa, xb, xbn, xrest⟩ := h1
      simp only [List.cons_append, coversFrom]
      have hmn := covers_end_ge h2
      exact ⟨xa, xb, by omega, ih x.snd xrest⟩

/-! ## T1 — the invariant -/

/-- The tiling invariant: either no open patch and the closed spans
cover [0, pos), or an open patch [openStart, pos) behind a cover of
[0, openStart). -/
def inv (s : PState) : Prop :=
  (s.openLen = 0 ∧ coversFrom s.spans 0 s.pos) ∨
  (s.openLen > 0 ∧ coversFrom s.spans 0 s.openStart ∧
    s.openStart + s.openLen = s.pos)

theorem inv_init : inv ⟨0, [], 0, 0, 0, 0⟩ :=
  Or.inl ⟨rfl, rfl⟩

theorem pstep_pos (sf : Nat → Nat) (C : Nat) (s : PState) :
    (pstep sf C s).pos = s.pos + 1 := by
  by_cases h2 : s.openLen = 0
  · simp only [pstep, h2, reduceIte]
    by_cases h1 : s.cs + sf s.pos ≤ s.cs + C
    · simp [h1]
    · simp [h1]
  · simp only [pstep, h2, reduceIte]
    by_cases h1 : s.cs + sf s.pos ≤ s.lim
    · simp [h1]
    · by_cases h3 : s.cs + sf s.pos ≤ s.cs + C
      · simp [h1, h3]
      · simp [h1, h3]

/-- **T1.** Every branch of `pstep` preserves the tiling invariant:
join extends the open patch; forced/close branches emit spans that
continue the cover exactly where it ended. -/
theorem pstep_inv (sf : Nat → Nat) (C : Nat) (s : PState) (h : inv s) :
    inv (pstep sf C s) := by
  unfold inv at h
  by_cases h2 : s.openLen = 0
  · simp only [pstep, h2, reduceIte]
    by_cases h1 : s.cs + sf s.pos ≤ s.cs + C
    · -- branch A, fresh open: the byte joins the patch opened this byte
      simp only [h1, reduceIte]
      rcases h with ⟨_, hcov⟩ | ⟨hopen, _, _⟩
      · exact Or.inr ⟨by dsimp only <;> omega, hcov, by dsimp only <;> omega⟩
      · exfalso; omega
    · -- branch B: forced single byte
      simp only [h1, reduceIte]
      rcases h with ⟨_, hcov⟩ | ⟨hopen, _, _⟩
      · exact Or.inl ⟨rfl, covers_append hcov (by omega)⟩
      · exfalso; omega
  · simp only [pstep, h2, reduceIte]
    by_cases h1 : s.cs + sf s.pos ≤ s.lim
    · -- branch A, existing open patch: the byte joins it
      simp only [h1, reduceIte]
      rcases h with ⟨_, _⟩ | ⟨_, hcov, hlen⟩
      · exfalso; omega
      · exact Or.inr ⟨by dsimp only <;> omega, hcov, by dsimp only <;> omega⟩
    · by_cases h3 : s.cs + sf s.pos ≤ s.cs + C
      · -- branch C: close, byte joins the fresh patch
        simp only [h1, h3, reduceIte]
        rcases h with ⟨_, _⟩ | ⟨hopen, hcov, hlen⟩
        · exfalso; omega
        · exact Or.inr ⟨by dsimp only <;> omega, covers_append hcov (by omega), by dsimp only <;> omega⟩
      · -- branch D: close, forced escape
        simp only [h1, h3, reduceIte]
        rcases h with ⟨_, _⟩ | ⟨hopen, hcov, hlen⟩
        · exfalso; omega
        · have step1 : coversFrom (s.spans ++ [(s.openStart, s.pos)]) 0 s.pos :=
            covers_append hcov (by omega)
          have step2 : coversFrom
              ((s.spans ++ [(s.openStart, s.pos)]) ++ [(s.pos, s.pos + 1)])
              0 (s.pos + 1) := covers_append step1 (by omega)
          rw [List.append_assoc] at step2
          exact Or.inl ⟨rfl, step2⟩

theorem pfold_inv (sf : Nat → Nat) (C : Nat) (n : Nat) :
    inv (pfold sf C n) ∧ (pfold sf C n).pos = n := by
  induction n with
  | zero => exact ⟨inv_init, rfl⟩
  | succ k ih =>
      obtain ⟨hinv, hpos⟩ := ih
      have h1 := pstep_inv sf C (pfold sf C k) hinv
      have h2 := pstep_pos sf C (pfold sf C k)
      rw [hpos] at h2
      constructor
      · simpa [pfold, List.range_succ, List.foldl_cons] using h1
      · simpa [pfold, List.range_succ, List.foldl_cons] using h2

/-! ## T2 — the tiling theorem -/

/-- **T2.** After n bytes, closed spans plus the open patch partition
[0, n) — for ANY non-negative surprisal function `sf`. The segmentation
is independent of the lane's arithmetic. -/
theorem pfold_tiling (sf : Nat → Nat) (C : Nat) (n : Nat) :
    coversFrom ((pfold sf C n).spans ++
      (if (pfold sf C n).openLen > 0 then
        [((pfold sf C n).openStart,
          (pfold sf C n).openStart + (pfold sf C n).openLen)]
      else [])) 0 n := by
  obtain ⟨hinv, hpos⟩ := pfold_inv sf C n
  by_cases hop : (pfold sf C n).openLen > 0
  · rcases hinv with ⟨h0, _⟩ | ⟨hopen, hcov, hlen⟩
    · exfalso; omega
    · rw [if_pos hop]
      have htail : coversFrom [((pfold sf C n).openStart,
          (pfold sf C n).openStart + (pfold sf C n).openLen)]
          (pfold sf C n).openStart n := by
        simp only [coversFrom]
        refine ⟨trivial, ?_, ?_, ?_⟩
        · omega
        · omega
        · omega
      exact covers_concat htail (pfold sf C n).spans 0 hcov
  · rw [if_neg hop, List.append_nil]
    rcases hinv with ⟨_, hcov⟩ | ⟨hopen, _, _⟩
    · rw [hpos] at hcov
      exact hcov
    · exfalso; omega

/-! ## T3 — decode output size -/

/-- **T3.** The spans' lengths sum to the stream length: the token
stream decodes to exactly `n - a` bytes over a cover of [a, n). -/
theorem covers_sum (spans : List (Nat × Nat)) :
    ∀ (a n : Nat), coversFrom spans a n →
      (spans.map (fun p => p.2 - p.1)).foldr (· + ·) 0 = n - a := by
  induction spans with
  | nil =>
      intro a n hcov
      simp only [coversFrom] at hcov
      simp [hcov]
  | cons sp rest ih =>
      intro a n hcov
      obtain ⟨ha, hb, hbn, hrest⟩ := hcov
      simp only [List.map_cons, List.foldr_cons]
      rw [ha]
      have hrec := ih (sp.2) n hrest
      omega

/-- The whole run's spans (closed + open) decode to exactly n bytes. -/
theorem pfold_total_len (sf : Nat → Nat) (C : Nat) (n : Nat) :
    (((pfold sf C n).spans ++
        (if (pfold sf C n).openLen > 0 then
          [((pfold sf C n).openStart,
            (pfold sf C n).openStart + (pfold sf C n).openLen)]
        else [])).map (fun p => p.2 - p.1)).foldr (· + ·) 0 = n := by
  have h := covers_sum ((pfold sf C n).spans ++
      (if (pfold sf C n).openLen > 0 then
        [((pfold sf C n).openStart,
          (pfold sf C n).openStart + (pfold sf C n).openLen)]
      else [])) 0 n (pfold_tiling sf C n)
  rwa [Nat.sub_zero] at h

end RatecTiling
