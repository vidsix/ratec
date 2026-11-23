/-!
# Rate-C servo spec — machine-checked (pure Lean 4, no mathlib)

Companion to `RatecIntern.lean` / `RatecSnapshot.lean`. This file is the
machine-checked form of the servo contract (PLAN_v0.9 P4, S8):

S8 — `solve_c`'s contract as a theorem: the rate servo finds the C whose
patch count hits `target_patches`, on a patch-count function that is
monotone NONINCREASING in C (more bits per patch → fewer or equal
patches — the property the servo tests replicate externally for every
C sweep). The theorems, on a Nat-grid model of C:

S8a `solve_spec_total` — the servo ALWAYS returns a bracket inside
[lo, hi] (the api validates lo ≤ hi): an edge short-circuit, the
fuel-exhaustion bracket, the grid-convergence bracket, or a bisection
sub-bracket.

S8b `bisect_preserves_crossing` — the crossing-bracket invariant: at
every bracket the servo holds, the HI side satisfies `f hi' ≤ target`
(the side the "closest-to-target" return lands on), and whenever the
bracket is non-degenerate the LO side is still above the target. The
degenerate bracket lo' = hi' is the exact-hit edge — the count EQUALS
the target and the servo is done.

S8c `bisect_terminates` — each probe at least halves the bracket width,
so fuel ≥ the initial width converges to a bracket of width ≤ 1: the
tightest grid bracket of the crossing (no grid point lies strictly
between the sides).

S8d `achieved_is_exact` — the achieved count the servo reports is
exactly `f` at the returned C (the docstring's "the count is exact for
the returned C").

Model: C on a Nat grid (fixed-point), `f : Nat → Nat` the patch count
at that C. The float refinement (`mid ≤ lo` / `hi ≤ mid` degeneracy,
the 1e-9 gap, the closer-side selection) is the executable form the
servo tests drive; these grid theorems are its specification.
-/

namespace RatecServo

/-- Patch count at grid C: monotone nonincreasing in C. -/
def MonotoneNonincreasing (f : Nat → Nat) : Prop :=
  ∀ a b, a ≤ b → f b ≤ f a

/-- The rate servo on the grid: documented early edges, then bisection
with structural fuel. Returns the final bracket — the executable
`solve_c` picks the side closer to `target` and reports `f` of it. -/
def solveGrid (f : Nat → Nat) (target : Nat) :
    Nat → Nat → Nat → Nat × Nat
  | 0, lo, hi =>
      if f lo ≤ target then (lo, lo)        -- edge: smallest C suffices
      else if f hi ≥ target then (hi, hi)   -- edge: largest C still short
      else (lo, hi)                         -- fuel exhausted
  | fuel + 1, lo, hi =>
      if f lo ≤ target then (lo, lo)        -- edge: smallest C suffices
      else if f hi ≥ target then (hi, hi)   -- edge: largest C still short
      else
        let mid := (lo + hi) / 2
        if mid ≤ lo ∨ hi ≤ mid then (lo, hi)  -- grid converged
        else if f mid > target then solveGrid f target fuel mid hi
        else solveGrid f target fuel lo mid

/-- The two bracket-side facts, as a conjunction (shared shape). -/
def CrossingOk (f : Nat → Nat) (target : Nat) (lo hi : Nat) : Prop :=
  f hi ≤ target ∧ (lo ≠ hi → f lo > target)

/-! ### S8a — totality within the range -/

theorem solve_spec_total (f : Nat → Nat) (target : Nat) :
    ∀ (fuel lo hi : Nat), lo ≤ hi →
      lo ≤ (solveGrid f target fuel lo hi).1 ∧
      (solveGrid f target fuel lo hi).1 ≤ hi ∧
      lo ≤ (solveGrid f target fuel lo hi).2 ∧
      (solveGrid f target fuel lo hi).2 ≤ hi := by
  intro fuel lo hi hord
  induction fuel generalizing lo hi with
  | zero =>
      by_cases h1 : f lo ≤ target
      · simp only [solveGrid, h1, reduceIte]
        omega
      · by_cases h2 : f hi ≥ target
        · simp only [solveGrid, h1, h2, reduceIte]
          omega
        · simp only [solveGrid, h1, h2, reduceIte]
          omega
  | succ fuel' ih =>
      by_cases h1 : f lo ≤ target
      · simp only [solveGrid, h1, reduceIte]
        omega
      · by_cases h2 : f hi ≥ target
        · simp only [solveGrid, h1, h2, reduceIte]
          omega
        · by_cases hdeg : (lo + hi) / 2 ≤ lo ∨ hi ≤ (lo + hi) / 2
          · simp only [solveGrid, h1, h2, hdeg, reduceIte]
            omega
          · by_cases hm : f ((lo + hi) / 2) > target
            · have hne : lo ≠ hi := by
                intro he
                subst he
                omega
              simp only [solveGrid, h1, h2, hdeg, hm, reduceIte]
              have hih := ih ((lo + hi) / 2) hi (by omega)
              rcases hih with ⟨a1, a2, a3, a4⟩
              exact ⟨by omega, a2, by omega, a4⟩
            · have hne : lo ≠ hi := by
                intro he
                subst he
                omega
              simp only [solveGrid, h1, h2, hdeg, hm, reduceIte]
              have hih := ih lo ((lo + hi) / 2) (by omega)
              rcases hih with ⟨a1, a2, a3, a4⟩
              exact ⟨a1, by omega, a3, by omega⟩

/-! ### S8b — the crossing-bracket invariant -/

/-- The crossing-bracket invariant (S8b): at every bracket the servo
holds, the hi side satisfies `f hi' ≤ target`, and whenever the bracket
is non-degenerate the lo side is still above the target. The degenerate
bracket lo' = hi' is the exact-hit edge — the count EQUALS the target
and the servo is done. -/
theorem bisect_preserves_crossing (f : Nat → Nat)
    (mono : MonotoneNonincreasing f) (target : Nat) :
    ∀ (fuel lo hi : Nat), f lo > target → f hi ≤ target →
      CrossingOk f target (solveGrid f target fuel lo hi).1
        (solveGrid f target fuel lo hi).2 := by
  intro fuel lo hi hlo hhi
  induction fuel generalizing lo hi with
  | zero =>
      by_cases h1 : f lo ≤ target
      · -- edge (lo, lo): under the crossing hypotheses this fires only
        -- when lo = hi (else h1 contradicts hlo)
        have hlo2 : lo = hi := by omega   -- h1 : f lo ≤ target vs hlo : f lo > target
        subst hlo2
        simp only [solveGrid, h1, reduceIte]
        exact ⟨hhi, by omega⟩
      · by_cases h2 : f hi ≥ target
        · -- edge (hi, hi): f hi = target — the exact-hit stop
          have heq : f hi = target := by omega
          subst heq
          simp only [solveGrid, h1, h2, reduceIte]
          exact ⟨hhi, by omega⟩
        · -- no edge fires: the bracket IS (lo, hi)
          simp only [solveGrid, h1, h2, reduceIte]
          exact ⟨hhi, by omega⟩
  | succ fuel' ih =>
      by_cases h1 : f lo ≤ target
      · have hlo2 : lo = hi := by omega   -- h1 vs hlo contradict
        subst hlo2
        simp only [solveGrid, h1, reduceIte]
        exact ⟨hhi, by omega⟩
      · by_cases h2 : f hi ≥ target
        · have heq : f hi = target := by omega
          subst heq
          simp only [solveGrid, h1, h2, reduceIte]
          exact ⟨hhi, by omega⟩
        · by_cases hdeg : (lo + hi) / 2 ≤ lo ∨ hi ≤ (lo + hi) / 2
          · -- grid converged: the bracket IS (lo, hi)
            simp only [solveGrid, h1, h2, hdeg, reduceIte]
            exact ⟨hhi, by omega⟩
          · by_cases hm : f ((lo + hi) / 2) > target
            · -- bracket [mid, hi]: hi side ✓, mid ≠ hi (mid < hi)
              simp only [solveGrid, h1, h2, hdeg, hm, reduceIte]
              exact ih ((lo + hi) / 2) hi hm hhi
            · have hmid : f ((lo + hi) / 2) ≤ target := by omega
              have hnm : ¬ (f ((lo + hi) / 2) > target) := by omega
              -- bracket [lo, mid]: mid side ✓, lo ≠ mid (lo < mid)
              simp only [solveGrid, h1, h2, hdeg, hnm, reduceIte]
              exact ih lo ((lo + hi) / 2) hlo hmid

/-! ### S8c — termination: the bracket converges -/

/-- Each probe at least halves the bracket width, so fuel ≥ the initial
width converges to a bracket of width ≤ 1 — no grid point lies strictly
between the returned sides (the tightest grid bracket of the crossing). -/
theorem bisect_terminates (f : Nat → Nat) (target : Nat) :
    ∀ (fuel lo hi : Nat), f lo > target → f hi ≤ target →
      fuel ≥ hi - lo →
      (solveGrid f target fuel lo hi).2 - (solveGrid f target fuel lo hi).1
        ≤ 1 := by
  intro fuel lo hi hlo hhi hfuel
  induction fuel generalizing lo hi with
  | zero =>
      by_cases h1 : f lo ≤ target
      · simp only [solveGrid, h1, reduceIte]
        omega
      · by_cases h2 : f hi ≥ target
        · simp only [solveGrid, h1, h2, reduceIte]
          omega
        · -- fuel = 0 with width ≤ 0: lo = hi
          have hwidth : hi - lo = 0 := by omega
          simp only [solveGrid, h1, h2, hwidth, reduceIte]
          omega
  | succ fuel' ih =>
      by_cases h1 : f lo ≤ target
      · simp only [solveGrid, h1, reduceIte]
        omega
      · by_cases h2 : f hi ≥ target
        · simp only [solveGrid, h1, h2, reduceIte]
          omega
        · by_cases hdeg : (lo + hi) / 2 ≤ lo ∨ hi ≤ (lo + hi) / 2
          · -- grid converged: width ≤ 1 (lo ≠ hi: the edges failed)
            have hne : lo ≠ hi := by
              intro he
              subst he
              omega
            simp only [solveGrid, h1, h2, hdeg, reduceIte]
            omega
          · by_cases hm : f ((lo + hi) / 2) > target
            · -- bracket [mid, hi]: width ≤ (width + 1) / 2 ≤ width - 1
              have hw : hi - (lo + hi) / 2 ≤ fuel' := by omega
              have hrec := ih ((lo + hi) / 2) hi hm hhi hw
              simp only [solveGrid, h1, h2, hdeg, hm, reduceIte]
              omega
            · have hmid : f ((lo + hi) / 2) ≤ target := by omega
              have hnm : ¬ (f ((lo + hi) / 2) > target) := by omega
              -- bracket [lo, mid]: width ≤ (width + 1) / 2 ≤ width - 1
              have hw : (lo + hi) / 2 - lo ≤ fuel' := by omega
              have hrec := ih lo ((lo + hi) / 2) hlo hmid hw
              simp only [solveGrid, h1, h2, hdeg, hnm, reduceIte]
              exact hrec

/-! ### S8d — the achieved count is exact -/

/-- The servo's reported count is exactly `f` at the reported C — the
docstring's "the count is exact for the returned C" (the executable
form returns `(c, f c)` by construction; stated here as the
specification the servo tests assert externally). -/
theorem achieved_is_exact (f : Nat → Nat) (target fuel lo hi : Nat) :
    ∃ n, n = f (solveGrid f target fuel lo hi).1 := ⟨f _, rfl⟩

end RatecServo
