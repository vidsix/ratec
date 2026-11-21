/-!
# Rate-C vocabulary — machine-checked core lemmas (pure Lean 4, no mathlib)

Setting. A frozen causal byte model `q` yields per-byte surprisal; rounded up to
integers, every byte costs `cost b ≤ 8` bits. Fix a budget `B` (in the architecture
`B = C - 2`, the `-2` absorbing the rANS `⌈·⌉+1` rounding, so codes fit in `C` bits).
A patch `w` (in a fixed context) is *admissible* when its accumulated cost `≤ B`.
Under `q`, an admissible patch has probability `≥ 2^-B`, so its dyadic mass is
`μ(w) = 2^(B - cost w)`.

What is proved here, with no `sorry`:

P1 `kraft_count` / `kraft_from_costs` — the table size is forced *within a
   (context, length) class*: at most `2^B` admissible patches of any fixed length
   after any fixed context (fixed-length string events are disjoint under q, so
   their Kraft sum is ≤ 1). Across MIXED lengths the admissible set is NOT bounded
   by `2^B` (exhaustive counterexample in `verify_ratec.py`: 84 > 64 at B = 6) —
   which is why the length embedding ℓ is a load-bearing part of the address:
   (index, ℓ) is per-context injective. The `2^C` vocabulary is not a design
   choice; it is the same equation as the budget rule, one class at a time.

P2 `cut_band` — the greedy cut is well formed: every emitted patch costs `≤ B`,
   and every *closed* (non-final) patch costs `≥ B - 7`, i.e. interior patches sit
   in a uniform band of width 8 around the budget. This is the machine-checked
   form of "every unit token carries (approximately) the same amount of
   information".

P3 `ratec_codes` — the index exists, constructively: the cumulative-mass map
   (the skeleton of the arithmetic/rANS code) assigns every admissible patch a
   *distinct* index in `[0, 2^B)`.

Not proved here (prose level): the exact chain rule `p(x) = Π_k p(w_k | w_<k)`,
a one-line measure argument from the determinism of the cut, verified
exhaustively in `verify_ratec.py`.
-/

-- ============================================================
-- Helpers
-- ============================================================

theorem ge1_length_le_sum : ∀ (l : List Nat), (∀ x ∈ l, 1 ≤ x) → l.length ≤ l.sum
  | [], _ => Nat.le.refl
  | x :: xs, h => by
      have h1 : 1 ≤ x := h x (by simp)
      have ih := ge1_length_le_sum xs (fun y hy => h y (by simp [hy]))
      simp only [List.length_cons, List.sum_cons]
      omega

-- ============================================================
-- P1: Kraft counting bound — the table size is forced
-- ============================================================

/-- If every mass is ≥ 1 and the masses sum to ≤ 2^B, there are at most 2^B items.
In our setting the masses are `2^(B - cost)` over admissible patches of ONE fixed
length after ONE context: fixed-length string events are disjoint under q, so
their Kraft sum is ≤ 1 and the class has ≤ 2^B elements. (Mixed lengths do not
satisfy the Kraft hypothesis — see the header note.) -/
theorem kraft_count (masses : List Nat) (B : Nat)
    (hge : ∀ m ∈ masses, 1 ≤ m) (hk : masses.sum ≤ 2 ^ B) :
    masses.length ≤ 2 ^ B :=
  Nat.le_trans (ge1_length_le_sum masses hge) hk

/-- Same statement directly over admissible patch *costs* of one length class. -/
theorem kraft_from_costs (costs : List Nat) (B : Nat)
    (hcosts : ∀ c ∈ costs, c ≤ B)
    (hk : (costs.map (fun c => 2 ^ (B - c))).sum ≤ 2 ^ B) :
    costs.length ≤ 2 ^ B := by
  have hge : ∀ m ∈ costs.map (fun c => 2 ^ (B - c)), 1 ≤ m := by
    intro m hm
    simp only [List.mem_map] at hm
    obtain ⟨c, -, rfl⟩ := hm
    exact Nat.two_pow_pos _
  have h1 := ge1_length_le_sum _ hge
  have h2 : costs.length = (costs.map (fun c => 2 ^ (B - c))).length :=
    (List.length_map _).symm
  omega

-- ============================================================
-- P2: the greedy cut — budget invariant and uniform band
-- ============================================================

/-- Greedy cut. Accumulate byte costs; when folding the next byte `c` would push
the running cost over `B`, the current patch is closed and `c` opens the next one.
Returns `(closed patches, final open patch)`. -/
def cut (B : Nat) (acc : List Nat) : List Nat → List (List Nat) × List Nat
  | [] => ([], acc)
  | c :: cs =>
      if acc.sum + c ≤ B then cut B (acc ++ [c]) cs
      else
        let (cl, op) := cut B [c] cs
        (acc :: cl, op)

theorem cut_nil (B : Nat) (acc : List Nat) : cut B acc [] = ([], acc) := rfl

theorem cut_cons_pos {B : Nat} {acc : List Nat} {c : Nat} {cs : List Nat}
    (h : acc.sum + c ≤ B) :
    cut B acc (c :: cs) = cut B (acc ++ [c]) cs := by
  simp only [cut, if_pos h]

theorem cut_cons_neg {B : Nat} {acc : List Nat} {c : Nat} {cs : List Nat}
    (h : ¬ (acc.sum + c ≤ B)) :
    cut B acc (c :: cs) = (acc :: (cut B [c] cs).1, (cut B [c] cs).2) := by
  simp only [cut, if_neg h]

/-- Every closed patch costs ≥ B - 7 (bytes cost ≤ 8), and the final open patch
costs ≤ B. So every *interior* patch sits in the band `[B-7, B]`: uniform
information load, guaranteed by construction. -/
theorem cut_band (B : Nat) (acc : List Nat) (cs : List Nat)
    (hacc : acc.sum ≤ B) (hcs : ∀ c ∈ cs, c ≤ 8) (hB : 8 ≤ B) :
    (∀ p ∈ (cut B acc cs).1, B - 7 ≤ p.sum) ∧ ((cut B acc cs).2).sum ≤ B := by
  induction cs generalizing acc with
  | nil =>
      rw [cut_nil]
      exact ⟨fun p hp => by simp at hp, hacc⟩
  | cons c cs ih =>
      have hcB : c ≤ B := Nat.le_trans (hcs c (by simp)) hB
      have hcs' : ∀ x ∈ cs, x ≤ 8 := fun x hx => hcs x (List.mem_cons_of_mem _ hx)
      by_cases h : acc.sum + c ≤ B
      · rw [cut_cons_pos h]
        refine ih (acc ++ [c]) ?_ hcs'
        have h1 : (acc ++ [c]).sum = acc.sum + c := by
          rw [List.sum_append, List.sum_singleton]
        omega
      · rw [cut_cons_neg h]
        obtain ⟨h1, h2⟩ := ih [c] (by show [c].sum ≤ B; rw [List.sum_singleton]; exact hcB) hcs'
        refine ⟨?_, h2⟩
        intro p hp
        rcases List.mem_cons.mp hp with rfl | hp'
        · -- `acc` was closed by byte `c`: acc.sum + c ≥ B + 1, and c ≤ 8
          have hc8 : c ≤ 8 := hcs c (by simp)
          omega
        · exact h1 p hp'

-- ============================================================
-- P3: the explicit code — cumulative-mass (arithmetic-code skeleton)
-- ============================================================

/-- Suffix sums of the mass list. For patch `i` (in the fixed order of the list),
`csum ms[i+1]` is the total mass of everything *after* it — the starting offset of
its dyadic interval, i.e. its index. This is the Shannon–Fano–Elias /
arithmetic-coding skeleton: code length = information content. -/
def csum : List Nat → List Nat
  | [] => [0]
  | m :: ms => (m :: ms).sum :: csum ms

theorem nodup_tail {l : List Nat} (h : l.Nodup) : l.tail.Nodup := by
  cases l with
  | nil => exact h
  | cons a t =>
      have htail : (a :: t).tail = t := rfl
      rw [htail]
      exact (List.pairwise_cons.mp h).2

theorem csum_le : ∀ (l : List Nat), ∀ x ∈ csum l, x ≤ l.sum := by
  intro l
  induction l with
  | nil => intro x hx; simp only [csum, List.mem_singleton] at hx; omega
  | cons m ms ih =>
      intro x hx
      simp only [csum, List.mem_cons] at hx
      rcases hx with rfl | hx'
      · omega
      · have h1 := ih x hx'
        have h2 : ms.sum + m = (m :: ms).sum := by rw [List.sum_cons]; omega
        omega

/-- Every code emitted is strictly below the total mass — hence, given the Kraft
bound, strictly below `2^B`: all indices land inside the table. -/
theorem code_lt (l : List Nat) (h1 : ∀ x ∈ l, 1 ≤ x) :
    ∀ x ∈ (csum l).tail, x < l.sum := by
  cases l with
  | nil => intro x hx; simp [csum] at hx
  | cons m ms =>
      intro x hx
      have htail : (csum (m :: ms)).tail = csum ms := rfl
      rw [htail] at hx
      have hle := csum_le ms x hx
      have hm : 1 ≤ m := h1 m (by simp)
      have h2 : ms.sum + m = (m :: ms).sum := by rw [List.sum_cons]; omega
      omega

/-- Codes are pairwise distinct: the mass list is all-positive, so the suffix sums
strictly decrease along the list. Distinct patches ⟹ distinct indices. -/
theorem csum_nodup (l : List Nat) (h1 : ∀ x ∈ l, 1 ≤ x) : (csum l).Nodup := by
  induction l with
  | nil => simp [csum]
  | cons m ms ih =>
      have hmem : 1 ≤ m := h1 m (by simp)
      have htail_nodup : (csum ms).Nodup :=
        ih (fun x hx => h1 x (List.mem_cons_of_mem _ hx))
      have hhead : ∀ x ∈ csum ms, x < (m :: ms).sum := by
        intro x hx
        have hle := csum_le ms x hx
        have h2 : ms.sum + m = (m :: ms).sum := by rw [List.sum_cons]; omega
        omega
      have hnotin : (m :: ms).sum ∉ csum ms := by
        intro hx
        have := hhead _ hx
        omega
      simp only [csum, List.nodup_cons]
      exact ⟨hnotin, htail_nodup⟩

/-- **The rate-C vocabulary exists, constructively (per length class).** Given a
list of admissible patch costs of one fixed length (per context) whose dyadic
masses satisfy the Kraft bound `Σ 2^(B - cost) ≤ 2^B`, the cumulative-mass map
assigns every patch a distinct index in `[0, 2^B)`. Combined with
`kraft_from_costs`, each (context, length) class *fits* the `2^B`-entry table and
this map realizes the assignment; the full address is `(index, ℓ)`. -/
theorem ratec_codes (costs : List Nat) (B : Nat)
    (hk : (costs.map (fun c => 2 ^ (B - c))).sum ≤ 2 ^ B) :
    ((csum (costs.map (fun c => 2 ^ (B - c)))).tail).Nodup
      ∧ (∀ x ∈ (csum (costs.map (fun c => 2 ^ (B - c)))).tail, x < 2 ^ B) := by
  have h1 : ∀ x ∈ costs.map (fun c => 2 ^ (B - c)), 1 ≤ x := by
    intro x hx
    simp only [List.mem_map] at hx
    obtain ⟨c, -, rfl⟩ := hx
    exact Nat.two_pow_pos _
  refine ⟨nodup_tail (csum_nodup _ h1), ?_⟩
  intro x hx
  have hlt := code_lt _ h1 x hx
  exact Nat.lt_of_lt_of_le hlt hk

-- ============================================================
-- Concrete demonstrations
-- ============================================================

-- Budget B = 14 (i.e. C = 16). Greedy cut of a cost stream: closed-patch sums
-- all land inside the band [B-7, B] = [7, 14].
#eval (cut 14 [] [3, 5, 2, 8, 1, 4, 6, 3, 7, 2, 1, 1, 5]).1.map (·.sum)

-- The codes: masses 2^(14 - cost) for patches with costs [2, 4, 6, 8];
-- four distinct indices, all < 2^14.
#eval (csum ([2, 4, 6, 8].map (fun c => 2 ^ (14 - c)))).tail

#print axioms kraft_count
#print axioms kraft_from_costs
#print axioms cut_band
#print axioms ratec_codes
