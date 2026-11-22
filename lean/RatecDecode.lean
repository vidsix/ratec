/-!
# Rate-C decode completeness — machine-checked (pure Lean 4, no mathlib)

Companion to `RatecIntern.lean` (S5/S6). This file is the machine-checked
form of the decode-completeness claim (PLAN_v0.9 P4, S7):

S7 `decode_roundtrip` — with the book built by the intern (first-seen
order over the stream's patch contents) and the ids the intern assigns,
`decode` reproduces the stream EXACTLY: every id resolves to its patch's
byte content and the concatenation is the stream flattened. No third
outcome: `decode` is `some bytes` exactly when every id is in range,
else the documented error (`none`; the C raises coded `ARGUMENT`,
"unknown patch id"). The C's two-pass decode (bounds pass, then gather)
makes the outcome ATOMIC — a bad id anywhere yields the error and no
partial output — which the `all`-guard models.

Model (mirrors `decode_bytes` in `_rcengine.c`):
- a patch's identity is its CONTENT (a List Nat stands in for its bytes);
- the book is `book patches` — the intern's first-seen order (S5);
- an id is the patch's first-occurrence position (`posOf`);
- `inBounds` is the bounds pass, `fetch` the gather read.
-/

namespace RatecDecode

/-- The intern's first-seen walk over patch contents (S5's `novel`,
element type = a patch's byte list). -/
def novel (seen : List (List Nat)) : List (List Nat) → List (List Nat)
  | [] => []
  | x :: rest =>
      if x ∈ seen then novel seen rest
      else x :: novel (seen ++ [x]) rest

/-- The book: first-seen order of the stream's patch contents. -/
def book (patches : List (List Nat)) : List (List Nat) := novel [] patches

/-- Position of the FIRST occurrence of x (0 if absent; ids only ever
query contents that are present). -/
def posOf (x : List Nat) : List (List Nat) → Nat
  | [] => 0
  | y :: rest => if y = x then 0 else posOf x rest + 1

/-- The bounds pass: is `id` a valid book index? -/
def inBounds (id : Nat) : List (List Nat) → Bool
  | [] => false
  | _ :: rest =>
      match id with
      | 0 => true
      | k + 1 => inBounds k rest

/-- The gather read: the id's book content ([] can't be reached when
inBounds passed). -/
def fetch (id : Nat) : List (List Nat) → List Nat
  | [] => []
  | c :: rest =>
      match id with
      | 0 => c
      | k + 1 => fetch k rest

/-- Decode: bounds pass over ALL ids, then the gather — all-or-error,
no partial output on a bad id. -/
def decode (B : List (List Nat)) (ids : List Nat) : Option (List Nat) :=
  if ids.all (fun id => inBounds id B) then
    some ((ids.map (fun id => fetch id B)).flatten)
  else none

/-- The ids the intern assigns: per patch, its first-occurrence position
in the book. -/
def idsSeq (patches : List (List Nat)) : List Nat :=
  patches.map (fun c => posOf c (book patches))

/-! ### the book-lookup lemmas -/

/-- bounds pass ⟺ index within length (the C's `id >= bn` check). -/
theorem inBounds_iff_lt : ∀ (id : Nat) (B : List (List Nat)),
    inBounds id B = true ↔ id < B.length := by
  intro id B
  induction B generalizing id with
  | nil => cases id <;> simp [inBounds]
  | cons c rest ih =>
      cases id with
      | zero =>
          simp only [inBounds, List.length_cons, reduceIte]
          simp
      | succ k =>
          simp only [inBounds, List.length_cons, reduceIte]
          rw [ih k]
          omega

/-- A member's position in the intern walk is a real index. -/
theorem posOf_lt_of_mem {x : List Nat} :
    ∀ (stream : List (List Nat)) (seen : List (List Nat)),
      x ∉ seen → x ∈ stream →
      posOf x (novel seen stream) < (novel seen stream).length := by
  intro stream
  induction stream with
  | nil => intro seen _ h; cases h
  | cons y rest ih =>
      intro seen hns hin
      simp only [novel]
      split
      · next hy =>
          have hx : x ∈ rest := by
            rcases List.mem_cons.mp hin with he | hm
            · subst he
              exact absurd hy hns
            · exact hm
          exact ih seen hns hx
      · next =>
          rw [List.length_cons]
          rcases List.mem_cons.mp hin with he | hm
          · subst he
            simp only [posOf, List.mem_cons_self, reduceIte]
            omega
          · by_cases hne : y = x
            · simp only [posOf, hne, reduceIte]
              omega
            · simp only [posOf, hne, reduceIte]
              have hx : x ∉ seen ++ [y] := by
                intro hb
                rcases List.mem_append.mp hb with h1 | h2
                · exact hns h1
                · simp only [List.mem_singleton] at h2
                  exact hne h2.symm
              have hlt := ih (seen ++ [y]) hx hm
              exact Nat.succ_lt_succ hlt

/-- The gather read at a member's first-occurrence position IS the
member: the book lookup never lies. -/
theorem fetch_posOf_self {x : List Nat} :
    ∀ (stream : List (List Nat)) (seen : List (List Nat)),
      x ∉ seen → x ∈ stream →
      fetch (posOf x (novel seen stream)) (novel seen stream) = x := by
  intro stream
  induction stream with
  | nil => intro seen _ h; cases h
  | cons y rest ih =>
      intro seen hns hin
      simp only [novel]
      split
      · next hy =>
          have hx : x ∈ rest := by
            rcases List.mem_cons.mp hin with he | hm
            · subst he
              exact absurd hy hns
            · exact hm
          exact ih seen hns hx
      · next =>
          rcases List.mem_cons.mp hin with he | hm
          · subst he
            simp only [posOf, List.mem_cons_self, reduceIte]
            simp only [fetch, reduceIte]
          · by_cases hne : y = x
            · simp only [posOf, hne, reduceIte]
              simp only [fetch, hne, reduceIte]
            · simp only [posOf, hne, reduceIte]
              simp only [fetch, hne, reduceIte]
              have hx : x ∉ seen ++ [y] := by
                intro hb
                rcases List.mem_append.mp hb with h1 | h2
                · exact hns h1
                · simp only [List.mem_singleton] at h2
                  exact hne h2.symm
              have := ih (seen ++ [y]) hx hm
              simp only [this]

/-- pointwise map equality (core has no map_congr_left; one-liner). -/
private theorem map_congr_left {α β : Type} (f g : α → β) (l : List α)
    (h : ∀ a, a ∈ l → f a = g a) : l.map f = l.map g := by
  induction l with
  | nil => rfl
  | cons y rest ih =>
      simp only [List.map_cons]
      rw [h y List.mem_cons_self,
        ih (fun a ha => h a (List.mem_cons_of_mem _ ha))]

/-! ### S7 — the completeness theorems -/

/-- Every id the intern assigns passes the bounds pass (Lemma A per
element). -/
theorem ids_in_range (patches : List (List Nat)) :
    (idsSeq patches).all (fun id => inBounds id (book patches)) = true := by
  unfold idsSeq book
  rw [List.all_eq_true]
  intro id hid
  rcases List.mem_map.mp hid with ⟨c, hc, heq⟩
  have gen : posOf c (novel [] patches) < (novel [] patches).length :=
    posOf_lt_of_mem patches [] (by simp) hc
  rw [← heq]
  exact (inBounds_iff_lt _ _).mpr gen

/-- S7 — THE theorem: decoding the intern's ids through the intern's
book reproduces the stream exactly, byte for byte. -/
theorem decode_roundtrip (patches : List (List Nat)) :
    decode (book patches) (idsSeq patches) = some (patches.flatten) := by
  unfold decode idsSeq book
  have hin_all : (patches.map (fun c => posOf c (novel [] patches))).all
      (fun id => inBounds id (novel [] patches)) = true := by
    rw [List.all_eq_true]
    intro a ha
    rcases List.mem_map.mp ha with ⟨c, hc, heq⟩
    have hlt := posOf_lt_of_mem patches [] (by simp) hc
    rw [← heq]
    exact (inBounds_iff_lt _ _).mpr hlt
  split
  · have hmap : (patches.map (fun c => posOf c (novel [] patches))).map
        (fun id => fetch id (novel [] patches)) = patches := by
      rw [List.map_map]
      show (patches.map fun c =>
          fetch (posOf c (novel [] patches)) (novel [] patches)) = patches
      exact (map_congr_left (fun c => fetch (posOf c (novel [] patches))
                (novel [] patches)) (fun c => c) patches
        (fun c hc => fetch_posOf_self patches [] (by simp) hc)).trans
        (List.map_id patches)
    rw [hmap]
  · next hn => exact absurd hin_all hn

/-- witness lemma: a failed all-guard yields a violating element -/
private theorem all_false_witness {p : Nat → Bool} :
    ∀ (ids : List Nat), ids.all p = false →
      ∃ id, id ∈ ids ∧ p id = false := by
  intro ids
  induction ids with
  | nil => intro h; simp [List.all_nil] at h
  | cons a rest ih =>
      intro h
      simp only [List.all_cons] at h
      cases hp : p a with
      | false => exact ⟨a, List.mem_cons_self .., hp⟩
      | true =>
          have hr : rest.all p = false := by
            cases hrest : rest.all p with
            | false => rfl
            | true =>
                rw [hp, hrest] at h
                simp at h
          rcases ih hr with ⟨id, hid, hbad⟩
          exact ⟨id, List.mem_cons_of_mem _ hid, hbad⟩

/-- No third outcome, error side: a decode fails EXACTLY when some id
fails the bounds pass — the documented `ARGUMENT` / "unknown patch id"
error, never a partial output. -/
theorem decode_error_iff (B : List (List Nat)) (ids : List Nat) :
    decode B ids = none ↔ ∃ id, id ∈ ids ∧ inBounds id B = false := by
  unfold decode
  constructor
  · intro h
    by_cases hall : ids.all (fun id => inBounds id B) = true
    · rw [hall] at h
      simp at h
    · have hfalse : ids.all (fun id => inBounds id B) = false := by
        cases hlist : ids.all (fun id => inBounds id B) with
        | true => exact absurd hlist hall
        | false => rfl
      exact all_false_witness ids hfalse
  · -- the documented error: any out-of-range id fails the bounds pass,
    -- and the two-pass decode emits NO partial output
    intro hex
    rcases hex with ⟨id, hid, hbad⟩
    by_cases hall : ids.all (fun id => inBounds id B) = true
    · exfalso
      have := List.all_eq_true.mp hall id hid
      rw [this] at hbad
      simp at hbad
    · have hfalse : ids.all (fun id => inBounds id B) = false := by
        cases hlist : ids.all (fun id => inBounds id B) with
        | true => exact absurd hlist hall
        | false => rfl
      rw [hfalse]
      simp

end RatecDecode
