/-!
# Rate-C parallel intern order — machine-checked (pure Lean 4, no mathlib)

Companion to `RateC.lean` (vocabulary), `RatecSerial.lean` (execution
model), `RatecTiling.lean` (patcher tiling). This file is the
machine-checked form of the v0.9.0 parallel intern pass's correctness
claim:

S5 `ids_par_eq_seq` — interning a span tiling through K contiguous
chunks (private per-chunk first-seen tables, then an ordered merge
appending each chunk's locally-new contents) assigns the SAME ids as
the sequential first-seen pass over the whole stream — for ANY chunk
count and any chunking.

The model mirrors `iintern_job` / `imerge` in `_rcengine.c`:
- a patch's identity is its CONTENT (a Nat stands in for the bytes);
- `novel seen l` walks l keeping every content not already in `seen`,
  threading each kept content into `seen` — exactly the sequential
  intern AND one worker's private table (insert order = first-seen);
- `dedup l = novel [] l` — the sequential first-seen order;
- `mergeFold` = the ordered merge, folding chunks left to right,
  appending each chunk's novel contents (the merge's book probes);
- ids = for each span, the position of its content in the global
  first-seen list (`posOf`).
-/

namespace RatecIntern

/-- Walk l keeping every content not already in `seen`; each kept
content joins `seen`. This is the intern: a private table's insert
order (and the sequential engine's book order) is exactly
`novel [] l`. -/
def novel (seen : List Nat) : List Nat → List Nat
  | [] => []
  | x :: rest =>
      if x ∈ seen then novel seen rest
      else x :: novel (seen ++ [x]) rest

/-- The sequential first-seen order. -/
def dedup (l : List Nat) : List Nat := novel [] l

/-- Position of the FIRST occurrence of x (0 if absent; ids only ever
query contents that are present in the merged list). -/
def posOf (x : Nat) : List Nat → Nat
  | [] => 0
  | y :: rest => if y = x then 0 else posOf x rest + 1

/-- The sequential ids: per span, the position of its content in the
global first-seen list. -/
def idsSeq (contents : List Nat) : List Nat :=
  contents.map (fun c => posOf c (dedup contents))

/-- The ordered merge: fold chunks left to right, appending each
chunk's novel contents; the final accumulator is the merged list.
Mirrors `imerge`: chunks in order, each chunk's distinct patches in
local first-seen order, book probes assign global ids. -/
def mergeFold (acc : List Nat) : List (List Nat) → List Nat
  | [] => acc
  | ch :: rest => mergeFold (acc ++ novel acc ch) rest

/-- The parallel ids: per chunk, per span, the position of the span's
content in the merged list. -/
def idsPar (chunks : List (List Nat)) : List Nat :=
  let merged := mergeFold [] chunks
  (chunks.map (fun ch => ch.map (fun c => posOf c merged))).flatten

/-! ## the structural lemma: novel decomposes per chunk -/

theorem novel_append (seen : List Nat) (a b : List Nat) :
    novel seen (a ++ b) =
      novel seen a ++ novel (seen ++ novel seen a) b := by
  induction a generalizing seen with
  | nil => simp [novel]
  | cons y as ih =>
      by_cases hseen : y ∈ seen
      · simp only [List.cons_append, novel, hseen, reduceIte]
        rw [ih]
      · simp only [List.cons_append, novel, hseen, reduceIte]
        rw [ih, List.append_assoc, List.cons_append, List.nil_append]

/-! ## the merge computes the global first-seen order -/

/-- map distributes over flatten (the per-chunk id maps flatten to
the same ids as the flat map). -/
theorem flatten_map_map (g : Nat → Nat) (chunks : List (List Nat)) :
    (chunks.map (fun ch => ch.map g)).flatten = chunks.flatten.map g := by
  induction chunks with
  | nil => simp
  | cons ch rest ih =>
      simp only [List.map_cons, List.flatten_cons, List.map_append, ih]

theorem mergeFold_eq (acc : List Nat) (chunks : List (List Nat)) :
    mergeFold acc chunks =
      acc ++ novel acc chunks.flatten := by
  induction chunks generalizing acc with
  | nil => simp [mergeFold, novel]
  | cons ch rest ih =>
      simp only [mergeFold, List.flatten_cons, ih]
      rw [novel_append, List.append_assoc]

/-- The parallel merge's id list IS the sequential first-seen list. -/
theorem merged_is_dedup (chunks : List (List Nat)) :
    mergeFold [] chunks = dedup chunks.flatten := by
  rw [mergeFold_eq, List.nil_append]
  rfl

/-! ## S5 — the main theorem -/

theorem ids_par_eq_seq (chunks : List (List Nat)) :
    idsPar chunks = idsSeq chunks.flatten := by
  unfold idsPar idsSeq
  rw [merged_is_dedup, flatten_map_map]

end RatecIntern
