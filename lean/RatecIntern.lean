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

S6 `resolve_eq_gid` / `ids_stable_interleave` — the READ-STABLE merge
protocol (PLAN_v0.9 P4.6, licenses the cold-book parallel intern): a
reader probing the live tables while the ordered merge inserts may
linearize at ANY merge step k; whatever the probe returns (a frozen
book hit, a primary hit, or a miss resolved by the merge path), the
resolved id is the sequential id. The protocol the C must maintain for
the model to apply:
  * the primary grows by inserting the novel list IN ORDER — after k
    inserts its contents are exactly `(novel book stream).take k`
    (`primaryAt` below); no reordering, ever;
  * the frozen table is read-only for the merge's duration;
  * the epoch check makes every probe ATOMIC w.r.t. restructuring: a
    probe either linearizes at a step boundary (may read the primary)
    or reads only frozen data (a book hit or a miss). The
    memory-ordering discipline (acquire/release atomics) that makes
    the linearization hold is the C implementation's responsibility;
    this file proves the ALGORITHMIC soundness: at every linearization
    point, resolution = the sequential id — there is no interleaving
    and no step k that changes an id, and a miss has no third outcome.

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

/-! ## S6 — the read-stable merge protocol (PLAN_v0.9 P4.6)

The model of a reader concurrent with the ordered merge: the merge
walks the novel list in order (insert k = the k-th novel content), a
reader's probe linearizes at some step k (the epoch check makes the
probe atomic w.r.t. restructuring), and the reader resolves from the
frozen book, the growing primary, or — on a miss — the merge path.
The state the primary can present at step k is a PREFIX of the novel
list, and a prefix hit carries the same global id as the final list.
-/

/-- The growing primary's contents at merge step k: exactly the first
k novel contents, in merge order. The C must insert in merge order for
this to hold — it is the invariant the epoch check protects. -/
def primaryAt (book : List Nat) (stream : List Nat) (k : Nat) : List Nat :=
  (novel book stream).take k

/-- A probe against the live tables at merge step k: the frozen book
(read-only, always stable), then the epoch-checked primary, else a
miss. A primary hit returns the stored GLOBAL id (the merge assigns
book.length + the local position at insert time). -/
def probeAt (book : List Nat) (stream : List Nat) (x : Nat) (k : Nat) :
    Option Nat :=
  if x ∈ book then some (posOf x book)
  else if x ∈ primaryAt book stream k then
    some (book.length + posOf x (primaryAt book stream k))
  else none

/-- The global id: the content's position in the FINAL merged contents
(the book with the novel list appended — S5's `mergeFold_eq` is
exactly this shape with the frozen book as the accumulator). -/
def gid (book : List Nat) (stream : List Nat) (x : Nat) : Nat :=
  posOf x (book ++ novel book stream)

/-- The reader's total resolution: the probe's id, or — on a miss — the
merge path's id (the content will be inserted by the ordered merge and
the remap assigns its global id). -/
def resolve (book : List Nat) (stream : List Nat) (x : Nat) (k : Nat) :
    Nat :=
  match probeAt book stream x k with
  | some id => id
  | none => book.length + posOf x (novel book stream)

/-! ### the position lemmas -/

/-- x in the prefix ⇒ x's position is stable under appending. -/
theorem posOf_mem_append (x : Nat) (l : List Nat) : ∀ (r : List Nat),
    x ∈ l → posOf x (l ++ r) = posOf x l := by
  induction l with
  | nil => intro r h; cases h
  | cons y rest ih =>
      intro r h
      simp only [List.cons_append, posOf]
      by_cases hxy : y = x
      · simp [hxy]
      · have hrest : x ∈ rest := by
          rcases List.mem_cons.mp h with he | hmem
          · exact absurd he.symm hxy
          · exact hmem
        rw [ih r hrest]

/-- x in a prefix of the list ⇒ its position in the prefix is its
position in the whole (first occurrence). -/
theorem posOf_take_mem (x : Nat) (l : List Nat) : ∀ (k : Nat),
    x ∈ l.take k → posOf x l = posOf x (l.take k) := by
  induction l with
  | nil => intro k h; exact absurd h (by simp)
  | cons y rest ih =>
      intro k h
      cases k with
      | zero => exact absurd h (by simp)
      | succ k' =>
          simp only [List.take] at h ⊢
          by_cases hxy : y = x
          · simp only [posOf, hxy, reduceIte]
          · have hmem : x ∈ rest.take k' := by
              rcases List.mem_cons.mp h with he | hm
              · exact absurd he.symm hxy
              · exact hm
            simp only [posOf, hxy, reduceIte]
            rw [ih k' hmem]

/-! ### the novel-list lemmas -/

/-- A novel content is never in the book (novel only keeps non-members). -/
theorem novel_not_mem_book (x : Nat) (book : List Nat) : ∀ (stream : List Nat),
    x ∈ novel book stream → x ∉ book := by
  intro stream
  induction stream generalizing book with
  | nil => intro h; cases h
  | cons y rest ih =>
      intro h
      simp only [novel] at h
      split at h
      · exact ih book h
      · next hnb =>
          rcases List.mem_cons.mp h with he | hm
          · subst he
            exact hnb
          · exact fun hb => ih (book ++ [y]) hm (List.mem_append_left [y] hb)

/-- Every stream content lives in the book or in the novel list — no
third place (this is what makes the miss outcome closed). -/
theorem mem_stream_novel_or_book (x : Nat) (book : List Nat) :
    ∀ (stream : List Nat), x ∈ stream →
      x ∈ book ∨ x ∈ novel book stream := by
  intro stream
  induction stream generalizing book with
  | nil => intro h; cases h
  | cons y rest ih =>
      intro h
      rcases List.mem_cons.mp h with he | hm
      · subst he
        by_cases hb : x ∈ book
        · exact Or.inl hb
        · simp only [novel, hb, reduceIte]
          exact Or.inr List.mem_cons_self
      · by_cases hb : y ∈ book
        · simp only [novel, hb, reduceIte]
          exact ih book hm
        · simp only [novel, hb, reduceIte]
          rcases ih (book ++ [y]) hm with hb' | hn'
          · rcases List.mem_append.mp hb' with h1 | h2
            · exact Or.inl h1
            · simp only [List.mem_singleton] at h2
              subst h2
              exact Or.inr List.mem_cons_self
          · exact Or.inr (List.mem_cons_of_mem _ hn')

/-- A novel content's global id: the book's length plus its position in
the novel list (the book occupies the id space's head). -/
theorem posOf_novel_append (x : Nat) : ∀ (book nl : List Nat),
    x ∉ book → x ∈ nl →
    posOf x (book ++ nl) = book.length + posOf x nl := by
  intro book
  induction book with
  | nil => intro nl _ hn; simp
  | cons y rest ih =>
      intro nl hnb hn
      simp only [List.cons_append, posOf]
      have hne : ¬ (y = x) := by
        intro he
        apply hnb
        rw [he]
        exact List.mem_cons_self
      have hnrest : x ∉ rest := by
        intro hr
        exact hnb (List.mem_cons_of_mem y hr)
      simp only [hne, reduceIte]
      rw [ih nl hnrest hn]
      rw [List.length_cons]
      omega

/-! ### S6 — the resolution theorems -/

/-- S6a — every probe HIT at any step k returns exactly the global id:
a frozen book hit (prefix-stable) and a primary hit (the merge-order
prefix carries the final position) coincide with the final list. -/
theorem probe_hit_eq_gid (book : List Nat) (stream : List Nat) (x : Nat)
    (k : Nat) (id : Nat) (h : probeAt book stream x k = some id) :
    id = gid book stream x := by
  unfold probeAt primaryAt at h
  unfold gid
  split at h
  · next hb =>
      simp at h
      rw [← h]
      rw [posOf_mem_append x book (novel book stream) hb]
  · next hnb =>
      split at h
      · next hp =>
          simp at h
          rw [← h]
          rw [← posOf_take_mem x (novel book stream) k hp]
          rw [posOf_novel_append x book (novel book stream) hnb
            (List.take_subset k (novel book stream) hp)]
      · simp at h

/-- S6b — a miss is not a third outcome: the content lives in the
novel list (it is a stream content), the ordered merge will insert it,
and the merge path resolves exactly the global id. -/
theorem probe_miss_eq_gid (book : List Nat) (stream : List Nat) (x : Nat)
    (k : Nat) (h : probeAt book stream x k = none) (hx : x ∈ stream) :
    gid book stream x = book.length + posOf x (novel book stream) := by
  have hnb : ¬ (x ∈ book) := by
    intro hb
    unfold probeAt at h
    simp only [hb, reduceIte] at h
    simp at h
  rcases mem_stream_novel_or_book x book stream hx with hb | hn
  · exact absurd hb hnb
  · unfold gid
    rw [posOf_novel_append x book (novel book stream) hnb hn]

/-- S6 — THE theorem: whatever the probe returns at ANY step k (any
interleaving that respects the epoch check), the resolution is the
sequential id. This is the read-stable property: concurrency cannot
change an id, so the pre-probe may overlap the ordered merge once the
C maintains the two protocol invariants (merge-order inserts; frozen
read-only table). -/
theorem resolve_eq_gid (book : List Nat) (stream : List Nat) (x : Nat)
    (k : Nat) (hx : x ∈ stream) :
    resolve book stream x k = gid book stream x := by
  unfold resolve
  cases h : probeAt book stream x k with
  | some id =>
      show id = gid book stream x
      exact probe_hit_eq_gid book stream x k id h
  | none =>
      show book.length + posOf x (novel book stream) = gid book stream x
      exact (probe_miss_eq_gid book stream x k h hx).symm

/-- The sequential ids for a stream interned into a pre-existing book:
first-seen order with the book occupying the id space's head. (The
book's own duplicate-freeness is the engine's maintained invariant —
the snapshot load path validates it.) -/
def idsSeqBook (book : List Nat) (stream : List Nat) : List Nat :=
  stream.map (fun c => gid book stream c)

/-- pointwise map equality (core has no map_congr_left; one-liner). -/
private theorem map_congr_left {α β : Type} (f g : α → β) (l : List α)
    (h : ∀ a, a ∈ l → f a = g a) : l.map f = l.map g := by
  induction l with
  | nil => rfl
  | cons y rest ih =>
      simp only [List.map_cons]
      rw [h y List.mem_cons_self,
        ih (fun a ha => h a (List.mem_cons_of_mem _ ha))]

/-- S6c — the per-stream statement: for ANY assignment of
linearization steps to contents (the interleaving, here any function
ks : content ↦ step), the ids the protocol assigns are exactly the
sequential ids with the book as head. Together with S5 (the ordered
merge itself computes the first-seen order for any chunking), this
discharges the read-stable design: no interleaving changes an id, and
no resolution path returns anything but the sequential id. -/
theorem ids_stable_interleave (book : List Nat) (stream : List Nat)
    (ks : Nat → Nat) :
    stream.map (fun c => resolve book stream c (ks c))
      = idsSeqBook book stream := by
  apply map_congr_left
  intro c hc
  exact resolve_eq_gid book stream c (ks c) hc

end RatecIntern
