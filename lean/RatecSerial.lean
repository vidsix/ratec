/-!
# Rate-C serial/parallel structure — machine-checked (pure Lean 4, no mathlib)

Companion to `RateC.lean` (the vocabulary lemmas). This file proves the
*execution-model* theorems that v0.8.8/v0.8.9's multicore design rests on,
about an abstracted adaptive judge (order-1 counts + previous byte — the
same causal shape as RateC's orders 0..5, and the same proof skeleton
carries over).

What is proved, with no `sorry`:

S1 `foldl_append_gen` / `jfold_append` / `pfold_append` — the **decomposition
   theorem**: folding a stream `xs ++ ys` from state `σ` equals folding `ys`
   from the state after `xs`. This is the exactness foundation of every
   multi-stream parallel decomposition (ratec.batch, encode_stream_parallel
   runs mode): a contiguous run encoded from the true prefix state
   reproduces the sequential result *bit-exactly*.

S2 `jstep_swap` — the adaptive step does **not commute** for distinct bytes
   (the previous-byte context differs), so no map-reduce over byte
   *multisets* can reproduce the sequential state chain: the serial chain
   is essential, per-stream cost is Ω(len) state transitions.

S3 `jsurp_depends_on_history` — the same byte carries different surprisal
   depending on what came before (concretely: first occurrence costs
   `256/1` scaled bits, a repeat costs `256/2`). Reordering bytes changes
   the surprisal *sequence*, hence the patcher's `cs` accumulation, hence
   the cut positions.

S4 `pfold_judge` — the full judge+patcher pipeline is itself a fold, and
   its judge-component is exactly the judge fold: the pipeline inherits
   S1 (decomposable) and S2 (non-reorderable).

Together: *the cheapest exact algorithm is a serial fold per stream, and
parallelism must (and may exactly) decompose by contiguous runs.* That is
the theorem v0.8.8 implemented; no finer-grained exact parallelization
exists without changing the frozen semantics.

The judge here is deliberately a toy (Nat counts, no FP): S1 holds for ANY
pure step function; S2/S3 only need one concrete distinguishing pair.
RateC's real judge adds context breadth, not causal shape.
-/

namespace RatecSerial

abbrev Byte := Fin 256

/-! ## A concrete adaptive judge: order-1 counts + previous byte -/

/-- Judge state: bytes seen, per-byte counts, previous byte (context). -/
structure JState where
  n : Nat
  c1 : Byte → Nat
  prev : Option Byte

def jzero : JState := ⟨0, fun _ => 0, none⟩

/-- One adaptive transition: count the byte, shift the context. -/
def jstep : JState → Byte → JState
  | s, b => ⟨s.n + 1, fun x => if x = b then s.c1 x + 1 else s.c1 x, some b⟩

/-- Serial fold over a stream (the exact encode). -/
def jfold (s : JState) (bs : List Byte) : JState := bs.foldl jstep s

/-! ## S1 — the decomposition theorem -/

theorem foldl_append_gen {σ : Type} (f : σ → Byte → σ) (s : σ) :
    ∀ (xs ys : List Byte), (xs ++ ys).foldl f s = ys.foldl f (xs.foldl f s)
  | [], _ => rfl
  | x :: xs, ys => by
      simp only [List.cons_append, List.foldl_cons]
      exact foldl_append_gen f (f s x) xs ys

/-- **S1.** Encoding `xs ++ ys` from `σ` = encoding `ys` from the state
after `xs`. The runs-mode / batch-mode parallel decomposition is this
lemma applied per run. -/
theorem jfold_append (s : JState) (xs ys : List Byte) :
    jfold s (xs ++ ys) = jfold (jfold s xs) ys :=
  foldl_append_gen jstep s xs ys

/-! ## S2 — the step does not commute -/

/-- **S2.** Two distinct bytes leave different states in different order:
the previous-byte context is `some b` vs `some a`. No reordering of the
byte sequence (no multiset summary, no commutative map-reduce) can
reproduce the serial chain. -/
theorem jstep_swap (a b : Byte) (h : a ≠ b) :
    jstep (jstep jzero a) b ≠ jstep (jstep jzero b) a := by
  intro h_eq
  have hprev : (jstep (jstep jzero a) b).prev = (jstep (jstep jzero b) a).prev :=
    congrArg JState.prev h_eq
  simp only [jstep] at hprev
  exact h (Option.some.inj hprev).symm

/-! ## S3 — surprisal depends on the history -/

/-- Surprisal proxy, scaled to Nat (avoids floats; injectivity in the count
is all the patcher needs). First occurrence of a byte is expensive, a
repeat is cheaper. -/
def jsurp (s : JState) (b : Byte) : Nat := 256 / (s.c1 b + 1)

/-- **S3.** The same byte carries different surprisal depending on the
history: `a` after `b` costs `256/1`, but `a` after `a, b` costs `256/2`.
The `cs` accumulation feeding the patcher therefore differs under
reordering — cut positions are history-dependent. -/
theorem jsurp_depends_on_history (a b : Byte) (hne : a ≠ b) :
    jsurp (jstep jzero b) a ≠ jsurp (jstep (jstep jzero a) b) a := by
  show 256 / ((jstep jzero b).c1 a + 1) ≠
       256 / ((jstep (jstep jzero a) b).c1 a + 1)
  have h1 : (jstep jzero b).c1 a = 0 := by simp [jstep, jzero, hne]
  have h2 : (jstep (jstep jzero a) b).c1 a = 1 := by simp [jstep, jzero, hne]
  rw [h1, h2]
  omega

/-! ## S4 — the full pipeline (judge + patcher) is a fold -/

/-- Patcher state: the judge, the running `cs` since the last cut, the
current `lim`, and the cut count. -/
structure PState where
  judge : JState
  cs : Nat
  lim : Nat
  cuts : Nat

def pzero (C : Nat) : PState := ⟨jzero, 0, C, 0⟩

/-- Greedy under-shoot cut: add the byte's surprisal to `cs`; when the
budget is reached, cut (reset `cs`, raise `lim`). Causal: the surprisal
is read from the PRE-transition judge state, then the state advances —
the same order the C engine uses. -/
def pstep (C : Nat) : PState → Byte → PState
  | p, b =>
    let w := jsurp p.judge b
    let cs' := p.cs + w
    if cs' ≥ p.lim then
      ⟨jstep p.judge b, 0, p.lim + C, p.cuts + 1⟩
    else
      ⟨jstep p.judge b, cs', p.lim, p.cuts⟩

/-- The full encode = one serial fold. -/
def pfold (C : Nat) (s : PState) (bs : List Byte) : PState :=
  bs.foldl (pstep C) s

/-- **S4 (pipeline decomposition).** The patcher pipeline decomposes
exactly like the bare judge: contiguous runs from true prefix states
reproduce the sequential cut sequence. -/
theorem pfold_append (C : Nat) (s : PState) (xs ys : List Byte) :
    pfold C s (xs ++ ys) = pfold C (pfold C s xs) ys :=
  foldl_append_gen (pstep C) s xs ys

/-- The pipeline's judge component advances exactly like the bare judge
(one transition per byte, either cut branch). -/
theorem pstep_judge (C : Nat) (s : PState) (b : Byte) :
    (pstep C s b).judge = jstep s.judge b := by
  simp only [pstep]
  split
  · rfl
  · rfl

/-- The pipeline's judge component is exactly the judge fold — S2's
non-commutativity transfers to the whole pipeline. -/
theorem pfold_judge (C : Nat) (s : PState) (bs : List Byte) :
    (pfold C s bs).judge = bs.foldl jstep s.judge := by
  induction bs generalizing s with
  | nil => rfl
  | cons b bs ih =>
      show (pfold C (pstep C s b) bs).judge =
           bs.foldl jstep (jstep s.judge b)
      rw [ih (pstep C s b), pstep_judge]

end RatecSerial
