# v1.0.0rc4 — the release-prep candidate

The rc that turns the 1.0.0 go/no-go into ONE machine-checkable
command, and closes every finding from the rc3 fitness review.

## The release gate (`bench/release_gate.py`)

The docs/STABILITY.md freeze audit, executed — not remembered — as a
single script with a GO/NO-GO verdict and exit code:

1. **manifest immutability** — the api_freeze regen must leave the
   committed record surface-identical (revision/timestamp-immune;
   any surface drift is a FAIL).
2. **the freeze test** green.
3. **provenance hashes** — 22/22 artifacts identical to the committed
   record; any drift listed by name.
4. **perf floors** PASS.
5. **quiet-box envelope** present as the floor reference.
6. **lean no-sorry** — every lean/*.lean compiled on the PINNED
   toolchain (auto-located via ELAN_HOME, the reinstall recipe in the
   failure hint), with an ACTIVE-sorry scanner that strips lean
   block/line comments first (docstring mentions of the word are not
   sorries).
7. **make test AND the full pytest suite** green.
8. **the nightly soak** — the last N scheduled GitHub-Actions
   nightlies on main green with an intact daily cadence (N >= 7).
   Reports the streak, the gap list, and — while short — the earliest
   date the week completes assuming green continues.

`python3 bench/release_gate.py` prints the verdict table;
`--json out.json` writes the machine record. Criterion 8 is the only
networked check (GITHUB_TOKEN optional; the repo is public).

## Review findings closed

- README's improvement-record list extended with the rc2 and rc3
  entries; the stale "(current)" marker on the rc1 item removed.
- docs/vllm_recipe.md: the cosmetic "(rc2)" section header dropped.

## Fit-for-1.0.0 status

Criteria 1-7 are green NOW (re-proven at this tag); criterion 8 is a
CALENDAR criterion — the nightly soak week — which no code change can
accelerate. rc4 ships the gate so the final decision is mechanical:
when the streak reaches 7, the gate says GO and 1.0.0 is one tag away.

## Gates (executed at this tag)

make test green, full pytest green, tables drift PASS, all seven LEAN
files compile no-sorry on the pinned v4.36.0 toolchain, floors PASS
against the standing envelope, integration proof PASS (tokenizers
decode byte-exact, official GGUF reader, snapshot continuation
bit-exact), provenance 22/22, manifest at 1.0.0rc4 — the frozen core
identical to the v0.9.3 freeze, the rc3 additive surface unchanged.

## Known-open for 1.0.0

- The nightly-green week (STABILITY.md step 8): the streak stood at 1/7
  at the rc3 review; rc4 carries the gate that verifies it. When the
  gate says GO, tag 1.0.0.
