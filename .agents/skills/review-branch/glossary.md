# Review-Branch Glossary

Shared vocabulary for the branch-review subsystem. Read it before you repair the suite.
"The review setup", "the review agents" and "our reviewers" all mean the things defined
here. Use these terms precisely.

## The subsystem

- **the suite** (a.k.a. *review suite*, *the review setup*) — the `/review-branch`
  orchestrator skill **plus** its `review-*` subagents, taken together. When the user says
  "our review setup missed X", they mean the suite.
- **`/review-branch`** — the orchestrator skill. Computes the branch diff, computes
  per-reviewer preconditions, dispatches reviewers in waves, collects findings, gates.
  Source: `.agents/skills/review-branch/SKILL.md`.
- **reviewer** (a.k.a. *review agent*, *review subagent*, *review lane*) — one
  `review-*` subagent owning exactly one concern. Source files: `.agents/agents/review-*.md`.
- **lane** — a reviewer's single concern. "In its lane" / "out of its lane" decides
  which reviewer owns a finding. One agent = one lane is the governing rule
  (`.agents/agents/AGENTS.md`).

## Dispatch model

- **wave** — a dispatch tier. Wave 1 = code-correctness reviewers (parallel); wave 2 =
  test-/cassette-/docs-shape reviewers; wave 3 = cross-cutting & sibling-consistency
  audits. Waves run in order, but only the gating lanes hold a later wave back.
- **gating lanes** — `review-runtime-behavior`, `review-public-api`,
  `review-integration-impact`. The only lanes whose open BLOCKING findings hold waves 2
  and 3 back, because they mean the diff's behavior or public surface is about to change
  — what a later wave would otherwise pin. Every other lane's BLOCKING findings are
  reported and must still be resolved before merge; they don't delay dispatch.
- **gate / `gate_status`** — on wave 1, `closed` when the **gating lanes** have zero
  BLOCKING findings on the current `diff_hash` (`gating_blocking_total == 0`); `open`
  otherwise. Waves 2 and 3 both dispatch on that one gate; their own `gate_status`
  (zero BLOCKING in the wave) is reporting only.
- **precondition** — the per-reviewer trigger the orchestrator computes from the diff
  (e.g. `public_modules_changed`, `cassettes_changed`). A reviewer whose precondition is
  false is recorded `skipped`, not run.
- **`diff_hash`** — sha256 of the cassette-stripped code diff. Changing it resets the
  suite to wave 1.

## Reviewer ids

The reviewer table in `SKILL.md` lists every reviewer id with its wave and dispatch trigger.

## Feedback & improvement

- **miss** — a real defect a **human** reviewer caught (on a PR) that the suite did
  **not** flag. The unit of input to a *suite repair*.
- **miss mode** — *how* the suite failed on a given miss, established by reading the
  actual run transcript (never guessed):
  - *not-dispatched* — the owning reviewer's precondition was false, so it never ran.
  - *dispatched-but-skipped* — it ran but self-skipped (e.g. spec-conformance NL-gate).
  - *ran-silent* — it ran, looked at the area, and raised nothing.
  - *flagged-but-overruled* — it raised the finding but the orchestrator demoted/dropped it.
  - *reasoned-wrong* — it explicitly reasoned about the area and reached a wrong
    conclusion (the highest-value mode — the lane exists but its method is flawed).
- **class** — the generalized form of a miss, stated so it applies beyond the one
  instance (e.g. "empirical-premise inflation", not "missed the memory tool").
- **owner** — the existing reviewer whose lane the class belongs to. The dedup
  discipline (`.agents/agents/AGENTS.md`) prefers fixing the owner over adding
  a new reviewer; a new mini-agent requires the option-3 proof bar.
- **suite repair** — a change that turns a *miss* into a fix to the suite: read the run
  transcript, diagnose the *miss mode*, generalize to a *class*, map it to an *owner*, and edit
  that owner.
- **suite history** — the git history of the suite's source files. Each suite repair lands as a
  commit that records miss → class → fix, so
  `git log -- .agents/skills/review-branch .agents/agents` shows whether a class was already
  addressed.
- **claim** — one finding, taken as an assertion to be tested: what it asserts, at which
  file and symbol, at what severity.
- **adjudication** — the claim-check's verdict on a claim: *CONFIRMED* (evidence reproduces
  it), *RESCOPED* (true, but the code pre-dates the diff — leaves the gate, needs a tracking
  artifact), *REFUTED* (evidence contradicts it — leaves every count), *UNSETTLED* (not
  settled — stands unchanged).
- **`review-claim-check`** — the agent that adjudicates. Runs once per wave (`/review-branch`
  Step 5b), after that wave's reviewers and before the gate. The one agent allowed to read a
  finished report; it raises no findings of its own. Source:
  `.agents/agents/review-claim-check.md`.
- **false positive** — a claim the evidence refutes. The mirror of a *miss*: a miss is a real
  defect the suite did not raise, a false positive is a non-defect the suite did. Misses feed
  suite repairs; false positives are filtered by `review-claim-check` before the user
  sees them.
