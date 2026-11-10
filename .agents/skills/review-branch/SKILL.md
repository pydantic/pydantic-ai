---
name: review-branch
description: Review a resolved branch against its canonical base. Use when asked to run review-branch; ask before starting the review suite on your own initiative.
---

# Review Branch

Review one committed snapshot of the branch with the `review-*` subagents, in three waves.
A claim-check adjudicates each wave. `review-state.json` carries the run state, and one PR
comment (the beacon) mirrors it.

Reference files live beside this file, in `.agents/skills/review-branch/` (`REVIEW_SKILL`).
Read each one when a step names it.

| File | Holds |
|---|---|
| `glossary.md` | Suite vocabulary: lane, gate, miss, adjudication |
| `state.md` | `review-state.json` schema and field meanings |
| `verdict-rubric.md` | How a reviewer assigns `Verdict: do\|defer\|skip` to a finding |
| `beacon.md` | PR comment template, buckets, verdict transitions, post/edit script |
| `prepare.py` | Step 1 snapshot and blame map |
| `preconditions.py` | Step 3 dispatch preconditions and reviewer inputs |
| `internals-model.md`, `maintainer-mindset.md` | Background the lanes cite: the runtime walkthrough and maintainer review principles |

The lanes live in `.agents/agents/`; its `AGENTS.md` holds the rules for writing one. CI runs
`test_prepare.py`, `test_preconditions.py` and `mechanical/test_checks.py`. The weekly
`Review Branch Drift` workflow runs `test_drift.py`, which fails when a lane or
`preconditions.py` names a path, family member or surface the code no longer matches.

## This suite and the pre-push gate

`pre-push-review` owns the gate in `pushing-commits-to-the-repo`: one fresh reviewer, kept away
from branch-continuity state. This suite is a separate, wider review that runs when asked. Its
claim-check reads `pr-decisions.md` so a settled decision can refute or rescope a finding, which
the gate's context contract forbids. A run of this suite therefore neither satisfies the gate nor
spends its review budget.

## Invocation gate

**If nobody asked for this review, ask before you dispatch anything.** The suite starts about
15 subagents and posts a PR comment. That cost fits a diff with real surface, not a model-id
addition or a one-file fix.

An explicit invocation runs immediately: the user typed `/review-branch`, or an orchestrating
agent sent it into this session. That invocation is the choice.

When you reached this skill on your own judgment, ask the user with your harness's structured
question tool (AskUserQuestion in Claude Code) before Step 0:

| Option | What it means |
|---|---|
| Run the structured swarm | This skill as written: waves, fixed lanes, gating, `review-state.json`, beacon. |
| Review it yourself | Decompose the diff, dispatch only the subagents its weak points call for (an individual `review-*` lane included), and smoke-test what you doubt. No waves, no state file, no beacon. |

Name the files touched and the kind of change in the question, so the user chooses on the diff.
Take the answer as given. On "review it yourself", do not fall back into the swarm.

## Waves and the gate

- **Wave 1** — code correctness. **Wave 2** — tests, cassettes, docs. **Wave 3** — cross-cutting
  and sibling-consistency audits.
- **Gating lanes**: `review-runtime-behavior`, `review-public-api`, `review-integration-impact`.
  Their BLOCKING findings mean the diff's behavior or public surface is about to change.
- Dispatch waves 2 and 3 only when wave 1's gate is closed. Dispatch wave 3 only after every
  wave-2 reviewer is `complete` or `skipped`. Wave-2 findings never hold wave 3.
- A BLOCKING finding in a non-gating lane does not hold dispatch. It still holds PASS.
- Reason: test, cassette and docs reviewers pin the behavior the code commits to. Running them
  while a gating lane is open pins behavior that is about to change.
- Wave 3 runs last because its fixes change the code, which resets the run to wave 1. No
  earlier wave reads wave-3 output.

Run `review-docs` on every branch. PR readiness is context for it, never a skip or automatic
deferral gate. After the waves settle, fix the reported docs gaps to match the final behavior. Leave a
docs gap open only when its wording depends on an unresolved design question, and record that
dependency as an open required action. Prepare a new snapshot after committing the fixes.

## Reviewers

`preconditions.py` decides dispatch for every reviewer (Step 3). Each reviewer re-confirms its
own trigger and self-skips, so the gates may over-fire.

| Reviewer id | Wave | Dispatched when | Extra inputs |
|---|---|---|---|
| `pattern` | 1 | code changes | skill `.agents/skills/review-patterns/SKILL.md` |
| `auto-review` | 1 | code changes | skill `.agents/skills/auto-review/SKILL.md` |
| `review-public-api` (gating) | 1 | public modules changed | — |
| `review-integration-impact` (gating) | 1 | core symbols or wrapped-class files touched, a package `.py` file outside slim changed, or a gating `scripts/durable_exec_contract.json` entry touched | `core_symbols_path`, `wrapped_class_files_path`, `durable_exec_entries_path` |
| `review-runtime-behavior` (gating) | 1 | a package `.py` file changed | — |
| `review-spec-coverage` | 1 | spec URL in the PR body | `spec_url` |
| `review-spec-conformance` | 1 | code changes | `pr_number` |
| `review-code-reuse` | 1 | new package `.py` file, or new top-level def/class | `new_py_files_path` |
| `review-trust-boundary` | 1 | UI, SSRF, sandbox, workspace, CI workflow or prompt-framing path or token changed | — |
| `review-suspend-resume` | 1 | pause-boundary path or token changed | — |
| `review-concurrency-lifecycle` | 1 | realtime path changed, or a changed package line names an async lifecycle primitive | — |
| `review-ambient-matrix` | 1 | changed line conditions on an ambient dimension | `ambient_branch_lines_path`, `cassettes_list_path` |
| `review-precedent` | 1 | a modified (not only added) file | `pr_number` |
| `review-test-shape` | 2 | ≥10 new test functions and ≥10 added test lines that reach a private symbol | — |
| `review-cassettes` | 2 | cassettes changed, or a `models/` or realtime codec file changed | `cassettes_list_path` |
| `review-tests` | 2 | test files changed, or a new public def in a package | — |
| `review-docs` | 2 | every branch | `pr_ready_path` |
| `review-cross-cutting` | 3 | code changes | — |
| `review-sibling-consistency` | 3 | family member modified or added, or a peer-type branch changed | `sibling_family_files_path`, `peer_type_branches_path` |

Input files live in `.agents/skills/review-branch/tmp/`: `core-symbols-touched.txt`,
`wrapped-class-files.txt`, `new-py-files.txt`, `ambient-branch-lines.txt`, `cassettes.list`,
`sibling-family-files.txt`, `peer-type-branches.txt`, `durable-exec-entries.txt`, `pr-ready-for-docs.txt`,
`spec-url.txt`.

`review-claim-check` is not a lane. It runs once per wave over that wave's reports (Step 5b).

No lane owns clai2 terminal UX: rendering, idle CPU, and how an interaction feels. This is a
deliberate scope boundary. A static diff review cannot judge it, and `src/pydantic_clai2/AGENTS.md`
→ Testing makes the author verify every affected UX feature in tmux.

Dispatch each `review-*` agent by the id its harness gives the lane file in `.agents/agents/`:

| Harness | Lane id | General-purpose agent |
|---|---|---|
| Claude Code | the lane's `name`, e.g. `review-tests` | `general-purpose` |
| OpenCode | `review/` plus the lane's `name`, e.g. `review/review-tests` | `general` |

That loads the agent's lane prompt, so the dispatch prompt carries only per-run inputs and the
rules below. If the harness answers "unknown subagent type", or loads no agent from
`.agents/agents/`, report the agent-discovery failure. Never fall back to the general-purpose
agent with the role inlined. `pattern` and `auto-review` are skills: dispatch each as the
general-purpose agent, told to read its skill file and review `code_diff_path`.

### Review models

Use the [light tier](../pushing-commits-to-the-repo/SKILL.md#model-tiers) for every suite dispatch,
including `pattern`, `auto-review`, and `review-claim-check`, and set its effort from the PR's
complexity. The lane files carry no model, so pass it on every dispatch.
For Codex, use `fork_turns: 'none'` and pass the chosen effort when supported.

### Dispatch prompt

Every dispatch prompt carries these inputs, one per line:

- `code_diff_path: .agents/skills/review-branch/tmp/code.diff`
- `blame_map_path: .agents/skills/review-branch/tmp/code.blame-map.txt`
- `branch_files_path: .agents/skills/review-branch/tmp/branch-files.txt`
- `merge_base:` the content of `tmp/merge-base.txt`. Read the file; never recompute it.
- `review_head:` the content of `tmp/head-sha.txt`
- `worktree_root:` the repository root
- waves 2 and 3: `wave_state_path: .agents/skills/review-branch/review-state.json`
- the reviewer's extra inputs from the table

Then paste this block verbatim:

> **Fixed base.** Diff ONLY against the provided `merge_base` SHA and the files in `branch_files_path`. Never run `git diff main`, `git diff origin/main`, `git diff upstream/main`, or `git merge-base` to recompute the base — in this worktree those refs may be stale or point at a fork, and recomputing will silently pull unrelated files into your review and make you report the manifest as wrong when it isn't.
>
> **Diff authority.** The provided `code.diff` is authoritative for WHAT you review. When you Read an on-disk file for context and its content **diverges from `code.diff`** — the working tree carries an uncommitted fix, or is otherwise ahead of / behind the diff — **review the diff-as-given.** Never treat the on-disk file as the newer/correct truth: do NOT downgrade or drop a finding because the tree "already fixed" it, and do NOT raise a finding that exists only on disk and not in the diff. An on-disk Read alone does NOT tell you what is committed — if you want to characterize the divergence, reconcile it with `git show <review_head>:<path>` / `git status` and surface it as a one-line note, never as a reason to override, upgrade, or downgrade a finding. The diff is what merges; the working tree is not.
>
> **Behavioral findings.** If a finding is a *behavioral* claim — "X is hidden until Y", "this tool is deferred until loaded", "this path short-circuits", "no leak occurs" — you may raise it but you may NOT settle it from reading alone. Tag it `needs-runtime-verification` and state what executable check would confirm it (a throwaway `uv run --no-sync python -c '...'` or a targeted test). A confident static trace is not a verdict on runtime behavior.

When `branch-files.txt` has a path under `src/pydantic_ai_harness/`, `tests/harness/`, `docs/harness/`,
`src/pydantic_clai2/`, `tests/clai2/`, `pydantic_ai_slim/pydantic_ai/realtime/`, `tests/realtime/`,
`docs/realtime/` or `.github/workflows/`, add this line, keeping only the sentences whose paths appear:

> **Package rules.** For paths under `src/pydantic_ai_harness/`, `tests/harness/` or `docs/harness/`, also apply `src/pydantic_ai_harness/AGENTS.md` and the parts of `src/pydantic_ai_harness/agent_docs/review-checklist.md` inside your lane. For `tests/harness/`, also apply `tests/harness/AGENTS.md`. For `docs/harness/`, also apply `docs/harness/AGENTS.md`. For `src/pydantic_clai2/` or `tests/clai2/`, apply `src/pydantic_clai2/AGENTS.md`. For `pydantic_ai_slim/pydantic_ai/realtime/` or `tests/realtime/`, apply `pydantic_ai_slim/pydantic_ai/realtime/AGENTS.md`. For `docs/realtime/`, apply `docs/realtime/AGENTS.md`. For `.github/workflows/`, apply `.github/workflows/AGENTS.md`.

Name only files that appear in `branch-files.txt` (Step 2).

## Steps

### 0. Establish the review head

Read the PR number, base branch and head SHA with
`gh pr view --json number,baseRefName,headRefOid`. Distinguish a confirmed local-only branch from
a failed GitHub request. For a PR, move a clean local HEAD that is only behind the PR head (as
after GitHub's "Update branch") to that head:

```bash
# PR_NUMBER and PR_HEAD (headRefOid) come from `gh pr view`.
git cat-file -e "$PR_HEAD^{commit}" 2>/dev/null \
  || git fetch origin "$PR_HEAD" --quiet 2>/dev/null \
  || git fetch upstream "$PR_HEAD" --quiet 2>/dev/null \
  || git fetch origin "pull/$PR_NUMBER/head" --quiet 2>/dev/null || true
if [ "$(git rev-parse HEAD)" != "$PR_HEAD" ] && git merge-base --is-ancestor HEAD "$PR_HEAD" 2>/dev/null \
  && [ -z "$(git status --porcelain)" ]; then
  git reset --hard "$PR_HEAD" --quiet
fi
```

Then verify that local HEAD matches the PR head. A mismatch means local is ahead, diverged, or
behind with uncommitted changes. Finish and push intended local commits before reviewing a PR.
Never discard local commits or uncommitted changes to make the heads match.

### 1. Prepare the committed snapshot and blame map

Resolve the PR's actual base branch. Use `main` only for a confirmed local branch targeting
main. Run the bundled script; never reconstruct the diff or write an ad hoc blame parser.

```bash
# BASE_REF is the resolved target branch.
REVIEW_SKILL=.agents/skills/review-branch
if git remote get-url upstream >/dev/null 2>&1; then BASE_REMOTE=upstream; else BASE_REMOTE=origin; fi
git fetch "$BASE_REMOTE" "$BASE_REF" --quiet || exit 1
BASE="$BASE_REMOTE/$BASE_REF"
uv run --no-project python "$REVIEW_SKILL/prepare.py" --repo . --base "$BASE" || exit 1

MERGE_BASE=$(cat .agents/skills/review-branch/tmp/merge-base.txt)
REVIEW_HEAD=$(cat .agents/skills/review-branch/tmp/head-sha.txt)
```

Stop on a failed fetch: a stale base produces the wrong review scope. The script rejects an
unresolved index, an unfinished merge/rebase/cherry-pick/revert, and conflicts with the fetched
base. Resolve and commit conflicts, then retry. The script never changes the checkout.

It writes these worktree-local artifacts to `.agents/skills/review-branch/tmp/`:

| Artifact | Meaning |
|---|---|
| `base-ref.txt`, `base-sha.txt` | Canonical base ref and its fetched commit |
| `merge-base.txt`, `head-sha.txt` | Immutable endpoints for the review |
| `branch-files.txt` | Files changed between those endpoints — the authority for changed files |
| `code.diff` | Diff excluding cassettes |
| `cassettes.list` | Cassette changes, `<status> <path>` |
| `diff-hash.txt` | SHA-256 of `code.diff` |
| `code.blame-map.txt` | `<path>:<new-line-no> NEW\|PRE-EXISTING <full-commit-sha>` per added text line |

A commit reachable from the head but not the merge base is **NEW**, merge-resolution commits
included. Every other introducing commit is **PRE-EXISTING**. Paths with whitespace, quotes or
backslashes are JSON-quoted. Binary files, submodule pointers and pure renames have no blame rows.

Stop on preparation failure. Never reuse an earlier map or relabel unresolved entries. Read
committed content with `git show "$REVIEW_HEAD:<path>"`. If HEAD changes during the review,
prepare a new snapshot before reporting the branch as reviewed.

Report the base, pinned head, code-diff size and cassette count. When `origin` is the only
remote, name it, so the base can be checked.

### 1b. Mechanical pre-pass

The mechanical scanner reads the checkout. Run it only when tracked files match `REVIEW_HEAD`;
otherwise report it as skipped. The committed snapshot still governs the review.

```bash
if git diff --quiet "$REVIEW_HEAD" --; then
  MECH_STATUS=0
  uv run --no-project python "$REVIEW_SKILL/mechanical/runner.py" \
    --repo . --diff "$MERGE_BASE" --json \
    > .agents/skills/review-branch/tmp/mechanical.json || MECH_STATUS=$?
  [ "$MECH_STATUS" -le 1 ] || exit "$MECH_STATUS"
  # Exit 1 can mean reported findings. Require valid JSON before using the output.
  jq -e '.findings | type == "array"' .agents/skills/review-branch/tmp/mechanical.json >/dev/null || exit 1
fi
```

Pass valid mechanical output to reviewers as supporting evidence. Never use it as a wave gate.

### 2. Verify scope

`branch-files.txt` is the authority for changed files. Before a dispatch prompt says "X was
modified", check `grep -Fxq -- "<path>" .agents/skills/review-branch/tmp/branch-files.txt`. Git
quotes unusual paths in that list; decode them before comparing. When in doubt, omit the claim:
the reviewer has the diff.

### 3. Compute dispatch preconditions

```bash
uv run --no-project python "$REVIEW_SKILL/preconditions.py" --repo . ${PR_NUMBER:+--pr "$PR_NUMBER"} || exit 1
```

`PR_NUMBER` is the Step 0 PR number, empty for a local branch. The script writes the reviewer
input files and `tmp/preconditions.json`, and prints the precondition table. Report that table.
`reviewers.<id>.dispatch` in `preconditions.json` decides each dispatch, and `.reason` is the
skip reason you record.

### 4. Read or initialize state

Read `.agents/skills/review-branch/review-state.json` (schema: `$REVIEW_SKILL/state.md`).
Compare its `merge_base`, `head_sha` and `diff_hash` with the prepared snapshot.

- All three match: resume incomplete reviewers from the saved state.
- Any differs: reset every reviewer status and adjudication. A changed file list cannot detect
  changed content, and identical patches can sit on different history.

On a reset, archive prior wave reports under the worktree-local scratch directory before
starting fresh wave directories. Never use a shared `/tmp/review-branch`. Keep the beacon id,
human decisions and applied-finding history. Never carry an earlier verdict into a new snapshot.
Initialize wave 1 as `pending` and later waves as `blocked`, and write the state before dispatch.

### 5. Dispatch the current wave

Dispatch every `pending` reviewer of the current wave in one message, so they run concurrently.
Record each reviewer that `preconditions.json` skips as `skipped` with its reason. When wave 1's
gate is open, record wave-2 and wave-3 reviewers as `blocked` with
`reason: "gating lanes open (<N> blocking)"`.

### 5b. Adjudicate the wave's findings

Reviewers never read each other's reports, so no lane can check a peer's claim. Before
computing the gate, dispatch one `review-claim-check` over the wave's reports. A false BLOCKING
then never holds a later wave, and a refuted claim never reaches the beacon.

Retry failed reviewers first. When the wave produced zero findings, record adjudication as
`skipped`. Otherwise dispatch after every reviewer in the wave is `complete` or `skipped`, with
inputs `wave_number`, `reports_dir: .agents/skills/review-branch/tmp/wave<N>/`, `code_diff_path`,
`blame_map_path`, `branch_files_path`, `pr_decisions_path: .claude/skills/branch-context/pr-decisions.md`,
`merge_base`, `worktree_root`, and the verbatim block from "Dispatch prompt". Save its report to
`tmp/wave<N>/claim-check.md`.

Apply its verdicts before Step 6:

- **CONFIRMED** — the finding counts as its reviewer reported it.
- **RESCOPED** — the finding leaves the wave's `blocking` and `required` counts. Carry it as a
  `defer` with the tracking artifact the claim-check named, and open that artifact in this run.
  A defer with no artifact is invalid.
- **REFUTED** — the finding leaves every count and never reaches the beacon.
- **UNSETTLED** — the finding counts exactly as its reviewer reported it.

The claim-check never adds a finding. Drop any finding its report raises and note the lane
violation in the run output.

### 6. Collect results, update state, compute the gate

For each reviewer that ran, save its report to `tmp/wave<N>/<reviewer-id>.md`. Parse BLOCKING
and WARNING counts from its Summary block, and count the findings whose `Verdict:` line is `do`
(required actions, whatever their severity). Subtract the claim-check's REFUTED and RESCOPED
findings, then record `status: complete`, `blocking`, `warnings`, `required` and `report_path`.
State carries post-adjudication counts.

A failed or unfinished reviewer never counts as zero findings. Retry it before closing its wave.
A wave with findings needs completed adjudication; `skipped` adjudication counts only for zero
findings.

- `blocking_total` and `required_total` sum the wave's reviewers.
- Wave 1 only: `gating_blocking_total` sums `blocking` over the three gating lanes.
- Wave 1 `gate_status` is `closed` when every reviewer is complete or skipped, adjudication is
  complete, and `gating_blocking_total == 0`.
- Waves 2 and 3 use the same rule with `blocking_total == 0`. Their gate is reporting only.
- `required_total` never holds a gate. It holds PASS (Step 7).

**Write the wave's results to `review-state.json`, atomically, before dispatching the next
wave.** Wave-2 and wave-3 reviewers read `wave_state_path` to check the gate, and a stale file
makes them defer. A dispatch prompt never substitutes for the file.

When wave 1's gate is closed and `current_wave` is 1, set it to 2, write state, and dispatch
wave 2 in this run. When `current_wave` is 2 and every wave-2 reviewer is complete or skipped,
set it to 3, write state, and dispatch wave 3 in this run.

**Premise-invalid convergence.** When two or more wave-1 reviewers independently land the same
structural finding that the claim-check CONFIRMED — *no first-party code path produces this
input*, or *the change inverts an intentional, documented design choice* — emit the secondary
verdict **REJECT (premise-invalid)** alongside "needs maintainer", recommending close. Nothing
weaker triggers it.

### 6b. Targeted resolved-thread cross-check

Check a finding against the PR's resolved review threads only when it hangs on one of two
questions:

- **Which way, or why this way**: the finding second-guesses a design choice a resolved thread
  settled. Demote it to INFO and cite the thread.
- **Already dismissed**: the finding restates something a resolved thread decided against. Drop
  it and cite the thread.

Fetch resolved threads with `gh api` GraphQL (`isResolved`). Report a thread marked resolved
whose claim no longer matches the code as **resolved-but-stale**, under its own heading.

Write what a thread settles into `branch-context/pr-decisions.md` (append-only, thread link as
Source), so the next session reads the decision instead of re-mining GitHub. A change to scope
or success criteria goes to `issue-brief.md` via `/refresh-issue-brief`. Skip the fetch when
`pr-decisions.md` already records the decision.

### 7. Combined report

```
## Branch Review: <branch-name>

Code diff: <N> lines across <M> files · Cassettes: <P> new, <Q> modified
Diff hash: sha256:<short> (changed since last run: yes/no)
Current wave: <N> · Wave-1 gate: <open/closed>

### Wave <N> — <name>
| Reviewer | Status | Blocking | Required | Warnings | Report |
<inlined report per reviewer that found something>
(wave not dispatched: "Wave <N> deferred — the gating lanes have <N> blocking findings.")

### Summary
- New-in-branch issues: <N> · Pre-existing (optional flyby): <N>
- Resolved-but-stale threads: <N> · Demoted by a resolved thread: <N> (logged to pr-decisions.md)
- Refuted by claim-check: <N> · Rescoped to pre-existing: <N> (tracked at: <links>)
- Blocking: <N> · Required actions (do): <N> · Warnings (defer/skip): <N>
- Verdict: PASS | REVIEW BEFORE PUSHING | REJECT (premise-invalid)
- Next action: <one line>
```

The verdict is **PASS** only when the snapshot still matches HEAD, every applicable reviewer
completed, all three waves are closed, zero NEW blocking findings remain, and `required_total`
is 0 across all waves. A `do` finding holds PASS exactly like a blocker: severity is not
actionability. Otherwise the verdict is REVIEW BEFORE PUSHING. Render REJECT (premise-invalid)
alongside that verdict, never instead of escalation.

### 8. Post the PR review-state comment (PR only)

Skip this step when no PR exists, or when you were told to hold comments. Post only when
external communication is authorized. Before publishing PASS, verify that the PR head still
equals `review_head`; if it moved, restart from Step 0.

Write every edit of the comment for a maintainer who reads it cold: no diff open, no Checks
tab, no memory of earlier rounds. From the comment alone they must answer three questions:

1. What state is this PR in now: reviewing, blocked, required actions open, or passed?
2. How did it get there: what changed since the last edit, and if the verdict flipped, why —
   above all, were the required items resolved, and by whom?
3. Is anything left for me to do?

Build and post the comment by following `$REVIEW_SKILL/beacon.md`. Before saving an edit,
reread it as that maintainer and check that all three answers are on the page.

## Design constraints

- Reviewers see the cassette-stripped `code.diff`. Dispatch prompts never re-filter it.
- Review artifacts stay in the worktree-local `.agents/skills/review-branch/tmp/`, so worktrees
  can review concurrently.
- Reviewers are read-only and self-contained: they read only the orchestrator's inputs, never
  another reviewer's report. The claim-check is the one exception, for its own wave only.
- `review-state.json` is the single state file. The beacon mirrors it.
