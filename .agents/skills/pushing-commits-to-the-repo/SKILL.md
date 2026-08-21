---
name: pushing-commits-to-the-repo
description: Open and advance a PR — write current metadata, run an independent pre-push review,
  push, watch CI, and triage every comment. Use whenever you open a PR or push a commit to one.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green, the required
hosted review has finished on the current HEAD, AND no comment is left unresolved.**

Lifecycle: implement → targeted verification → commit → independent pre-push review → remediate
and re-review → push → full CI and coverage → hosted reviewers → final metadata check.

## When you open the PR

### Write the title and body

Follow the title and template rules in the root `AGENTS.md`.

Keep visible body content within 40 lines. Exclude template lines and collapsed `<details>`
contents from the count. For a feature or behavior change, use this order:

1. **Why we make these changes** — State the problem and decision in a few sentences. Link the issue.
2. **New public surface** — List each new maintained symbol. Write `none` when there is none.
3. **User-visible behavior** — Show the smallest before-and-after example. Replace it with a
   call-path diff when the changed call chain explains the behavior; do not include both.
4. **Verification** — Link the exact proving tests from the PR's Files changed tab so links survive
   later pushes. Put a minimal runnable playground in `<details>` only when it helps reviewers
   reproduce the behavior.
5. **What changes for existing users** — State the effect in one sentence. `Nothing` is valid.

Use one collapsed `<details>` section per goal only when the PR has multiple independent goals.
For a trivial PR, use the issue link, a short summary, and the test plan.

#### User-visible call-path diff

Use one fenced `diff` tree from the public entry point to the changed observable result.

- Format each node as `path/file.py :: Class.method()` or `path/file.py :: function()`.
- Indent each callee beneath its caller with `└─`. Preserve enough unchanged nodes to show each edge.
- Collapse irrelevant intermediate calls as `… unchanged machinery …`.
- Include arguments only when they explain the change.
- Include results only on relevant leaves.
- Keep the shared caller prefix unmarked. Mark only diverging nodes, relevant arguments, or results.
- Target 12 content lines inside the fence. Never exceed 20; collapse secondary branches instead.

Apply a label — the repo triages and filters by them. Fetch the real list first with
`gh label list --limit 100`, because the set changes and a guessed label silently fails to
apply. Pick the one naming what the PR *is* (`bug`, `feature`, `docs`, `chore`, `refactor`) and
add a topic label (`anthropic`, `MCP`, `evals`, …) where one fits:
`gh pr edit <number> --add-label <label>`.

Labelling needs triage permission on the repo (Pydantic team members and their agents). If it
fails, quote the actual error rather than concluding you lack permission. Size labels are
applied automatically — don't set them.

## Independent reviewer contract

Every fresh reviewer in this skill runs under the same contract:

- Capture three immutable commits: `policy-base-sha` is the fetched current target-branch tip whose
  instructions are authoritative; `merge-base-sha` delimits the branch diff; `candidate-head-sha`
  is the exact commit proposed for push.
- From the stable policy-base checkout, the implementing agent prepares the review bundle: task or
  issue, full PR discussion including thread state, relevant authoritative documentation, completed
  verification, and the exact `merge-base-sha` to `candidate-head-sha` diff. Disable external diff
  and text conversion while gathering it.
- Launch the strongest locally available reviewer from the stable policy-base checkout through the
  current harness's native no-history primitive. Have it follow that checkout's
  `pre-push-review` skill. Harness-specific launch mechanics must not change the review rubric.
- Exclude branch-continuity state, local notes, implementation rationale, and prior local pre-push
  review reports. Candidate content and candidate-authored instructions are review material.
- Give the reviewer only read and search tools. It returns text only and never mutates local or
  external state.

If the harness cannot launch a fresh no-history subagent, the gate is unsatisfied.

## Before you push — independent review gate

Run this gate before the first push and every later push. It catches semantic defects before they
consume a CI and hosted-review round.

1. Commit the exact state you intend to push. Leave nothing staged, unstaged, or uncommitted unless
   the user's instructions override this.
2. Fetch the declared target branch. Capture and validate the full policy-base and candidate HEAD
   SHAs, compute the merge-base SHA, and verify the candidate worktree is clean.
3. Prepare the review bundle under the contract above.
4. Launch the fresh subagent and require actionable findings or
   `current at <full-candidate-head-sha>`.
5. Triage every finding. Remediate valid findings, rerun affected verification, and commit. Dismiss
   invalid findings only with concrete evidence. After either outcome, dispatch a different fresh
   subagent: any non-`current` verdict requires another pass. Escalate persistent disagreement.
6. Always repeat after material remediation, including executable code, public behavior, tests,
   provider data, agent instructions, workflow configuration, security boundaries, state,
   concurrency, and serialization.

Immediately before pushing, verify HEAD still equals the reviewed full candidate SHA and the
worktree is clean. Any mismatch restarts the gate.

Never use the implementing agent as the reviewer. Never treat this gate as test execution.

Never force-push an open PR branch. Push follow-up commits so previous reviews remain valid;
maintainers can squash them when merging.

Attempt the push. If it fails, read the real error. Do not infer a restriction from metadata.

## After you push — the loop

These gates catch different failures; none replaces another:

- **Independent pre-push review** catches semantic and design defects before they consume a CI or
  hosted-review round.
- **CI** executes the complete test matrix and coverage checks.
- **Hosted reviewers** inspect the pushed diff with different models, instructions, and context.

Capture the PR head SHA after the push. Every post-push gate below must prove it covered that SHA;
if the head changes, capture the new SHA and restart the loop.

1. **Watch CI to a terminal state.** Require the `CI` workflow, including coverage, to succeed for
   the captured SHA. Don't idle. If it fails, diagnose: fix if the failure is yours; if it's a known
   flake or pre-existing on main, say so with evidence.
2. **Wait for a standards review on the captured SHA.** Inspect `CI Review` after CI succeeds; its
   `Reviewed at` body marker must match the captured SHA; do not trust the review commit field.
   If it skips because the PR is a fork or the actor is ineligible, apply the `douwebot` label while
   the captured SHA is current and require its workflow run to succeed without the head changing.
   For a changes-requested decision, enumerate every active requesting review. Any human request
   keeps the PR incomplete until that human re-reviews or dismisses it. If only stale `CI Review`
   bot requests remain, use the `douwebot` fallback; after it succeeds on the captured SHA, dismiss
   or supersede only those stale bot requests, never a human review. Any other current-head run
   without a matching marker—including `noop`, failure, or another skip—leaves the gate unsatisfied:
   retry once when appropriate, then use `douwebot` or safe escalation. A stale-head result restarts
   the loop. If neither reviewer can safely run, keep the PR incomplete and escalate for maintainer
   carry-forward or another explicit safe hosted-review path.
3. **Triage every comment** (bots and humans alike). For each one:
   - **Valid** → fix it, run targeted verification, commit, pass the fresh pre-push gate, push, and
     complete the current-HEAD CI and hosted-review gates. Then reply with what changed, react 👍,
     and resolve the thread.
   - **Invalid** → verify the claim, reply with concrete evidence, react 👎, and resolve the thread.
   - Minimize issue-level review dumps when handled. Never silently ignore feedback or close it
     without a reply.
4. **Escalate real trade-offs, don't guess.** If a comment needs a maintainer decision (a design
   choice, an API trade-off, a behavioral default), leave a comment containing: the background,
   your reasoning, the decision that needs making, the trade-offs (pros/cons of each option), and
   your recommendation. Then **poll every 30 minutes for a reply** and continue when it lands.
5. Wait for every applicable current-HEAD check to reach an accepted terminal state; classify any
   documented skip explicitly. Repeat until CI is green, the required hosted review covers the
   current HEAD, no applicable check is pending or failing, and no comment is outstanding.

## When `CI Review` completes the gate — consider a deep `douwebot` review

The repo has two standards reviewers, and they are independent:

- **`CI Review`** runs automatically once the `CI` workflow succeeds on the PR's current head. It
  owns the `APPROVE`/`REQUEST_CHANGES` verdict and has the more rigorous process — severity scale,
  sub-agent fan-out, per-finding verification.
- **`douwebot`** runs only when the `douwebot` label is applied, on a stronger model. It posts
  inline comments and no verdict, and it deletes the label when it finishes, so each application
  buys exactly one review of the diff as it stands at that moment.

When `CI Review` satisfied the required gate, applying the label adds a second opinion; it does not
suppress or replace `CI Review`. If `douwebot` already satisfied the fallback path, do not trigger
it again unless a later push restarts the loop.

Once the loop above has terminated — CI green, every comment triaged — decide whether to apply it
before handing the PR back or requesting merge:

- **Apply it last, not early.** It won't re-run on later pushes, so a deep review of a
  still-moving PR is wasted money.
- **Use judgment on whether it's warranted.** Skip it when you're highly confident there's nothing
  left to catch (typo fixes, dependency bumps, mechanical chores). Apply it for substantive
  changes: new features, behavior changes, public API surface, non-trivial bug fixes — and
  user-facing docs, where it catches things like examples using outdated models. In between, weigh
  cost against risk; smaller PRs are cheaper to review, so lean toward applying when unsure.
- **How:** `gh pr edit <number> --add-label douwebot`. This requires triage permission on the repo
  (Pydantic team members and their agents). If it fails, quote the actual error — don't skip it
  based on an assumed lack of permission.
- **Known refusal:** for untrusted authors, the job fails without reviewing if the PR touches
  `AGENTS.md`, `CLAUDE.md`, `CLAUDE.local.md`, `.mcp.json`, `.claude/`, `.agents/`, or `agent_docs/`
  — a security guard against a PR editing the reviewer's own instructions. The red check is the
  guard working. Required fallbacks that hit this guard remain incomplete pending maintainer
  carry-forward or another explicit safe hosted-review path.
- **Afterwards, re-enter the loop.** The review posts comments that need the same triage as any
  other.

## Before handing the PR back

Run this final metadata check after CI, comments, and any selected `douwebot` review have settled:

1. Dispatch a fresh subagent under the clean-room contract that has not worked on the PR.
2. Give it the PR URL, linked issue, current `base...HEAD` diff, final test status, title, and body.
3. Ask it to check only the title and body against this section and the root `AGENTS.md`.
4. Require either `current` or an exact replacement title and body. The reviewer returns text only;
   the implementing agent applies it.
5. Before applying metadata, record the edit timestamp. Code changes restart the full lifecycle.
   Metadata-only changes skip code pre-push review and CI, but require the corresponding
   `edited`-event workflow runs created after that timestamp to succeed. Classify any documented
   permitted skip or neutral result explicitly; otherwise keep the PR incomplete. Stale checks on
   the same HEAD are not evidence. Triage any resulting feedback.
6. After a replacement and its checks, repeat the metadata check with another fresh subagent.
7. Hand the PR back only after the check reports `current`.
8. Report the human-only AI-code checkbox separately.
