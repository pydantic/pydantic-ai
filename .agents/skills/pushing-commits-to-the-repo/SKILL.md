---
name: pushing-commits-to-the-repo
description: What to do when you open a PR and every time you push — label the PR, run a local
  review, watch CI to green, triage every review comment to a reply and a reaction, and escalate
  genuine design trade-offs to maintainers. Use whenever you open a PR or push a commit to one.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green AND no comment
is left unresolved.**

## When you open the PR

Apply a label — the repo triages and filters by them. Fetch the real list first with
`gh label list --limit 100`, because the set changes and a guessed label silently fails to
apply. Pick the one naming what the PR *is* (`bug`, `feature`, `docs`, `chore`, `refactor`) and
add a topic label (`anthropic`, `MCP`, `evals`, …) where one fits:
`gh pr edit <number> --add-label <label>`.

Labelling needs triage permission on the repo (Pydantic team members and their agents). If it
fails, quote the actual error rather than concluding you lack permission. Size labels are
applied automatically — don't set them.

## Before you push
- Commit the exact state you intend to push. Leave nothing staged, unstaged or uncommitted unless
  the user's instructions override this.
- Run `pre-push-review`. Address every finding, commit the fixes, and repeat the review until it
  returns no findings. This applies before the first PR push and between every later PR iteration.
- Never force-push an open PR branch. Push follow-up commits so previous reviews remain valid;
  maintainers can squash them when merging.
- Attempt the push. If it fails, read the real error — do not preemptively decide you lack
  permission from a flag or setting.

## After you push — the loop
1. **Watch CI to a terminal state.** Don't idle. If it fails, diagnose: fix if the failure is
   yours; if it's a known flake or pre-existing on main, say so with evidence.
2. **Triage every comment** (bots and humans alike). For each one:
   - **Valid** → fix it, then reply saying what changed, and react 👍.
   - **Invalid** → reply explaining concretely why (with code evidence), and react 👎.
   - Never silently ignore a comment, and never resolve a thread without a reply.
3. **Escalate real trade-offs, don't guess.** If a comment needs a maintainer decision (a design
   choice, an API trade-off, a behavioral default), leave a comment containing: the background,
   your reasoning, the decision that needs making, the trade-offs (pros/cons of each option), and
   your recommendation. Then **poll every 30 minutes for a reply** and continue when it lands.
4. Repeat until CI is green and no comment is outstanding.

## When the loop completes — consider a deep `douwebot` review

The repo has two standards reviewers, and they are independent:

- **`CI Review`** runs automatically once the `CI` workflow succeeds on the PR's current head. It
  owns the `APPROVE`/`REQUEST_CHANGES` verdict and has the more rigorous process — severity scale,
  sub-agent fan-out, per-finding verification.
- **`douwebot`** runs only when the `douwebot` label is applied, on a stronger model. It posts
  inline comments and no verdict, and it deletes the label when it finishes, so each application
  buys exactly one review of the diff as it stands at that moment.

Applying the label adds a second opinion; it does not suppress or replace `CI Review`.

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
- **Known refusal:** the job fails without reviewing if the PR touches `AGENTS.md`, `CLAUDE.md`, or
  anything under `.claude/` — a security guard against a PR editing the reviewer's own
  instructions. Don't apply the label to those PRs; the red check is the guard working.
- **Afterwards, re-enter the loop.** The review posts comments that need the same triage as any
  other.
