---
name: pushing-commits-to-the-repo
description: What to do after you push — watch CI to green, triage every review comment to a reply
  and a reaction, and escalate genuine design trade-offs to maintainers. Use whenever you push a
  commit to a PR.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green AND no comment
is left unresolved.**

## Before you push
- Attempt the push. If it fails, read the real error — do not preemptively decide you lack
  permission from a flag or setting.
- Leave nothing unstaged or uncommitted locally, unless the user's instructions override this.

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

## When the loop completes — consider a deep `auto-review`

Adding the `auto-review` label triggers a one-shot in-depth review (the `Review` job in
`bots.yml`) that is significantly more expensive and thorough than the per-push CI reviewer.
The label removes itself after the run, so each application buys exactly one review of the
diff as it stands at that moment.

Once the loop above has terminated — CI green, every comment triaged — decide whether to apply
it before handing the PR back or requesting merge:

- **Apply it last, not early.** It won't re-run on later pushes, so a deep review of a
  still-moving PR is wasted money. Wait until CI is green and feedback from the per-push
  reviewer and any other bots has been addressed.
- **Use judgment on whether it's warranted.** Skip it when you're highly confident there's
  nothing left to catch (typo fixes, dependency bumps, mechanical chores). Apply it for
  substantive changes: new features, behavior changes, public API surface, non-trivial bug
  fixes — and user-facing docs, where it catches things like examples using outdated models.
  In between, weigh cost against risk; smaller PRs are cheaper to review, so lean toward
  applying when unsure.
- **How:** `gh pr edit <number> --add-label auto-review`. This requires triage permission on
  the repo (Pydantic team members and their agents). If it fails, quote the actual error —
  don't skip it based on an assumed lack of permission.
- **Known refusal:** the review job exits without reviewing if the PR touches `AGENTS.md`,
  `CLAUDE.md`, or anything under `.claude/` (a security guard). Don't apply the label to
  those PRs.
- **Afterwards, re-enter the loop.** The review posts comments that need the same triage as
  any other. If your fixes are substantial enough to deserve another deep pass, you may
  re-apply the label — but usually once is enough.
