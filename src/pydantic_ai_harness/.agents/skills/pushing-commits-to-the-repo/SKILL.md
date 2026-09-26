---
name: pushing-commits-to-the-repo
description: What to do when you open a PR and every time you push -- label the PR, watch CI to
  green, triage every review comment to a reply and a reaction, and escalate genuine design
  trade-offs to maintainers. Use whenever you open a PR or push a commit to one.
---

# pushing-commits-to-the-repo

Pushing starts a loop; it does not end the task. **Work stops only when CI is green AND no comment
is left unresolved.**

## When you open the PR

Apply a label -- the repo triages and filters by them. Fetch the real list first with
`gh label list --limit 100`, because the set changes and a guessed label silently fails to apply.
Pick the one naming what the PR *is* (`bug`, `enhancement`, `documentation`) and add a topic label
where one fits (`capability`, `primitive`, `agent-feature`, `core-change`, `code-mode`, `media`,
`externalization`, ...): `gh pr edit <number> --add-label <label>`.

Labelling needs triage permission on the repo. If it fails, quote the actual error rather than
concluding you lack permission.

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
