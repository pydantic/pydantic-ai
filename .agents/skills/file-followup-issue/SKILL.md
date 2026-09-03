---
name: file-followup-issue
description: Capture a real, net-positive bug or improvement you discovered that is OUT OF SCOPE for the current PR — the PR already closes its issue and is stable, and the new work is non-trivial (adds its own decisions, carries uncertainty, or widens blast radius beyond what the issue needed). Opens a GitHub issue with full context for a future session and does NOT touch the current PR. NOT for trivial in-scope fixes (just do those) or non-issues (dismiss those).
---

# File a follow-up issue

You discovered work that is real and worth doing, but doesn't belong in the current PR. Park it as a GitHub issue with enough context that a future session can pick it up cold and open its own PR — then get back to the current PR without touching it.

## Gate — all three must hold

Only file an issue when **all** are true:

1. **The current PR is complete and stable** — it closes its original issue; this discovery is not needed for that.
2. **The discovery is real and net-positive** — a genuine bug or improvement, not a nitpick.
3. **It's non-trivial and out of scope** — fixing it here would add its own design decisions, carry unresolved uncertainty, or widen the diff's blast radius past what the issue needed.

If a gate fails, do the alternative and **stop** — don't file:

- Trivial **and** in-scope → just fix it in this PR.
- In-scope but larger → it belongs in this PR's plan, not a new issue.
- Not actually a real problem → drop it (or, if it's a review thread, dismiss it).

## Steps

1. **Dedup.** Search for an existing issue before creating:
   ```bash
   gh issue list --state all --search "<3-5 distinctive keywords>" --limit 20
   ```
   If a clear match exists, link to it instead of creating a duplicate — and skip to step 4 (log) referencing the existing issue.

2. **Confirm before creating** (outward action). Show the user the drafted title + body and get a yes.

3. **Create** with the template below:
   Both title and body quote code and reviewer text, where `$(...)` and backticks would be
   expanded by the shell before `gh` ever runs. Write each to its own file with your editor tool,
   then load the title through `cat` — a shell variable's contents are not reparsed for
   substitution:
   ```bash
   title=$(cat <path to the title file you wrote>)
   gh issue create --title "$title" --body-file <path to the body file you wrote>
   ```
   Use the ownership shortcut only when the acting account can assign issues and apply repository labels.
   If a maintainer will tackle the follow-up directly or soon, use
   `--assignee <maintainer-login> --label pydanty:skip`. This includes an AICA acting for that
   maintainer. Assign the maintainer who will own the work. The label prevents Pydanty from running
   triage or `retry-pr` on the issue.

   Do not use the ownership shortcut for an external contributor. External contributors cannot
   apply repository labels, so leave their follow-up issues in normal triage.

   Add `--label <name>` only for labels you've confirmed exist (`gh label list`); a missing label fails the command.

4. **Log the deferral** in `pr-decisions.md` so the current PR's reviewers see *why* this was parked rather than fixed:
   ```bash
   .agents/skills/branch-context/append-pr-decision.sh \
     "defer: <short title of discovery>" \
     "Filed as #<N> instead of fixing here" \
     "Out of scope for this PR — <decisions / uncertainty / blast-radius, pick one>" \
     "<new issue URL>" \
     "-"
   ```

5. **Report** the issue URL. **Do not expand the current PR** — that's the whole point.

## Issue body template

```markdown
## Background
<what we were doing and where this surfaced; link the originating PR>

## The problem
<the bug or improvement, concrete and specific>

## Why it's out of scope for the originating PR
<the specific reason: own design decisions, unresolved uncertainty, or blast radius>

## Where
- `path/to/file.py` — `Class.method`   <!-- filepath + symbol, no line numbers -->

## Suggested direction (low-confidence)
<optional starting point — mark clearly as a lead, not a decision. Omit if you have none.>
```
