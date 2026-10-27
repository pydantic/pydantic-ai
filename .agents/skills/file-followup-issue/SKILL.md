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
   Preserve that issue's ownership and routing labels. Deduplication does not authorize a new route.

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
   Apply the [maintainer routing mechanics](#maintainer-routing-mechanics) for a maintainer's selected route.

4. **Log the deferral** in `pr-decisions.md` so the current PR's reviewers see *why* this was parked rather than fixed:
   For a newly created maintainer issue, include the selected route and human owner (or no local owner) in the decision.
   Bind contributor-derived fields with quoted heredocs, as in [adopt-pr](../adopt-pr/SKILL.md#step-5--backfill-pr-decisionsmd-from-resolved-threads).
   This also applies when deduplication skipped issue creation and Step 3 has no bound title.
   Never paste contributor text into double-quoted command arguments.

```bash
title=$(cat <<'EOF'
defer: <short title of discovery>
EOF
)
decision=$(cat <<'EOF'
Filed as #<N> instead of fixing here
EOF
)
why=$(cat <<'EOF'
Out of scope for this PR — <decisions / uncertainty / blast-radius, pick one>
EOF
)
source=$(cat <<'EOF'
<new issue URL>
EOF
)
.agents/skills/branch-context/append-pr-decision.sh \
  --title "$title" --decision "$decision" --why "$why" --source "$source"
```

5. **Report** the issue URL. **Do not expand the current PR** — that's the whole point.

## Maintainer routing mechanics

For issues filed on a maintainer's behalf, honor their chosen route; otherwise ask which route they want.
Combine this choice with draft approval when both are needed. This routing also applies to non-follow-up issues.
Confirm labels against the [checked-in catalog](../pushing-commits-to-the-repo/labels.tsv) before using them; a missing label fails the command.

- **General queue:** create without `--assignee` or `pydanty:skip`, then run
  `gh issue edit <issue-number> --add-label pydanty:triage-issue`.
  This explicitly requests Pydanty triage, which team-authored issues otherwise skip.
- **Immediate work:** create with `--assignee <maintainer-login> --label pydanty:skip`.
  Assign the human maintainer who owns the work. The label suppresses Pydanty triage and `retry-pr` triggers.
  Carry that ownership onto the PR using the [PR skill](../pushing-commits-to-the-repo/SKILL.md#carry-maintainer-ownership-onto-the-pr).

Do not apply these maintainer routes for an external contributor. External contributors cannot
apply repository labels, so leave their issues in normal triage.

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
