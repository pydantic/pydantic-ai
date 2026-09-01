---
name: refresh-issue-brief
description: Re-fetch the GitHub issue(s) linked in issue-brief.md and update the brief if new comments shift scope, criteria, or constraints. User-invoked only — never run automatically. Use when the user flags new issue activity (e.g. "check the latest comments on the issue").
---

# Refresh Brief

Re-fetch the issue(s) referenced in `.claude/skills/branch-context/issue-brief.md` and update the brief only where genuinely new content warrants it.

## When to run

Only when the user tells you to. Typical triggers:
- "check the latest comments on the issue"
- "the issue got updated, re-read it"
- "has anything changed upstream on the issue?"

Never run autonomously. The brief is meant to stay stable across many sessions — refreshing it every time would defeat the purpose.

## Process

### 1. Parse current brief

Read `.claude/skills/branch-context/issue-brief.md` frontmatter:
- `issues:` list (numbers + URLs)
- `last_fetched_at`
- `last_fetched_comment_count`

If `issues:` is empty (free-text problem, no linked issue), print: "No linked issue to refresh. Update the brief manually if scope changed." and exit.

### 2. Fetch current state

For each issue:
```bash
gh issue view <N> --json title,state,comments,labels,body
```

Collect:
- Current comment count
- Any comments with `createdAt > last_fetched_at`
- Current title / state / labels (flag changes)

### 3. Compare

For each issue, build a short summary of deltas:
- New comments (count + one-line summary of each)
- Title changed? state changed (reopened/closed)? labels added/removed?
- Body edited? (compare first 200 chars against what the brief cites)

If no deltas anywhere: update `last_fetched_at` + `last_fetched_comment_count` in the brief frontmatter and print "No changes since last fetch." Exit.

### 4. Classify deltas

For each delta, classify the impact:
- **Scope shift** — issue now asks for more/less than before → update `## Scope`
- **New constraint** — e.g. a maintainer said "must preserve backwards-compat" → update `## Constraints`
- **New success criterion** — e.g. "also needs to cover provider X" → update `## Success criteria`
- **Reference worth adding** — linked PR / related issue → update `## References`
- **Noise** — clarifying questions, bike-shedding, reactions → ignore

### 5. Confirm with user

Before writing any changes, use the harness's structured question mechanism for any non-noise delta:
- Summarize the delta in one line
- Show the proposed brief edit
- Options: "Apply", "Skip", "Edit differently"

### 6. Write updates

For approved deltas, edit `.claude/skills/branch-context/issue-brief.md`:
- Update the relevant section(s) — keep the tight format from the brief template
- Always update frontmatter: `last_fetched_at` (ISO now), `last_fetched_comment_count` (fresh total)

### 7. Log a decision entry

If any delta changed scope/criteria/constraints, append one line to `pr-decisions.md`:

```bash
.claude/skills/branch-context/append-pr-decision.sh \
  "brief refresh: <short title>" \
  "<what changed in the brief>" \
  "<which comment/event triggered it>" \
  "<link to the comment or issue event>" \
  "-"
```

Use iter `-` (this runs outside the ralph loop).

## Rules

- Never rewrite the whole brief — targeted edits only
- Never add prose research — the brief stays a contract, not a log
- Always cite the triggering comment via URL in the decision entry
- If the user says "just re-read it" without approving changes, still update the frontmatter timestamps so we know when we last checked
