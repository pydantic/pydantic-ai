---
name: initialize-worktree
description: Study a problem or GitHub issue in a fresh worktree, plan a fix, optionally implement it, and open the initial PR. Use when starting work on a new problem — not for advancing existing PRs.
---

# Initialize Worktree — From Problem to PR

Study a problem or GitHub issue, research the codebase, plan a fix, optionally implement it, and open the initial PR. Assumes the worktree already exists (created via `pyai-worktree`).

## Startup

1. Read `CLAUDE.md` and `CLAUDE.local.md` for project context
2. Skim the DDD+ protocol + reviewer priority in the autoloaded `CLAUDE.local.md` for shared vocabulary

## Step 1 — Parse Input

Parse `$ARGUMENTS`:
- **GitHub issue URL or number** (e.g. `https://github.com/pydantic/pydantic-ai/issues/1234` or `#1234`) → fetch via `gh issue view <number> --json title,body,comments,labels,state,updatedAt`
- **Free text** → treat as problem description
- **Empty** → ask the user to link an issue or describe the problem.

## Step 2 — Study

Inspect the codebase to understand the problem:
- For issues: read all comments to understand discussion, decisions, maintainer guidance
- Identify key files, affected areas, related tests
- Search for prior attempts or related PRs if relevant

Be thorough — this research informs everything downstream.

## Step 3 — Write the branch-context files

Write `.claude/skills/branch-context/issue-brief.md` from scratch (replace the template). Use this schema — keep it tight, this file is autoloaded on every future session and token cost compounds:

```markdown
---
last_fetched_at: <ISO timestamp now>
last_fetched_comment_count: <count from gh issue view>
branch: <git rev-parse --abbrev-ref HEAD>
related_pr: TBD
issues:
  - number: <N>
    url: <url>
    title: "<title>"
    type: <bug|improvement|feature>
    role: <primary|related|follow-up>
    updated_at: <issue updatedAt from GitHub>
---

# Issue Brief

## Issues
- [#<N> — <title>](<url>) — type: <...> — role: <...>

## Problem
<1–2 sentences>

## Current behavior vs. expected
(bugs only — omit for features/improvements)

## Scope
**In:**
- <...>

**Out:**
- <...>

## Success criteria
| # | Criterion | Test |
|---|-----------|------|
| 1 | <...> | <test path or "to add"> |

## Constraints
- <backwards-compat / public-API / provider-parity statements>

## Affected surface
- <file> — <role>

## References
- [<label>](<url>) — <relevance>
```

Rules:
- Multi-issue branches: list every linked issue in the `issues:` frontmatter + `## Issues` section. Share one success-criteria table across them unless they're genuinely independent.
- Free-text problems (no issue): use `issues: []`, set `role: n/a`, keep everything else.
- Don't paste research prose here — it belongs in `local-notes/`. This file is a contract, not a log.

Then initialize `.claude/skills/branch-context/pr-decisions.md` by copying `pr-decisions.template.md` if it doesn't already exist (a fresh worktree starts with an empty log).

## Step 4 — Assess Complexity

Make an automatic assessment, then confirm it through the harness's structured question mechanism when available:

**Complex** (plan-only PR) — indicators:
- Issue has extensive discussion with multiple viewpoints
- Multiple ambiguous design decisions
- Cross-cutting architectural changes
- Large surface area across many files
- User says so

**Simple** (implement + PR) — indicators:
- Clear bug fix with obvious solution
- Straightforward feature addition
- Small, contained change
- Obvious solution path from the research

Ask: 'Based on my research, I assess this as [simple/complex]. [1-2 sentence reasoning]. Should I implement directly, or open a plan-only PR for discussion first?'

## Step 5a — Complex Path (plan-only PR)

1. Write plan to `.claude/plans/<branch-name>.md`
2. Commit the plan file only
3. Push and create a **draft** PR:
   - Title: concise description of the change
   - Body: follow the tracked `pushing-commits-to-the-repo` skill. Put each closing issue reference
     immediately after the attribution line, then include the plan and note that implementation
     follows after alignment.
4. Update `.claude/skills/branch-context/issue-brief.md`: set `related_pr` in frontmatter to the new PR URL
5. Print: 'Draft PR created with plan. Discuss in the PR, then implement after alignment.'

## Step 5b — Simple Path (implement + PR)

1. Write plan to `.claude/plans/<branch-name>.md`
2. Implement changes per plan
3. Run `make format && make lint` — fix issues until clean
4. Commit all changes, then follow the tracked `pushing-commits-to-the-repo` skill through its
   independent pre-push review gate
5. Push and create PR:
   - Title: concise description of the change
   - Body: follow the tracked `pushing-commits-to-the-repo` skill, including closing issue placement
     and the collapsed PR-decisions section.
   - If issue exists: include `Closes #<number>` or a non-closing issue link as appropriate.
6. Update `.claude/skills/branch-context/issue-brief.md`: set `related_pr` in frontmatter to the new PR URL
7. Continue the tracked push lifecycle through current-head CI, hosted review, feedback close-out,
   and final metadata; do not stop at PR creation

## Notes

- PR feedback is inspected with the `pr-review-feedback` helpers and handled through the tracked
  push lifecycle, or by `/pr-orchestrator` from the manager
- Always create the plan file regardless of complexity — it documents intent
- For the PR body, follow the tracked `pushing-commits-to-the-repo` skill and root `AGENTS.md`.
