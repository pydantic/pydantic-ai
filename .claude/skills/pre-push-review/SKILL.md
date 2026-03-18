---
description: Review the current branch against main, simulating the automated CI review from the bots workflow
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(git diff:*)
  - Bash(git log:*)
  - Bash(git merge-base:*)
  - Bash(git status:*)
  - Bash(git rev-parse:*)
  - WebSearch
  - WebFetch
---

# Pre-push Review

Simulate the automated CI review job locally before pushing or opening a PR.

## How it works

The review criteria are defined in the CI workflow, not duplicated here. Read the
**review prompt** directly from the source of truth:

- `.github/workflows/bots.yml` — find the `review` job's `prompt:` field (the YAML
  block starting with `Review this pull request`). This contains the full review
  criteria, priorities, and guidelines that CI uses.

Adapt those criteria for a local (pre-PR) review as described below.

## Steps

### 1. Read the CI review prompt

Read `.github/workflows/bots.yml` and extract the review prompt from the
`anthropics/claude-code-action` step in the `review` job. This is your primary
reference for what to look for and how to prioritize findings.

### 2. Gather local context

Since there's no PR yet, gather equivalent context locally:

```bash
# Determine the base branch (default: main)
git merge-base main HEAD

# Changed files
git diff main...HEAD --stat

# Function-context diffs (same -W flag the CI script uses)
git diff main...HEAD -W
```

For the diff, read it in manageable chunks — don't try to load everything at once
if it's large. Prioritize "core implementation" files over tests and generated files
(like `uv.lock` and cassettes).

### 3. Read relevant AGENTS.md files

Read the root `CLAUDE.md` (already in your system prompt) plus any directory-specific
`AGENTS.md` files for directories that contain changed files. The CI script checks
these paths:

- `docs/AGENTS.md`
- `pydantic_ai_slim/pydantic_ai/models/AGENTS.md`
- `tests/AGENTS.md`

Only read ones relevant to the changed directories.

### 4. Review

Apply the review criteria from the CI prompt, with these local adaptations:

- **Skip PR-specific checks**: no PR description, linked issues, duplicate PR checks,
  or prior review comments to consider.
- **Skip "should this PR exist" check**: assume the user intends to make these changes.
- **Output findings as text** instead of posting inline comments.

Focus areas (in priority order, per the CI prompt):

1. **Public API** — abstractions, class hierarchy, method signatures, type safety, backward compat
2. **Concepts & behavior** — design decisions, tradeoffs needing discussion
3. **Documentation** — voice, patterns, completeness
4. **Tests** — coverage, style, integration vs unit
5. **Code style** — AGENTS.md rule compliance, idiomatic Python

If there are high-level problems that would require significant rework, focus on those
and hold off on lower-level nits.

### 5. Present findings

Organize findings by priority tier. For each finding:

- Reference the specific file and line (`file:line`)
- Explain the issue concisely
- Suggest a fix if appropriate

If there are no findings, say so.

## Comment quality

Every finding must earn its place. Your review should never add noise:

- **Actionable only**: each finding must request a specific change, flag a concern that
  needs discussion, or suggest a concrete improvement. Do not comment on positive aspects
  ("excellent design", "good choice") — those are noise.
- **Concise**: 1-3 sentences per finding is almost always enough. No unnecessary lists,
  subheadings, or emojis.
- **Non-repetitive**: don't flag the same issue multiple times unless it appears in
  meaningfully different contexts.
- **No filler**: do not pad the review with observations that don't require action.
  If a choice is correct, don't mention it. If code follows the project's patterns,
  don't praise it. Focus exclusively on what needs to change or be discussed.
- **Friendly but not sycophantic**: use the tone of a helpful project maintainer.
  No compliments, no "great work", just clear, direct feedback.

If there are high-level problems that would require significant rework, focus on those
and skip lower-level nits entirely — they'll need to be revisited anyway.
