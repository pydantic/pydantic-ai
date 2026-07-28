---
description: Run a high-judgment local review of the current branch before pushing, both before a
  PR exists and between PR iterations
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash(gh issue view:*)
  - Bash(gh pr view:*)
  - Bash(git diff:*)
  - Bash(git log:*)
  - Bash(git merge-base:*)
  - Bash(git status:*)
  - Bash(git rev-parse:*)
  - WebSearch
  - WebFetch
---

# Pre-push Review

Use the strongest locally available reviewer to catch problems while they are still cheap
to fix. Run this before the first push and again before every later push to an existing PR.

This is the local counterpart to `douwebot`: a high-judgment standards review paid for by
the developer's model subscription. It is independent of the automatic `CI Review`, which
runs on GitHub after CI passes.

## Read the review rubric

Read the `prompt:` of the `douwebot` review job in `.github/workflows/bots.yml`. It is the
source of truth for what to look for, how to prioritise concerns, and what makes feedback
useful. Apply its review judgment and comment-quality rules, but ignore its hosted workflow
mechanics: triggers, checkout, model selection, pre-gathered file paths, and GitHub comment
tools.

Read the root `AGENTS.md`, `agent_docs/index.md` and its relevant topic guides, plus every
directory-specific `AGENTS.md` governing a changed file.

## Gather local and PR context

First run `gh pr view` for the current branch.

- **If a PR exists**, read its title, body, base branch, linked issue, comments and reviews.
  Review the entire branch diff against that base, not just the latest commit. Use the
  existing discussion to avoid duplicate findings and to detect concerns that remain
  unresolved after an iteration.
- **If no PR exists**, use `main` as the base and review against the task context available
  locally. Skip only PR metadata that does not exist; scope and readiness are still valid
  review concerns.

Gather the corresponding local state:

```bash
git status --short
git merge-base <base> HEAD
git diff <base>...HEAD --stat
git diff <base>...HEAD -W
git diff HEAD
```

The last command includes staged and unstaged work that has not reached `HEAD`. Read a large
diff in chunks, core implementation before tests, and skip generated files (`uv.lock`,
cassettes).

## Return the review locally

Do not post comments, submit a GitHub review, or modify the branch. Return only actionable
findings as text: `file:line`, the problem, and the concrete fix. Put higher-level concerns
before lower-level ones, following the ordering in the `douwebot` rubric. Say plainly when
there are no findings.
