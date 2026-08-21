---
name: pre-push-review
description: Run the repository's high-judgment standards review locally on the exact candidate commit
  before it is pushed
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

When `.github/workflows/bots.yml` contains the `douwebot` job, read its `prompt:`; that is the
source of truth. Apply its judgment and comment-quality rules, but ignore hosted-workflow mechanics.
In repositories without that workflow, apply the same core rubric:

- Is the work ready, correctly scoped, non-duplicative, and aligned with maintainer guidance?
- Does it meet root and directory instructions, including public API and compatibility requirements?
- Does any behavior, design decision, or trade-off require explicit maintainer consideration?
- Review in priority order: public API, concepts and behavior, documentation, tests, code quality.
- If a high-level problem may invalidate lower-level work, report it first and defer the lower-level
  pass until remediation.
- Report only actionable concerns. Be concise, concrete, non-repetitive, and friendly without praise.

Read the root `AGENTS.md`, `agent_docs/index.md` and its relevant topic guides, plus every
directory-specific `AGENTS.md` governing a changed file.

## Gather local and PR context

If the dispatcher supplied a PR number, run `gh pr view <number>`; do not infer it from the base
checkout's current branch.

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
git rev-parse <review-base-sha> <candidate-head-sha>
git diff <review-base-sha>...<candidate-head-sha> --stat
git diff <review-base-sha>...<candidate-head-sha> -W
```

Read a large diff in chunks, core implementation before tests, and skip generated files
(`uv.lock`, cassettes).

## Return the review locally

Do not post comments, submit a GitHub review, or modify the branch. Return only actionable
findings as text: `file:line`, the problem, and the concrete fix. Put higher-level concerns
before lower-level ones, following the ordering in the `douwebot` rubric. If there are no
findings, return exactly `current at <full-candidate-head-sha>`.
