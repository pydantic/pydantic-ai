---
name: pre-push-review
description: Run the repository's high-judgment standards review locally on the exact candidate commit
  before it is pushed
allowed-tools: Read Glob Grep WebSearch WebFetch
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

Read the stable base checkout's root `AGENTS.md`, `agent_docs/index.md` and its relevant
topic guides, plus every directory-specific `AGENTS.md` governing a changed file.

## Receive trusted review context

The implementing agent, not the fresh reviewer, prepares a read-only review bundle from the stable
base checkout:

1. Validate the supplied values as full commit SHAs and confirm both are commit objects.
2. Confirm the candidate worktree is clean and its HEAD equals the candidate SHA.
3. Collect the task or issue and, when a PR exists, its title, body, comments, reviews, inline review
   threads, and resolution state with trusted read-only GitHub tooling.
4. Capture the exact endpoint diff with external diff and text conversion disabled:

```bash
git -c diff.external= -c diff.trustExitCode=false diff --no-ext-diff --no-textconv --stat <review-base-sha> <candidate-head-sha> --
git -c diff.external= -c diff.trustExitCode=false diff --no-ext-diff --no-textconv -W <review-base-sha> <candidate-head-sha> --
```

Store that material outside the candidate worktree and give the fresh reviewer only its path, the
exact SHAs, and the stable base checkout. The reviewer reads stable instructions and repository
context directly, but reads candidate content only from the bundle. Read a large diff in chunks,
core implementation before tests, and skip generated files (`uv.lock`, cassettes).

## Return the review locally

Do not post comments, submit a GitHub review, or modify the branch. Return only actionable
findings as text: `file:line`, the problem, and the concrete fix. Put higher-level concerns
before lower-level ones, following the ordering in the `douwebot` rubric. If there are no
findings, return exactly `current at <full-candidate-head-sha>`.
