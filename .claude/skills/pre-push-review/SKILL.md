---
description: Review the current branch against main, simulating the automatic CI Review before there is a PR
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

Run the `CI Review` reviewer's judgment against your branch before there is a PR for it
to run on.

## Where the review itself is defined

Read these rather than reviewing from memory — this skill deliberately does not restate
them, so it cannot drift from what CI actually does:

- **What to look for, how to prioritise, how to word a finding** —
  `.github/workflows/shared/prompts/pydantic-ai-pr-review.md`. This is the complete
  reviewer prompt.
- **The severity scale, the false-positive catalog, and the calibration examples** — the
  `review-instructions.md` heredoc in `scripts/gather-pydantic-ai-review-context.sh`. At
  runtime the reviewer reads that file out of `.review-context/`; locally, read the
  heredoc.
- **How the reviewer is triggered, gated and fed its context** —
  `.github/workflows/pydantic-ai-pr-review.md`.

(`bots.yml`'s `douwebot` job is a different reviewer — on-demand, label-triggered, no
verdict. It is not what this skill simulates.)

## Gather the context locally

There is no PR, so assemble the equivalent yourself:

```bash
git merge-base main HEAD
git diff main...HEAD --stat
git diff main...HEAD -W
```

Read a large diff in chunks, core implementation before tests, and skip generated files
(`uv.lock`, cassettes).

Then read the root `AGENTS.md` plus the directory-specific `AGENTS.md` for each directory
containing changed files.

## Local adaptations

The prompt assumes a PR exists. Four things change without one:

- **Skip everything PR-scoped** — no description, linked issues, duplicate-PR check, or
  prior review threads.
- **Skip "should this change exist"** — the user has already decided it should.
- **Skip the verdict.** There is nothing to approve or request changes on; the findings
  are the output.
- **Emit findings as text**, not inline comments — file:line, the problem, the fix.
  Group by severity, highest first. Say so plainly when there are none.
