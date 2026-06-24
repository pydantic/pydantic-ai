---
# Shared pre-gathered context and review-thread handling for gh-aw review prompts.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
---

## Pre-gathered context

A pre-agent step ran `scripts/gather-pydantic-ai-review-context.sh` and
wrote everything you need to `/tmp/gh-aw/.review-context/`. **Read these
files instead of calling the GitHub API.**

- `pr-details.json` — title, body, author, branches, labels, draft/state.
- `pr-size.txt` — `{N} files, {M} diff lines`.
- `changed-files.txt` — paths in this PR with `+N -M` change counts and the
  matching `diff/<path>.diff` filename.
- `file-orderings/az.txt`, `file-orderings/za.txt`,
  `file-orderings/largest.txt` — the same file list in three orderings.
- `diff/<path>.diff` — per-file diffs with function context, annotated
  with `NL:<n>` for new-side and `OL:<n>` for old-side line numbers.
  **Inline comments require an `NL:` line.**
- `pr-comments.txt` — issue-style PR discussion.
- `review-comments.txt` — inline review threads with diff hunks and
  per-thread `RESOLVED` / `UNRESOLVED` / `OUTDATED` state.
- `related-issues.txt` — linked issues referenced by the PR body.
- `agents-md.txt` — `AGENTS.md` excerpts for directories the PR touches.

The annotated diffs are the **source of truth** for what changed.

**If a file is missing** (the pre-agent step may have warned), fall back to
`gh pr view` / `gh pr diff` for that piece — but only that piece. Don't
re-fetch what is already on disk.

## Handling existing review threads

For each thread in `review-comments.txt`, the **state** field tells you what
to do with any finding that would land on the same `path:line`:

- `[UNRESOLVED]` — already flagged. **Do not duplicate.**
- `[RESOLVED]` with a reviewer reply (e.g. "intentional", "won't fix") —
  decision is final. **Do not re-flag.**
- `[RESOLVED]` without a reply — author likely fixed it. **Do not re-raise**
  unless your reading shows the fix introduced a new problem.
- `[OUTDATED]` — the code has shifted under the comment. Only re-flag if
  the issue still applies to the *current* diff.

When in doubt, do not duplicate. Redundant comments erode trust.
