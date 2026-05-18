---
name: branch-context
description: Branch-local issue brief and decisions log. Read-only reference — two data files under this dir are autoloaded into every session via `@` imports in CLAUDE.local.md. Don't invoke to "do" anything; read the files directly.
allowed-tools: Read
---

# Branch Context

This directory holds two files that persist branch-specific context across every Claude session in this worktree:

- `issue-brief.md` — synthesis of the issue(s) the branch addresses. Written by `/initialize-worktree`. Refreshed only on request via `/refresh-issue-brief` (when the user flags that the issue has new comments).
- `pr-decisions.md` — append-only log of non-obvious decisions made while working the PR. Written by ralph-phase agents and the stop hook; never rewritten or summarized.

Both files are loaded automatically: `CLAUDE.local.md` has `@` imports at the top, so any session that reads `CLAUDE.local.md` picks them up.

## When to write

- **issue-brief.md** — write only via `/initialize-worktree` (fresh) or `/refresh-issue-brief` (incremental update). Never edit in-session freestyle; it's a synthesis, not a scratchpad.
- **pr-decisions.md** — append an entry whenever you make a decision that the issue didn't already spell out. Entry format below. Keep entries brief — one line each for decision, why, source link.

### pr-decisions.md entry format

```
## YYYY-MM-DD · <short title> · iter <ralph iteration, or "-" if not in ralph loop>
- Decision: <one line>
- Why: <one line>
- Source: <link to comment/thread/issue/commit — this is mandatory>
- Supersedes: <ref to earlier entry title, if this overrides one>
```

No prose. No multi-paragraph rationale. If the decision needs more context, the link carries it.

## When to read

Both files are autoloaded — you already have them. Explicit re-reads only needed after `/refresh-issue-brief` updates the brief mid-session, or after appending a decision and wanting to verify formatting.

## Scope boundary

- This is not the place for research notes, investigation logs, or interim reports — those go in `local-notes/`.
- This is not the place for implementation state (ralph-state.json, goals.json, plan-output.md) — those live in `start-worktree-loop/`.
- Decisions logged here should be *load-bearing for future reviewers* — ones that would be confusing without the link. Ordinary implementation details don't belong.
