---
name: branch-context
description: Branch-local issue brief and decisions log. Use when you need to read or append durable PR decisions that should survive across AICA sessions.
allowed-tools: Read, Bash(.agents/skills/branch-context/status.sh:*), Bash(.agents/skills/branch-context/append-pr-decision.sh:*)
---

# Branch Context

This skill manages two ignored, branch-local files under `local-notes/branch-context/`:

- `issue-brief.md` — synthesis of the issue(s) the branch addresses.
- `pr-decisions.md` — append-only log of non-obvious decisions made while working the PR.

Templates live beside this skill as `issue-brief.template.md` and `pr-decisions.template.md`. Do not commit live branch context files.

## When to write

- **issue-brief.md** — write only when bootstrapping or refreshing branch context from the issue/PR. It is a synthesis, not a scratchpad.
- **pr-decisions.md** — append whenever you make a decision that the issue didn't already spell out. Keep entries brief: one line each for decision, why, source link.

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

Read `local-notes/branch-context/issue-brief.md` before reviving an existing branch or after the user says issue/PR context changed. Read `pr-decisions.md` before changing an approach that may have been deliberately chosen earlier.

## Scope boundary

- This is not the place for research notes, investigation logs, or interim reports — those go in `local-notes/`.
- This is not the place for implementation state, test logs, or review reports.
- Decisions logged here should be *load-bearing for future reviewers* — ones that would be confusing without the link. Ordinary implementation details don't belong.
