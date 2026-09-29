---
name: branch-context
description: Branch-local durable PR state -- issue brief, decisions log, and session handoffs. Read the brief, decisions, and latest handoff at session start; append decisions as you work; write a handoff only on user request or explicit session turnover.
allowed-tools: Read, Write, Edit, Bash
---

# Branch Context

This directory is the home for durable PR/branch state that outlives a single
session. Three surfaces:

| File / dir | Role | Lifetime |
|------------|------|----------|
| `issue-brief.md` | Synthesis of the issue(s) this branch addresses | Rewritten only when adopting or re-syncing a branch |
| `pr-decisions.md` | Append-only log of non-obvious PR-shaping decisions | Append forever; supersede, never edit |
| `handoffs/` + `handoffs-index.md` | Append-only session handoffs for the next agent | Never overwrite a handoff file; index points at the latest |

The instances (`issue-brief.md`, `pr-decisions.md`, `handoffs-index.md`,
`handoffs/`) are per-branch and git-ignored. The scaffolding (this file, the
scripts, the `*.template.md` files) is committed. Instantiate the instances from
the templates, or via `/adopt-pr` when a PR already exists.

## Session defaults

**On start, before coding:**

1. Confirm the branch matches the brief's `branch:` field
   (`git rev-parse --abbrev-ref HEAD`).
2. Read `issue-brief.md`, `pr-decisions.md`, and `handoffs-index.md`.
3. Read the latest handoff -- the last entry in `handoffs-index.md` points at
   its file under `handoffs/`. That is what the previous session left for you.
   If the index has no entries, there is no handoff yet; start from the brief.
4. If the brief is still the unfilled template, populate it first: run
   `/adopt-pr` when a PR already exists, or write it from the linked issue when
   starting fresh.

**While working -- persist without being asked:**

- A non-obvious decision (picking path A over B, a plan deviation, an ambiguous
  thread resolution) goes into `pr-decisions.md` via `append-pr-decision.sh` in
  the same turn you make it, not "later".
- Disk in this directory is the continuity channel, not chat history. Do not
  rely on the conversation surviving a context clear or a new session.
- Do not write a handoff because the context "feels full" -- that misjudges and
  primes an early stop. Write a handoff only when the user asks, or when session
  turnover is already decided (the user is clearing, switching tools, or ending
  the sitting).

## When to write each surface

### `issue-brief.md`

Rewritten only when adopting a branch (`/adopt-pr`) or re-syncing after new
issue activity. Do not freestyle-edit it mid-session.

### `pr-decisions.md`

Append whenever you make a decision the issue did not already spell out:

```bash
.claude/skills/branch-context/append-pr-decision.sh \
  --title "<short title>" \
  --decision "<one-line decision>" \
  --why "<one-line why>" \
  --source "<source url -- mandatory>" \
  [--iter N] [--supersedes "<earlier title>"]
```

Named flags are preferred (they resist arg-order mistakes). Positional order is
`title decision why source [iter] [supersedes]` -- do not pass `iter` as the
second argument.

Entry shape (the script writes this):

```
## YYYY-MM-DD · <short title> · iter <N or "-">
- Decision: <one line>
- Why: <one line>
- Source: <link -- mandatory>
- Supersedes: <earlier title, if applicable>
```

When the decision restates a modal claim (always / never / only if / must /
unless), quote that clause verbatim instead of paraphrasing -- "always X unless
Y" compressed to "always X" is a different instruction.

### Handoffs

One handoff per session, newest last. Never overwrite another session's handoff.

```bash
.claude/skills/branch-context/append-handoff.sh [--writer <name>] "<one-line summary>" [path-to-body.md]
```

`--writer` tags the index line with the skill that produced the handoff. If you
omit the body path, the script writes a stub you must fill in via Write/Edit
before stopping; prefer writing the full body first and passing its path. To
revise a handoff you already wrote this session, edit the file in place rather
than appending a second index entry.

Handoff body sections (required):

```markdown
# Handoff · YYYY-MM-DD · <summary>

## Done
- ...

## Next
- ... (ordered; first item is what the next agent starts on)

## Commitments & constraints carried forward
- ... (verbatim; or "none")

## Key paths
- `path` -- why

## Open questions
- ... (or "none")

## Branch-context pointers
- Brief: issue-brief.md (still valid? yes/no)
- Decisions appended this session: <titles or "none">
- Related plan file (if any): <path>
```

**Commitments & constraints is a required check, not an optional extra.** Before
writing the handoff, sweep the session for two things and quote them verbatim --
do not paraphrase, the modality is the payload:

- **Constraints the user stated** that are not already in the brief's
  Constraints section. "Always X unless Y" and "always X" are different
  instructions, and a one-line paraphrase is where the qualifier gets dropped.
- **Promises you made and have not kept** -- to the user ("I'll add the
  regression test next"), or on the record in a PR/review comment ("I'll file a
  follow-up issue", "I'll re-run this once CI clears"). A promise made to a
  reviewer and then dropped across a session boundary is the expensive kind: the
  next agent cannot know it exists, and the reviewer is still waiting.

A longer, accurate handoff beats a short lossy one. Do not compress this section
to save space.

## Session turnover

Write the handoff (and any unlogged decisions) when the user turns the session
over. If the harness has a plan mode, capture the remaining work as a concrete
plan (next steps, files, verification), persist the handoff, then exit plan mode
so the fresh session inherits it. If the harness has no plan mode, persist the
handoff and tell the user to start a new session in this worktree whose first
action is to read `handoffs-index.md` and the latest handoff.

## Scope boundary

- Not for research notes or repro scripts -- those belong outside this
  directory.
- Not for durable codebase facts that outlive the PR -- those belong in
  longer-lived project docs or memory. Rule of thumb: if removing the linked
  thread would make this PR's diff confusing, it is a decision; if the fact still
  helps after merge, it belongs elsewhere.

## Helpers

```bash
.claude/skills/branch-context/status.sh              # JSON: brief/decisions/handoffs state
.claude/skills/branch-context/append-pr-decision.sh ...
.claude/skills/branch-context/append-handoff.sh ...
```
