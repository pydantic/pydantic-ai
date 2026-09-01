---
name: branch-context
description: Branch-local durable PR state — issue brief, decisions log, and session handoffs. Autoloaded brief + decisions + handoffs index every session; persist decisions as you work; EPEH or /handoff only on user request or explicit session turnover.
---

# Branch Context

This directory is the **single home** for durable PR/branch state across sessions and harnesses. Three live surfaces:

| File / dir | Role | Lifetime |
|------------|------|----------|
| `issue-brief.md` | Synthesis of the issue(s) this branch addresses | Rewritten only by `/initialize-worktree`, `/refresh-issue-brief`, `/adopt-pr` |
| `pr-decisions.md` | Append-only log of non-obvious PR-shaping decisions | Append forever; supersede, never edit |
| `handoffs/` + `handoffs-index.md` | Append-only session handoffs for the next agent | Never overwrite a handoff file; index points at latest |

Autoload (via `CLAUDE.local.md` `@` imports): brief, decisions, **handoffs-index** (not full handoff bodies). Full handoffs live under `handoffs/` — read the latest path from the index.

## Session defaults (every agent, every harness)

**On start (before coding):**

1. Confirm branch matches the brief's `branch:` field (`git rev-parse --abbrev-ref HEAD`).
2. Read the autoloaded brief + decisions + handoffs-index.
3. **Read your lane's latest handoff — and only your lane's:**

   ```bash
   .agents/skills/branch-context/latest-handoff.sh   # prints one path, or nothing
   ```

   Read exactly the file it prints. If it prints nothing, **your lane has no handoff**: say so and start from the live board. Other lanes' entries are visible in the index and are *not* yours — reading one makes you adopt PRs another manager drives. Never pick an entry by eye off the index; the script resolves the lane for you.
4. If the brief is still the unfilled template → `/initialize-worktree` or `/adopt-pr` first.
5. **Load the skills this session runs on before acting.** Always `i-have-adhd` (how David reads: lead with the result or decision, use the harness's structured question mechanism when available, no preamble/recap/closers). In the **manager** worktree also `pr-orchestrator` — it owns the tmux multi-PR workflow and its `helpers/` are the interface to every worker window; don't hand-roll `tmux send-keys` when a helper exists. Loading these late costs a half-session of output David skims past.

**Every handoff body must repeat step 5 explicitly** — a fresh agent reads the handoff before it reads this file.

**While working — persist without being asked:**

- Non-obvious decision (path A over B, plan deviation, ambiguous thread resolution) → append via `append-pr-decision.sh` **in the same turn** as the decision, not "later."
- Do not rely on chat history surviving a clear or a new session. Disk in this dir is the continuity channel.
- Do **not** self-trigger a handoff because you "think context is full" — agents misjudge that and it primes early stops. Handoff/EPEH only when the user asks, or when session turnover is already decided (user clearing, switching harness, ending the sitting).

## When to write each surface

### `issue-brief.md`

Only via `/initialize-worktree` (fresh), `/refresh-issue-brief` (user-flagged issue activity), or `/adopt-pr` (existing PR). Never freestyle-edit mid-session.

### `pr-decisions.md`

Append whenever you make a decision the issue didn't already spell out:

```bash
# Named flags preferred (avoids arg-order mangling):
.agents/skills/branch-context/append-pr-decision.sh \
  --title "<short title>" \
  --decision "<one-line decision>" \
  --why "<one-line why>" \
  --source "<source url — mandatory>" \
  [--iter N] [--supersedes "<earlier title>"]
# Positional compat: title decision why source [iter] [supersedes]
# — do NOT put iter as the second arg.
```

Entry shape (script writes this):

```
## YYYY-MM-DD · <short title> · iter <N or "-">
- Decision: <one line>
- Why: <one line>
- Source: <link — mandatory>
- Supersedes: <earlier title, if applicable>
```

### Handoffs

Never overwrite another session's handoff. One handoff per session:

```bash
.agents/skills/branch-context/append-handoff.sh [--writer <skill>] "<one-line summary>" [path-to-body.md]
```

Pass `--writer` when a skill owns the handoff (`--writer manager-handoff`); it tags the index line so a reader can see which writer produced it. Call it again in the same session and it **amends** your existing handoff — same file, same index line — rather than appending a rival entry the next agent would have to choose between.

**Lanes.** Every entry stores an immutable `lane-id:<id>` from the host conversation/thread when
available (`$CLAUDE_CODE_HOST_SESSION_ID` or `$CODEX_THREAD_ID`), then tmux, with `$HANDOFF_LANE` as
the explicit override. It also carries `lane:<label>` for display. Several managers share this
worktree and index; `latest-handoff.sh` matches only the exact lane ID. Labels default to a short id
and live in `handoffs/.lanes` (`<id> <label>`) — rename one there to something human
(`Manager 2 - daily`) without changing handoff ownership. Start the handoff body with the label and
the board it covers, so a successor whose live board disagrees can catch the mismatch.

If you omit the body path, the script opens a stub you (the agent) must fill via Write/Edit before stopping. Prefer writing the full body first, then calling the script with that path — or write the body to the path the script prints.

Handoff body sections (required):

```markdown
# Handoff · YYYY-MM-DD · <summary>

## Done
- …

## Next
- … (ordered; first item is what the next agent starts on)

## Commitments & constraints carried forward
- … (verbatim; or "none")

## Key paths
- `path` — why

## Open questions
- … (or "none")

## Branch-context pointers
- Brief: issue-brief.md (still valid? yes/no)
- Decisions appended this session: <titles or "none">
- Related plan file (if any): <path>
```

**Commitments & constraints is a required check, not an optional extra.** Before writing the handoff, sweep the session for two things and **quote them verbatim — do not paraphrase**:

- **Constraints the user stated** that aren't already in `issue-brief.md`'s Constraints section. Modality is the payload: "always X **unless** Y" and "always X" are different instructions, and a one-line paraphrase is exactly where the qualifier gets dropped.
- **Promises you made and haven't kept** — to David ("I'll add the regression test next"), or *on the record* in a PR/review comment ("I'll file a follow-up issue", "I'll re-run this once CI clears"). A promise made to a reviewer and then silently dropped across a session boundary is the expensive kind: the next agent can't know it exists, and the reviewer is still waiting.

A longer, accurate handoff beats a short lossy one. Never compress this section to save space.

## EPEH — enter-plan-exit-handoff

Use when the **user** asks to hand off, clear and continue, or otherwise turn the session over. Not when you guess the context window is full.

1. **Enter plan mode** (if the harness has it) — remaining work as a concrete plan (next steps, files, verification).
2. **Persist before exit:**
   - any unlogged decisions → `append-pr-decision.sh`
   - write handoff body under `handoffs/` → `append-handoff.sh` (index updated)
   - optional: point the plan at the latest handoff path
3. **Exit plan / turn over** — harness-specific:

| Harness | What to do |
|---------|------------|
| **Claude Code** | `ExitPlanMode` → choose **clear context and continue** with the plan. Fresh session inherits autoloaded branch-context + should read latest handoff. |
| **Grok Build** | Plan mode has **no** clear-context-on-approve. After writing handoff + plan: stop and tell the user to start a **new** `grok` session in this worktree. First action for the new agent: read handoffs-index + latest handoff. (`/compact` / `/flush` are not a full handoff.) |
| **Codex** | Write handoff + plan; use new session / fork if available. Do not assume clear-context-on-plan-exit. Next session: same first-read as Grok. |
| **Pi** | Prefer a plan-mode **extension** that writes the handoff then seeds a new session. If no extension: same file out as Grok, and propose (or install) an extension that automates EPEH. |
| **OpenCode** | Write the handoff + plan, then start a new session/thread with the harness's supported command or UI. Re-read the lane handoff before acting; do not assume plan exit clears context. |

Self-programmable harnesses (Pi, and any that can install extensions): **implement EPEH once** rather than re-documenting the manual out every time.

### `/handoff` alias

User-invoked `/handoff` runs the **persist steps only** (decisions + handoff file + index) without requiring plan mode. Same writers, same paths. EPEH = plan mode + that persist + harness exit.

## Scope boundary

- Not for research notes / repro scripts → `local-notes/`
- Not for durable codebase facts that outlive the PR → per-worktree memory
- `pr-decisions.md` vs memory: if removing the linked thread would make **this PR's** diff confusing → decisions. If the fact still helps after merge → memory. Often both.

## Helpers

```bash
.agents/skills/branch-context/status.sh          # JSON: brief/decisions/handoffs state
.agents/skills/branch-context/append-pr-decision.sh …
.agents/skills/branch-context/append-handoff.sh …
```
