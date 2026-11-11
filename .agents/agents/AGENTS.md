---
disable: true
---

<!-- OpenCode loads every Markdown file under `.opencode/agents/review/` as an agent, this one included. `disable: true` keeps it out of OpenCode's agent list. Claude Code skips it because it has no `name`. -->

# Authoring the review lanes

These are the lanes of `/review-branch` (`.agents/skills/review-branch/SKILL.md`). The skill dispatches them; a user never invokes one directly. Each one is meant to be a single, sharp lane.

## One file, two harnesses

`.claude/agents/review` and `.opencode/agents/review` link to this directory, so each lane file loads in both Claude Code and OpenCode.

- Name each file after its lane: `<name>.md`.
- Claude Code reads `name`, `description`, `tools` and `color`, and dispatches the lane by its `name`.
- OpenCode reads `description`, `mode: subagent` and `permissions`, and dispatches the lane as `review/<file stem>`. It ignores the Claude keys.
- Keep `mode: subagent` in every lane. OpenCode silently drops a file in Claude's format, one without an OpenCode-only key.
- Keep `permissions` in step with `tools`: a lane without `Edit` or `Write` denies `edit`.
- Never add a `model` key. A Claude model id makes OpenCode drop the file, and an OpenCode model id fails in Claude. The dispatcher passes the light-tier model on every dispatch ([Model tiers](../skills/pushing-commits-to-the-repo/SKILL.md#model-tiers)).

After a frontmatter change, check that both harnesses list the lane: `opencode debug agents --standalone` for OpenCode, and the session's agent list for Claude Code.

## The one principle: atomic, non-overlapping focus

One agent = one concern. More agents is fine; **overlapping agents are not**. The more atomically focused an agent is, the better it performs — a reviewer that owns exactly one question gives sharper findings than one juggling three. Two agents that both "sort of" check the same thing is the failure mode to avoid: findings get duplicated, and each does its shared part worse than a dedicated owner would.

## Before creating a new agent (mandatory)

**Read every agent file in this folder first.** No exceptions. You are looking for any existing agent whose instructions already cover — even partially — the concern you are about to add. This dedup pass is the gate; skipping it is how the suite rots into overlap.

A new lane also needs its dispatch precondition in `.agents/skills/review-branch/preconditions.py` and its row in the skill's reviewer table.

## When the new concern overlaps an existing agent

You have discretion to resolve it one of three ways:

1. **Remove it from the old agent, keep it in the new** — when the new agent is the more natural owner.
2. **Drop it from the new agent** — when the old agent already owns it well; defer to the old.
3. **Carve it out of both into a new, dedicated mini-agent** — only when the overlapping aspect is substantial enough to deserve its own lane.

Options 1 and 2 are routine — use judgment. **Option 3 requires proof, not intuition.**

## Proof bar for option 3 (carving out a new mini-agent)

Before splitting an aspect into its own agent, demonstrate from real runs that carrying it is hurting the agent(s) that share it. Read — or have a subagent read — the transcripts of previous `/review-branch` runs, including runs from removed worktrees. Establish at least one of:

- **Focus dilution** — the carrying agent's findings degraded because the extra aspect distracted it (e.g. it missed findings in its core lane on the runs where the shared aspect was in play, or its output visibly splits attention).
- **Token blowout** — that agent consistently burned substantially more tokens than its peers, indicating the bundled aspect does enough work to stand on its own.

Intuition that "this feels like its own thing" is not sufficient. If you cannot find the evidence, fold the aspect into one existing owner (option 1 or 2) instead.

## Updating an existing agent

Keep it in its lane. If an edit would add a *second* concern to an agent, treat that as the "new agent" case above: re-read the suite, check for a better owner, and do not quietly broaden an agent's scope — scope creep is overlap in slow motion.
