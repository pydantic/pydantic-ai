# PR Decisions Log

Append-only record of decisions made during this PR's lifecycle that weren't obvious from the linked issue.

## Entry format

```
## YYYY-MM-DD · <short title> · iter <N or "-">
- Decision: <one line>
- Why: <one line>
- Source: <link to comment/thread/issue/commit — mandatory>
- Session: <id of the harness session that made the decision — optional>
- Supersedes: <earlier entry title, if applicable>
```

No prose. No multi-paragraph rationale. The Source link carries the full context.

**One exception — modality.** When the Decision restates a modal claim (always / never / only if / must / unless), quote that clause verbatim instead of paraphrasing it. Brevity everywhere else, but "always X unless Y" compressed to "always X" is a different instruction, and the next agent has no way to notice the qualifier went missing.

**Session.** Pass `--session` to `append-pr-decision.sh` with your harness session id. A later agent uses it to find the session that made the decision. Claude Code sets the id in `$CLAUDE_CODE_SESSION_ID`. Codex sets it in `$CODEX_THREAD_ID`. The field is best-effort: when neither is set, or the script rejects the id, append the entry without it.

## When to append

- You made a pick between two valid implementations and the reviewer later asked about it
- A thread resolution chose path A over B
- A maintainer answered an escalation and that answer settled an ambiguity
- You deviated from the plan while coding (`.claude/plans/<branch>.md`, or `PLAN.md` on a plan-only PR)
- A CI finding forced a design change

## When NOT to append

- Routine implementation work already captured by the diff
- Decisions already spelled out in the issue or in `issue-brief.md`
- Research notes (those go in `local-notes/`)

---

<!-- entries below, newest at bottom -->
