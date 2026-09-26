# PR Decisions Log

Append-only record of decisions made during this PR's lifecycle that weren't obvious from the linked issue.

## Entry format

```
## YYYY-MM-DD · <short title> · iter <N or "-">
- Decision: <one line>
- Why: <one line>
- Source: <link to comment/thread/issue/commit -- mandatory>
- Supersedes: <earlier entry title, if applicable>
```

No prose. No multi-paragraph rationale. The Source link carries the full context.

**One exception -- modality.** When the Decision restates a modal claim (always / never / only if / must / unless), quote that clause verbatim instead of paraphrasing it. Brevity everywhere else, but "always X unless Y" compressed to "always X" is a different instruction, and the next agent has no way to notice the qualifier went missing.

## When to append

- You made a pick between two valid implementations and the reviewer later asked about it
- A thread resolution chose path A over B
- An ambiguity was settled during the work
- You deviated from the plan while coding
- A CI finding forced a design change

## When NOT to append

- Routine implementation work already captured by the diff
- Decisions already spelled out in the issue or in `issue-brief.md`
- Research notes (those belong outside this directory)

---

<!-- entries below, newest at bottom -->
