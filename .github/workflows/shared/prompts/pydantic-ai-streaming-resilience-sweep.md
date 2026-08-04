<!--
Default/seed prompt for the Pydantic AI Streaming Resilience Sweep agent.

This file is the COMPLETE prompt. It is the verbatim fallback when the
Logfire managed variable
`gh_aw_pydantic_ai_streaming_resilience_sweep_prompt` is unset or
unreachable. To iterate on the live prompt, edit that Logfire variable (start
from this file's content below the comment); no recompile or commit is
needed. Keep this file in sync as the reviewed default.
-->

# Pydantic AI Streaming Resilience Sweep

The streaming/agent-loop code is in `pydantic_ai_slim/pydantic_ai/` (`_agent_graph`, `result`, `messages`,
`run`) and the AG-UI / Vercel adapters.

## Objective

Find one concrete bug in the **streaming state machine**. Streaming is the
largest topical cluster and harbors hard-to-spot ordering and lifecycle bugs.
Pick **one** focus area per run:

- `run_stream` / `StreamedRunResult` lifecycle (early exit, double-consume,
  `get_output()` before/after completion).
- `agent.iter` / `_next_node` graph node ordering and assertions.
- `event_stream_handler` event sequence (start → deltas → end), including for
  tool calls and thinking parts.
- Partial / aborted / cancelled streams (network drop, `break`, timeout) and
  cleanup.
- AG-UI / Vercel adapter event ordering and terminal events.
- Usage / final message assembly when the stream ends or errors mid-way.

## How to Verify — mandatory

Use `TestModel`/`FunctionModel` or a recorded fixture to drive a deterministic
stream. Write a **new** minimal test asserting the **event/None sequence and
final message** — e.g. no deltas after the final part, tool-call/return
pairing intact, `usage` populated on completion, no `StopAsyncIteration`/
assertion leakage on early exit. Do not run and report the existing suite.

## What to Look For

- Events emitted out of order, duplicated, or missing a terminal event.
- State leaking between consumption attempts; `get_output()` returning stale
  or partial data.
- Exceptions/asserts surfacing to the user on normal early termination.
- Final `ModelResponse`/usage missing parts that were streamed.
- Cancellation not cleaning up the underlying provider stream.

## What to Skip

- Provider-specific delta mapping bugs (→ provider mapping sweep).
- Speculation without a deterministic failing reproduction.
- Behavior already tracked by an open issue — **search issues first**.

## Deduplication — mandatory BEFORE filing an issue

Open issues were prefetched before the sandbox started. First narrow the local
corpus to streaming-labelled issues. This covers both prior
`[streaming-resilience-sweep]` findings and human-filed streaming issues:

```
jq '.[] | select(any(.labels[]; .name == "streaming")) | {number, title, url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Only if that is inconclusive, widen to a full open-issue scan and grep locally
for "stream_output" / "stream_text":

```
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Do not enumerate issues with `gh` from inside the sandbox; list requests can
stall until the workflow times out.

If a matching issue exists, call `mcp__safeoutputs__noop` immediately.

## Sandbox notes

- Use `FunctionModel` with a simple stream function for reproductions — avoid
  complex model setups.

## Quality Gate — When to Noop

`mcp__safeoutputs__noop` is the expected outcome most runs. Only file with a deterministic,
minimal, failing streaming reproduction and captured event trace.

## Issue Format

**Title:** `Streaming: <short bug summary>`

**Body:**

> ## Impact
> [Who is affected — streaming users, AG-UI clients, `agent.iter` users]
>
> ## Focus Area & Code Path
> [Which streaming surface; `file:line`]
>
> ## Reproduction
> [The new streaming test — full code — and the command]
>
> ## Expected vs Actual
> **Expected event/result sequence:** … **Actual:** … [captured trace]
>
> ## Evidence
> - [Captured event trace / output; `path:line` references]
>
> ## Adversarial review
> - **Reproduced on `main`:** [exact command + real captured trace — confirm the asymmetry/failure actually exists, not a false premise]
> - **Existing tests checked:** [streaming tests read; none assert the current behavior, and the fix doesn't break them]
> - **Ruled out by-design:** [sibling streaming methods behave the same / nearby comment / maintainer decision checked]
> - **Not a duplicate:** [label-filtered dedup returned nothing]
