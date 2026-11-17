---
title: Tool Budgets
description: "Cap how many times a Pydantic AI agent can call each tool, or any tool, in a run, and refuse exact repeats, with a tool result the model can act on instead of an exception."
---

# Tool Budgets

Cap how many times the model can call each tool, or any tool, in one run, and refuse a call that repeats an earlier one exactly. An over-budget or repeated call is answered with a tool result that tells the model what it hit, and the run continues. Use it to keep an agent from spending a run on one expensive tool, such as web search.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ToolBudgets

budgets = ToolBudgets(per_tool={'web_search': 3}, total=20, refuse_repeats={'web_search'})
agent = Agent('{{ anthropic:large }}', capabilities=[budgets])
```

| Field | Refuses a call when |
|---|---|
| `per_tool` | The named tool already ran this many times in the run. |
| `total` | This many tool calls, across all tools, already ran in the run. |
| `refuse_repeats` | The named tool already ran in the run with the same arguments. |

A budget must be at least 1. [`UsageLimits.tool_calls_limit`](../agent.md#usage-limits) stays the hard ceiling: it raises `UsageLimitExceeded` and ends the run. A refused call is still a call the model made, so it counts toward that limit.

## What the model sees

A refused call doesn't run. Its tool result names the budget and how much of it is used:

```text
3 of 3 `web_search` calls used; no more are allowed in this run.
20 of 20 tool calls used; no more are allowed in this run.
`web_search` already ran with these exact arguments in this run (tool call `call_abc`); its result, or a summary of it, is earlier in this conversation. Don't call it again with the same arguments.
```

The refusal is a [`ToolReturn`][pydantic_ai.messages.ToolReturn] whose `metadata`, which the model doesn't see, is `{'origin': 'pydantic-ai-harness.tool_budgets.refusal.v1', 'reason': reason}`, with `reason` one of `'per_tool'`, `'total'` or `'repeat'` (`ToolBudgetReason`). Find refusals in history by that `origin`, not by their text.

## How calls are counted

- A call counts once its execution starts, whether the tool then returns, raises [`ModelRetry`](../tools-advanced.md#tool-retries), fails, or raises `CallDeferred`, since a tool that defers may already have started its work. A call that another capability skips or holds for approval, such as a [Tool Call Judge](tool-call-judge.md) block or an approval request, doesn't count. A refused call doesn't count either.
- Repeats are opt-in per tool, because a polling tool repeats its call on purpose. Two calls are the same when the tool name matches and the arguments, as the model sent them before defaults are filled in, are equal as JSON with sorted keys. A call that failed isn't remembered, so it can be made again.
- Parallel calls in one response are counted in the order they start, so they can't overshoot a budget, and an identical parallel call is refused as a repeat. A call is checked against calls that started but haven't finished, so if another capability then skips one of those or holds it for approval, a parallel call refused because of it still doesn't run.
- Every [function tool](../tools.md) call counts, including MCP tools and tools other capabilities add. Calls the provider runs itself ([native tools](../native-tools.md)) and output tools don't. `per_tool` and `refuse_repeats` use the tool's name as the model sees it, after any prefix.
- Counts live in the instance `for_run` creates for each run, not in message history, so [compaction](compaction.md) that summarizes earlier calls away doesn't hand a budget back, and a repeat stays refused after its earlier result has been summarized.
- A run that continues an earlier conversation through `message_history` starts at zero. So does a run resumed after a [tool approval](../deferred-tools.md#human-in-the-loop-tool-approval) or other deferred call resolved outside the run, rather than by a [handler](../deferred-tools.md#resolving-deferred-calls-with-a-handler) inside it: with approval rounds, `per_tool={'web_search': 3}` allows three calls per round, not per task.

## Durable execution

The checks run in `wrap_tool_execute`, in workflow code, under [durable execution](../durable_execution/overview.md). When Temporal, DBOS, or Prefect replays a run, they run again in the same order over the journaled responses and reach the same refusals, so a replay doesn't double-count.

## Telemetry

Each refusal emits a zero-duration `tool call refused` span through `ctx.tracer`, with `tool_budgets.reason` and `tool_budgets.tool_name`. It's a no-op unless the agent is [instrumented](../logfire.md). Calls that run emit nothing beyond core's tool span.

## Composition

- Several `ToolBudgets` on one agent each count the calls they let run, so every budget applies.
- [Detect Loops](detect-loops.md) notices repetition after the fact and nudges or stops the run; `ToolBudgets` refuses an exact repeat before it runs, for the tools you list.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_budgets/).

## API reference

::: pydantic_ai_harness.ToolBudgets

::: pydantic_ai_harness.tool_budgets.ToolBudgetReason
