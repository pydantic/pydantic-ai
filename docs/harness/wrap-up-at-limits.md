---
title: Wrap Up At Limits
description: "End a Pydantic AI agent run with a final answer, instead of `UsageLimitExceeded`, when a usage limit would stop it."
---

# Wrap Up At Limits

End a run with a final answer when a [usage limit](../agent.md#usage-limits) would otherwise stop it with [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded]. Use it where the work done before the limit is worth returning, such as a research or coding agent running on a request, tool-call or token budget.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import WrapUpAtLimits

agent = Agent('{{ anthropic:large }}', capabilities=[WrapUpAtLimits()])


async def research(topic: str) -> str:
    result = await agent.run(topic, usage_limits=UsageLimits(request_limit=10, tool_calls_limit=30))
    return result.output
```

When the run's [`UsageLimits`][pydantic_ai.usage.UsageLimits] leave room for one more model request but not for another after it, `WrapUpAtLimits` makes that request the last:

- It sets `tool_choice='none'` and keeps the tool definitions, so the cached tools and instructions still match. Some providers still miss part of the cache: Anthropic caches messages per `tool_choice`, and [some cases](../tools-advanced.md#tool-choice-caching) send a trimmed tool list. With [tool output](../output.md#tool-output), `tool_choice='none'` sends only the output tools.
- It adds a request-scoped [instruction capability part](../capabilities/custom.md#capability-parts), a `WrapUpPart`, asking the model to answer now with what it has. The model's answer is the run's output.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | `DEFAULT_WRAP_UP_PROMPT` | What the model is told on the final request |

## When the final request is made

| Limit | The final request |
|---|---|
| `request_limit` | The last request the limit allows |
| `tool_calls_limit` | The first request after which the tool calls left are fewer than the run's largest batch so far, or none are left |
| `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit` | The first request after which a final request is projected to go over the limit |

- A batch of tool calls larger than any before it that would go over `tool_calls_limit` still raises `UsageLimitExceeded`, before any of its calls run. Answering such a batch as not executed and wrapping up after it needs core to stop counting denied tool results against the limit ([#8840](https://github.com/pydantic/pydantic-ai/pull/8840)); the request stays open in [#11846](https://github.com/pydantic/pydantic-ai/issues/11846).
- Token use is projected from this run's responses. Input and total tokens are assumed to grow on each further request by as much as the last response grew over the one before, plus the final prompt; output tokens to be at most the largest response's. A final request that uses more than projected still raises `UsageLimitExceeded`.
- `cost_limit` and `per_request_input_tokens_limit` aren't projected, and raise as before.
- If the final answer fails output validation, no request is left for the retry and the run raises `UsageLimitExceeded`.
- A run given no `usage_limits` gets the default [`UsageLimits`](../agent.md#usage-limits), whose `request_limit` is 50, so it wraps up on its 50th request instead of raising.
- A model that still calls tools on the final request, as a provider that doesn't enforce `tool_choice='none'` allows, has them run as usual, unless `tool_calls_limit` is used up, in which case they raise `UsageLimitExceeded`. After calls that ran, the next request is made the last again, or raises when no request is left.

## Telling a wrapped-up answer from a finished one

The final prompt is a request-scoped [`WrapUpPart`][pydantic_ai_harness.wrap_up_at_limits.WrapUpPart], an [instruction capability part][pydantic_ai.messages.InstructionCapabilityPart] whose `limit` is the [`LimitName`][pydantic_ai_harness.wrap_up_at_limits.LimitName] that triggered it. `wrapped_up_by()` reads it back:

```python
from pydantic_ai import AgentRunResult
from pydantic_ai_harness import wrapped_up_by


def report(result: AgentRunResult[str]) -> str:
    if limit := wrapped_up_by(result.new_messages()):
        return f'{result.output}\n\n(cut short by `{limit}`)'
    return result.output
```

## Durable execution

`WrapUpAtLimits` keeps no state of its own: it decides from the run's usage and message history. Under [durable execution](../durable_execution/overview.md) its hooks run in workflow code, so when Temporal, DBOS or Prefect replays a run, they make the same request the last.

## Telemetry

`WrapUpAtLimits` marks its decision with a zero-duration span on an [instrumented](../logfire.md) agent:

| Span | Attribute | Meaning |
|---|---|---|
| `run wrapped up` | `wrap_up_at_limits.limit` | The `LimitName` that made the request the last |

The span carries no prompt text. The final request itself shows as an ordinary model request.

## Composition

- [`WarnNearLimits`](compaction.md#warnnearlimits-warn-instead-of-rewrite) warns the model as limits approach. `WrapUpAtLimits` acts on the last request. Use both to have the model wind down first.
- It decides on the usage when its hook runs. List it after capabilities that spend from the run's usage before a request, such as `SummarizingCompaction`, so it sees what they spent.
- [Compaction](compaction.md) still runs on the final request. A strategy that makes a model request of its own, such as `SummarizingCompaction`, has no request left there and raises `UsageLimitExceeded`; wrap it in `FallbackCompaction(fallback_on=(UnexpectedModelBehavior, UsageLimitExceeded))` with a strategy that makes no model requests. Skipping compaction on the final request needs a core ordering hook ([#11454](https://github.com/pydantic/pydantic-ai/pull/11454)).
- In [Subagents](subagents.md), give a sub-agent its own `SubAgent.usage_limits` and add `WrapUpAtLimits` to its capabilities: it returns a wrapped-up answer to the parent instead of reaching its budget. A sub-agent that shares the parent's budget (no `usage_limits` of its own) would spend the parent's last request on its answer, so the parent then raises.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/wrap_up_at_limits/).

## API reference

::: pydantic_ai_harness.WrapUpAtLimits

::: pydantic_ai_harness.wrap_up_at_limits.wrapped_up_by

::: pydantic_ai_harness.wrap_up_at_limits.WrapUpPart

::: pydantic_ai_harness.wrap_up_at_limits.LimitName

::: pydantic_ai_harness.wrap_up_at_limits.DEFAULT_WRAP_UP_PROMPT
