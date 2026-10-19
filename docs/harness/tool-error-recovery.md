---
title: Tool Error Recovery
description: Decide in one place what happens when a tool call raises. Retry it in place, report it to the model as a failed result, return a substitute value, or let it propagate, with per-run budgets and logging.
---

# Tool Error Recovery

Decide in one place what happens when a tool call raises: retry it in place, report it to the model as a failed result, return a substitute value, or let it propagate.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_error_recovery/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

An unhandled exception in a tool ends the agent run, and the user gets an error instead of an answer. Tools fail for reasons that need different responses: a timeout from an upstream service is worth one more attempt, a missing record is something the model should hear about and work around, and a `KeyError` is a bug that should surface. Handling this in every tool is repetitive, and tools from an MCP server or a third-party toolset cannot be changed at all.

## The solution

`ToolErrorRecovery` wraps every tool call on the agent. When a call raises, a classifier returns one of four outcomes:

| Outcome | Effect |
|---|---|
| `retry` | Call the tool again with the same arguments, up to `max_attempts` times, with no model round trip in between |
| `inform` | Report a terminal failure to the model by raising [`ToolFailed`][pydantic_ai.exceptions.ToolFailed]; the tool result carries `outcome='failed'` |
| `fallback` | Return a substitute value as the tool result |
| `propagate` | Re-raise the exception |

Install `pydantic-ai-harness`; no extra is required.

```bash
pip/uv-add pydantic-ai-harness
```

```python {names="defined"}
import httpx

from pydantic_ai import Agent
from pydantic_ai_harness import ToolErrorRecovery

agent = Agent('{{ openai:medium }}', capabilities=[ToolErrorRecovery()])


@agent.tool_plain
async def check_stock(sku: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f'https://inventory.example.com/stock/{sku}')
        response.raise_for_status()
        return response.text
```

With the default policy, a connect timeout from the inventory service is retried in place, up to three attempts. A `404` raises `HTTPStatusError`, which reaches the model as a failed tool call it can tell the user about. A `KeyError` in your own code still ends the run. Every recovery is logged.

## Classifying failures

The default policy retries `TimeoutError`, `ConnectionError`, and the httpx connect and timeout errors, propagates the common programming-error types, and reports everything else to the model. Pass your own `classify` to decide by tool name, exception type, or both. It may take a leading [`RunContext`][pydantic_ai.tools.RunContext], may be async, and runs again on every attempt, so a different error on a later attempt is judged on its own.

```python {names="defined"}
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.tool_error_recovery import (
    RecoveryOutcome,
    RecoveryPolicy,
    ToolErrorRecovery,
    default_classify,
)


class OutOfStock(LookupError):
    pass


def classify(ctx: RunContext[Any], call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
    if isinstance(error, OutOfStock):
        return RecoveryOutcome.inform(label='out of stock', expose_message=True)
    if call.tool_name == 'charge_card':
        return RecoveryOutcome.propagate()
    return default_classify(ctx, call, error)


recovery = ToolErrorRecovery(RecoveryPolicy(classify=classify))
```

`OutOfStock` is raised by your own tools, so its message is written for the model and `expose_message=True` passes it on. Failures of `charge_card` propagate whatever the exception. Everything else keeps the default behavior.

By default the text the model sees names the exception type and omits `str(error)`, because exception messages can carry secrets. `label`, `expose_message`, and a custom `format_error` on the policy opt out of that.

## Retrying in place

`retry` repeats the call with the same validated arguments and no model round trip, so it is only safe for tools whose effect can be repeated; a failure after a partial side effect would apply the effect again. The default policy therefore retries only errors raised before a request was sent.

`retry` is not [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]. The model is not asked for new arguments, and the core [per-tool retry budget](../tools-advanced.md#tool-retries) is not consumed. To have the model choose new arguments, raise `ModelRetry` [from the tool or a hook](../hooks.md#triggering-retries-with-modelretry).

## Falling back to a value

`fallback` returns a substitute as if the tool had returned it: same type, successful outcome, so code and output schemas that depend on the tool's result keep working. The classifier sees the call, so the substitute can depend on the arguments. Here a rate service that stays unreachable after the retries answers from the last known value:

```python {names="defined"}
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.tool_error_recovery import RecoveryOutcome, default_classify

last_known_rates: dict[str, float] = {}


def classify(ctx: RunContext[Any], call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
    if call.tool_name == 'exchange_rate' and isinstance(error, ConnectionError):
        rate = last_known_rates.get(call.args_as_dict()['currency'])
        if rate is not None:
            stale = RecoveryOutcome.fallback({'rate': rate, 'stale': True}, label='stale rate')
            return RecoveryOutcome.retry(3, on_exhausted=stale)
    return default_classify(ctx, call, error)
```

The model is not told that a fallback happened; `label` only reaches the log. Put the fact into the value, as `stale` does here, when the model should treat the result differently. Use `inform` instead when the model should know that the call failed.

## Budgets and observability

`max_recoveries` caps the failures smoothed over per run across all tools, and `per_tool_recoveries` caps them per tool name. Once a cap is reached the error propagates, so a service that is down does not produce an unbounded series of recovered calls. The counters count final recoveries, not attempts, and reset per run.

Every recovery is logged on the `pydantic_ai_harness.tool_error_recovery` logger: INFO per retry, WARNING per `inform` or `fallback`, ERROR when a failure propagates, each record carrying `recovery_tool`, `recovery_action`, `recovery_error`, `recovery_label`, `recovery_attempt`, and `recovery_max_attempts`. The capability emits no spans of its own; core already spans the tool call, and where the log records go is the application's choice.

## What it does not do

It does not turn bugs into tool failures: the default policy propagates `KeyError`, `TypeError`, and the other programming-error types, and a classifier that reports them to the model instead hides defects. It does not touch `ModelRetry`, `ToolFailed`, `CallDeferred`, `ApprovalRequired`, or `SkipToolExecution` raised by a tool; those pass through unclassified. It does not make a tool idempotent; that stays the tool's job, which is why `retry` is limited to errors raised before a request was sent.

## Composition

Capabilities wrap in list order. List `ToolErrorRecovery` before a capability that puts a deadline on tool execution, or the deadline wraps the recovery and its [`HookTimeoutError`][pydantic_ai.capabilities.HookTimeoutError] ends the run before the classifier sees it. Capabilities listed after it run their tool hooks once per attempt. Two instances nest, each with its own budget: an inner `inform` or `fallback` is final, and an inner `propagate` reaches the outer one.

## Durable execution

The hook is not a durable operation, so no `id` is needed under Temporal, DBOS, or Prefect. It runs in orchestration context around the tool call the engine checkpoints, and the budget counters live on the run's copy of the capability.

## Not spec-serializable

`ToolErrorRecovery.get_serialization_name()` returns `None`: `classify` and `format_error` are callables, which cannot be serialized to an [agent spec](../agent-spec.md).

## API reference

::: pydantic_ai_harness.tool_error_recovery.ToolErrorRecovery

::: pydantic_ai_harness.tool_error_recovery.RecoveryPolicy

::: pydantic_ai_harness.tool_error_recovery.RecoveryOutcome
