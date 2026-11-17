---
title: Report Tool Errors
description: "Send an exception a Pydantic AI tool raises back to the model as a failed tool result, so the run continues instead of ending."
---

# Report Tool Errors

Send an exception a tool raises back to the model as a failed tool result, so the run continues and the model can try another way. Without it, a tool like `divide(a, b)` called with `b=0` raises `ZeroDivisionError` out of `agent.run()`. Use it when an agent runs tools you don't control, and an unhandled edge case in one of them shouldn't end the run.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ReportToolErrors

agent = Agent('{{ anthropic:large }}', capabilities=[ReportToolErrors()])


@agent.tool_plain
def divide(a: int, b: int) -> float:
    return a / b
```

## What the model sees

The failed call gets a tool result with `outcome='failed'` that names the tool, the exception type and its message:

```text
`divide` raised ZeroDivisionError: division by zero
```

To hide internals such as file paths or hostnames, pass `format_error`, which builds the text from the failed [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] and the exception:

```python
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness import ReportToolErrors


def hide_internals(call: ToolCallPart, error: Exception) -> str:
    return f'`{call.tool_name}` failed with {type(error).__name__}.'


report = ReportToolErrors(format_error=hide_internals)
```

## What it leaves alone

- The tool stays available and nothing counts the failures: the model decides whether to call it again. [`UsageLimits`](../agent.md#usage-limits), such as `request_limit`, stop a model that keeps calling a failing tool.
- `ModelRetry`, `ToolFailed`, `ApprovalRequired`, `CallDeferred` and `SkipToolExecution` keep their usual meaning, since core doesn't pass them to `on_tool_execute_error`.
- A tool that raises `ModelRetry` more often than its `retries` allow still ends the run with `UnexpectedModelBehavior`, so a tool's retry limit keeps working. [Coder](coder.md) reports those to the model instead.
- Output tools and argument validation errors aren't affected.

To get the fail-fast behavior back, leave `ReportToolErrors` out of `capabilities`.

## Telemetry

`ReportToolErrors` emits no spans of its own: core's tool span already marks the call as an error and, when content is included, records the text the model sees as its result.

## Composition

- Core runs `on_tool_execute_error` hooks from the last capability in `capabilities` to the first, passing each the error the one before it raised. A capability listed after `ReportToolErrors` can handle an error first. One listed before it gets the `ToolFailed`.
- `ReportToolErrors` passes on a `ToolFailed`, `ModelRetry`, `ApprovalRequired`, `CallDeferred` or `SkipToolExecution` that an earlier hook raised, so with several on one agent, the last one listed formats the error.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_tool_errors/).

## API reference

::: pydantic_ai_harness.ReportToolErrors
