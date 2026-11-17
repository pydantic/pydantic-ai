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

To hide internals such as file paths or hostnames, pass `format_error`, which builds the text from the run context and a `ToolErrorInfo` carrying the tool name, its arguments, the tool call ID and the exception:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import ReportToolErrors
from pydantic_ai_harness.report_tool_errors import ToolErrorInfo


def hide_internals(ctx: RunContext[None], info: ToolErrorInfo) -> str:
    return f'`{info.name}` failed with {type(info.error).__name__}.'


report = ReportToolErrors(format_error=hide_internals)
```

To report only some exceptions, pass `exceptions`. Any other exception ends the run as it would without the capability, so a programming error such as a `KeyError` still fails fast:

```python
from pydantic_ai_harness import ReportToolErrors

report = ReportToolErrors(exceptions=(TimeoutError, ConnectionError))
```

## How it works

`ReportToolErrors` wraps the agent's toolset and turns an exception from a tool into a [`ToolFailed`][pydantic_ai.exceptions.ToolFailed], as if the tool had raised it, so the rest of the agent treats the call like any other failed call:

- The tool stays available and nothing counts the failures: the model decides whether to call it again. [`UsageLimits`](../agent.md#usage-limits), such as `request_limit`, stop a model that keeps calling a failing tool.
- `ModelRetry`, `ToolFailed`, `ApprovalRequired`, `CallDeferred` and `SkipToolExecution` pass through unchanged, and so does `RunCancelled` from a sub-agent, which core already reports. `UsageLimitExceeded` and `UserError` still end the run: a spent budget or a setup bug is not something the model can work around.
- A tool that raises `ModelRetry` more often than its `retries` allow still ends the run with `UnexpectedModelBehavior`, so a tool's retry limit keeps working. [Coder](coder.md) reports those to the model instead.
- Every function tool is covered, including MCP tools and tools other capabilities add. Output tools and argument validation errors aren't.

To get the fail-fast behavior back, leave `ReportToolErrors` out of `capabilities`.

## Telemetry

`ReportToolErrors` emits no spans of its own: core's tool span already marks the call as an error and, when content is included, records the text the model sees as its result.

## Composition

- A [Tool Guardrails](guardrails.md) `result_guard` screens the reported text like any failed tool result, so it can redact what the exception message reveals.
- Capabilities' `on_tool_execute_error` hooks don't see the exception: they see a failed call, as for a tool that raised `ToolFailed`.
- Several `ReportToolErrors` on one agent: the last one listed wraps the tools most closely, so it formats the error.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_tool_errors/).

## API reference

::: pydantic_ai_harness.ReportToolErrors

::: pydantic_ai_harness.report_tool_errors.ToolErrorInfo
