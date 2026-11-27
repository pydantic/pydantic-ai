# Report Tool Errors

Send an exception a tool raises back to the model as a failed tool result, so the run continues and the model can try another way. Without it, a tool like `divide(a, b)` called with `b=0` raises `ZeroDivisionError` out of `agent.run()`. Use it when an agent runs tools you don't control, and an unhandled edge case in one of them shouldn't end the run.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_tool_errors/)

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ReportToolErrors

agent = Agent('anthropic:claude-opus-5-5', capabilities=[ReportToolErrors()])


@agent.tool_plain
def divide(a: int, b: int) -> float:
    return a / b
```

## What the model sees

The failed call gets a tool result with `outcome='failed'` that names the tool and the exception type:

```text
`divide` raised ZeroDivisionError
```

The exception's message isn't included, since it can carry file paths, hostnames or connection strings. When your tools' messages are meant for the model, pass `format_error`, which builds the text from the run context and a `ToolErrorInfo` carrying the tool name, its arguments, the tool call ID and the exception:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import ReportToolErrors
from pydantic_ai_harness.report_tool_errors import ToolErrorInfo


def with_message(ctx: RunContext[None], info: ToolErrorInfo) -> str:
    return f'`{info.name}` raised {type(info.error).__name__}: {info.error}'


report = ReportToolErrors(format_error=with_message)
```

To report only some exceptions, pass `exceptions`. Any other exception ends the run as it would without the capability, so a programming error such as a `KeyError` still fails fast:

```python
from pydantic_ai_harness import ReportToolErrors

report = ReportToolErrors(exceptions=(TimeoutError, ConnectionError))
```

## How it works

`ReportToolErrors` wraps the agent's toolset and turns an exception from a tool into a [`ToolFailed`](https://pydantic.dev/docs/ai/api/pydantic-ai/exceptions/#pydantic_ai.exceptions.ToolFailed), as if the tool had raised it, so the rest of the agent treats the call like any other failed call:

- The tool stays available and nothing counts the failures: the model decides whether to call it again. [`UsageLimits.request_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) stops a model that keeps calling a failing tool; `tool_calls_limit` counts only successful calls. To cap calls to one tool, use [Tool Budgets](../tool_budgets/).
- `ModelRetry`, `ToolFailed`, `ApprovalRequired`, `CallDeferred` and `SkipToolExecution` pass through unchanged, and so does `RunCancelled` from a sub-agent, which core already reports. `UsageLimitExceeded` and `UserError` still end the run: a spent budget or a setup bug is not something the model can work around.
- A tool that raises `ModelRetry` more often than its `retries` allow still ends the run with `UnexpectedModelBehavior`, so a tool's retry limit keeps working. [Coder](../coder/) reports those to the model instead.
- Every function tool is covered, including MCP tools and tools other capabilities add. Output tools and argument validation errors aren't.

To get the fail-fast behavior back, leave `ReportToolErrors` out of `capabilities`.

## Durable execution

Under [Temporal](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/), a tool runs in an activity, and an exception it raises goes to the activity's retry policy. When that gives up, the workflow gets Temporal's `ActivityError`, which names the tool's exception only by its type, so `ReportToolErrors` leaves it alone and the run fails as it would without the capability.

## Telemetry

`ReportToolErrors` emits no spans of its own: core's tool span already marks the call as an error and, when content is included, records the text the model sees as its result.

## Composition

- A [Tool Guardrails](../guardrails/) `result_guard` screens the reported text like any failed tool result, so it can redact what a custom `format_error` includes.
- `on_tool_execute_error` hooks aren't called for an exception `ReportToolErrors` reports, whatever order the capabilities are listed in, so a capability that logs or recovers in that hook no longer sees it. Narrow `exceptions` to leave those types to the hook.
- Under [Code Mode](../code_mode/), a tool that a snippet calls raises the reported `ToolFailed` into the snippet instead of its own exception, so `except ZeroDivisionError:` in the snippet no longer catches it. Narrow `exceptions` to leave such types to the snippet.
- Several `ReportToolErrors` on one agent: the last one listed wraps the tools most closely, so it formats the error.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_tool_errors/).
