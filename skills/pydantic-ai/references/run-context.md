# RunContext Reference

Source: `pydantic_ai_slim/pydantic_ai/_run_context.py`

## Overview

`RunContext[DepsT]` provides contextual information about the current agent run to tools, system prompts, and validators. It's the primary mechanism for dependency injection and accessing run state.

```python
from pydantic_ai import Agent, RunContext

@agent.tool
async def my_tool(ctx: RunContext[MyDeps], query: str) -> str:
    # Access dependencies, usage, messages, retry info, etc.
    return f"User {ctx.deps.user_id}: {query}"
```

## Core Fields

### `ctx.deps` — Dependencies

Your custom dependencies object passed via `run(deps=...)`. Type matches `deps_type` on the agent.

```python
ctx.deps              # Your dependencies object (AgentDepsT)
ctx.deps.db           # Access fields on your deps class
ctx.deps.api_client   # E.g., httpx.AsyncClient
```

### `ctx.model` — Model Instance

The `Model` instance being used for this run.

```python
ctx.model             # Model instance (e.g., OpenAIModel)
ctx.model.model_name  # Get the model name string
```

### `ctx.usage` — Token Usage

`RunUsage` with cumulative token counts for this run.

```python
ctx.usage.input_tokens    # Total input tokens so far
ctx.usage.output_tokens   # Total output tokens so far
ctx.usage.requests        # Number of model requests made
ctx.usage.total_tokens    # input_tokens + output_tokens
```

### `ctx.prompt` — Original User Prompt

The original prompt passed to `run()` or `run_sync()`.

```python
ctx.prompt            # str | Sequence[UserContent] | None
```

### `ctx.messages` — Conversation History

List of all messages exchanged in the conversation so far.

```python
ctx.messages          # list[ModelMessage]
len(ctx.messages)     # Number of messages
ctx.messages[-1]      # Most recent message
```

### `ctx.run_id` — Run Identifier

Unique identifier for this agent run.

```python
ctx.run_id            # str | None
```

### `ctx.metadata` — Run Metadata

Metadata dict associated with this run (if configured via `run(metadata=...)`).

```python
ctx.metadata          # dict[str, Any] | None
```

## Tool-Specific Fields

### `ctx.tool_name` — Current Tool Name

Name of the tool being called (only set during tool execution).

```python
ctx.tool_name         # str | None
```

### `ctx.tool_call_id` — Tool Call ID

The ID of the current tool call from the model.

```python
ctx.tool_call_id      # str | None
```

### `ctx.retry` — Current Retry Count

Number of retries so far for this specific operation.

- **For tool calls:** Number of retries of this specific tool
- **For output validation:** Number of output validation retries

```python
ctx.retry             # int (0 on first attempt)
```

### `ctx.max_retries` — Maximum Retries

Maximum retries allowed for this operation.

- **For tool calls:** Maximum retries configured for the specific tool
- **For output validation:** Maximum output validation retries

```python
ctx.max_retries       # int
```

### `ctx.retries` — Retry Counts by Tool

Dict mapping tool names to their retry counts.

```python
ctx.retries           # dict[str, int]
ctx.retries.get('my_tool', 0)  # Retries for a specific tool
```

### `ctx.tool_call_approved` — Approval Status

Whether a deferred tool call has been approved (for human-in-the-loop workflows).

```python
ctx.tool_call_approved    # bool (True if approved)
```

### `ctx.tool_call_metadata` — Deferred Tool Metadata

Metadata from `DeferredToolResults.metadata[tool_call_id]`, available when `tool_call_approved=True`.

```python
ctx.tool_call_metadata    # Any
```

### `ctx.run_step` — Current Step Number

The current step number in the agent run loop.

```python
ctx.run_step          # int (0-indexed)
```

## Validation Fields

### `ctx.validation_context` — Pydantic Context

Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) for tool args and outputs.

```python
ctx.validation_context    # Any
```

### `ctx.partial_output` — Streaming Partial Flag

Whether the output passed to an output validator is partial (during streaming).

```python
ctx.partial_output    # bool
```

## Tracing/Instrumentation Fields

### `ctx.tracer` — OpenTelemetry Tracer

The OpenTelemetry tracer for creating spans within the run.

```python
ctx.tracer            # opentelemetry.trace.Tracer
```

### `ctx.trace_include_content` — Content Tracing Flag

Whether to include message content in traces.

```python
ctx.trace_include_content    # bool
```

### `ctx.instrumentation_version` — Instrumentation Version

Instrumentation settings version, if instrumentation is enabled.

```python
ctx.instrumentation_version  # int
```

## Properties

### `ctx.last_attempt` — Final Retry Check

Returns `True` if this is the last retry attempt before an error is raised.

```python
ctx.last_attempt      # bool (True if retry == max_retries)

# Common pattern: only log on final attempt
if ctx.last_attempt:
    logger.error(f"Tool {ctx.tool_name} failed after {ctx.retry} retries")
```

## Practical Examples

### Accessing Dependencies in Tools

```python {title="run_context_deps.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    user_id: int
    api_key: str


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def get_user_data(ctx: RunContext[Deps]) -> str:
    """Fetch user data using dependencies."""
    return f'Data for user {ctx.deps.user_id}'


result = agent.run_sync('Get my data', deps=Deps(user_id=42, api_key='key'))
print(result.output)
#> Here is your data: Data for user 42
```

### Using Retry Information

```python {title="run_context_retry.py"}
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-5')


@agent.tool(retries=3)
async def flaky_api(ctx: RunContext[None], query: str) -> str:
    """Call an API that may need retries."""
    if ctx.retry < 2:
        raise ModelRetry(f'Attempt {ctx.retry + 1} failed, retrying...')
    return f'Success on attempt {ctx.retry + 1}'


result = agent.run_sync('Call the API')
print(result.output)
#> The API returned: Success on attempt 3
```

### Checking Last Attempt

```python {title="run_context_last.py"}
import logging

from pydantic_ai import Agent, ModelRetry, RunContext

logger = logging.getLogger(__name__)

agent = Agent('openai:gpt-5')


@agent.tool(retries=2)
async def important_action(ctx: RunContext[None]) -> str:
    """Action with logging on final attempt."""
    if ctx.last_attempt:
        logger.warning('Final attempt for important_action')
    # ... perform action
    return 'done'


result = agent.run_sync('Do the important action')
print(result.output)
#> The action is complete: done
```

## Key Type

| Type | Import | Description |
|------|--------|-------------|
| `RunContext` | `pydantic_ai.RunContext` | Context passed to tools, system prompts, validators |

## See Also

- [dependencies.md](dependencies.md) — Dependency injection patterns
- [tools.md](tools.md) — Tool registration and usage
- [exceptions.md](exceptions.md) — ModelRetry and error handling
