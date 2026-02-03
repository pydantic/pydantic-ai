# Extensibility Patterns

Source: `pydantic_ai_slim/pydantic_ai/toolsets/wrapper.py`, `pydantic_ai_slim/pydantic_ai/models/wrapper.py`

## Philosophy

PydanticAI is intentionally lightweight. Before requesting a framework feature, consider building on these extension points. The framework provides strong primitives that enable arbitrarily complex behaviors without core changes.

**The pattern:** Wrap, don't fork. Compose, don't modify.

## Decision Tree

| I need... | Try this pattern |
|-----------|------------------|
| Logging/caching/auth for tool calls | `WrapperToolset` — override `call_tool()` |
| Retry logic/cost tracking for model requests | `WrapperModel` — override `request()` |
| Conditional tool availability | `FilteredToolset` or `ToolPrepareFunc` |
| Per-run state or behavior | `RunContext` + `ctx.deps` |
| Custom output processing | Output validators or `TextOutput` |
| Agent-level middleware | `WrapperAgent` — override `iter()` |

## WrapperToolset Pattern

Subclass `WrapperToolset` to intercept and modify tool execution. Override `call_tool()` for execution-time behavior, or `get_tools()` for tool availability.

**Source:** `pydantic_ai_slim/pydantic_ai/toolsets/wrapper.py`

```python
from typing import Any

from pydantic_ai import RunContext, WrapperToolset
from pydantic_ai.toolsets import ToolsetTool


class LoggingToolset(WrapperToolset):
    """Log all tool calls with timing."""

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool
    ) -> Any:
        import time
        start = time.perf_counter()
        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
            elapsed = time.perf_counter() - start
            print(f'Tool {name} completed in {elapsed:.2f}s')
            return result
        except Exception as e:
            print(f'Tool {name} failed: {e}')
            raise


# Usage: wrap any toolset
logging_toolset = LoggingToolset(my_toolset)
agent = Agent('openai:gpt-5', toolsets=[logging_toolset])
```

### What to Override

| Method | Purpose |
|--------|---------|
| `call_tool()` | Intercept tool execution (logging, caching, auth, validation) |
| `get_tools()` | Modify available tools at runtime |
| `__aenter__`/`__aexit__` | Resource management (connections, sessions) |

### Real Examples in Codebase

- **`ApprovalRequiredToolset`** — Raises `ApprovalRequired` before tool execution
  - Source: `pydantic_ai_slim/pydantic_ai/toolsets/approval_required.py`
- **`FilteredToolset`** — Filters tools based on context and definition
  - Source: `pydantic_ai_slim/pydantic_ai/toolsets/filtered.py`

## WrapperModel Pattern

Subclass `WrapperModel` to intercept model requests. Override `request()` for request/response modification, or `request_stream()` for streaming.

**Source:** `pydantic_ai_slim/pydantic_ai/models/wrapper.py`

```python
from dataclasses import dataclass

from pydantic_ai.models import Model, KnownModelName, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.settings import ModelSettings


@dataclass(init=False)
class RetryingModel(WrapperModel):
    """Retry failed requests with exponential backoff."""

    max_retries: int = 3

    def __init__(self, wrapped: Model | KnownModelName, max_retries: int = 3):
        super().__init__(wrapped)
        self.max_retries = max_retries

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        import asyncio
        for attempt in range(self.max_retries):
            try:
                return await super().request(messages, model_settings, model_request_parameters)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError('Unreachable')


# Usage
model = RetryingModel('openai:gpt-5', max_retries=3)
agent = Agent(model)
```

### What to Override

| Method | Purpose |
|--------|---------|
| `request()` | Intercept non-streaming requests |
| `request_stream()` | Intercept streaming requests |
| `count_tokens()` | Custom token counting |

### Real Examples in Codebase

- **`InstrumentedModel`** — Adds OpenTelemetry tracing to all requests
  - Source: `pydantic_ai_slim/pydantic_ai/models/instrumented.py`
- **`FallbackModel`** — Tries multiple models in sequence (not a WrapperModel, but similar pattern)
  - Source: `pydantic_ai_slim/pydantic_ai/models/fallback.py`

## RunContext Patterns

Use `RunContext` fields and `ctx.deps` for per-run state and behavior without wrapper classes.

### Using Built-in Context Fields

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-5')


@agent.tool(retries=3)
async def smart_retry(ctx: RunContext, query: str) -> str:
    """Adjust behavior based on retry state."""
    # ctx.retry — current retry count (0 on first attempt)
    # ctx.max_retries — maximum retries for this tool
    # ctx.last_attempt — True if this is the final attempt
    # ctx.run_step — current step in agent loop

    if ctx.retry > 0:
        # On retry, try a different approach
        query = f'[RETRY {ctx.retry}] {query}'

    if ctx.last_attempt:
        # Final attempt — log for debugging
        print(f'Final attempt for query: {query}')

    return f'Processed: {query}'
```

### Using Dependencies for Stateful Abstractions

```python
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    """Dependencies with caching and rate limiting."""
    cache: dict = field(default_factory=dict)
    request_count: int = 0
    max_requests: int = 100


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def cached_lookup(ctx: RunContext[Deps], key: str) -> str:
    """Tool with caching via deps."""
    if key in ctx.deps.cache:
        return ctx.deps.cache[key]

    ctx.deps.request_count += 1
    if ctx.deps.request_count > ctx.deps.max_requests:
        raise RuntimeError('Rate limit exceeded')

    result = f'Value for {key}'
    ctx.deps.cache[key] = result
    return result
```

See [run-context.md](run-context.md) for all available fields.

## Dynamic Tool Patterns

### ToolPrepareFunc — Per-Tool Configuration

Modify individual tool definitions before each agent step:

```python
from dataclasses import replace

from pydantic_ai import Agent, RunContext, Tool, ToolDefinition


async def hide_on_retry(
    ctx: RunContext, tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Hide tool after first attempt to force different approach."""
    if ctx.retry > 0:
        return None  # Tool not available on retry
    return tool_def


agent = Agent(
    'openai:gpt-5',
    tools=[Tool(my_func, prepare=hide_on_retry)],
)
```

### FilteredToolset — Conditional Availability

Filter entire toolsets based on context:

```python
from pydantic_ai import FilteredToolset

# Only expose tools based on user role
admin_only = FilteredToolset(
    all_tools,
    filter_func=lambda ctx, tool_def: (
        not tool_def.name.startswith('admin_') or ctx.deps.is_admin
    ),
)
```

See [toolsets.md](toolsets.md) for more toolset patterns.

## WrapperAgent Pattern

For agent-level middleware (less common), subclass `WrapperAgent`:

```python
from pydantic_ai.agent.wrapper import WrapperAgent


class AuditedAgent(WrapperAgent):
    """Log all agent runs for audit."""

    async def iter(self, user_prompt, **kwargs):
        print(f'Starting run: {user_prompt[:50]}...')
        async with super().iter(user_prompt, **kwargs) as run:
            yield run
        print(f'Run completed: {run.result.output}')
```

**Source:** `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`

## Existing Implementations

Study these framework implementations as patterns for your own extensions:

| Implementation | Pattern | Purpose |
|----------------|---------|---------|
| `ApprovalRequiredToolset` | WrapperToolset | Human-in-the-loop tool approval |
| `FilteredToolset` | WrapperToolset | Context-based tool filtering |
| `InstrumentedModel` | WrapperModel | OpenTelemetry observability |
| `FallbackModel` | Model composition | Multi-provider resilience |
| `LangChainToolset` | AbstractToolset | External framework adapter |

## See Also

- [toolsets.md](toolsets.md) — Toolset composition and filtering
- [models.md](models.md) — Model configuration and FallbackModel
- [run-context.md](run-context.md) — All RunContext fields
- [tools-advanced.md](tools-advanced.md) — ToolPrepareFunc and advanced tool patterns
- [dependencies.md](dependencies.md) — Dependency injection patterns
