# Extensibility & Customization

Pydantic AI is designed to be extended. The three core abstractions — **agents**, **models**, and **toolsets** — each have a wrapper base class that lets you intercept, modify, or augment behavior without rewriting internals.

This guide covers:

- [Wrapper Agents](#wrapper-agents) — intercept agent runs
- [Wrapper Models](#wrapper-models) — intercept model requests
- [Wrapper Toolsets](#wrapper-toolsets) — intercept tool calls

All wrappers follow the same pattern: subclass the wrapper base, override the methods you care about, and delegate the rest to `self.wrapped`.

## Wrapper Agents

[`WrapperAgent`][pydantic_ai.agent.WrapperAgent] wraps an existing agent and delegates all calls to it. Subclass it to add behavior before or after agent runs — such as logging, metrics, guardrails, or durable execution.

```python {test="skip"}
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic_ai.agent import WrapperAgent
from pydantic_ai.run import AgentRun


class TimedAgent(WrapperAgent):
    """Agent wrapper that logs the duration of each run."""

    @asynccontextmanager
    async def iter(self, *args, **kwargs) -> AsyncIterator[AgentRun]:
        start = time.monotonic()
        async with self.wrapped.iter(*args, **kwargs) as run:
            yield run
        elapsed = time.monotonic() - start
        print(f'Run completed in {elapsed:.2f}s')
```

### When to use WrapperAgent

- **Logging and metrics** — record run durations, token usage, or custom events
- **Guardrails** — validate inputs/outputs before or after runs
- **Durable execution** — the built-in [`PrefectAgent`][pydantic_ai.durable_exec.prefect.PrefectAgent], [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent], and [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent] all extend `WrapperAgent` to offload model requests and tool calls to their respective frameworks

### Key methods to override

| Method | Purpose |
|---|---|
| `iter()` | The core method — wraps the agent run lifecycle |
| `override()` | Temporarily override agent config (model, deps, etc.) |

All `run()`, `run_sync()`, and `run_stream()` methods call `iter()` internally, so overriding `iter()` is usually sufficient.

## Wrapper Models

[`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] wraps a model to intercept or modify requests and responses. This is useful for caching, rate limiting, request/response transforms, or custom routing logic.

```python {test="skip"}
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.settings import ModelSettings


class LoggingModel(WrapperModel):
    """Model wrapper that logs each request."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        print(f'Sending {len(messages)} messages to {self.model_name}')
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        print(f'Received response with {response.usage.output_tokens} output tokens')
        return response
```

### Built-in model wrappers

Pydantic AI ships with several model wrappers that extend [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel]:

- [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel] — adds OpenTelemetry tracing and metrics
- [`ConcurrencyLimitedModel`][pydantic_ai.models.concurrency.ConcurrencyLimitedModel] — limits concurrent requests to a model

[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] also provides model composition (trying multiple models in sequence on failure), but it extends [`Model`][pydantic_ai.models.Model] directly rather than `WrapperModel` since it wraps multiple models.

### Key methods to override

| Method | Purpose |
|---|---|
| `request()` | Intercept a standard (non-streaming) model request |
| `request_stream()` | Intercept a streaming model request |
| `count_tokens()` | Modify token counting behavior |
| `customize_request_parameters()` | Modify tool definitions or output schemas before a request |
| `prepare_request()` | Modify settings and parameters together before a request |

## Wrapper Toolsets

[`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] wraps a toolset to intercept tool definitions and/or tool calls. This is the most commonly extended wrapper.

For a full example, see [Wrapping a Toolset](toolsets.md#wrapping-a-toolset) in the toolsets guide.

### Built-in toolset wrappers

Pydantic AI provides several ready-made toolset wrappers, all available as chainable methods on any toolset:

| Wrapper | Method | Purpose |
|---|---|---|
| [`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset] | `.filtered()` | Include/exclude tools dynamically based on context |
| [`PrefixedToolset`][pydantic_ai.toolsets.PrefixedToolset] | `.prefixed()` | Add a prefix to tool names to avoid conflicts |
| [`RenamedToolset`][pydantic_ai.toolsets.RenamedToolset] | `.renamed()` | Rename tools using a mapping dict |
| [`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset] | `.prepared()` | Modify tool definitions at runtime |
| [`ApprovalRequiredToolset`][pydantic_ai.toolsets.ApprovalRequiredToolset] | `.approval_required()` | Require human approval before tool execution |

These can be chained:

```python {test="skip"}
toolset.filtered(my_filter).prefixed('weather').prepared(my_prepare_func)
```

### Key methods to override

| Method | Purpose |
|---|---|
| `get_tools()` | Modify which tools are available and their definitions |
| `call_tool()` | Intercept tool execution (logging, validation, caching) |
| `__aenter__` / `__aexit__` | Manage resources (connections, sessions) |

## Composition patterns

### Stacking wrappers

Wrappers can be composed by nesting them:

```python {test="skip"}
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.instrumented import InstrumentedModel, InstrumentationSettings

# Stack: instrumentation → concurrency limit → base model
model = InstrumentedModel(
    ConcurrencyLimitedModel('openai:gpt-5.2', limiter=5),
    InstrumentationSettings(),
)
```

### Building a custom toolset from scratch

If the wrapper pattern doesn't fit, you can subclass [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] directly and implement `get_tools()` and `call_tool()`:

```python {test="skip"}
from pydantic_ai import AbstractToolset, RunContext, ToolsetTool


class MyCustomToolset(AbstractToolset):
    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        # Return your tool definitions
        ...

    async def call_tool(self, name: str, tool_args: dict, ctx: RunContext, tool: ToolsetTool):
        # Execute the tool
        ...
```

See the [toolsets documentation](toolsets.md#building-a-custom-toolset) for a complete example.
