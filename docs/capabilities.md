
# Capabilities

A capability is a reusable, composable unit of agent behavior. Instead of threading multiple arguments through your `Agent` constructor — instructions here, model settings there, a toolset somewhere else, a history processor on yet another parameter — you can bundle related behavior into a single capability and pass it via the [`capabilities`][pydantic_ai.agent.Agent.__init__] parameter.

Capabilities can provide any combination of:

* **Model settings** — static or per-step configuration
* **Tools** — via toolsets or builtin tools
* **Instructions** — static or dynamic system prompt additions
* **Lifecycle hooks** — intercept and modify model requests, tool calls, and the overall run

This makes them the primary extension point for Pydantic AI. Whether you're building a memory system, a guardrail, a cost tracker, or an approval workflow, a capability is the right abstraction.

## Using built-in capabilities

Pydantic AI ships with several capabilities that cover common needs:

| Capability | What it provides | Spec |
|---|---|:---:|
| [`BuiltinTool`][pydantic_ai.capabilities.BuiltinTool] | Registers a [builtin tool](builtin-tools.md) with the agent | Yes |
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | Web search — builtin when supported, local fallback otherwise | Yes |
| [`WebFetch`][pydantic_ai.capabilities.WebFetch] | URL fetching — builtin when supported, custom local fallback | Yes |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | Image generation — builtin when supported, custom local fallback | Yes |
| [`MCP`][pydantic_ai.capabilities.MCP] | MCP server — builtin when supported, direct connection otherwise | Yes |
| [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] | Filters or modifies tool definitions per step | — |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] | — |
| [`HistoryProcessor`][pydantic_ai.capabilities.HistoryProcessor] | Wraps a [history processor](message-history.md) | — |

The **Spec** column indicates whether the capability can be used in [agent specs](#agent-specs) (YAML/JSON). Capabilities marked **—** take non-serializable arguments (callables, toolset objects) and can only be used in Python code.

```python {title="builtin_capabilities.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    model_settings={'max_tokens': 8192},
    capabilities=[
        WebSearch(),
    ],
)
```

Instructions and model settings are configured directly via the `instructions` and `model_settings` parameters on `Agent` (or `AgentSpec`). Capabilities are for behavior that goes beyond simple configuration — tools, lifecycle hooks, and custom extensions. They compose well, especially when you want to reuse the same configuration across multiple agents or load it from a [spec file](#agent-specs).

### Builtin tool capabilities

[`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], and [`MCP`][pydantic_ai.capabilities.MCP] wrap [builtin tools](builtin-tools.md) with automatic local fallbacks. When the model supports the builtin natively, it's used directly. When it doesn't, a local function tool handles it instead — so your agent works across models without code changes.

Each accepts `builtin` and `local` keyword arguments to control which side is used:

```python {title="builtin_tool_capabilities.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, WebFetch, WebSearch

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        # Auto-detects DuckDuckGo as local fallback
        WebSearch(),
        # Builtin URL fetching; provide local= for fallback
        WebFetch(),
        # Auto-detects transport from URL
        MCP(url='https://mcp.example.com/api'),
    ],
)
```

To force builtin-only (errors on unsupported models instead of falling back to local):

```python {title="builtin_only.py" test="skip" lint="skip"}
MCP(url='https://mcp.example.com/api', local=False)
```

To force local-only (never use the builtin, even when the model supports it):

```python {title="local_only.py" test="skip" lint="skip"}
MCP(url='https://mcp.example.com/api', builtin=False)
```

Constraint fields like `allowed_domains` or `blocked_domains` require the builtin — the local fallback can't enforce them. When these are set and the model doesn't support the builtin, a [`UserError`][pydantic_ai.exceptions.UserError] is raised:

```python {title="constraints.py" test="skip" lint="skip"}
# Only search example.com — requires builtin support
WebSearch(allowed_domains=['example.com'])
```

## Building a custom capability

To build your own capability, subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] and override the methods you need. There are two categories: **configuration methods** that are called once at agent construction, and **lifecycle hooks** that fire during each run.

### Providing configuration

The simplest capabilities just provide static configuration. Here's a `KnowsCurrentTime` capability that injects the current time into the system prompt:

```python {title="custom_capability_config.py"}
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """Tells the agent what time it is."""

    def get_instructions(self):
        return f'The current date and time is {datetime.now().isoformat()}.'


agent = Agent('openai:gpt-5.2', capabilities=[KnowsCurrentTime()])
result = agent.run_sync('What time is it?')
print(result.output)
#> The current time is 3:45 PM.
```

A capability that provides tools can return a pre-built [toolset](toolsets.md) from [`get_toolset`][pydantic_ai.capabilities.AbstractCapability.get_toolset]:

```python {title="custom_capability_tools.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


def _add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def _multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


_math_toolset = FunctionToolset(tools=[_add, _multiply])


@dataclass
class MathTools(AbstractCapability[Any]):
    """Provides basic math operations."""

    def get_toolset(self):
        return _math_toolset


agent = Agent('openai:gpt-5.2', capabilities=[MathTools()])
result = agent.run_sync('What is 2 + 3?')
print(result.output)
#> The answer is 5.0
```

The full set of configuration methods:

| Method | Return type | Purpose |
|---|---|---|
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`AgentInstructions`][pydantic_ai._instructions.AgentInstructions] ` \| None` | System prompt additions (static strings, [template strings](#template-strings), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | Model settings dict, or a callable for [per-step settings](#dynamic-model-settings) |
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AgentToolset`][pydantic_ai.toolsets.AgentToolset] ` \| None` | A [toolset](toolsets.md) to register with the agent |
| [`get_builtin_tools()`][pydantic_ai.capabilities.AbstractCapability.get_builtin_tools] | `Sequence[AbstractBuiltinTool]` | [Builtin tools](builtin-tools.md) to register |
| [`get_wrapper_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] | [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] ` \| None` | [Wrap the agent's assembled toolset](#toolset-wrapping) |

### Dynamic model settings

When model settings need to vary per step — for example, enabling thinking only on retry — return a callable from [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings]:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.settings import ModelSettings


@dataclass
class ThinkingOnRetry(AbstractCapability[None]):
    """Enables thinking mode when the agent is retrying."""

    def get_model_settings(self):
        def resolve(ctx: RunContext[None]) -> ModelSettings:
            if ctx.run_step > 1:
                return ModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 5000})
            return ModelSettings()

        return resolve


agent = Agent('openai:gpt-5.2', capabilities=[ThinkingOnRetry()])
result = agent.run_sync('hello')
print(result.output)
#> Hello! How can I help you today?
```

The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] where `ctx.model_settings` contains the merged result of all layers resolved before this capability (model defaults and agent-level settings).

## Lifecycle hooks

Capabilities can hook into five lifecycle points, each with up to four variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action succeeds (in reverse capability order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it
* **`on_*_error`** — fires when the action fails (after `wrap_*` has had its chance to recover), can observe, transform, or recover from errors

### Short-circuit exceptions

Three exceptions allow hooks to short-circuit the normal flow with a replacement value:

| Exception | Raised in | Effect |
|---|---|---|
| [`SkipModelRequest(response)`][pydantic_ai.exceptions.SkipModelRequest] | `before_model_request`, `wrap_model_request` | Skips the model call, uses the provided `ModelResponse` instead |
| [`SkipToolValidation(validated_args)`][pydantic_ai.exceptions.SkipToolValidation] | `before_tool_validate`, `wrap_tool_validate` | Skips argument validation, uses the provided `dict` as validated args |
| [`SkipToolExecution(result)`][pydantic_ai.exceptions.SkipToolExecution] | `before_tool_execute`, `wrap_tool_execute` | Skips tool execution, uses the provided value as the tool result |

### Run hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_run`][pydantic_ai.capabilities.AbstractCapability.before_run] | `(ctx: RunContext) -> None` | Observe-only notification that a run is starting |
| [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run] | `(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult` | Modify the final result |
| [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] | `(ctx: RunContext, *, handler: () -> AgentRunResult) -> AgentRunResult` | Wrap the entire run |
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | `(ctx: RunContext, *, error: BaseException) -> AgentRunResult` | Handle run errors (see [error hooks](#error-hooks)) |

`wrap_run` supports error recovery: if `handler()` raises and `wrap_run` catches the exception and returns a result instead, the error is suppressed and the recovery result is used. This works with both [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] and [`agent.iter()`][pydantic_ai.agent.Agent.iter].

### Node hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_node_run`][pydantic_ai.capabilities.AbstractCapability.before_node_run] | `(ctx: RunContext, *, node: AgentNode) -> AgentNode` | Observe or replace the node before execution |
| [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run] | `(ctx: RunContext, *, node: AgentNode, result: NodeResult) -> NodeResult` | Modify the result (next node or `End`) |
| [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] | `(ctx: RunContext, *, node: AgentNode, handler: (AgentNode) -> AgentNode \| End) -> AgentNode \| End` | Wrap each graph node execution |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | `(ctx: RunContext, *, node: AgentNode, error: BaseException) -> AgentNode \| End` | Handle node errors (see [error hooks](#error-hooks)) |

The [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] hook fires for every node in the [agent graph](agent.md#iterating-over-an-agents-graph) ([`UserPromptNode`][pydantic_ai.UserPromptNode], [`ModelRequestNode`][pydantic_ai.ModelRequestNode], [`CallToolsNode`][pydantic_ai.CallToolsNode]). The `handler` executes the node and returns the next node (or [`End`][pydantic_graph.nodes.End]). Override this to observe node transitions, add per-step logging, or modify graph progression:

!!! note
    `wrap_node_run` hooks are called automatically by [`agent.run()`][pydantic_ai.agent.AbstractAgent.run], [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], and [`agent_run.next()`][pydantic_ai.run.AgentRun.next]. However, they are **not** called when iterating with bare `async for node in agent_run:` over [`agent.iter()`][pydantic_ai.agent.Agent.iter], since that uses the graph run's internal iteration. Always use `agent_run.next(node)` to advance the run if you need `wrap_node_run` hooks to fire.

```python {title="node_logging_example.py"}
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapNodeRunHandler,
)


@dataclass
class NodeLogger(AbstractCapability[Any]):
    """Logs each node that executes during a run."""

    nodes: list[str] = field(default_factory=lambda: [])

    async def wrap_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]
    ) -> NodeResult[Any]:
        self.nodes.append(type(node).__name__)
        return await handler(node)


logger = NodeLogger()
agent = Agent('openai:gpt-5.2', capabilities=[logger])
agent.run_sync('hello')
print(logger.nodes)
#> ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']
```

You can also use `wrap_node_run` to modify graph progression — for example, limiting the number of model requests per run:

```python {title="node_modification_example.py" test="skip" lint="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_graph import End

from pydantic_ai import ModelRequestNode, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, WrapNodeRunHandler
from pydantic_ai.result import FinalResult


@dataclass
class MaxModelRequests(AbstractCapability[Any]):
    """Limits the number of model requests per run by ending early."""

    max_requests: int = 5
    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> 'MaxModelRequests':
        return MaxModelRequests(max_requests=self.max_requests)  # fresh per run

    async def wrap_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]
    ) -> NodeResult[Any]:
        if isinstance(node, ModelRequestNode):
            self.count += 1
            if self.count > self.max_requests:
                return End(FinalResult(output='Max model requests reached'))
        return await handler(node)
```

See [Iterating Over an Agent's Graph](agent.md#iterating-over-an-agents-graph) for more about the agent graph and its node types.

### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] | `(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext` | Modify messages, settings, or parameters before the model call |
| [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse` | Modify the model's response |
| [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, handler: (ModelRequestContext) -> ModelResponse) -> ModelResponse` | Wrap the model call |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | `(ctx: RunContext, *, request_context: ModelRequestContext, error: Exception) -> ModelResponse` | Handle model request errors (see [error hooks](#error-hooks)) |

[`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] bundles `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof.

### Tool hooks

Tool processing has two phases: **validation** (parsing and validating the model's JSON arguments against the tool's schema) and **execution** (running the tool function). Each phase has its own hooks:

**Validation hooks** — `args` is the raw `str | dict[str, Any]` from the model before validation, or the validated `dict[str, Any]` after:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_validate`][pydantic_ai.capabilities.AbstractCapability.before_tool_validate] | `(ctx, *, call: ToolCallPart, args: str \| dict) -> str \| dict` | Modify raw args before validation (e.g. JSON repair) |
| [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate] | `(ctx, *, call: ToolCallPart, args: dict) -> dict` | Modify validated args |
| [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate] | `(ctx, *, call: ToolCallPart, args: str \| dict, handler) -> dict` | Wrap the validation step |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | `(ctx, *, call: ToolCallPart, args: str \| dict, error) -> dict` | Handle validation errors (see [error hooks](#error-hooks)) |

**Execution hooks** — `args` is always the validated `dict[str, Any]`:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict) -> dict` | Modify args before execution |
| [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict, result: Any) -> Any` | Modify execution result |
| [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict, handler) -> Any` | Wrap execution |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | `(ctx, *, call: ToolCallPart, args: dict, error: Exception) -> Any` | Handle execution errors (see [error hooks](#error-hooks)) |

### Tool preparation

Capabilities can filter or modify which tool definitions the model sees on each step via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools]. This controls tool **visibility**, not execution — use execution hooks for that.

```python {title="prepare_tools_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import ToolDefinition


@dataclass
class HideDangerousTools(AbstractCapability[Any]):
    """Hides tools matching certain name prefixes from the model."""

    hidden_prefixes: tuple[str, ...] = ('delete_', 'drop_')

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [td for td in tool_defs if not any(td.name.startswith(p) for p in self.hidden_prefixes)]


agent = Agent('openai:gpt-5.2', capabilities=[HideDangerousTools()])


@agent.tool_plain
def delete_file(path: str) -> str:
    """Delete a file."""
    return f'deleted {path}'


@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file."""
    return f'contents of {path}'


result = agent.run_sync('hello')
# The model only sees `read_file`, not `delete_file`
```

The list includes all tool kinds (function, output, unapproved) — use `tool_def.kind` to distinguish. This hook runs after the agent-level [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc]. For simple cases, the built-in [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] capability wraps a callable without needing a custom subclass.

### Toolset wrapping

While `prepare_tools` modifies tool *definitions* per step, [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] lets a capability wrap the agent's entire assembled toolset with a [`WrapperToolset`](toolsets.md#changing-tool-execution). This is more powerful — it can intercept tool execution, replace tools entirely, or apply any cross-cutting behavior.

The wrapper receives the combined non-output toolset (after the agent-level [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc] wrapping). Output tools are added separately and are not affected.

```python {title="wrapper_toolset_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset


@dataclass
class NamespaceTools(AbstractCapability[Any]):
    """Prefixes all tool names with a namespace."""

    namespace: str

    def get_wrapper_toolset(
        self, toolset: AbstractToolset[Any]
    ) -> AbstractToolset[Any]:
        return PrefixedToolset(toolset, prefix=self.namespace)


agent = Agent('openai:gpt-5.2', capabilities=[NamespaceTools(namespace='myapp')])


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'


result = agent.run_sync('hello')
# The model sees `myapp_greet` instead of `greet`
```

!!! note
    `prepare_tools` can also be expressed as a wrapper: `get_wrapper_toolset(toolset) -> toolset.prepared(fn)`. The difference is that `prepare_tools` (the capability hook) operates on tool *definitions* for all tool kinds per step, while `get_wrapper_toolset` wraps the non-output *toolset* once per run (during toolset assembly), intercepting tool execution rather than just modifying definitions.

### Event stream hook

For runs with event streaming ([`run_stream_events`][pydantic_ai.agent.AbstractAgent.run_stream_events], [`event_stream_handler`][pydantic_ai.agent.Agent.__init__], [UI event streams](ui/overview.md)), capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream] | `(ctx, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

`wrap_run_event_stream` is an async generator — yield events directly without needing an inner function:

```python {title="event_stream_example.py" test="skip"}
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    TextPart,
)


@dataclass
class StreamAuditor(AbstractCapability[Any]):
    """Logs tool calls and text output during streamed runs."""

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        async for event in stream:
            if isinstance(event, FunctionToolCallEvent):
                print(f'Tool called: {event.part.tool_name}')
            elif isinstance(event, FunctionToolResultEvent):
                print(f'Tool result: {event.tool_return.content!r}')
            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                print(f'Text: {event.part.content!r}')
            yield event
```

For building web UIs that transform streamed events into protocol-specific formats (like SSE), see the [UI event streams](ui/overview.md) documentation and the [`UIEventStream`][pydantic_ai.ui.UIEventStream] base class.

### Error hooks

Each lifecycle point has an `on_*_error` hook — the error counterpart to `after_*`. While `after_*` hooks fire on success, `on_*_error` hooks fire on failure (after `wrap_*` has had its chance to recover):

```
before_X → wrap_X(handler)
  ├─ success ─────────→ after_X (modify result)
  └─ failure → on_X_error
        ├─ re-raise ──→ (error propagates, after_X not called)
        └─ recover ───→ after_X (modify recovered result)
```

Error hooks use **raise-to-propagate, return-to-recover** semantics:

- **Raise the original error** — propagates the error unchanged *(default)*
- **Raise a different exception** — transforms the error
- **Return a result** — suppresses the error and uses the returned value

| Hook | Fires when | Recovery type |
|---|---|---|
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | Agent run fails | Return [`AgentRunResult`][pydantic_ai.run.AgentRunResult] |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | Graph node fails | Return next node or [`End`][pydantic_graph.nodes.End] |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | Model request fails | Return [`ModelResponse`][pydantic_ai.messages.ModelResponse] |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | Tool validation fails | Return validated args `dict` |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | Tool execution fails | Return any tool result |

```python {title="error_hooks_example.py" test="skip" lint="skip"}
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext


@dataclass
class ErrorLogger(AbstractCapability[Any]):
    """Logs all errors that occur during agent runs."""

    errors: list[str] = field(default_factory=list)

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.errors.append(f'Model error: {error}')
        # Return a fallback response to recover
        return ModelResponse(parts=[TextPart(content='Service temporarily unavailable.')])

    async def on_tool_execute_error(
        self, ctx: RunContext[Any], *, call: Any, tool_def: Any, args: dict[str, Any], error: Exception
    ) -> Any:
        self.errors.append(f'Tool {call.tool_name} failed: {error}')
        raise error  # Re-raise to let the normal retry flow handle it
```

### Per-run state isolation

By default, a capability instance is shared across all runs of an agent. If your capability accumulates mutable state that should not leak between runs, override [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] to return a fresh instance:

```python {title="per_run_state.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext


@dataclass
class RequestCounter(AbstractCapability[Any]):
    """Counts model requests per run."""

    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> 'RequestCounter':
        return RequestCounter()  # fresh instance for each run

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.count += 1
        return request_context


counter = RequestCounter()
agent = Agent('openai:gpt-5.2', capabilities=[counter])

# The shared counter stays at 0 because for_run returns a fresh instance
agent.run_sync('first run')
agent.run_sync('second run')
print(counter.count)
#> 0
```

## Example: building a guardrail

A guardrail is a capability that intercepts model requests or responses to enforce safety rules. Here's one that scans model responses for potential PII and redacts it:

```python {title="guardrail_example.py"}
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import ModelRequestContext


@dataclass
class PIIRedactionGuardrail(AbstractCapability[Any]):
    """Redacts email addresses and phone numbers from model responses."""

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                # Redact email addresses
                part.content = re.sub(
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    '[EMAIL REDACTED]',
                    part.content,
                )
                # Redact phone numbers (simple US pattern)
                part.content = re.sub(
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                    '[PHONE REDACTED]',
                    part.content,
                )
        return response


agent = Agent('openai:gpt-5.2', capabilities=[PIIRedactionGuardrail()])
result = agent.run_sync("What's Jane's contact info?")
print(result.output)
#> You can reach Jane at [EMAIL REDACTED] or [PHONE REDACTED].
```

## Example: building a logging middleware

The `wrap_*` pattern is useful when you need to observe or time both the input and output of an operation. Here's a capability that logs every model request and tool call:

```python {title="logging_middleware_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition


@dataclass
class VerboseLogging(AbstractCapability[Any]):
    """Logs model requests and tool executions."""

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        print(f'  Model request (step {ctx.run_step}, {len(request_context.messages)} messages)')
        #>   Model request (step 1, 1 messages)
        response = await handler(request_context)
        print(f'  Model response: {len(response.parts)} parts')
        #>   Model response: 1 parts
        return response

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        print(f'  Tool call: {call.tool_name}({args})')
        result = await handler(args)
        print(f'  Tool result: {result!r}')
        return result


agent = Agent('openai:gpt-5.2', capabilities=[VerboseLogging()])
result = agent.run_sync('hello')
print(f'Output: {result.output}')
#> Output: Hello! How can I help you today?
```

## Composition

When multiple capabilities are passed to an agent, they are composed into a single [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability]:

* **Configuration** is merged: instructions concatenate, model settings merge additively (later capabilities override earlier ones), toolsets combine, builtin tools collect.
* **`before_*`** hooks fire in capability order: `cap1 → cap2 → cap3`.
* **`after_*`** hooks fire in reverse order: `cap3 → cap2 → cap1`.
* **`wrap_*`** hooks nest as middleware: `cap1` wraps `cap2` wraps `cap3` wraps the actual operation. The first capability is the outermost layer.

This means the first capability in the list has the first and last say on the operation — it sees the original input in its `wrap_*` before handler, and it sees the final output after handler returns.

## Template strings

[`TemplateStr`][pydantic_ai.TemplateStr] lets you write Handlebars-style templates (`{{variable}}`) that are rendered against the agent's [dependencies](dependencies.md) at runtime. Template strings work anywhere instructions or descriptions are accepted — in Python code, in capabilities, and in [agent specs](#agent-specs).

```python {title="template_instructions.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, TemplateStr


@dataclass
class UserProfile:
    name: str
    role: str


agent = Agent(
    'openai:gpt-5.2',
    deps_type=UserProfile,
    instructions=TemplateStr('You are assisting {{name}}, who is a {{role}}.'),
)
result = agent.run_sync('hello', deps=UserProfile(name='Alice', role='engineer'))
print(result.output)
#> Hello! How can I help you today?
```

Template variables are resolved from the fields of the `deps` object. When a `deps_type` is provided, template variable names are validated at construction time.

In [agent specs](#agent-specs), strings containing `{{` are automatically converted to template strings — no explicit `TemplateStr` wrapper is needed. This applies to the `instructions` and `description` fields.

## Agent specs

Capabilities integrate with the YAML/JSON agent spec system, allowing you to define agents declaratively:

```yaml {title="agent.yaml" test="skip"}
model: anthropic:claude-opus-4-6
instructions: You are a helpful research assistant.
model_settings:
  max_tokens: 8192
capabilities:
  - WebSearch
```

### Spec syntax

Capabilities in specs support three forms:

* `'MyCapability'` — no arguments, calls `MyCapability.from_spec()`
* `{'MyCapability': value}` — single positional argument, calls `MyCapability.from_spec(value)`
* `{'MyCapability': {key: value, ...}}` — keyword arguments, calls `MyCapability.from_spec(**kwargs)`

### Custom capabilities in specs

To make a custom capability work with specs, it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

```python {title="custom_spec_capability.py" test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class RateLimit(AbstractCapability[Any]):
    """Limits requests per minute."""

    rpm: int = 60


# In YAML: `- RateLimit: {rpm: 30}`
# In Python:
agent = Agent.from_spec(
    AgentSpec(model='test', capabilities=[{'RateLimit': {'rpm': 30}}]),
    custom_capability_types=[RateLimit],
)
```

Override [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] when the constructor takes types that can't be represented in YAML/JSON. The spec fields should mirror the dataclass fields, but with serializable types:

```python {title="from_spec_override_example.py" test="skip" lint="skip"}
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import ToolDefinition


@dataclass
class ConditionalTools(AbstractCapability[Any]):
    """Hides tools unless a condition is met."""

    condition: Callable[[RunContext[Any]], bool]  # not serializable
    hidden_tools: list[str] = ()

    @classmethod
    def from_spec(cls, hidden_tools: list[str]) -> 'ConditionalTools':
        # In the spec, there's no condition callable — always hide
        return cls(condition=lambda ctx: True, hidden_tools=hidden_tools)

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        if self.condition(ctx):
            return [td for td in tool_defs if td.name not in self.hidden_tools]
        return tool_defs
```

In YAML this would be `- ConditionalTools: {hidden_tools: [dangerous_tool]}`. In Python code, the full constructor is available: `ConditionalTools(condition=my_check, hidden_tools=['dangerous_tool'])`.

Pass custom capability types via the `custom_capability_types` parameter so the spec resolver can find them.

### Loading specs

[`Agent.from_file`][pydantic_ai.Agent.from_file] loads a spec from a YAML or JSON file and constructs an agent:

```python {title="from_file_example.py" test="skip"}
from pydantic_ai import Agent

agent = Agent.from_file('agent.yaml')
```

[`Agent.from_spec`][pydantic_ai.Agent.from_spec] accepts a dict or [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] instance and supports additional keyword arguments that supplement or override the spec:

```python {title="from_spec_example.py" test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_spec(
    {
        'model': 'anthropic:claude-opus-4-6',
        'instructions': 'You are helping {{user_name}}.',
        'capabilities': ['WebSearch'],
    },
    deps_type=UserContext,
)
```

Keyword arguments interact with spec fields as follows:

* **Scalar fields** (`model`, `name`, `retries`, `end_strategy`, etc.) — the keyword argument overrides the spec value when provided.
* **`instructions`** — merged: spec instructions come first, then keyword argument instructions.
* **`capabilities`** — merged: spec capabilities come first, then keyword argument capabilities.
* **`model_settings`** — merged additively: keyword argument settings override matching spec settings.
* **`output_type`** — takes precedence over `output_schema` from the spec.

When `deps_type` is passed, [template strings](#template-strings) in the spec's `instructions`, `description`, and capability arguments are compiled and validated against the deps type at construction time.

For more control over spec loading, use [`AgentSpec.from_file`][pydantic_ai.agent.spec.AgentSpec.from_file] to load the spec separately before passing it to `Agent.from_spec`.

### `AgentSpec`

The [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] model represents the full spec structure:

| Field | Type | Description |
|---|---|---|
| `model` | `str` | Model name (required) |
| `name` | `str \| None` | Agent name |
| `description` | `str \| None` | Agent description (supports [templates](#template-strings)) |
| `instructions` | `str \| list[str] \| None` | System prompt instructions (supports [templates](#template-strings)) |
| `model_settings` | `dict \| None` | Model settings |
| `capabilities` | `list` | Capabilities (see [spec syntax](#spec-syntax)) |
| `deps_schema` | `dict \| None` | JSON Schema for [template string](#template-strings) validation (see below) |
| `output_schema` | `dict \| None` | JSON Schema for [structured output](output.md) (see below) |
| `retries` | `int` | Default tool retries (default: `1`) |
| `output_retries` | `int \| None` | Output validation retries |
| `end_strategy` | `EndStrategy` | When to stop (`'early'` or `'exhaustive'`) |
| `tool_timeout` | `float \| None` | Default tool timeout in seconds |
| `instrument` | `bool \| None` | Enable [Logfire](logfire.md) instrumentation |
| `metadata` | `dict \| None` | Agent metadata |

#### `deps_schema`

When loading a spec without a Python `deps_type`, `deps_schema` provides a JSON Schema that is used to validate [template string](#template-strings) variable names at construction time. It does **not** validate the actual deps object at runtime — it only ensures that template variables like `{{user_name}}` correspond to properties defined in the schema.

#### `output_schema`

When provided (and no `output_type` keyword argument is passed to `from_spec`), `output_schema` defines the structure the model should produce as its final output. Under the hood, it creates a [`StructuredDict`][pydantic_ai.output.StructuredDict] output type: the JSON Schema is sent to the model API so the model knows what structure to produce, and the response is returned as a `dict[str, Any]`.

!!! note
    The model's response is not validated against the schema's `properties` or `required` fields — it is accepted as a plain dict. The schema serves as an instruction to the model, not a runtime validation constraint.

```yaml {title="agent_with_schema.yaml" test="skip"}
model: anthropic:claude-opus-4-6
deps_schema:
  type: object
  properties:
    user_name:
      type: string
  required: [user_name]
output_schema:
  type: object
  properties:
    answer:
      type: string
    confidence:
      type: number
  required: [answer, confidence]
instructions: "You are helping {{user_name}}. Always include a confidence score."
capabilities:
  - WebSearch
```

### Saving specs

[`AgentSpec.to_file`][pydantic_ai.agent.spec.AgentSpec.to_file] saves a spec to YAML or JSON and optionally generates a companion JSON Schema file for editor autocompletion:

```python {title="save_spec_example.py" test="skip"}
from pydantic_ai.agent.spec import AgentSpec

spec = AgentSpec(
    model='anthropic:claude-opus-4-6',
    instructions='You are a helpful assistant.',
    capabilities=['WebSearch'],
)
spec.to_file('agent.yaml')
# Also generates ./agent_schema.json for editor autocompletion
```

The generated JSON Schema file enables autocompletion and validation in editors that support the [YAML Language Server](https://github.com/redhat-developer/yaml-language-server) protocol. Pass `schema_path=None` to skip schema generation.
