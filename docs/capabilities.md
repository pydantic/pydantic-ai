# Capabilities

A capability is a reusable, composable unit of agent behavior. Instead of threading multiple arguments through your `Agent` constructor — [instructions](agent.md#instructions) here, [model settings](agent.md#model-run-settings) there, a [toolset](toolsets.md) somewhere else, a [history processor](message-history.md#processing-message-history) on yet another parameter — you can bundle related behavior into a single capability and pass it via the [`capabilities`][pydantic_ai.agent.Agent.__init__] parameter.

Capabilities can provide any combination of:

* **Tools** — via [toolsets](toolsets.md) or [builtin tools](builtin-tools.md)
* **Lifecycle hooks** — intercept and modify model requests, tool calls, and the overall run
* **Instructions** — static or dynamic [instruction](agent.md#instructions) additions
* **Model settings** — static or per-step [model settings](agent.md#model-run-settings)

This makes them the primary extension point for Pydantic AI. Whether you're building a memory system, a guardrail, a cost tracker, or an approval workflow, a capability is the right abstraction.

## Built-in capabilities

Pydantic AI ships with several capabilities that cover common needs:

| Capability | What it provides | Spec |
|---|---|:---:|
| [`Thinking`][pydantic_ai.capabilities.Thinking] | Enables model [thinking/reasoning](thinking.md) at configurable effort | Yes |
| [`Hooks`][pydantic_ai.capabilities.Hooks] | Decorator-based [lifecycle hook](hooks.md) registration | — |
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | Web search — builtin when supported, [local fallback](common-tools.md#duckduckgo-search-tool) with [`duckduckgo` extra](install.md#slim-install) | Yes |
| [`WebFetch`][pydantic_ai.capabilities.WebFetch] | URL fetching — builtin when supported, [local fallback](common-tools.md#web-fetch-tool) with [`web-fetch` extra](install.md#slim-install) | Yes |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | Image generation — builtin when supported, subagent fallback via `fallback_model` | Yes |
| [`MCP`][pydantic_ai.capabilities.MCP] | MCP server — builtin when supported, direct connection otherwise | Yes |
| [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] | Filters or modifies [tool definitions](tools.md) per step | — |
| [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] | Wraps a capability and prefixes its tool names | Yes |
| [`BuiltinTool`][pydantic_ai.capabilities.BuiltinTool] | Registers a [builtin tool](builtin-tools.md) with the agent | Yes |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] | — |
| [`HistoryProcessor`][pydantic_ai.capabilities.HistoryProcessor] | Wraps a [history processor](message-history.md#processing-message-history) | — |
| [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] | Uses a custom thread executor for [sync functions](tools-advanced.md#thread-executor-for-long-running-servers) | — |

The **Spec** column indicates whether the capability can be used in [agent specs](agent-spec.md) (YAML/JSON). Capabilities marked **—** take non-serializable arguments (callables, toolset objects) and can only be used in Python code.

```python {title="builtin_capabilities.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

[Instructions](agent.md#instructions) and [model settings](agent.md#model-run-settings) are configured directly via the `instructions` and `model_settings` parameters on `Agent` (or [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec]). Capabilities are for behavior that goes beyond simple configuration — tools, lifecycle hooks, and custom extensions. They compose well, especially when you want to reuse the same configuration across multiple agents or load it from a [spec file](agent-spec.md).

### Thinking

The [`Thinking`][pydantic_ai.capabilities.Thinking] capability enables model [thinking/reasoning](thinking.md) at a configurable effort level. It's the simplest way to enable thinking across providers:

```python {title="thinking_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Thinking(effort='high')])
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

See [Thinking](thinking.md) for provider-specific details and the [unified thinking settings](thinking.md#unified-thinking-settings).

### Compaction

Provider-specific compaction capabilities manage conversation context size by compacting older messages into summaries:

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](models/anthropic.md#message-compaction) |

You can also use [`PerModelCapability`][pydantic_ai.capabilities.PerModelCapability] to route to the right compaction implementation based on the model:

```python {title="compaction_routed.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import PerModelCapability
from pydantic_ai.models.openai import OpenAICompaction
from pydantic_ai.models.anthropic import AnthropicCompaction

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[PerModelCapability(routes={
        'openai': OpenAICompaction(message_count_threshold=10),
        'anthropic': AnthropicCompaction(token_threshold=100_000),
    })],
)
```

### ThreadExecutor

The [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] capability provides a custom [`Executor`][concurrent.futures.Executor] for running sync tool functions and other sync callbacks in threads. This is useful in long-running servers (e.g. FastAPI) where the default ephemeral threads from [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync] can accumulate under sustained load:

```python {test="skip"}
from concurrent.futures import ThreadPoolExecutor

from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')
agent = Agent('openai:gpt-5.2', capabilities=[ThreadExecutor(executor)])
```

See [Thread executor for long-running servers](tools-advanced.md#thread-executor-for-long-running-servers) for more details.
### Hooks

The [`Hooks`][pydantic_ai.capabilities.Hooks] capability provides decorator-based [lifecycle hook](#hooking-into-the-lifecycle) registration — the easiest way to intercept model requests, tool calls, and other events without subclassing [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability]:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
    agent_name = ctx.agent.name if ctx.agent else 'unknown'
    print(f'[{agent_name}] Sending {len(request_context.messages)} messages')
    return request_context

agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[hooks])
```

All hooks receive [`RunContext`][pydantic_ai.tools.RunContext], which provides access to the running agent via [`ctx.agent`][pydantic_ai.tools.RunContext.agent] — useful for logging, metrics, and other cross-cutting concerns that need to identify which agent is running.

See the dedicated [Hooks](hooks.md) page for the full API: decorator and constructor registration, timeouts, tool filtering, wrap hooks, per-event hooks, and more.

### Provider-adaptive tools

[`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], and [`MCP`][pydantic_ai.capabilities.MCP] provide model-agnostic access to common tool types. When the model supports the tool natively (as a [builtin tool](builtin-tools.md)), it's used directly. When it doesn't, a local function tool handles it instead — so your agent works across providers without code changes.

Each accepts `builtin` and `local` keyword arguments to control which side is used:

```python {title="provider_adaptive_tools.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, ImageGeneration, WebFetch, WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        # Auto-detects DuckDuckGo as local fallback
        WebSearch(),
        # Auto-detects markdownify-based fetch as local fallback
        WebFetch(),
        # Builtin when the model supports it; falls back to a subagent
        # running the specified LLM with image generation support
        ImageGeneration(fallback_model='openai-responses:gpt-5.4'),
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

Some constraint fields require the builtin because the local fallback can't enforce them. When these are set and the model doesn't support the builtin, a [`UserError`][pydantic_ai.exceptions.UserError] is raised. For example, [`WebSearch`][pydantic_ai.capabilities.WebSearch] domain constraints require the builtin, while [`WebFetch`][pydantic_ai.capabilities.WebFetch] enforces them locally:

```python {title="constraints.py" test="skip" lint="skip"}
# Only search example.com — requires builtin support
WebSearch(allowed_domains=['example.com'])

# Only fetch example.com — enforced locally when builtin is unavailable
WebFetch(allowed_domains=['example.com'])
```

All of these capabilities are subclasses of [`BuiltinOrLocalTool`][pydantic_ai.capabilities.BuiltinOrLocalTool], which you can use directly or subclass to build your own provider-adaptive tools. For example, to pair [`CodeExecutionTool`][pydantic_ai.builtin_tools.CodeExecutionTool] with a local fallback:

```python {title="custom_builtin_or_local.py" test="skip" lint="skip"}
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.capabilities import BuiltinOrLocalTool

cap = BuiltinOrLocalTool(builtin=CodeExecutionTool(), local=my_local_executor)
```

### PrepareTools

[`PrepareTools`][pydantic_ai.capabilities.PrepareTools] wraps a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a capability, for filtering or modifying [tool definitions](tools.md) per step:

```python {title="prepare_tools_builtin.py"}
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import PrepareTools


async def hide_dangerous(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [td for td in tool_defs if not td.name.startswith('delete_')]


agent = Agent('openai:gpt-5.2', capabilities=[PrepareTools(hide_dangerous)])


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

For more complex tool preparation logic, see [Tool preparation](#tool-preparation) under lifecycle hooks.

### PrefixTools

[`PrefixTools`][pydantic_ai.capabilities.PrefixTools] wraps another capability and prefixes all of its tool names, useful for namespacing when composing multiple capabilities that might have conflicting tool names:

```python {title="prefix_tools_example.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, PrefixTools

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        PrefixTools(MCP(url='https://api1.example.com'), prefix='api1'),
        PrefixTools(MCP(url='https://api2.example.com'), prefix='api2'),
    ],
)
```

Every [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] has a convenience method [`prefix_tools`][pydantic_ai.capabilities.AbstractCapability.prefix_tools] that returns a [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] wrapper:

```python {title="prefix_convenience.py" test="skip" lint="skip"}
MCP(url='https://mcp.example.com/api').prefix_tools('mcp')
```

## Building custom capabilities

To build your own capability, subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] and override the methods you need. There are two categories: **configuration methods** that are called at agent construction (except [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] which is called per-run), and **lifecycle hooks** that fire during each run.

### Providing tools

A capability that provides tools returns a [toolset](toolsets.md) from [`get_toolset`][pydantic_ai.capabilities.AbstractCapability.get_toolset]. This can be a pre-built [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] instance, or a callable that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns one dynamically:

```python {title="custom_capability_tools.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

math_toolset = FunctionToolset()


@math_toolset.tool_plain
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@math_toolset.tool_plain
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@dataclass
class MathTools(AbstractCapability[Any]):
    """Provides basic math operations."""

    def get_toolset(self) -> AgentToolset[Any] | None:
        return math_toolset


agent = Agent('openai:gpt-5.2', capabilities=[MathTools()])
result = agent.run_sync('What is 2 + 3?')
print(result.output)
#> The answer is 5.0
```

For [builtin tools](builtin-tools.md), override [`get_builtin_tools`][pydantic_ai.capabilities.AbstractCapability.get_builtin_tools] to return a sequence of [`AgentBuiltinTool`][pydantic_ai.tools.AgentBuiltinTool] instances (which includes both [`AbstractBuiltinTool`][pydantic_ai.builtin_tools.AbstractBuiltinTool] objects and callables that receive [`RunContext`][pydantic_ai.tools.RunContext]).

#### Toolset wrapping

[`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] lets a capability wrap the agent's entire assembled toolset with a [`WrapperToolset`](toolsets.md#changing-tool-execution). This is more powerful than providing tools — it can intercept tool execution, add logging, or apply cross-cutting behavior.

The wrapper receives the combined non-output toolset (after any agent-level [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc] wrapping). Output tools are added separately and are not affected.

```python {title="wrapper_toolset_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset


@dataclass
class LoggingToolset(WrapperToolset[Any]):
    """Logs all tool calls."""

    async def call_tool(
        self, tool_name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        print(f'  Calling tool: {tool_name}')
        return await super().call_tool(tool_name, tool_args, *args, **kwargs)


@dataclass
class LogToolCalls(AbstractCapability[Any]):
    """Wraps the agent's toolset to log all tool calls."""

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        return LoggingToolset(wrapped=toolset)


agent = Agent('openai:gpt-5.2', capabilities=[LogToolCalls()])


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'


result = agent.run_sync('hello')
# Tool calls are logged as they happen
```

!!! note
    `get_wrapper_toolset` wraps the non-output *toolset* once per run (during toolset assembly), intercepting tool execution. This is different from the [`prepare_tools`](#tool-preparation) hook, which operates on tool *definitions* per step and controls visibility rather than execution.

### Providing instructions

[`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions] adds [instructions](agent.md#instructions) to the agent. Since it's called once at agent construction, return a callable if you need dynamic values:

```python {title="custom_capability_config.py"}
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """Tells the agent what time it is."""

    def get_instructions(self):
        def _get_time(ctx: RunContext[Any]) -> str:
            return f'The current date and time is {datetime.now().isoformat()}.'

        return _get_time


agent = Agent('openai:gpt-5.2', capabilities=[KnowsCurrentTime()])
result = agent.run_sync('What time is it?')
print(result.output)
#> The current time is 3:45 PM.
```

Instructions can also use [template strings](agent-spec.md#template-strings) (`TemplateStr('Hello {{name}}')`) for Handlebars-style templates rendered against the agent's [dependencies](dependencies.md). In Python code, a callable with [`RunContext`][pydantic_ai.tools.RunContext] is generally preferred for IDE autocomplete.

### Providing model settings

[`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] returns [model settings](agent.md#model-run-settings) as a dict or a callable for per-step settings.

When model settings need to vary per step — for example, enabling thinking only on retry — return a callable:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class ThinkingOnRetry(AbstractCapability[None]):
    """Enables thinking mode when the agent is retrying."""

    def get_model_settings(self):
        def resolve(ctx: RunContext[None]) -> ModelSettings:
            if ctx.run_step > 1:
                return ModelSettings(thinking='high')
            return ModelSettings()

        return resolve


agent = Agent('openai:gpt-5.2', capabilities=[ThinkingOnRetry()])
result = agent.run_sync('hello')
print(result.output)
#> Hello! How can I help you today?
```

The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] where `ctx.model_settings` contains the merged result of all layers resolved before this capability (model defaults and agent-level settings).

### Configuration methods reference

| Method | Return type | Purpose |
|---|---|---|
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AgentToolset`][pydantic_ai.toolsets.AgentToolset] ` \| None` | A [toolset](toolsets.md) to register (or a callable for [dynamic toolsets](toolsets.md#dynamically-building-a-toolset)) |
| [`get_builtin_tools()`][pydantic_ai.capabilities.AbstractCapability.get_builtin_tools] | `Sequence[`[`AgentBuiltinTool`][pydantic_ai.tools.AgentBuiltinTool]`]` | [Builtin tools](builtin-tools.md) to register (including callables) |
| [`get_wrapper_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] | [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] ` \| None` | [Wrap the agent's assembled toolset](#toolset-wrapping) |
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`AgentInstructions`][pydantic_ai._instructions.AgentInstructions] ` \| None` | [Instructions](agent.md#instructions) (static strings, [template strings](agent-spec.md#template-strings), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | [Model settings](agent.md#model-run-settings) dict, or a callable for per-step settings |

### Hooking into the lifecycle

Capabilities can hook into five lifecycle points, each with up to four variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action succeeds (in reverse capability order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it
* **`on_*_error`** — fires when the action fails (after `wrap_*` has had its chance to recover), can observe, transform, or recover from errors

!!! tip
    For quick, application-level hooks without subclassing, use the [`Hooks`](hooks.md) capability instead.

#### Run hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_run`][pydantic_ai.capabilities.AbstractCapability.before_run] | `(ctx: RunContext) -> None` | Observe-only notification that a run is starting |
| [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run] | `(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult` | Modify the final result |
| [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] | `(ctx: RunContext, *, handler: WrapRunHandler) -> AgentRunResult` | Wrap the entire run |
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | `(ctx: RunContext, *, error: BaseException) -> AgentRunResult` | Handle run errors (see [error hooks](#error-hooks)) |

`wrap_run` supports error recovery: if `handler()` raises and `wrap_run` catches the exception and returns a result instead, the error is suppressed and the recovery result is used. This works with both [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] and [`agent.iter()`][pydantic_ai.agent.Agent.iter].

#### Node hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_node_run`][pydantic_ai.capabilities.AbstractCapability.before_node_run] | `(ctx: RunContext, *, node: AgentNode) -> AgentNode` | Observe or replace the node before execution |
| [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run] | `(ctx: RunContext, *, node: AgentNode, result: NodeResult) -> NodeResult` | Modify the result (next node or `End`) |
| [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] | `(ctx: RunContext, *, node: AgentNode, handler: WrapNodeRunHandler) -> NodeResult` | Wrap each graph node execution |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | `(ctx: RunContext, *, node: AgentNode, error: BaseException) -> NodeResult` | Handle node errors (see [error hooks](#error-hooks)) |

[`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] fires for every node in the [agent graph](agent.md#iterating-over-an-agents-graph) ([`UserPromptNode`][pydantic_ai.UserPromptNode], [`ModelRequestNode`][pydantic_ai.ModelRequestNode], [`CallToolsNode`][pydantic_ai.CallToolsNode]). Override this to observe node transitions, add per-step logging, or modify graph progression:

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

    nodes: list[str] = field(default_factory=list)

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

#### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] | `(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext` | Modify messages, settings, parameters, or model before the model call |
| [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse` | Modify the model's response |
| [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, handler: WrapModelRequestHandler) -> ModelResponse` | Wrap the model call |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | `(ctx: RunContext, *, request_context: ModelRequestContext, error: Exception) -> ModelResponse` | Handle model request errors (see [error hooks](#error-hooks)) |

[`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] bundles `model`, `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof. To swap the model for a given request, set `request_context.model` to a different [`Model`][pydantic_ai.models.Model] instance.

To skip the model call entirely and provide a replacement response, raise [`SkipModelRequest(response)`][pydantic_ai.exceptions.SkipModelRequest] from `before_model_request` or `wrap_model_request`.

#### Tool hooks

Tool processing has two phases: **validation** (parsing and validating the model's JSON arguments against the tool's schema) and **execution** (running the tool function). Each phase has its own hooks.

All tool hooks receive a `tool_def` parameter with the [`ToolDefinition`][pydantic_ai.tools.ToolDefinition].

**Validation hooks** — `args` is the raw `str | dict[str, Any]` from the model before validation, or the validated `dict[str, Any]` after:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_validate`][pydantic_ai.capabilities.AbstractCapability.before_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs) -> RawToolArgs` | Modify raw args before validation (e.g. JSON repair) |
| [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs` | Modify validated args |
| [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, handler: WrapToolValidateHandler) -> ValidatedToolArgs` | Wrap the validation step |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, error: Exception) -> ValidatedToolArgs` | Handle validation errors (see [error hooks](#error-hooks)) |

To skip validation and provide pre-validated args, raise [`SkipToolValidation(args)`][pydantic_ai.exceptions.SkipToolValidation] from `before_tool_validate` or `wrap_tool_validate`.

**Execution hooks** — `args` is always the validated `dict[str, Any]`:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs` | Modify args before execution |
| [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: Any) -> Any` | Modify execution result |
| [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, handler: WrapToolExecuteHandler) -> Any` | Wrap execution |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, error: Exception) -> Any` | Handle execution errors (see [error hooks](#error-hooks)) |

To skip execution and provide a replacement result, raise [`SkipToolExecution(result)`][pydantic_ai.exceptions.SkipToolExecution] from `before_tool_execute` or `wrap_tool_execute`.

#### Tool preparation

Capabilities can filter or modify which tool definitions the model sees on each step via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools]. This controls tool **visibility**, not execution — use execution hooks for that.

```python {title="prepare_tools_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability


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

#### Event stream hook

For runs with event streaming ([`run_stream_events`][pydantic_ai.agent.AbstractAgent.run_stream_events], [`event_stream_handler`][pydantic_ai.agent.Agent.__init__], [UI event streams](ui/overview.md)), capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream] | `(ctx: RunContext, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

```python {title="event_stream_example.py" test="skip"}
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
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

#### Error hooks

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

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart


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

### Wrapping capabilities

[`WrapperCapability`][pydantic_ai.capabilities.WrapperCapability] wraps another capability and delegates all methods to it — similar to [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] for toolsets. Subclass it to override specific methods while delegating the rest:

```python {title="wrapper_capability_example.py" test="skip" lint="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import WrapperCapability


@dataclass
class AuditedCapability(WrapperCapability[Any]):
    """Wraps any capability and logs its model requests."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        print(f'Request from {type(self.wrapped).__name__}')
        return await super().before_model_request(ctx, request_context)
```

The built-in [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] is an example of a `WrapperCapability` — it wraps another capability and prefixes its tool names.

### Per-run state isolation

By default, a capability instance is shared across all runs of an agent. If your capability accumulates mutable state that should not leak between runs, override [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] to return a fresh instance:

```python {title="per_run_state.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability


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

### Composition

When multiple capabilities are passed to an agent, they are composed into a single [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability]:

* **Configuration** is merged: instructions concatenate, model settings merge additively (later capabilities override earlier ones), toolsets combine, builtin tools collect.
* **`before_*`** hooks fire in capability order: `cap1 → cap2 → cap3`.
* **`after_*`** hooks fire in reverse order: `cap3 → cap2 → cap1`.
* **`wrap_*`** hooks nest as middleware: `cap1` wraps `cap2` wraps `cap3` wraps the actual operation. The first capability is the outermost layer.

This means the first capability in the list has the first and last say on the operation — it sees the original input in its `wrap_*` before handler, and it sees the final output after handler returns.

## Examples

### Guardrail (PII redaction)

A guardrail is a capability that intercepts model requests or responses to enforce safety rules. Here's one that scans model responses for potential PII and redacts it:

```python {title="guardrail_example.py"}
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart


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

### Logging middleware

The `wrap_*` pattern is useful when you need to observe or time both the input and output of an operation. Here's a capability that logs every model request and tool call:

```python {title="logging_middleware_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart


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

## Third-party capabilities

Capabilities are the recommended way for third-party packages to extend Pydantic AI, since they can bundle tools with hooks, instructions, and model settings. See [Extensibility](extensibility.md) for the full ecosystem, including [third-party toolsets](toolsets.md#third-party-toolsets) that can also be wrapped as capabilities.

### Task Management

Capabilities for task planning and progress tracking help agents organize complex work:

* [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) - `TodoCapability` with `add_todo`, `read_todos`, `write_todos`, `update_todo_status`, and `remove_todo` tools. Supports subtasks, dependencies, and PostgreSQL persistence. Also available as a lower-level `TodoToolset`.

### Context Management

Capabilities for managing long conversations help agents stay within context limits:

* [`summarization-pydantic-ai`](https://github.com/vstorm-co/summarization-pydantic-ai) - Four capabilities for managing long conversations: `ContextManagerCapability` (real-time token tracking, auto-compression at a configurable threshold, and large tool-output truncation); `SummarizationCapability` (LLM-powered history compression); `SlidingWindowCapability` (zero-cost message trimming); `LimitWarnerCapability` (injects a finish-soon hint before hard context limits). Also available as standalone `history_processors`: `SummarizationProcessor`, `SlidingWindowProcessor`, and `LimitWarnerProcessor`.

### Multi-Agent Orchestration

Capabilities for spawning and delegating to specialized subagents help agents tackle complex, parallelizable work:

* [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai) - `SubAgentCapability` adds tools for multi-agent delegation: `task` (spawn a subagent), `check_task`, `wait_tasks`, `list_active_tasks`, `soft_cancel_task`, `hard_cancel_task`, and `answer_subagent`. Supports sync, async, and auto execution modes, nested subagents, and runtime agent creation. Also available as a lower-level toolset via `create_subagent_toolset`.

### Guardrails & Safety

Capabilities for cost control, input/output filtering, and tool permissions help keep agents safe and within budget:

* [`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) - Ready-to-use guardrail capabilities: `CostTracking` (tracks token usage and USD cost per run, raises `BudgetExceededError` on budget overrun); `ToolGuard` (block or require approval for specific tools); `InputGuard` and `OutputGuard` (custom sync or async validation functions); `PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, and `NoRefusals` content shields.

### File Operations & Sandboxing

Capabilities for filesystem access and sandboxed code execution help agents work with files and run code safely:

* [`pydantic-ai-backend`](https://github.com/vstorm-co/pydantic-ai-backend) - `ConsoleCapability` registers `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute` tools with a fine-grained permission system. Backends include `StateBackend` (in-memory, for testing), `LocalBackend` (real filesystem), `DockerSandbox` (isolated container execution), and `CompositeBackend` (routing across backends). Also available as a lower-level `ConsoleToolset`.

### Agent Skills

Capabilities that implement [Agent Skills](https://agentskills.io) support help agents efficiently discover and perform specific tasks:

* [`pydantic-ai-skills`](https://github.com/DougTrajano/pydantic-ai-skills) - `SkillsCapability` implements Agent Skills support with progressive disclosure (load skills on-demand to reduce tokens). Supports filesystem and programmatic skills; compatible with [agentskills.io](https://agentskills.io).

To add your package to this page, open a pull request.

## Publishing capabilities

To make a custom capability usable in [agent specs](agent-spec.md), it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

```python {title="custom_spec_capability.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, AgentSpec
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

Users register custom capability types via the `custom_capability_types` parameter on [`Agent.from_spec`][pydantic_ai.Agent.from_spec] or [`Agent.from_file`][pydantic_ai.Agent.from_file].

Override [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] when the constructor takes types that can't be represented in YAML/JSON. The spec fields should mirror the dataclass fields, but with serializable types:

```python {title="from_spec_override_example.py" test="skip" lint="skip"}
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class ConditionalTools(AbstractCapability[Any]):
    """Hides tools unless a condition is met."""

    condition: Callable[[RunContext[Any]], bool]  # not serializable
    hidden_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, hidden_tools: list[str]) -> 'ConditionalTools[Any]':
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

See [Extensibility](extensibility.md) for packaging conventions and the broader extension ecosystem.
