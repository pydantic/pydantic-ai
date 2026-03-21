
# Capabilities

A capability is a reusable, composable unit of agent behavior. Instead of threading multiple arguments through your `Agent` constructor — instructions here, model settings there, a toolset somewhere else, a history processor on yet another parameter — you can bundle related behavior into a single capability and pass it via the [`capabilities`][pydantic_ai.agent.Agent.__init__] parameter.

Capabilities can provide any combination of:

* **Instructions** — static or dynamic system prompt additions
* **Model settings** — static or per-step configuration
* **Tools** — via toolsets or builtin tools
* **Lifecycle hooks** — intercept and modify model requests, tool calls, and the overall run

This makes them the primary extension point for Pydantic AI. Whether you're building a memory system, a guardrail, a cost tracker, or an approval workflow, a capability is the right abstraction.

## Using built-in capabilities

Pydantic AI ships with several capabilities that cover common needs:

| Capability | What it provides |
|---|---|
| [`Instructions`][pydantic_ai.capabilities.Instructions] | Static or template-based system prompt instructions |
| [`ModelSettings`][pydantic_ai.capabilities.ModelSettings] | Static or dynamic model settings |
| [`Thinking`][pydantic_ai.capabilities.Thinking] | Enables model thinking/reasoning mode |
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | Registers the web search [builtin tool](builtin-tools.md) |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] |
| [`HistoryProcessorCapability`][pydantic_ai.capabilities.HistoryProcessorCapability] | Wraps a [history processor](message-history.md) |

```python {title="builtin_capabilities.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instructions, ModelSettings, Thinking, WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    capabilities=[
        Instructions('You are a research assistant. Be thorough and cite sources.'),
        Thinking(),
        WebSearch(),
        ModelSettings({'max_tokens': 8192}),
    ],
)
```

These are equivalent to passing the same configuration through separate `Agent` parameters, but they compose better — especially when you want to reuse the same configuration across multiple agents, or load it from a [spec file](#agent-specs).

## Building a custom capability

To build your own capability, subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] and override the methods you need. There are two categories: **configuration methods** that are called once at agent construction, and **lifecycle hooks** that fire during each run.

### Providing configuration

The simplest capabilities just provide static configuration:

```python {title="custom_capability_config.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """A capability that tells the agent what time it is."""

    def get_instructions(self):
        from datetime import datetime

        return f'The current date and time is {datetime.now().isoformat()}.'


@dataclass
class MathTools(AbstractCapability[Any]):
    """A capability that provides math tools."""

    def get_toolset(self):
        toolset = FunctionToolset()

        @toolset.tool_plain
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @toolset.tool_plain
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        return toolset


agent = Agent(
    TestModel(),
    capabilities=[
        KnowsCurrentTime(),
        MathTools(),
    ],
)
result = agent.run_sync('What is 2 + 3?')
print(result.output)
#> {"add":0.0,"multiply":0.0}
```

The configuration methods are:

| Method | Return type | Purpose |
|---|---|---|
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`Instructions`][pydantic_ai._instructions.Instructions] ` \| None` | System prompt additions (static strings, [template strings](#template-instructions), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | Model settings dict, or a callable for [per-step settings](#dynamic-model-settings) |
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] ` \| None` | A [toolset](toolsets.md) to register with the agent |
| [`get_builtin_tools()`][pydantic_ai.capabilities.AbstractCapability.get_builtin_tools] | `Sequence[AbstractBuiltinTool]` | [Builtin tools](builtin-tools.md) to register |

### Template instructions

Instructions can use Handlebars-style templates that are rendered against the agent's [dependencies](dependencies.md) at runtime:

```python {title="template_instructions.py" test="skip"}
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai._template import TemplateStr
from pydantic_ai.capabilities import Instructions


@dataclass
class UserProfile:
    name: str
    role: str


agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    deps_type=UserProfile,
    capabilities=[
        Instructions(TemplateStr('You are assisting {{name}}, who is a {{role}}.')),
    ],
)
```

Template strings are automatically detected (by the presence of `{{`) when loading from [spec files](#agent-specs).

### Dynamic model settings

When model settings need to vary per step, return a callable from `get_model_settings()`:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings


@dataclass
class AdaptiveTokenLimit(AbstractCapability[None]):
    """Increases max_tokens on retries."""

    base_tokens: int = 4096

    def get_model_settings(self):
        base = self.base_tokens

        def resolve(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=base * (ctx.run_step + 1))

        return resolve


agent = Agent(TestModel(), capabilities=[AdaptiveTokenLimit()])
result = agent.run_sync('hello')
print(result.output)
#> success (no tool calls)
```

The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] where `ctx.model_settings` contains the merged result of all layers resolved before this capability (model defaults and agent-level settings).

## Lifecycle hooks

Capabilities can hook into four lifecycle points, each with three variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action (in reverse order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it

### Run hooks

| Hook | Signature | Purpose |
|---|---|---|
| `before_run` | `(ctx) -> None` | Observe-only notification that a run is starting |
| `after_run` | `(ctx, *, result) -> AgentRunResult` | Modify the final result |
| `wrap_run` | `(ctx, *, handler) -> AgentRunResult` | Wrap the entire run |

### Run step hooks

| Hook | Signature | Purpose |
|---|---|---|
| `wrap_run_step` | `(ctx, *, node, handler) -> AgentNode \| End` | Wrap each graph node execution |

The `wrap_run_step` hook fires for every node in the agent graph (`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`). The `handler` executes the node and returns the next node. Override this to observe node transitions, add logging, or modify graph progression.

### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| `before_model_request` | `(ctx, request_context) -> BeforeModelRequestContext` | Modify messages, settings, or parameters before the model call |
| `after_model_request` | `(ctx, *, response) -> ModelResponse` | Modify the model's response |
| `wrap_model_request` | `(ctx, *, request_context, handler) -> ModelResponse` | Wrap the model call |

[`BeforeModelRequestContext`][pydantic_ai.capabilities.BeforeModelRequestContext] bundles `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof.

### Tool hooks

| Hook | Signature | Purpose |
|---|---|---|
| `before_tool_validate` | `(ctx, *, call, args) -> args` | Modify raw args before validation (e.g. JSON repair) |
| `after_tool_validate` | `(ctx, *, call, args) -> args` | Modify validated args |
| `wrap_tool_validate` | `(ctx, *, call, args, handler) -> args` | Wrap validation |
| `before_tool_execute` | `(ctx, *, call, args) -> args` | Modify args before execution |
| `after_tool_execute` | `(ctx, *, call, args, result) -> result` | Modify execution result |
| `wrap_tool_execute` | `(ctx, *, call, args, handler) -> result` | Wrap execution |

### Tool preparation

In addition to the lifecycle hooks above, capabilities can filter or modify which tools the model sees on each step via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools]:

```python {title="prepare_tools_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


@dataclass
class HideDangerousTools(AbstractCapability[Any]):
    """Hides tools matching certain names unless explicitly enabled."""

    hidden_prefixes: tuple[str, ...] = ('delete_', 'drop_')

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [td for td in tool_defs if not any(td.name.startswith(p) for p in self.hidden_prefixes)]


agent = Agent(TestModel(), capabilities=[HideDangerousTools()])


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

The list includes all tool kinds (function, output, unapproved) — use `tool_def.kind` to distinguish. This hook runs after the agent-level [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc].

### Event stream hook

For streamed runs, capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| `wrap_run_event_stream` | `(ctx, *, stream) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

## Example: building a guardrail

A guardrail is a capability that uses hooks to enforce safety rules. Here's one that blocks tool calls to certain tools:

```python {title="guardrail_example.py"}
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


@dataclass
class ToolApprovalGuardrail(AbstractCapability[Any]):
    """Blocks tool calls that aren't in the allowed set."""

    allowed_tools: set[str] = field(default_factory=set)

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if self.allowed_tools and call.tool_name not in self.allowed_tools:
            raise SkipToolExecution(result=f'Tool {call.tool_name!r} is not allowed.')
        return args


def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=f'Tool said: {part.content}')])
    if info.function_tools:
        tool = info.function_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{}', tool_call_id='c1')])
    return ModelResponse(parts=[TextPart(content='no tools')])


agent = Agent(
    FunctionModel(model_fn),
    capabilities=[ToolApprovalGuardrail(allowed_tools={'safe_tool'})],
)


@agent.tool_plain
def dangerous_tool() -> str:
    return 'this should not run'


result = agent.run_sync('do something')
print(result.output)
#> Tool said: Tool 'dangerous_tool' is not allowed.
```

The three `Skip*` exceptions ([`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest], [`SkipToolValidation`][pydantic_ai.exceptions.SkipToolValidation], [`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution]) let `before_*` and `wrap_*` hooks short-circuit the normal flow with a replacement value.

## Example: building a cost tracker

A cost tracker demonstrates the `wrap_*` pattern — wrapping the run to observe the outcome and accumulating state across model requests:

```python {title="cost_tracker_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult


@dataclass
class CostTracker(AbstractCapability[Any]):
    """Tracks token usage across model requests."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    request_count: int = 0

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        if response.usage:
            self.total_input_tokens += response.usage.input_tokens or 0
            self.total_output_tokens += response.usage.output_tokens or 0
        self.request_count += 1
        return response

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        return result


tracker = CostTracker()
agent = Agent(TestModel(), capabilities=[tracker])
result = agent.run_sync('hello')
print(f'Requests: {tracker.request_count}')
#> Requests: 1
```

## Composition

When multiple capabilities are passed to an agent, they are composed into a single [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability]:

* **Configuration** is merged: instructions concatenate, model settings merge additively (later capabilities override earlier ones), toolsets combine, builtin tools collect.
* **`before_*`** hooks fire in capability order: `cap1 → cap2 → cap3`.
* **`after_*`** hooks fire in reverse order: `cap3 → cap2 → cap1`.
* **`wrap_*`** hooks nest as middleware: `cap1` wraps `cap2` wraps `cap3` wraps the actual operation. The first capability is the outermost layer.

This means the first capability in the list has the first and last say on the operation — it sees the original input in its `wrap_*` before handler, and it sees the final output after handler returns.

## Per-run state isolation

By default, a capability instance is shared across all runs of an agent. If your capability accumulates mutable state that should not leak between runs, override [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] to return a fresh instance:

```python {title="per_run_state.py" test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability


@dataclass
class RequestCounter(AbstractCapability[Any]):
    """Counts model requests per run."""

    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> 'RequestCounter':
        return RequestCounter()  # fresh instance for each run

    async def before_model_request(self, ctx, request_context):
        self.count += 1
        return request_context
```

## Agent specs

Capabilities integrate with the YAML/JSON agent spec system, allowing you to define agents declaratively:

```yaml {title="agent.yaml" test="skip"}
model: anthropic:claude-sonnet-4-20250514
instructions: You are a helpful research assistant.
capabilities:
  - Thinking
  - WebSearch
  - ModelSettings:
      max_tokens: 8192
```

Load it with [`Agent.from_spec`][pydantic_ai.Agent.from_spec]:

```python {title="from_spec_example.py" test="skip"}
from pydantic_ai import Agent

agent = Agent.from_spec('agent.yaml')
```

### Spec syntax

Capabilities in specs support three forms:

* `'MyCapability'` — no arguments, calls `MyCapability.from_spec()`
* `{'MyCapability': value}` — single positional argument, calls `MyCapability.from_spec(value)`
* `{'MyCapability': {key: value, ...}}` — keyword arguments, calls `MyCapability.from_spec(**kwargs)`

### Custom capabilities in specs

To make your custom capability work with specs, ensure it has a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] that accepts serializable arguments:

```python {title="custom_spec_capability.py" test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability


@dataclass
class RateLimit(AbstractCapability[Any]):
    """Limits requests per minute."""

    rpm: int = 60

    @classmethod
    def from_spec(cls, rpm: int = 60) -> 'RateLimit':
        return cls(rpm=rpm)


# In YAML: `- RateLimit: {rpm: 30}`
# In Python:
agent = Agent.from_spec(
    {'model': 'test', 'capabilities': [{'RateLimit': {'rpm': 30}}]},
    custom_capability_types=[RateLimit],
)
```

Pass custom capability types via [`custom_capability_types`][pydantic_ai.Agent.from_spec] so the spec resolver can find them.

### `AgentSpec`

The [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] model represents the full spec structure. Beyond capabilities, it supports:

| Field | Type | Description |
|---|---|---|
| `model` | `str` | Model name (required) |
| `name` | `str \| None` | Agent name |
| `description` | `str \| None` | Agent description (supports templates) |
| `instructions` | `str \| list[str] \| None` | System prompt instructions (supports templates) |
| `model_settings` | `dict \| None` | Model settings |
| `capabilities` | `list[CapabilitySpec]` | Capabilities |
| `retries` | `int` | Default tool retries |
| `output_retries` | `int \| None` | Output validation retries |
| `end_strategy` | `EndStrategy` | When to stop (`'early'` or `'exhaustive'`) |
| `tool_timeout` | `float \| None` | Default tool timeout in seconds |
| `instrument` | `bool \| None` | Enable [Logfire](logfire.md) instrumentation |
| `metadata` | `dict \| None` | Agent metadata |

Specs can be loaded from files with [`AgentSpec.from_file`][pydantic_ai.agent.spec.AgentSpec.from_file] and saved with [`AgentSpec.to_file`][pydantic_ai.agent.spec.AgentSpec.to_file], which also generates a JSON schema for editor autocompletion.
