
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
| [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] | Filters or modifies tool definitions per step |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] |
| [`HistoryProcessor`][pydantic_ai.capabilities.HistoryProcessor] | Wraps a [history processor](message-history.md) |

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

The simplest capabilities just provide static configuration. Here's a `KnowsCurrentTime` capability that injects the current time into the system prompt:

```python {title="custom_capability_config.py"}
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """Tells the agent what time it is."""

    def get_instructions(self):
        return f'The current date and time is {datetime.now().isoformat()}.'


agent = Agent(TestModel(), capabilities=[KnowsCurrentTime()])
result = agent.run_sync('What time is it?')
print(result.output)
#> success (no tool calls)
```

A capability that provides tools can return a pre-built [toolset](toolsets.md) from [`get_toolset`][pydantic_ai.capabilities.AbstractCapability.get_toolset]:

```python {title="custom_capability_tools.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
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


agent = Agent(TestModel(), capabilities=[MathTools()])
result = agent.run_sync('What is 2 + 3?')
print(result.output)
#> {"_add":0.0,"_multiply":0.0}
```

The full set of configuration methods:

| Method | Return type | Purpose |
|---|---|---|
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`AgentInstructions`][pydantic_ai._instructions.AgentInstructions] ` \| None` | System prompt additions (static strings, [template strings](#template-instructions), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | Model settings dict, or a callable for [per-step settings](#dynamic-model-settings) |
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AgentToolset`][pydantic_ai.toolsets.AgentToolset] ` \| None` | A [toolset](toolsets.md) to register with the agent |
| [`get_builtin_tools()`][pydantic_ai.capabilities.AbstractCapability.get_builtin_tools] | `Sequence[AbstractBuiltinTool]` | [Builtin tools](builtin-tools.md) to register |

### Template instructions

Instructions can use Handlebars-style templates that are rendered against the agent's [dependencies](dependencies.md) at runtime:

```python {title="template_instructions.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, TemplateStr
from pydantic_ai.capabilities import Instructions
from pydantic_ai.models.test import TestModel


@dataclass
class UserProfile:
    name: str
    role: str


agent = Agent(
    TestModel(),
    deps_type=UserProfile,
    capabilities=[
        Instructions(TemplateStr('You are assisting {{name}}, who is a {{role}}.')),
    ],
)
result = agent.run_sync('hello', deps=UserProfile(name='Alice', role='engineer'))
print(result.output)
#> success (no tool calls)
```

Template strings are automatically detected (by the presence of `{{`) when loading from [spec files](#agent-specs).

### Dynamic model settings

When model settings need to vary per step — for example, enabling thinking only on retry — return a callable from [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings]:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
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


agent = Agent(TestModel(), capabilities=[ThinkingOnRetry()])
result = agent.run_sync('hello')
print(result.output)
#> success (no tool calls)
```

The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] where `ctx.model_settings` contains the merged result of all layers resolved before this capability (model defaults and agent-level settings).

## Lifecycle hooks

Capabilities can hook into four lifecycle points, each with up to three variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action (in reverse capability order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it

### Short-circuit exceptions

Before diving into the individual hooks, it's worth knowing about three exceptions that allow `before_*` and `wrap_*` hooks to short-circuit the normal flow:

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

### Node hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] | `(ctx: RunContext, *, node: AgentNode, handler: (AgentNode) -> AgentNode \| End) -> AgentNode \| End` | Wrap each graph node execution |

The [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] hook fires for every node in the [agent graph](agent.md#iterating-over-an-agents-graph) ([`UserPromptNode`][pydantic_ai._agent_graph.UserPromptNode], [`ModelRequestNode`][pydantic_ai._agent_graph.ModelRequestNode], [`CallToolsNode`][pydantic_ai._agent_graph.CallToolsNode]). The `handler` executes the node and returns the next node (or [`End`][pydantic_graph.nodes.End]). Override this to observe node transitions, add per-step logging, or modify graph progression:

```python {title="node_logging_example.py"}
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


@dataclass
class NodeLogger(AbstractCapability[Any]):
    """Logs each node that executes during a run."""

    nodes: list[str] = field(default_factory=lambda: [])

    async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
        self.nodes.append(type(node).__name__)
        return await handler(node)


logger = NodeLogger()
agent = Agent(TestModel(), capabilities=[logger])
agent.run_sync('hello')
print(logger.nodes)
#> ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']
```

### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] | `(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext` | Modify messages, settings, or parameters before the model call |
| [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] | `(ctx: RunContext, *, response: ModelResponse) -> ModelResponse` | Modify the model's response |
| [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, handler: (ModelRequestContext) -> ModelResponse) -> ModelResponse` | Wrap the model call |

[`ModelRequestContext`][pydantic_ai.capabilities.ModelRequestContext] bundles `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof.

### Tool hooks

Tool processing has two phases: **validation** (parsing and validating the model's JSON arguments against the tool's schema) and **execution** (running the tool function). Each phase has its own hooks:

**Validation hooks** — `args` is the raw `str | dict[str, Any]` from the model before validation, or the validated `dict[str, Any]` after:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_validate`][pydantic_ai.capabilities.AbstractCapability.before_tool_validate] | `(ctx, *, call: ToolCallPart, args: str \| dict) -> str \| dict` | Modify raw args before validation (e.g. JSON repair) |
| [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate] | `(ctx, *, call: ToolCallPart, args: dict) -> dict` | Modify validated args |
| [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate] | `(ctx, *, call: ToolCallPart, args: str \| dict, handler) -> dict` | Wrap the validation step |

**Execution hooks** — `args` is always the validated `dict[str, Any]`:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict) -> dict` | Modify args before execution |
| [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict, result: Any) -> Any` | Modify execution result |
| [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute] | `(ctx, *, call: ToolCallPart, args: dict, handler) -> Any` | Wrap execution |

### Tool preparation

Capabilities can filter or modify which tool definitions the model sees on each step via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools]. This controls tool **visibility**, not execution — use execution hooks for that.

```python {title="prepare_tools_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


@dataclass
class HideDangerousTools(AbstractCapability[Any]):
    """Hides tools matching certain name prefixes from the model."""

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

The list includes all tool kinds (function, output, unapproved) — use `tool_def.kind` to distinguish. This hook runs after the agent-level [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc]. For simple cases, the built-in [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] capability wraps a callable without needing a custom subclass.

### Event stream hook

For streamed runs ([`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream], [UI event streams](ui/overview.md)), capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream] | `(ctx, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

```python {title="event_stream_example.py" test="skip"}
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, TextPart


@dataclass
class StreamLogger(AbstractCapability[Any]):
    """Logs text parts as they stream."""

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        async def _wrap():
            async for event in stream:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    print(f'Streaming text: {event.part.content!r}')
                yield event

        return _wrap()
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
from pydantic_ai.models.test import TestModel


@dataclass
class PIIRedactionGuardrail(AbstractCapability[Any]):
    """Redacts email addresses and phone numbers from model responses."""

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
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


agent = Agent(TestModel(), capabilities=[PIIRedactionGuardrail()])
result = agent.run_sync('hello')
print(result.output)
#> success (no tool calls)
```

## Example: building a logging middleware

The `wrap_*` pattern is useful when you need to observe or time both the input and output of an operation. Here's a capability that logs every model request and tool call:

```python {title="logging_middleware_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import ModelRequestContext
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.test import TestModel


@dataclass
class VerboseLogging(AbstractCapability[Any]):
    """Logs model requests and tool executions."""

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Any,
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
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        print(f'  Tool call: {call.tool_name}({args})')
        result = await handler(args)
        print(f'  Tool result: {result!r}')
        return result


agent = Agent(TestModel(), capabilities=[VerboseLogging()])
result = agent.run_sync('hello')
print(f'Output: {result.output}')
#> Output: success (no tool calls)
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

```python {title="per_run_state.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import ModelRequestContext
from pydantic_ai.models.test import TestModel


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
agent = Agent(TestModel(), capabilities=[counter])

# The shared counter stays at 0 because for_run returns a fresh instance
agent.run_sync('first run')
agent.run_sync('second run')
print(counter.count)
#> 0
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

To make a custom capability work with specs, it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

```python {title="custom_spec_capability.py" test="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class RateLimit(AbstractCapability[Any]):
    """Limits requests per minute."""

    rpm: int = 60


# In YAML: `- RateLimit: {rpm: 30}`
# In Python:
agent = Agent.from_spec(
    {'model': 'test', 'capabilities': [{'RateLimit': {'rpm': 30}}]},
    custom_capability_types=[RateLimit],
)
```

Override [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] only when the constructor takes non-serializable types (like callables) or when you need to transform the spec arguments into a different constructor signature.

Pass custom capability types via the `custom_capability_types` parameter so the spec resolver can find them.

### `AgentSpec`

The [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] model represents the full spec structure. Beyond capabilities, it supports:

| Field | Type | Description |
|---|---|---|
| `model` | `str` | Model name (required) |
| `name` | `str \| None` | Agent name |
| `description` | [`TemplateStr`][pydantic_ai.TemplateStr] ` \| str \| None` | Agent description (supports templates) |
| `instructions` | [`TemplateStr`][pydantic_ai.TemplateStr] ` \| str \| list \| None` | System prompt instructions (supports templates) |
| `model_settings` | `dict \| None` | Model settings |
| `capabilities` | `list[CapabilitySpec]` | Capabilities |
| `retries` | `int` | Default tool retries |
| `output_retries` | `int \| None` | Output validation retries |
| `end_strategy` | `EndStrategy` | When to stop (`'early'` or `'exhaustive'`) |
| `tool_timeout` | `float \| None` | Default tool timeout in seconds |
| `instrument` | `bool \| None` | Enable [Logfire](logfire.md) instrumentation |
| `metadata` | `dict \| None` | Agent metadata |

Specs can be loaded from files with [`AgentSpec.from_file`][pydantic_ai.agent.spec.AgentSpec.from_file] and saved with [`AgentSpec.to_file`][pydantic_ai.agent.spec.AgentSpec.to_file], which also generates a JSON schema for editor autocompletion.
