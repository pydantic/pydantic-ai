# Building Custom Capabilities

To build your own [capability](overview.md), subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] and override the methods you need. There are two categories: **configuration methods** that are called at agent construction (except [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] which is called per-run), and **lifecycle hooks** that fire during each run.

Custom capability classes can be plain classes or dataclasses. The shared metadata attributes — [`id`][pydantic_ai.capabilities.AbstractCapability.id], [`description`][pydantic_ai.capabilities.AbstractCapability.description], and [`defer_loading`][pydantic_ai.capabilities.AbstractCapability.defer_loading] — are optional declarations on the capability object for always-available capabilities. If `id` is omitted there, Pydantic AI derives a run-local id from the class name and disambiguates duplicates within the run. Deferred capabilities require an explicit stable `id`.

```python {title="custom_capability_plain.py"}
from typing import Any

from pydantic_ai.capabilities import AbstractCapability


class MyCapability(AbstractCapability[Any]):
    """A custom capability."""
```

Use a dataclass when you want generated constructor parameters for your own configuration fields, or for the shared metadata fields:

```python {title="custom_capability_dataclass.py"}
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability


@dataclass
class MyCapability(AbstractCapability[None]):
    label: str
```

If you define a custom `__init__`, set only the metadata you want to expose. There is no `super().__init__()` or `__post_init__()` requirement:

```python {title="custom_capability_init.py"}
from pydantic_ai.capabilities import AbstractCapability


class MyCapability(AbstractCapability[None]):
    def __init__(
        self,
        label: str,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.label = label
```

When [`defer_loading=True`](on-demand.md), provide a stable explicit `id`; history replay depends on it, and Pydantic AI rejects deferred capabilities without one. For always-available capabilities, omitting `id` still derives a run-local id from the class name.

## Providing tools

A capability that provides tools returns a [toolset](../toolsets.md) from [`get_toolset`][pydantic_ai.capabilities.AbstractCapability.get_toolset]. This can be a pre-built [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] instance, or a callable that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns one dynamically:

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

For [native tools](../native-tools.md), override [`get_native_tools`][pydantic_ai.capabilities.AbstractCapability.get_native_tools] to return a sequence of [`AgentNativeTool`][pydantic_ai.tools.AgentNativeTool] instances (which includes both [`AbstractNativeTool`][pydantic_ai.native_tools.AbstractNativeTool] objects and callables that receive [`RunContext`][pydantic_ai.tools.RunContext]).

### Toolset wrapping

[`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] lets a capability wrap the agent's entire assembled toolset with a [`WrapperToolset`](../toolsets.md#changing-tool-execution). This is more powerful than providing tools — it can intercept tool execution, add logging, or apply cross-cutting behavior.

The wrapper receives the combined non-output toolset (after the [`prepare_tools`](#tool-preparation) hook has wrapped it). Output tools are added separately and are not affected.

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
    `get_wrapper_toolset` wraps the non-output *toolset* once per run (during toolset assembly). The [`prepare_tools`](#tool-preparation) and [`prepare_output_tools`](#tool-preparation) hooks also flow through `PreparedToolset` wrappers, so all three integrate at the toolset level — `get_wrapper_toolset` runs around `prepare_tools` (it sees the prepared defs), and `prepare_output_tools` wraps the output toolset independently.

## Providing instructions

[`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions] adds [instructions](../agent.md#instructions) to the agent. Since it's called once at agent construction, return a callable if you need dynamic values:

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

Instructions can also use [template strings](../agent-spec.md#template-strings) (`TemplateStr('Hello {{name}}')`) for Handlebars-style templates rendered against the agent's [dependencies](../dependencies.md). In Python code, a callable with [`RunContext`][pydantic_ai.tools.RunContext] is generally preferred for IDE autocomplete.

## Providing model settings

[`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] returns [model settings](../agent.md#model-run-settings) as a dict or a callable for per-step settings.

When model settings need to vary per step — for example, enabling thinking only on retry, or forcing a specific [`tool_choice`](../tools-advanced.md#dynamic-tool-choice-via-capabilities) until a tool has been called — return a callable:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class ThinkingOnRetry(AbstractCapability):
    """Enables thinking mode when the agent is retrying."""

    def get_model_settings(self):
        def resolve(ctx: RunContext) -> ModelSettings:
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

## Selecting the model

Override [`get_model()`][pydantic_ai.capabilities.AbstractCapability.get_model] when model selection is one part of a larger custom capability. Return a [`Model`][pydantic_ai.models.Model], a model ID string, or a sync/async callable taking [`ModelSelectionContext`][pydantic_ai.models.ModelSelectionContext]. This example chooses a model from dependencies on every request step:

```python {title="custom_model_selection.py"}
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import AbstractCapability, ModelSelector


@dataclass
class Deps:
    """Dependencies that influence model selection."""

    task_complexity: Literal['standard', 'complex']


class AdaptiveModel(AbstractCapability[Deps]):
    """Select a model for each request step."""

    def get_model(self) -> ModelSelector[Deps]:
        return self.select_model

    def select_model(self, ctx: ModelSelectionContext[Deps]) -> str:
        return 'openai:gpt-5.6-sol' if ctx.deps.task_complexity == 'complex' else 'openai:gpt-5.6-luna'


agent = Agent(deps_type=Deps, capabilities=[AdaptiveModel()])
```

[`get_model()`][pydantic_ai.capabilities.AbstractCapability.get_model] is a synchronous configuration method, but the [`ModelSelector`][pydantic_ai.capabilities.ModelSelector] it returns may be synchronous or asynchronous. [`ModelSelectionContext`][pydantic_ai.models.ModelSelectionContext] is separate from [`RunContext`][pydantic_ai.tools.RunContext] because a complete run context requires the model currently being selected. It includes dependencies, the request step, message history, and usage. Keep `get_model()` itself cheap; perform I/O in an async selector.

A model or model ID returned directly from `get_model()` is resolved once per run. A selector returned from `get_model()` is evaluated before every logical model request step.

A capability's model slots in below a call-site `run(model=...)` argument and a run-level `spec=` model, and above the agent constructor's model. From highest to lowest priority:

`run()`/`iter()` argument › run `spec=` model › capability `get_model()` › agent constructor.

An [`override(model=...)`][pydantic_ai.agent.AbstractAgent.override] still wins over all of these. An explicit model skips capability selection entirely.

Later model contributions override earlier ones. If [`for_run()`][pydantic_ai.capabilities.AbstractCapability.for_run] leaves the capability unchanged, its bootstrap selection is reused on step one; if it returns a replacement with a different selector, that selector makes a new step-one selection.

Fallback is complementary to selection: return a configured [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] when request failures should be retried on another model.

## Resolving model IDs

Override [`resolve_model_id()`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id] when an application-specific string needs custom provider construction, credentials, or registry lookup. Unlike model selection, resolution is first-wins: capabilities are tried in order, and normal [`infer_model()`][pydantic_ai.models.infer_model] behavior is used only if every resolver returns `None`.

```python {title="custom_model_id_resolution.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelResolutionContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass
class Deps:
    """Per-user provider credentials."""

    openai_api_key: str


class UserModelResolver(AbstractCapability[Deps]):
    """Resolve user-scoped model IDs with per-user credentials."""

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[Deps],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        if not model_id.startswith('user:'):
            return None

        def provider_factory(provider_name: str) -> Provider[Any]:
            if provider_name == 'openai':
                return OpenAIProvider(api_key=ctx.deps.openai_api_key)
            return infer_provider(provider_name)

        return infer_model(model_id.removeprefix('user:'), provider_factory)


agent = Agent('user:openai:gpt-5.6-sol', deps_type=Deps, capabilities=[UserModelResolver()])
```

The constructor ID remains a string through [`for_agent()`][pydantic_ai.capabilities.AbstractCapability.for_agent], so a bound capability can install a resolver without default inference first constructing a provider with different configuration or credentials.

Resolution results are cached by model ID and resolver tree for the duration of one run. If a per-step selector returns the same string again, Pydantic AI reuses the same model, provider, and client rather than invoking the resolver again. To deliberately resolve differently on a later step, select a different ID or return a [`Model`][pydantic_ai.models.Model] instance directly from the selector.

## Model selection lifecycle and limitations

Bootstrap resolution uses the capability tree after `for_agent()` binding but before `for_run()`, because resolving the first model is what makes a full [`RunContext`][pydantic_ai.tools.RunContext] possible. If `for_run()` returns a replacement capability, strings selected for step one or later steps use that replacement's resolver chain. When `for_run()` leaves the capability unchanged, the already-resolved bootstrap model is reused for step one.

Model selection and resolution are eager hooks, so deferred capabilities do not contribute them, even after they are loaded. Run-spec capabilities are known during bootstrap and can supply the first model. A [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc], or another capability whose model is only introduced by `for_run()`, requires an existing bootstrap model because `for_run()` receives a full `RunContext`; it may replace that model starting with step one, but cannot bootstrap a model-less agent. If selecting a new model after loading a deferred capability would be useful for your application, please open an issue describing the desired step and continuation semantics.

Dynamic selection is not currently supported by durable execution capabilities. Durable runs need model IDs registered before execution and must recreate the same selected model during replay or cross-run resumption. Pass an explicit registered model for durable execution. Resuming a suspended provider request in a separate ordinary run likewise requires an explicit model when the previous model came from a selector.

## Configuration methods reference

| Method | Return type | Purpose |
|---|---|---|
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AgentToolset`][pydantic_ai.toolsets.AgentToolset] ` \| None` | A [toolset](../toolsets.md) to register (or a callable for [dynamic toolsets](../toolsets.md#dynamically-building-a-toolset)) |
| [`get_native_tools()`][pydantic_ai.capabilities.AbstractCapability.get_native_tools] | `Sequence[`[`AgentNativeTool`][pydantic_ai.tools.AgentNativeTool]`]` | [Native tools](../native-tools.md) to register (including callables) |
| [`get_wrapper_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] | [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] ` \| None` | [Wrap the agent's assembled toolset](#toolset-wrapping) |
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`AgentInstructions`][pydantic_ai._instructions.AgentInstructions] ` \| None` | [Instructions](../agent.md#instructions) (static strings, [template strings](../agent-spec.md#template-strings), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | [Model settings](../agent.md#model-run-settings) dict, or a callable for per-step settings |
| [`get_model()`][pydantic_ai.capabilities.AbstractCapability.get_model] | [`AgentModel`][pydantic_ai.capabilities.AgentModel] `\| None` | Static or per-step [model selection](#selecting-the-model) |
| [`resolve_model_id()`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id] | [`Model`][pydantic_ai.models.Model] `\| None` | [Resolve a selected model ID](#resolving-model-ids) using the agent and run dependencies |

## Binding to an agent

Override [`for_agent()`][pydantic_ai.capabilities.AbstractCapability.for_agent] when a reusable capability needs to inspect the agent it is attached to. The hook runs once during [`Agent`][pydantic_ai.Agent] construction, after the agent's own model, name, and toolsets are available and before capability contributions are extracted:

```python {title="agent_bound_capability.py"}
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Self

from pydantic_ai import Agent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class AgentIdentity(AbstractCapability[Any]):
    """Add the bound agent's name to its instructions."""

    agent_name: str | None = None

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> Self:
        return replace(self, agent_name=agent.name)

    def get_instructions(self) -> str:
        return f'You are the {self.agent_name} agent.'


identity = AgentIdentity()
support = Agent('openai:gpt-5.2', name='support', capabilities=[identity])
sales = Agent('openai:gpt-5.2', name='sales', capabilities=[identity])
```

`for_agent()` is synchronous because it binds configuration during agent construction, before run dependencies or a lifecycle context exist. Keep it free of I/O; asynchronous run-specific setup belongs in [`for_run()`][pydantic_ai.capabilities.AbstractCapability.for_run].

Return a new bound copy rather than mutating the original when the same capability may be attached to multiple agents. [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] and [`WrapperCapability`][pydantic_ai.capabilities.WrapperCapability] propagate binding to their children, and the bound copy participates in all configuration hooks, including `get_model()` and `resolve_model_id()`.

The parameter is typed as [`AbstractAgent`][pydantic_ai.agent.AbstractAgent] so reusable capabilities depend only on the portable agent interface and remain compatible with custom agent implementations. Runs through a [`WrapperAgent`][pydantic_ai.agent.WrapperAgent] are delegated to its wrapped agent, so Pydantic AI's built-in wrappers do not rebind the capability to the outer wrapper.

`for_agent()` sees the constructor model exactly as the caller supplied it. In particular, a model ID remains a string while binding runs, so a bound capability can introduce `resolve_model_id()` without the default inference first constructing a provider with the wrong configuration or credentials. If the bound capability tree has no resolver and `defer_model_check=False`, normal model inference happens after binding.

Capabilities passed directly to [`run()`][pydantic_ai.agent.AbstractAgent.run] or through a run spec are bound once for that run before bootstrap model selection. A [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] is itself bound before the run; because its returned value is normally an independently reusable capability, that value is also bound before its own `for_run()` is called. In contrast, a specialized run-bound capability returned by an ordinary capability's [`for_run()`][pydantic_ai.capabilities.AbstractCapability.for_run] is not passed through `for_agent()` again.

## Capability lifecycle

Binding hooks establish which capability participates in a run; lifecycle hooks then intercept the work it performs. The high-level order is:

`for_agent()` → bootstrap model selection and resolution → `for_run()` → per-step selection and preparation → model request → tool/output processing → run completion

| Phase | Capability work | What is available |
|---|---|---|
| Agent binding | [`for_agent()`][pydantic_ai.capabilities.AbstractCapability.for_agent] | Agent name, raw constructor model, toolsets, and other constructor configuration; no run dependencies or `RunContext` |
| Run bootstrap | [`get_model()`][pydantic_ai.capabilities.AbstractCapability.get_model], then [`resolve_model_id()`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id] if the selection is a string | Dependencies, message history, usage, and the lower-precedence model through selection/resolution contexts; no complete `RunContext` yet |
| Run binding | [`for_run()`][pydantic_ai.capabilities.AbstractCapability.for_run] | A complete [`RunContext`][pydantic_ai.tools.RunContext] containing the bootstrap model; may return a run-scoped replacement capability |
| Each logical model step | Post-`for_run()` model selection/resolution, model settings, tool preparation, and message preparation | The selected model is installed in `RunContext` before its settings, profile-sensitive tools, and model-specific message preparation are evaluated |
| Model request and response | Model request, tool, output, node, and event-stream [hooks](#hooking-into-the-lifecycle) | The fully prepared request and the live run state appropriate to each hook |
| Run completion | `after_run`, `on_run_error`, and `wrap_run` completion | Final result or error, accumulated messages, and usage |

If `for_run()` returns the original capability, the bootstrap model selection is reused for step one. A replacement capability can select a different model for step one. Continuation polling within one logical step remains pinned to that step's selected model.

## Hooking into the lifecycle

Capabilities can hook into five lifecycle points, each with up to four variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action succeeds (in reverse capability order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it
* **`on_*_error`** — fires when the action fails (after `wrap_*` has had its chance to recover), can observe, transform, or recover from errors

!!! tip
    For quick, application-level hooks without subclassing, use the [`Hooks`](../hooks.md) capability instead.

### Run hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_run`][pydantic_ai.capabilities.AbstractCapability.before_run] | `(ctx: RunContext) -> None` | Observe-only notification that a run is starting |
| [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run] | `(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult` | Modify the final result |
| [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] | `(ctx: RunContext, *, handler: WrapRunHandler) -> AgentRunResult` | Wrap the entire run |
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | `(ctx: RunContext, *, error: BaseException) -> AgentRunResult` | Handle run errors (see [error hooks](#error-hooks)) |

`wrap_run` supports error recovery: if `handler()` raises and `wrap_run` catches the exception and returns a result instead, the error is suppressed and the recovery result is used. This works with both [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] and [`agent.iter()`][pydantic_ai.agent.Agent.iter].

### Node hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_node_run`][pydantic_ai.capabilities.AbstractCapability.before_node_run] | `(ctx: RunContext, *, node: AgentNode) -> AgentNode` | Observe or replace the node before execution |
| [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run] | `(ctx: RunContext, *, node: AgentNode, result: NodeResult) -> NodeResult` | Modify the result (next node or `End`) |
| [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] | `(ctx: RunContext, *, node: AgentNode, handler: WrapNodeRunHandler) -> NodeResult` | Wrap each graph node execution |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | `(ctx: RunContext, *, node: AgentNode, error: Exception) -> NodeResult` | Handle node errors (see [error hooks](#error-hooks)) |

[`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] fires for every node in the [agent graph](../agent.md#iterating-over-an-agents-graph) ([`UserPromptNode`][pydantic_ai.UserPromptNode], [`ModelRequestNode`][pydantic_ai.ModelRequestNode], [`CallToolsNode`][pydantic_ai.CallToolsNode]). Override this to observe node transitions, add per-step logging, or modify graph progression:

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

See [Iterating Over an Agent's Graph](../agent.md#iterating-over-an-agents-graph) for more about the agent graph and its node types.

### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] | `(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext` | Modify messages, settings, parameters, or model before the model call |
| [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse` | Modify the model's response |
| [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, handler: WrapModelRequestHandler) -> ModelResponse` | Wrap the model call |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | `(ctx: RunContext, *, request_context: ModelRequestContext, error: Exception) -> ModelResponse` | Handle model request errors (see [error hooks](#error-hooks)) |

[`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] bundles `model`, `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof. To swap the model for a given request, set `request_context.model` to a different [`Model`][pydantic_ai.models.Model] instance.

To skip the model call entirely and provide a replacement response, raise [`SkipModelRequest(response)`][pydantic_ai.exceptions.SkipModelRequest] from `before_model_request` or `wrap_model_request`.

### Tool hooks

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

### Output hooks

Like tool processing, [output](../output.md) processing has two phases: **validation** (parsing the model's raw output against the output schema) and **processing** (extracting the value and calling any [output function](../output.md#output-functions)). Each phase has its own hooks.

All output hooks receive an `output_context` parameter with [`OutputContext`][pydantic_ai.capabilities.OutputContext] (mode, output type, schema info, and tool call details for [tool output](../output.md#tool-output)).

**Validate hooks** fire only for structured output that requires parsing (prompted, native, tool, union output). They do not fire for plain text or image output. **Process hooks** fire for **all output types** including text, structured, and image output. For [tool output](../output.md#tool-output), only output hooks fire — tool hooks are skipped entirely.

**Validation hooks** — fire for structured output only; `output` is `str` (raw text) or `dict` (tool args):

| Hook | Signature | Purpose |
|---|---|---|
| [`before_output_validate`][pydantic_ai.capabilities.AbstractCapability.before_output_validate] | `(ctx, *, output_context, output: RawOutput) -> RawOutput` | Modify raw output before validation (e.g. JSON repair) |
| [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate] | `(ctx, *, output_context, output: Any) -> Any` | Modify validated output |
| [`wrap_output_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_output_validate] | `(ctx, *, output_context, output: RawOutput, handler) -> Any` | Wrap the validation step |
| [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error] | `(ctx, *, output_context, output: RawOutput, error: ValidationError \| ModelRetry) -> Any` | Handle validation errors (see [error hooks](#error-hooks)) |

**Processing hooks** — fire for all output types; `output` is the validated/raw output. Output validators ([`@agent.output_validator`][pydantic_ai.Agent.output_validator]) run inside the processing pipeline (within `wrap_output_process`), so `after_output_process` sees the fully validated result:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_output_process`][pydantic_ai.capabilities.AbstractCapability.before_output_process] | `(ctx, *, output_context, output: Any) -> Any` | Modify output before processing |
| [`after_output_process`][pydantic_ai.capabilities.AbstractCapability.after_output_process] | `(ctx, *, output_context, output: Any) -> Any` | Modify processed result |
| [`wrap_output_process`][pydantic_ai.capabilities.AbstractCapability.wrap_output_process] | `(ctx, *, output_context, output: Any, handler) -> Any` | Wrap processing |
| [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error] | `(ctx, *, output_context, output: Any, error: Exception) -> Any` | Handle processing errors (see [error hooks](#error-hooks)) |

Output validate and process hooks can raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to try again with a custom message — the same pattern used in [output functions](../output.md#output-functions) and [output validators](../output.md#output-validator-functions). See [Triggering retries with `ModelRetry`](../hooks.md#triggering-retries-with-modelretry) for the full pattern.

### Tool preparation

Capabilities can filter or modify which tool definitions the model sees on each step via two hooks:

- [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] — receives **function** tools only. Use this for filtering or modifications to tools the model can call directly.
- [`prepare_output_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_output_tools] — receives [output tools][pydantic_ai.output.ToolOutput] only, with `ctx.retry`/`ctx.max_retries` reflecting the **output** side of the agent retry budget, matching the [output hook](#output-hooks) lifecycle.

Both hooks operate at the toolset level — the result flows into both the model's request parameters and `ToolManager.tools`, so filtering also blocks tool execution.

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

For simple cases, the built-in [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] / [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] capabilities wrap a callable without a custom subclass.

### Event stream hook

For runs with event streaming ([`run_stream_events`][pydantic_ai.agent.AbstractAgent.run_stream_events], [`event_stream_handler`][pydantic_ai.agent.Agent.__init__], [UI event streams](../ui/overview.md)), capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream] | `(ctx: RunContext, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

```python {title="event_stream_example.py"}
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    PartStartEvent,
    TextPart,
    ToolCallEvent,
    ToolResultEvent,
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
            if isinstance(event, ToolCallEvent):
                print(f'Tool called: {event.part.tool_name}')
            elif isinstance(event, ToolResultEvent):
                print(f'Tool result: {event.part.content!r}')
            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                print(f'Text: {event.part.content!r}')
            yield event
```

Matching against [`ToolCallEvent`][pydantic_ai.messages.ToolCallEvent] and [`ToolResultEvent`][pydantic_ai.messages.ToolResultEvent] handles both function tool calls ([`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] / [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]) and output tool calls ([`OutputToolCallEvent`][pydantic_ai.messages.OutputToolCallEvent] / [`OutputToolResultEvent`][pydantic_ai.messages.OutputToolResultEvent]). Match against the specific subclass when you need to treat them differently. [Deferred tool calls](../deferred-tools.md#observing-deferred-tool-calls-in-a-stream) additionally emit batch-level [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] / [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent].

For building web UIs that transform streamed events into protocol-specific formats (like SSE), see the [UI event streams](../ui/overview.md) documentation and the [`UIEventStream`][pydantic_ai.ui.UIEventStream] base class.

### Error hooks

Each lifecycle point has an `on_*_error` hook — the error counterpart to `after_*`. While `after_*` hooks fire on success, `on_*_error` hooks fire on failure (after `wrap_*` has had its chance to recover):

```text
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
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | Graph node fails | Return next node or [`End`][pydantic_graph.basenode.End] |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | Model request fails | Return [`ModelResponse`][pydantic_ai.messages.ModelResponse] |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | Tool validation fails | Return validated args `dict` |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | Tool execution fails | Return any tool result |
| [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error] | Output validation fails | Return validated output |
| [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error] | Output execution fails | Return any output result |

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

### Deferred tool calls

Capabilities can resolve [deferred tool calls](../deferred-tools.md) — calls that require approval, or that are executed externally — directly from the agent run, without ending the run and waiting for a follow-up:

| Hook | Signature | Purpose |
|---|---|---|
| [`handle_deferred_tool_calls`][pydantic_ai.capabilities.AbstractCapability.handle_deferred_tool_calls] | `(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults \| None` | Resolve some or all pending approval/external calls inline |

Multiple capabilities can each handle a subset: dispatch accumulates results across the chain, passing only the still-unresolved requests to the next capability. Returning `None` (or a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with no entries) declines handling. Anything still unresolved bubbles up as a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output for the caller to handle.

For application code that just needs to plug in a handler, use the dedicated [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] capability — see [Resolving deferred calls with a handler](../deferred-tools.md#resolving-deferred-calls-with-a-handler).

## Wrapping capabilities

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

## Per-run state isolation

After construction-time [`for_agent()`][pydantic_ai.capabilities.AbstractCapability.for_agent] binding, the resulting capability instance is shared across all runs of an agent. If your capability accumulates mutable state that should not leak between runs, override [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] to return a fresh instance:

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

## Dynamically building a capability

Capabilities can be built dynamically ahead of each agent run using a function that takes the agent [`RunContext`][pydantic_ai.tools.RunContext] and returns a capability or `None`. This is useful when the capability — its instructions, model settings, hooks, or contributed toolset — depends on information specific to a run, like its [dependencies](../dependencies.md).

To register a dynamic capability, pass a function that takes [`RunContext`][pydantic_ai.tools.RunContext] to the `capabilities` argument of the [`Agent`][pydantic_ai.Agent] constructor or `agent.run()`. Sync and async functions are both supported. The function is called once per run and the returned capability replaces it for the rest of the run, so its instructions, model settings, toolsets, native tools, and hooks all flow through normally.

```python {title="dynamic_capability.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


@dataclass
class Skill(AbstractCapability[str]):
    """Per-user skill loaded from a database at run time."""

    name: str
    role: Literal['admin', 'guest']

    def get_instructions(self) -> str:
        return f'You can use the {self.name} skill (role: {self.role}).'


# Pretend this comes from a database keyed by user.
SKILLS = {
    'alice': Skill(name='refunds', role='admin'),
    'bob': Skill(name='lookup', role='guest'),
}


def user_skill(ctx: RunContext[str]) -> AbstractCapability[str] | None:
    return SKILLS.get(ctx.deps)


agent = Agent(TestModel(), deps_type=str, capabilities=[user_skill])

result = agent.run_sync('hi', deps='alice')
print(result.all_messages()[0].instructions)
#> You can use the refunds skill (role: admin).
```

_(This example is complete, it can be run "as is")_

To return more than one capability from a single factory, wrap them in a [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability].

!!! note "Durable execution (Temporal, DBOS, Prefect)"

    A dynamic capability whose resolved capability contributes only instructions, model settings, native tools, hooks, or `prepare_tools`/`get_wrapper_toolset` (i.e. no `get_toolset()` of its own) works seamlessly with durable execution — the factory runs in the workflow alongside the rest of the agent loop. This covers the common "load this user's skill from the database and add its instructions" pattern.

    However, dynamic capabilities that contribute their own toolset via `get_toolset()` are not yet supported with durable execution. The toolset is only known at run time, so it bypasses the durable wrapper's construction-time toolset registration and would attempt I/O directly inside the workflow. As a workaround, register the toolsets statically via `Agent(toolsets=[...])` (where they get wrapped properly) and have the dynamic capability reference them indirectly — e.g. via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] to scope which tools are visible per-run, rather than constructing the toolset inside the factory. Full support is tracked in [#5253](https://github.com/pydantic/pydantic-ai/issues/5253).

## Composition and middleware semantics

When multiple capabilities are passed to an agent, they are composed into a single [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] that follows **middleware semantics** — the same pattern used by web frameworks like Django and Starlette:

* **Configuration** is merged: instructions concatenate, model settings merge additively (later capabilities override earlier ones), toolsets combine, native tools collect.
* **`before_*`** hooks fire in capability order (outermost to innermost): `cap1 → cap2 → cap3`.
* **`after_*`** hooks fire in reverse order (innermost to outermost): `cap3 → cap2 → cap1`.
* **`wrap_*`** hooks nest as middleware: `cap1` wraps `cap2` wraps `cap3` wraps the actual operation. The first capability is the **outermost** layer.
* **`get_wrapper_toolset`** follows the same nesting: the first capability's wrapper is outermost.

This means the first capability in the list has the first and last say on the operation — it sees the original input before any other capability, and it sees the final output after all inner capabilities have processed it.

## Ordering

By default, capabilities are composed in the order you list them. When a capability needs to be at a specific position regardless of where the user lists it, override [`get_ordering`][pydantic_ai.capabilities.AbstractCapability.get_ordering] to return a [`CapabilityOrdering`][pydantic_ai.capabilities.CapabilityOrdering]:

```python {title="capability_ordering_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    CombinedCapability,
)


@dataclass
class InstrumentationCapability(AbstractCapability[Any]):
    """Must wrap all other capabilities to trace everything."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


@dataclass
class PlainCapability(AbstractCapability[Any]):
    pass


# InstrumentationCapability ends up first regardless of list order
combined = CombinedCapability([PlainCapability(), InstrumentationCapability()])
assert type(combined.capabilities[0]) is InstrumentationCapability
```

The available constraints are:

* **`position`** — `'outermost'` or `'innermost'`. Places the capability in a tier before (or after) all capabilities without that position. Multiple capabilities can share a tier; original list order breaks ties within it.
* **`wraps`** — list of capabilities this one wraps around (is outside of). Each entry can be a capability **type** (matches all instances via `issubclass`) or a specific **instance** (matches by identity). Use when your capability needs to see the output of another: `CapabilityOrdering(wraps=[OtherCapability])`.
* **`wrapped_by`** — list of capabilities that wrap around this one (are outside of it). Accepts types or instances, like `wraps`. The inverse of `wraps`.
* **`requires`** — list of capability types that must be present. Raises [`UserError`][pydantic_ai.exceptions.UserError] if any are missing. Does not imply ordering.

When constraints are declared, [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] topologically sorts its children at construction time, preserving user-provided order as a tiebreaker.

[`Hooks`][pydantic_ai.capabilities.Hooks] supports ordering via the `ordering` parameter, so you can declare ordering constraints without subclassing:

```python {title="hooks_ordering_example.py"}
from pydantic_ai.capabilities import CapabilityOrdering, CombinedCapability, Hooks

logging_hooks = Hooks(ordering=CapabilityOrdering(position='outermost'))
rate_limit_hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[logging_hooks]))

# logging_hooks ends up outermost; rate_limit_hooks is wrapped by it
combined = CombinedCapability([rate_limit_hooks, logging_hooks])
assert combined.capabilities[0] is logging_hooks
assert combined.capabilities[1] is rate_limit_hooks
```


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


## Publishing capabilities

To make a custom capability usable in [agent specs](../agent-spec.md), it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

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

See [Extensibility](../extensibility.md) for packaging conventions and the broader extension ecosystem.
