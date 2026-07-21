from __future__ import annotations as _annotations

import asyncio
import contextvars
import dataclasses
import functools
import inspect
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from contextvars import ContextVar
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeGuard, cast, overload

import anyio
from opentelemetry.trace import NoOpTracer
from pydantic.alias_generators import to_snake
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Self, TypeIs, TypeVar

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION
from pydantic_ai._spec import load_from_registry
from pydantic_ai.capabilities._deferred_capability_loader import DeferredCapabilityLoader

from .. import (
    _agent_graph,
    _instructions,
    _output,
    _system_prompt,
    _utils,
    concurrency as _concurrency,
    exceptions,
    messages as _messages,
    models,
    usage as _usage,
)
from .._agent_graph import (
    CallToolsNode,
    EndStrategy,
    ModelRequestNode,
    UserPromptNode,
    build_run_context,
    capture_run_messages,
)
from .._deferred_capabilities import parse_loaded_capabilities
from .._instructions import AgentInstructions
from .._output import OutputToolset
from .._template import TemplateStr, validate_from_spec_args
from .._warnings import PydanticAIDeprecationWarning
from ..capabilities import (
    AbstractCapability,
    AgentCapability,
    AgentModel,
    CombinedCapability,
    ModelSelection,
    ModelSelector,
    ToolSearch as ToolSearchCap,
)
from ..capabilities._dynamic import wrap_capability_funcs
from ..capabilities._ordering import find_capability, has_capability_type
from ..capabilities._pending_messages import PendingMessageDrainCapability
from ..capabilities.instrumentation import Instrumentation as InstrumentationCap
from ..models.instrumented import InstrumentationSettings, InstrumentedModel
from ..native_tools import AbstractNativeTool
from ..native_tools._tool_search import ToolSearchTool
from ..output import OutputDataT, OutputSpec, StructuredDict
from ..run import AgentRun, AgentRunResult
from ..settings import ModelSettings, merge_model_settings
from ..tool_manager import ParallelExecutionMode, ToolManager
from ..tools import (
    AgentDepsT,
    AgentNativeTool,
    ArgsValidatorFunc,
    DeferredToolResults,
    DocstringFormat,
    GenerateToolJsonSchema,
    NativeToolFunc,
    RunContext,
    SystemPromptFunc,
    Tool,
    ToolDefinition,
    ToolDenied,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
    ToolsPrepareFunc,
)
from ..toolsets import AbstractToolset, AgentToolset
from ..toolsets._dynamic import (
    DynamicToolset,
    ToolsetFunc,
)
from ..toolsets._tool_search import parse_discovered_tools
from ..toolsets.combined import CombinedToolset
from ..toolsets.function import FunctionToolset
from ..toolsets.prepared import PreparedToolset
from .abstract import (
    AbstractAgent,
    AgentMetadata,
    AgentModelSettings,
    AgentRetries,
    EventStreamHandler,
    EventStreamProcessor,
    RunOutputDataT,
)
from .spec import AgentSpec, get_capability_registry
from .wrapper import WrapperAgent

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from pydantic_graph import GraphRunContext

    from ..realtime import AudioRetention, KnownRealtimeModelName, RealtimeModel, RealtimeModelSettings, RealtimeSession
    from ..ui._web import ModelsParam

__all__ = (
    'AbstractAgent',
    'Agent',
    'AgentModelSettings',
    'AgentRetries',
    'AgentRun',
    'AgentRunResult',
    'NativeToolFunc',
    'CallToolsNode',
    'EndStrategy',
    'EventStreamHandler',
    'EventStreamProcessor',
    'InstrumentationSettings',
    'ModelRequestNode',
    'ParallelExecutionMode',
    'UserPromptNode',
    'WrapperAgent',
    'capture_run_messages',
    'PydanticAIDeprecationWarning',
    'ToolsPrepareFunc',
    'ToolDenied',
)


@dataclasses.dataclass(frozen=True)
class _ResolvedAgentRetries:
    """Fully resolved retry budgets used internally."""

    tools: int
    output: int


def _is_model(value: object) -> TypeIs[models.Model[Any]]:
    """Narrow a value to a concrete model without losing its client type to `Unknown`."""
    return isinstance(value, models.Model)


def _normalize_agent_retries(retries: AgentRetries, *, default: int = 1) -> _ResolvedAgentRetries:
    """Resolve normalized retry overrides into concrete retry budgets.

    Missing keys in an `AgentRetries` dict fall back to `default`, so internal code can work with a
    single concrete shape.
    """
    return _ResolvedAgentRetries(tools=retries.get('tools', default), output=retries.get('output', default))


def _normalize_agent_retry_overrides(
    retries: int | AgentRetries | None,
    *,
    int_means: Literal['both', 'output'] = 'both',
) -> AgentRetries:
    """Normalize retry input without filling missing keys.

    This is used while merging layered configuration. At run/override time, `int_means='output'`
    treats `retries=N` as an output-budget override only.
    """
    if retries is None:
        return {}
    if isinstance(retries, int):
        if int_means == 'output':
            return {'output': retries}
        return {'tools': retries, 'output': retries}
    return retries.copy()


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)


@dataclasses.dataclass
class _ResolvedSpec:
    """Result of resolving an AgentSpec for use at run/override time."""

    capability: CombinedCapability[Any] | None
    instructions: list[str | SystemPromptFunc[Any]]
    model: str | None
    model_settings: ModelSettings | None
    metadata: dict[str, Any] | None
    name: str | None
    output_retries: int | None


@dataclasses.dataclass(init=False)
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the output type they return, [`OutputDataT`][pydantic_ai.output.OutputDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[object, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-5.2')
    result = agent.run_sync('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.
    ```
    """

    _model: models.Model | models.KnownModelName | str | None

    _name: str | None
    _description: TemplateStr[AgentDepsT] | str | None
    end_strategy: EndStrategy
    """The strategy for handling function tool calls the model requests alongside a result that ends the run.

    That result usually comes from an output tool call, but with `NativeOutput`, `PromptedOutput`, or image
    output it comes from the structured text or image the model returns in the same response. Plain,
    unstructured text (`str` or `TextOutput`) is not treated as such a result: since the model isn't told
    its text is final, `end_strategy` never skips tools on its account, even under `'early'`.

    Defaults to `'graceful'`. See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for the behavior of
    each strategy.
    """

    model_settings: AgentModelSettings[AgentDepsT] | None
    """Optional model request settings to use for this agent's runs, by default.

    Can be a static `ModelSettings` dict or a callable that takes a
    [`RunContext`][pydantic_ai.tools.RunContext] and returns `ModelSettings`.
    Callables are called before each model request, allowing dynamic per-step settings.

    Note, if `model_settings` is also provided at run time, those settings will be merged
    on top of the agent-level settings, with the run-level argument taking priority.
    """

    _output_type: OutputSpec[OutputDataT]

    _instrument: InstrumentationSettings | bool | None
    """Backing store for the `instrument` attribute. Read internally by
    `_resolve_instrumentation_settings()`; the public `agent.instrument` property reads/writes
    this field directly."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False
    _metadata: AgentMetadata[AgentDepsT] | None = dataclasses.field(repr=False)

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _output_schema: _output.OutputSchema[OutputDataT] = dataclasses.field(repr=False)
    _output_validators: list[_output.OutputValidator[AgentDepsT, OutputDataT]] = dataclasses.field(repr=False)
    _instructions: list[str | SystemPromptFunc[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_toolset: FunctionToolset[AgentDepsT] = dataclasses.field(repr=False)
    _output_toolset: OutputToolset[AgentDepsT] | None = dataclasses.field(repr=False)
    _user_toolsets: list[AbstractToolset[AgentDepsT]] = dataclasses.field(repr=False)
    _max_output_retries: int = dataclasses.field(repr=False)
    _max_tool_retries: int = dataclasses.field(repr=False)
    _tool_timeout: float | None = dataclasses.field(repr=False)
    _validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = dataclasses.field(repr=False)

    _event_stream_handler: EventStreamHandler[AgentDepsT] | None = dataclasses.field(repr=False)

    _concurrency_limiter: _concurrency.AbstractConcurrencyLimiter | None = dataclasses.field(repr=False)

    _entered_count: int = dataclasses.field(repr=False)
    _exit_stack: AsyncExitStack | None = dataclasses.field(repr=False)

    @functools.cached_property
    def _enter_lock(self) -> anyio.Lock:
        # We use a cached_property for this because `anyio.Lock` binds to the event loop on which
        # it's first used; deferring creation until first access ensures it binds to the correct
        # running loop and avoids issues with Temporal's workflow sandbox.
        return anyio.Lock()

    # `__init__` keeps an overload pair purely so Pyright resolves a class-union `output_type`
    # (`Foo | Bar`) as `type[Foo | Bar]` rather than a bare `UnionType`; on a non-overloaded
    # signature Pyright rejects the union argument. The two overloads are intentionally
    # identical, so the second one overlaps the first.
    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = object,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'graceful',
        metadata: AgentMetadata[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
    ) -> None: ...

    @overload
    def __init__(  # pyright: ignore[reportOverlappingOverload]
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = object,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'graceful',
        metadata: AgentMetadata[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = object,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'graceful',
        metadata: AgentMetadata[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
    ) -> None:
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provided,
                you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
            output_type: The type of the output data, used to validate the data returned by the model,
                defaults to `str`.
            instructions: Instructions to use for this agent, you can also register instructions via a function with
                [`instructions`][pydantic_ai.agent.Agent.instructions] or pass additional, temporary, instructions when executing a run.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.agent.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[object, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            description: A human-readable description of the agent, attached to the agent run span as
                `gen_ai.agent.description` when instrumentation is enabled.
            model_settings: Optional model request settings to use for this agent's runs, by default.
                Can be a static `ModelSettings` dict or a callable that takes a
                [`RunContext`][pydantic_ai.tools.RunContext] and returns `ModelSettings`.
                Callables are called before each model request, allowing dynamic per-step settings.
            retries: Per-category retry budgets for tools and output validation. Pass an `int` to set the same
                budget for both, or an [`AgentRetries`][pydantic_ai.AgentRetries] dict to set them
                individually (e.g. `retries={'tools': 3, 'output': 1}`). Defaults to 1 for both.
                On the text path, `output` is a global budget shared across all output-validation retries
                in a run; on the tool path it is the default per-tool `max_retries` for each output tool,
                overridable via [`ToolOutput(max_retries=...)`][pydantic_ai.output.ToolOutput.max_retries].
                The `output` budget can be overridden per run via `agent.run(retries=...)` (and friends).
                For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
            validation_context: Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) used to validate tool arguments and outputs.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.agent.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain].
            toolsets: Toolsets to register with the agent, including MCP servers and functions which take a run context
                and return a toolset. See [`ToolsetFunc`][pydantic_ai.toolsets.ToolsetFunc] for more information.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `True`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.agent.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            metadata: Optional metadata to store with each run.
                Provide a dictionary of primitives, or a callable returning one
                computed from the [`RunContext`][pydantic_ai.tools.RunContext] on each run.
                Metadata is resolved when a run starts and recomputed after a successful run finishes so it
                can reflect the final state.
                Resolved metadata can be read after the run completes via
                [`AgentRun.metadata`][pydantic_ai.agent.AgentRun],
                [`AgentRunResult.metadata`][pydantic_ai.agent.AgentRunResult], and
                [`StreamedRunResult.metadata`][pydantic_ai.result.StreamedRunResult],
                and is attached to the agent run span when instrumentation is enabled.
            tool_timeout: Default timeout in seconds for tool execution. If a tool takes longer than this,
                the tool is considered to have failed and a retry prompt is returned to the model (counting towards the retry limit).
                Individual tools can override this with their own timeout. Defaults to None (no timeout).
            max_concurrency: Optional limit on concurrent agent runs. Can be an integer for simple limiting,
                a [`ConcurrencyLimit`][pydantic_ai.ConcurrencyLimit] for advanced configuration with backpressure,
                a [`ConcurrencyLimiter`][pydantic_ai.ConcurrencyLimiter] for sharing limits across
                multiple agents, or None (default) for no limiting. When the limit is reached, additional calls
                to `run()` or `iter()` will wait until a slot becomes available.
            capabilities: Optional list of [capabilities](https://ai.pydantic.dev/capabilities/) to configure the agent with,
                including functions which take a run context and return a capability.
                See [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] for more information.
                Custom capabilities can be created by subclassing
                [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability].
        """
        self._name = name
        self._description = description
        self.end_strategy = end_strategy

        capabilities = wrap_capability_funcs(capabilities)

        _inject_auto_capabilities(capabilities)

        self._root_capability = CombinedCapability(capabilities)

        # Keep the constructor value untouched while capabilities bind. A capability may interpret
        # model IDs itself, so eagerly inferring a string here could construct the wrong provider
        # (and perform its authentication/configuration side effects) before `for_agent()` can add
        # the appropriate resolver.
        self._model = model

        self.model_settings = model_settings

        self._output_type = output_type
        self._instrument = None
        self._metadata = metadata
        self._deps_type = deps_type

        self._output_schema = _output.OutputSchema[OutputDataT].build(output_type)
        self._output_validators = []

        self._instructions = _instructions.normalize_instructions(instructions)

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        retry_overrides = _normalize_agent_retry_overrides(retries)
        resolved_retries = _normalize_agent_retries(retry_overrides)
        self._max_tool_retries = resolved_retries.tools
        self._max_output_retries = resolved_retries.output
        self._tool_timeout = tool_timeout
        if self._tool_timeout is not None and self._tool_timeout <= 0:
            raise exceptions.UserError(f'tool_timeout must be > 0, got {self._tool_timeout}')

        self._validation_context = validation_context

        self._output_toolset = self._output_schema.toolset
        if self._output_toolset and self._output_toolset.max_retries is None:
            self._output_toolset.max_retries = self._max_output_retries

        self._function_toolset = _AgentFunctionToolset(
            tools,
            max_retries=self._max_tool_retries,
            timeout=self._tool_timeout,
            output_schema=self._output_schema,
        )

        # Agent-direct toolsets
        agent_toolsets = list(toolsets or [])
        self._dynamic_toolsets = [
            DynamicToolset[AgentDepsT](toolset_func=toolset)
            for toolset in agent_toolsets
            if not isinstance(toolset, AbstractToolset)
        ]
        self._user_toolsets = [toolset for toolset in agent_toolsets if isinstance(toolset, AbstractToolset)]

        # Populated by durable-execution subclasses; base agents use the run-level kwarg.
        self._event_stream_handler = None

        self._concurrency_limiter = _concurrency.normalize_to_limiter(max_concurrency)

        self._override_name: ContextVar[_utils.Option[str]] = ContextVar('_override_name', default=None)
        self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
        self._override_model: ContextVar[_utils.Option[models.Model | models.KnownModelName | str]] = ContextVar(
            '_override_model', default=None
        )
        self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
            '_override_toolsets', default=None
        )
        self._override_tools: ContextVar[
            _utils.Option[Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]]
        ] = ContextVar('_override_tools', default=None)
        self._override_native_tools: ContextVar[_utils.Option[Sequence[AgentNativeTool[AgentDepsT]]]] = ContextVar(
            '_override_native_tools', default=None
        )
        self._override_instructions: ContextVar[_utils.Option[list[str | SystemPromptFunc[AgentDepsT]]]] = ContextVar(
            '_override_instructions', default=None
        )
        self._override_metadata: ContextVar[_utils.Option[AgentMetadata[AgentDepsT]]] = ContextVar(
            '_override_metadata', default=None
        )
        self._override_model_settings: ContextVar[_utils.Option[AgentModelSettings[AgentDepsT]]] = ContextVar(
            '_override_model_settings', default=None
        )
        self._override_output_retries: ContextVar[_utils.Option[int]] = ContextVar(
            '_override_output_retries', default=None
        )
        self._override_root_capability: ContextVar[_utils.Option[CombinedCapability[AgentDepsT]]] = ContextVar(
            '_override_root_capability', default=None
        )
        self._entered_count = 0
        self._exit_stack = None
        self._entered_model_ids: set[int] = set()
        self._entered_models_by_selection: dict[tuple[int, str], models.Model] = {}

        # Initialize capability-contributed fields before binding so `for_agent` can safely
        # inspect `agent.toolsets`. Contributions from the bound capability are extracted below.
        self._cap_toolsets: list[AgentToolset[AgentDepsT]] = []
        self._cap_instructions: list[str | SystemPromptFunc[AgentDepsT]] = []
        self._cap_native_tools: list[AgentNativeTool[AgentDepsT]] = []
        self._cap_model_settings: AgentModelSettings[AgentDepsT] | None = None

        self._root_capability = self._root_capability.for_agent(self)
        if model is not None and not defer_model_check and not self._root_capability.has_resolve_model_id:
            self._model = models.infer_model(model)

        # Validate the bound tree so a replacement returned by `for_agent` is subject to the
        # same eager ID checks as the capability originally passed to the constructor.
        static_capabilities: list[AbstractCapability[AgentDepsT]] = []
        self._root_capability.apply(static_capabilities.append)
        _validate_capability_ids(static_capabilities)

        self._cap_instructions = _instructions.normalize_instructions(self._root_capability.get_instructions())
        self._cap_native_tools = list(self._root_capability.get_native_tools())
        _validate_native_tool_ids(self._cap_native_tools, source='agent capabilities')
        self._cap_model_settings = self._root_capability.get_model_settings()
        cap_toolset = self._root_capability.get_toolset()
        if cap_toolset is not None:
            self._cap_toolsets = [cap_toolset]

    @overload
    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any] | AgentSpec,
        *,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[object, str]: ...

    @overload
    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any] | AgentSpec,
        *,
        deps_type: type[T],
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[T, str]: ...

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any] | AgentSpec,
        *,
        deps_type: type[Any] = type(None),
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[Any, Any]:
        """Construct an Agent from a spec dict or `AgentSpec`.

        This allows defining agents declaratively in YAML/JSON/dict form.
        Keyword arguments supplement the spec: scalar spec fields (like `name`,
        `retries`) are used as defaults that explicit arguments override, while
        `capabilities` from both sources are merged.

        Args:
            spec: The agent specification, either a dict or an `AgentSpec` instance.
            deps_type: The type of the dependencies for the agent. When provided,
                template strings in capabilities (e.g. `"Hello {{name}}"`) are
                compiled and validated against this type.
            custom_capability_types: Additional capability classes to make available
                beyond the built-in defaults.
            model: Override the model from the spec.
            output_type: The type of the output data, defaults to `str`.
            instructions: Instructions for the agent.
            system_prompt: Static system prompts.
            name: The agent name, overrides spec `name` if provided.
            description: The agent description, overrides spec `description` if provided.
            model_settings: Model request settings.
            retries: Retry budgets for tools and output validation. Pass an `int` to set the same budget
                for both, or an [`AgentRetries`][pydantic_ai.AgentRetries] dict to set them individually.
                Overrides spec `retries` if provided.
            validation_context: Pydantic validation context for tool arguments and outputs.
            tools: Tools to register with the agent.
            toolsets: Toolsets to register with the agent.
            defer_model_check: Defer model evaluation until first run.
            end_strategy: Strategy for tool calls alongside a final result, overrides spec `end_strategy` if provided.
            metadata: Metadata to store with each run, overrides spec `metadata` if provided.
            tool_timeout: Default timeout for tool execution, overrides spec `tool_timeout` if provided.

            max_concurrency: Limit on concurrent agent runs.
            capabilities: Additional capabilities merged with those from the spec.

        Returns:
            A new Agent instance.
        """
        validated_spec, template_context = _validate_spec(spec, deps_type)

        effective_output_type: OutputSpec[Any]
        if output_type is not str:
            effective_output_type = output_type
        elif validated_spec.output_schema is not None:
            effective_output_type = StructuredDict(validated_spec.output_schema)
        else:
            effective_output_type = str

        # Merge instructions from spec and arg
        merged_instructions = _instructions.normalize_instructions(validated_spec.instructions)
        merged_instructions.extend(_instructions.normalize_instructions(instructions))

        all_capabilities: list[AgentCapability[Any]] = list(
            _capabilities_from_spec(validated_spec, custom_capability_types, template_context)
        )
        if capabilities:
            all_capabilities.extend(capabilities)

        effective_model = model or validated_spec.model
        if effective_model is None:
            raise exceptions.UserError(
                '`model` must be provided either in the spec or as a keyword argument to `from_spec()`.'
            )

        agent = Agent(
            model=effective_model,
            output_type=effective_output_type,
            instructions=merged_instructions or None,
            system_prompt=system_prompt,
            deps_type=deps_type,
            name=name or validated_spec.name,
            description=description or validated_spec.description,
            model_settings=merge_model_settings(
                cast(ModelSettings, validated_spec.model_settings) if validated_spec.model_settings else None,
                model_settings,
            ),
            retries=_merge_retries_with_spec(retries, validated_spec),
            validation_context=validation_context,
            tools=tools,
            toolsets=toolsets,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy if end_strategy is not None else validated_spec.end_strategy,
            metadata=metadata if metadata is not None else validated_spec.metadata,
            tool_timeout=tool_timeout if tool_timeout is not None else validated_spec.tool_timeout,
            max_concurrency=max_concurrency,
            capabilities=all_capabilities,
        )
        return agent

    @overload
    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        fmt: Literal['yaml', 'json'] | None = None,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[object, str]: ...

    @overload
    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        fmt: Literal['yaml', 'json'] | None = None,
        deps_type: type[T],
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[T, str]: ...

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        *,
        fmt: Literal['yaml', 'json'] | None = None,
        deps_type: type[Any] = type(None),
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        model: models.Model | models.KnownModelName | str | None = None,
        output_type: OutputSpec[Any] = str,
        instructions: AgentInstructions[Any] = None,
        system_prompt: str | Sequence[str] = (),
        name: str | None = None,
        description: TemplateStr[Any] | str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int | AgentRetries | None = None,
        validation_context: Any = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        metadata: AgentMetadata[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[Any, Any]:
        """Construct an Agent from a YAML or JSON spec file.

        This is a convenience method equivalent to
        `Agent.from_spec(AgentSpec.from_file(path), ...)`.

        The file format is inferred from the extension (`.yaml`/`.yml` or `.json`)
        unless overridden with the `fmt` argument.

        All other arguments are forwarded to [`from_spec`][pydantic_ai.Agent.from_spec].
        """
        spec = AgentSpec.from_file(path, fmt=fmt)
        agent = cls.from_spec(
            spec,
            deps_type=deps_type,
            custom_capability_types=custom_capability_types,
            model=model,
            output_type=output_type,
            instructions=instructions,
            system_prompt=system_prompt,
            name=name,
            description=description,
            model_settings=model_settings,
            retries=retries,
            validation_context=validation_context,
            tools=tools,
            toolsets=toolsets,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,
            metadata=metadata,
            tool_timeout=tool_timeout,
            max_concurrency=max_concurrency,
            capabilities=capabilities,
        )
        return agent

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all agents that don't explicitly add an `Instrumentation` capability."""
        Agent._instrument_default = instrument

    @property
    def instrument(self) -> InstrumentationSettings | bool | None:
        """Instrumentation settings applied to this agent."""
        return self._instrument

    @instrument.setter
    def instrument(self, value: InstrumentationSettings | bool | None) -> None:
        self._instrument = value

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        """The default model configured for this agent."""
        return self._model

    @model.setter
    def model(self, value: models.Model | models.KnownModelName | str | None) -> None:
        """Set the default model configured for this agent.

        We allow `str` here since the actual list of allowed models changes frequently.
        """
        self._model = value

    @property
    def name(self) -> str | None:
        """The name of the agent, used for logging.

        If `None`, we try to infer the agent name from the call frame when the agent is first run.
        """
        name_ = self._override_name.get()
        return name_.value if name_ else self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the name of the agent, used for logging."""
        self._name = value

    @property
    def description(self) -> str | None:
        """A human-readable description of the agent.

        If the description is a TemplateStr, returns the raw template source.
        The rendered description is available at runtime via OTel span attributes.
        """
        if self._description is None:
            return None
        return str(self._description)

    @description.setter
    def description(self, value: TemplateStr[AgentDepsT] | str | None) -> None:
        """Set the description of the agent."""
        self._description = value

    def render_description(self, deps: AgentDepsT = None) -> str | None:
        """Return the agent description, rendering any TemplateStr with the given deps."""
        if self._description is None:
            return None
        if isinstance(self._description, TemplateStr):
            return self._description.render(deps)
        return self._description

    @property
    def deps_type(self) -> type:
        """The type of dependencies used by the agent."""
        return self._deps_type

    @property
    def output_type(self) -> OutputSpec[OutputDataT]:
        """The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`."""
        return self._output_type

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """Optional handler for events from the model's streaming response and the agent's execution of tools."""
        return self._event_stream_handler

    def __repr__(self) -> str:
        return f'{type(self).__name__}(model={self.model!r}, name={self.name!r}, end_strategy={self.end_strategy!r}, model_settings={self.model_settings!r}, output_type={self.output_type!r})'

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(  # noqa: C901
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncGenerator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ],
                        timestamp=datetime.datetime(...),
                        run_id='...',
                        conversation_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-5.2',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                        conversation_id='...',
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        # Tool retries cannot be overridden per run: `int` is treated as the output budget. An explicit
        # `retries={'tools': ...}` is rejected so the value isn't silently dropped.
        retry_overrides = _normalize_agent_retry_overrides(retries, int_means='output')
        if 'tools' in retry_overrides:
            raise exceptions.UserError(
                'Per-run `retries` cannot set tool retries: tool retries can only be configured at agent '
                "construction time. Use `retries={'output': ...}` (or `retries=<int>` to override the output "
                'budget) here, and `Agent(retries=...)` for tool retries.'
            )

        # Resolve the root capability (override > agent default) up front: it's needed both for the
        # capability-supplied model fallback below and for run-time capability assembly further down.
        override_cap = self._override_root_capability.get()
        base_capability = self._effective_root_capability()

        # Resolve spec contributions (additive at run time)
        resolved = self._resolve_spec(spec)

        effective_output_retries = retry_overrides.get('output')
        if resolved is not None:
            # Model: spec as fallback (run param > spec > agent)
            if model is None and resolved.model is not None:
                model = resolved.model
            # Output retries: run param > spec > agent default
            if effective_output_retries is None and resolved.output_retries is not None:
                effective_output_retries = resolved.output_retries
            # Instructions: spec instructions are additional
            if resolved.instructions:
                extra = resolved.instructions
                if instructions is not None:
                    existing = _instructions.normalize_instructions(instructions)
                    existing.extend(extra)
                    instructions = existing
                else:
                    instructions = extra
            # Model settings: merge spec settings under run settings (only static dicts)
            if resolved.model_settings is not None:
                if model_settings is None or not callable(model_settings):
                    model_settings = merge_model_settings(resolved.model_settings, model_settings)
                # If model_settings is a callable, spec model_settings are handled via the capability layer
            # Metadata: merge spec metadata under run metadata
            if resolved.metadata is not None:
                if metadata is not None:
                    if callable(metadata):
                        _spec_meta = resolved.metadata
                        _orig_metadata = metadata

                        def _merged_meta(ctx: RunContext[AgentDepsT]) -> dict[str, Any]:
                            return {**(_spec_meta or {}), **_orig_metadata(ctx)}

                        metadata = _merged_meta
                    else:
                        metadata = {**resolved.metadata, **metadata}
                else:
                    metadata = resolved.metadata

        # `override(retries=...)` wins over the run kwarg + spec, matching the precedence
        # of `model`/`deps`/`instructions`/etc. (see `Agent._get_model`). This keeps testing
        # fixtures that wrap call sites in `agent.override(retries=N)` effective even when
        # production code passes its own `run(retries=...)`.
        override_output_retries = self._override_output_retries.get()
        if override_output_retries is not None:
            effective_output_retries = override_output_retries.value

        deps = self._get_deps(deps)
        usage = usage or _usage.RunUsage()

        # Run/spec capabilities that already exist before `for_run()` can participate in
        # bootstrap model selection and ID resolution. A capability function may only
        # contribute after a bootstrap model exists because its contract requires RunContext.
        extra_capabilities: list[AbstractCapability[AgentDepsT]] = []
        if resolved is not None and resolved.capability is not None:
            extra_capabilities.append(resolved.capability)
        extra_capabilities.extend(wrap_capability_funcs(capabilities))
        extra_capabilities = [capability.for_agent(self) for capability in extra_capabilities]
        model_layers: list[AbstractCapability[AgentDepsT]] = [base_capability, *extra_capabilities]
        bootstrap_capability: AbstractCapability[AgentDepsT]
        if len(model_layers) > 1:
            bootstrap_capability = CombinedCapability(model_layers)
        else:
            bootstrap_capability = model_layers[0]
        resolved_models_by_selection: dict[tuple[int, str], models.Model] = {}

        # Explicit run/spec/override models are authoritative. Otherwise the capability model
        # contribution selects the initial model needed to construct RunContext and resolve
        # `for_run()`; dynamic contributions are evaluated again for later request steps.
        model_is_explicit = model is not None or self._override_model.get() is not None
        model_contribution = None if model_is_explicit else bootstrap_capability.get_model()
        self._check_dynamic_model_resume(model_contribution, message_history)

        has_default_model = self._override_model.get() is not None or model is not None or self.model is not None
        default_model = (
            await self._resolve_model_selection(
                self._pick_raw_model(model),
                capability=bootstrap_capability,
                deps=deps,
                resolved_models=resolved_models_by_selection,
            )
            if has_default_model
            else None
        )
        if model_contribution is not None:
            selection_ctx = models.ModelSelectionContext(
                agent=self,
                deps=deps,
                model=default_model,
                run_step=1,
                messages=list(message_history) if message_history else [],
                usage=usage,
            )
            model_used = await self._evaluate_model_contribution(
                model_contribution,
                capability=bootstrap_capability,
                ctx=selection_ctx,
                resolved_models=resolved_models_by_selection,
            )
        elif default_model is not None:
            model_used = default_model
        else:
            raise exceptions.UserError('`model` must either be set on the agent or included when calling it.')
        del model
        output_schema = self._prepare_output_schema(output_type)

        output_type_ = output_type or self.output_type

        # We consider it a user error if a user tries to restrict the result type while having an output validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        output_validators = self._output_validators

        # Resolve the effective per-output-tool default: run arg > spec > agent init default
        effective_output_toolset_max_retries = (
            effective_output_retries if effective_output_retries is not None else self._max_output_retries
        )

        output_toolset = self._output_toolset
        if output_schema != self._output_schema or output_validators:
            output_toolset = output_schema.toolset
            if output_toolset:
                # Clone before mutating max_retries when the toolset is the shared agent-level
                # instance (output_schema == self._output_schema, branch hit via output_validators);
                # when output_schema differs, output_schema.toolset is already a fresh per-run instance.
                if output_toolset is self._output_toolset and effective_output_retries is not None:
                    output_toolset = copy(output_toolset)
                if output_toolset.max_retries is None or effective_output_retries is not None:
                    output_toolset.max_retries = effective_output_toolset_max_retries
                output_toolset.output_validators = output_validators
        elif output_toolset is not None and effective_output_retries is not None:
            # Clone before mutating max_retries so concurrent runs don't race on the
            # shared agent-level toolset.
            output_toolset = copy(output_toolset)
            output_toolset.max_retries = effective_output_toolset_max_retries

        # Build the graph
        graph = _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)

        # Build the initial state
        state = _agent_graph.GraphAgentState(
            message_history=list(message_history) if message_history else [],
            usage=usage,
            output_retries_used=0,
            run_step=0,
            conversation_id=_agent_graph.resolve_conversation_id(conversation_id, message_history),
        )

        # Build a resolver that computes model settings per-step, in order of precedence: run > agent > model
        model_settings_override = self._override_model_settings.get()
        agent_model_settings = (
            model_settings_override.value if model_settings_override is not None else self.model_settings
        )
        run_model_settings = model_settings if model_settings_override is None else None

        # Validate `tool_choice` on the static baseline. Callable layers (agent-level callable,
        # run-level callable, capability-supplied) may inject `'required'` or `list[str]` per-step
        # and are trusted to adapt across steps; static dict values would lock every step into a
        # tool call and prevent the agent from producing a final response.
        baseline_settings: ModelSettings | None = model_used.settings
        if not callable(agent_model_settings):
            baseline_settings = merge_model_settings(baseline_settings, agent_model_settings)
        if not callable(run_model_settings):
            baseline_settings = merge_model_settings(baseline_settings, run_model_settings)
        if baseline_settings:
            tool_choice = baseline_settings.get('tool_choice')
            if tool_choice == 'required' or isinstance(tool_choice, list):
                raise exceptions.UserError(
                    f'`tool_choice={tool_choice!r}` prevents the agent from producing a final response '
                    f'because output tools are excluded. Use `ToolOrOutput` to combine specific function '
                    f"tools with output capability, return a callable from a capability's "
                    f'`get_model_settings()` to vary `tool_choice` per step, or use '
                    f'`pydantic_ai.direct.model_request` for single-shot model calls.'
                )

        usage_limits = usage_limits or _usage.UsageLimits()

        # Resolve instrumentation: an explicit `InstrumentedModel` (passed by the user
        # to `Agent(model=...)`, e.g. by `logfire.instrument_pydantic_ai(model)`) wins,
        # then `Agent.instrument_all()` / `agent.instrument = ...` (read via
        # `_resolve_instrumentation_settings`). When detected, unwrap so the rest of the
        # run uses the plain model — the `Instrumentation` capability injected below
        # provides the spans.
        if isinstance(model_used, InstrumentedModel):
            instrumentation_settings: InstrumentationSettings | None = model_used.instrumentation_settings
            model_used = model_used.wrapped
        else:
            instrumentation_settings = self._resolve_instrumentation_settings()

        if instrumentation_settings is not None:
            tracer = instrumentation_settings.tracer
            instrumentation_cap: InstrumentationCap | None = InstrumentationCap(settings=instrumentation_settings)
        else:
            tracer = NoOpTracer()
            instrumentation_cap = None

        # Build initial RunContext for for_run lifecycle hooks. Includes every
        # field that's already known here — `tool_manager` and `validation_context`
        # are populated later by `build_run_context` once the run is iterating.
        initial_ctx = RunContext[AgentDepsT](
            deps=deps,
            agent=self,
            model=model_used,
            usage=usage,
            usage_limits=usage_limits,
            prompt=user_prompt,
            messages=state.message_history,
            tracer=tracer,
            trace_include_content=instrumentation_settings is not None and instrumentation_settings.include_content,
            instrumentation_version=instrumentation_settings.version
            if instrumentation_settings
            else DEFAULT_INSTRUMENTATION_VERSION,
            run_step=0,
            pending_messages=state.pending_messages,
            run_id=state.run_id,
            conversation_id=state.conversation_id,
        )

        # Resolve run metadata up front so capability and toolset `for_run` hooks
        # can see it on `RunContext.metadata`. Metadata factories receive the
        # `initial_ctx` above (no `tool_manager` / `validation_context` yet); they
        # will be invoked again at the end of the run with the full final state,
        # so any field that becomes available later still ends up reflected in
        # `agent_run.metadata`. Factories should be pure mappings over the run
        # context, not perform IO or have side effects.
        state.metadata = self._get_metadata(initial_ctx, metadata)
        initial_ctx.metadata = state.metadata

        # Resolve the capability layers and extract their per-run contributions. Shared with
        # `realtime_session` via `_resolve_run_capabilities` so both wire capabilities up identically;
        # this call site keeps the graph-only surroundings: the `InstrumentedModel` unwrap and
        # instrumentation-settings resolution above, the deferred loader (`inject_deferred_loader=True`),
        # the output toolset below, and the layered `get_model_settings` closure. Keep those in sync
        # with the realtime call site.
        resolved_caps = await self._resolve_run_capabilities(
            initial_ctx,
            base_capability=base_capability,
            extra_capabilities=extra_capabilities,
            instrumentation_cap=instrumentation_cap,
            inject_deferred_loader=True,
            base_is_override=override_cap is not None,
        )
        run_capability = resolved_caps.run_capability
        capabilities_dict = resolved_caps.capabilities
        cap_instructions = resolved_caps.instructions
        cap_native_tools = resolved_caps.native_tools
        cap_model_settings = resolved_caps.model_settings
        cap_toolsets = resolved_caps.toolsets

        # Whether any capability's `for_run` swapped a model-layer contribution during resolution; the
        # per-step model-selection block below keys off this. The model layers are the tail of the
        # resolved layers (the `Instrumentation` capability, when injected, sits at the front).
        resolved_layers = resolved_caps.resolved_layers
        model_layer_start = len(resolved_layers) - len(model_layers)
        model_layers_unchanged = all(
            resolved_layers[model_layer_start + index] is layer for index, layer in enumerate(model_layers)
        )

        # Build model settings resolver using per-run capability. Shared with `realtime_session` via
        # `_layer_model_settings` (agent -> capability -> run order; the model's own settings are the
        # base for a graph run). Resolved per model-request step here; once at connect in a session.
        def get_model_settings(run_context: RunContext[AgentDepsT]) -> ModelSettings | None:
            # A capability can select a different model per step, so the base is the step's live
            # `run_context.model` settings, not a captured initial model. A graph run always uses a
            # request-response `Model` here; realtime has its own settings path and never reaches this.
            # (Hoisted to a local first so pyright narrows cleanly after `RunContext.model` widened to
            # `AbstractModel`; member access on the narrowed attribute directly trips a false positive.)
            step_model = run_context.model
            return _layer_model_settings(
                run_context,
                (agent_model_settings, cap_model_settings, run_model_settings),
                base=step_model.settings if isinstance(step_model, models.Model) else None,
            )

        # Build toolset with per-run capability contributions
        toolset = self._get_toolset(
            output_toolset=output_toolset,
            additional_toolsets=toolsets,
            cap_toolsets=cap_toolsets,
            run_capability=run_capability,
            max_output_retries=effective_output_toolset_max_retries,
        )
        toolset = await toolset.for_run(initial_ctx)
        tool_manager = ToolManager[AgentDepsT](
            toolset, root_capability=run_capability, default_max_retries=self._max_tool_retries
        )

        # Build instructions with per-run capability contributions
        instructions_literal, instructions_functions = self._get_instructions(
            additional_instructions=instructions,
            cap_instructions=cap_instructions,
        )

        async def get_instructions(
            run_context: RunContext[AgentDepsT],
        ) -> list[_messages.InstructionPart] | None:
            parts: list[_messages.InstructionPart] = []

            if instructions_literal:
                parts.append(_messages.InstructionPart(content=instructions_literal, dynamic=False))

            for func in instructions_functions:
                text = await func.run(run_context)
                if text:
                    parts.append(_messages.InstructionPart(content=text, dynamic=True))

            return parts or None

        # The deferred capabilities the model has already loaded in prior steps; the graph
        # refreshes this from history before each model request, so the seed only matters
        # for pre-first-step access. Non-deferred capabilities are folded in by the
        # `RunContext.available_capability_ids` property.
        loaded_capability_ids = parse_loaded_capabilities(message_history) if message_history else set[str]()
        discovered_tool_names = parse_discovered_tools(message_history) if message_history else set[str]()

        run_model_contribution = None if model_is_explicit else run_capability.get_model()
        self._check_dynamic_model_resume(run_model_contribution, message_history)
        model_selector: ModelSelector[AgentDepsT] | None
        model_selected_for_step: int | None
        capability_owns_current_model: bool
        if model_layers_unchanged:
            model_selector = (
                model_contribution if callable(model_contribution) and not _is_model(model_contribution) else None
            )
            model_selected_for_step = 1 if model_selector is not None else None
            capability_owns_current_model = model_contribution is not None
        elif callable(run_model_contribution) and not _is_model(run_model_contribution):
            # The bootstrap model was only needed to construct RunContext for `for_run`.
            # The replacement selector makes the authoritative step-one choice in the graph,
            # but the discarded bootstrap model still needs its lifecycle managed.
            model_selector = run_model_contribution
            model_selected_for_step = None
            capability_owns_current_model = True
        elif run_model_contribution is not None:
            model_used = await self._resolve_model_selection(
                run_model_contribution,
                capability=run_capability,
                deps=deps,
                resolved_models=resolved_models_by_selection,
            )
            model_selector = None
            model_selected_for_step = None
            capability_owns_current_model = True
        elif default_model is not None:
            model_used = default_model
            model_selector = None
            model_selected_for_step = None
            capability_owns_current_model = False
        else:
            raise exceptions.UserError(
                'A capability removed the bootstrap model in `for_run()` but the agent has no default model.'
            )

        async def evaluate_model_selector(
            selector: ModelSelector[AgentDepsT], selection_ctx: models.ModelSelectionContext[AgentDepsT]
        ) -> models.Model:
            return await self._evaluate_model_contribution(
                selector,
                capability=run_capability,
                ctx=selection_ctx,
                resolved_models=resolved_models_by_selection,
            )

        model_stack: AsyncExitStack | None = None
        entered_model_ids = self._entered_model_ids.copy()

        async def enter_model(selected_model: models.Model) -> None:
            model_identity = id(selected_model)
            if model_identity in entered_model_ids:
                return
            assert model_stack is not None
            await model_stack.enter_async_context(selected_model)
            entered_model_ids.add(model_identity)

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT](
            user_deps=deps,
            agent=self,
            prompt=user_prompt,
            new_message_index=len(message_history) if message_history else 0,
            resumed_request=None,
            resumed_request_index=None,
            model=model_used,
            model_selector=model_selector,
            model_selected_for_step=model_selected_for_step,
            evaluate_model_selector=evaluate_model_selector,
            enter_model=enter_model,
            get_model_settings=get_model_settings,
            usage_limits=usage_limits,
            max_output_retries=effective_output_toolset_max_retries,
            end_strategy=self.end_strategy,
            output_schema=output_schema,
            output_validators=output_validators,
            validation_context=self._validation_context,
            root_capability=run_capability,
            capabilities=capabilities_dict,
            loaded_capability_ids=loaded_capability_ids,
            discovered_tool_names=discovered_tool_names,
            native_tools=cap_native_tools,
            tool_manager=tool_manager,
            tracer=tracer,
            get_instructions=get_instructions,
            instrumentation_settings=instrumentation_settings,
        )

        user_prompt_node = _agent_graph.UserPromptNode[AgentDepsT](
            user_prompt=user_prompt,
            deferred_tool_results=deferred_tool_results,
            instructions=instructions_literal,
            instructions_functions=instructions_functions,
            system_prompts=self._system_prompts,
            system_prompt_functions=self._system_prompt_functions,
            system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
        )

        agent_name = self.name or 'agent'

        async with AsyncExitStack() as stack:
            model_stack = stack
            await stack.enter_async_context(
                _concurrency.get_concurrency_context(self._concurrency_limiter, f'agent:{agent_name}')
            )
            if capability_owns_current_model:
                await enter_model(model_used)
            graph_run = await stack.enter_async_context(
                graph.iter(
                    inputs=user_prompt_node,
                    state=state,
                    deps=graph_deps,
                    span=None,
                    infer_name=False,
                )
            )
            agent_run = AgentRun(graph_run)
            self._resolve_and_store_metadata(agent_run.ctx, metadata)

            # Build RunContext for run lifecycle hooks
            run_ctx = _agent_graph.build_run_context(agent_run.ctx)

            # wrap_run cooperative hand-off protocol:
            #
            # 1. _do_run() calls before_run, sets _run_ready, then awaits _run_done.
            # 2. wrap_run wraps _do_run via the capability middleware chain.
            # 3. We await either _run_ready (handler started) or _wrap_task completion
            #    (short-circuit: wrap_run returned without calling handler).
            # 4. We yield agent_run to the caller for iteration.
            # 5. When the caller finishes (or an error occurs), we set _run_done.
            # 6. _do_run resumes: returns the result (success) or re-raises the error.
            # 7. If wrap_run catches the error and returns a recovery result, we use it.
            #    Otherwise the original error propagates.
            _run_ready = asyncio.Event()
            _run_done = asyncio.Event()
            _run_error: BaseException | None = None
            _wrap_context: list[tuple[ContextVar[Any], Any]] | None = None

            async def _do_run() -> AgentRunResult[Any]:
                nonlocal _wrap_context
                await run_capability.before_run(run_ctx)
                # Capture context vars set by wrap_run/before_run so
                # they can be propagated to the outer task where
                # agent_run.next() (and therefore node hooks) execute.
                _current_ctx = contextvars.copy_context()
                _wrap_context = [
                    (var, _current_ctx[var])
                    for var in _current_ctx
                    if var not in _outer_context or _outer_context[var] is not _current_ctx[var]
                ]
                _run_ready.set()
                await _run_done.wait()
                if _run_error is not None:
                    # Raise the original node error, not the potentially
                    # transformed version from context manager __aexit__ chains.
                    raise agent_run._node_error or _run_error  # pyright: ignore[reportPrivateUsage]
                r = agent_run.result
                assert r is not None
                return r

            _outer_context = contextvars.copy_context()
            _wrap_task = asyncio.create_task(run_capability.wrap_run(run_ctx, handler=_do_run))
            # Wait for handler to start or wrap_run to complete (short-circuit)
            _ready_waiter = asyncio.create_task(_run_ready.wait())
            try:
                await asyncio.wait({_ready_waiter, _wrap_task}, return_when=asyncio.FIRST_COMPLETED)
            except BaseException as exc:
                # Unblock `_do_run` before draining, mirroring the streaming handoff: if
                # `before_run`'s durable step absorbed the CancelledError (e.g. Temporal's
                # cooperative cancellation) and returned, `_do_run` is parked on
                # `_run_done.wait()`. Set `_run_error` so the survivor re-raises this error
                # instead of asserting on a not-yet-produced result, then set `_run_done` so
                # it can exit and `cancel_and_drain`'s gather can complete (it discards the
                # survivor's exception). Harmless no-op when `_wrap_task` really died
                # cancelled — it's already unwinding. See https://github.com/pydantic/pydantic-ai/issues/6422.
                _run_error = exc
                _run_done.set()
                await _utils.cancel_and_drain(_ready_waiter, _wrap_task)
                raise
            else:
                await _utils.cancel_and_drain(_ready_waiter)

            # Propagate context vars set by wrap_run/before_run to
            # the outer task so that agent_run.next() (and therefore
            # node hooks) can see them.
            _context_tokens: list[tuple[ContextVar[Any], contextvars.Token[Any]]] = []
            # Note: indexing instead of tuple unpacking because pyright
            # can't resolve types through nonlocal + Optional unpacking.
            for _cv_pair in _wrap_context or ():
                _context_tokens.append((_cv_pair[0], _cv_pair[0].set(_cv_pair[1])))

            # Register context var restore on the stack so it happens in LIFO order
            # (after toolset exit, before graph run exit).
            def _restore_context_vars() -> None:
                for _var, _token in _context_tokens:
                    _var.reset(_token)

            stack.callback(_restore_context_vars)

            # Enter toolset AFTER context vars are propagated so that
            # toolset __aenter__/__aexit__ run inside the run span context
            # (set by the Instrumentation capability's wrap_run).
            await stack.enter_async_context(toolset)

            async def _finalize_result(r: AgentRunResult[Any]) -> None:
                """Call after_run, store the result override, and clear any pending error."""
                nonlocal _run_error
                r = await run_capability.after_run(run_ctx, result=r)
                agent_run._result_override = r  # pyright: ignore[reportPrivateUsage]
                _run_error = None

            _short_circuited = _wrap_task.done() and not _run_ready.is_set()
            if _short_circuited:
                await _finalize_result(_wrap_task.result())

            try:
                yield agent_run
            except BaseException as _exc:
                # Use the original node error if available, since context manager
                # __aexit__ chains (GraphRun → anyio TaskGroup) may transform
                # the exception (e.g. into CancelledError or ExceptionGroup).
                _run_error = agent_run._node_error or _exc  # pyright: ignore[reportPrivateUsage]
                # Don't attempt recovery for GeneratorExit/KeyboardInterrupt —
                # awaiting _wrap_task during cleanup could delay shutdown.
                if isinstance(_run_error, (GeneratorExit, KeyboardInterrupt)):
                    raise
                # Don't re-raise yet — give wrap_run a chance to recover.
                # If wrap_run catches the error from handler() and returns
                # a recovery result, the exception will be suppressed.
            finally:
                if agent_run.result is not None:
                    self._resolve_and_store_metadata(agent_run.ctx, metadata)

                if not _short_circuited:
                    _run_done.set()
                    if _run_error is None and agent_run.result is not None:
                        await _finalize_result(await _wrap_task)
                    elif _run_error is not None:
                        # Error path: await wrap_run to see if it recovers.
                        # _do_run() re-raises _run_error; if wrap_run catches
                        # it and returns a result, recovery succeeds.
                        try:
                            await _finalize_result(await _wrap_task)
                        except BaseException as _wrap_exc:
                            # Attach wrap_run's own errors as context so they're
                            # visible in tracebacks (but don't mask the original).
                            # Skip CancelledError: it's expected cancellation propagation,
                            # and setting __context__ on it causes hangs on Python 3.10.
                            if not isinstance(_wrap_exc, asyncio.CancelledError) and _wrap_exc is not _run_error:
                                _run_error.__context__ = (
                                    _wrap_exc  # pragma: no cover — only fires for bugs in wrap_run implementations
                                )
                    elif (
                        not _wrap_task.done()
                    ):  # pragma: no branch — _run_done.set() can't complete _wrap_task synchronously
                        _wrap_task.cancel()
                        try:
                            await _wrap_task
                        except (asyncio.CancelledError, BaseException):
                            pass

            # If wrap_run didn't recover, give on_run_error a chance.
            if _run_error is not None:
                try:
                    _result = await run_capability.on_run_error(run_ctx, error=_run_error)
                except BaseException as _on_error_exc:
                    _run_error = _on_error_exc
                else:
                    await _finalize_result(_result)

            # If on_run_error didn't recover either, re-raise.
            # In an @asynccontextmanager, not re-raising suppresses the exception.
            if _run_error is not None:
                raise _run_error

    def _get_metadata(
        self,
        ctx: RunContext[AgentDepsT],
        additional_metadata: AgentMetadata[AgentDepsT] | None = None,
    ) -> dict[str, Any] | None:
        metadata_override = self._override_metadata.get()
        if metadata_override is not None:
            return self._resolve_metadata_config(metadata_override.value, ctx)

        base_metadata = self._resolve_metadata_config(self._metadata, ctx)
        run_metadata = self._resolve_metadata_config(additional_metadata, ctx)

        if base_metadata and run_metadata:
            return {**base_metadata, **run_metadata}
        return run_metadata or base_metadata

    def _resolve_metadata_config(
        self,
        config: AgentMetadata[AgentDepsT] | None,
        ctx: RunContext[AgentDepsT],
    ) -> dict[str, Any] | None:
        if config is None:
            return None
        metadata = config(ctx) if callable(config) else config
        return metadata

    def _resolve_and_store_metadata(
        self,
        graph_run_ctx: GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]],
        metadata: AgentMetadata[AgentDepsT] | None,
    ) -> dict[str, Any] | None:
        run_context = build_run_context(graph_run_ctx)
        resolved_metadata = self._get_metadata(run_context, metadata)
        graph_run_ctx.state.metadata = resolved_metadata
        return resolved_metadata

    def _resolve_spec(
        self,
        spec: dict[str, Any] | AgentSpec | None,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
    ) -> _ResolvedSpec | None:
        """Validate and instantiate capabilities from a spec, returning contributions.

        Returns None if spec is None.
        """
        if spec is None:
            return None

        validated_spec, template_context = _validate_spec(spec, self._deps_type)

        capabilities = list(_capabilities_from_spec(validated_spec, custom_capability_types, template_context))
        combined = CombinedCapability(capabilities) if capabilities else None

        retry_overrides = _retry_overrides_from_spec(validated_spec)
        if 'tools' in retry_overrides:
            warnings.warn(
                "AgentSpec retry field 'tools' is not supported at run/override time and will be ignored",
                UserWarning,
                stacklevel=3,
            )

        # Warn for unsupported fields with non-default values. Read via `__dict__` to avoid
        # triggering pydantic deprecation warnings on deprecated spec fields.
        for field_name in _UNSUPPORTED_SPEC_FIELDS:
            field_info = type(validated_spec).model_fields[field_name]
            if validated_spec.__dict__[field_name] != field_info.default:
                warnings.warn(
                    f'AgentSpec field {field_name!r} is not supported at run/override time and will be ignored',
                    UserWarning,
                    stacklevel=3,
                )

        return _ResolvedSpec(
            capability=combined,
            instructions=_instructions.normalize_instructions(validated_spec.instructions)
            if validated_spec.instructions
            else [],
            model=validated_spec.model,
            model_settings=cast(ModelSettings, validated_spec.model_settings)
            if validated_spec.model_settings
            else None,
            metadata=validated_spec.metadata,
            name=validated_spec.name,
            output_retries=retry_overrides.get('output'),
        )

    @contextmanager
    def override(  # noqa: C901
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        native_tools: Sequence[AgentNativeTool[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        instructions: AgentInstructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
        metadata: AgentMetadata[AgentDepsT] | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | _utils.Unset = _utils.UNSET,
        retries: int | AgentRetries | _utils.Unset = _utils.UNSET,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> Generator[None]:
        """Context manager to temporarily override agent configuration.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            native_tools: The native tools to use instead of the agent's configured native tools.
            instructions: The instructions to use instead of the instructions registered with the agent.
                Note: this also replaces capability-contributed instructions (e.g. from
                [`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions]).
            metadata: The metadata to use instead of the metadata passed to the agent constructor. When set, any
                per-run `metadata` argument is ignored.
            model_settings: The model settings to use instead of the model settings passed to the agent constructor.
                When set, any per-run `model_settings` argument is ignored.
            retries: The retry budgets to use instead of the agent-level configuration. Pass an `int` to
                override the output-validation budget, or an [`AgentRetries`][pydantic_ai.AgentRetries]
                dict for finer control. When set, any per-run `retries` argument is ignored. Tool retries
                cannot be overridden via `override()`.
            spec: Optional agent spec providing defaults for override. Explicit params take precedence
                over spec values. When the spec includes `capabilities`, they replace (not merge with)
                the agent's existing capabilities. To add capabilities without replacing, pass `spec`
                to `run()` or `iter()` instead.
        """
        # Tool retries cannot be overridden via `override()`. An int means "override output only".
        override_output_retries: int | _utils.Unset
        if _utils.is_set(retries):
            retry_overrides = _normalize_agent_retry_overrides(retries, int_means='output')
            if 'tools' in retry_overrides:
                raise exceptions.UserError(
                    '`agent.override(retries=...)` cannot set tool retries: tool retries can only be '
                    "configured at agent construction time. Use `retries={'output': ...}` (or "
                    '`retries=<int>` to override the output budget) here.'
                )
            override_output_retries = retry_overrides.get('output', _utils.UNSET)
        else:
            override_output_retries = _utils.UNSET

        resolved = self._resolve_spec(spec)

        # A spec capability replaces the agent's root capability for the duration of the
        # override. Build it before resolving an overridden model so custom model IDs can
        # be preserved for that capability's async, deps-aware resolver.
        if resolved is not None and resolved.capability is not None:
            override_caps = list(resolved.capability.capabilities)
            _inject_auto_capabilities(override_caps)
            override_capability: CombinedCapability[AgentDepsT] | None = CombinedCapability(override_caps).for_agent(
                self
            )
        else:
            override_capability = None

        # Apply spec values as defaults where explicit params are not set
        if resolved is not None:
            if not _utils.is_set(name) and resolved.name is not None:
                name = resolved.name
            if not _utils.is_set(model) and resolved.model is not None:
                model = resolved.model
            if not _utils.is_set(instructions) and resolved.instructions:
                instructions = resolved.instructions
            if not _utils.is_set(model_settings) and resolved.model_settings is not None:
                model_settings = resolved.model_settings
            if not _utils.is_set(metadata) and resolved.metadata is not None:
                metadata = resolved.metadata
            if not _utils.is_set(override_output_retries) and resolved.output_retries is not None:
                override_output_retries = resolved.output_retries

        if _utils.is_set(name):
            name_token = self._override_name.set(_utils.Some(name))
        else:
            name_token = None

        if _utils.is_set(deps):
            deps_token = self._override_deps.set(_utils.Some(deps))
        else:
            deps_token = None

        if _utils.is_set(model):
            model_capability = override_capability or self._effective_root_capability()
            override_model = (
                model if isinstance(model, str) and model_capability.has_resolve_model_id else models.infer_model(model)
            )
            model_token = self._override_model.set(_utils.Some(override_model))
        else:
            model_token = None

        if _utils.is_set(toolsets):
            toolsets_token = self._override_toolsets.set(_utils.Some(toolsets))
        else:
            toolsets_token = None

        if _utils.is_set(tools):
            tools_token = self._override_tools.set(_utils.Some(tools))
        else:
            tools_token = None

        if _utils.is_set(native_tools):
            native_tools_token = self._override_native_tools.set(_utils.Some(native_tools))
        else:
            native_tools_token = None

        if _utils.is_set(instructions):
            normalized_instructions = _instructions.normalize_instructions(instructions)
            instructions_token = self._override_instructions.set(_utils.Some(normalized_instructions))
        else:
            instructions_token = None

        if _utils.is_set(metadata):
            metadata_token = self._override_metadata.set(_utils.Some(metadata))
        else:
            metadata_token = None

        if _utils.is_set(model_settings):
            model_settings_token = self._override_model_settings.set(_utils.Some(model_settings))
        else:
            model_settings_token = None

        if _utils.is_set(override_output_retries):
            output_retries_token = self._override_output_retries.set(_utils.Some(override_output_retries))
        else:
            output_retries_token = None

        # Set capability from spec, replacing the agent's existing root capability.
        # Auto-inject infrastructure capabilities since the override replaces
        # (not merges with) the agent's root capability.
        if override_capability is not None:
            cap_token = self._override_root_capability.set(_utils.Some(override_capability))
        else:
            cap_token = None

        try:
            yield
        finally:
            if name_token is not None:
                self._override_name.reset(name_token)
            if deps_token is not None:
                self._override_deps.reset(deps_token)
            if model_token is not None:
                self._override_model.reset(model_token)
            if toolsets_token is not None:
                self._override_toolsets.reset(toolsets_token)
            if tools_token is not None:
                self._override_tools.reset(tools_token)
            if native_tools_token is not None:
                self._override_native_tools.reset(native_tools_token)
            if instructions_token is not None:
                self._override_instructions.reset(instructions_token)
            if metadata_token is not None:
                self._override_metadata.reset(metadata_token)
            if model_settings_token is not None:
                self._override_model_settings.reset(model_settings_token)
            if output_retries_token is not None:
                self._override_output_retries.reset(output_retries_token)
            if cap_token is not None:
                self._override_root_capability.reset(cap_token)

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], str | None], /
    ) -> Callable[[RunContext[AgentDepsT]], str | None]: ...

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str | None]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]: ...

    @overload
    def instructions(self, func: Callable[[], str | None], /) -> Callable[[], str | None]: ...

    @overload
    def instructions(self, func: Callable[[], Awaitable[str | None]], /) -> Callable[[], Awaitable[str | None]]: ...

    @overload
    def instructions(self, /) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]]: ...

    def instructions(
        self,
        func: SystemPromptFunc[AgentDepsT] | None = None,
        /,
    ) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]] | SystemPromptFunc[AgentDepsT]:
        """Decorator to register an instructions function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used bare (`agent.instructions`).

        Overloads for every possible signature of `instructions` are included so the decorator doesn't obscure
        the type of the function.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.instructions
        def simple_instructions() -> str:
            return 'foobar'

        @agent.instructions
        async def async_instructions(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: SystemPromptFunc[AgentDepsT],
            ) -> SystemPromptFunc[AgentDepsT]:
                self._instructions.append(func_)
                return func_

            return decorator
        else:
            self._instructions.append(func)
            return func

    async def system_prompt_parts(
        self,
        *,
        deps: AgentDepsT = None,
        model: models.Model | models.KnownModelName | str | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        prompt: str | Sequence[_messages.UserContent] | None = None,
        usage: _usage.RunUsage | None = None,
        model_settings: ModelSettings | None = None,
    ) -> list[_messages.SystemPromptPart]:
        """Resolve the agent's configured system prompts into `SystemPromptPart`s.

        See [`AbstractAgent.system_prompt_parts`][pydantic_ai.agent.AbstractAgent.system_prompt_parts].
        """
        deps = self._get_deps(deps)
        usage = usage or _usage.RunUsage()
        messages = list(message_history or [])
        capability = self._effective_root_capability()
        has_default_model = self._override_model.get() is not None or model is not None or self.model is not None
        default_model = (
            await self._resolve_model_selection(self._pick_raw_model(model), capability=capability, deps=deps)
            if has_default_model
            else None
        )
        if model is None and self._override_model.get() is None:
            contribution = capability.get_model()
            if contribution is not None:
                selection_ctx = models.ModelSelectionContext(
                    agent=self,
                    deps=deps,
                    model=default_model,
                    run_step=1,
                    messages=messages,
                    usage=usage,
                )
                selected_model = await self._evaluate_model_contribution(
                    contribution, capability=capability, ctx=selection_ctx
                )
            elif default_model is not None:
                selected_model = default_model
            else:
                raise exceptions.UserError('`model` must either be set on the agent or supplied by a capability.')
        else:
            assert default_model is not None
            selected_model = default_model
        run_context = RunContext[AgentDepsT](
            deps=deps,
            agent=self,
            model=selected_model,
            usage=usage,
            prompt=prompt,
            messages=messages,
            model_settings=model_settings,
            run_step=1,
        )
        return await _system_prompt.resolve_system_prompts(
            self._system_prompts, self._system_prompt_functions, run_context
        )

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], str | None], /
    ) -> Callable[[RunContext[AgentDepsT]], str | None]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str | None]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str | None], /) -> Callable[[], str | None]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str | None]], /) -> Callable[[], Awaitable[str | None]]: ...

    @overload
    def system_prompt(
        self, /, *, dynamic: bool = False
    ) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]]: ...

    def system_prompt(
        self,
        func: SystemPromptFunc[AgentDepsT] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]] | SystemPromptFunc[AgentDepsT]:
        """Decorator to register a system prompt function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used either bare (`agent.system_prompt`) or as a function call
        (`agent.system_prompt(...)`), see the examples below.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Args:
            func: The function to decorate
            dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
                see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt(dynamic=True)
        async def async_system_prompt(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: SystemPromptFunc[AgentDepsT],
            ) -> SystemPromptFunc[AgentDepsT]:
                runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
                self._system_prompt_functions.append(runner)
                if dynamic:  # pragma: lax no cover
                    self._system_prompt_dynamic_functions[func_.__qualname__] = runner
                return func_

            return decorator
        else:
            assert not dynamic, "dynamic can't be True in this case"
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
            return func

    @overload
    def output_validator(
        self, func: Callable[[RunContext[AgentDepsT], OutputDataT], OutputDataT], /
    ) -> Callable[[RunContext[AgentDepsT], OutputDataT], OutputDataT]: ...

    @overload
    def output_validator(
        self, func: Callable[[RunContext[AgentDepsT], OutputDataT], Awaitable[OutputDataT]], /
    ) -> Callable[[RunContext[AgentDepsT], OutputDataT], Awaitable[OutputDataT]]: ...

    @overload
    def output_validator(
        self, func: Callable[[OutputDataT], OutputDataT], /
    ) -> Callable[[OutputDataT], OutputDataT]: ...

    @overload
    def output_validator(
        self, func: Callable[[OutputDataT], Awaitable[OutputDataT]], /
    ) -> Callable[[OutputDataT], Awaitable[OutputDataT]]: ...

    def output_validator(
        self, func: _output.OutputValidatorFunc[AgentDepsT, OutputDataT], /
    ) -> _output.OutputValidatorFunc[AgentDepsT, OutputDataT]:
        """Decorator to register an output validator function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `output_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```python
        from pydantic_ai import Agent, ModelRetry, RunContext

        agent = Agent('test', deps_type=str)

        @agent.output_validator
        def output_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.output_validator
        async def output_validator_deps(ctx: RunContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.output)
        #> success (no tool calls)
        ```
        """
        self._output_validators.append(_output.OutputValidator[AgentDepsT, Any](func))
        return func

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=int)

        @agent.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            args_validator: custom method to validate tool arguments after schema validation has passed,
                before execution. The validator receives the already-validated and type-converted parameters,
                with `RunContext` as the first argument.
                Should raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on validation failure,
                return `None` on success.
                See [`ArgsValidatorFunc`][pydantic_ai.tools.ArgsValidatorFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether this tool acts as a barrier that runs alone, not overlapping with other tool calls.
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            timeout: Timeout in seconds for tool execution. If the tool takes longer, a retry prompt is returned to the model.
                Overrides the agent-level `tool_timeout` if set. Defaults to None (no timeout).
            defer_loading: Whether to hide this tool until it's discovered via tool search. Defaults to False.
                See [Tool Search](../tools-advanced.md#tool-search) for more info.
            include_return_schema: Whether to include the return schema in the tool definition sent to the model.
                If `None`, defaults to `False` unless the [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is used.
        """

        def tool_decorator(
            func_: ToolFuncContext[AgentDepsT, ToolParams],
        ) -> ToolFuncContext[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                takes_ctx=True,
                name=name,
                description=description,
                retries=retries,
                prepare=prepare,
                args_validator=args_validator,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
                timeout=timeout,
                defer_loading=defer_loading,
                include_return_schema=include_return_schema,
            )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Any:
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":123,"spam":3.14}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            args_validator: custom method to validate tool arguments after schema validation has passed,
                before execution. The validator receives the already-validated and type-converted parameters,
                with [`RunContext`][pydantic_ai.tools.RunContext] as the first argument — even though the
                tool function itself does not take `RunContext` when using `tool_plain`.
                Should raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] on validation failure,
                return `None` on success.
                See [`ArgsValidatorFunc`][pydantic_ai.tools.ArgsValidatorFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether this tool acts as a barrier that runs alone, not overlapping with other tool calls.
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info. Defaults to False.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
            timeout: Timeout in seconds for tool execution. If the tool takes longer, a retry prompt is returned to the model.
                Overrides the agent-level `tool_timeout` if set. Defaults to None (no timeout).
            defer_loading: Whether to hide this tool until it's discovered via tool search. Defaults to False.
                See [Tool Search](../tools-advanced.md#tool-search) for more info.
            include_return_schema: Whether to include the return schema in the tool definition sent to the model.
                If `None`, defaults to `False` unless the [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] capability is used.
        """

        def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                takes_ctx=False,
                name=name,
                description=description,
                retries=retries,
                prepare=prepare,
                args_validator=args_validator,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
                timeout=timeout,
                defer_loading=defer_loading,
                include_return_schema=include_return_schema,
            )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    @overload
    def toolset(self, func: ToolsetFunc[AgentDepsT], /) -> ToolsetFunc[AgentDepsT]: ...

    @overload
    def toolset(
        self,
        /,
        *,
        per_run_step: bool = True,
        id: str | None = None,
    ) -> Callable[[ToolsetFunc[AgentDepsT]], ToolsetFunc[AgentDepsT]]: ...

    def toolset(
        self,
        func: ToolsetFunc[AgentDepsT] | None = None,
        /,
        *,
        per_run_step: bool = True,
        id: str | None = None,
    ) -> Any:
        """Decorator to register a toolset function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.

        Can decorate a sync or async functions.

        The decorator can be used bare (`agent.toolset`).

        Example:
        ```python
        from pydantic_ai import AbstractToolset, Agent, FunctionToolset, RunContext

        agent = Agent('test', deps_type=str)

        @agent.toolset
        async def simple_toolset(ctx: RunContext[str]) -> AbstractToolset[str]:
            return FunctionToolset()
        ```

        Args:
            func: The toolset function to register.
            per_run_step: Whether to re-evaluate the toolset for each run step. Defaults to True.
            id: An optional unique ID for the dynamic toolset. Required for use with durable execution
                environments like Temporal, where the ID identifies the toolset's activities within the workflow.
        """

        def toolset_decorator(func_: ToolsetFunc[AgentDepsT]) -> ToolsetFunc[AgentDepsT]:
            self._dynamic_toolsets.append(DynamicToolset(func_, per_run_step=per_run_step, id=id))
            return func_

        return toolset_decorator if func is None else toolset_decorator(func)

    def _pick_raw_model(
        self, model: models.Model | models.KnownModelName | str | None
    ) -> models.Model | models.KnownModelName | str:
        if some_model := self._override_model.get():
            return some_model.value
        if model is not None:
            return model
        if self.model is not None:
            return self.model
        raise exceptions.UserError('`model` must either be set on the agent or included when calling it.')

    def _effective_root_capability(self) -> CombinedCapability[AgentDepsT]:
        """Return the override capability when present, otherwise the configured root."""
        override = self._override_root_capability.get()
        return override.value if override is not None else self._root_capability

    async def _resolve_model_selection(
        self,
        selection: ModelSelection,
        *,
        capability: AbstractCapability[AgentDepsT],
        deps: AgentDepsT,
        resolved_models: dict[tuple[int, str], models.Model] | None = None,
    ) -> models.Model:
        """Resolve a concrete model selection through the capability chain."""
        if not isinstance(selection, str):
            return selection
        cache_key = (id(capability), selection)
        if resolved_models is not None and (resolved_model := resolved_models.get(cache_key)) is not None:
            return resolved_model
        if entered_model := self._entered_models_by_selection.get(cache_key):
            if resolved_models is not None:
                resolved_models[cache_key] = entered_model
            return entered_model
        resolution_ctx = models.ModelResolutionContext(agent=self, deps=deps)
        resolved = await capability.resolve_model_id(resolution_ctx, model_id=selection)
        resolved_model = resolved if resolved is not None else models.infer_model(selection)
        if resolved_models is not None:
            resolved_models[cache_key] = resolved_model
        return resolved_model

    async def _evaluate_model_contribution(
        self,
        contribution: AgentModel[AgentDepsT],
        *,
        capability: AbstractCapability[AgentDepsT],
        ctx: models.ModelSelectionContext[AgentDepsT],
        resolved_models: dict[tuple[int, str], models.Model] | None = None,
    ) -> models.Model:
        """Evaluate a static or dynamic model contribution and resolve its result."""
        selection = contribution(ctx) if callable(contribution) and not _is_model(contribution) else contribution
        if inspect.isawaitable(selection):
            selection = await selection
        return await self._resolve_model_selection(
            selection, capability=capability, deps=ctx.deps, resolved_models=resolved_models
        )

    @staticmethod
    def _check_dynamic_model_resume(
        contribution: AgentModel[AgentDepsT] | None,
        message_history: Sequence[_messages.ModelMessage] | None,
    ) -> None:
        """Reject cross-run continuation when a selector cannot reconstruct the pinned model."""
        if (
            callable(contribution)
            and not _is_model(contribution)
            and message_history
            and isinstance(message_history[-1], _messages.ModelResponse)
            and message_history[-1].state == 'suspended'
        ):
            raise exceptions.UserError(
                'Cannot resume a suspended response with a dynamic capability model: the model '
                'that created the provider-side job cannot be reconstructed unambiguously. Pass '
                'that model explicitly to `run(model=...)` when resuming.'
            )

    def _get_model_outside_run(self, model: models.Model | models.KnownModelName | str | None = None) -> models.Model:
        """Resolve a configured or static capability model where run deps are unavailable."""
        capability = self._effective_root_capability()
        if model is not None or self._override_model.get() is not None:
            selection = self._pick_raw_model(model)
            return selection if _is_model(selection) else models.infer_model(selection)
        contribution = capability.get_model()
        if callable(contribution) and not _is_model(contribution):
            raise exceptions.UserError(
                'The capability model is dynamic and can only be selected during a run with run dependencies. '
                'Pass a concrete model explicitly.'
            )
        selection = contribution if contribution is not None else self._pick_raw_model(None)
        if isinstance(selection, str) and capability.has_resolve_model_id:
            raise exceptions.UserError(
                'The configured model ID is resolved by a capability using run dependencies. '
                'Pass a concrete model explicitly.'
            )
        return selection if _is_model(selection) else models.infer_model(selection)

    def _resolve_instrumentation_settings(self) -> InstrumentationSettings | None:
        """Resolve effective `InstrumentationSettings` from `Agent.instrument_all` / `agent.instrument`."""
        instrument = self._instrument if self._instrument is not None else self._instrument_default
        if not instrument:
            return None
        return InstrumentationSettings() if instrument is True else instrument

    def _get_deps(self: Agent[T, OutputDataT], deps: T) -> T:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps.get():
            return some_deps.value
        else:
            return deps

    async def _resolve_run_capabilities(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        base_capability: AbstractCapability[AgentDepsT],
        extra_capabilities: list[AbstractCapability[AgentDepsT]],
        instrumentation_cap: InstrumentationCap | None,
        inject_deferred_loader: bool,
        base_is_override: bool,
    ) -> _ResolvedRunCapabilities[AgentDepsT]:
        """Resolve the per-run capability layers and extract their contributions.

        Shared by [`iter`][pydantic_ai.agent.AbstractAgent.iter] / [`run`][pydantic_ai.agent.AbstractAgent.run]
        and [`realtime_session`][pydantic_ai.agent.Agent.realtime_session] so both wire capabilities up
        identically: the outermost `Instrumentation` injection, per-layer `for_run` resolution (never
        composing first — see below), optional deferred-loader injection, and the native-tool /
        instruction / model-settings / toolset contributions with `override(native_tools=...)` folded
        in. Each caller keeps its own surrounding logic (instrumentation *settings* resolution,
        model-settings layering, output toolset, the graph vs. the connection) and cross-references
        this method so the two stay in sync.

        `ctx` must already carry `metadata` (and, when instrumented, `tracer` / `trace_include_content`),
        since capability `for_run` hooks observe it. `base_is_override` is whether `base_capability`
        came from `override(root_capability=...)` (only the graph run supports that today), used solely
        for the native-tool validation error's `source`.
        """
        run_layers: list[AbstractCapability[AgentDepsT]] = [base_capability, *extra_capabilities]
        # Prepend `Instrumentation` (outermost, so its spans wrap everything) unless the user already
        # added one themselves — mirroring the explicit-capability-wins precedence.
        if instrumentation_cap is not None and not has_capability_type(run_layers, InstrumentationCap):
            run_layers.insert(0, instrumentation_cap)

        # Resolve `for_run` per layer instead of composing a `CombinedCapability` first (which would
        # gather over the same children): the `override(native_tools=...)` merge below needs the
        # *resolved* extras — their native tools may only materialize in `for_run`, e.g. from a
        # capability function's returned capability — and the composed tree re-flattens/re-sorts its
        # children and can't hand them back. Nor can the extras be re-resolved afterwards to peek:
        # `for_run` is documented as called once per run and may have per-run side effects (the
        # durable-exec integrations rely on this for deterministic replay). Composing from the resolved
        # pieces yields the same structure as resolving a pre-composed tree, since the same
        # flatten-and-sort runs on the same resolved children either way.
        resolved_layers = await _utils.gather(*(cap.for_run(ctx) for cap in run_layers))
        # The extras are the tail of `run_layers` (instrumentation, if added, is at the front). Slicing
        # from the front avoids the `[-0:]` full-list pitfall when there are no extras.
        resolved_extras = resolved_layers[len(resolved_layers) - len(extra_capabilities) :]
        run_capability = CombinedCapability(resolved_layers) if len(resolved_layers) > 1 else resolved_layers[0]

        # Re-extract get_*() from the resolved capability if anything is contributed per-run.
        capabilities = _build_run_capabilities(run_capability)
        # Inject the loader only if a deferred capability is present AND `for_run` didn't already return
        # one, or a second loader toolset then errors on the reserved `load_capability` name
        # (cf. https://github.com/pydantic/pydantic-ai/issues/5047).
        if inject_deferred_loader and (
            any(capability.defer_loading is True for capability in capabilities.values())
            and not has_capability_type([run_capability], DeferredCapabilityLoader)
        ):
            run_capability = CombinedCapability([run_capability, DeferredCapabilityLoader()])
            capabilities = _build_run_capabilities(run_capability)

        # Only read contributions from the resolved tree when the run actually has capabilities beyond
        # the plain agent default; otherwise fall back to the init-time snapshots.
        if run_capability is not base_capability or base_is_override:
            source_cap: AbstractCapability[AgentDepsT] | None = run_capability
        else:
            source_cap = None
        if source_cap is not None:
            instructions = _instructions.normalize_instructions(source_cap.get_instructions())
            native_tools = list(source_cap.get_native_tools())
            model_settings = source_cap.get_model_settings()
            cap_toolset = source_cap.get_toolset()
            toolsets: list[AgentToolset[AgentDepsT]] | None = [cap_toolset] if cap_toolset is not None else []
        else:
            instructions = None  # use init-time defaults
            native_tools = self._cap_native_tools
            model_settings = self._cap_model_settings
            toolsets = None

        # Native tool ids are validated per layer, from each layer's *resolved* form (contributions
        # from e.g. capability functions only materialize in `for_run`). Conflicting definitions
        # sharing a `unique_id` *within* a layer are ambiguous; last-wins *across* layers is the
        # intentional override mechanism. Instrumentation contributes no native tools.
        base_native_tools = [
            tool
            for cap in resolved_layers[: len(resolved_layers) - len(extra_capabilities)]
            for tool in cap.get_native_tools()
        ]
        _validate_native_tool_ids(
            base_native_tools,
            source='override spec capabilities' if base_is_override else 'agent capabilities',
        )
        extra_native_tools: list[AgentNativeTool[AgentDepsT]] = [
            tool for cap in resolved_extras for tool in cap.get_native_tools()
        ]
        _validate_native_tool_ids(extra_native_tools, source='run capabilities')

        # `override(native_tools=...)` replaces the agent's *baseline* native tools while still
        # preserving any additional per-run capability-contributed native tools on top.
        if some_native_tools := self._override_native_tools.get():
            _validate_native_tool_ids(some_native_tools.value, source='override native_tools')
            native_tools = [*some_native_tools.value, *extra_native_tools]

        return _ResolvedRunCapabilities(
            run_capability=run_capability,
            capabilities=capabilities,
            instructions=instructions,
            native_tools=native_tools,
            model_settings=model_settings,
            toolsets=toolsets,
            resolved_layers=resolved_layers,
        )

    def _get_instructions(
        self,
        additional_instructions: AgentInstructions[AgentDepsT] = None,
        cap_instructions: list[str | SystemPromptFunc[AgentDepsT]] | None = None,
    ) -> tuple[str | None, list[_system_prompt.SystemPromptRunner[AgentDepsT]]]:
        """Prepare agent-level instructions, splitting them into literal strings and functions.

        Toolset instructions are collected separately during run execution.

        Args:
            additional_instructions: Additional instructions to include for this run.
            cap_instructions: Instructions from capabilities, resolved at run time.

        Returns:
            A tuple of (literal_instructions, instruction_functions) where:
            - literal_instructions: Combined literal string instructions or None
            - instruction_functions: List of instruction functions that need to be evaluated at runtime
        """
        override_instructions = self._override_instructions.get()
        if override_instructions:
            # Override replaces all instructions, including capability contributions.
            instructions = override_instructions.value
        else:
            instructions = self._instructions.copy()
            instructions.extend(cap_instructions if cap_instructions is not None else self._cap_instructions)
            if additional_instructions is not None:
                instructions.extend(_instructions.normalize_instructions(additional_instructions))

        literal_parts: list[str] = []
        functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []

        for instruction in instructions:
            if isinstance(instruction, str):
                literal_parts.append(instruction)
            else:
                # TemplateStr instances land here too: they are callable with a
                # RunContext parameter, so SystemPromptRunner handles them like
                # any other system prompt function.
                functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](instruction))

        literal = '\n'.join(literal_parts).strip() or None
        return literal, functions

    def _get_toolset(
        self,
        output_toolset: AbstractToolset[AgentDepsT] | None | _utils.Unset = _utils.UNSET,
        additional_toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        cap_toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        run_capability: AbstractCapability[AgentDepsT] | None = None,
        max_output_retries: int | None = None,
    ) -> AbstractToolset[AgentDepsT]:
        """Get the complete toolset.

        Args:
            output_toolset: The output toolset to use instead of the one built at agent construction time.
            additional_toolsets: Additional toolsets to add, unless toolsets have been overridden.
            cap_toolsets: Per-run capability toolsets to use instead of the init-time capability toolsets.
            run_capability: The per-run capability instance, used to apply wrapper toolsets.
            max_output_retries: The effective output retry budget for this run (run kwarg / spec / agent default).
                Used as `ctx.max_retries` for the `prepare_output_tools` capability hook so it sees the
                same budget the run will actually enforce. Falls back to the agent-level default.
        """
        toolsets = list(self._build_toolset_list(cap_toolsets=cap_toolsets))
        # Don't add additional toolsets if the toolsets have been overridden
        if additional_toolsets and self._override_toolsets.get() is None:
            toolsets = [*toolsets, *additional_toolsets]

        toolset: AbstractToolset[AgentDepsT] = CombinedToolset(toolsets)

        if run_capability is not None:
            # Dispatch the `prepare_tools` capability hook through a `PreparedToolset` wrapped
            # **inside** any other capability `get_wrapper_toolset` results (e.g. `ToolSearch`,
            # `CodeMode`): filter/modify defs first, let other toolset transformations layer on
            # top. The hook sees **function** tools only — output tools route through
            # `prepare_output_tools` below.
            fn_cap = run_capability

            async def _dispatch_prepare_tools(
                ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return await fn_cap.prepare_tools(ctx, tool_defs)

            toolset = PreparedToolset(toolset, _dispatch_prepare_tools)

            # Capability wrapper toolsets (including ToolSearch and CodeMode) are
            # applied here via get_wrapper_toolset, around the prepare_tools wrap above.
            toolset = run_capability.get_wrapper_toolset(toolset) or toolset

        output_toolset = output_toolset if _utils.is_set(output_toolset) else self._output_toolset
        if output_toolset is not None:
            if run_capability is not None:
                # Dispatch the new `prepare_output_tools` capability hook through a `PreparedToolset`
                # wrapped around the output toolset specifically — so the hook only sees output
                # tools, and the filtered/modified defs flow into `ToolManager.tools` and the model
                # request parameters together. Override `ctx.max_retries` to the agent's output
                # retry budget (matches `_build_output_run_context`'s contract — see https://github.com/pydantic/pydantic-ai/issues/4745).
                # `output_toolset.max_retries` is set to `max_output_retries` at agent construction.
                output_cap = run_capability
                effective_max_output_retries = (
                    max_output_retries if max_output_retries is not None else self._max_output_retries
                )

                async def _dispatch_prepare_output_tools(
                    ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
                ) -> list[ToolDefinition]:
                    output_ctx = replace(ctx, max_retries=effective_max_output_retries)
                    return await output_cap.prepare_output_tools(output_ctx, tool_defs)

                output_toolset = PreparedToolset(output_toolset, _dispatch_prepare_output_tools)
            toolset = CombinedToolset([output_toolset, toolset])

        return toolset

    @property
    def root_capability(self) -> CombinedCapability[AgentDepsT]:
        """The root capability of the agent, containing all registered capabilities."""
        return self._root_capability

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """All toolsets registered on the agent, including a function toolset holding tools that were registered on the agent directly.

        Output tools are not included.
        """
        return self._build_toolset_list()

    def _build_toolset_list(
        self,
        cap_toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
    ) -> list[AbstractToolset[AgentDepsT]]:
        """Build the list of toolsets, optionally with per-run capability toolsets."""
        toolsets: list[AbstractToolset[AgentDepsT]] = []

        if some_tools := self._override_tools.get():
            function_toolset = _AgentFunctionToolset(
                some_tools.value,
                max_retries=self._max_tool_retries,
                timeout=self._tool_timeout,
                output_schema=self._output_schema,
            )
        else:
            function_toolset = self._function_toolset
        toolsets.append(function_toolset)

        if some_user_toolsets := self._override_toolsets.get():
            toolsets.extend(some_user_toolsets.value)
        else:
            toolsets.extend(self._user_toolsets)
            toolsets.extend(self._dynamic_toolsets)
            for cap_ts in cap_toolsets if cap_toolsets is not None else self._cap_toolsets:
                if isinstance(cap_ts, AbstractToolset):
                    toolsets.append(cap_ts)  # pyright: ignore[reportUnknownArgumentType]
                else:  # pragma: no cover — get_toolset() always returns AbstractToolset
                    toolsets.append(DynamicToolset(cap_ts))

        return toolsets

    @overload
    def _prepare_output_schema(self, output_type: None) -> _output.OutputSchema[OutputDataT]: ...

    @overload
    def _prepare_output_schema(
        self, output_type: OutputSpec[RunOutputDataT]
    ) -> _output.OutputSchema[RunOutputDataT]: ...

    def _prepare_output_schema(self, output_type: OutputSpec[Any] | None) -> _output.OutputSchema[Any]:
        if output_type is not None:
            if self._output_validators:
                raise exceptions.UserError('Cannot set a custom run `output_type` when the agent has output validators')
            schema = _output.OutputSchema.build(output_type)
        else:
            schema = self._output_schema

        return schema

    @asynccontextmanager
    async def realtime_session(
        self,
        model: RealtimeModel | KnownRealtimeModelName | str,
        *,
        deps: AgentDepsT = None,
        model_settings: RealtimeModelSettings | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        usage: _usage.RunUsage | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        conversation_id: str | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        audio_retention: AudioRetention = 'transcript_only',
    ) -> AsyncGenerator[RealtimeSession]:
        """Open a realtime speech-to-speech session backed by the agent's tools.

        The session connects to a realtime `model` and automatically executes tool calls using the
        agent's registered tools, sending the results back to the model.

        This mirrors [`run`][pydantic_ai.agent.AbstractAgent.run] / [`iter`][pydantic_ai.agent.AbstractAgent.iter]
        for the parameters that map to a long-lived, bidirectional session. Parameters that are
        specific to the request-response graph — `output_type`, `retries`, `event_stream_handler`,
        `deferred_tool_results` — do not apply; structured output should be delegated to a normal
        [`Agent`][pydantic_ai.Agent] (see the realtime docs). Of the
        `capabilities` lifecycle, only the **tool** hooks (`prepare_tools`,
        `before`/`after`/`wrap` `tool_validate`, and `before`/`after`/`wrap`/`on_error` `tool_execute`)
        run, since the session validates and executes tools but has no model-request/graph/output
        stages.

        Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.realtime.openai import OpenAIRealtimeModel

        agent = Agent(instructions='You are a helpful voice assistant.')

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'Sunny in {city}'

        async def main():
            rt = OpenAIRealtimeModel('gpt-realtime')
            async with agent.realtime_session(model=rt) as session:
                await session.send_audio(b'...')
                async for event in session:
                    print(event)
        ```

        Args:
            model: The realtime model to connect to.
            deps: Dependencies passed to tool functions.
            model_settings: Optional realtime settings overriding the model's defaults for this session.
            instructions: Additional instructions for this session, combined with the agent's
                instructions. Dynamic instruction functions (`@agent.instructions`) are evaluated
                once at connect time (there is no per-request rebuild in a realtime session).
            toolsets: Optional additional toolsets for this session, on top of the agent's.
            capabilities: Optional additional capabilities for this session. Only the tool-lifecycle
                hooks apply (see above); model-request/graph/output hooks are not invoked.
            usage: Optional [`RunUsage`][pydantic_ai.usage.RunUsage] to accumulate token usage into;
                exposed as `session.usage`. A fresh one is used when omitted.
            usage_limits: Optional [`UsageLimits`][pydantic_ai.usage.UsageLimits]. Token and
                tool-call limits are enforced as usage accrues; a breach raises
                [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] from the session's
                event iterator, matching how `run` / `iter` surface a usage limit.
            metadata: Optional metadata set on the [`RunContext`][pydantic_ai.tools.RunContext]
                available to tools and capabilities.
            conversation_id: Optional conversation id, set on the run context and the telemetry span
                so a realtime session can be correlated with other runs.
            message_history: Prior conversation to seed the session with. Replayable text, transcripts,
                thinking, tool rounds, images, and supported retained user audio are projected to the
                provider's initial conversation items; unrepresentable content raises `UserError`. The
                history is included in [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages]
                (but not `new_messages()`). Hand off from a prior session or a standard
                [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] by passing its messages here.
            audio_retention: How much spoken audio the session retains in its history, on top of
                transcripts. Defaults to `'transcript_only'` (drop audio bytes); see
                [`AudioRetention`][pydantic_ai.realtime.AudioRetention].
        """
        from ..realtime import RealtimeModel, RealtimeSession, infer_realtime_model

        if not isinstance(model, RealtimeModel):
            model = infer_realtime_model(model)

        deps = self._get_deps(deps)
        run_context = RunContext[AgentDepsT](
            deps=deps,
            agent=self,
            model=model,
            usage=usage if usage is not None else _usage.RunUsage(),
            model_settings=None,
            conversation_id=conversation_id,
            # A realtime session has no run identity yet (`run_id` stays unset); it gains one once
            # exchange-level hooks land and each exchange becomes an addressable unit.
            max_retries=self._max_tool_retries,
        )

        # Instrumentation: inject an `Instrumentation` capability (outermost) so tool spans flow through
        # `ToolManager.handle_call`'s `wrap_tool_execute` hook — the single, canonical source of tool
        # spans — exactly as `run`/`iter` do (via `_resolve_run_capabilities` below). A realtime model is
        # a `RealtimeModel`, never an `InstrumentedModel`, so there's no wrapped model to unwrap; the
        # settings come straight from `_resolve_instrumentation_settings()`. The helper skips injection if
        # the user already supplied an `Instrumentation` capability (agent- or call-level).
        extra_capabilities = wrap_capability_funcs(capabilities)
        instrumentation_settings = self._resolve_instrumentation_settings()
        instrumentation_cap = (
            InstrumentationCap(settings=instrumentation_settings) if instrumentation_settings is not None else None
        )

        # The session-level `realtime` span and per-response `chat` spans are hand-managed by
        # `RealtimeSession` (there are no realtime capability hooks yet to hang them on — those move onto
        # exchange-level capability hooks when they land). Until then, drive them from the settings that
        # will actually win: an explicit `Instrumentation` capability's (agent- or call-level) over the
        # `instrument=`-derived ones, matching the precedence `_resolve_run_capabilities` applies to the
        # tool spans.
        explicit_instrumentation = find_capability([self._root_capability, *extra_capabilities], InstrumentationCap)
        session_instrumentation_settings = (
            explicit_instrumentation.settings if explicit_instrumentation is not None else instrumentation_settings
        )
        # Mirror `iter`'s `RunContext`: expose the resolved tracer (a `NoOpTracer` when uninstrumented)
        # and content-tracing flag, and resolve metadata before `for_run` so capability/toolset hooks see
        # it (same ordering as the graph run).
        run_context.tracer = (
            session_instrumentation_settings.tracer if session_instrumentation_settings is not None else NoOpTracer()
        )
        run_context.trace_include_content = (
            session_instrumentation_settings is not None and session_instrumentation_settings.include_content
        )
        if session_instrumentation_settings is not None:
            run_context.instrumentation_version = session_instrumentation_settings.version
        run_context.metadata = self._get_metadata(run_context, metadata)

        # Resolve the capability layers and extract their contributions, exactly as `run`/`iter` do via
        # the shared helper. Realtime keeps its own surroundings: no `InstrumentedModel` unwrap, no
        # deferred loader (`inject_deferred_loader=False`), no root-capability override, once-only model
        # settings (below), and the `_keep_native` / `supported_native_tools` gate (below). Keep this in
        # sync with the `iter` call site.
        resolved_caps = await self._resolve_run_capabilities(
            run_context,
            base_capability=self._root_capability,
            extra_capabilities=extra_capabilities,
            instrumentation_cap=instrumentation_cap,
            inject_deferred_loader=False,
            base_is_override=False,
        )
        run_capability = resolved_caps.run_capability
        run_context.capabilities = resolved_caps.capabilities

        # Regular agent and capability model settings intentionally do not apply to realtime sessions.
        # A future capability hook dedicated to realtime settings can add that behavior deliberately.
        effective_model_settings: RealtimeModelSettings | None = model.settings.copy() if model.settings else None
        if model_settings:
            if effective_model_settings is None:
                effective_model_settings = model_settings.copy()
            else:
                effective_model_settings.update(model_settings)

        # Native (provider built-in) tools, e.g. via `capabilities=[NativeTool(WebSearchTool())]`. Only
        # concrete tools are forwarded; dynamic native-tool functions aren't resolved for realtime. The
        # auto-injected optional `ToolSearchTool` is dropped (mirroring the graph) — there's no
        # tool-search corpus and realtime providers don't support it. The helper already folded in
        # `override(native_tools=...)` and any per-call capability native tools.
        def _keep_native(tool: AgentNativeTool[AgentDepsT]) -> TypeGuard[AbstractNativeTool]:
            return isinstance(tool, AbstractNativeTool) and not (isinstance(tool, ToolSearchTool) and tool.optional)

        native_tools = [t for t in resolved_caps.native_tools if _keep_native(t)]

        # Validate the full native-tool set (capability-contributed and `override(native_tools=...)`)
        # against the model's declared support up front — mirroring the classic model's
        # `supported_native_tools` check — so an unsupported tool fails with a clear error here, before
        # connecting, rather than mid-session. This is the signal a caller or capability needs to fall
        # back (e.g. to a local tool); the session itself does not fall back automatically.
        model_profile = model.profile
        supported_native_tools = model_profile.get('supported_native_tools', frozenset())
        if unsupported_native_tools := [t for t in native_tools if not isinstance(t, tuple(supported_native_tools))]:
            unsupported = ', '.join(sorted(type(t).__name__ for t in unsupported_native_tools))
            supported = ', '.join(sorted(t.__name__ for t in supported_native_tools)) or 'none'
            raise exceptions.UserError(
                f'The {model.model_name!r} realtime model does not support the {unsupported} native tool(s). '
                f'Supported native tools: {supported}.'
            )

        toolset = self._get_toolset(
            output_toolset=None,
            additional_toolsets=toolsets,
            cap_toolsets=resolved_caps.toolsets,
            run_capability=run_capability,
        )
        toolset = await toolset.for_run(run_context)
        async with toolset:
            # A dedicated usage for tool execution keeps the session the single authority for
            # `session.usage.tool_calls` (counted as `ToolCall`s stream in); token usage still lands
            # on the session via provider `Usage` events.
            tool_context = dataclasses.replace(run_context, usage=_usage.RunUsage())
            tool_manager = await ToolManager[AgentDepsT](
                toolset, root_capability=run_capability, default_max_retries=self._max_tool_retries
            ).for_run_step(tool_context)
            tool_defs = tool_manager.tool_defs

            # Evaluate literal + dynamic instructions once, then fold in toolset-contributed
            # instructions, mirroring the run/iter graph. Capability-contributed instructions come from
            # the resolved capabilities (like `iter`), not just the init-time snapshot.
            literal, instruction_functions = self._get_instructions(
                additional_instructions=instructions, cap_instructions=resolved_caps.instructions
            )
            instruction_parts = [literal, *[await fn.run(run_context) for fn in instruction_functions]]
            instruction_parts.extend(
                part.content
                for part in _instructions.normalize_toolset_instructions(
                    await tool_manager.toolset.get_instructions(run_context)
                )
            )
            resolved_instructions = '\n'.join(part for part in instruction_parts if part).strip()
            request_messages = [
                *(message_history or ()),
                _messages.ModelRequest(parts=[], instructions=resolved_instructions or None),
            ]
            model_request_parameters = models.ModelRequestParameters(
                function_tools=tool_defs,
                native_tools=native_tools,
            )

            if message_history and not model_profile.get('supports_session_seeding', False):
                raise exceptions.UserError(
                    f'The {model.model_name!r} realtime model does not support seeding a session with '
                    '`message_history`.'
                )

            async with model.connect(
                messages=request_messages,
                model_settings=effective_model_settings,
                model_request_parameters=model_request_parameters,
            ) as connection:
                session = RealtimeSession(
                    connection,
                    tool_manager,
                    instrumentation=session_instrumentation_settings,
                    model_name=model.model_name,
                    agent_name=self.name,
                    usage=run_context.usage,
                    usage_limits=usage_limits,
                    audio_retention=audio_retention,
                    message_history=message_history,
                    profile=model_profile,
                    conversation_id=conversation_id,
                    output_modality=(effective_model_settings or {}).get('output_modality', 'audio'),
                )
                async with session:
                    yield session

    async def __aenter__(self) -> Self:
        """Enter the agent context.

        This will start all [`MCPToolset`s][pydantic_ai.mcp.MCPToolset] registered as `toolsets` so they are ready to be used,
        and enter the model so the provider's HTTP client will be closed cleanly on exit.

        This is a no-op if the agent has already been entered.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    toolset = self._get_toolset()
                    await exit_stack.enter_async_context(toolset)

                    capability = self._effective_root_capability()
                    capability_model = capability.get_model()
                    override_model = self._override_model.get()
                    if override_model is not None:
                        static_selection = override_model.value
                    elif callable(capability_model) and not _is_model(capability_model):
                        # Dynamic capability models are entered by the run that selects them.
                        static_selection = None
                    elif capability_model is not None:
                        static_selection = capability_model
                    else:
                        static_selection = self.model
                    if static_selection is not None and not (
                        isinstance(static_selection, str) and capability.has_resolve_model_id
                    ):
                        model = (
                            static_selection if _is_model(static_selection) else models.infer_model(static_selection)
                        )
                        await exit_stack.enter_async_context(model)
                        self._entered_model_ids.add(id(model))
                        if isinstance(static_selection, str):
                            self._entered_models_by_selection[id(capability), static_selection] = model

                    self._exit_stack = exit_stack.pop_all()
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack is not None:
                try:
                    await self._exit_stack.aclose()
                finally:
                    self._exit_stack = None
                    self._entered_model_ids.clear()
                    self._entered_models_by_selection.clear()

    def set_mcp_sampling_model(self, model: models.Model | models.KnownModelName | str | None = None) -> None:
        """Set the sampling model on all [`MCPToolset`s][pydantic_ai.mcp.MCPToolset] registered with the agent.

        If no sampling model is provided, the agent's model will be used.
        """
        try:
            sampling_model = models.infer_model(model) if model else self._get_model_outside_run()
        except exceptions.UserError as e:
            capability = self._effective_root_capability()
            if model is None and (callable(capability.get_model()) or capability.has_resolve_model_id):
                raise exceptions.UserError(
                    'The capability model requires run dependencies and cannot be used for MCP sampling setup. '
                    'Pass a concrete model explicitly.'
                ) from e
            raise exceptions.UserError('No sampling model provided and no model set on the agent.') from e

        from ..mcp import MCPToolset

        def _set_sampling_model(toolset: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(toolset, MCPToolset):
                toolset.set_sampling_model(sampling_model)

        self._get_toolset().apply(_set_sampling_model)

    def to_web(
        self,
        *,
        models: ModelsParam = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        instructions: str | None = None,
        html_source: str | Path | None = None,
    ) -> Starlette:
        """Create a Starlette app that serves a web chat UI for this agent.

        This method returns a pre-configured Starlette application that provides a web-based
        chat interface for interacting with the agent. By default, the UI is fetched from a
        CDN and cached on first use.

        The returned Starlette application can be mounted into a FastAPI app or run directly
        with any ASGI server (uvicorn, hypercorn, etc.).

        Note that the `deps` and `model_settings` will be the same for each request.
        To provide different `deps` for each request use the lower-level adapters directly.

        The agent's configured native tools (registered via `capabilities=[NativeTool(...)]`
        or higher-level capabilities like `WebSearch()`) are automatically exposed as
        options in the UI.

        Args:
            models: Additional models to make available in the UI. Can be:
                - A sequence of model names/instances (e.g., `['openai:gpt-5', 'anthropic:claude-sonnet-4-6']`)
                - A dict mapping display labels to model names/instances
                  (e.g., `{'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-6'}`)
                The agent's model is always included. Native tool support is automatically
                determined from each model's profile.
            deps: Optional dependencies to use for all requests.
            model_settings: Optional settings to use for all model requests.
            instructions: Optional extra instructions to pass to each agent run.
            html_source: Path or URL for the chat UI HTML. Can be:
                - None (default): Fetches from CDN and caches locally
                - A Path instance: Reads from the local file
                - A URL string (http:// or https://): Fetches from the URL
                - A file path string: Reads from the local file

        Returns:
            A configured Starlette application ready to be served (e.g., with uvicorn)

        Example:
            ```python
            from pydantic_ai import Agent
            from pydantic_ai.capabilities import NativeTool
            from pydantic_ai.native_tools import WebSearchTool

            agent = Agent('openai:gpt-5', capabilities=[NativeTool(WebSearchTool())])

            # Simple usage - uses agent's model and native tools
            app = agent.to_web()

            # Or provide additional models for UI selection
            app = agent.to_web(models=['openai:gpt-5', 'anthropic:claude-sonnet-4-6'])

            # Then run with: uvicorn app:app --reload
            ```
        """
        from ..ui._web import create_web_app

        return create_web_app(
            self,
            models=models,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
            html_source=html_source,
        )


def _merge_retries_with_spec(
    explicit: int | AgentRetries | None,
    spec: AgentSpec,
) -> AgentRetries | None:
    """Merge an explicit `retries=` value with the retry fields on an `AgentSpec`.

    Explicit kwarg keys win over spec keys.
    """
    merged = _retry_overrides_from_spec(spec)
    merged.update(_normalize_agent_retry_overrides(explicit))
    if not merged:
        return None
    return merged


def _retry_overrides_from_spec(spec: AgentSpec) -> AgentRetries:
    """Return retry fields explicitly configured on an `AgentSpec`."""
    return _normalize_agent_retry_overrides(spec.retries) if 'retries' in spec.model_fields_set else {}


_UNSUPPORTED_SPEC_FIELDS: tuple[str, ...] = (
    'description',
    'end_strategy',
    'tool_timeout',
    'output_schema',
    'deps_schema',
)
"""AgentSpec fields that are not supported at run/override time."""

_AUTO_INJECT_CAPABILITY_TYPES: tuple[type[AbstractCapability[Any]], ...] = (
    ToolSearchCap,
    PendingMessageDrainCapability,
)
"""Infrastructure capabilities auto-injected when not already present."""


def _inject_auto_capabilities(capabilities: list[AbstractCapability[Any]]) -> None:
    """Ensure all auto-injected infrastructure capabilities are present.

    Each capability's own `CapabilityOrdering` (e.g. `position='outermost'`)
    determines its final placement, so insertion order here doesn't matter.
    """
    for cap_type in _AUTO_INJECT_CAPABILITY_TYPES:
        if not has_capability_type(capabilities, cap_type):
            capabilities.append(cap_type())


def _validate_capability_ids(capabilities: Sequence[AbstractCapability[Any]]) -> set[str]:
    """Validate capability `id`s and return the set of explicit ones.

    Rejects deferred capabilities that lack an explicit `id` and explicit ids used by more than
    one capability. Shared by two call sites: construction-time validation over the
    statically-provided capabilities (so misconfiguration fails fast in `Agent(...)` rather than
    on the first run), and run-time assembly in `_build_run_capabilities`, which also covers
    capabilities supplied per-run or returned by `for_run` and so can't be checked at construction.
    """
    explicit_ids: set[str] = set()
    for cap in capabilities:
        if cap.defer_loading is True and cap.id is None:
            raise exceptions.UserError(
                'Deferred capabilities must use stable explicit `id` values. '
                'Pass `id=...` when using `defer_loading=True`.'
            )
        if cap.id is None:
            continue
        if cap.id in explicit_ids:
            raise exceptions.UserError(
                f'Capability id {cap.id!r} is used by multiple capabilities. '
                'Capability ids must be unique within a run.'
            )
        explicit_ids.add(cap.id)
    return explicit_ids


def _validate_native_tool_ids(native_tools: Sequence[AgentNativeTool[Any]], *, source: str) -> None:
    """Reject native tools that share a `unique_id` but carry conflicting definitions.

    Native tools are keyed by `unique_id` when request parameters are deduplicated (see
    `Model.prepare_request`). That dedup is intentionally last-wins *across* layers, so a run-level
    native tool can override an agent-level default with the same id. *Within* a single layer,
    though, two different tools sharing an id are ambiguous: the silent last-wins would bind a
    stable id (e.g. an `MCPServerTool` id) to an unexpected definition such as a different server
    URL or authorization token. Fail fast here instead. Identical duplicates are allowed and
    collapsed later.

    `NativeToolFunc` callables are skipped: they have no stable `unique_id` to key on.
    """
    seen: dict[str, AbstractNativeTool] = {}
    for tool in native_tools:
        if not isinstance(tool, AbstractNativeTool):
            continue
        existing = seen.setdefault(tool.unique_id, tool)
        if existing is not tool and existing != tool:
            raise exceptions.UserError(
                f'Native tool id {tool.unique_id!r} maps to conflicting definitions in {source}. '
                'Native tool ids must be unique within a capability layer.'
            )


@dataclasses.dataclass
class _ResolvedRunCapabilities(Generic[AgentDepsT]):
    """The per-run capability state shared by `run`/`iter` and `realtime_session`.

    Produced by [`Agent._resolve_run_capabilities`][]: the resolved capability tree plus the
    contributions extracted from it (instructions, native tools, model settings, toolsets), so both a
    graph run and a realtime session wire capabilities up identically. See the cross-references on the
    two call sites for the surrounding logic each keeps to itself.
    """

    run_capability: AbstractCapability[AgentDepsT]
    capabilities: dict[str, AbstractCapability[AgentDepsT]]
    instructions: list[str | SystemPromptFunc[AgentDepsT]] | None
    native_tools: list[AgentNativeTool[AgentDepsT]]
    model_settings: AgentModelSettings[AgentDepsT] | None
    toolsets: list[AgentToolset[AgentDepsT]] | None
    resolved_layers: list[AbstractCapability[AgentDepsT]]
    """Each run layer after `for_run`, in order (instrumentation first when injected). The graph run
    compares the model-layer slice against its pre-resolution `model_layers` to detect whether any
    capability changed the model contribution during resolution (`model_layers_unchanged`)."""


def _layer_model_settings(
    run_context: RunContext[AgentDepsT],
    layers: Sequence[AgentModelSettings[AgentDepsT] | None],
    *,
    base: ModelSettings | None = None,
) -> ModelSettings | None:
    """Merge model-settings layers left-to-right, stamping `run_context.model_settings` before each.

    Each layer is a static `ModelSettings`, a callable resolved against the run context, or `None`.
    Stamping the merged-so-far onto `run_context.model_settings` before a callable layer runs lets it
    observe the previous layers — the agent -> capability -> run order both `iter` (per model-request
    step) and `realtime_session` (once, at connect) rely on. `base` is the model's own settings for a
    graph run; a realtime model has none, so it defaults to `None`.
    """
    merged = base
    run_context.model_settings = merged
    for layer in layers:
        resolved = layer(run_context) if callable(layer) else layer
        merged = merge_model_settings(merged, resolved)
        run_context.model_settings = merged
    return merged


def _build_run_capabilities(capability: AbstractCapability[AgentDepsT]) -> dict[str, AbstractCapability[AgentDepsT]]:
    capabilities: list[AbstractCapability[AgentDepsT]] = []
    capability.apply(capabilities.append)

    explicit_ids = _validate_capability_ids(capabilities)

    by_id: dict[str, AbstractCapability[AgentDepsT]] = {}
    for cap in capabilities:
        capability_id = cap.id
        if capability_id is None:
            base_id = to_snake(type(cap).__name__)
            capability_id = base_id
            suffix = 2
            while capability_id in by_id or capability_id in explicit_ids:
                capability_id = f'{base_id}_{suffix}'
                suffix += 1

        by_id[capability_id] = cap

    return by_id


def _validate_spec(
    spec: dict[str, Any] | AgentSpec,
    deps_type: type[Any],
) -> tuple[AgentSpec, dict[str, Any]]:
    """Validate a spec dict/object and build the template context.

    Shared by `Agent.from_spec()` and `Agent._resolve_spec()`.

    Returns:
        A tuple of (validated_spec, template_context).
    """
    template_context: dict[str, Any] = {
        'deps_type': deps_type if deps_type is not type(None) else None,
    }
    if isinstance(spec, dict):
        validated_spec = AgentSpec.model_validate(spec, context=template_context)
    else:
        validated_spec = spec
    template_context['deps_schema'] = validated_spec.deps_schema
    return validated_spec, template_context


def _capabilities_from_spec(
    spec: AgentSpec,
    custom_capability_types: Sequence[type[AbstractCapability[Any]]],
    template_context: dict[str, Any],
) -> list[AbstractCapability[Any]]:
    """Instantiate capabilities from an AgentSpec using the capability registry.

    Shared by `Agent.from_spec()` and `Agent._resolve_spec()`.
    """
    from pydantic_ai.agent import spec as _agent_spec

    registry = get_capability_registry(custom_capability_types)

    def _instantiate_cap(
        cap_cls: type[AbstractCapability[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AbstractCapability[Any]:
        args, kwargs = validate_from_spec_args(cap_cls, args, kwargs, template_context)
        return cap_cls.from_spec(*args, **kwargs)

    # Set context so nested from_spec calls (e.g. PrefixTools) can reuse the registry
    ctx = _agent_spec.CapabilitySpecContext(registry=registry, instantiate=_instantiate_cap)
    token = _agent_spec.capability_spec_context.set(ctx)
    try:
        capabilities: list[AbstractCapability[Any]] = []
        for cap_spec in spec.capabilities:
            capability = load_from_registry(
                registry,
                cap_spec,
                label='capability',
                custom_types_param='custom_capability_types',
                instantiate=_instantiate_cap,
            )
            capabilities.append(capability)
        return capabilities
    finally:
        _agent_spec.capability_spec_context.reset(token)


@dataclasses.dataclass(init=False)
class _AgentFunctionToolset(FunctionToolset[AgentDepsT]):
    output_schema: _output.OutputSchema[Any]

    def __init__(
        self,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
        *,
        max_retries: int | None = None,
        timeout: float | None = None,
        id: str | None = None,
        output_schema: _output.OutputSchema[Any],
    ):
        self.output_schema = output_schema
        super().__init__(tools, max_retries=max_retries, timeout=timeout, id=id)

    @property
    def id(self) -> str:
        return '<agent>'

    @property
    def label(self) -> str:
        return 'the agent'
