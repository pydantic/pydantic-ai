from __future__ import annotations as _annotations

import asyncio
import contextvars
import dataclasses
import functools
import inspect
import json
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

import anyio
from opentelemetry.baggage import set_baggage as _otel_set_baggage
from opentelemetry.context import attach as _otel_attach, detach as _otel_detach
from opentelemetry.trace import NoOpTracer, use_span
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Self, TypeVar, deprecated

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION, InstrumentationNames
from pydantic_ai._spec import load_from_registry

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
    HistoryProcessor,
    ModelRequestNode,
    UserPromptNode,
    build_run_context,
    capture_run_messages,
)
from .._instructions import AgentInstructions
from .._output import OutputToolset
from .._template import TemplateStr, validate_from_spec_args
from ..builtin_tools import AbstractBuiltinTool
from ..capabilities import AbstractCapability, AgentCapability, CombinedCapability
from ..capabilities._dynamic import wrap_capability_funcs
from ..capabilities._ordering import has_capability_type
from ..capabilities._tool_search import ToolSearch as ToolSearchCap
from ..capabilities.builtin_tool import BuiltinTool as BuiltinToolCap
from ..capabilities.prepare_tools import PrepareOutputTools, PrepareTools
from ..capabilities.process_history import ProcessHistory
from ..models.instrumented import InstrumentationSettings, InstrumentedModel, instrument_model
from ..output import OutputDataT, OutputSpec, StructuredDict
from ..run import AgentRun, AgentRunResult
from ..settings import ModelSettings, merge_model_settings
from ..tool_manager import ParallelExecutionMode, ToolManager
from ..tools import (
    AgentBuiltinTool,
    AgentDepsT,
    ArgsValidatorFunc,
    BuiltinToolFunc,
    DeferredToolResults,
    DocstringFormat,
    GenerateToolJsonSchema,
    RunContext,
    Tool,
    ToolDefinition,
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
from ..toolsets.combined import CombinedToolset
from ..toolsets.function import FunctionToolset
from ..toolsets.prepared import PreparedToolset
from .abstract import (
    AbstractAgent,
    AgentMetadata,
    AgentModelSettings,
    EventStreamHandler,
    EventStreamProcessor,
    RunOutputDataT,
)
from .spec import AgentSpec, get_capability_registry
from .wrapper import WrapperAgent

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from pydantic_graph import GraphRunContext

    from ..builtin_tools import AbstractBuiltinTool
    from ..mcp import MCPServer
    from ..ui._web import ModelsParam

__all__ = (
    'AbstractAgent',
    'Agent',
    'AgentModelSettings',
    'AgentRun',
    'AgentRunResult',
    'BuiltinToolFunc',
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
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)


@dataclasses.dataclass
class _ResolvedSpec:
    """Result of resolving an AgentSpec for use at run/override time."""

    capability: CombinedCapability[Any] | None
    instructions: list[str | _system_prompt.SystemPromptFunc[Any]]
    model: str | None
    model_settings: ModelSettings | None
    metadata: dict[str, Any] | None
    name: str | None


@dataclasses.dataclass(init=False)
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the output type they return, [`OutputDataT`][pydantic_ai.output.OutputDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

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
    """The strategy for handling multiple tool calls when a final result is found.

    - `'early'` (default): Output tools are executed first. Once a valid final result is found, remaining function and output tool calls are skipped
    - `'graceful'`: Output tools are executed first. Once a valid final result is found, remaining output tool calls are skipped, but function tools are still executed
    - `'exhaustive'`: Output tools are executed first, then all function tools are executed. The first valid output tool result becomes the final output
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

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False
    _metadata: AgentMetadata[AgentDepsT] | None = dataclasses.field(repr=False)

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _output_schema: _output.OutputSchema[OutputDataT] = dataclasses.field(repr=False)
    _output_validators: list[_output.OutputValidator[AgentDepsT, OutputDataT]] = dataclasses.field(repr=False)
    _instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_toolset: FunctionToolset[AgentDepsT] = dataclasses.field(repr=False)
    _output_toolset: OutputToolset[AgentDepsT] | None = dataclasses.field(repr=False)
    _user_toolsets: list[AbstractToolset[AgentDepsT]] = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)
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

    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int = 1,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
    ) -> None: ...

    @overload
    @deprecated('`mcp_servers` is deprecated, use `toolsets` instead.')
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: AgentInstructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int = 1,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        mcp_servers: Sequence[MCPServer] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
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
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        description: TemplateStr[AgentDepsT] | str | None = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        retries: int = 1,
        validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Any,
    ):
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
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            description: A human-readable description of the agent, attached to the agent run span as
                `gen_ai.agent.description` when instrumentation is enabled.
            model_settings: Optional model request settings to use for this agent's runs, by default.
                Can be a static `ModelSettings` dict or a callable that takes a
                [`RunContext`][pydantic_ai.tools.RunContext] and returns `ModelSettings`.
                Callables are called before each model request, allowing dynamic per-step settings.
            retries: The default number of retries to allow for tool calls and output validation, before raising an error.
                For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
            validation_context: Pydantic [validation context](https://docs.pydantic.dev/latest/concepts/validators/#validation-context) used to validate tool arguments and outputs.
            output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.agent.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain].
            builtin_tools: The builtin tools that the agent will use. This depends on the model, as some models may not
                support certain tools. If the model doesn't support the builtin tools, an error will be raised.
            prepare_tools: Custom function to prepare the tool definition of all tools for each step, except output tools.
                This is useful if you want to customize the definition of multiple tools or you want to register
                a subset of tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            prepare_output_tools: Custom function to prepare the tool definition of all output tools for each step.
                This is useful if you want to customize the definition of multiple output tools or you want to register
                a subset of output tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            toolsets: Toolsets to register with the agent, including MCP servers and functions which take a run context
                and return a toolset. See [`ToolsetFunc`][pydantic_ai.toolsets.ToolsetFunc] for more information.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.agent.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            instrument: Set to True to automatically instrument with OpenTelemetry,
                which will use Logfire if it's configured.
                Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
                If this isn't set, then the last value set by
                [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
                will be used, which defaults to False.
                See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
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
            history_processors: Optional list of callables to process the message history before sending it to the model.
                Each processor takes a list of messages and returns a modified list of messages.
                Processors can be sync or async and are applied in sequence.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools.
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
        if model is None or defer_model_check:
            self._model = model
        else:
            self._model = models.infer_model(model)

        self._name = name
        self._description = description
        self.end_strategy = end_strategy

        self.history_processors: list[HistoryProcessor[AgentDepsT]] = list(history_processors or [])

        capabilities = wrap_capability_funcs(capabilities)
        for history_processor in self.history_processors:
            capabilities.append(ProcessHistory(history_processor))
        for builtin_tool in builtin_tools:
            capabilities.append(BuiltinToolCap(builtin_tool))
        if prepare_tools is not None:
            capabilities.append(PrepareTools(prepare_tools))
        if prepare_output_tools is not None:
            capabilities.append(PrepareOutputTools(prepare_output_tools))

        _inject_auto_capabilities(capabilities)

        self._root_capability = CombinedCapability(capabilities)

        self.model_settings = model_settings

        self._output_type = output_type
        self.instrument = instrument
        self._metadata = metadata
        self._deps_type = deps_type

        if mcp_servers := _deprecated_kwargs.pop('mcp_servers', None):
            if toolsets is not None:  # pragma: no cover
                raise TypeError('`mcp_servers` and `toolsets` cannot be set at the same time.')
            warnings.warn('`mcp_servers` is deprecated, use `toolsets` instead', DeprecationWarning)
            toolsets = mcp_servers

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        self._output_schema = _output.OutputSchema[OutputDataT].build(output_type)
        self._output_validators = []

        self._instructions = _instructions.normalize_instructions(instructions)
        self._cap_instructions = _instructions.normalize_instructions(self._root_capability.get_instructions())

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        self._max_result_retries = output_retries if output_retries is not None else retries
        self._max_tool_retries = retries
        self._tool_timeout = tool_timeout

        self._validation_context = validation_context

        self._cap_builtin_tools = list(self._root_capability.get_builtin_tools())

        self._cap_model_settings = self._root_capability.get_model_settings()

        self._output_toolset = self._output_schema.toolset
        if self._output_toolset and self._output_toolset.max_retries is None:
            self._output_toolset.max_retries = self._max_result_retries

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

        # Capability-contributed toolsets (stored separately for per-run re-extraction)
        cap_toolset = self._root_capability.get_toolset()
        self._cap_toolsets: list[AgentToolset[AgentDepsT]] = [cap_toolset] if cap_toolset is not None else []

        self._event_stream_handler = event_stream_handler

        self._concurrency_limiter = _concurrency.normalize_to_limiter(max_concurrency)

        self._override_name: ContextVar[_utils.Option[str]] = ContextVar('_override_name', default=None)
        self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
        self._override_model: ContextVar[_utils.Option[models.Model]] = ContextVar('_override_model', default=None)
        self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
            '_override_toolsets', default=None
        )
        self._override_tools: ContextVar[
            _utils.Option[Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]]
        ] = ContextVar('_override_tools', default=None)
        self._override_builtin_tools: ContextVar[_utils.Option[Sequence[AgentBuiltinTool[AgentDepsT]]]] = ContextVar(
            '_override_builtin_tools', default=None
        )
        self._override_instructions: ContextVar[
            _utils.Option[list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]]
        ] = ContextVar('_override_instructions', default=None)
        self._override_metadata: ContextVar[_utils.Option[AgentMetadata[AgentDepsT]]] = ContextVar(
            '_override_metadata', default=None
        )
        self._override_model_settings: ContextVar[_utils.Option[AgentModelSettings[AgentDepsT]]] = ContextVar(
            '_override_model_settings', default=None
        )
        self._override_root_capability: ContextVar[_utils.Option[CombinedCapability[AgentDepsT]]] = ContextVar(
            '_override_root_capability', default=None
        )
        self._entered_count = 0
        self._exit_stack = None

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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[None, str]: ...

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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
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
            retries: Default retries for tool calls and output validation, overrides spec `retries` if provided.
            validation_context: Pydantic validation context for tool arguments and outputs.
            output_retries: Max retries for output validation, overrides spec `output_retries` if provided.
            tools: Tools to register with the agent.
            builtin_tools: Builtin tools for the agent.
            prepare_tools: Custom function to prepare tool definitions.
            prepare_output_tools: Custom function to prepare output tool definitions.
            toolsets: Toolsets to register with the agent.
            defer_model_check: Defer model evaluation until first run.
            end_strategy: Strategy for tool calls alongside a final result, overrides spec `end_strategy` if provided.
            instrument: Instrumentation settings, overrides spec `instrument` if provided.
            metadata: Metadata to store with each run, overrides spec `metadata` if provided.
            history_processors: Processors for message history.
            event_stream_handler: Handler for streaming events.
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

        return Agent(
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
            retries=retries if retries is not None else validated_spec.retries,
            validation_context=validation_context,
            output_retries=output_retries if output_retries is not None else validated_spec.output_retries,
            tools=tools,
            builtin_tools=builtin_tools,
            prepare_tools=prepare_tools,
            prepare_output_tools=prepare_output_tools,
            toolsets=toolsets,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy if end_strategy is not None else validated_spec.end_strategy,
            instrument=instrument if instrument is not None else validated_spec.instrument,
            metadata=metadata if metadata is not None else validated_spec.metadata,
            history_processors=history_processors,
            event_stream_handler=event_stream_handler,
            tool_timeout=tool_timeout if tool_timeout is not None else validated_spec.tool_timeout,
            max_concurrency=max_concurrency,
            capabilities=all_capabilities,
        )

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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: _concurrency.AnyConcurrencyLimit = None,
        capabilities: Sequence[AgentCapability[Any]] | None = None,
    ) -> Agent[None, str]: ...

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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
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
        retries: int | None = None,
        validation_context: Any = None,
        output_retries: int | None = None,
        tools: Sequence[Tool[Any] | ToolFuncEither[Any, ...]] = (),
        builtin_tools: Sequence[AgentBuiltinTool[Any]] = (),
        prepare_tools: ToolsPrepareFunc[Any] | None = None,
        prepare_output_tools: ToolsPrepareFunc[Any] | None = None,
        toolsets: Sequence[AgentToolset[Any]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy | None = None,
        instrument: InstrumentationSettings | bool | None = None,
        metadata: AgentMetadata[Any] | None = None,
        history_processors: Sequence[HistoryProcessor[Any]] | None = None,
        event_stream_handler: EventStreamHandler[Any] | None = None,
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
        return cls.from_spec(
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
            output_retries=output_retries,
            tools=tools,
            builtin_tools=builtin_tools,
            prepare_tools=prepare_tools,
            prepare_output_tools=prepare_output_tools,
            toolsets=toolsets,
            defer_model_check=defer_model_check,
            end_strategy=end_strategy,
            instrument=instrument,
            metadata=metadata,
            history_processors=history_processors,
            event_stream_handler=event_stream_handler,
            tool_timeout=tool_timeout,
            max_concurrency=max_concurrency,
            capabilities=capabilities,
        )

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all agents where `instrument` is not set."""
        Agent._instrument_default = instrument

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
        return f'{type(self).__name__}(model={self.model!r}, name={self.name!r}, end_strategy={self.end_strategy!r}, model_settings={self.model_settings!r}, output_type={self.output_type!r}, instrument={self.instrument!r})'

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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
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
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        # Resolve spec contributions (additive at run time)
        resolved = self._resolve_spec(spec)
        if resolved is not None:
            # Model: spec as fallback (run param > spec > agent)
            if model is None and resolved.model is not None:
                model = resolved.model
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

        model_used = self._get_model(model)
        del model

        deps = self._get_deps(deps)
        output_schema = self._prepare_output_schema(output_type)

        output_type_ = output_type or self.output_type

        # We consider it a user error if a user tries to restrict the result type while having an output validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        output_validators = self._output_validators

        output_toolset = self._output_toolset
        if output_schema != self._output_schema or output_validators:
            output_toolset = output_schema.toolset
            if output_toolset:
                if output_toolset.max_retries is None:
                    output_toolset.max_retries = self._max_result_retries
                output_toolset.output_validators = output_validators

        # Build the graph
        graph = _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)

        # Build the initial state
        usage = usage or _usage.RunUsage()
        state = _agent_graph.GraphAgentState(
            message_history=list(message_history) if message_history else [],
            usage=usage,
            retries=0,
            run_step=0,
            conversation_id=_agent_graph.resolve_conversation_id(conversation_id, message_history),
        )

        # Build a resolver that computes model settings per-step, in order of precedence: run > agent > model
        model_settings_override = self._override_model_settings.get()
        agent_model_settings = (
            model_settings_override.value if model_settings_override is not None else self.model_settings
        )
        run_model_settings = model_settings if model_settings_override is None else None

        usage_limits = usage_limits or _usage.UsageLimits()

        if isinstance(model_used, InstrumentedModel):
            instrumentation_settings = model_used.instrumentation_settings
            tracer = model_used.instrumentation_settings.tracer
        else:
            instrumentation_settings = None
            tracer = NoOpTracer()

        # Build initial RunContext for for_run lifecycle hooks
        initial_ctx = RunContext[AgentDepsT](
            deps=deps,
            agent=self,
            model=model_used,
            usage=usage,
            prompt=user_prompt,
            messages=state.message_history,
            tracer=tracer,
            run_step=0,
        )

        # Determine root capability: override > agent default
        override_cap = self._override_root_capability.get()
        base_capability = override_cap.value if override_cap is not None else self._root_capability

        # Merge spec and run-time capabilities additively with the base capability
        extra_capabilities: list[AbstractCapability[AgentDepsT]] = []
        if resolved is not None and resolved.capability is not None:
            extra_capabilities.append(resolved.capability)
        extra_capabilities.extend(wrap_capability_funcs(capabilities))
        if extra_capabilities:
            effective_capability = CombinedCapability([base_capability, *extra_capabilities])
        else:
            effective_capability = base_capability

        # Per-run capability: re-extract get_*() if for_run returns a different instance
        run_capability = await effective_capability.for_run(initial_ctx)
        cap_toolsets: list[AgentToolset[AgentDepsT]] | None

        if run_capability is not effective_capability:
            source_cap = run_capability
        elif override_cap is not None or extra_capabilities:
            source_cap = effective_capability
        else:
            source_cap = None

        if source_cap is not None:
            cap_instructions = _instructions.normalize_instructions(source_cap.get_instructions())
            cap_builtin_tools = list(source_cap.get_builtin_tools())
            cap_model_settings = source_cap.get_model_settings()
            cap_ts = source_cap.get_toolset()
            cap_toolsets = [cap_ts] if cap_ts is not None else []
        else:
            cap_instructions = None  # use init-time defaults
            cap_builtin_tools = self._cap_builtin_tools
            cap_model_settings = self._cap_model_settings
            cap_toolsets = None

        if some_builtin_tools := self._override_builtin_tools.get():
            cap_builtin_tools = list(some_builtin_tools.value)

        # Build model settings resolver using per-run capability
        def get_model_settings(run_context: RunContext[AgentDepsT]) -> ModelSettings | None:
            # Resolve settings in layers, each merged on top of the previous.
            # Before calling each callable, set run_context.model_settings so it
            # can see the merged result of all previous layers.
            merged = model_used.settings

            run_context.model_settings = merged
            resolved_agent = (
                agent_model_settings(run_context) if callable(agent_model_settings) else agent_model_settings
            )
            merged = merge_model_settings(merged, resolved_agent)

            # Capability settings (from custom capabilities that override get_model_settings), cached at init
            run_context.model_settings = merged
            cap_settings = cap_model_settings
            resolved_cap = cap_settings(run_context) if callable(cap_settings) else cap_settings
            merged = merge_model_settings(merged, resolved_cap)

            run_context.model_settings = merged
            resolved_run = run_model_settings(run_context) if callable(run_model_settings) else run_model_settings
            merged = merge_model_settings(merged, resolved_run)

            run_context.model_settings = merged
            return merged

        # Build toolset with per-run capability contributions
        toolset = self._get_toolset(
            output_toolset=output_toolset,
            additional_toolsets=toolsets,
            cap_toolsets=cap_toolsets,
            run_capability=run_capability,
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

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT](
            user_deps=deps,
            agent=self,
            prompt=user_prompt,
            new_message_index=len(message_history) if message_history else 0,
            resumed_request=None,
            model=model_used,
            get_model_settings=get_model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            output_schema=output_schema,
            output_validators=output_validators,
            validation_context=self._validation_context,
            root_capability=run_capability,
            builtin_tools=[*cap_builtin_tools, *(builtin_tools or [])],
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
        instrumentation_names = InstrumentationNames.for_version(
            instrumentation_settings.version if instrumentation_settings else DEFAULT_INSTRUMENTATION_VERSION
        )

        span_attributes: dict[str, str] = {
            'model_name': model_used.model_name if model_used else 'no-model',
            'agent_name': agent_name,
            'gen_ai.agent.name': agent_name,
            'gen_ai.agent.call.id': state.run_id,
            'gen_ai.conversation.id': state.conversation_id,
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': f'{agent_name} run',
        }
        if self._description is not None:
            if isinstance(self._description, TemplateStr):
                span_attributes['gen_ai.agent.description'] = self._description.render(deps)
            else:
                span_attributes['gen_ai.agent.description'] = self._description

        run_span = tracer.start_span(
            instrumentation_names.get_agent_run_span_name(agent_name),
            attributes=span_attributes,
        )
        run_metadata: dict[str, Any] | None = None
        try:
            async with AsyncExitStack() as stack:
                if run_span.is_recording():
                    ctx = _otel_set_baggage('gen_ai.agent.name', agent_name)
                    ctx = _otel_set_baggage('gen_ai.agent.call.id', state.run_id, context=ctx)
                    ctx = _otel_set_baggage('gen_ai.conversation.id', state.conversation_id, context=ctx)
                    token = _otel_attach(ctx)
                    stack.callback(_otel_detach, token)
                await stack.enter_async_context(
                    _concurrency.get_concurrency_context(self._concurrency_limiter, f'agent:{agent_name}')
                )
                graph_run = await stack.enter_async_context(
                    graph.iter(
                        inputs=user_prompt_node,
                        state=state,
                        deps=graph_deps,
                        span=use_span(run_span) if run_span.is_recording() else None,
                        infer_name=False,
                    )
                )
                await stack.enter_async_context(toolset)
                agent_run = AgentRun(graph_run)
                run_metadata = self._resolve_and_store_metadata(agent_run.ctx, metadata)

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
                await asyncio.wait({_ready_waiter, _wrap_task}, return_when=asyncio.FIRST_COMPLETED)
                _ready_waiter.cancel()

                # Propagate context vars set by wrap_run/before_run to
                # the outer task so that agent_run.next() (and therefore
                # node hooks) can see them.
                _context_tokens: list[tuple[ContextVar[Any], contextvars.Token[Any]]] = []
                # Note: indexing instead of tuple unpacking because pyright
                # can't resolve types through nonlocal + Optional unpacking.
                for _cv_pair in _wrap_context or ():
                    _context_tokens.append((_cv_pair[0], _cv_pair[0].set(_cv_pair[1])))

                async def _finalize_result(r: AgentRunResult[Any]) -> None:
                    """Call after_run, store the result override, and clear any pending error."""
                    nonlocal _run_error
                    r = await run_capability.after_run(run_ctx, result=r)
                    agent_run._result_override = r  # pyright: ignore[reportPrivateUsage]
                    _run_error = None

                try:
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
                            run_metadata = self._resolve_and_store_metadata(agent_run.ctx, metadata)
                        else:
                            run_metadata = graph_run.state.metadata

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
                                    if (
                                        not isinstance(_wrap_exc, asyncio.CancelledError)
                                        and _wrap_exc is not _run_error
                                    ):
                                        _run_error.__context__ = _wrap_exc  # pragma: no cover — only fires for bugs in wrap_run implementations
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
                finally:
                    # Always restore context vars, even on
                    # GeneratorExit/KeyboardInterrupt.
                    for _var, _token in _context_tokens:
                        _var.reset(_token)

                final_result = agent_run.result
                if (
                    instrumentation_settings
                    and instrumentation_settings.include_content
                    and run_span.is_recording()
                    and final_result is not None
                ):
                    run_span.set_attribute(
                        'final_result',
                        (
                            final_result.output
                            if isinstance(final_result.output, str)
                            else json.dumps(InstrumentedModel.serialize_any(final_result.output))
                        ),
                    )
        finally:
            try:
                if instrumentation_settings and run_span.is_recording():
                    run_span.set_attributes(
                        self._run_span_end_attributes(
                            instrumentation_settings,
                            usage,
                            state.message_history,
                            graph_deps.new_message_index,
                            run_metadata,
                            model_request_parameters=state.last_model_request_parameters,
                        )
                    )
            finally:
                run_span.end()

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

    def _run_span_end_attributes(
        self,
        settings: InstrumentationSettings,
        usage: _usage.RunUsage,
        message_history: list[_messages.ModelMessage],
        new_message_index: int,
        metadata: dict[str, Any] | None = None,
        model_request_parameters: models.ModelRequestParameters | None = None,
    ) -> dict[str, str | int | float | bool]:
        if settings.version == 1:
            attrs = {
                'all_messages_events': json.dumps(
                    [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(message_history)]
                )
            }
        else:
            last_instructions = InstrumentedModel._get_instructions(message_history, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
            attrs: dict[str, Any] = {
                'pydantic_ai.all_messages': json.dumps(settings.messages_to_otel_messages(list(message_history))),
                **settings.system_instructions_attributes(last_instructions),
            }

            # If this agent run was provided with existing history, store an attribute indicating the point at which the
            # new messages begin.
            if new_message_index > 0:
                attrs['pydantic_ai.new_message_index'] = new_message_index

            # If the instructions for this agent run were not always the same, store an attribute that indicates that.
            # This can signal to an observability UI that different steps in the agent run had different instructions.
            # Note: We purposely only look at "new" messages because they are the only ones produced by this agent run.
            if any(
                (
                    isinstance(m, _messages.ModelRequest)
                    and m.instructions is not None
                    and m.instructions != last_instructions
                )
                for m in message_history[new_message_index:]
            ):
                attrs['pydantic_ai.variable_instructions'] = True

        if metadata is not None:
            attrs['metadata'] = json.dumps(InstrumentedModel.serialize_any(metadata))

        usage_attrs = (
            {
                k.replace('gen_ai.usage.', 'gen_ai.aggregated_usage.', 1): v
                for k, v in usage.opentelemetry_attributes().items()
            }
            if settings.use_aggregated_usage_attribute_names
            else usage.opentelemetry_attributes()
        )

        return {
            **usage_attrs,
            **attrs,
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        **{k: {'type': 'array'} if isinstance(v, str) else {} for k, v in attrs.items()},
                        'final_result': {'type': 'object'},
                    },
                }
            ),
        }

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

        # Warn for unsupported fields with non-default values
        for field_name in _UNSUPPORTED_SPEC_FIELDS:
            field_info = type(validated_spec).model_fields[field_name]
            if getattr(validated_spec, field_name) != field_info.default:
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
        builtin_tools: Sequence[AgentBuiltinTool[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        instructions: AgentInstructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
        metadata: AgentMetadata[AgentDepsT] | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | _utils.Unset = _utils.UNSET,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent configuration.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            builtin_tools: The builtin tools to use instead of the agent's configured builtin tools.
                Per-run `builtin_tools` are still added to these.
            instructions: The instructions to use instead of the instructions registered with the agent.
                Note: this also replaces capability-contributed instructions (e.g. from
                [`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions]).
            metadata: The metadata to use instead of the metadata passed to the agent constructor. When set, any
                per-run `metadata` argument is ignored.
            model_settings: The model settings to use instead of the model settings passed to the agent constructor.
                When set, any per-run `model_settings` argument is ignored.
            spec: Optional agent spec providing defaults for override. Explicit params take precedence
                over spec values. When the spec includes `capabilities`, they replace (not merge with)
                the agent's existing capabilities. To add capabilities without replacing, pass `spec`
                to `run()` or `iter()` instead.
        """
        resolved = self._resolve_spec(spec)

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

        if _utils.is_set(name):
            name_token = self._override_name.set(_utils.Some(name))
        else:
            name_token = None

        if _utils.is_set(deps):
            deps_token = self._override_deps.set(_utils.Some(deps))
        else:
            deps_token = None

        if _utils.is_set(model):
            model_token = self._override_model.set(_utils.Some(models.infer_model(model)))
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

        if _utils.is_set(builtin_tools):
            builtin_tools_token = self._override_builtin_tools.set(_utils.Some(builtin_tools))
        else:
            builtin_tools_token = None

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

        # Set capability from spec, replacing the agent's existing root capability.
        # Auto-inject infrastructure capabilities since the override replaces
        # (not merges with) the agent's root capability.
        if resolved is not None and resolved.capability is not None:
            override_caps = list(resolved.capability.capabilities)
            _inject_auto_capabilities(override_caps)
            override_capability = CombinedCapability(override_caps)
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
            if builtin_tools_token is not None:
                self._override_builtin_tools.reset(builtin_tools_token)
            if instructions_token is not None:
                self._override_instructions.reset(instructions_token)
            if metadata_token is not None:
                self._override_metadata.reset(metadata_token)
            if model_settings_token is not None:
                self._override_model_settings.reset(model_settings_token)
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
    def instructions(
        self, /
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def instructions(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
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
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
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
        run_context = RunContext[AgentDepsT](
            deps=deps,
            agent=self,
            model=self._get_model(model),
            usage=usage or _usage.RunUsage(),
            prompt=prompt,
            messages=list(message_history or []),
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
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def system_prompt(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
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
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
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
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
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
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
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

    def _get_model(self, model: models.Model | models.KnownModelName | str | None) -> models.Model:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The model used
        """
        model_: models.Model
        if some_model := self._override_model.get():
            # we don't want `override()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must either be set on the agent or included when calling it. '
                    '(Even when `override(model=...)` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
        elif model is not None:
            model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
        else:
            raise exceptions.UserError('`model` must either be set on the agent or included when calling it.')

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_model(model_, instrument)

    def _get_deps(self: Agent[T, OutputDataT], deps: T) -> T:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps.get():
            return some_deps.value
        else:
            return deps

    def _get_instructions(
        self,
        additional_instructions: AgentInstructions[AgentDepsT] = None,
        cap_instructions: list[str | _system_prompt.SystemPromptFunc[AgentDepsT]] | None = None,
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
    ) -> AbstractToolset[AgentDepsT]:
        """Get the complete toolset.

        Args:
            output_toolset: The output toolset to use instead of the one built at agent construction time.
            additional_toolsets: Additional toolsets to add, unless toolsets have been overridden.
            cap_toolsets: Per-run capability toolsets to use instead of the init-time capability toolsets.
            run_capability: The per-run capability instance, used to apply wrapper toolsets.
        """
        toolsets = list(self._build_toolset_list(cap_toolsets=cap_toolsets))
        # Don't add additional toolsets if the toolsets have been overridden
        if additional_toolsets and self._override_toolsets.get() is None:
            toolsets = [*toolsets, *additional_toolsets]

        toolset: AbstractToolset[AgentDepsT] = CombinedToolset(toolsets)

        if run_capability is not None:
            # Dispatch the `prepare_tools` capability hook through a `PreparedToolset` wrapped
            # **inside** any other capability `get_wrapper_toolset` results (e.g. `ToolSearch`,
            # `CodeMode`), matching the original ordering of the agent-level `prepare_tools=`
            # kwarg in main: filter/modify defs first, let other toolset transformations layer
            # on top. The hook sees **function** tools only — output tools route through
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
                # retry budget (matches `_build_output_run_context`'s contract — see #4745).
                # `output_toolset.max_retries` is set to `max_result_retries` at agent construction.
                output_cap = run_capability
                output_max_retries = self._max_result_retries

                async def _dispatch_prepare_output_tools(
                    ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
                ) -> list[ToolDefinition]:
                    output_ctx = replace(ctx, max_retries=output_max_retries)
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

    async def __aenter__(self) -> Self:
        """Enter the agent context.

        This will start all [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] registered as `toolsets` so they are ready to be used,
        and enter the model so the provider's HTTP client will be closed cleanly on exit.

        This is a no-op if the agent has already been entered.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    toolset = self._get_toolset()
                    await exit_stack.enter_async_context(toolset)

                    if self.model is not None:
                        model = self._get_model(None)
                        await exit_stack.enter_async_context(model)

                    self._exit_stack = exit_stack.pop_all()
            self._entered_count += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0 and self._exit_stack is not None:
                await self._exit_stack.aclose()
                self._exit_stack = None

    def set_mcp_sampling_model(self, model: models.Model | models.KnownModelName | str | None = None) -> None:
        """Set the sampling model on all MCP servers registered with the agent.

        If no sampling model is provided, the agent's model will be used.
        """
        try:
            sampling_model = models.infer_model(model) if model else self._get_model(None)
        except exceptions.UserError as e:
            raise exceptions.UserError('No sampling model provided and no model set on the agent.') from e

        from ..mcp import MCPServer

        def _set_sampling_model(toolset: AbstractToolset[AgentDepsT]) -> None:
            if isinstance(toolset, MCPServer):
                toolset.sampling_model = sampling_model

        self._get_toolset().apply(_set_sampling_model)

    def to_web(
        self,
        *,
        models: ModelsParam = None,
        builtin_tools: list[AbstractBuiltinTool] | None = None,
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

        Args:
            models: Additional models to make available in the UI. Can be:
                - A sequence of model names/instances (e.g., `['openai:gpt-5', 'anthropic:claude-sonnet-4-6']`)
                - A dict mapping display labels to model names/instances
                  (e.g., `{'GPT 5': 'openai:gpt-5', 'Claude': 'anthropic:claude-sonnet-4-6'}`)
                The agent's model is always included. Builtin tool support is automatically
                determined from each model's profile.
            builtin_tools: Additional builtin tools to make available in the UI.
                The agent's configured builtin tools are always included. Tool labels
                in the UI are derived from the tool's `label` property.
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
            from pydantic_ai.builtin_tools import WebSearchTool

            agent = Agent('openai:gpt-5', builtin_tools=[WebSearchTool()])

            # Simple usage - uses agent's model and builtin tools
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
            builtin_tools=builtin_tools,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
            html_source=html_source,
        )

    @asynccontextmanager
    @deprecated(
        '`run_mcp_servers` is deprecated, use `async with agent:` instead. If you need to set a sampling model on all MCP servers, use `agent.set_mcp_sampling_model()`.'
    )
    async def run_mcp_servers(
        self, model: models.Model | models.KnownModelName | str | None = None
    ) -> AsyncIterator[None]:
        """Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.

        Deprecated: use [`async with agent`][pydantic_ai.agent.Agent.__aenter__] instead.
        If you need to set a sampling model on all MCP servers, use [`agent.set_mcp_sampling_model()`][pydantic_ai.agent.Agent.set_mcp_sampling_model].

        Returns: a context manager to start and shutdown the servers.
        """
        try:
            self.set_mcp_sampling_model(model)
        except exceptions.UserError:
            if model is not None:
                raise

        async with self:
            yield


_UNSUPPORTED_SPEC_FIELDS: tuple[str, ...] = (
    'description',
    'end_strategy',
    'retries',
    'output_retries',
    'tool_timeout',
    'instrument',
    'output_schema',
    'deps_schema',
)
"""AgentSpec fields that are not supported at run/override time."""

_AUTO_INJECT_CAPABILITY_TYPES: tuple[type[AbstractCapability[Any]], ...] = (ToolSearchCap,)
"""Infrastructure capabilities auto-injected when not already present."""


def _inject_auto_capabilities(capabilities: list[AbstractCapability[Any]]) -> None:
    """Ensure all auto-injected infrastructure capabilities are present.

    Each capability's own ``CapabilityOrdering`` (e.g. ``position='outermost'``)
    determines its final placement, so insertion order here doesn't matter.
    """
    for cap_type in _AUTO_INJECT_CAPABILITY_TYPES:
        if not has_capability_type(capabilities, cap_type):
            capabilities.append(cap_type())


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
