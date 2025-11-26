from __future__ import annotations as _annotations

import dataclasses
import inspect
import json
import warnings
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, overload

from opentelemetry.trace import NoOpTracer, use_span
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Self, TypeVar, deprecated

from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION, InstrumentationNames

from .. import (
    _agent_graph,
    _output,
    _system_prompt,
    _utils,
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
    capture_run_messages,
)
from .._output import OutputToolset
from .._tool_manager import ToolManager
from ..builtin_tools import AbstractBuiltinTool
from ..models.instrumented import InstrumentationSettings, InstrumentedModel, instrument_model
from ..output import OutputDataT, OutputSpec
from ..run import AgentRun, AgentRunResult
from ..settings import ModelSettings, merge_model_settings
from ..tools import (
    AgentDepsT,
    DeferredToolResults,
    DocstringFormat,
    GenerateToolJsonSchema,
    RunContext,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
    ToolsPrepareFunc,
)
from ..toolsets import AbstractToolset
from ..toolsets._dynamic import (
    DynamicToolset,
    ToolsetFunc,
)
from ..toolsets.combined import CombinedToolset
from ..toolsets.function import FunctionToolset
from ..toolsets.prepared import PreparedToolset
from .abstract import AbstractAgent, EventStreamHandler, Instructions, RunOutputDataT
from .wrapper import WrapperAgent

if TYPE_CHECKING:
    from ..mcp import MCPServer

__all__ = (
    'Agent',
    'AgentRun',
    'AgentRunResult',
    'capture_run_messages',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'InstrumentationSettings',
    'WrapperAgent',
    'AbstractAgent',
    'EventStreamHandler',
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)


@dataclasses.dataclass(init=False)
class Agent(AbstractAgent[AgentDepsT, OutputDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the output type they return, [`OutputDataT`][pydantic_ai.output.OutputDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.
    ```
    """

    _model: models.Model | models.KnownModelName | str | None

    _name: str | None
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    _output_type: OutputSpec[OutputDataT]

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

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
    _prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)
    _max_tool_retries: int = dataclasses.field(repr=False)

    _event_stream_handler: EventStreamHandler[AgentDepsT] | None = dataclasses.field(repr=False)

    _enter_lock: Lock = dataclasses.field(repr=False)
    _entered_count: int = dataclasses.field(repr=False)
    _exit_stack: AsyncExitStack | None = dataclasses.field(repr=False)

    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> None: ...

    @overload
    @deprecated('`mcp_servers` is deprecated, use `toolsets` instead.')
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        mcp_servers: Sequence[MCPServer] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: Instructions[AgentDepsT] = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        builtin_tools: Sequence[AbstractBuiltinTool] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provided,
                you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
            output_type: The type of the output data, used to validate the data returned by the model,
                defaults to `str`.
            instructions: Instructions to use for this agent, you can also register instructions via a function with
                [`instructions`][pydantic_ai.Agent.instructions] or pass additional, temporary, instructions when executing a run.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow for tool calls and output validation, before raising an error.
                For model request retries, see the [HTTP Request Retries](../retries.md) documentation.
            output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
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
                [override the model][pydantic_ai.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            instrument: Set to True to automatically instrument with OpenTelemetry,
                which will use Logfire if it's configured.
                Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
                If this isn't set, then the last value set by
                [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
                will be used, which defaults to False.
                See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
            history_processors: Optional list of callables to process the message history before sending it to the model.
                Each processor takes a list of messages and returns a modified list of messages.
                Processors can be sync or async and are applied in sequence.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools.
        """
        if model is None or defer_model_check:
            self._model = model
        else:
            self._model = models.infer_model(model)

        self._name = name
        self.end_strategy = end_strategy
        self.model_settings = model_settings

        self._output_type = output_type
        self.instrument = instrument
        self._deps_type = deps_type

        if mcp_servers := _deprecated_kwargs.pop('mcp_servers', None):
            if toolsets is not None:  # pragma: no cover
                raise TypeError('`mcp_servers` and `toolsets` cannot be set at the same time.')
            warnings.warn('`mcp_servers` is deprecated, use `toolsets` instead', DeprecationWarning)
            toolsets = mcp_servers

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        self._output_schema = _output.OutputSchema[OutputDataT].build(output_type)
        self._output_validators = []

        self._instructions = self._normalize_instructions(instructions)

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        self._max_result_retries = output_retries if output_retries is not None else retries
        self._max_tool_retries = retries

        self._builtin_tools = builtin_tools

        self._prepare_tools = prepare_tools
        self._prepare_output_tools = prepare_output_tools

        self._output_toolset = self._output_schema.toolset
        if self._output_toolset:
            self._output_toolset.max_retries = self._max_result_retries

        self._function_toolset = _AgentFunctionToolset(
            tools, max_retries=self._max_tool_retries, output_schema=self._output_schema
        )
        self._dynamic_toolsets = [
            DynamicToolset[AgentDepsT](toolset_func=toolset)
            for toolset in toolsets or []
            if not isinstance(toolset, AbstractToolset)
        ]
        self._user_toolsets = [toolset for toolset in toolsets or [] if isinstance(toolset, AbstractToolset)]

        self.history_processors = history_processors or []

        self._event_stream_handler = event_stream_handler

        self._override_name: ContextVar[_utils.Option[str]] = ContextVar('_override_name', default=None)
        self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
        self._override_model: ContextVar[_utils.Option[models.Model]] = ContextVar('_override_model', default=None)
        self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
            '_override_toolsets', default=None
        )
        self._override_tools: ContextVar[
            _utils.Option[Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]]]
        ] = ContextVar('_override_tools', default=None)
        self._override_instructions: ContextVar[
            _utils.Option[list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]]
        ] = ContextVar('_override_instructions', default=None)

        self._enter_lock = Lock()
        self._entered_count = 0
        self._exit_stack = None

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
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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

        agent = Agent('openai:gpt-4o')

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
                        run_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        run_id='...',
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
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

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
                output_toolset.max_retries = self._max_result_retries
                output_toolset.output_validators = output_validators
        toolset = self._get_toolset(output_toolset=output_toolset, additional_toolsets=toolsets)
        tool_manager = ToolManager[AgentDepsT](toolset)

        # Build the graph
        graph = _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)

        # Build the initial state
        usage = usage or _usage.RunUsage()
        state = _agent_graph.GraphAgentState(
            message_history=list(message_history) if message_history else [],
            usage=usage,
            retries=0,
            run_step=0,
        )

        # Merge model settings in order of precedence: run > agent > model
        merged_settings = merge_model_settings(model_used.settings, self.model_settings)
        model_settings = merge_model_settings(merged_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()

        instructions_literal, instructions_functions = self._get_instructions(additional_instructions=instructions)

        async def get_instructions(run_context: RunContext[AgentDepsT]) -> str | None:
            parts = [
                instructions_literal,
                *[await func.run(run_context) for func in instructions_functions],
            ]

            parts = [p for p in parts if p]
            if not parts:
                return None
            return '\n\n'.join(parts).strip()

        if isinstance(model_used, InstrumentedModel):
            instrumentation_settings = model_used.instrumentation_settings
            tracer = model_used.instrumentation_settings.tracer
        else:
            instrumentation_settings = None
            tracer = NoOpTracer()

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT](
            user_deps=deps,
            prompt=user_prompt,
            new_message_index=len(message_history) if message_history else 0,
            model=model_used,
            model_settings=model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            output_schema=output_schema,
            output_validators=output_validators,
            history_processors=self.history_processors,
            builtin_tools=[*self._builtin_tools, *(builtin_tools or [])],
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

        run_span = tracer.start_span(
            instrumentation_names.get_agent_run_span_name(agent_name),
            attributes={
                'model_name': model_used.model_name if model_used else 'no-model',
                'agent_name': agent_name,
                'gen_ai.agent.name': agent_name,
                'logfire.msg': f'{agent_name} run',
            },
        )

        try:
            async with graph.iter(
                inputs=user_prompt_node,
                state=state,
                deps=graph_deps,
                span=use_span(run_span) if run_span.is_recording() else None,
                infer_name=False,
            ) as graph_run:
                async with toolset:
                    agent_run = AgentRun(graph_run)
                    yield agent_run
                    if (final_result := agent_run.result) is not None and run_span.is_recording():
                        if instrumentation_settings and instrumentation_settings.include_content:
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
                            instrumentation_settings, usage, state.message_history, graph_deps.new_message_index
                        )
                    )
            finally:
                run_span.end()

    def _run_span_end_attributes(
        self,
        settings: InstrumentationSettings,
        usage: _usage.RunUsage,
        message_history: list[_messages.ModelMessage],
        new_message_index: int,
    ):
        if settings.version == 1:
            attrs = {
                'all_messages_events': json.dumps(
                    [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(message_history)]
                )
            }
        else:
            # Store the last instructions here for convenience
            last_instructions = InstrumentedModel._get_instructions(message_history)  # pyright: ignore[reportPrivateUsage]
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

        return {
            **usage.opentelemetry_attributes(),
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

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
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

        if _utils.is_set(instructions):
            normalized_instructions = self._normalize_instructions(instructions)
            instructions_token = self._override_instructions.set(_utils.Some(normalized_instructions))
        else:
            instructions_token = None

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
            if instructions_token is not None:
                self._override_instructions.reset(instructions_token)

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def instructions(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def instructions(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

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

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

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
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
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
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
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
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
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
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
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
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
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
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
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
    ) -> Callable[[ToolsetFunc[AgentDepsT]], ToolsetFunc[AgentDepsT]]: ...

    def toolset(
        self,
        func: ToolsetFunc[AgentDepsT] | None = None,
        /,
        *,
        per_run_step: bool = True,
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
        """

        def toolset_decorator(func_: ToolsetFunc[AgentDepsT]) -> ToolsetFunc[AgentDepsT]:
            self._dynamic_toolsets.append(DynamicToolset(func_, per_run_step=per_run_step))
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

    def _normalize_instructions(
        self,
        instructions: Instructions[AgentDepsT],
    ) -> list[str | _system_prompt.SystemPromptFunc[AgentDepsT]]:
        if instructions is None:
            return []
        if isinstance(instructions, str) or callable(instructions):
            return [instructions]
        return list(instructions)

    def _get_instructions(
        self,
        additional_instructions: Instructions[AgentDepsT] = None,
    ) -> tuple[str | None, list[_system_prompt.SystemPromptRunner[AgentDepsT]]]:
        override_instructions = self._override_instructions.get()
        if override_instructions:
            instructions = override_instructions.value
        else:
            instructions = self._instructions.copy()
            if additional_instructions is not None:
                instructions.extend(self._normalize_instructions(additional_instructions))

        literal_parts: list[str] = []
        functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []

        for instruction in instructions:
            if isinstance(instruction, str):
                literal_parts.append(instruction)
            else:
                functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](instruction))

        literal = '\n'.join(literal_parts).strip() or None
        return literal, functions

    def _get_toolset(
        self,
        output_toolset: AbstractToolset[AgentDepsT] | None | _utils.Unset = _utils.UNSET,
        additional_toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractToolset[AgentDepsT]:
        """Get the complete toolset.

        Args:
            output_toolset: The output toolset to use instead of the one built at agent construction time.
            additional_toolsets: Additional toolsets to add, unless toolsets have been overridden.
        """
        toolsets = self.toolsets
        # Don't add additional toolsets if the toolsets have been overridden
        if additional_toolsets and self._override_toolsets.get() is None:
            toolsets = [*toolsets, *additional_toolsets]

        toolset = CombinedToolset(toolsets)

        # Copy the dynamic toolsets to ensure each run has its own instances
        def copy_dynamic_toolsets(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if isinstance(toolset, DynamicToolset):
                return dataclasses.replace(toolset)
            else:
                return toolset

        toolset = toolset.visit_and_replace(copy_dynamic_toolsets)

        if self._prepare_tools:
            toolset = PreparedToolset(toolset, self._prepare_tools)

        output_toolset = output_toolset if _utils.is_set(output_toolset) else self._output_toolset
        if output_toolset is not None:
            if self._prepare_output_tools:
                output_toolset = PreparedToolset(output_toolset, self._prepare_output_tools)
            toolset = CombinedToolset([output_toolset, toolset])

        return toolset

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """All toolsets registered on the agent, including a function toolset holding tools that were registered on the agent directly.

        Output tools are not included.
        """
        toolsets: list[AbstractToolset[AgentDepsT]] = []

        if some_tools := self._override_tools.get():
            function_toolset = _AgentFunctionToolset(
                some_tools.value, max_retries=self._max_tool_retries, output_schema=self._output_schema
            )
        else:
            function_toolset = self._function_toolset
        toolsets.append(function_toolset)

        if some_user_toolsets := self._override_toolsets.get():
            user_toolsets = some_user_toolsets.value
        else:
            user_toolsets = [*self._user_toolsets, *self._dynamic_toolsets]
        toolsets.extend(user_toolsets)

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

        This will start all [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] registered as `toolsets` so they are ready to be used.

        This is a no-op if the agent has already been entered.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    toolset = self._get_toolset()
                    await exit_stack.enter_async_context(toolset)

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


@dataclasses.dataclass(init=False)
class _AgentFunctionToolset(FunctionToolset[AgentDepsT]):
    output_schema: _output.OutputSchema[Any]

    def __init__(
        self,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
        *,
        max_retries: int = 1,
        id: str | None = None,
        output_schema: _output.OutputSchema[Any],
    ):
        self.output_schema = output_schema
        super().__init__(tools, max_retries=max_retries, id=id)

    @property
    def id(self) -> str:
        return '<agent>'

    @property
    def label(self) -> str:
        return 'the agent'

    def add_tool(self, tool: Tool[AgentDepsT]) -> None:
        if tool.requires_approval and not self.output_schema.allows_deferred_tools:
            raise exceptions.UserError(
                'To use tools that require approval, add `DeferredToolRequests` to the list of output types for this agent.'
            )
        super().add_tool(tool)
