from __future__ import annotations as _annotations

import dataclasses
import inspect
import json
import warnings
from asyncio import Lock
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, cast, final, overload

from opentelemetry.trace import NoOpTracer, use_span
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Literal, Never, Self, TypeIs, TypeVar, deprecated

from pydantic_graph import End, Graph, GraphRun, GraphRunContext
from pydantic_graph._utils import get_event_loop

from . import (
    _agent_graph,
    _output,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from ._agent_graph import HistoryProcessor
from ._output import OutputToolset
from ._tool_manager import ToolManager
from .models.instrumented import InstrumentationSettings, InstrumentedModel, instrument_model
from .output import OutputDataT, OutputSpec
from .profiles import ModelProfile
from .result import AgentStream, FinalResult, StreamedRunResult
from .settings import ModelSettings, merge_model_settings
from .tools import (
    AgentDepsT,
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
from .toolsets import AbstractToolset
from .toolsets.combined import CombinedToolset
from .toolsets.function import FunctionToolset
from .toolsets.prepared import PreparedToolset
from .usage import Usage, UsageLimits

# Re-exporting like this improves auto-import behavior in PyCharm
capture_run_messages = _agent_graph.capture_run_messages
EndStrategy = _agent_graph.EndStrategy
CallToolsNode = _agent_graph.CallToolsNode
ModelRequestNode = _agent_graph.ModelRequestNode
UserPromptNode = _agent_graph.UserPromptNode

if TYPE_CHECKING:
    from fasta2a.applications import FastA2A
    from fasta2a.broker import Broker
    from fasta2a.schema import AgentProvider, Skill
    from fasta2a.storage import Storage
    from starlette.middleware import Middleware
    from starlette.routing import BaseRoute, Route
    from starlette.types import ExceptionHandler, Lifespan

    from pydantic_ai.mcp import MCPServer

    from .ag_ui import AGUIApp

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
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
RunOutputDataT = TypeVar('RunOutputDataT')
"""Type variable for the result data of a run where `output_type` was customized on the run call."""


@final
@dataclasses.dataclass(init=False)
class Agent(Generic[AgentDepsT, OutputDataT]):
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
    #> Paris
    ```
    """

    model: models.Model | models.KnownModelName | str | None
    """The default model configured for this agent.

    We allow `str` here since the actual list of allowed models changes frequently.
    """

    name: str | None
    """The name of the agent, used for logging.

    If `None`, we try to infer the agent name from the call frame when the agent is first run.
    """
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    output_type: OutputSpec[OutputDataT]
    """
    The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`.
    """

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _deprecated_result_tool_name: str | None = dataclasses.field(repr=False)
    _deprecated_result_tool_description: str | None = dataclasses.field(repr=False)
    _output_schema: _output.BaseOutputSchema[OutputDataT] = dataclasses.field(repr=False)
    _output_validators: list[_output.OutputValidator[AgentDepsT, OutputDataT]] = dataclasses.field(repr=False)
    _instructions: str | None = dataclasses.field(repr=False)
    _instructions_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_toolset: FunctionToolset[AgentDepsT] = dataclasses.field(repr=False)
    _output_toolset: OutputToolset[AgentDepsT] | None = dataclasses.field(repr=False)
    _user_toolsets: Sequence[AbstractToolset[AgentDepsT]] = dataclasses.field(repr=False)
    _prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)

    _enter_lock: Lock = dataclasses.field(repr=False)
    _entered_count: int = dataclasses.field(repr=False)
    _exit_stack: AsyncExitStack | None = dataclasses.field(repr=False)

    @overload
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        output_type: OutputSpec[OutputDataT] = str,
        instructions: str
        | _system_prompt.SystemPromptFunc[AgentDepsT]
        | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
        | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    ) -> None: ...

    @overload
    @deprecated(
        '`result_type`, `result_tool_name` & `result_tool_description` are deprecated, use `output_type` instead. `result_retries` is deprecated, use `output_retries` instead.'
    )
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        result_type: type[OutputDataT] = str,
        instructions: str
        | _system_prompt.SystemPromptFunc[AgentDepsT]
        | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
        | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        result_tool_name: str = _output.DEFAULT_OUTPUT_TOOL_NAME,
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    ) -> None: ...

    @overload
    @deprecated('`mcp_servers` is deprecated, use `toolsets` instead.')
    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        result_type: type[OutputDataT] = str,
        instructions: str
        | _system_prompt.SystemPromptFunc[AgentDepsT]
        | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
        | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        result_tool_name: str = _output.DEFAULT_OUTPUT_TOOL_NAME,
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        mcp_servers: Sequence[MCPServer] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
    ) -> None: ...

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        # TODO change this back to `output_type: _output.OutputType[OutputDataT] = str,` when we remove the overloads
        output_type: Any = str,
        instructions: str
        | _system_prompt.SystemPromptFunc[AgentDepsT]
        | Sequence[str | _system_prompt.SystemPromptFunc[AgentDepsT]]
        | None = None,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        prepare_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        prepare_output_tools: ToolsPrepareFunc[AgentDepsT] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
        history_processors: Sequence[HistoryProcessor[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Any,
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provide,
                you must provide the model when calling it. We allow `str` here since the actual list of allowed models changes frequently.
            output_type: The type of the output data, used to validate the data returned by the model,
                defaults to `str`.
            instructions: Instructions to use for this agent, you can also register instructions via a function with
                [`instructions`][pydantic_ai.Agent.instructions].
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow before raising an error.
            output_retries: The maximum number of retries to allow for output validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
            prepare_tools: Custom function to prepare the tool definition of all tools for each step, except output tools.
                This is useful if you want to customize the definition of multiple tools or you want to register
                a subset of tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            prepare_output_tools: Custom function to prepare the tool definition of all output tools for each step.
                This is useful if you want to customize the definition of multiple output tools or you want to register
                a subset of output tools for a given step. See [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc]
            toolsets: Toolsets to register with the agent, including MCP servers.
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
        """
        if model is None or defer_model_check:
            self.model = model
        else:
            self.model = models.infer_model(model)

        self.end_strategy = end_strategy
        self.name = name
        self.model_settings = model_settings

        if 'result_type' in _deprecated_kwargs:
            if output_type is not str:  # pragma: no cover
                raise TypeError('`result_type` and `output_type` cannot be set at the same time.')
            warnings.warn('`result_type` is deprecated, use `output_type` instead', DeprecationWarning, stacklevel=2)
            output_type = _deprecated_kwargs.pop('result_type')

        self.output_type = output_type

        self.instrument = instrument

        self._deps_type = deps_type

        self._deprecated_result_tool_name = _deprecated_kwargs.pop('result_tool_name', None)
        if self._deprecated_result_tool_name is not None:
            warnings.warn(
                '`result_tool_name` is deprecated, use `output_type` with `ToolOutput` instead',
                DeprecationWarning,
                stacklevel=2,
            )

        self._deprecated_result_tool_description = _deprecated_kwargs.pop('result_tool_description', None)
        if self._deprecated_result_tool_description is not None:
            warnings.warn(
                '`result_tool_description` is deprecated, use `output_type` with `ToolOutput` instead',
                DeprecationWarning,
                stacklevel=2,
            )
        result_retries = _deprecated_kwargs.pop('result_retries', None)
        if result_retries is not None:
            if output_retries is not None:  # pragma: no cover
                raise TypeError('`output_retries` and `result_retries` cannot be set at the same time.')
            warnings.warn(
                '`result_retries` is deprecated, use `max_result_retries` instead', DeprecationWarning, stacklevel=2
            )
            output_retries = result_retries

        if mcp_servers := _deprecated_kwargs.pop('mcp_servers', None):
            if toolsets is not None:  # pragma: no cover
                raise TypeError('`mcp_servers` and `toolsets` cannot be set at the same time.')
            warnings.warn('`mcp_servers` is deprecated, use `toolsets` instead', DeprecationWarning)
            toolsets = mcp_servers

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        default_output_mode = (
            self.model.profile.default_structured_output_mode if isinstance(self.model, models.Model) else None
        )

        self._output_schema = _output.OutputSchema[OutputDataT].build(
            output_type,
            default_mode=default_output_mode,
            name=self._deprecated_result_tool_name,
            description=self._deprecated_result_tool_description,
        )
        self._output_validators = []

        self._instructions = ''
        self._instructions_functions = []
        if isinstance(instructions, (str, Callable)):
            instructions = [instructions]
        for instruction in instructions or []:
            if isinstance(instruction, str):
                self._instructions += instruction + '\n'
            else:
                self._instructions_functions.append(_system_prompt.SystemPromptRunner(instruction))
        self._instructions = self._instructions.strip() or None

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        self._max_result_retries = output_retries if output_retries is not None else retries
        self._prepare_tools = prepare_tools
        self._prepare_output_tools = prepare_output_tools

        self._output_toolset = self._output_schema.toolset
        if self._output_toolset:
            self._output_toolset.max_retries = self._max_result_retries

        self._function_toolset = FunctionToolset(tools, max_retries=retries)
        self._user_toolsets = toolsets or ()

        self.history_processors = history_processors or []

        self._override_deps: ContextVar[_utils.Option[AgentDepsT]] = ContextVar('_override_deps', default=None)
        self._override_model: ContextVar[_utils.Option[models.Model]] = ContextVar('_override_model', default=None)
        self._override_toolsets: ContextVar[_utils.Option[Sequence[AbstractToolset[AgentDepsT]]]] = ContextVar(
            '_override_toolsets', default=None
        )

        self._enter_lock = _utils.get_async_lock()
        self._entered_count = 0
        self._exit_stack = None

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all agents where `instrument` is not set."""
        Agent._instrument_default = instrument

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    @overload
    @deprecated('`result_type` is deprecated, use `output_type` instead.')
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> Paris
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if 'result_type' in _deprecated_kwargs:  # pragma: no cover
            if output_type is not str:
                raise TypeError('`result_type` and `output_type` cannot be set at the same time.')
            warnings.warn('`result_type` is deprecated, use `output_type` instead.', DeprecationWarning, stacklevel=2)
            output_type = _deprecated_kwargs.pop('result_type')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        async with self.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            toolsets=toolsets,
        ) as agent_run:
            async for _ in agent_run:
                pass

        assert agent_run.result is not None, 'The graph run did not finish properly'
        return agent_run.result

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @overload
    @deprecated('`result_type` is deprecated, use `output_type` instead.')
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, Any]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
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
                    instructions=None,
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
                        ]
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='Paris')],
                        usage=Usage(
                            requests=1, request_tokens=56, response_tokens=1, total_tokens=57
                        ),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                    )
                ),
                End(data=FinalResult(output='Paris')),
            ]
            '''
            print(agent_run.result.output)
            #> Paris
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        model_used = self._get_model(model)
        del model

        if 'result_type' in _deprecated_kwargs:  # pragma: no cover
            if output_type is not str:
                raise TypeError('`result_type` and `output_type` cannot be set at the same time.')
            warnings.warn('`result_type` is deprecated, use `output_type` instead.', DeprecationWarning, stacklevel=2)
            output_type = _deprecated_kwargs.pop('result_type')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        output_schema = self._prepare_output_schema(output_type, model_used.profile)

        output_type_ = output_type or self.output_type

        # We consider it a user error if a user tries to restrict the result type while having an output validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        output_validators = cast(list[_output.OutputValidator[AgentDepsT, RunOutputDataT]], self._output_validators)

        output_toolset = self._output_toolset
        if output_schema != self._output_schema or output_validators:
            output_toolset = cast(OutputToolset[AgentDepsT], output_schema.toolset)
            if output_toolset:
                output_toolset.max_retries = self._max_result_retries
                output_toolset.output_validators = output_validators

        # Build the graph
        graph: Graph[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[Any]] = (
            _agent_graph.build_agent_graph(self.name, self._deps_type, output_type_)
        )

        # Build the initial state
        usage = usage or _usage.Usage()
        state = _agent_graph.GraphAgentState(
            message_history=message_history[:] if message_history else [],
            usage=usage,
            retries=0,
            run_step=0,
        )

        if isinstance(model_used, InstrumentedModel):
            instrumentation_settings = model_used.instrumentation_settings
            tracer = model_used.instrumentation_settings.tracer
        else:
            instrumentation_settings = None
            tracer = NoOpTracer()

        run_context = RunContext[AgentDepsT](
            deps=deps,
            model=model_used,
            usage=usage,
            prompt=user_prompt,
            messages=state.message_history,
            tracer=tracer,
            trace_include_content=instrumentation_settings is not None and instrumentation_settings.include_content,
            run_step=state.run_step,
        )

        toolset = self._get_toolset(output_toolset=output_toolset, additional_toolsets=toolsets)
        # This will raise errors for any name conflicts
        run_toolset = await ToolManager[AgentDepsT].build(toolset, run_context)

        # Merge model settings in order of precedence: run > agent > model
        merged_settings = merge_model_settings(model_used.settings, self.model_settings)
        model_settings = merge_model_settings(merged_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()
        agent_name = self.name or 'agent'
        run_span = tracer.start_span(
            'agent run',
            attributes={
                'model_name': model_used.model_name if model_used else 'no-model',
                'agent_name': agent_name,
                'logfire.msg': f'{agent_name} run',
            },
        )

        async def get_instructions(run_context: RunContext[AgentDepsT]) -> str | None:
            parts = [
                self._instructions,
                *[await func.run(run_context) for func in self._instructions_functions],
            ]

            model_profile = model_used.profile
            if isinstance(output_schema, _output.PromptedOutputSchema):
                instructions = output_schema.instructions(model_profile.prompted_output_template)
                parts.append(instructions)

            parts = [p for p in parts if p]
            if not parts:
                return None
            return '\n\n'.join(parts).strip()

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, RunOutputDataT](
            user_deps=deps,
            prompt=user_prompt,
            new_message_index=new_message_index,
            model=model_used,
            model_settings=model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            output_schema=output_schema,
            output_validators=output_validators,
            history_processors=self.history_processors,
            tool_manager=run_toolset,
            tracer=tracer,
            get_instructions=get_instructions,
            instrumentation_settings=instrumentation_settings,
        )
        start_node = _agent_graph.UserPromptNode[AgentDepsT](
            user_prompt=user_prompt,
            instructions=self._instructions,
            instructions_functions=self._instructions_functions,
            system_prompts=self._system_prompts,
            system_prompt_functions=self._system_prompt_functions,
            system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
        )

        try:
            async with graph.iter(
                start_node,
                state=state,
                deps=graph_deps,
                span=use_span(run_span) if run_span.is_recording() else None,
                infer_name=False,
            ) as graph_run:
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
                    run_span.set_attributes(self._run_span_end_attributes(state, usage, instrumentation_settings))
            finally:
                run_span.end()

    def _run_span_end_attributes(
        self, state: _agent_graph.GraphAgentState, usage: _usage.Usage, settings: InstrumentationSettings
    ):
        return {
            **usage.opentelemetry_attributes(),
            'all_messages_events': json.dumps(
                [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(state.message_history)]
            ),
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        'all_messages_events': {'type': 'array'},
                        'final_result': {'type': 'object'},
                    },
                }
            ),
        }

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    @overload
    @deprecated('`result_type` is deprecated, use `output_type` instead.')
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> Rome
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        if 'result_type' in _deprecated_kwargs:  # pragma: no cover
            if output_type is not str:
                raise TypeError('`result_type` and `output_type` cannot be set at the same time.')
            warnings.warn('`result_type` is deprecated, use `output_type` instead.', DeprecationWarning, stacklevel=2)
            output_type = _deprecated_kwargs.pop('result_type')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        return get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=False,
                toolsets=toolsets,
            )
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent],
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @overload
    @deprecated('`result_type` is deprecated, use `output_type` instead.')
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(  # noqa C901
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> London
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        # TODO: We need to deprecate this now that we have the `iter` method.
        #   Before that, though, we should add an event for when we reach the final result of the stream.
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        if 'result_type' in _deprecated_kwargs:  # pragma: no cover
            if output_type is not str:
                raise TypeError('`result_type` and `output_type` cannot be set at the same time.')
            warnings.warn('`result_type` is deprecated, use `output_type` instead.', DeprecationWarning, stacklevel=2)
            output_type = _deprecated_kwargs.pop('result_type')

        _utils.validate_empty_kwargs(_deprecated_kwargs)

        yielded = False
        async with self.iter(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=False,
            toolsets=toolsets,
        ) as agent_run:
            first_node = agent_run.next_node  # start with the first node
            assert isinstance(first_node, _agent_graph.UserPromptNode)  # the first node should be a user prompt node
            node = first_node
            while True:
                if self.is_model_request_node(node):
                    graph_ctx = agent_run.ctx
                    async with node.stream(graph_ctx) as stream:

                        async def stream_to_final(s: AgentStream) -> FinalResult[AgentStream] | None:
                            async for event in stream:
                                if isinstance(event, _messages.FinalResultEvent):
                                    return FinalResult(s, event.tool_name, event.tool_call_id)
                            return None

                        final_result = await stream_to_final(stream)
                        if final_result is not None:
                            if yielded:
                                raise exceptions.AgentRunError('Agent run produced final results')  # pragma: no cover
                            yielded = True

                            messages = graph_ctx.state.message_history.copy()

                            async def on_complete() -> None:
                                """Called when the stream has completed.

                                The model response will have been added to messages by now
                                by `StreamedRunResult._marked_completed`.
                                """
                                last_message = messages[-1]
                                assert isinstance(last_message, _messages.ModelResponse)
                                tool_calls = [
                                    part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
                                ]

                                parts: list[_messages.ModelRequestPart] = []
                                async for _event in _agent_graph.process_function_tools(
                                    graph_ctx.deps.tool_manager,
                                    tool_calls,
                                    final_result,
                                    graph_ctx,
                                    parts,
                                ):
                                    pass
                                if parts:
                                    messages.append(_messages.ModelRequest(parts))

                            yield StreamedRunResult(
                                messages,
                                graph_ctx.deps.new_message_index,
                                stream,
                                on_complete,
                            )
                            break
                next_node = await agent_run.next(node)
                if not isinstance(next_node, _agent_graph.AgentNode):
                    raise exceptions.AgentRunError(  # pragma: no cover
                        'Should have produced a StreamedRunResult before getting here'
                    )
                node = cast(_agent_graph.AgentNode[Any, Any], next_node)

        if not yielded:
            raise exceptions.AgentRunError('Agent run finished without producing a final result')  # pragma: no cover

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, or toolsets.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
        """
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

        try:
            yield
        finally:
            if deps_token is not None:
                self._override_deps.reset(deps_token)
            if model_token is not None:
                self._override_model.reset(model_token)
            if toolsets_token is not None:
                self._override_toolsets.reset(toolsets_token)

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
                self._instructions_functions.append(_system_prompt.SystemPromptRunner(func_))
                return func_

            return decorator
        else:
            self._instructions_functions.append(_system_prompt.SystemPromptRunner(func))
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

    @deprecated('`result_validator` is deprecated, use `output_validator` instead.')
    def result_validator(self, func: Any, /) -> Any:
        warnings.warn(
            '`result_validator` is deprecated, use `output_validator` instead.', DeprecationWarning, stacklevel=2
        )
        return self.output_validator(func)  # type: ignore

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
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
        """

        def tool_decorator(
            func_: ToolFuncContext[AgentDepsT, ToolParams],
        ) -> ToolFuncContext[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                True,
                name,
                retries,
                prepare,
                docstring_format,
                require_parameter_descriptions,
                schema_generator,
                strict,
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
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
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
        """

        def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
            # noinspection PyTypeChecker
            self._function_toolset.add_function(
                func_,
                False,
                name,
                retries,
                prepare,
                docstring_format,
                require_parameter_descriptions,
                schema_generator,
                strict,
            )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

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

    def _get_toolset(
        self,
        output_toolset: AbstractToolset[AgentDepsT] | None | _utils.Unset = _utils.UNSET,
        additional_toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    ) -> AbstractToolset[AgentDepsT]:
        """Get the complete toolset.

        Args:
            output_toolset: The output toolset to use instead of the one built at agent construction time.
            additional_toolsets: Additional toolsets to add.
        """
        if some_user_toolsets := self._override_toolsets.get():
            user_toolsets = some_user_toolsets.value
        elif additional_toolsets is not None:
            user_toolsets = [*self._user_toolsets, *additional_toolsets]
        else:
            user_toolsets = self._user_toolsets

        all_toolsets = [self._function_toolset, *user_toolsets]

        if self._prepare_tools:
            all_toolsets = [PreparedToolset(CombinedToolset(all_toolsets), self._prepare_tools)]

        output_toolset = output_toolset if _utils.is_set(output_toolset) else self._output_toolset
        if output_toolset is not None:
            if self._prepare_output_tools:
                output_toolset = PreparedToolset(output_toolset, self._prepare_output_tools)
            all_toolsets = [output_toolset, *all_toolsets]

        return CombinedToolset(all_toolsets)

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:  # pragma: no branch
                    # if we couldn't find the agent in locals and globals are a different dict, try globals
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

    @property
    @deprecated(
        'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
    )
    def last_run_messages(self) -> list[_messages.ModelMessage]:
        raise AttributeError('The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.')

    def _prepare_output_schema(
        self, output_type: OutputSpec[RunOutputDataT] | None, model_profile: ModelProfile
    ) -> _output.OutputSchema[RunOutputDataT]:
        if output_type is not None:
            if self._output_validators:
                raise exceptions.UserError('Cannot set a custom run `output_type` when the agent has output validators')
            schema = _output.OutputSchema[RunOutputDataT].build(
                output_type,
                name=self._deprecated_result_tool_name,
                description=self._deprecated_result_tool_description,
                default_mode=model_profile.default_structured_output_mode,
            )
        else:
            schema = self._output_schema.with_default_mode(model_profile.default_structured_output_mode)

        schema.raise_if_unsupported(model_profile)

        return schema  # pyright: ignore[reportReturnType]

    @staticmethod
    def is_model_request_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.ModelRequestNode[T, S]]:
        """Check if the node is a `ModelRequestNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.ModelRequestNode)

    @staticmethod
    def is_call_tools_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.CallToolsNode[T, S]]:
        """Check if the node is a `CallToolsNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.CallToolsNode)

    @staticmethod
    def is_user_prompt_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.UserPromptNode[T, S]]:
        """Check if the node is a `UserPromptNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.UserPromptNode)

    @staticmethod
    def is_end_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[End[result.FinalResult[S]]]:
        """Check if the node is a `End`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, End)

    async def __aenter__(self) -> Self:
        """Enter the agent context.

        This will start all [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] registered as `toolsets` so they are ready to be used.

        This is a no-op if the agent has already been entered.
        """
        async with self._enter_lock:
            if self._entered_count == 0:
                self._exit_stack = AsyncExitStack()
                toolset = self._get_toolset()
                await self._exit_stack.enter_async_context(toolset)
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

        from .mcp import MCPServer

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

    def to_ag_ui(
        self,
        *,
        # Agent.iter parameters
        output_type: OutputSpec[OutputDataT] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[BaseRoute] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
        on_startup: Sequence[Callable[[], Any]] | None = None,
        on_shutdown: Sequence[Callable[[], Any]] | None = None,
        lifespan: Lifespan[AGUIApp[AgentDepsT, OutputDataT]] | None = None,
    ) -> AGUIApp[AgentDepsT, OutputDataT]:
        """Convert the agent to an AG-UI application.

        This allows you to use the agent with a compatible AG-UI frontend.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')
        app = agent.to_ag_ui()
        ```

        The `app` is an ASGI application that can be used with any ASGI server.

        To run the application, you can use the following command:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

        See [AG-UI docs](../ag-ui.md) for more information.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional list of toolsets to use for this agent, defaults to the agent's toolset.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.

        Returns:
            An ASGI application for running Pydantic AI agents with AG-UI protocol support.
        """
        from .ag_ui import AGUIApp

        return AGUIApp(
            agent=self,
            # Agent.iter parameters
            output_type=output_type,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            # Starlette
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

    def to_a2a(
        self,
        *,
        storage: Storage | None = None,
        broker: Broker | None = None,
        # Agent card
        name: str | None = None,
        url: str = 'http://localhost:8000',
        version: str = '1.0.0',
        description: str | None = None,
        provider: AgentProvider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan[FastA2A] | None = None,
    ) -> FastA2A:
        """Convert the agent to a FastA2A application.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')
        app = agent.to_a2a()
        ```

        The `app` is an ASGI application that can be used with any ASGI server.

        To run the application, you can use the following command:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```
        """
        from ._a2a import agent_to_a2a

        return agent_to_a2a(
            self,
            storage=storage,
            broker=broker,
            name=name,
            url=url,
            version=version,
            description=description,
            provider=provider,
            skills=skills,
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

    async def to_cli(self: Self, deps: AgentDepsT = None, prog_name: str = 'pydantic-ai') -> None:
        """Run the agent in a CLI chat interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.

        Example:
        ```python {title="agent_to_cli.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')

        async def main():
            await agent.to_cli()
        ```
        """
        from rich.console import Console

        from pydantic_ai._cli import run_chat

        await run_chat(stream=True, agent=self, deps=deps, console=Console(), code_theme='monokai', prog_name=prog_name)

    def to_cli_sync(self: Self, deps: AgentDepsT = None, prog_name: str = 'pydantic-ai') -> None:
        """Run the agent in a CLI chat interface with the non-async interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.

        ```python {title="agent_to_cli_sync.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o', instructions='You always respond in Italian.')
        agent.to_cli_sync()
        agent.to_cli_sync(prog_name='assistant')
        ```
        """
        return get_event_loop().run_until_complete(self.to_cli(deps=deps, prog_name=prog_name))


@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions=None,
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
                    ]
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='Paris')],
                    usage=Usage(
                        requests=1, request_tokens=56, response_tokens=1, total_tokens=57
                    ),
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                )
            ),
            End(data=FinalResult(output='Paris')),
        ]
        '''
        print(agent_run.result.output)
        #> Paris
    ```

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[OutputDataT]
    ]

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        traceparent = self._graph_run._traceparent(required=False)  # type: ignore[reportPrivateUsage]
        if traceparent is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return traceparent

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            self._graph_run.state, self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        next_node = self._graph_run.next_node
        if isinstance(next_node, End):
            return next_node
        if _agent_graph.is_agent_node(next_node):
            return next_node
        raise exceptions.AgentRunError(f'Unexpected node type: {type(next_node)}')  # pragma: no cover

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        graph_run_result = self._graph_run.result
        if graph_run_result is None:
            return None
        return AgentRunResult(
            graph_run_result.output.output,
            graph_run_result.output.tool_name,
            graph_run_result.state,
            self._graph_run.deps.new_message_index,
            self._traceparent(required=False),
        )

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node."""
        next_node = await self._graph_run.__anext__()
        if _agent_graph.is_agent_node(next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
                print(nodes)
                '''
                [
                    UserPromptNode(
                        user_prompt='What is the capital of France?',
                        instructions=None,
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
                            ]
                        )
                    ),
                    CallToolsNode(
                        model_response=ModelResponse(
                            parts=[TextPart(content='Paris')],
                            usage=Usage(
                                requests=1,
                                request_tokens=56,
                                response_tokens=1,
                                total_tokens=57,
                            ),
                            model_name='gpt-4o',
                            timestamp=datetime.datetime(...),
                        )
                    ),
                    End(data=FinalResult(output='Paris')),
                ]
                '''
                print('Final result:', agent_run.result.output)
                #> Final result: Paris
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        next_node = await self._graph_run.next(node)
        if _agent_graph.is_agent_node(next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    def usage(self) -> _usage.Usage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    def __repr__(self) -> str:  # pragma: no cover
        result = self._graph_run.result
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'


@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""

    _output_tool_name: str | None = dataclasses.field(repr=False)
    _state: _agent_graph.GraphAgentState = dataclasses.field(repr=False)
    _new_message_index: int = dataclasses.field(repr=False)
    _traceparent_value: str | None = dataclasses.field(repr=False)

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return self._traceparent_value

    @property
    @deprecated('`result.data` is deprecated, use `result.output` instead.')
    def data(self) -> OutputDataT:
        return self.output

    def _set_output_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the output tool.

        Useful if you want to continue the conversation and want to set the response to the output tool call.
        """
        if not self._output_tool_name:
            raise ValueError('Cannot set output tool return content when the return type is `str`.')

        messages = self._state.message_history
        last_message = messages[-1]
        for idx, part in enumerate(last_message.parts):
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._output_tool_name:
                # Only do deepcopy when we have to modify
                copied_messages = list(messages)
                copied_last = deepcopy(last_message)
                copied_last.parts[idx].content = return_content  # type: ignore[misc]
                copied_messages[-1] = copied_last
                return copied_messages

        raise LookupError(f'No tool call found with tool name {self._output_tool_name!r}.')

    @overload
    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def all_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    def all_messages(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: Deprecated, use `output_tool_return_content` instead.

        Returns:
            List of messages.
        """
        content = result.coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        if content is not None:
            return self._set_output_tool_return(content)
        else:
            return self._state.message_history

    @overload
    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def all_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes: ...

    def all_messages_json(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: Deprecated, use `output_tool_return_content` instead.

        Returns:
            JSON bytes representing the messages.
        """
        content = result.coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages(output_tool_return_content=content))

    @overload
    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def new_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    def new_messages(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: Deprecated, use `output_tool_return_content` instead.

        Returns:
            List of new messages.
        """
        content = result.coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return self.all_messages(output_tool_return_content=content)[self._new_message_index :]

    @overload
    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def new_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes: ...

    def new_messages_json(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: Deprecated, use `output_tool_return_content` instead.

        Returns:
            JSON bytes representing the new messages.
        """
        content = result.coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages(output_tool_return_content=content))

    def usage(self) -> _usage.Usage:
        """Return the usage of the whole run."""
        return self._state.usage
